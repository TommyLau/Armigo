// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture-cache blit admission selects direct copies or scaled sampling. Copy
// regions map onto sampler coordinates after validating the accelerated surface
// contract and rejecting unsupported operations.

#include "twod_blit.h"

#include "attachments.h"
#include "texture_residency.h"

#include <armigo/graphics/texture_format.h>
#include <armigo/graphics/texture_layout.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>

namespace armigo::platform::metal {
namespace {

using namespace armigo::graphics;

constexpr std::string_view blit_msl = R"(
#include <metal_stdlib>
using namespace metal;
struct BlitCorner { float2 position; float2 uv; };
struct BlitOut { float4 position [[position]]; float2 uv; };
vertex BlitOut blit_vertex(uint vid [[vertex_id]],
                           constant BlitCorner* corners [[buffer(0)]]) {
    return {float4(corners[vid].position, 0.0, 1.0), corners[vid].uv};
}
fragment float4 blit_float_frag(BlitOut in [[stage_in]],
                                texture2d<float> source [[texture(0)]],
                                sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
fragment uint4 blit_uint_frag(BlitOut in [[stage_in]],
                              texture2d<uint> source [[texture(0)]],
                              sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
fragment int4 blit_sint_frag(BlitOut in [[stage_in]],
                             texture2d<int> source [[texture(0)]],
                             sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
)";

RendererError failure(CommandStatus status, const std::string& detail) {
    return RendererError(CommandResult{status, detail});
}

// Storage geometry of one decoded 2D surface, mirroring the attachment
// descriptor construction: the backing extent, GOB-aligned row stride and
// storage row count come from the shared layout module.
struct SurfaceStorage {
    std::uint64_t address = 0;
    TextureDescriptor descriptor;
};

std::optional<RendererError> surface_storage(const TwodSurface& surface,
                                             TextureFormat format,
                                             std::string_view role,
                                             SurfaceStorage& out) {
    const TextureFormatInfo info = texture_format_info(format);
    if (surface.block_depth_log2 != 0)
        return failure(CommandStatus::Unsupported,
                       "2D blit " + std::string(role) +
                           " GOB block depth requires a 3D surface");
    const std::uint32_t bytes_per_pixel = info.bytes_per_block;
    if (bytes_per_pixel == 0)
        return failure(CommandStatus::Unsupported,
                       "2D blit " + std::string(role) + " format has no texel size");
    TextureLayoutDescription description{};
    if (surface.pitch) {
        if (surface.pitch_bytes == 0 || surface.pitch_bytes % bytes_per_pixel != 0)
            return failure(CommandStatus::Unsupported,
                           "2D blit " + std::string(role) +
                               " byte pitch is not a whole texel");
        description.width = surface.pitch_bytes / bytes_per_pixel;
        description.pitch_bytes = surface.pitch_bytes;
        description.storage = TextureStorageLayout::Pitch;
    } else {
        description.width = surface.width;
        description.storage = TextureStorageLayout::BlockLinear;
    }
    description.height = surface.height;
    description.depth = 1;
    description.levels = 1;
    description.layers = 1;
    description.block_width = info.block_width;
    description.block_height = info.block_height;
    description.bytes_per_block = info.bytes_per_block;
    description.gob_block_height_log2 = surface.block_height_log2;
    description.gob_block_depth_log2 = 0;
    const auto calculated = calculate_texture_size(description);
    if (const auto* layout_error = std::get_if<TextureLayoutError>(&calculated))
        return failure(CommandStatus::Unsupported,
                       "2D blit " + std::string(role) +
                           " storage layout is invalid: " + layout_error->message());
    const auto& layout = std::get<TextureLayoutDescription>(calculated);
    if (layout.total_size == 0 || layout.total_size > max_texture_backing_size)
        return failure(CommandStatus::BudgetExceeded,
                       "2D blit " + std::string(role) +
                           " storage extent is outside the texture budget");
    const TextureLevelLayout& base = layout.mips[0];
    out.address = surface.address;
    out.descriptor.address = surface.address;
    out.descriptor.backing_extent = layout.total_size;
    if (surface.pitch) {
        out.descriptor.row_bytes = surface.pitch_bytes;
        out.descriptor.storage_height = base.block_rows;
    } else {
        const std::size_t logical_row =
            std::size_t{base.block_columns} * info.bytes_per_block;
        out.descriptor.row_bytes =
            (logical_row + texture_gob_width_bytes - 1) / texture_gob_width_bytes *
            texture_gob_width_bytes;
        const std::size_t gob_rows =
            texture_gob_height_rows << base.gob_block_height_log2;
        out.descriptor.storage_height =
            (std::size_t{base.block_rows} + gob_rows - 1) / gob_rows * gob_rows;
    }
    out.descriptor.layer_stride = layout.layer_stride;
    out.descriptor.width = description.width;
    out.descriptor.height = surface.height;
    out.descriptor.depth = 1;
    out.descriptor.levels = 1;
    out.descriptor.layers = 1;
    out.descriptor.base_level = 0;
    out.descriptor.samples = 1;
    out.descriptor.block_height_log2 = surface.block_height_log2;
    out.descriptor.block_depth_log2 = 0;
    out.descriptor.format = format;
    out.descriptor.layout =
        surface.pitch ? TextureLayout::Pitch : TextureLayout::BlockLinear;
    out.descriptor.type = TextureType::Texture2D;
    out.descriptor.srgb = format == TextureFormat::R8G8B8A8Srgb;
    return std::nullopt;
}

// Latches blit completion through the shared registry. Like the draw path,
// deferred write-back keeps GPU results GPU-authoritative and the owner's
// coherence callback publishes them; otherwise the coherent boundary runs here.
class TwodActions final : public CompletionActions {
public:
    TwodActions(std::shared_ptr<ResourceMemoryRegistry> memory, bool defer)
        : memory_(std::move(memory)), defer_(defer) {}
    RendererStatus make_coherent() override {
        return defer_ ? memory_->internal_boundary() : memory_->submission_boundary();
    }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    // The committed reservation keeps every authoritative dirty allocation
    // registry-owned even on failed feedback; nothing needs transfer.
    RendererStatus preserve_failed_authority() override { return renderer_success(); }
private:
    std::shared_ptr<ResourceMemoryRegistry> memory_;
    bool defer_ = false;
};

// Retains one blit's native objects, access storage and reservation until the
// completion state retires them.
struct TwodLease final : ResourceLease {
    NSMutableArray<id>* retained = nil;
    std::shared_ptr<GpuAccessReservation> reservation;
    std::vector<RetainedResource> access_storage;
};

bool wait_for_dependencies(const std::vector<CompletionToken>& dependencies,
                           std::uint64_t timeout_ns, RendererError& error) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(timeout_ns);
    for (const auto& dependency : dependencies) {
        for (;;) {
            const auto state = dependency.snapshot();
            if (state.error) {
                error = *state.error;
                return false;
            }
            if (state.phase == SubmissionPhase::Complete) break;
            if (state.phase == SubmissionPhase::Failed) {
                error = failure(CommandStatus::Stopped, "2D blit dependency failed");
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                error = failure(CommandStatus::Stopped, "2D blit dependency timed out");
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return true;
}

// One vertex-function corner table: NDC positions of the destination
// rectangle plus the source rectangle's normalized coordinates. NDC and
// texture space share Metal's downward Y axis, so no flip is involved.
struct BlitCorner {
    float position[2];
    float uv[2];
};

id<MTLTexture> native_texture(const ResidentTexture& resident,
                              id<MTLTexture>* allocation = nullptr) {
    const auto resource =
        std::dynamic_pointer_cast<const TextureResource>(resident.storage);
    if (!resource) return nil;
    if (allocation) *allocation = resource->allocation();
    return resource->texture();
}

}  // namespace

RendererStatus execute_metal_twod_blit(MetalTwodResources& resources,
                                       const RenderStop& stop,
                                       const GpuAddressSpace& memory) noexcept {
    static_cast<void>(memory);
    @autoreleasepool {
        try {
            if (!resources.device || !resources.queue || !resources.memory ||
                !resources.textures || resources.submission.queue == 0 ||
                resources.submission.sequence == 0 ||
                resources.completion_timeout_ns == 0)
                return failure(
                    CommandStatus::Unsupported, "2D blit has no joined renderer domain");

            // 1. Decode and admit the captured 2D state.
            const auto decoded = decode_twod_blit(stop.registers);
            if (const auto* rejected = std::get_if<CommandResult>(&decoded))
                return RendererError(*rejected);
            const auto& blit = std::get<TwodBlit>(decoded);
            if (blit.operation != TwodOperation::SrcCopy) {
                char detail[96]{};
                std::snprintf(detail, sizeof detail,
                              "2D blit operation %u has no copy implementation",
                              unsigned(blit.operation));
                return failure(CommandStatus::Unsupported, detail);
            }
            if (blit.clip_enable)
                return failure(CommandStatus::Unsupported,
                               "2D blit clip rectangle is unsupported");
            for (const auto [surface, role] : {std::pair{&blit.source, "source"},
                                               {&blit.destination, "destination"}}) {
                if (surface->layer != 0)
                    return failure(
                        CommandStatus::Unsupported,
                        std::string("2D blit ") + role + " layer is unsupported");
                if (surface->depth > 1)
                    return failure(
                        CommandStatus::Unsupported,
                        std::string("2D blit ") + role + " depth is unsupported");
            }
            TextureFormat source_format{};
            TextureFormat destination_format{};
            if (!decode_twod_format(blit.source.format, source_format))
                return failure(
                    CommandStatus::Unsupported,
                    "2D blit source format has no decoded table entry");
            if (!decode_twod_format(blit.destination.format, destination_format))
                return failure(
                    CommandStatus::Unsupported,
                    "2D blit destination format has no decoded table entry");
            if (texture_numeric_class(source_format) !=
                texture_numeric_class(destination_format))
                return failure(
                    CommandStatus::Unsupported,
                    "2D blit format conversion crosses numeric classes");

            SurfaceStorage source_storage{};
            SurfaceStorage destination_storage{};
            if (const auto error =
                    surface_storage(blit.source, source_format, "source", source_storage))
                return *error;
            if (const auto error = surface_storage(blit.destination, destination_format,
                                                   "destination", destination_storage))
                return *error;

            // Zero extents perform no hardware work; report the faithful no-op.
            const std::int64_t source_width = std::int64_t(blit.source_x2) - blit.source_x1;
            const std::int64_t source_height = std::int64_t(blit.source_y2) - blit.source_y1;
            if (blit.destination_width == 0 || blit.destination_height == 0 ||
                source_width <= 0 || source_height <= 0)
                return renderer_success();

            // The destination rectangle must land inside the destination
            // surface; attachment writes cannot clip it.
            if (blit.destination_x1 < 0 || blit.destination_y1 < 0 ||
                std::uint64_t(blit.destination_x1) + blit.destination_width >
                    destination_storage.descriptor.width ||
                std::uint64_t(blit.destination_y1) + blit.destination_height >
                    destination_storage.descriptor.height) {
                char detail[160]{};
                std::snprintf(detail, sizeof detail,
                              "2D blit destination rectangle [%d,%d)+%ux%u escapes the "
                              "%ux%u surface",
                              blit.destination_x1, blit.destination_y1,
                              blit.destination_width, blit.destination_height,
                              destination_storage.descriptor.width,
                              destination_storage.descriptor.height);
                return failure(CommandStatus::Unsupported, detail);
            }

            // 2. Resolve both surfaces through the texture cache.
            auto retained_source =
                resources.memory->retain(source_storage.address,
                                         source_storage.descriptor.backing_extent,
                                         ResourceAccess::Read);
            if (const auto* error = std::get_if<RendererError>(&retained_source))
                return *error;
            auto retained_destination = resources.memory->retain(
                destination_storage.address,
                destination_storage.descriptor.backing_extent, ResourceAccess::Write);
            if (const auto* error = std::get_if<RendererError>(&retained_destination))
                return *error;
            const auto& source_range = std::get<RetainedRange>(retained_source);
            const auto& destination_range = std::get<RetainedRange>(retained_destination);
            TextureViewKey source_key{};
            source_key.storage.bytes = source_range.bytes;
            source_key.storage.descriptor = source_storage.descriptor;
            TextureViewKey destination_key{};
            destination_key.storage.bytes = destination_range.bytes;
            destination_key.storage.descriptor = destination_storage.descriptor;
            auto acquired_source = resources.textures->acquire(
                {source_range, source_key, ResourceAccess::Read});
            if (const auto* error = std::get_if<RendererError>(&acquired_source))
                return *error;
            auto acquired_destination = resources.textures->acquire(
                {destination_range, destination_key, ResourceAccess::Write});
            if (const auto* error = std::get_if<RendererError>(&acquired_destination))
                return *error;
            auto source = std::get<ResidentTexture>(std::move(acquired_source));
            auto destination = std::get<ResidentTexture>(std::move(acquired_destination));
            id<MTLTexture> source_allocation = nil;
            id<MTLTexture> destination_allocation = nil;
            id<MTLTexture> source_texture =
                native_texture(source, &source_allocation);
            id<MTLTexture> destination_texture =
                native_texture(destination, &destination_allocation);
            if (!source_texture || !destination_texture || !source_allocation ||
                !destination_allocation)
                return failure(CommandStatus::Unsupported,
                               "2D blit resident storage has no native texture");

            // 3. Reserve the access pair: the destination becomes GPU-dirty in
            // the registry like a draw, versioned with this submission identity.
            auto lease = std::make_shared<TwodLease>();
            auto actions =
                std::make_shared<TwodActions>(resources.memory,
                                              resources.defer_gpu_writeback);
            SubmissionCompletion completion(resources.submission, {lease}, actions);
            const std::array<GpuAccessRequest, 2> accesses{
                {{source_range, ResourceAccess::Read},
                 {destination_range, ResourceAccess::Write}}};
            auto reserved =
                resources.memory->reserve_accesses(accesses, completion.token());
            if (const auto* error = std::get_if<RendererError>(&reserved)) {
                static_cast<void>(completion.reject(error->command()));
                return *error;
            }
            lease->reservation =
                std::get<std::shared_ptr<GpuAccessReservation>>(std::move(reserved));
            lease->access_storage = {source.storage, destination.storage};

            // 4. Honor every pending producer before encoding parallel work.
            std::vector<CompletionToken> dependencies = source.dependencies;
            dependencies.insert(dependencies.end(), destination.dependencies.begin(),
                                destination.dependencies.end());
            auto reservation_dependencies = lease->reservation->dependencies();
            dependencies.insert(dependencies.end(), reservation_dependencies.begin(),
                                reservation_dependencies.end());
            RendererError dependency_error;
            if (!wait_for_dependencies(dependencies, resources.completion_timeout_ns,
                                       dependency_error)) {
                static_cast<void>(completion.reject(dependency_error.command()));
                return dependency_error;
            }
            if (const auto& committed = lease->reservation->commit(lease->access_storage);
                !std::holds_alternative<std::monostate>(committed)) {
                const auto& error = std::get<RendererError>(committed);
                static_cast<void>(completion.reject(error.command()));
                return error;
            }
            if (const auto& submitted = completion.submitted();
                !std::holds_alternative<std::monostate>(submitted))
                return std::get<RendererError>(submitted);

            // 5. Encode: same-size same-format in-bounds copies blit; scaled,
            // converting, out-of-bounds-source or flipped copies draw a
            // textured quad with the requested filter and clamped sampling.
            const bool source_in_bounds =
                blit.source_x1 >= 0 && blit.source_y1 >= 0 &&
                std::uint64_t(blit.source_x1) + std::uint64_t(source_width) <=
                    source_storage.descriptor.width &&
                std::uint64_t(blit.source_y1) + std::uint64_t(source_height) <=
                    source_storage.descriptor.height;
            const bool direct_copy =
                std::uint64_t(source_width) == blit.destination_width &&
                std::uint64_t(source_height) == blit.destination_height &&
                source_format == destination_format && source_in_bounds &&
                !blit.flip_x && !blit.flip_y;
            id<MTL4CommandAllocator> allocator = [resources.device newCommandAllocator];
            auto* residency_descriptor = [MTLResidencySetDescriptor new];
            residency_descriptor.initialCapacity = 8;
            NSError* allocation_error = nil;
            id<MTLResidencySet> residency = [resources.device
                newResidencySetWithDescriptor:residency_descriptor
                                        error:&allocation_error];
            NSMutableArray* retained =
                [NSMutableArray arrayWithObjects:allocator, residency, nil];
            if (!allocator || !residency || !retained)
                return failure(CommandStatus::Unsupported,
                                          "2D blit command allocation failed");
            [residency addAllocation:source_texture];
            [residency addAllocation:destination_texture];
            // Views alias one allocation per cache entry; residency follows the
            // draw path in pinning the underlying allocations too.
            if (source_allocation != source_texture) [residency addAllocation:source_allocation];
            if (destination_allocation != destination_texture)
                [residency addAllocation:destination_allocation];
            [residency commit];
            id<MTL4CommandBuffer> commands = [resources.device newCommandBuffer];
            if (!commands)
                return failure(CommandStatus::Unsupported,
                                          "2D blit command buffer allocation failed");
            [retained addObject:commands];
            [commands beginCommandBufferWithAllocator:allocator];
            [commands useResidencySet:residency];
            if (direct_copy) {
                id<MTL4ComputeCommandEncoder> transfer = [commands computeCommandEncoder];
                if (!transfer)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit copy encoder creation failed");
                [transfer barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                                visibilityOptions:MTL4VisibilityOptionDevice];
                [transfer copyFromTexture:source_texture
                              sourceSlice:0
                               sourceLevel:0
                              sourceOrigin:MTLOriginMake(blit.source_x1, blit.source_y1, 0)
                               sourceSize:MTLSizeMake(source_width, source_height, 1)
                                toTexture:destination_texture
                         destinationSlice:0
                          destinationLevel:0
                         destinationOrigin:MTLOriginMake(blit.destination_x1,
                                                          blit.destination_y1, 0)];
                [transfer endEncoding];
            } else {
                const auto destination_pixels = metal_render_format(
                    destination_format, TextureAspect::Color);
                if (const auto* error = std::get_if<CommandResult>(&destination_pixels))
                    return RendererError(*error);
                auto* compile_options = [MTLCompileOptions new];
                compile_options.languageVersion = MTLLanguageVersion4_0;
                id<MTLLibrary> library = [resources.device
                    newLibraryWithSource:[NSString
                                             stringWithUTF8String:std::string(blit_msl).c_str()]
                                  options:compile_options
                                    error:&allocation_error];
                if (!library)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit shader library compilation failed");
                [retained addObject:library];
                const auto numeric = texture_numeric_class(destination_format);
                NSString* fragment = @"blit_float_frag";
                if (numeric == TextureNumericClass::Uint) fragment = @"blit_uint_frag";
                if (numeric == TextureNumericClass::Sint) fragment = @"blit_sint_frag";
                auto* pipeline_descriptor = [MTLRenderPipelineDescriptor new];
                pipeline_descriptor.vertexFunction =
                    [library newFunctionWithName:@"blit_vertex"];
                pipeline_descriptor.fragmentFunction = [library newFunctionWithName:fragment];
                pipeline_descriptor.inputPrimitiveTopology =
                    MTLPrimitiveTopologyClassTriangle;
                pipeline_descriptor.colorAttachments[0].pixelFormat =
                    std::get<MTLPixelFormat>(destination_pixels);
                pipeline_descriptor.colorAttachments[0].blendingEnabled = NO;
                id<MTLRenderPipelineState> pipeline =
                    [resources.device newRenderPipelineStateWithDescriptor:
                                          pipeline_descriptor
                                                                    error:&allocation_error];
                if (!pipeline)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit pipeline creation failed");
                [retained addObject:pipeline];
                auto* sampler_descriptor = [MTLSamplerDescriptor new];
                sampler_descriptor.minFilter = sampler_descriptor.magFilter =
                    blit.linear_filter ? MTLSamplerMinMagFilterLinear
                                       : MTLSamplerMinMagFilterNearest;
                sampler_descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
                id<MTLSamplerState> sampler =
                    [resources.device newSamplerStateWithDescriptor:sampler_descriptor];
                if (!sampler)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit sampler creation failed");
                [retained addObject:sampler];

                // Corner table: the destination rectangle in NDC, the source
                // rectangle in normalized coordinates. Metal's NDC Y points
                // up while guest rows and texture V grow downward, so the
                // vertical mapping negates; sampler clamp-to-edge bounds the
                // source. Direction flips swap the walked axis.
                const float surface_width = float(destination_storage.descriptor.width);
                const float surface_height = float(destination_storage.descriptor.height);
                const float source_texture_width = float(source_storage.descriptor.width);
                const float source_texture_height = float(source_storage.descriptor.height);
                const auto ndc_x = [&](std::int32_t pixel) {
                    return 2.0f * float(pixel) / surface_width - 1.0f;
                };
                const auto ndc_y = [&](std::int32_t pixel) {
                    return 1.0f - 2.0f * float(pixel) / surface_height;
                };
                const auto u = [&](std::int32_t texel) {
                    return float(texel) / source_texture_width;
                };
                const auto v = [&](std::int32_t texel) {
                    return float(texel) / source_texture_height;
                };
                const std::int32_t source_left = blit.flip_x ? blit.source_x2 : blit.source_x1;
                const std::int32_t source_right = blit.flip_x ? blit.source_x1 : blit.source_x2;
                const std::int32_t source_top = blit.flip_y ? blit.source_y2 : blit.source_y1;
                const std::int32_t source_bottom = blit.flip_y ? blit.source_y1 : blit.source_y2;
                const std::int32_t destination_left = blit.destination_x1;
                const std::int32_t destination_right =
                    blit.destination_x1 + std::int32_t(blit.destination_width);
                const std::int32_t destination_top = blit.destination_y1;
                const std::int32_t destination_bottom =
                    blit.destination_y1 + std::int32_t(blit.destination_height);
                const BlitCorner corners[4] = {
                    {{ndc_x(destination_left), ndc_y(destination_top)},
                     {u(source_left), v(source_top)}},
                    {{ndc_x(destination_right), ndc_y(destination_top)},
                     {u(source_right), v(source_top)}},
                    {{ndc_x(destination_left), ndc_y(destination_bottom)},
                     {u(source_left), v(source_bottom)}},
                    {{ndc_x(destination_right), ndc_y(destination_bottom)},
                     {u(source_right), v(source_bottom)}},
                };
                id<MTLBuffer> corner_buffer = [resources.device
                    newBufferWithBytes:corners
                                length:sizeof(corners)
                               options:MTLResourceStorageModeShared];
                if (!corner_buffer)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit corner allocation failed");
                [retained addObject:corner_buffer];
                auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
                table_descriptor.maxBufferBindCount = 1;
                table_descriptor.maxTextureBindCount = 1;
                table_descriptor.maxSamplerStateBindCount = 1;
                id<MTL4ArgumentTable> vertex_table = [resources.device
                    newArgumentTableWithDescriptor:table_descriptor error:&allocation_error];
                id<MTL4ArgumentTable> fragment_table = [resources.device
                    newArgumentTableWithDescriptor:table_descriptor error:&allocation_error];
                if (!vertex_table || !fragment_table)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit argument table allocation failed");
                [retained addObject:vertex_table];
                [retained addObject:fragment_table];
                [vertex_table setAddress:corner_buffer.gpuAddress atIndex:0];
                [fragment_table setTexture:source_texture.gpuResourceID atIndex:0];
                [fragment_table setSamplerState:sampler.gpuResourceID atIndex:0];
                auto* pass = [MTL4RenderPassDescriptor new];
                auto* attachment = pass.colorAttachments[0];
                attachment.texture = destination_texture;
                attachment.loadAction = MTLLoadActionLoad;
                attachment.storeAction = MTLStoreActionStore;
                id<MTL4RenderCommandEncoder> render =
                    [commands renderCommandEncoderWithDescriptor:pass];
                if (!render)
                    return failure(CommandStatus::Unsupported,
                                              "2D blit render encoder creation failed");
                [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                              visibilityOptions:MTL4VisibilityOptionDevice];
                [render setRenderPipelineState:pipeline];
                [render setViewport:MTLViewport{0.0, 0.0, double(surface_width),
                                                double(surface_height), 0.0, 1.0}];
                [render setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
                [render setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
                [render drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0
                            vertexCount:4];
                [render endEncoding];
            }
            [commands endCommandBuffer];

            // 6. Execute synchronously and drive the completion state.
            dispatch_semaphore_t finished = dispatch_semaphore_create(0);
            __block NSError* execution_error = nil;
            auto* commit_options = [MTL4CommitOptions new];
            [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                // No guest-memory callback or borrowed C++ object escapes into
                // this block. Feedback retains every GPU allocation.
                static_cast<void>(retained.count);
                execution_error = feedback.error;
                dispatch_semaphore_signal(finished);
            }];
            id<MTL4CommandBuffer> submitted_buffers[] = {commands};
            [resources.queue commit:submitted_buffers count:1 options:commit_options];
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::nanoseconds(resources.completion_timeout_ns);
            while (dispatch_semaphore_wait(finished, DISPATCH_TIME_NOW))
            {
                if (std::chrono::steady_clock::now() >= deadline) {
                    const auto timed_out =
                        failure(CommandStatus::Stopped, "2D blit GPU completion timed out");
                    static_cast<void>(completion.feedback(timed_out.command()));
                    static_cast<void>(completion.finish());
                    static_cast<void>(completion.release_failed());
                    return timed_out;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (execution_error) {
                const auto gpu_failure =
                    failure(CommandStatus::Stopped, "2D blit GPU execution failed");
                static_cast<void>(completion.feedback(gpu_failure.command()));
                static_cast<void>(completion.finish());
                static_cast<void>(completion.release_failed());
                return gpu_failure;
            }
            static_cast<void>(completion.feedback({}));
            const auto finished_state = completion.finish();
            if (const auto* error = std::get_if<RendererError>(&finished_state)) {
                static_cast<void>(completion.release_failed());
                return *error;
            }
            return renderer_success();
        } catch (const RendererError& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return failure(CommandStatus::BudgetExceeded, "2D blit allocation failed");
        } catch (...) {
            return failure(CommandStatus::Stopped, "2D blit backend threw an exception");
        }
    }
}

}  // namespace armigo::platform::metal
