// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Full clears fold into render-pass load actions. Partial, masked, or scissored
// clears emit operations that preserve untouched data. Array-layer ranges map
// to Metal slice selection and layered render passes.

#include "clear_encoder.h"
#include "pass_counters.h"

#include "attachments.h"

#include <armigo/graphics/texture_format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace armigo::platform::metal {
namespace {

using graphics::AttachmentLoad;
using graphics::ClearPlan;
using graphics::CommandResult;
using graphics::CommandStatus;
using graphics::TextureAspect;
using graphics::TextureFormat;

constexpr std::string_view clear_msl = R"(
#include <metal_stdlib>
using namespace metal;
struct ClearValues { float4 color; float depth; uint stencil; };
vertex float4 clear_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    return float4(positions[vid], 0.0, 1.0);
}
// Layered clears instance one vertex per cleared layer; the instance selects
// the render-target slice (Vulkan ClearRect baseArrayLayer semantics).
struct ClearLayerOut { float4 position [[position]]; uint layer [[render_target_array_index]]; };
vertex ClearLayerOut clear_vertex_layered(uint vid [[vertex_id]], uint instance [[instance_id]]) {
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    return {float4(positions[vid], 0.0, 1.0), instance};
}
fragment float4 clear_color_frag(constant ClearValues &values [[buffer(0)]]) {
    return values.color;
}
fragment uint4 clear_uint_frag(constant ClearValues &values [[buffer(0)]]) {
    return as_type<uint4>(values.color);
}
fragment int4 clear_sint_frag(constant ClearValues &values [[buffer(0)]]) {
    return as_type<int4>(values.color);
}
struct ColorDepthOut { float4 color [[color(0)]]; float depth [[depth(any)]]; };
fragment ColorDepthOut clear_color_depth_frag(constant ClearValues &values [[buffer(0)]]) {
    return {values.color, values.depth};
}
struct DepthOut { float depth [[depth(any)]]; };
fragment DepthOut clear_depth_frag(constant ClearValues &values [[buffer(0)]]) {
    return {values.depth};
}
// Stencil writes go through the depth-stencil operation path (always-pass
// replace with the planned write mask), which merges the reference into the
// loaded contents; the fragment stencil output does not preserve masked bits.
fragment void clear_stencil_frag() { }
)";

using EncodeResult = graphics::RendererResult<EncodedClear>;

EncodeResult encode_failure(std::string_view detail) {
    return graphics::RendererError{CommandResult{CommandStatus::Unsupported, detail}};
}

// The depth/stencil descriptor of the plan: a combined clear carries it
// alongside the colour attachment; single-aspect clears reuse target.
const graphics::AttachmentPlan& ds_plan(const ClearPlan& plan) {
    return plan.depth_stencil ? *plan.depth_stencil : plan.target;
}

// Validates the bindings named by the plan's aspects and returns the mapped
// pixel formats per aspect (color nullopt for depth/stencil clears).
struct ValidatedBindings {
    MTLPixelFormat color = MTLPixelFormatInvalid;
    MTLPixelFormat depth = MTLPixelFormatInvalid;
    MTLPixelFormat stencil = MTLPixelFormatInvalid;
};

std::optional<CommandResult>
validate_bindings(const ClearPlan& plan, const ClearBindings& bindings,
                  ValidatedBindings& out) {
    const bool color = plan.color_mask != 0;
    const bool depth = plan.clear_depth;
    const bool stencil = plan.clear_stencil;
    const auto bind = [&](const graphics::AttachmentPlan& aspect,
                          const id<MTLTexture> texture) -> std::optional<CommandResult> {
        if (!texture)
            return CommandResult{CommandStatus::Unsupported,
                                 "planned clear aspect has no resident texture"};
        if (texture.textureType == MTLTextureType3D &&
            (texture.depth != 1 || aspect.layer != 0))
            return CommandResult{CommandStatus::Unsupported,
                                 "clear 3D depth plane is unsupported"};
        if (aspect.level >= texture.mipmapLevelCount)
            return CommandResult{CommandStatus::Unsupported,
                                 "clear attachment level or slice is out of range"};
        const std::uint64_t layers = texture.textureType == MTLTextureType2DArray
            ? texture.arrayLength : 1;
        // A layered clear needs an array texture (renderTargetArrayLength
        // addresses slices); every cleared range must stay inside storage.
        if (aspect.layer_count > 1 && texture.textureType != MTLTextureType2DArray)
            return CommandResult{CommandStatus::Unsupported,
                                 "layered clear requires an array texture"};
        if (std::uint64_t(aspect.layer) + aspect.layer_count > layers)
            return CommandResult{CommandStatus::Unsupported,
                                 "clear layer range exceeds the attachment"};
        if (bindings.slice != aspect.layer)
            return CommandResult{CommandStatus::Unsupported,
                                 "clear binding slice contradicts the planned layer"};
        if (std::uint64_t(plan.region.x) + plan.region.width > texture.width ||
            std::uint64_t(plan.region.y) + plan.region.height > texture.height)
            return CommandResult{CommandStatus::Unsupported,
                                 "clear region exceeds the attachment extent"};
        return std::nullopt;
    };
    const auto require_format = [](id<MTLTexture> texture, MTLPixelFormat expected,
                                   bool stencil_view) -> std::optional<CommandResult> {
        const bool packed_stencil_view = stencil_view &&
            expected == MTLPixelFormatDepth32Float_Stencil8 &&
            texture.pixelFormat == MTLPixelFormatX32_Stencil8;
        if (texture.pixelFormat != expected && !packed_stencil_view)
            return CommandResult{CommandStatus::Unsupported,
                                 "clear attachment pixel format contradicts the plan"};
        return std::nullopt;
    };
    if (color) {
        const auto mapped = metal_render_format(plan.target.texture.format,
                                                TextureAspect::Color);
        if (const auto* error = std::get_if<CommandResult>(&mapped)) return *error;
        out.color = std::get<MTLPixelFormat>(mapped);
        if (const auto error = require_format(bindings.color, out.color, false))
            return error;
        if (const auto error = bind(plan.target, bindings.color)) return error;
    }
    if (depth || stencil) {
        const graphics::AttachmentPlan& aspect = ds_plan(plan);
        if (depth) {
            const auto mapped = metal_render_format(aspect.texture.format,
                                                    TextureAspect::Depth);
            if (const auto* error = std::get_if<CommandResult>(&mapped)) return *error;
            out.depth = std::get<MTLPixelFormat>(mapped);
            if (const auto error = require_format(bindings.depth, out.depth, false))
                return error;
            if (const auto error = bind(aspect, bindings.depth)) return error;
        }
        if (stencil) {
            const auto mapped = metal_render_format(aspect.texture.format,
                                                    TextureAspect::Stencil);
            if (const auto* error = std::get_if<CommandResult>(&mapped)) return *error;
            out.stencil = std::get<MTLPixelFormat>(mapped);
            if (const auto error = require_format(bindings.stencil, out.stencil, true))
                return error;
            if (const auto error = bind(aspect, bindings.stencil)) return error;
            out.stencil = bindings.stencil.pixelFormat;
            if (depth && bindings.stencil != bindings.depth)
                return CommandResult{CommandStatus::Unsupported,
                                     "packed depth/stencil aspects must alias one texture"};
        }
    }
    if (!color && !depth && !stencil)
        return CommandResult{CommandStatus::Unsupported,
                             "clear names no color or depth/stencil aspect"};
    return std::nullopt;
}

// The pass descriptor for a folded clear: the load actions perform the whole
// clear, so the pass stays empty. A layered fold clears every slice of the
// arrayed attachments through renderTargetArrayLength (the Vulkan load-op
// behaviour for a full ClearRect layer range).
EncodeResult encode_fold(id<MTLDevice> device,
                         id<MTL4CommandAllocator> allocator,
                         const ClearPlan& plan,
                         const ClearBindings& bindings,
                         NSMutableArray<id>* retained,
                         id<MTLResidencySet> residency) {
    const graphics::AttachmentPlan& colour = plan.target;
    const graphics::AttachmentPlan& depth_stencil = ds_plan(plan);
    const auto full_range = [](const graphics::AttachmentPlan& aspect) {
        return aspect.layer == 0 && aspect.layer_count == aspect.texture.layers;
    };
    if ((plan.color_mask != 0 && !full_range(colour)) ||
        ((plan.clear_depth || plan.clear_stencil) && !full_range(depth_stencil)))
        return encode_failure("folded clear does not cover every cleared layer");
    const std::uint32_t layer_count = plan.color_mask != 0
        ? colour.layer_count : depth_stencil.layer_count;
    const auto values = metal_clear_color(plan.color_bits, plan.target.texture.format);
    if (plan.color_mask != 0) {
        if (const auto* error = std::get_if<CommandResult>(&values))
            return encode_failure(error->detail);
    }
    MTL4RenderPassDescriptor * pass = [MTL4RenderPassDescriptor new];
    if (!pass) return encode_failure("Metal clear pass allocation failed");
    if (layer_count > 1) pass.renderTargetArrayLength = layer_count;
    if (plan.color_mask != 0) {
        auto* attachment = pass.colorAttachments[0];
        attachment.texture = bindings.color;
        attachment.level = colour.level;
        attachment.slice = colour.layer;
        attachment.depthPlane = 0;
        attachment.loadAction = MTLLoadActionClear;
        attachment.storeAction = MTLStoreActionStore;
        const auto& color = std::get<std::array<double, 4>>(values);
        attachment.clearColor =
            MTLClearColorMake(color[0], color[1], color[2], color[3]);
    }
    if (plan.clear_depth) {
        pass.depthAttachment.texture = bindings.depth;
        pass.depthAttachment.level = depth_stencil.level;
        pass.depthAttachment.slice = depth_stencil.layer;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = plan.depth_value;
    }
    if (plan.clear_stencil) {
        pass.stencilAttachment.texture = bindings.stencil;
        pass.stencilAttachment.level = depth_stencil.level;
        pass.stencilAttachment.slice = depth_stencil.layer;
        pass.stencilAttachment.loadAction = MTLLoadActionClear;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        pass.stencilAttachment.clearStencil = plan.stencil_value;
    }
    id<MTL4CommandBuffer> commands = bindings.into ? bindings.into : [device newCommandBuffer];
    if (!commands) return encode_failure("Metal clear command buffer allocation failed");
    if (!bindings.into) {
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
    }
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return encode_failure("Metal clear render encoder creation failed");
    // Queue visibility applies to work in this encoder, including attachment
    // loads. A barrier in a separate empty encoder cannot order this clear.
    [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                 visibilityOptions:MTL4VisibilityOptionDevice];
    const auto measured_pass = bindings.pass_counters
        ? bindings.pass_counters->begin(device, render, pass, "clear") : PassCounters::invalid;
    if (bindings.pass_counters) bindings.pass_counters->end(render, measured_pass);
    [render endEncoding];
    if (!bindings.into) [commands endCommandBuffer];
    EncodedClear encoded;
    encoded.commands = commands;
    encoded.allocator = allocator;
    encoded.retained = retained;
    return encoded;
}

// The clear library and pipelines, built once per device (library) and per
// configuration (pipeline) instead of for every clear. An entry compiles
// under its own lock, outside the cache lock, on the first thread that asks;
// the others wait for it. A failed compile leaves the entry empty, so the next
// request tries again. Each compile is recorded for the thread that ran it.
struct ClearLibraryEntry {
    std::mutex mutex;
    id<MTLLibrary> library = nil;
};
struct ClearPipelineEntry {
    std::mutex mutex;
    id<MTLRenderPipelineState> pipeline = nil;
};
using ClearPipelineKey = std::tuple<void*, std::string, std::string, MTLPixelFormat, MTLColorWriteMask,
                                    MTLPixelFormat, MTLPixelFormat, std::uint32_t>;

struct ClearShaderCache {
    std::mutex mutex;
    std::map<void*, std::unique_ptr<ClearLibraryEntry>> libraries;
    std::map<ClearPipelineKey, std::unique_ptr<ClearPipelineEntry>> pipelines;
};

ClearShaderCache& clear_shader_cache() {
    // Never destroyed: replay threads clear until the process ends.
    static auto& cache = *new ClearShaderCache;
    return cache;
}

thread_local std::vector<ClearCompile> thread_clear_compiles;

std::uint64_t clear_now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

template <typename Entry, typename Map, typename Key>
Entry& clear_entry(Map& map, const Key& key) {
    auto& cache = clear_shader_cache();
    const std::lock_guard lock(cache.mutex);
    auto& slot = map[key];
    if (!slot) slot = std::make_unique<Entry>();
    return *slot;
}

// Locks an entry; a wait of more than a millisecond (another thread was
// compiling it) is recorded for this thread.
std::unique_lock<std::mutex> lock_clear_entry(std::mutex& mutex, const std::optional<ClearPipelineConfig>& config) {
    std::unique_lock lock(mutex, std::try_to_lock);
    if (lock.owns_lock()) return lock;
    const auto begin = clear_now_ns();
    lock.lock();
    if (const auto waited = clear_now_ns() - begin; waited > 1'000'000)
        thread_clear_compiles.push_back({config, waited, true});
    return lock;
}

id<MTLLibrary> clear_library(id<MTLDevice> device) {
    auto& entry = clear_entry<ClearLibraryEntry>(clear_shader_cache().libraries, (__bridge void*)device);
    const auto lock = lock_clear_entry(entry.mutex, std::nullopt);
    if (!entry.library) {
        const auto begin = clear_now_ns();
        auto* compile_options = [MTLCompileOptions new];
        // The fragment depth/stencil outputs require the modern language mode.
        compile_options.languageVersion = MTLLanguageVersion4_0;
        NSError* error = nil;
        entry.library = [device newLibraryWithSource:[NSString stringWithUTF8String:std::string(clear_msl).c_str()]
                                             options:compile_options error:&error];
        thread_clear_compiles.push_back({std::nullopt, clear_now_ns() - begin});
    }
    return entry.library;
}

id<MTLRenderPipelineState> clear_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          const ClearPipelineConfig& config) {
    auto& entry = clear_entry<ClearPipelineEntry>(
        clear_shader_cache().pipelines,
        ClearPipelineKey{(__bridge void*)device, config.vertex, config.fragment, config.color, config.mask,
                         config.depth, config.stencil, config.samples});
    const auto lock = lock_clear_entry(entry.mutex, config);
    if (entry.pipeline) return entry.pipeline;
    // The Metal 4 pipeline descriptor carries no depth/stencil pixel formats,
    // so the pipeline state is created through the classic descriptor, which
    // MTL4RenderCommandEncoder accepts directly.
    auto* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@(config.vertex.c_str())];
    descriptor.fragmentFunction = [library newFunctionWithName:@(config.fragment.c_str())];
    if (!descriptor.vertexFunction || !descriptor.fragmentFunction) return nil;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.rasterSampleCount = config.samples;
    if (config.color != MTLPixelFormatInvalid) {
        auto* attachment = descriptor.colorAttachments[0];
        attachment.pixelFormat = config.color;
        attachment.writeMask = config.mask;
        attachment.blendingEnabled = NO;
    }
    descriptor.depthAttachmentPixelFormat = config.depth;
    descriptor.stencilAttachmentPixelFormat = config.stencil;
    const auto begin = clear_now_ns();
    NSError* error = nil;
    entry.pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    thread_clear_compiles.push_back({config, clear_now_ns() - begin});
    return entry.pipeline;
}

// The pipeline for a partial clear: full-screen triangle writing the clear
// values through the planned write masks and depth/stencil state.
EncodeResult encode_operation(id<MTLDevice> device,
                              id<MTL4CommandAllocator> allocator,
                              const ClearPlan& plan,
                              const ClearBindings& bindings,
                              const ValidatedBindings& formats,
                              NSMutableArray<id>* retained,
                              id<MTLResidencySet> residency) {
    NSError* error = nil;
    id<MTLLibrary> library = clear_library(device);
    if (!library) return encode_failure("Metal clear shader library compilation failed");
    [retained addObject:library];

    const bool color = plan.color_mask != 0;
    const bool depth = plan.clear_depth;
    const bool stencil = plan.clear_stencil;
    const graphics::AttachmentPlan& depth_stencil = ds_plan(plan);
    const char* fragment = color && depth ? "clear_color_depth_frag"
        : color ? "clear_color_frag"
        : depth ? "clear_depth_frag"
        : "clear_stencil_frag";
    if (color && !depth) {
        const auto numeric = graphics::texture_numeric_class(plan.target.texture.format);
        if (numeric == graphics::TextureNumericClass::Uint) fragment = "clear_uint_frag";
        if (numeric == graphics::TextureNumericClass::Sint) fragment = "clear_sint_frag";
    }
    // A layered clear instances the covering triangle once per cleared layer
    // and addresses the slices through render_target_array_index using the planned base array layer and layer count.
    const std::uint32_t layer_count = color
        ? plan.target.layer_count : depth_stencil.layer_count;
    const char* vertex = layer_count > 1 ? "clear_vertex_layered" : "clear_vertex";
    const auto mask = static_cast<MTLColorWriteMask>(
        MTLColorWriteMaskRed * ((plan.color_mask >> 0) & 1) +
        MTLColorWriteMaskGreen * ((plan.color_mask >> 1) & 1) +
        MTLColorWriteMaskBlue * ((plan.color_mask >> 2) & 1) +
        MTLColorWriteMaskAlpha * ((plan.color_mask >> 3) & 1));
    // Unmasked channels keep their loaded contents; the fragment output never reaches them.
    // The pipeline rasterizes at the cleared attachments' sample count.
    id<MTLTexture> attached = color ? bindings.color : depth ? bindings.depth : bindings.stencil;
    const ClearPipelineConfig config{vertex, fragment, color ? formats.color : MTLPixelFormatInvalid,
                                     color ? mask : MTLColorWriteMaskNone,
                                     depth ? formats.depth : MTLPixelFormatInvalid,
                                     stencil ? formats.stencil : MTLPixelFormatInvalid,
                                     attached ? static_cast<std::uint32_t>(attached.sampleCount) : 1u};
    id<MTLRenderPipelineState> pipeline = clear_pipeline(device, library, config);
    if (!pipeline) return encode_failure("Metal clear pipeline creation failed");
    [retained addObject:pipeline];

    auto* depth_descriptor = [MTLDepthStencilDescriptor new];
    depth_descriptor.depthCompareFunction = MTLCompareFunctionAlways;
    depth_descriptor.depthWriteEnabled = depth;
    auto* face = [MTLStencilDescriptor new];
    face.stencilCompareFunction = MTLCompareFunctionAlways;
    // The stencil write lands through both the operation path (reference
    // replace) and the fragment stencil output; the write mask bounds both.
    face.stencilFailureOperation = stencil ? MTLStencilOperationReplace
                                           : MTLStencilOperationKeep;
    face.depthFailureOperation = stencil ? MTLStencilOperationReplace
                                         : MTLStencilOperationKeep;
    face.depthStencilPassOperation = stencil ? MTLStencilOperationReplace
                                             : MTLStencilOperationKeep;
    face.writeMask = stencil ? plan.stencil_write_mask : 0;
    depth_descriptor.frontFaceStencil = face;
    depth_descriptor.backFaceStencil = face;
    id<MTLDepthStencilState> depth_state =
        [device newDepthStencilStateWithDescriptor:depth_descriptor];
    if (!depth_state) return encode_failure("Metal clear depth-stencil state creation failed");
    [retained addObject:depth_state];

    struct ClearValues {
        std::uint32_t color[4];
        float depth;
        std::uint32_t stencil;
    } values{};
    // Float clear words are IEEE-754 bits; typed shaders reinterpret the same
    // payload as integers without an intervening lossy float conversion.
    std::copy(plan.color_bits.begin(), plan.color_bits.end(), values.color);
    values.depth = static_cast<float>(plan.depth_value);
    values.stencil = plan.stencil_value;
    id<MTLBuffer> values_buffer = [device newBufferWithBytes:&values
        length:sizeof(values) options:MTLResourceStorageModeShared];
    if (!values_buffer) return encode_failure("Metal clear values allocation failed");
    [retained addObject:values_buffer];
    if (bindings.into) [residency addAllocation:values_buffer];

    auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 1;
    table_descriptor.maxTextureBindCount = 1;
    table_descriptor.maxSamplerStateBindCount = 1;
    id<MTL4ArgumentTable> fragment_table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    if (!fragment_table) return encode_failure("Metal clear argument table allocation failed");
    [retained addObject:fragment_table];
    [fragment_table setAddress:values_buffer.gpuAddress atIndex:0];

    id<MTL4CommandBuffer> commands = bindings.into ? bindings.into : [device newCommandBuffer];
    if (!commands) return encode_failure("Metal clear command buffer allocation failed");
    if (!bindings.into) {
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
    }
    auto* pass = [MTL4RenderPassDescriptor new];
    if (!pass) return encode_failure("Metal clear pass allocation failed");
    if (layer_count > 1) pass.renderTargetArrayLength = layer_count;
    if (color) {
        auto* attachment = pass.colorAttachments[0];
        attachment.texture = bindings.color;
        attachment.level = plan.target.level;
        attachment.slice = plan.target.layer;
        attachment.depthPlane = 0;
        // Untouched pixels load; only the draw writes.
        attachment.loadAction = MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionStore;
    }
    if (depth) {
        pass.depthAttachment.texture = bindings.depth;
        pass.depthAttachment.level = depth_stencil.level;
        pass.depthAttachment.slice = depth_stencil.layer;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (stencil) {
        pass.stencilAttachment.texture = bindings.stencil;
        pass.stencilAttachment.level = depth_stencil.level;
        pass.stencilAttachment.slice = depth_stencil.layer;
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return encode_failure("Metal clear render encoder creation failed");
    // Queue visibility applies to work in this encoder, including attachment
    // loads. A barrier in a separate empty encoder cannot order this clear.
    [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                 visibilityOptions:MTL4VisibilityOptionDevice];
    const auto measured_pass = bindings.pass_counters
        ? bindings.pass_counters->begin(device, render, pass, "clear") : PassCounters::invalid;
    [render setRenderPipelineState:pipeline];
    [render setDepthStencilState:depth_state];
    [render setStencilFrontReferenceValue:plan.stencil_value
        backReferenceValue:plan.stencil_value];
    const NSUInteger width = bindings.color ? bindings.color.width
        : (bindings.depth ? bindings.depth.width : bindings.stencil.width);
    const NSUInteger height = bindings.color ? bindings.color.height
        : (bindings.depth ? bindings.depth.height : bindings.stencil.height);
    [render setViewport:MTLViewport{0.0, 0.0, double(width), double(height), 0.0, 1.0}];
    const bool full_region = plan.region.x == 0 && plan.region.y == 0 &&
                             plan.region.width == width &&
                             plan.region.height == height;
    if (!full_region) {
        // The scissor bounds the fragment writes, so texels outside the
        // region keep their loaded contents.
        [render setScissorRect:MTLScissorRect{NSUInteger(plan.region.x),
            NSUInteger(plan.region.y), NSUInteger(plan.region.width),
            NSUInteger(plan.region.height)}];
    }
    [render setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
    if (measured_pass != PassCounters::invalid) {
        auto& measured = bindings.pass_counters->passes[measured_pass];
        ++measured.draws;
        ++measured.pipeline_binds;
        ++measured.depth_binds;
        measured.depth_draws += bindings.depth != nil;
    }
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3
             instanceCount:layer_count > 1 ? NSUInteger(layer_count) : 1];
    if (bindings.pass_counters) bindings.pass_counters->end(render, measured_pass);
    [render endEncoding];
    if (!bindings.into) [commands endCommandBuffer];
    EncodedClear encoded;
    encoded.commands = commands;
    encoded.allocator = allocator;
    encoded.retained = retained;
    return encoded;
}

}  // namespace

graphics::RendererStatus
validate_clear(const graphics::ClearPlan& plan,
               const ClearBindings& bindings) noexcept {
    ValidatedBindings formats;
    if (const auto error = validate_bindings(plan, bindings, formats))
        return graphics::RendererError(*error);
    return graphics::renderer_success();
}

graphics::RendererResult<EncodedClear>
encode_clear(id<MTLDevice> device,
             const graphics::ClearPlan& plan,
             const ClearBindings& bindings) noexcept {
    if (!device) return encode_failure("Metal clear requires a device");
    ValidatedBindings formats;
    if (const auto error = validate_bindings(plan, bindings, formats))
        return encode_failure(error->detail);

    id<MTL4CommandAllocator> allocator = nil;
    id<MTLResidencySet> residency = bindings.into_residency;
    NSMutableArray* retained = [NSMutableArray array];
    if (!bindings.into) {
        allocator = [device newCommandAllocator];
        if (!allocator) return encode_failure("Metal clear allocator allocation failed");
        auto* residency_descriptor = [MTLResidencySetDescriptor new];
        residency_descriptor.initialCapacity = 8;
        NSError* error = nil;
        residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
        if (!residency) return encode_failure("Metal clear residency allocation failed");
        [retained addObject:allocator];
        [retained addObject:residency];
    } else if (!residency) {
        return encode_failure("Metal clear into a command buffer requires its residency set");
    }
    for (const id<MTLResource> resource :
         {bindings.color, bindings.depth, bindings.stencil}) {
        if (resource != nil) [residency addAllocation:resource];
    }
    if (!bindings.into) [residency commit];

    if (plan.fold_into_load || plan.fold_layers_into_load) {
        return encode_fold(device, allocator, plan, bindings, retained, residency);
    }
    return encode_operation(device, allocator, plan, bindings, formats,
                            retained, residency);
}

void prepare_clear_shaders(id<MTLDevice> device) noexcept {
    if (device) (void)clear_library(device);
}

bool prepare_clear_pipeline(id<MTLDevice> device, const ClearPipelineConfig& config) noexcept {
    static constexpr std::string_view vertices[]{"clear_vertex", "clear_vertex_layered"};
    static constexpr std::string_view fragments[]{"clear_color_depth_frag", "clear_color_frag", "clear_depth_frag",
                                                  "clear_stencil_frag", "clear_uint_frag", "clear_sint_frag"};
    if (!device || std::find(std::begin(vertices), std::end(vertices), config.vertex) == std::end(vertices) ||
        std::find(std::begin(fragments), std::end(fragments), config.fragment) == std::end(fragments) ||
        !config.samples || config.samples > 8 || (config.samples & (config.samples - 1)) ||
        ![device supportsTextureSampleCount:config.samples])
        return false;
    id<MTLLibrary> library = clear_library(device);
    return library && clear_pipeline(device, library, config);
}

std::vector<ClearCompile> take_clear_compiles() {
    return std::exchange(thread_clear_compiles, {});
}

}  // namespace armigo::platform::metal
