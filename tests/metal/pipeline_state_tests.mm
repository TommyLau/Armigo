// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// Synthetic, independently authored verification for the L9 Metal
// pipeline-state cache: pixel oracles for blend equations and factors, depth
// compare/write and front/back stencil; mutation coverage for every baked key
// field; dynamic-only changes that reuse the cached pipeline; forced hash
// collisions that stay distinct entries. All shaders and data are authored
// here; nothing is taken from a game.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <armigo/graphics/texture_format.h>
#include <attachments.h>
#include <device_context.h>
#include <native_pipeline_state.h>
#include <fragment_output.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <functional>
#include <optional>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;
namespace graphics = armigo::graphics;
namespace metal = armigo::platform::metal;

unsigned failures = 0;
#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                            \
            ++failures;                                                                             \
        }                                                                                           \
    } while (false)

constexpr std::uint32_t kWidth = 8;
constexpr std::uint32_t kHeight = 8;
constexpr std::size_t kPixelFloats = 4;  // RGBA32Float

metal::PipelineContextKey test_context(id<MTLDevice> device) {
    metal::PipelineContextKey context;
    context.device = device.registryID;
    context.compiler_chain = "synthetic-l9";
    context.capability_profile = "metal4-apple-silicon";
    return context;
}

class TestDeviceContext final : public metal::DeviceContext {
public:
    explicit TestDeviceContext(id<MTLDevice> device) : device_(device) {}
    id<MTLDevice> device() const noexcept override { return device_; }
    id<MTL4CommandQueue> queue() const noexcept override {
        if (!queue_) queue_ = [device_ newMTL4CommandQueue];
        return queue_;
    }
    const metal::PipelineContextKey& identity() const noexcept override {
        static const metal::PipelineContextKey context = test_context(device_);
        return context;
    }

private:
    id<MTLDevice> device_;
    mutable id<MTL4CommandQueue> queue_ = nil;
};

id<MTLDevice> ensure_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
        std::fprintf(stderr, "FAIL: unified-memory Metal 4 device is unavailable\n");
        std::exit(1);
    }
    return device;
}

// ---------------------------------------------------------------------------
// Synthetic shader library: one source, one serialized metallib blob, many
// entry points. Vertex stages place a six-vertex quad whose left and right
// triangles carry opposite winding; fragment stages return fixed constants.
// ---------------------------------------------------------------------------

std::string shader_source() {
    std::string source = R"(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; };
constant float2 k_quad[6] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                             float2(-1.0, 1.0), float2(1.0, -1.0), float2(1.0, 1.0)};
// The split quad draws opposite-wound halves for the front/back stencil test.
constant float2 k_split[6] = {float2(-1.0, -1.0), float2(0.0, -1.0), float2(-1.0, 1.0),
                              float2(0.0, -1.0), float2(0.0, 1.0),  float2(1.0, -1.0)};
vertex VOut vs_plain(uint vid [[vertex_id]]) {
    VOut out; out.position = float4(k_quad[vid], 0.5, 1.0); return out;
}
vertex VOut vs_split(uint vid [[vertex_id]]) {
    VOut out; out.position = float4(k_split[vid], 0.5, 1.0); return out;
}
vertex VOut vs_z025(uint vid [[vertex_id]]) {
    VOut out; out.position = float4(k_quad[vid], 0.25, 1.0); return out;
}
vertex VOut vs_z075(uint vid [[vertex_id]]) {
    VOut out; out.position = float4(k_quad[vid], 0.75, 1.0); return out;
}
// Metal requires a void-returning vertex function when rasterization is off.
vertex void vs_void(uint vid [[vertex_id]]) {
    static_cast<void>(vid);
}
fragment float4 fs_dummy() { return float4(0.0, 0.0, 0.0, 1.0); }
fragment uint4 fs_uint_out() { return uint4(64, 128, 200, 255); }
struct SparseOutput {
    float4 bound [[color(0)]];
    float4 unbound [[color(2)]];
};
fragment SparseOutput fs_sparse_out() {
    return {float4(0.2, 0.4, 0.6, 1.0), float4(1.0, 0.0, 1.0, 1.0)};
}
fragment void fs_no_color() {}
constant uint output_class [[function_constant(0)]];
constant bool output_float = output_class == 0;
constant bool output_uint = output_class == 1;
constant bool output_sint = output_class == 2;
struct AdaptiveOutput {
    float4 f [[color(0), function_constant(output_float)]];
    uint4 u [[color(0), function_constant(output_uint)]];
    int4 s [[color(0), function_constant(output_sint)]];
};
fragment AdaptiveOutput fs_adaptive() {
    AdaptiveOutput result;
    uint4 bits = uint4(0xffffffffu, 0x80000000u, 0x7fc12345u, 0x01000001u);
    if (output_float) result.f = as_type<float4>(bits);
    if (output_uint) result.u = bits;
    if (output_sint) result.s = as_type<int4>(bits);
    return result;
}
constant bool shader_depth_bias [[function_constant(8)]];
struct DepthBiasOutput {
    float4 color [[color(0)]];
    float depth [[depth(any), function_constant(shader_depth_bias)]];
};
fragment DepthBiasOutput fs_depth_bias() {
    DepthBiasOutput result;
    result.color = float4(0.3, 0.5, 0.7, 1.0);
    if (shader_depth_bias) result.depth = 0.25;
    return result;
}
fragment float4 fs_alpha() { return float4(0.8, 0.6, 0.4, 0.25); }
constant bool armigo_y_negate [[function_constant(9)]];
fragment float4 fs_y_direction() {
    return float4(armigo_y_negate ? 0.25 : 0.75, 0.0, 0.0, 1.0);
}
struct DepthExportOutput {
    float depth [[depth(any)]];
};
fragment DepthExportOutput fs_depth_export() {
    DepthExportOutput result;
    result.depth = 0.5;
    return result;
}
struct SampleMaskOutput {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};
fragment SampleMaskOutput fs_sample_mask() {
    SampleMaskOutput result;
    result.color = float4(0.25, 0.5, 0.75, 1.0);
    result.mask = 1u;
    return result;
}
)";
    for (unsigned index = 0; index < 6; ++index) {
        const float red = 0.1f * (index + 1);
        const float green = 0.2f + 0.05f * index;
        const float blue = 0.9f - 0.1f * index;
        source += "fragment float4 fs_c" + std::to_string(index) +
                  "() { return float4(" + std::to_string(red) + ", " + std::to_string(green) +
                  ", " + std::to_string(blue) + ", 1.0); }\n";
    }
    return source;
}

// Compiles the synthetic source with the same offline toolchain the
// repository's fixture wrapper uses (xcrun metal + metallib), yielding the
// executable-type library the runtime cache loads through newLibraryWithData.
std::vector<std::byte> compile_metallib(const std::string& source) {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("armigo-l9-pso-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "shader.metal";
    const auto air_path = directory / "shader.air";
    const auto library_path = directory / "shader.metallib";
    std::ofstream stream(source_path, std::ios::binary);
    stream << source;
    stream.close();
    const auto run_tool = [](const std::string& command) {
        const int status = std::system(command.c_str());
        if (status != 0) {
            std::fprintf(stderr, "FAIL: fixture compilation failed: %s\n", command.c_str());
            ++failures;
        }
        return status == 0;
    };
    const std::string triple = "xcrun --sdk macosx ";
    const bool compiled = run_tool(
        triple + "metal -std=metal4.0 -mmacosx-version-min=26.0 -Werror -c " +
        source_path.string() + " -o " + air_path.string()) &&
        run_tool(triple + "metallib " + air_path.string() + " -o " + library_path.string());
    std::vector<std::byte> bytes;
    if (compiled) {
        std::ifstream input(library_path, std::ios::binary);
        const std::vector<char> data{std::istreambuf_iterator<char>(input), {}};
        const auto* raw = reinterpret_cast<const std::byte*>(data.data());
        bytes.assign(raw, raw + data.size());
    }
    std::error_code cleanup;
    std::filesystem::remove_all(directory, cleanup);
    return bytes;
}

metal::AdmittedGraphicsStages make_stages(
    const std::vector<std::byte>& metallib, const std::string& vertex_entry,
    const std::string& fragment_entry,
    const std::vector<armigo::platform::CompiledColorOutput>& fragment_outputs = {},
    bool writes_fragment_depth = false, bool writes_sample_mask = false) {
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    for (auto* stage : {vertex.get(), fragment.get()}) stage->metallib = metallib;
    vertex->stage = armigo::shader_input::Stage::Vertex;
    vertex->entry_point = vertex_entry;
    fragment->stage = armigo::shader_input::Stage::Fragment;
    fragment->entry_point = fragment_entry;
    fragment->color_outputs = fragment_outputs;
    fragment->writes_fragment_depth = writes_fragment_depth;
    fragment->writes_sample_mask = writes_sample_mask;
    metal::ShaderFunctionKey vertex_key;
    vertex_key.package = "synthetic";
    vertex_key.variant = "v1";
    vertex_key.artifact = vertex_entry;
    vertex_key.entry_point = vertex_entry;
    metal::ShaderFunctionKey fragment_key = vertex_key;
    fragment_key.artifact = fragment_entry;
    fragment_key.entry_point = fragment_entry;
    return metal::AdmittedGraphicsStages{armigo::platform::CompiledGraphicsPair{vertex, fragment},
                                         vertex_key, fragment_key};
}

// ---------------------------------------------------------------------------
// Key builders.
// ---------------------------------------------------------------------------

metal::PipelineKey base_key(const metal::PipelineStateCache& cache) {
    metal::PipelineKey key;
    key.context = cache.context();
    key.vertex.package = key.fragment.package = "synthetic";
    key.vertex.variant = key.fragment.variant = "v1";
    key.vertex.artifact = key.vertex.entry_point = "vs_plain";
    key.fragment.artifact = key.fragment.entry_point = "fs_c0";
    key.colors[0].format = TextureFormat::R32G32B32A32Float;
    key.primitive = PrimitiveClass::Triangle;
    key.rasterization_enabled = true;
    key.coverage.samples = 1;
    key.coverage.sample_mask = ~0u;
    key.coverage.multisample = false;
    return key;
}

metal::PipelineKey keyed(const metal::PipelineStateCache& cache, const char* vertex_entry,
                         const char* fragment_entry) {
    metal::PipelineKey key = base_key(cache);
    key.vertex.artifact = key.vertex.entry_point = vertex_entry;
    key.fragment.artifact = key.fragment.entry_point = fragment_entry;
    return key;
}

const graphics::RendererError* error_of(const RendererStatus& status) {
    return std::get_if<RendererError>(&status);
}
template <class T>
const graphics::RendererError* error_of(const RendererResult<T>& result) {
    return std::get_if<RendererError>(&result);
}
std::string error_text(const graphics::RendererError& error) {
    return std::string(error.command().detail);
}

bool acquire_ok(metal::PipelineStateCache& cache, const metal::PipelineKey& key,
                const metal::AdmittedGraphicsStages& stages, metal::ResidentPipeline& out) {
    auto acquired = cache.acquire(key, stages);
    if (const auto* error = error_of(acquired)) {
        std::fprintf(stderr, "FAIL: acquire rejected a valid key: %s\n",
                     error_text(*error).c_str());
        ++failures;
        return false;
    }
    out = std::get<metal::ResidentPipeline>(std::move(acquired));
    return true;
}

// ---------------------------------------------------------------------------
// GPU harness: offscreen RGBA32Float passes with optional depth/stencil.
// ---------------------------------------------------------------------------

struct Canvas {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTLTexture> color = nil;
    id<MTLTexture> depth_stencil = nil;  // Depth32Float_Stencil8 when used
    bool stencil_used = false;
};

Canvas make_canvas(id<MTLDevice> device, id<MTL4CommandQueue> queue, bool depth, bool stencil) {
    Canvas canvas;
    canvas.device = device;
    canvas.queue = queue;
    auto* color_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                    width:kWidth height:kHeight mipmapped:NO];
    color_descriptor.storageMode = MTLStorageModePrivate;
    color_descriptor.usage = MTLTextureUsageRenderTarget;
    canvas.color = [device newTextureWithDescriptor:color_descriptor];
    CHECK(canvas.color != nil);
    if (depth || stencil) {
        auto* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                        width:kWidth height:kHeight mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget;
        canvas.depth_stencil = [device newTextureWithDescriptor:descriptor];
        canvas.stencil_used = stencil;
        CHECK(canvas.depth_stencil != nil);
    }
    return canvas;
}

struct DrawStep {
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLDepthStencilState> depth_stencil = nil;
    std::optional<std::array<float, 4>> blend_color;
    std::optional<std::pair<std::uint32_t, std::uint32_t>> stencil_references;
};

struct ClearValues {
    std::array<float, 4> color{0.0f, 0.0f, 0.0f, 1.0f};
    double depth = 1.0;
    std::uint32_t stencil = 0;
};

// Runs one render pass of DrawSteps and returns width*height RGBA float pixels.
std::vector<float> run_pass(const Canvas& canvas, const ClearValues& clear,
                            const std::vector<DrawStep>& steps) {
    id<MTL4CommandBuffer> commands = [canvas.device newCommandBuffer];
    id<MTL4CommandAllocator> allocator = [canvas.device newCommandAllocator];
    CHECK(commands != nil && allocator != nil);
    [commands beginCommandBufferWithAllocator:allocator];
    auto* pass = [MTL4RenderPassDescriptor new];
    auto* color_attachment = pass.colorAttachments[0];
    color_attachment.texture = canvas.color;
    color_attachment.loadAction = MTLLoadActionClear;
    color_attachment.storeAction = MTLStoreActionStore;
    color_attachment.clearColor = MTLClearColorMake(clear.color[0], clear.color[1],
                                                    clear.color[2], clear.color[3]);
    if (canvas.depth_stencil) {
        pass.depthAttachment.texture = canvas.depth_stencil;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = clear.depth;
        if (canvas.stencil_used) {
            pass.stencilAttachment.texture = canvas.depth_stencil;
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = clear.stencil;
        }
    }
    id<MTL4RenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    CHECK(encoder != nil);
    NSMutableArray* retained = [NSMutableArray arrayWithArray:@[commands, allocator, pass]];
    for (const auto& step : steps) {
        [encoder setRenderPipelineState:step.pipeline];
        if (step.depth_stencil) [encoder setDepthStencilState:step.depth_stencil];
        if (step.blend_color)
            [encoder setBlendColorRed:step.blend_color->at(0) green:step.blend_color->at(1)
                                 blue:step.blend_color->at(2) alpha:step.blend_color->at(3)];
        if (step.stencil_references)
            [encoder setStencilFrontReferenceValue:step.stencil_references->first
                                backReferenceValue:step.stencil_references->second];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    }
    [encoder endEncoding];

    id<MTLBuffer> readback = [canvas.device
        newBufferWithLength:kWidth * kHeight * kPixelFloats * sizeof(float)
        options:MTLResourceStorageModeShared];
    CHECK(readback != nil);
    [retained addObject:readback];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    CHECK(copy != nil);
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
               visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:canvas.color sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                toBuffer:readback destinationOffset:0
    destinationBytesPerRow:kWidth * kPixelFloats * sizeof(float)
    destinationBytesPerImage:0];
    [copy endEncoding];
    [commands endCommandBuffer];

    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [canvas.queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) !=
        0) {
        std::fprintf(stderr, "FAIL: pixel pass timed out\n");
        ++failures;
        return {};
    }
    if (execution_error) {
        std::fprintf(stderr, "FAIL: pixel pass failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        ++failures;
        return {};
    }
    const auto* pixels = static_cast<const float*>(readback.contents);
    return {pixels, pixels + kWidth * kHeight * kPixelFloats};
}

std::array<float, 4> pixel_at(const std::vector<float>& pixels, std::uint32_t x,
                              std::uint32_t y) {
    const std::size_t base = (std::size_t{y} * kWidth + x) * kPixelFloats;
    return {pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]};
}

bool close_to(float left, float right) { return std::fabs(left - right) < 1.0e-5f; }

bool same_color(const std::array<float, 4>& pixel, const std::array<float, 4>& expected) {
    for (std::size_t i = 0; i < 4; ++i)
        if (!close_to(pixel[i], expected[i])) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Forced hash collision. The implementation hashes the shader_emulation tag
// as its low 32 bits, so any two tags sharing that word land in one bucket
// while the keys stay distinct and equality must separate them.
// ---------------------------------------------------------------------------

std::pair<std::uint64_t, std::uint64_t> colliding_emulation_values() {
    return {0x00000001'0000002Aull, 0x00000002'0000002Aull};
}

// Type-erased resource accessors matching pipeline_state.mm's leases.
id<MTLRenderPipelineState> pso_of(const metal::ResidentPipeline& resident) {
    return dynamic_cast<const metal::MetalRenderPipelineResource&>(*resident.pipeline).pipeline();
}

id<MTLDepthStencilState> depth_stencil_of(graphics::RetainedResource resource) {
    return dynamic_cast<const metal::MetalDepthStencilResource&>(*resource).depth_stencil();
}

metal::ResidentPipeline acquire_pipeline(metal::PipelineStateCache& cache,
                                         const std::vector<std::byte>& metallib,
                                         const char* vertex_entry, const char* fragment_entry,
                                         const std::function<void(metal::PipelineKey&)>& tune = {}) {
    const auto stages = make_stages(metallib, vertex_entry, fragment_entry);
    auto key = keyed(cache, vertex_entry, fragment_entry);
    if (tune) tune(key);
    metal::ResidentPipeline resident;
    CHECK(acquire_ok(cache, key, stages, resident));
    return resident;
}

}  // namespace

int main() {
    id<MTLDevice> device = ensure_device();
    auto device_context = std::make_shared<TestDeviceContext>(device);
    auto cache_result = metal::create_pipeline_state_cache(device_context, 64ull * 1024 * 1024);
    CHECK(!error_of(cache_result));
    if (error_of(cache_result)) {
        std::fprintf(stderr, "FAIL: cache creation failed: %s\n",
                     error_text(*error_of(cache_result)).c_str());
        return 1;
    }
    auto& cache = **std::get_if<std::shared_ptr<metal::PipelineStateCache>>(&cache_result);
    const auto metallib = compile_metallib(shader_source());
    CHECK(!metallib.empty());
    if (metallib.empty()) return 1;

    // -- factory validation -----------------------------------------------------
    {
        auto empty_context = metal::create_pipeline_state_cache(nullptr, 1024);
        CHECK(error_of(empty_context) != nullptr);
        auto zero_budget = metal::create_pipeline_state_cache(device_context, 0);
        CHECK(error_of(zero_budget) != nullptr);
        CHECK(cache.context() == test_context(device));
    }

    // -- argument-buffer sampler capacity and lease-aware reclamation ---------
    {
        const auto sampler_limit =
            static_cast<std::uint64_t>(device.maxArgumentBufferSamplerCount);
        CHECK(sampler_limit != 0);
        if (sampler_limit != 0 && sampler_limit <= 4096) {
            auto limited_result =
                metal::create_pipeline_state_cache(device_context, 64ull * 1024 * 1024);
            CHECK(!error_of(limited_result));
            if (!error_of(limited_result)) {
                auto limited =
                    std::get<std::shared_ptr<metal::PipelineStateCache>>(limited_result);
                std::vector<graphics::RetainedResource> live_samplers;
                live_samplers.reserve(static_cast<std::size_t>(sampler_limit));
                for (std::uint64_t index = 0; index < sampler_limit; ++index) {
                    SamplerDescriptor descriptor;
                    descriptor.max_lod = static_cast<float>(index + 1);
                    auto created = limited->sampler(descriptor);
                    CHECK(!error_of(created));
                    if (error_of(created)) break;
                    live_samplers.push_back(
                        std::get<graphics::RetainedResource>(std::move(created)));
                }
                CHECK(live_samplers.size() == sampler_limit);
                if (live_samplers.size() == sampler_limit) {
                    SamplerDescriptor overflow;
                    overflow.max_lod = static_cast<float>(sampler_limit + 1);
                    const auto exhausted = limited->sampler(overflow);
                    CHECK(error_of(exhausted) != nullptr);
                    if (const auto* error = error_of(exhausted)) {
                        CHECK(error->command().status == CommandStatus::BudgetExceeded);
                        CHECK(error_text(*error).find("maxArgumentBufferSamplerCount") !=
                              std::string::npos);
                    }

                    live_samplers.front().reset();
                    auto reclaimed = limited->sampler(overflow);
                    CHECK(!error_of(reclaimed));
                }
            }
        }
    }

    // -- reuse: identical keys return one pipeline/state/sampler --------------
    metal::ResidentPipeline baseline;
    {
        const auto stages = make_stages(metallib, "vs_plain", "fs_c0");
        const auto key = keyed(cache, "vs_plain", "fs_c0");
        CHECK(acquire_ok(cache, key, stages, baseline));
        metal::ResidentPipeline again;
        CHECK(acquire_ok(cache, key, stages, again));
        CHECK(baseline.pipeline == again.pipeline);

        const DepthStencilState depth_key;
        auto depth_first = cache.depth_stencil(depth_key);
        auto depth_second = cache.depth_stencil(depth_key);
        CHECK(!error_of(depth_first) && !error_of(depth_second));
        CHECK(std::get<graphics::RetainedResource>(depth_first) ==
              std::get<graphics::RetainedResource>(depth_second));

        SamplerDescriptor sampler_key;
        sampler_key.minification = SamplerFilter::Nearest;
        sampler_key.magnification = SamplerFilter::Nearest;
        sampler_key.mip_filter = SamplerMipFilter::None;
        sampler_key.raw_words = {1, 2, 3, 4, 5, 6, 7, 8};
        auto sampler_first = cache.sampler(sampler_key);
        CHECK(!error_of(sampler_first));
        // raw_words records the decoded TSC input, not admitted state:
        // changing it must NOT allocate a second sampler.
        sampler_key.raw_words = {9, 9, 9, 9, 9, 9, 9, 9};
        auto sampler_again = cache.sampler(sampler_key);
        CHECK(!error_of(sampler_again));
        CHECK(std::get<graphics::RetainedResource>(sampler_first) ==
              std::get<graphics::RetainedResource>(sampler_again));
        // A real state change does.
        SamplerDescriptor modified_sampler = sampler_key;
        modified_sampler.magnification = SamplerFilter::Linear;
        auto sampler_other = cache.sampler(modified_sampler);
        CHECK(!error_of(sampler_other));
        CHECK(std::get<graphics::RetainedResource>(sampler_first) !=
              std::get<graphics::RetainedResource>(sampler_other));
        modified_sampler = sampler_key;
        modified_sampler.srgb_border_color = {17, 93, 211};
        auto sampler_packed = cache.sampler(modified_sampler);
        CHECK(!error_of(sampler_packed));
        CHECK(std::get<graphics::RetainedResource>(sampler_first) !=
              std::get<graphics::RetainedResource>(sampler_packed));
        modified_sampler.wrap.fill(SamplerWrap::ClampToBorder);
        modified_sampler.border_color = {0.2f, 0.4f, 0.7f, 0.3f};
        auto sampler_custom = cache.sampler(modified_sampler);
        CHECK(!error_of(sampler_custom));
        auto sampler_custom_again = cache.sampler(modified_sampler);
        CHECK(!error_of(sampler_custom_again));
        CHECK(std::get<graphics::RetainedResource>(sampler_custom) ==
              std::get<graphics::RetainedResource>(sampler_custom_again));
        modified_sampler.border_color[0] = 0.6f;
        auto sampler_custom_changed = cache.sampler(modified_sampler);
        CHECK(!error_of(sampler_custom_changed));
        CHECK(std::get<graphics::RetainedResource>(sampler_custom) !=
              std::get<graphics::RetainedResource>(sampler_custom_changed));

        // Dynamic-only state (viewport, scissor, blend constant, stencil
        // references, cull mode, winding, fill mode, depth bias) has no field
        // in the frozen key: equal keys keep reusing the same pipeline no
        // matter how the encoder-level state differs around them. The dynamic
        // snapshot below differs on every such field, yet the key - and so
        // the cached pipeline - is unchanged.
        RasterizerState dynamic_raster;  // cull/front/fill all differ from defaults below
        dynamic_raster.cull = CullMode::Front;
        dynamic_raster.front_face = FrontFace::Clockwise;
        dynamic_raster.front_fill = FillMode::Line;
        std::vector<ViewportState> dynamic_viewports(1);
        dynamic_viewports[0] = {1.0, 2.0, 3.0, 4.0, 0.25, 0.75};
        RenderState dynamic_state;
        dynamic_state.rasterizer = dynamic_raster;
        dynamic_state.viewports = dynamic_viewports;
        dynamic_state.blend_constant = {0.25f, 0.5f, 0.75f, 1.0f};
        dynamic_state.front_stencil_reference = 7;
        dynamic_state.back_stencil_reference = 9;
        dynamic_state.depth_bias = {true, true, true, 1.5f, 2.5f, 3.5f};
        static_cast<void>(dynamic_state);
        metal::ResidentPipeline dynamic_variant;
        CHECK(acquire_ok(cache, key, stages, dynamic_variant));
        CHECK(baseline.pipeline == dynamic_variant.pipeline);
    }

    // -- mutation coverage: every baked key field -----------------------------
    {
        const auto stages = make_stages(metallib, "vs_plain", "fs_c0");
        const auto key = keyed(cache, "vs_plain", "fs_c0");
        struct Mutation {
            const char* name;
            bool unsupported;  // true when the change must be structured stop
            std::function<void(metal::PipelineKey&)> apply;
            const char* vertex_entry = "vs_plain";
        };
        const std::vector<Mutation> mutations = {
            {"context.device", true,
             [&](metal::PipelineKey& k) { k.context.device = 0xDEADBEEF; }},
            {"context.compiler_chain", true,
             [&](metal::PipelineKey& k) { k.context.compiler_chain = "other"; }},
            {"context.capability_profile", true,
             [&](metal::PipelineKey& k) { k.context.capability_profile = "other"; }},
            {"vertex.package", true, [&](metal::PipelineKey& k) { k.vertex.package = "other"; }},
            {"vertex.variant", true, [&](metal::PipelineKey& k) { k.vertex.variant = "other"; }},
            {"vertex.artifact", true, [&](metal::PipelineKey& k) { k.vertex.artifact = "gone"; }},
            {"vertex.entry_point", true,
             [&](metal::PipelineKey& k) { k.vertex.entry_point = "vs_z025"; }},
            {"fragment.package", true,
             [&](metal::PipelineKey& k) { k.fragment.package = "other"; }},
            {"fragment.variant", true,
             [&](metal::PipelineKey& k) { k.fragment.variant = "other"; }},
            {"fragment.artifact", true,
             [&](metal::PipelineKey& k) { k.fragment.artifact = "gone"; }},
            {"fragment.entry_point", true,
             [&](metal::PipelineKey& k) { k.fragment.entry_point = "fs_c1"; }},
            {"vertex specialization added", true,
             [&](metal::PipelineKey& k) {
                 k.vertex.specialization.push_back({0, "u32", {std::byte{1}}});
             }},
            {"fragment specialization added", true,
             [&](metal::PipelineKey& k) {
                 k.fragment.specialization.push_back({0, "u32", {std::byte{1}}});
             }},
            {"attribute added", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 0, VertexAttribFormat::R32G32B32A32Float});
                 k.buffers.push_back({0, 16, VertexStep::PerVertex, 1});
             }},
            {"attribute format", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 0, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({0, 12, VertexStep::PerVertex, 1});
             }},
            {"attribute offset", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 4, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({0, 16, VertexStep::PerVertex, 1});
             }},
            {"buffer index", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 1, 0, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({1, 12, VertexStep::PerVertex, 1});
             }},
            {"buffer stride", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 0, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({0, 32, VertexStep::PerVertex, 1});
             }},
            {"buffer step", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 0, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({0, 12, VertexStep::PerInstance, 1});
             }},
            {"buffer rate", false,
             [&](metal::PipelineKey& k) {
                 k.attributes.push_back({0, 0, 0, VertexAttribFormat::R32G32B32Float});
                 k.buffers.push_back({0, 12, VertexStep::PerInstance, 2});
             }},
            {"color format", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].format = TextureFormat::R16G16B16A16Float;
             }},
            {"color blend enabled", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::SourceAlpha,
                                          BlendFactor::OneMinusSourceAlpha};
                 k.colors[0].blend.alpha = {BlendOperation::Add, BlendFactor::One,
                                            BlendFactor::OneMinusSourceAlpha};
             }},
            {"color rgb operation", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Max, BlendFactor::One,
                                          BlendFactor::One};
             }},
            {"color rgb source factor", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::One,
                                          BlendFactor::Zero};
             }},
            {"color rgb destination factor", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::SourceAlpha,
                                          BlendFactor::One};
             }},
            {"color alpha operation", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.alpha = {BlendOperation::Min, BlendFactor::One,
                                            BlendFactor::One};
             }},
            {"color alpha source factor", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.alpha = {BlendOperation::Add, BlendFactor::Zero,
                                            BlendFactor::One};
             }},
            {"color alpha destination factor", false,
             [&](metal::PipelineKey& k) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.alpha = {BlendOperation::Add, BlendFactor::One,
                                            BlendFactor::SourceAlpha};
             }},
            {"color write mask", false,
             [&](metal::PipelineKey& k) { k.colors[0].blend.write_mask = 0x5; }},
            {"second target format", false,
             [&](metal::PipelineKey& k) {
                 k.colors[1].format = TextureFormat::R16G16B16A16Float;
             }},
            {"depth format", false,
             [&](metal::PipelineKey& k) { k.depth_format = TextureFormat::D32Float; }},
            {"stencil format", false,
             [&](metal::PipelineKey& k) {
                 k.depth_format = TextureFormat::D32FloatS8Uint;
                 k.stencil_format = TextureFormat::D32FloatS8Uint;
             }},
            {"coverage samples", false,
             [&](metal::PipelineKey& k) {
                 k.coverage.samples = 4;
                 k.coverage.multisample = true;
             }},
            {"coverage sample mask", false,
             [&](metal::PipelineKey& k) { k.coverage.sample_mask = 0x3u; }},
            {"coverage multisample flag", true,
             [&](metal::PipelineKey& k) { k.coverage.multisample = true; }},
            {"coverage alpha to coverage", false,
             [&](metal::PipelineKey& k) {
                 k.coverage.samples = 4;
                 k.coverage.multisample = true;
                 k.coverage.alpha_to_coverage = true;
             }},
            {"coverage alpha to one", false,
             [&](metal::PipelineKey& k) {
                 k.coverage.samples = 4;
                 k.coverage.multisample = true;
                 k.coverage.alpha_to_one = true;
             }},
            {"primitive class", false,
             [&](metal::PipelineKey& k) { k.primitive = PrimitiveClass::Line; }},
            {"rasterization enabled", false,
             [&](metal::PipelineKey& k) { k.rasterization_enabled = false; },
             "vs_void"},
            {"shader emulation tag", false,
             [&](metal::PipelineKey& k) { k.shader_emulation = 0x1234; }},
        };
        for (const auto& mutation : mutations) {
            metal::PipelineKey mutated = key;
            mutation.apply(mutated);
            const bool overridden = std::string(mutation.vertex_entry) != "vs_plain";
            if (overridden) {
                mutated.vertex.artifact = mutated.vertex.entry_point = mutation.vertex_entry;
            }
            const auto mutation_stages =
                overridden ? make_stages(metallib, mutation.vertex_entry, "fs_c0") : stages;
            auto acquired = cache.acquire(mutated, mutation_stages);
            if (mutation.unsupported) {
                if (!error_of(acquired)) {
                    // Context and stage-identity changes must never be served
                    // from the cached entry; creation may legitimately fail.
                    std::fprintf(stderr, "FAIL: mutation %s was served as a pipeline\n",
                                 mutation.name);
                    ++failures;
                }
                continue;
            }
            if (const auto* error = error_of(acquired)) {
                std::fprintf(stderr, "FAIL: mutation %s failed: %s\n", mutation.name,
                             error_text(*error).c_str());
                ++failures;
                continue;
            }
            const auto& resident = std::get<metal::ResidentPipeline>(acquired);
            if (resident.pipeline == baseline.pipeline) {
                std::fprintf(stderr, "FAIL: mutation %s reused the cached pipeline\n",
                             mutation.name);
                ++failures;
            }
            // The untouched key must still hit its original entry.
            metal::ResidentPipeline original;
            CHECK(acquire_ok(cache, key, stages, original));
            CHECK(original.pipeline == baseline.pipeline);
        }
    }

    // -- forced hash collisions stay distinct ----------------------------------
    {
        const auto stages = make_stages(metallib, "vs_plain", "fs_c0");
        auto key_a = keyed(cache, "vs_plain", "fs_c0");
        auto key_b = key_a;
        const auto [emulation_a, emulation_b] = colliding_emulation_values();
        key_a.shader_emulation = emulation_a;
        key_b.shader_emulation = emulation_b;
        CHECK(key_a != key_b);
        metal::ResidentPipeline resident_a, resident_b;
        CHECK(acquire_ok(cache, key_a, stages, resident_a));
        CHECK(acquire_ok(cache, key_b, stages, resident_b));
        CHECK(resident_a.pipeline != resident_b.pipeline);
        metal::ResidentPipeline repeat_a, repeat_b;
        CHECK(acquire_ok(cache, key_a, stages, repeat_a));
        CHECK(acquire_ok(cache, key_b, stages, repeat_b));
        CHECK(repeat_a.pipeline == resident_a.pipeline);
        CHECK(repeat_b.pipeline == resident_b.pipeline);
    }

    // -- structured rejections --------------------------------------------------
    {
        struct Rejection {
            const char* name;
            std::function<void(metal::PipelineKey&, metal::AdmittedGraphicsStages&)> apply;
        };
        const std::vector<Rejection> rejections = {
            {"key contradicts admitted vertex key",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.vertex.package = "other";
             }},
            {"key contradicts admitted fragment key",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.fragment.entry_point = "other";
             }},
            {"dual-source blend factor",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::Source1Color,
                                          BlendFactor::One};
             }},
            {"blend on integer attachment",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages& s) {
                 s = make_stages(metallib, "vs_plain", "fs_uint_out",
                                 {armigo::platform::CompiledColorOutput{
                                     0, 0xF, armigo::platform::ColorOutputScalar::f32, 4}});
                 k.vertex.artifact = k.vertex.entry_point = "vs_plain";
                 k.fragment.artifact = k.fragment.entry_point = "fs_uint_out";
                 k.colors[0].format = TextureFormat::R8G8B8A8Uint;
                 k.colors[0].blend.enabled = true;
                 k.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::One,
                                          BlendFactor::Zero};
                 k.colors[0].blend.alpha = {BlendOperation::Add, BlendFactor::One,
                                            BlendFactor::Zero};
             }},
            {"float output against integer target",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages& s) {
                 s = make_stages(metallib, "vs_plain", "fs_uint_out",
                                 {armigo::platform::CompiledColorOutput{
                                     0, 0xF, armigo::platform::ColorOutputScalar::f32, 4}});
                 k.vertex.artifact = k.vertex.entry_point = "vs_plain";
                 k.fragment.artifact = k.fragment.entry_point = "fs_uint_out";
                 k.colors[0].format = TextureFormat::R8G8B8A8Uint;
             }},
            {"non-canonical attribute order",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.buffers = {{0, 16, VertexStep::PerVertex, 1},
                              {1, 16, VertexStep::PerVertex, 1}};
                 k.attributes = {{2, 0, 0, VertexAttribFormat::R32G32B32A32Float},
                                 {1, 1, 0, VertexAttribFormat::R32G32B32A32Float}};
             }},
            {"attribute without buffer layout",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.attributes = {{0, 3, 0, VertexAttribFormat::R32G32B32A32Float}};
             }},
            {"stride beyond Metal maximum",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.attributes = {{0, 0, 0, VertexAttribFormat::R32G32B32A32Float}};
                 k.buffers = {{0, 0x10000, VertexStep::PerVertex, 1}};
             }},
            {"instance rate zero",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.attributes = {{0, 0, 0, VertexAttribFormat::R32G32B32A32Float}};
                 k.buffers = {{0, 16, VertexStep::PerInstance, 0}};
             }},
            {"unsupported sample count",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 CHECK(![device supportsTextureSampleCount:3]);
                 k.coverage.samples = 3;
             }},
            {"alpha to coverage without multisample",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.coverage.alpha_to_coverage = true;
             }},
            {"stencil attachment without depth",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.stencil_format = TextureFormat::D32FloatS8Uint;
             }},
            {"unrenderable color format",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.colors[0].format = TextureFormat::R32G32B32Float;
             }},
            {"depth-compressed color format",
             [&](metal::PipelineKey& k, metal::AdmittedGraphicsStages&) {
                 k.colors[0].format = TextureFormat::Bc3Unorm;
             }},
        };
        for (const auto& rejection : rejections) {
            auto key = keyed(cache, "vs_plain", "fs_c0");
            auto stages_copy = make_stages(metallib, "vs_plain", "fs_c0");
            rejection.apply(key, stages_copy);
            const auto acquired = cache.acquire(key, stages_copy);
            if (!error_of(acquired)) {
                std::fprintf(stderr, "FAIL: rejection %s created a pipeline\n", rejection.name);
                ++failures;
            }
        }

        // An attribute may extend past its stream stride (overlapping records):
        // Maxwell and Metal both fetch vertex * stride + offset exactly.
        {
            metal::ResidentPipeline overlapping;
            auto key = keyed(cache, "vs_plain", "fs_c0");
            auto stages_copy = make_stages(metallib, "vs_plain", "fs_c0");
            key.attributes = {{0, 0, 8, VertexAttribFormat::R32G32B32A32Float}};
            key.buffers = {{0, 12, VertexStep::PerVertex, 1}};
            if (!acquire_ok(cache, key, stages_copy, overlapping)) {
                std::fprintf(stderr, "FAIL: attribute past its stride did not acquire\n");
                ++failures;
            }
        }

        // A constant (zero) stream keyed with stride zero still creates a
        // pipeline: Metal validates a nonzero stride even for constant steps.
        {
            metal::ResidentPipeline constant;
            auto key = keyed(cache, "vs_plain", "fs_c0");
            auto stages_copy = make_stages(metallib, "vs_plain", "fs_c0");
            key.attributes = {{0, 0, 0, VertexAttribFormat::R32G32B32A32Float}};
            key.buffers = {{0, 0, VertexStep::Constant, 1}};
            if (!acquire_ok(cache, key, stages_copy, constant)) {
                std::fprintf(stderr, "FAIL: constant vertex stream did not acquire\n");
                ++failures;
            }
        }

        // Fragment depth and sample-mask exports have admitted pipeline
        // paths. Each export acquires against its real fixture entry point,
        // the sample-mask export is admitted without multisampling, a
        // multisampled variant gets its own pipeline, and repeats hit the
        // same cache entry.
        {
            metal::ResidentPipeline depth_export, sample_mask, repeat;
            const auto depth_stages =
                make_stages(metallib, "vs_plain", "fs_depth_export", {}, true, false);
            const auto depth_key = keyed(cache, "vs_plain", "fs_depth_export");
            if (!acquire_ok(cache, depth_key, depth_stages, depth_export)) {
                std::fprintf(stderr, "FAIL: fragment depth export did not acquire\n");
                ++failures;
            }
            const auto mask_stages =
                make_stages(metallib, "vs_plain", "fs_sample_mask", {}, false, true);
            auto mask_key = keyed(cache, "vs_plain", "fs_sample_mask");
            if (!acquire_ok(cache, mask_key, mask_stages, sample_mask)) {
                std::fprintf(stderr, "FAIL: fragment sample-mask export did not acquire\n");
                ++failures;
            }
            CHECK(acquire_ok(cache, mask_key, mask_stages, repeat));
            CHECK(sample_mask.pipeline == repeat.pipeline);
            CHECK(depth_export.pipeline != sample_mask.pipeline);
            auto multisampled = mask_key;
            multisampled.coverage.samples = 4;
            multisampled.coverage.multisample = true;
            metal::ResidentPipeline multisample_mask;
            if (acquire_ok(cache, multisampled, mask_stages, multisample_mask))
                CHECK(multisample_mask.pipeline != sample_mask.pipeline);
        }

        // y-direction-v1: a stage that reads S2R SR_YDIRECTION keys one bool
        // constant (index 9) carrying the guest YControl sign. Both signs are
        // distinct valid pipelines; any other specialization shape stops.
        {
            const auto base = make_stages(metallib, "vs_plain", "fs_y_direction");
            auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
                *base.stages.fragment);
            fragment->consumes_y_direction = true;
            const auto consuming = metal::AdmittedGraphicsStages{
                {base.stages.vertex, fragment}, base.vertex_key, base.fragment_key};

            // Without the constant the key contradicts the consuming stage.
            if (!error_of(cache.acquire(keyed(cache, "vs_plain", "fs_y_direction"), consuming))) {
                std::fprintf(stderr, "FAIL: y-direction key without its constant acquired\n");
                ++failures;
            }
            const auto with_sign = [&](bool negate) {
                auto key = keyed(cache, "vs_plain", "fs_y_direction");
                key.fragment.specialization.push_back({9, "bool", {std::byte(negate ? 1 : 0)}});
                return key;
            };
            metal::ResidentPipeline positive, negative, repeat;
            if (!acquire_ok(cache, with_sign(false), consuming, positive) ||
                !acquire_ok(cache, with_sign(true), consuming, negative)) {
                std::fprintf(stderr, "FAIL: y-direction sign variants did not acquire\n");
                ++failures;
            } else {
                CHECK(positive.pipeline != negative.pipeline);
                CHECK(acquire_ok(cache, with_sign(false), consuming, repeat));
                CHECK(positive.pipeline == repeat.pipeline);
            }
            auto malformed = with_sign(false);
            malformed.fragment.specialization.back().type = "u32";
            if (!error_of(cache.acquire(malformed, consuming))) {
                std::fprintf(stderr, "FAIL: malformed y-direction constant acquired\n");
                ++failures;
            }
            auto truncated = with_sign(false);
            truncated.fragment.specialization.back().value.clear();
            if (!error_of(cache.acquire(truncated, consuming))) {
                std::fprintf(stderr, "FAIL: empty y-direction constant acquired\n");
                ++failures;
            }
            // A stage that does not read SR_YDIRECTION must not gain variants.
            auto plain = make_stages(metallib, "vs_plain", "fs_c0");
            auto plain_key = keyed(cache, "vs_plain", "fs_c0");
            plain_key.fragment.specialization.push_back({9, "bool", {std::byte{0}}});
            if (!error_of(cache.acquire(plain_key, plain))) {
                std::fprintf(stderr, "FAIL: non-consuming stage acquired a y-direction variant\n");
                ++failures;
            }
        }
        struct SamplerRejection {
            const char* name;
            std::function<void(SamplerDescriptor&)> apply;
        };
        const std::vector<SamplerRejection> sampler_rejections = {
            {"non-finite lod bias", [&](SamplerDescriptor& s) { s.lod_bias = std::nanf(""); }},
            {"non-finite border color",
             [&](SamplerDescriptor& s) { s.border_color[2] = INFINITY; }},
            {"inverted lod range",
             [&](SamplerDescriptor& s) {
                 s.min_lod = 4.0f;
                 s.max_lod = 2.0f;
             }},
            {"anisotropy out of range",
             [&](SamplerDescriptor& s) { s.max_anisotropy = 32.0f; }},
            {"anisotropy not a power of two",
             [&](SamplerDescriptor& s) { s.max_anisotropy = 3.0f; }},
            {"compare function without comparison",
             [&](SamplerDescriptor& s) { s.compare_function = SamplerCompareFunction::Less; }},
            {"unestablished normalization",
             [&](SamplerDescriptor& s) { s.float_coordinate_normalization = true; }},
        };
        for (const auto& rejection : sampler_rejections) {
            SamplerDescriptor sampler_key;
            rejection.apply(sampler_key);
            const auto created = cache.sampler(sampler_key);
            if (!error_of(created)) {
                std::fprintf(stderr, "FAIL: sampler rejection %s created a sampler\n",
                             rejection.name);
                ++failures;
            }
        }
    }

    // -- pixel oracles ------------------------------------------------------------
    const std::array<float, 4> background{0.25f, 0.5f, 0.75f, 1.0f};
    {
        const Canvas canvas = make_canvas(device, device_context->queue(), false, false);
        const std::array<float, 4> source{0.8f, 0.6f, 0.4f, 0.25f};
        const std::array<float, 4> blend_constant{0.5f, 0.25f, 0.125f, 0.75f};
        struct BlendCase {
            const char* name;
            BlendOperation op;
            BlendFactor src;
            BlendFactor dst;
            bool constant;
        };
        const std::vector<BlendCase> cases = {
            {"add src-alpha", BlendOperation::Add, BlendFactor::SourceAlpha,
             BlendFactor::OneMinusSourceAlpha, false},
            {"add one-one", BlendOperation::Add, BlendFactor::One, BlendFactor::One, false},
            {"subtract", BlendOperation::Subtract, BlendFactor::SourceColor,
             BlendFactor::OneMinusSourceAlpha, false},
            {"reverse subtract", BlendOperation::ReverseSubtract, BlendFactor::SourceColor,
             BlendFactor::DestinationAlpha, false},
            {"min", BlendOperation::Min, BlendFactor::One, BlendFactor::One, false},
            {"max", BlendOperation::Max, BlendFactor::One, BlendFactor::One, false},
            {"constant color", BlendOperation::Add, BlendFactor::ConstantColor,
             BlendFactor::OneMinusConstantColor, true},
        };
        for (const auto& blend_case : cases) {
            auto resident = acquire_pipeline(cache, metallib, "vs_plain", "fs_alpha",
                                             [&](metal::PipelineKey& key) {
                                                 key.colors[0].blend.enabled = true;
                                                 key.colors[0].blend.rgb = {blend_case.op,
                                                                            blend_case.src,
                                                                            blend_case.dst};
                                                 key.colors[0].blend.alpha = {
                                                     BlendOperation::Add, BlendFactor::One,
                                                     BlendFactor::Zero};
                                             });
            ClearValues clear;
            clear.color = background;
            DrawStep step;
            step.pipeline = pso_of(resident);
            if (blend_case.constant) step.blend_color = blend_constant;
            const auto pixels = run_pass(canvas, clear, {step});
            if (pixels.empty()) continue;
            const auto pick = [&](BlendFactor factor, std::size_t channel) {
                switch (factor) {
                    case BlendFactor::Zero: return 0.0f;
                    case BlendFactor::One: return 1.0f;
                    case BlendFactor::SourceColor: return source[channel];
                    case BlendFactor::OneMinusSourceColor: return 1.0f - source[channel];
                    case BlendFactor::SourceAlpha: return source[3];
                    case BlendFactor::OneMinusSourceAlpha: return 1.0f - source[3];
                    case BlendFactor::DestinationColor: return background[channel];
                    case BlendFactor::OneMinusDestinationColor: return 1.0f - background[channel];
                    case BlendFactor::DestinationAlpha: return background[3];
                    case BlendFactor::OneMinusDestinationAlpha: return 1.0f - background[3];
                    case BlendFactor::ConstantColor: return blend_constant[channel];
                    case BlendFactor::OneMinusConstantColor: return 1.0f - blend_constant[channel];
                    case BlendFactor::ConstantAlpha: return blend_constant[3];
                    case BlendFactor::OneMinusConstantAlpha: return 1.0f - blend_constant[3];
                    default: return 0.0f;  // dual-source factors never reach a pass
                }
            };
            std::array<float, 4> expected{};
            for (std::size_t channel = 0; channel < 4; ++channel) {
                // The alpha equation is fixed at Add One,Zero: every channel's
                // factors are the case's, except alpha uses One and Zero.
                const bool alpha_channel = channel == 3;
                const BlendFactor src = alpha_channel ? BlendFactor::One : blend_case.src;
                const BlendFactor dst = alpha_channel ? BlendFactor::Zero : blend_case.dst;
                const float src_value = source[channel] * pick(src, channel);
                const float dst_value = background[channel] * pick(dst, channel);
                switch (alpha_channel ? BlendOperation::Add : blend_case.op) {
                    case BlendOperation::Add: expected[channel] = src_value + dst_value; break;
                    case BlendOperation::Subtract:
                        expected[channel] = src_value - dst_value;
                        break;
                    case BlendOperation::ReverseSubtract:
                        expected[channel] = dst_value - src_value;
                        break;
                    case BlendOperation::Min:
                        expected[channel] = std::min(src_value, dst_value);
                        break;
                    case BlendOperation::Max:
                        expected[channel] = std::max(src_value, dst_value);
                        break;
                }
            }
            const auto pixel = pixel_at(pixels, 2, 3);
            if (!same_color(pixel, expected)) {
                std::fprintf(stderr,
                             "FAIL: blend case %s produced {%g, %g, %g, %g}, expected "
                             "{%g, %g, %g, %g}\n",
                             blend_case.name, pixel[0], pixel[1], pixel[2], pixel[3],
                             expected[0], expected[1], expected[2], expected[3]);
                ++failures;
            }
        }

        // Write masks: only masked channels change. Armigo's write mask puts R
        // at bit 0; Metal's puts alpha there, so this guards the conversion.
        {
            auto resident = acquire_pipeline(cache, metallib, "vs_plain", "fs_c3",
                                             [&](metal::PipelineKey& key) {
                                                 key.colors[0].blend.write_mask = 0x1;
                                             });
            ClearValues clear;
            clear.color = background;
            DrawStep step;
            step.pipeline = pso_of(resident);
            const auto pixels = run_pass(canvas, clear, {step});
            if (!pixels.empty()) {
                const auto pixel = pixel_at(pixels, 4, 4);
                if (!close_to(pixel[0], 0.4f) || !close_to(pixel[1], background[1]) ||
                    !close_to(pixel[2], background[2]) || !close_to(pixel[3], background[3])) {
                    std::fprintf(stderr,
                                 "FAIL: red-only write mask produced {%g, %g, %g, %g}\n",
                                 pixel[0], pixel[1], pixel[2], pixel[3]);
                    ++failures;
                }
            }
        }
    }

    // -- pixel oracles: depth compare and write ---------------------------------
    {
        const Canvas canvas = make_canvas(device, device_context->queue(), true, false);
        const auto depth_state = [&](CompareFunction compare, bool write) {
            DepthStencilState state;
            state.depth_test = true;
            state.depth_write = write;
            state.depth_compare = compare;
            auto created = cache.depth_stencil(state);
            CHECK(!error_of(created));
            return depth_stencil_of(std::get<graphics::RetainedResource>(created));
        };
        struct DepthCase {
            const char* name;
            CompareFunction compare;
            bool expect_pass;  // quad z=0.75 against cleared depth 0.5
        };
        const std::vector<DepthCase> cases = {
            {"never", CompareFunction::Never, false},
            {"less", CompareFunction::Less, false},
            {"equal", CompareFunction::Equal, false},
            {"less equal", CompareFunction::LessEqual, false},
            {"greater", CompareFunction::Greater, true},
            {"not equal", CompareFunction::NotEqual, true},
            {"greater equal", CompareFunction::GreaterEqual, true},
            {"always", CompareFunction::Always, true},
        };
        for (const auto& depth_case : cases) {
            auto resident = acquire_pipeline(cache, metallib, "vs_z075", "fs_c1");
            ClearValues clear;
            clear.color = background;
            clear.depth = 0.5;
            DrawStep step;
            step.pipeline = pso_of(resident);
            step.depth_stencil = depth_state(depth_case.compare, false);
            const auto pixels = run_pass(canvas, clear, {step});
            if (pixels.empty()) continue;
            const auto painted = pixel_at(pixels, 3, 3);
            const bool passed = !same_color(painted, background);
            if (passed != depth_case.expect_pass) {
                std::fprintf(stderr, "FAIL: depth compare %s %s\n", depth_case.name,
                             passed ? "passed unexpectedly" : "did not pass");
                ++failures;
            }
        }
        // Depth write: the second draw overwrites the first where it passes,
        // and the third draw is filtered by the second draw's written depth.
        {
            auto low = acquire_pipeline(cache, metallib, "vs_z025", "fs_c0");
            auto high = acquire_pipeline(cache, metallib, "vs_z075", "fs_c1");
            auto middle = acquire_pipeline(cache, metallib, "vs_plain", "fs_c2");
            ClearValues clear;
            clear.color = background;
            clear.depth = 1.0;
            const std::vector<DrawStep> steps = {
                {pso_of(low), depth_state(CompareFunction::Always, true), std::nullopt,
                 std::nullopt},
                {pso_of(high), depth_state(CompareFunction::Greater, true), std::nullopt,
                 std::nullopt},
                {pso_of(middle), depth_state(CompareFunction::Greater, false), std::nullopt,
                 std::nullopt},
            };
            const auto pixels = run_pass(canvas, clear, steps);
            if (!pixels.empty()) {
                // Depth sequence: 0.25 written, then 0.75 (greater) overwrites,
                // then 0.5 is not greater than 0.75 and paints nowhere: the
                // high quad's color must win everywhere.
                const auto pixel = pixel_at(pixels, 5, 5);
                if (!same_color(pixel, {0.2f, 0.25f, 0.8f, 1.0f})) {
                    std::fprintf(stderr, "FAIL: depth write sequence ended {%g, %g, %g, %g}\n",
                                 pixel[0], pixel[1], pixel[2], pixel[3]);
                    ++failures;
                }
            }
        }
    }

    // -- pixel oracles: front/back stencil ---------------------------------------
    {
        const Canvas canvas = make_canvas(device, device_context->queue(), true, true);
        const auto stencil_state = [&](CompareFunction compare, StencilOperation fail,
                                       StencilOperation depth_fail, StencilOperation pass,
                                       std::uint32_t compare_mask,
                                       std::uint32_t write_mask) {
            StencilFaceState face;
            face.compare = compare;
            face.fail = fail;
            face.depth_fail = depth_fail;
            face.pass = pass;
            face.compare_mask = compare_mask;
            face.write_mask = write_mask;
            DepthStencilState state;
            state.stencil_test = true;
            state.front = face;
            state.back = face;
            auto created = cache.depth_stencil(state);
            CHECK(!error_of(created));
            return depth_stencil_of(std::get<graphics::RetainedResource>(created));
        };
        // Probe: front and back faces replace with different reference values,
        // so each triangle half carries a distinct stencil value afterwards.
        // Which physical half is "front" is Metal's winding decision; the
        // assertions below hold for either assignment.
        const auto two_sided = [&](StencilFaceState front, StencilFaceState back) {
            DepthStencilState state;
            state.stencil_test = true;
            state.front = front;
            state.back = back;
            auto created = cache.depth_stencil(state);
            CHECK(!error_of(created));
            return depth_stencil_of(std::get<graphics::RetainedResource>(created));
        };
        const auto face = [](CompareFunction compare, StencilOperation pass) {
            StencilFaceState result;
            result.compare = compare;
            result.pass = pass;
            return result;
        };
        // The split quad draws two opposite-wound triangle halves, so exactly
        // one half is front-facing. The probe replaces with different front and
        // back references; the paint draws address each reference through its
        // own face, which holds whichever winding Metal assigned.
        auto probe = acquire_pipeline(cache, metallib, "vs_split", "fs_c0");
        auto paint_b = acquire_pipeline(cache, metallib, "vs_split", "fs_c1");
        auto paint_c = acquire_pipeline(cache, metallib, "vs_split", "fs_c2");
        const auto probe_stencil = two_sided(face(CompareFunction::Always,
                                                  StencilOperation::Replace),
                                             face(CompareFunction::Always,
                                                  StencilOperation::Replace));
        const auto equal_one = stencil_state(CompareFunction::Equal, StencilOperation::Keep,
                                             StencilOperation::Keep, StencilOperation::Keep,
                                             ~0u, ~0u);
        const auto front_equal_one = two_sided(face(CompareFunction::Equal,
                                                    StencilOperation::Keep),
                                               face(CompareFunction::Never,
                                                    StencilOperation::Keep));
        const auto back_equal_two = two_sided(face(CompareFunction::Never,
                                                   StencilOperation::Keep),
                                              face(CompareFunction::Equal,
                                                   StencilOperation::Keep));
        ClearValues clear;
        clear.color = background;
        clear.depth = 1.0;
        const std::vector<DrawStep> steps = {
            {pso_of(probe), probe_stencil, std::nullopt, std::make_pair(1u, 2u)},
            {pso_of(paint_b), front_equal_one, std::nullopt, std::make_pair(1u, 2u)},
            {pso_of(paint_c), back_equal_two, std::nullopt, std::make_pair(1u, 2u)},
        };
        const auto pixels = run_pass(canvas, clear, steps);
        if (!pixels.empty()) {
            const auto left = pixel_at(pixels, 1, 4);
            const auto right = pixel_at(pixels, 6, 6);
            const std::array<float, 4> color_b{0.2f, 0.25f, 0.8f, 1.0f};
            const std::array<float, 4> color_c{0.3f, 0.3f, 0.7f, 1.0f};
            const bool left_is_b = same_color(left, color_b);
            const bool left_is_c = same_color(left, color_c);
            CHECK(left_is_b != left_is_c);
            const auto& expected_right = left_is_b ? color_c : color_b;
            if (!same_color(right, expected_right)) {
                std::fprintf(stderr,
                             "FAIL: stencil halves are {%g,%g,%g,%g} and {%g,%g,%g,%g}\n",
                             left[0], left[1], left[2], left[3], right[0], right[1], right[2],
                             right[3]);
                ++failures;
            }
        }

        // A compare mask of zero passes without inspecting the reference, and
        // a write mask of zero leaves the stored value untouched, so a later
        // Equal draw still sees the probed values.
        {
            const auto mask_write =
                stencil_state(CompareFunction::Equal, StencilOperation::Keep,
                              StencilOperation::Keep, StencilOperation::Replace, 0u, ~0u);
            const auto equal_all = stencil_state(CompareFunction::Equal,
                                                 StencilOperation::Keep,
                                                 StencilOperation::Keep,
                                                 StencilOperation::Keep, ~0u, ~0u);
            auto probe_two = acquire_pipeline(cache, metallib, "vs_split", "fs_c3");
            auto paint_old = acquire_pipeline(cache, metallib, "vs_split", "fs_c1");
            auto paint_new = acquire_pipeline(cache, metallib, "vs_split", "fs_c4");
            const std::vector<DrawStep> masked = {
                {pso_of(probe_two), probe_stencil, std::nullopt, std::make_pair(3u, 4u)},
                {pso_of(probe_two), mask_write, std::nullopt, std::make_pair(9u, 9u)},
                {pso_of(paint_old), equal_one, std::nullopt, std::make_pair(3u, 4u)},
                {pso_of(paint_new), equal_all, std::nullopt, std::make_pair(9u, 9u)},
            };
            const auto masked_pixels = run_pass(canvas, clear, masked);
            if (!masked_pixels.empty()) {
                const auto left = pixel_at(masked_pixels, 1, 4);
                const auto right = pixel_at(masked_pixels, 6, 6);
                // The compare-mask-zero replace rewrote both halves to nine, so
                // the Equal-3/4 paint must find nothing and the Equal-9 paint
                // must cover everything.
                const std::array<float, 4> color_four{0.5f, 0.4f, 0.5f, 1.0f};
                if (!same_color(left, color_four) || !same_color(right, color_four)) {
                    std::fprintf(stderr,
                                 "FAIL: masked stencil halves are {%g,%g,%g,%g} and "
                                 "{%g,%g,%g,%g}\n",
                                 left[0], left[1], left[2], left[3], right[0], right[1],
                                 right[2], right[3]);
                    ++failures;
                }
            }
        }

        // Depth-fail operations: with the depth compare failing everywhere,
        // the depth-fail op replaces the value while the pass op keeps.
        {
            StencilFaceState fail_face;
            fail_face.compare = CompareFunction::Always;
            fail_face.fail = StencilOperation::Keep;
            fail_face.depth_fail = StencilOperation::Replace;
            fail_face.pass = StencilOperation::Keep;
            StencilFaceState keep_face;
            keep_face.compare = CompareFunction::Equal;
            DepthStencilState fail_key;
            fail_key.depth_test = true;
            fail_key.depth_write = false;
            fail_key.depth_compare = CompareFunction::Greater;  // 0.5 vs cleared 1.0 fails
            fail_key.stencil_test = true;
            fail_key.front = fail_face;
            fail_key.back = fail_face;
            auto fail_created = cache.depth_stencil(fail_key);
            CHECK(!error_of(fail_created));
            auto probe = acquire_pipeline(cache, metallib, "vs_plain", "fs_dummy");
            auto paint = acquire_pipeline(cache, metallib, "vs_plain", "fs_c5");
            ClearValues fail_clear;
            fail_clear.color = background;
            fail_clear.depth = 1.0;
            const std::vector<DrawStep> fail_steps = {
                {pso_of(probe),
                 depth_stencil_of(std::get<graphics::RetainedResource>(fail_created)),
                 std::nullopt, std::make_pair(7u, 9u)},
                {pso_of(paint), equal_one, std::nullopt, std::make_pair(7u, 9u)},
            };
            const auto fail_pixels = run_pass(canvas, fail_clear, fail_steps);
            if (!fail_pixels.empty()) {
                // Every fragment failed depth, so the depth-fail replace wrote
                // 7 (front) and 9 (back) and the Equal paint must cover both
                // halves. A pass-op implementation would write nothing.
                const std::array<float, 4> color_five{0.6f, 0.45f, 0.4f, 1.0f};
                const auto left = pixel_at(fail_pixels, 2, 3);
                const auto right = pixel_at(fail_pixels, 5, 5);
                if (!same_color(left, color_five) || !same_color(right, color_five)) {
                    std::fprintf(stderr,
                                 "FAIL: depth-fail stencil halves are {%g,%g,%g,%g} and "
                                 "{%g,%g,%g,%g}\n",
                                 left[0], left[1], left[2], left[3], right[0], right[1],
                                 right[2], right[3]);
                    ++failures;
                }
            }
        }
    }

    // -- typed integer output on an integer attachment ---------------------------
    {
        auto stages = make_stages(
            metallib, "vs_plain", "fs_uint_out",
            {armigo::platform::CompiledColorOutput{
                0, 0xF, armigo::platform::ColorOutputScalar::u32, 4}});
        auto key = keyed(cache, "vs_plain", "fs_uint_out");
        key.colors[0].format = TextureFormat::R8G8B8A8Uint;
        metal::ResidentPipeline resident;
        CHECK(acquire_ok(cache, key, stages, resident));
    }

    // Maxwell drops a fragment export whose logical target is unbound. Keep
    // the sparse output location in reflection while the pipeline masks it;
    // the bound target must still render and the same key must remain cached.
    {
        using Scalar = armigo::platform::ColorOutputScalar;
        auto stages = make_stages(metallib, "vs_plain", "fs_sparse_out",
            {{0, 15, Scalar::f32, 4}, {2, 15, Scalar::f32, 4}});
        auto key = keyed(cache, "vs_plain", "fs_sparse_out");
        metal::ResidentPipeline resident, repeated;
        CHECK(acquire_ok(cache, key, stages, resident));
        CHECK(acquire_ok(cache, key, stages, repeated));
        CHECK(repeated.pipeline == resident.pipeline);
        ClearValues initial;
        initial.color = {0.75f, 0.5f, 0.25f, 1.0f};
        const auto canvas = make_canvas(device, device_context->queue(), false, false);
        const auto pixels = run_pass(canvas, initial,
            {{pso_of(resident), nil, std::nullopt, std::nullopt}});
        CHECK(!pixels.empty());
        if (!pixels.empty())
            CHECK(same_color(pixel_at(pixels, 3, 3), {0.2f, 0.4f, 0.6f, 1.0f}));
    }

    // The same library and function specialize by attachment class. Both the
    // admitted reflection and exact function constants must match the key.
    {
        using Scalar = armigo::platform::ColorOutputScalar;
        std::array<metal::ResidentPipeline, 3> residents;
        const std::array formats{TextureFormat::R32G32B32A32Float,
            TextureFormat::R32G32B32A32Uint, TextureFormat::R32G32B32A32Sint};
        for (unsigned index = 0; index < formats.size(); ++index) {
            auto stages = make_stages(metallib, "vs_plain", "fs_adaptive",
                {{0, 15, static_cast<Scalar>(index), 4}});
            auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
                *stages.stages.fragment);
            fragment->fragment_output_types_from_render_targets = true;
            stages.stages.fragment = fragment;
            stages.fragment_key.specialization = metal::fragment_output_specialization(*fragment);
            auto key = keyed(cache, "vs_plain", "fs_adaptive");
            key.fragment = stages.fragment_key;
            key.colors[0].format = formats[index];
            CHECK(acquire_ok(cache, key, stages, residents[index]));
            metal::ResidentPipeline repeated;
            CHECK(acquire_ok(cache, key, stages, repeated));
            CHECK(repeated.pipeline == residents[index].pipeline);
            for (unsigned earlier = 0; earlier < index; ++earlier)
                CHECK(residents[earlier].pipeline != residents[index].pipeline);
            if (index == 0) {
                auto sparse_fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(*fragment);
                sparse_fragment->color_outputs[0].component_mask = 1;
                auto sparse_stages = stages;
                sparse_stages.stages.fragment = sparse_fragment;
                sparse_stages.fragment_key.variant = "sparse-red";
                auto sparse_key = key;
                sparse_key.fragment = sparse_stages.fragment_key;
                metal::ResidentPipeline sparse_pipeline;
                CHECK(acquire_ok(cache, sparse_key, sparse_stages, sparse_pipeline));
                ClearValues initial;
                initial.color = {0.25f, 0.5f, 0.75f, 1.0f};
                auto canvas = make_canvas(device, device_context->queue(), false, false);
                const auto pixels = run_pass(canvas, initial,
                    {{pso_of(sparse_pipeline), nil, std::nullopt, std::nullopt}});
                CHECK(!pixels.empty());
                if (!pixels.empty()) {
                    const auto pixel = pixel_at(pixels, 3, 3);
                    CHECK(pixel[1] == 0.5f && pixel[2] == 0.75f && pixel[3] == 1.0f);
                }
            }
            for (unsigned mutation = 0; mutation < 6; ++mutation) {
                auto bad_key = key;
                auto bad_stages = stages;
                auto& constants = bad_key.fragment.specialization;
                switch (mutation) {
                case 0: constants.clear(); break;
                case 1: constants[0].value[0] = std::byte{3}; break;
                case 2: constants[0].type = "f32"; break;
                case 3: constants[0].value.pop_back(); break;
                case 4: constants.push_back(constants[0]); break;
                case 5: constants[0].index = 1; break;
                }
                bad_stages.fragment_key = bad_key.fragment;
                CHECK(error_of(cache.acquire(bad_key, bad_stages)) != nullptr);
            }
            auto mismatched = key;
            mismatched.colors[0].format = formats[(index + 1) % formats.size()];
            CHECK(error_of(cache.acquire(mismatched, stages)) != nullptr);
        }
    }

    // A valid adaptive fragment can export no colors. It consumes no class
    // constants, and an attached integer target must not imply a float export.
    {
        auto stages = make_stages(metallib, "vs_plain", "fs_no_color");
        auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
            *stages.stages.fragment);
        fragment->fragment_output_types_from_render_targets = true;
        stages.stages.fragment = fragment;
        auto key = keyed(cache, "vs_plain", "fs_no_color");
        key.colors[0].format = TextureFormat::R32G32B32A32Uint;
        metal::ResidentPipeline resident;
        CHECK(acquire_ok(cache, key, stages, resident));
    }

    // Function constant 8 changes the fragment ABI while the attachment
    // formats stay fixed, so fallback activation must identify a distinct PSO.
    {
        using Scalar = armigo::platform::ColorOutputScalar;
        auto stages = make_stages(metallib, "vs_plain", "fs_depth_bias",
            {{0, 15, Scalar::f32, 4}});
        auto capable = std::make_shared<armigo::platform::CompiledGraphicsStage>(
            *stages.stages.fragment);
        capable->emulates_depth_bias = true;
        stages.stages.fragment = capable;

        auto inactive_key = keyed(cache, "vs_plain", "fs_depth_bias");
        inactive_key.depth_format = TextureFormat::D32Float;
        metal::ResidentPipeline inactive, inactive_repeated, active, active_repeated;
        CHECK(acquire_ok(cache, inactive_key, stages, inactive));
        CHECK(acquire_ok(cache, inactive_key, stages, inactive_repeated));
        CHECK(inactive_repeated.pipeline == inactive.pipeline);

        auto active_key = inactive_key;
        active_key.shader_depth_bias = true;
        CHECK(active_key != inactive_key);
        CHECK(acquire_ok(cache, active_key, stages, active));
        CHECK(acquire_ok(cache, active_key, stages, active_repeated));
        CHECK(active_repeated.pipeline == active.pipeline);
        CHECK(active.pipeline != inactive.pipeline);

        auto missing_depth = active_key;
        missing_depth.depth_format.reset();
        CHECK(error_of(cache.acquire(missing_depth, stages)) != nullptr);

        auto missing_capability = stages;
        auto incapable = std::make_shared<armigo::platform::CompiledGraphicsStage>(*capable);
        incapable->emulates_depth_bias = false;
        missing_capability.stages.fragment = incapable;
        CHECK(error_of(cache.acquire(active_key, missing_capability)) != nullptr);
    }

    // -- configured budget rejects before native creation -------------------------
    {
        auto trimmed = metal::create_pipeline_state_cache(device_context, 1);
        CHECK(!error_of(trimmed));
        if (!error_of(trimmed)) {
            auto& small_cache =
                **std::get_if<std::shared_ptr<metal::PipelineStateCache>>(&trimmed);
            const auto stages = make_stages(metallib, "vs_plain", "fs_c0");
            const auto key = keyed(small_cache, "vs_plain", "fs_c0");
            auto acquired = small_cache.acquire(key, stages);
            CHECK(error_of(acquired) != nullptr);
            CHECK(!error_of(small_cache.trim(0)));
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "%u pipeline-state checks failed\n", failures);
        return 1;
    }
    std::printf("graphics.metal4_pipeline_state passed\n");
    return 0;
}
