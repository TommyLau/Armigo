// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>
#include "metal_renderer.h"
#include "device_context.h"
#include "buffer_cache.h"
#include "texture_cache.h"
#include "texture_format.h"
#include "texture_residency.h"
#include "native_resources.h"
#include "../../src/shader/pipeline/msl/custom_border.h"
#include <armigo/graphics/resource_memory_registry.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {
using namespace armigo;
using namespace graphics;
namespace m = platform::metal;
namespace gm = shader::graphics_msl;
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); ++failures; } } while (false)

template<class T> T take(RendererResult<T> result) {
    if (const auto* error = std::get_if<RendererError>(&result)) {
        const auto command = error->command();
        std::fprintf(stderr, "Cannot continue: %.*s\n", int(command.detail.size()), command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}
bool ok(const RendererStatus& result) {
    if (const auto* error = std::get_if<RendererError>(&result)) {
        const auto command = error->command();
        std::fprintf(stderr, "Status: %.*s\n", int(command.detail.size()), command.detail.data());
        return false;
    }
    return true;
}
std::uint64_t deadline() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        (std::chrono::steady_clock::now() + std::chrono::seconds(20)).time_since_epoch()).count();
}

struct Memory final : ResourceMemoryBackend {
    std::vector<std::byte> bytes = std::vector<std::byte>(256 * 1024, std::byte{0});
    unsigned reads = 0, writes = 0;
    RendererStatus read_cpu(BackingRange range, std::span<std::byte> out) override {
        ++reads;
        if (range.offset > bytes.size() || range.size > bytes.size() - range.offset || out.size() != range.size)
            return RendererError({CommandStatus::MemoryFault, "synthetic read bounds"});
        std::copy_n(bytes.begin() + range.offset, out.size(), out.begin());
        return renderer_success();
    }
    RendererStatus write_cpu(BackingRange range, std::span<const std::byte> in) override {
        ++writes;
        if (range.offset > bytes.size() || range.size > bytes.size() - range.offset || in.size() != range.size)
            return RendererError({CommandStatus::MemoryFault, "synthetic write bounds"});
        std::copy(in.begin(), in.end(), bytes.begin() + range.offset);
        return renderer_success();
    }
    RendererStatus wait_gpu(CompletionToken token) override {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < end) {
            auto state = token.snapshot();
            if (state.error) return *state.error;
            if (state.phase == SubmissionPhase::GpuComplete || state.phase == SubmissionPhase::Publishing ||
                state.phase == SubmissionPhase::Complete) return renderer_success();
            if (state.phase == SubmissionPhase::Accepted)
                return RendererError({CommandStatus::Stopped, "synthetic producer is not submitted"});
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return RendererError({CommandStatus::Stopped, "synthetic feedback timeout"});
    }
};

struct Effects final : CompletionActions {
    unsigned coherent = 0, prepared = 0, published = 0;
    std::vector<unsigned>* publication_order = nullptr;
    unsigned publication_id = 0;
    RendererStatus make_coherent() override { ++coherent; return renderer_success(); }
    RendererStatus prepare_publication() override { ++prepared; return renderer_success(); }
    void publish() noexcept override {
        ++published;
        if (publication_order) publication_order->push_back(publication_id);
    }
    RendererStatus preserve_failed_authority() override { return renderer_success(); }
};

struct PipelineLease final : m::MetalRenderPipelineResource {
    id<MTLRenderPipelineState> value;
    id<MTLLibrary> library;
    id<MTLRenderPipelineState> pipeline() const noexcept override { return value; }
};
struct DepthLease final : m::MetalDepthStencilResource {
    id<MTLDepthStencilState> value;
    id<MTLDepthStencilState> depth_stencil() const noexcept override { return value; }
};
struct SamplerLease final : m::MetalSamplerResource {
    id<MTLSamplerState> value;
    id<MTLSamplerState> sampler() const noexcept override { return value; }
};

// L9 is a concurrent lane. This deliberately small test double builds real
// Metal pipelines from synthetic source while recording the frozen key.
struct Pipelines final : m::PipelineStateCache {
    std::shared_ptr<m::DeviceContext> context_;
    id<MTLLibrary> library;
    id<MTL4Compiler> compiler;
    std::vector<m::PipelineKey> keys;
    std::vector<DepthStencilState> depth_stencil_keys;
    std::vector<SamplerDescriptor> sampler_keys;
    explicit Pipelines(std::shared_ptr<m::DeviceContext> context) : context_(std::move(context)) {
        NSError* error = nil;
        NSMutableString* source = [NSMutableString stringWithString:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
            "struct Out { float4 position [[position]]; };\n"
            "vertex Out vertex_main(uint id [[vertex_id]]) {\n"
            "float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)}; return {float4(p[id%3],0,1)}; }\n"
            "struct In { float3 p [[attribute(0)]]; };\n"
            "vertex Out vertex_stream(In input [[stage_in]]) { return {float4(input.p,1)}; }\n"
            "fragment float4 fragment_main(constant float* cb [[buffer(0)]], constant uint* sizes [[buffer(17)]]) {\n"
            "return float4(cb[0],cb[2],float(sizes[0])/16.0,1); }\n"
            "fragment float4 fragment_bytes(constant uchar* cb [[buffer(0)]]) {\n"
            "return float4(cb[0],cb[1],cb[2],cb[3])/255.0; }\n"
            "fragment void fragment_empty() {}\n"
            "struct InstIn { float3 p [[attribute(0)]]; float3 c [[attribute(1)]]; };\n"
            "struct InstOut { float4 position [[position]]; float3 color; };\n"
            "vertex InstOut vertex_instances(InstIn in [[stage_in]], uint id [[instance_id]], uint base [[base_instance]]) {\n"
            "return {float4(in.p.x*0.25-0.75+float(id-base)*0.5,in.p.y,0,1),in.c}; }\n"
            "fragment float4 fragment_instances(InstOut in [[stage_in]]) { return float4(in.color,1); }\n"
            "struct StorageOut { float4 position [[position]]; float value; };\n"
            "vertex StorageOut vertex_storage_read(constant uint* cb [[buffer(0)]], constant uint* sizes [[buffer(17)]], device const uint* storage [[buffer(3)]], uint id [[vertex_id]]) {\n"
            "float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)}; ulong base = ulong(cb[4]) | (ulong(cb[5]) << 32); ulong address = base + 4; uint value = 0;\n"
            "if (sizes[2] >= 28 && cb[5] != 0 && cb[6] >= 8 && address >= base && address - base <= ulong(cb[6] - 4)) value = storage[1];\n"
            "return {float4(p[id%3],0,1), as_type<float>(value)}; }\n"
            "fragment float4 fragment_storage_color(StorageOut in [[stage_in]]) { return float4(in.value,0,0,1); }\n"
            "fragment float4 fragment_storage_overlap(StorageOut in [[stage_in]], constant uint* cb [[buffer(0)]], device const uint* storage [[buffer(3)]]) {\n"
            "uint value = cb[10] >= 8 ? storage[1] : 0; return float4(in.value + as_type<float>(value),0,0,1); }\n"
            "fragment float4 fragment_storage_update(float4 position [[position]], constant uint* cb [[buffer(0)]], constant uint* sizes [[buffer(17)]], device uint* storage [[buffer(3)]]) {\n"
            "ulong base = ulong(cb[4]) | (ulong(cb[5]) << 32); ulong address = base + 4;\n"
            "if (sizes[2] >= 28 && cb[6] >= 8 && address >= base && address - base <= ulong(cb[6] - 4) && uint(position.x) == 0 && uint(position.y) == 0) storage[1] = as_type<uint>(0.75f);\n"
            "return float4(0,0,0,1); }\n"
            "fragment float4 tex_1d(texture1d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,0.75); }\n"
            "fragment float4 tex_1da(texture1d_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,0.75,1); }\n"
            "fragment float4 tex_2d(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float2(0.75)); }\n"
            "fragment float4 tex_2d_step(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { float4 v=t.sample(s,float2(0.75)); return float4(v.r+0.25,v.g+0.125,v.b,1); }\n"
            "fragment float4 tex_2da(texture2d_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float2(0.75),1); }\n"
            "fragment float4 tex_2da_zero(texture2d_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float2(0.75),0); }\n"
            "fragment float4 tex_3d(texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float3(0.75)); }\n"
            "fragment float4 tex_3d_step(texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { float4 v=t.sample(s,float3(0.75)); return float4(v.r+0.125,0,0,1); }\n"
            "fragment float4 tex_cube(texturecube<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float3(0,0,-1)); }\n"
            "fragment float4 tex_cubea(texturecube_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s,float3(0,0,-1),1); }\n"
            "fragment float4 tex_rect(texture2d<float,access::read> t [[texture(0)]]) { return t.read(uint2(1,1)); }\n"
            "fragment float4 tex_uint(texture2d<uint,access::read> t [[texture(0)]]) { return float4(t.read(uint2(1,1)))/255.0; }\n"
            "fragment float4 tex_sint(texture2d<int,access::read> t [[texture(0)]]) { return float4(t.read(uint2(1,1)) + int4(128))/255.0; }\n"];
        const std::string custom_border_source =
            std::string(shader::pipeline::msl::custom_border_msl) + R"metal(
inline float4 synthetic_custom_border(
    texture2d<float> source, sampler state,
    const device ArmigoCustomBorder* support, float2 coordinates) {
    float4 black = source.sample(state, coordinates);
    if (support[0].enabled != 0u) {
        const float coverage = support[0].coverage.sample(state, coordinates).x;
        black = armigo_custom_border(black, coverage, support[0]);
    }
    return black;
}
fragment float4 custom_border_corner(
    texture2d<float> source [[texture(3)]],
    sampler state [[sampler(5)]],
    const device ArmigoCustomBorder* support [[buffer(2)]]) {
    return synthetic_custom_border(source, state, support,
                                   float2(-0.0625, -0.0625));
}
)metal";
        [source appendString:[NSString
            stringWithUTF8String:custom_border_source.c_str()]];
        [source appendString:
            @"struct ArgumentManyResources {\n"
             "array<texture2d<float>, 20> resource0;\n"
             "array<sampler, 20> resource1;\n"
             "};\n"
             "fragment float4 argument_many(const device ArgumentManyResources& resources [[buffer(2)]]) {\n"
             "float4 color = float4(0.0);\n"];
        for (unsigned index = 0; index != 20; ++index) {
            [source appendFormat:
                @"color += resources.resource0[%u].sample(resources.resource1[%u], float2(0.5)) * %u.0;\n",
                index, index, index + 1];
        }
        [source appendString:
            @"return color / 210.0;\n"
             "}\n"
             "struct ArgumentArrayResources { array<texture2d_array<float>, 20> resource0; array<sampler, 20> resource1; };\n"
             "fragment float4 argument_array_step(const device ArgumentArrayResources& resources [[buffer(2)]]) {\n"
             "float4 value = resources.resource0[0].sample(resources.resource1[0], float2(0.5), 0);\n"
             "return float4(value.r + 0.25, value.g + 0.125, value.b, 1.0);\n"
             "}\n"
             "fragment float4 direct_high(texture2d<float> texture [[texture(127)]], sampler state [[sampler(15)]]) { return texture.sample(state, float2(0.5)); }\n"
             "fragment float4 argument_each(float4 position [[position]], const device ArgumentManyResources& resources [[buffer(2)]]) {\n"
             "uint index = (uint(position.x) + uint(position.y) * 16) % 20;\n"
             "uint probe = uint(position.y) / 4;\n"
             "float2 uv = probe == 0 ? float2(0.53) : probe == 1 ? float2(-0.21,0.53) : probe == 2 ? float2(1.21,0.53) : float2(0.53,-0.21);\n"
             "return resources.resource0[index].sample(resources.resource1[index], uv);\n"
             "}\n"
             "fragment float4 argument_first(const device ArgumentManyResources& resources [[buffer(2)]]) {\n"
             "float4 value = resources.resource0[0].sample(resources.resource1[0], float2(0.5));\n"
             "return float4(value.rgb, 1.0);\n"
             "}\n"
             "fragment float4 argument_first_step(const device ArgumentManyResources& resources [[buffer(2)]]) {\n"
             "float4 value = resources.resource0[0].sample(resources.resource1[0], float2(0.5));\n"
             "return float4(value.r + 0.25, value.g + 0.125, value.b, 1.0);\n"
             "}\n"];
        library = [context_->device() newLibraryWithSource:source options:nil error:&error];
        if (!library) std::fprintf(stderr, "Shader: %s\n", error.localizedDescription.UTF8String);
        CHECK(library != nil);
        compiler = [context_->device() newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        CHECK(compiler != nil);
    }
    const m::PipelineContextKey& context() const noexcept override { return context_->identity(); }
    RendererResult<m::ResidentPipeline> acquire(const m::PipelineKey& key,
                                               const m::AdmittedGraphicsStages& stages) override {
        keys.push_back(key);
        auto* descriptor = [MTL4RenderPipelineDescriptor new];
        auto* vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = library;
        vertex.name = [NSString stringWithUTF8String:key.vertex.entry_point.c_str()];
        auto* fragment = [MTL4LibraryFunctionDescriptor new];
        fragment.library = library;
        fragment.name = [NSString stringWithUTF8String:key.fragment.entry_point.c_str()];
        descriptor.vertexFunctionDescriptor = vertex;
        descriptor.fragmentFunctionDescriptor = fragment;
        if (key.colors[0].format) {
            descriptor.colorAttachments[0].pixelFormat =
                key.colors[0].format == TextureFormat::R8Unorm
                    ? MTLPixelFormatR8Unorm
                    : MTLPixelFormatRGBA8Unorm;
            descriptor.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
            descriptor.colorAttachments[0].blendingState = MTL4BlendStateDisabled;
        }
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        auto* layout = [MTLVertexDescriptor vertexDescriptor];
        for (const auto& attribute : key.attributes) {
            CHECK(attribute.format == VertexAttribFormat::R32G32B32Float);
            layout.attributes[attribute.location].format = MTLVertexFormatFloat3;
            layout.attributes[attribute.location].offset = attribute.offset;
            layout.attributes[attribute.location].bufferIndex = attribute.buffer_index;
        }
        for (const auto& buffer : key.buffers) {
            layout.layouts[buffer.buffer_index].stride = buffer.stride;
            layout.layouts[buffer.buffer_index].stepFunction = buffer.step == VertexStep::PerVertex
                ? MTLVertexStepFunctionPerVertex : buffer.step == VertexStep::PerInstance
                ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionConstant;
            layout.layouts[buffer.buffer_index].stepRate = buffer.rate;
        }
        descriptor.vertexDescriptor = layout;
        NSError* error = nil;
        auto lease = std::make_shared<PipelineLease>();
        lease->library = library;
        lease->value = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!lease->value) {
            const std::string detail = error.localizedDescription.UTF8String;
            return RendererError({CommandStatus::Unsupported, detail});
        }
        return m::ResidentPipeline{key, lease, stages};
    }
    RendererResult<RetainedResource> depth_stencil(const DepthStencilState& key) override {
        depth_stencil_keys.push_back(key);
        auto* descriptor = [MTLDepthStencilDescriptor new];
        descriptor.depthCompareFunction = MTLCompareFunctionAlways;
        descriptor.depthWriteEnabled = key.depth_test && key.depth_write;
        if (key.stencil_test) {
            descriptor.frontFaceStencil = [MTLStencilDescriptor new];
            descriptor.backFaceStencil = [MTLStencilDescriptor new];
        }
        auto lease = std::make_shared<DepthLease>();
        lease->value = [context_->device() newDepthStencilStateWithDescriptor:descriptor];
        return RetainedResource(lease);
    }
    RendererResult<RetainedResource> sampler(const SamplerDescriptor& descriptor) override {
        sampler_keys.push_back(descriptor);
        const auto properties = platform::metal_sampler_properties(descriptor);
        if (const auto* error = std::get_if<TextureError>(&properties))
            return RendererError({CommandStatus::Unsupported, error->message()});
        auto lease = std::make_shared<SamplerLease>();
        lease->value = [context_->device() newSamplerStateWithDescriptor:
            platform::metal_sampler_descriptor(std::get<platform::MetalSamplerProperties>(properties))];
        return RetainedResource(lease);
    }
    RendererStatus trim(std::uint64_t) override { return renderer_success(); }
};

struct Stages final : m::GraphicsStageProvider {
    m::AdmittedGraphicsStages admitted;
    explicit Stages(bool stream) {
        auto vertex = std::make_shared<platform::CompiledGraphicsStage>();
        auto fragment = std::make_shared<platform::CompiledGraphicsStage>();
        vertex->metallib = {std::byte{1}};
        fragment->metallib = {std::byte{2}};
        vertex->stage = shader_input::Stage::Vertex;
        fragment->stage = shader_input::Stage::Fragment;
        vertex->entry_point = stream ? "vertex_stream" : "vertex_main";
        fragment->entry_point = "fragment_main";
        vertex->vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
        vertex->interface.position_output_mask = 15;
        fragment->interface.color_output_mask = 15;
        fragment->color_outputs.push_back({0, 15, platform::ColorOutputScalar::f32, 4});
        fragment->constant_buffer_windows.push_back({0, 16, 0});
        fragment->constant_buffer_support_index = 17;
        if (stream) {
            vertex->vertex_class_inputs.push_back({0, VertexFormatClass::float_class});
            vertex->interface.generic_inputs.push_back({0, 7});
        }
        admitted.stages = {vertex, fragment};
        admitted.vertex_key = {0, "synthetic", "v", "v-artifact", vertex->entry_point, {}};
        admitted.fragment_key = {0, "synthetic", "f", "f-artifact", fragment->entry_point, {}};
    }
    RendererResult<m::AdmittedGraphicsStages> lookup(const RenderStop&, const GpuAddressSpace&) override {
        return admitted;
    }
};

constexpr std::uint64_t gpu_base = 0x100000;
constexpr std::size_t target_offset = 65536;
constexpr std::size_t sample_output_offset = 196608;
constexpr std::size_t uniform_offset = 8192, vertex_offset = 12288;
constexpr std::size_t index_offset = 16384;
constexpr std::size_t empty_vertex_offset = 20480;
RenderStop snapshot(bool stream = false) {
    RenderStop stop;
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    stop.draw = DrawDescription{false, 4, 0, 3, 0, 0, 1};
    stop.color_targets[0] = {
        gpu_base + target_offset, 64, 16, 0xd5, 0x1000, 1, 0, 0};
    stop.registers[0x487] = 1;
    auto& r = stop.registers;
    r[0x0df] = 1;
    r[0x280] = std::bit_cast<std::uint32_t>(8.0f);
    r[0x281] = std::bit_cast<std::uint32_t>(-8.0f);
    r[0x282] = std::bit_cast<std::uint32_t>(0.5f);
    r[0x283] = std::bit_cast<std::uint32_t>(8.0f);
    r[0x284] = std::bit_cast<std::uint32_t>(8.0f);
    r[0x285] = std::bit_cast<std::uint32_t>(0.5f);
    r[0x286] = 0x6420;
    r[0x300] = r[0x301] = 16u << 16;
    r[0x303] = std::bit_cast<std::uint32_t>(1.0f);
    r[0x36b] = r[0x36c] = 0x1b02;
    r[0x380] = 1;
    r[0x381] = r[0x382] = 16u << 16;
    r[0x3e4] = 1;
    r[0x3fd] = r[0x3fe] = 16u << 16;
    r[0x4b9] = 1;
    r[0x4c3] = 8;
    r[0x647] = 0x901;
    r[0x648] = 0x405;
    r[0x64b] = 1;
    r[0x64f] = 0x181d;
    r[0x680] = 0x1111;
    stop.constant_buffers[4][0] = {gpu_base + uniform_offset, 4, true};
    if (stream) {
        r[0x458] = (2u << 21) | (7u << 27);
        r[0x700] = 0x1000 | 12;
        r[0x702] = gpu_base + vertex_offset;
        r[0x7c1] = gpu_base + vertex_offset + 71;
    }
    return stop;
}

RenderStop attachment_less_snapshot(std::uint32_t extent, bool stream = false) {
    auto stop = snapshot(stream);
    if (stream) {
        stop.registers[0x702] = gpu_base + empty_vertex_offset;
        stop.registers[0x7c1] = gpu_base + empty_vertex_offset + 35;
    }
    stop.draw->indexed = true;
    stop.registers[0x5f3] = gpu_base + index_offset;
    stop.registers[0x5f5] = gpu_base + index_offset + 5;
    stop.registers[0x5f6] = 1;
    stop.color_targets = {};
    stop.registers[0x487] = 0;
    stop.registers[0x680] = 0;
    stop.depth = {0, 0x13, 0, 0x1000, extent, extent, true};
    stop.registers[0x3f8] = 0;
    stop.registers[0x3f9] = 0;
    stop.registers[0x3fa] = 0x13;
    stop.registers[0x3fb] = 0;
    stop.registers[0x3fc] = 0x1000;
    stop.registers[0x48a] = extent;
    stop.registers[0x48b] = extent;
    stop.registers[0x48c] = 0x10001;
    stop.registers[0x54e] = 1;
    stop.registers[0x280] = std::bit_cast<std::uint32_t>(float(extent) * 0.5f);
    stop.registers[0x281] = std::bit_cast<std::uint32_t>(float(extent) * -0.5f);
    stop.registers[0x283] = std::bit_cast<std::uint32_t>(float(extent) * 0.5f);
    stop.registers[0x284] = std::bit_cast<std::uint32_t>(float(extent) * 0.5f);
    stop.registers[0x300] = extent << 16;
    stop.registers[0x301] = extent << 16;
    stop.registers[0x381] = extent << 16;
    stop.registers[0x382] = extent << 16;
    stop.registers[0x3fd] = extent << 16;
    stop.registers[0x3fe] = extent << 16;

    // Valid depth/stencil tests remain enabled in guest state but must be
    // disabled in native state because no depth or stencil storage is bound.
    stop.registers[0x4b3] = 1;
    stop.registers[0x4ba] = 1;
    stop.registers[0x4c3] = 8;
    stop.registers[0x4e0] = 1;
    stop.registers[0x4e1] = 1;
    stop.registers[0x4e2] = 1;
    stop.registers[0x4e3] = 1;
    stop.registers[0x4e4] = 8;
    return stop;
}

void configure_target(RenderStop& stop, std::size_t offset,
                      std::uint32_t width_or_pitch, std::uint32_t height,
                      std::uint32_t format, std::uint32_t tile_mode,
                      std::uint32_t surface_width,
                      std::uint32_t surface_height) {
    const std::uint64_t address = gpu_base + offset;
    stop.color_targets[0] = {address, width_or_pitch, height, format,
                             tile_mode, 1, 0x1000, 0};
    stop.registers[0x200] = static_cast<std::uint32_t>(address >> 32);
    stop.registers[0x201] = static_cast<std::uint32_t>(address);
    stop.registers[0x202] = width_or_pitch;
    stop.registers[0x203] = height;
    stop.registers[0x204] = format;
    stop.registers[0x205] = tile_mode;
    stop.registers[0x206] = 1;
    stop.registers[0x207] = 0x1000;
    stop.registers[0x208] = 0;
    stop.registers[0x280] =
        std::bit_cast<std::uint32_t>(float(surface_width) * 0.5f);
    stop.registers[0x281] =
        std::bit_cast<std::uint32_t>(float(surface_height) * -0.5f);
    stop.registers[0x283] =
        std::bit_cast<std::uint32_t>(float(surface_width) * 0.5f);
    stop.registers[0x284] =
        std::bit_cast<std::uint32_t>(float(surface_height) * 0.5f);
    stop.registers[0x300] = surface_width << 16;
    stop.registers[0x301] = surface_height << 16;
    stop.registers[0x381] = surface_width << 16;
    stop.registers[0x382] = surface_height << 16;
    stop.registers[0x3fd] = surface_width << 16;
    stop.registers[0x3fe] = surface_height << 16;
}

std::size_t gob_offset(std::size_t x_bytes, std::size_t y,
                       std::uint32_t block_height_log2,
                       std::uint32_t block_depth_log2,
                       std::size_t logical_row_bytes) {
    const std::size_t block_height = std::size_t{1} << block_height_log2;
    const std::size_t block_depth = std::size_t{1} << block_depth_log2;
    const std::size_t tiles_x = (logical_row_bytes + 63) / 64;
    const std::size_t tile =
        (y / (8 * block_height)) * tiles_x + x_bytes / 64;
    const std::size_t gob = (y / 8) % block_height;
    const std::size_t within = (x_bytes & 15) | ((y & 1) << 4) |
                               ((x_bytes & 16) << 1) | ((y & 6) << 5) |
                               ((x_bytes & 32) << 3);
    return tile * 512 * block_height * block_depth + gob * 512 + within;
}

std::size_t volume_gob_offset(std::size_t x_bytes, std::size_t y,
                              std::size_t z,
                              std::uint32_t block_height_log2,
                              std::uint32_t block_depth_log2,
                              std::size_t logical_row_bytes,
                              std::size_t logical_rows) {
    const std::size_t block_height = std::size_t{1} << block_height_log2;
    const std::size_t block_depth = std::size_t{1} << block_depth_log2;
    const std::size_t tiles_x = (logical_row_bytes + 63) / 64;
    const std::size_t tiles_y =
        (logical_rows + 8 * block_height - 1) / (8 * block_height);
    const std::size_t tile =
        ((z / block_depth * tiles_y + y / (8 * block_height)) * tiles_x +
         x_bytes / 64);
    const std::size_t gob =
        (z % block_depth) * block_height + (y / 8) % block_height;
    const std::size_t within = (x_bytes & 15) | ((y & 1) << 4) |
                               ((x_bytes & 16) << 1) | ((y & 6) << 5) |
                               ((x_bytes & 32) << 3);
    return tile * 512 * block_height * block_depth + gob * 512 + within;
}

std::vector<std::byte> expected_block_linear(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t bytes_per_texel, std::uint32_t block_height_log2,
    std::uint32_t block_depth_log2, std::span<const std::byte> texel,
    std::byte sentinel, std::size_t storage_size) {
    std::vector<std::byte> expected(storage_size, sentinel);
    const std::size_t row_bytes = std::size_t{width} * bytes_per_texel;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t offset = gob_offset(
                x * bytes_per_texel, y, block_height_log2,
                block_depth_log2, row_bytes);
            std::copy(texel.begin(), texel.end(), expected.begin() + offset);
        }
    }
    return expected;
}

struct Fixture {
    std::shared_ptr<Memory> backend = std::make_shared<Memory>();
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<m::DeviceContext> device;
    std::shared_ptr<Pipelines> pipelines;
    m::RendererServices services;
    std::shared_ptr<m::MetalRenderer> renderer;
    explicit Fixture(bool stream = false) {
        device = take(m::create_device_context());
        memory = take(make_resource_memory({1, backend, CoherenceMode::SubmissionBoundary, 4096}));
        CHECK(ok(memory->register_backing({{1, 1, 1}, backend->bytes.size(), backend})));
        CHECK(ok(memory->register_mapping({{1, 1, 1}, {1, 1, 1},
            {gpu_base, backend->bytes.size(), 0, 0x10000000, backend->bytes.size(), 4096}})));
        pipelines = std::make_shared<Pipelines>(device);
        services = {device, memory, std::make_shared<Stages>(stream),
            take(m::create_texture_cache(device, memory, 16 * 1024 * 1024)),
            take(m::create_buffer_cache(device, memory, 1024 * 1024)), pipelines,
            take(m::create_binding_planner(pipelines, device)),
            take(m::create_submission_queue(device, memory, 64, 4))};
        renderer = take(m::create_renderer({16 * 1024 * 1024, 1024 * 1024, 4, 64}, services));
        const std::array<float, 18> positions{-1,-1,0, 3,-1,0, -1,3,0, -1,-1,0, 3,-1,0, -1,3,0};
        std::memcpy(backend->bytes.data() + vertex_offset, positions.data(), sizeof(positions));
        uniform(1.0f);
    }
    void uniform(float value) {
        auto transaction = take(memory->prepare_cpu_write({{1, 1, 1}, uniform_offset, 4}));
        std::memcpy(backend->bytes.data() + uniform_offset, &value, 4);
        transaction->commit();
    }
    SubmissionReceipt enqueue(RenderStop stop, std::uint64_t sequence, std::shared_ptr<Effects> effects) {
        return take(renderer->enqueue({{1, sequence}, std::make_shared<const RenderStop>(std::move(stop)),
            memory, {}, std::move(effects)}));
    }
    void pixels(unsigned left, unsigned right, unsigned blue) {
        for (unsigned y = 0; y != 16; ++y) for (unsigned x = 0; x != 16; ++x) {
            const auto offset = target_offset + y * 64 + x * 4;
            CHECK(std::to_integer<unsigned>(backend->bytes[offset]) == (x < 8 ? left : right));
            CHECK(backend->bytes[offset + 1] == std::byte{0});
            CHECK(std::to_integer<unsigned>(backend->bytes[offset + 2]) == blue);
            CHECK(backend->bytes[offset + 3] == std::byte{255});
        }
    }
};

void use_outputless_fragment(Fixture& fixture) {
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    fragment->entry_point = "fragment_empty";
    fragment->interface.color_output_mask = 0;
    fragment->color_outputs.clear();
    fragment->constant_buffer_windows.clear();
    fragment->constant_buffer_support_index.reset();
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact = "outputless-fragment-artifact";
}

struct ExplicitTexturePlanner final : m::BindingPlanner {
    std::shared_ptr<m::BindingPlanner> delegate;
    std::optional<std::uint32_t> first_level;
    std::optional<std::uint32_t> first_layer;
    ResourceAccess access = ResourceAccess::Read;

    ExplicitTexturePlanner(std::shared_ptr<m::BindingPlanner> delegate_,
                           std::optional<std::uint32_t> first_level_,
                           std::optional<std::uint32_t> first_layer_,
                           ResourceAccess access_)
        : delegate(std::move(delegate_)), first_level(first_level_),
          first_layer(first_layer_), access(access_) {}

    RendererResult<m::Bindings> prepare(
        const RenderStop& stop, const m::AdmittedGraphicsStages& stages,
        const DrawPlan& draw, ResourceMemory& memory, BufferCache& buffers,
        TextureCache& textures, const m::BindingReuse& reuse) override {
        auto prepared = delegate->prepare(
            stop, stages, draw, memory, buffers, textures, reuse);
        if (const auto* error = std::get_if<RendererError>(&prepared))
            return *error;
        auto bindings = std::get<m::Bindings>(std::move(prepared));
        if (bindings.textures.size() != 1)
            return RendererError({CommandStatus::Unsupported,
                                  "synthetic explicit view requires one texture"});
        auto& binding = bindings.textures.front();
        const BackingRange bytes = binding.texture.key.storage.bytes;
        auto retained = std::ranges::find_if(bindings.accesses,
            [&](const GpuAccessRequest& request) {
                return request.range.bytes == bytes;
            });
        if (retained == bindings.accesses.end())
            return RendererError({CommandStatus::Unsupported,
                                  "synthetic explicit view lacks texture access"});
        auto key = binding.texture.key;
        if (first_level) {
            key.first_level = *first_level;
            key.level_count = 1;
        }
        if (first_layer) {
            key.first_layer = *first_layer;
            key.layer_count = 1;
        }
        RetainedRange range = retained->range;
        if (access != retained->access) {
            auto upgraded = memory.retain(
                key.storage.descriptor.address,
                key.storage.descriptor.backing_extent, access);
            if (const auto* error = std::get_if<RendererError>(&upgraded))
                return *error;
            range = std::get<RetainedRange>(std::move(upgraded));
        }
        auto acquired = textures.acquire({range, key, access});
        if (const auto* error = std::get_if<RendererError>(&acquired))
            return *error;
        binding.texture = std::get<ResidentTexture>(std::move(acquired));
        bindings.dependencies.insert(bindings.dependencies.end(),
            binding.texture.dependencies.begin(),
            binding.texture.dependencies.end());
        retained->range = std::move(range);
        retained->access = access;
        return bindings;
    }
};

void write_word(Memory& memory, std::size_t offset, std::uint32_t value) {
    for (unsigned byte = 0; byte != 4; ++byte) {
        memory.bytes[offset + byte] =
            std::byte((value >> (byte * 8)) & 0xff);
    }
}

constexpr std::size_t storage_descriptor_cbuf_offset = 4096;
constexpr std::uint32_t storage_descriptor_byte_offset = 16;
constexpr std::size_t storage_data_offset = 49152;
constexpr std::uint64_t high_storage_alias = 0x100200000ULL;

void write_storage_descriptor(Fixture& fixture, RenderStop& stop,
                              std::size_t stage_index,
                              std::uint64_t address, std::uint32_t size,
                              bool enabled = true,
                              std::uint32_t cbuf_size = 64) {
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, storage_descriptor_cbuf_offset, 64}));
    write_word(*fixture.backend,
               storage_descriptor_cbuf_offset + storage_descriptor_byte_offset,
               static_cast<std::uint32_t>(address));
    write_word(*fixture.backend,
               storage_descriptor_cbuf_offset + storage_descriptor_byte_offset + 4,
               static_cast<std::uint32_t>(address >> 32));
    write_word(*fixture.backend,
               storage_descriptor_cbuf_offset + storage_descriptor_byte_offset + 8,
               size);
    transaction->commit();
    stop.constant_buffers[stage_index][2] = {
        gpu_base + storage_descriptor_cbuf_offset, cbuf_size, enabled};
}

void write_storage_words(Fixture& fixture,
                         std::span<const std::uint32_t> words) {
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, storage_data_offset, words.size_bytes()}));
    std::memcpy(fixture.backend->bytes.data() + storage_data_offset,
                words.data(), words.size_bytes());
    transaction->commit();
}

void use_vertex_storage_reader(Fixture& fixture) {
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto vertex = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.vertex);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    vertex->entry_point = "vertex_storage_read";
    vertex->constant_buffers.clear();
    vertex->constant_buffer_windows = {{2, 64, 0}};
    vertex->constant_buffer_support_index = 17;
    vertex->storage_buffers = {{
        2, storage_descriptor_byte_offset, 3,
        platform::CompiledShaderResourceAccess::Read, 4}};
    vertex->interface.generic_outputs = {{0, 1}};
    fragment->entry_point = "fragment_storage_color";
    fragment->constant_buffers.clear();
    fragment->constant_buffer_windows.clear();
    fragment->constant_buffer_support_index.reset();
    fragment->storage_buffers.clear();
    fragment->interface.generic_inputs = {{0, 1}};
    stages->admitted.stages = {vertex, fragment};
    stages->admitted.vertex_key.entry_point = vertex->entry_point;
    stages->admitted.vertex_key.artifact = "synthetic-vertex-storage-read";
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact = "synthetic-fragment-storage-color";
}

void use_fragment_storage_update(
    Fixture& fixture,
    platform::CompiledShaderResourceAccess access =
        platform::CompiledShaderResourceAccess::Write,
    std::uint32_t alignment = 4) {
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto vertex = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.vertex);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    vertex->interface.generic_outputs.clear();
    vertex->constant_buffers.clear();
    vertex->constant_buffer_windows.clear();
    vertex->constant_buffer_support_index.reset();
    vertex->storage_buffers.clear();
    fragment->entry_point = "fragment_storage_update";
    fragment->constant_buffers.clear();
    fragment->constant_buffer_windows = {{2, 64, 0}};
    fragment->constant_buffer_support_index = 17;
    fragment->storage_buffers = {{
        2, storage_descriptor_byte_offset, 3, access, alignment}};
    fragment->interface.generic_inputs.clear();
    stages->admitted.stages = {vertex, fragment};
    stages->admitted.vertex_key.entry_point = vertex->entry_point;
    stages->admitted.vertex_key.artifact = "synthetic-vertex-storage-update";
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact = "synthetic-fragment-storage-update";
}

constexpr std::size_t argument_tic_offset = 24576;
constexpr std::size_t argument_tsc_offset = argument_tic_offset + 1024;
constexpr std::size_t argument_source_offset = 32768;
constexpr std::size_t argument_source_stride = 1024;
constexpr std::size_t argument_handle_offset = 9216;
constexpr std::size_t argument_handle_stride = 192;
constexpr unsigned argument_resource_count = 20;
using ArgumentColors = std::array<std::array<std::byte, 4>, argument_resource_count>;

platform::CompiledShaderResource argument_texture_resource() {
    platform::CompiledShaderResource resource;
    resource.resource_id = 0;
    resource.array_length = argument_resource_count;
    resource.cbuf_slot = 4;
    resource.handle_byte_offset = 0;
    resource.kind = platform::CompiledShaderResourceKind::SampledTexture;
    resource.access = platform::CompiledShaderResourceAccess::Read;
    resource.comparison = false;
    resource.normalized_coordinates = true;
    resource.texture_descriptor = platform::CompiledShaderTextureDescriptor{
        platform::CompiledShaderTextureDimension::d2, false, false,
        platform::CompiledShaderTextureScalar::f32,
        platform::TextureComponentFormat::floating, true, false};
    resource.location.kind =
        platform::CompiledShaderResourceLocationKind::ArgumentBufferMember;
    resource.location.byte_offset = 0;
    resource.location.count = argument_resource_count;
    resource.location.stride = 8;
    return resource;
}

platform::CompiledShaderResource argument_sampler_resource() {
    platform::CompiledShaderResource resource;
    resource.resource_id = 1;
    resource.array_length = argument_resource_count;
    resource.cbuf_slot = 4;
    resource.handle_byte_offset = argument_resource_count * 4;
    resource.kind = platform::CompiledShaderResourceKind::Sampler;
    resource.access = platform::CompiledShaderResourceAccess::Read;
    resource.comparison = false;
    resource.normalized_coordinates = true;
    resource.location.kind =
        platform::CompiledShaderResourceLocationKind::ArgumentBufferMember;
    resource.location.byte_offset = argument_resource_count * 8;
    resource.location.count = argument_resource_count;
    resource.location.stride = 8;
    return resource;
}

void use_argument_fragment(Fixture& fixture, const char* entry, bool array = false) {
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    fragment->entry_point = entry;
    fragment->constant_buffer_windows.clear();
    fragment->constant_buffer_support_index.reset();
    fragment->textures.clear();
    fragment->texture_facts.clear();
    fragment->texture_handle_cbuf = 4;
    fragment->resources = {
        argument_texture_resource(), argument_sampler_resource()};
    fragment->resources[0].texture_descriptor->array = array;
    fragment->argument_buffer = platform::CompiledArgumentBufferLayout{
        1, 2, argument_resource_count * 2 * 8, 8};
    fragment->legacy_texture_bindings_compatible = false;
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = entry;
    stages->admitted.fragment_key.artifact =
        std::string("synthetic-") + entry;
}

ArgumentColors argument_colors() {
    ArgumentColors colors{};
    for (unsigned index = 0; index != colors.size(); ++index) {
        colors[index] = {
            std::byte{static_cast<unsigned char>(13 + index * 9)},
            std::byte{static_cast<unsigned char>(211 - index * 7)},
            std::byte{static_cast<unsigned char>((index * 47 + 19) & 0xff)},
            std::byte{255}};
    }
    return colors;
}

void configure_argument_inputs(
    Fixture& fixture, RenderStop& stop, const ArgumentColors& colors,
    std::array<std::optional<std::size_t>, argument_resource_count> sources,
    unsigned handle_blocks = 1) {
    const std::size_t source_end = argument_source_offset +
                                   argument_resource_count * argument_source_stride;
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, argument_handle_offset,
         source_end - argument_handle_offset}));
    for (unsigned block = 0; block != handle_blocks; ++block) {
        const auto base = argument_handle_offset + block * argument_handle_stride;
        for (unsigned index = 0; index != argument_resource_count; ++index) {
            const auto descriptor = (index + block) % argument_resource_count;
            write_word(*fixture.backend, base + index * 4, descriptor);
            write_word(*fixture.backend,
                       base + argument_resource_count * 4 + index * 4,
                       descriptor << 20);
        }
    }

    constexpr std::array<std::uint32_t, 5> wrap_modes{0, 1, 2, 3, 5};
    for (unsigned index = 0; index != argument_resource_count; ++index) {
        const auto tic = argument_tic_offset + index * texture_descriptor_size;
        std::uint32_t format_word = 0x08;
        constexpr std::array<std::uint32_t, 4> swizzle{2, 3, 4, 5};
        for (unsigned channel = 0; channel != 4; ++channel) {
            format_word |= 2u << (7 + channel * 3);
            format_word |= swizzle[channel] << (19 + channel * 3);
        }
        const std::uint64_t address = sources[index]
            ? gpu_base + *sources[index] : 0;
        write_word(*fixture.backend, tic, format_word);
        write_word(*fixture.backend, tic + 4,
                   static_cast<std::uint32_t>(address));
        write_word(*fixture.backend, tic + 8,
                   static_cast<std::uint32_t>(address >> 32) | (3u << 21));
        write_word(*fixture.backend, tic + 12, 0);
        write_word(*fixture.backend, tic + 16, 15u | (1u << 23));
        write_word(*fixture.backend, tic + 20, 15u | (1u << 31));
        write_word(*fixture.backend, tic + 24, 0);
        write_word(*fixture.backend, tic + 28, 0);

        const auto tsc = argument_tsc_offset + index * texture_descriptor_size;
        const auto wrap_u = wrap_modes[index % wrap_modes.size()];
        const auto wrap_v = wrap_modes[index / wrap_modes.size()];
        write_word(*fixture.backend, tsc,
                   wrap_u | (wrap_v << 3) | (2u << 6));
        write_word(*fixture.backend, tsc + 4,
                   1u | (1u << 4) | (1u << 6));
        for (unsigned word = 2; word != 8; ++word)
            write_word(*fixture.backend, tsc + word * 4, 0);

        if (sources[index] && *sources[index] >= argument_source_offset &&
            *sources[index] < source_end) {
            std::fill_n(fixture.backend->bytes.begin() + *sources[index],
                        argument_source_stride, std::byte{0});
            for (std::size_t y = 0; y != 16; ++y) {
                for (std::size_t x = 0; x != 16; ++x) {
                    const auto texel = *sources[index] +
                        gob_offset(x * 4, y, 0, 0, 64);
                    std::copy(colors[index].begin(), colors[index].end(),
                              fixture.backend->bytes.begin() + texel);
                }
            }
        }
    }
    transaction->commit();

    stop.registers[0x982] = 4;
    stop.registers[0x55e] = gpu_base + argument_tic_offset;
    stop.registers[0x55f] = argument_resource_count - 1;
    stop.registers[0x558] = gpu_base + argument_tsc_offset;
    stop.registers[0x559] = argument_resource_count - 1;
    stop.constant_buffers[4][4] = {
        gpu_base + argument_handle_offset,
        static_cast<std::uint32_t>(argument_resource_count * 2 * 4), true};
}

std::array<std::optional<std::size_t>, argument_resource_count>
argument_sources() {
    std::array<std::optional<std::size_t>, argument_resource_count> sources;
    for (unsigned index = 0; index != sources.size(); ++index)
        sources[index] = argument_source_offset + index * argument_source_stride;
    return sources;
}

void check_pitch_color(const Memory& memory, std::size_t offset,
                       const std::array<std::byte, 4>& color,
                       unsigned tolerance = 0) {
    for (std::size_t pixel = 0; pixel != 16 * 16; ++pixel) {
        for (unsigned channel = 0; channel != color.size(); ++channel) {
            const auto actual = std::to_integer<unsigned>(
                memory.bytes[offset + pixel * 4 + channel]);
            const auto expected = std::to_integer<unsigned>(color[channel]);
            CHECK(actual + tolerance >= expected &&
                  actual <= expected + tolerance);
        }
    }
}

void check_block_color(const Memory& memory, std::size_t offset,
                       const std::array<std::byte, 4>& color) {
    for (std::size_t y = 0; y != 16; ++y) {
        for (std::size_t x = 0; x != 16; ++x) {
            const auto texel = offset + gob_offset(x * 4, y, 0, 0, 64);
            CHECK(std::equal(color.begin(), color.end(),
                             memory.bytes.begin() + texel));
        }
    }
}

void persistent_argument_buffer_many_resources() {
    Fixture fixture;
    auto draw = snapshot();
    configure_target(draw, target_offset, 64, 16, 0xd5, 0x1000, 16, 16);
    const auto colors = argument_colors();
    configure_argument_inputs(
        fixture, draw, colors, argument_sources());
    use_argument_fragment(fixture, "argument_many");

    const auto receipt = fixture.enqueue(
        draw, 1, std::make_shared<Effects>());
    CHECK(receipt.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());

    std::array<std::byte, 4> expected{};
    for (unsigned channel = 0; channel != expected.size(); ++channel) {
        unsigned weighted = 0;
        for (unsigned index = 0; index != colors.size(); ++index) {
            weighted += std::to_integer<unsigned>(colors[index][channel]) *
                        (index + 1);
        }
        expected[channel] = std::byte{static_cast<unsigned char>(
            (weighted + 105) / 210)};
    }
    check_pitch_color(*fixture.backend, target_offset, expected, 1);
    CHECK(fixture.pipelines->sampler_keys.size() == argument_resource_count);
    for (unsigned left = 0; left != argument_resource_count; ++left) {
        for (unsigned right = left + 1; right != argument_resource_count;
             ++right) {
            CHECK(fixture.pipelines->sampler_keys[left].raw_words[0] !=
                  fixture.pipelines->sampler_keys[right].raw_words[0]);
        }
    }
}

void persistent_argument_buffer_each_member() {
    Fixture fixture;
    auto draw = snapshot();
    const auto colors = argument_colors();
    configure_argument_inputs(fixture, draw, colors, argument_sources());
    use_argument_fragment(fixture, "argument_each");
    auto write = take(fixture.memory->prepare_cpu_write({
        {1, 1, 1}, argument_source_offset,
        argument_resource_count * argument_source_stride}));
    for (unsigned index = 0; index != argument_resource_count; ++index) {
        for (unsigned y = 0; y != 16; ++y) {
            for (unsigned x = 0; x != 16; ++x) {
                const auto offset = argument_source_offset + index * argument_source_stride +
                    gob_offset(x * 4, y, 0, 0, 64);
                fixture.backend->bytes[offset] = std::byte(x * 13 + index);
                fixture.backend->bytes[offset + 1] = std::byte(y * 13 + index);
            }
        }
    }
    write->commit();
    const auto receipt = fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    // Interior probes identify every texture. Probing only one outside axis at
    // a time distinguishes every sampler configuration without border masking.
    constexpr unsigned negative_texels[] = {12, 3, 0, 0, 3};
    constexpr unsigned positive_texels[] = {3, 12, 15, 0, 15};
    for (unsigned y = 0; y != 16; ++y) {
        for (unsigned x = 0; x != 16; ++x) {
            const auto index = (x + y * 16) % argument_resource_count;
            const auto wrap_u = index % 5, wrap_v = index / 5;
            const auto probe = y / 4;
            const bool border = ((probe == 1 || probe == 2) && wrap_u == 3) ||
                                (probe == 3 && wrap_v == 3);
            std::array<std::byte, 4> expected{};
            if (!border) {
                const auto source_x = probe == 1 ? negative_texels[wrap_u] :
                    probe == 2 ? positive_texels[wrap_u] : 8;
                const auto source_y = probe == 3 ? negative_texels[wrap_v] : 8;
                expected = {std::byte(source_x * 13 + index),
                    std::byte(source_y * 13 + index), colors[index][2], std::byte{255}};
            }
            const auto offset = target_offset + (y * 16 + x) * 4;
            for (unsigned channel = 0; channel != 4; ++channel) {
                const auto actual = fixture.backend->bytes[offset + channel];
                if (actual != expected[channel])
                    std::fprintf(stderr, "Argument member %u probe %u channel %u: %u != %u\n",
                        index, probe, channel, std::to_integer<unsigned>(actual),
                        std::to_integer<unsigned>(expected[channel]));
                CHECK(actual == expected[channel]);
            }
        }
    }
}

void persistent_direct_resource_high_indices() {
    Fixture fixture;
    auto draw = snapshot();
    const auto colors = argument_colors();
    configure_argument_inputs(fixture, draw, colors, argument_sources());
    use_argument_fragment(fixture, "direct_high");
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.fragment);
    fragment->argument_buffer.reset();
    fragment->resources[0].array_length = 1;
    fragment->resources[0].location = {platform::CompiledShaderResourceLocationKind::Direct,
        platform::CompiledShaderResourceNamespace::Texture, 127, 0, 1, 0};
    fragment->resources[1].array_length = 1;
    fragment->resources[1].location = {platform::CompiledShaderResourceLocationKind::Direct,
        platform::CompiledShaderResourceNamespace::Sampler, 15, 0, 1, 0};
    stages->admitted.stages.fragment = fragment;
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    check_pitch_color(*fixture.backend, target_offset, colors[0]);
}

void persistent_argument_buffer_typed_null() {
    Fixture fixture;
    auto draw = snapshot();
    configure_target(draw, target_offset, 64, 16, 0xd5, 0x1000, 16, 16);
    auto sources = argument_sources();
    sources[0].reset();
    configure_argument_inputs(
        fixture, draw, argument_colors(), sources);
    use_argument_fragment(fixture, "argument_first");

    const auto receipt = fixture.enqueue(
        draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    check_pitch_color(
        *fixture.backend, target_offset,
        {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{255}});
}

void persistent_argument_buffer_feedback_substitution(bool array = false) {
    Fixture fixture;
    fixture.uniform(0.25f);
    auto producer = snapshot();
    configure_target(producer, target_offset, 16, 16, 0xd5, 0,
                     16, 16);

    auto feedback = snapshot();
    configure_target(feedback, target_offset, 16, 16, 0xd5, 0,
                     16, 16);
    auto sources = argument_sources();
    sources[0] = target_offset;
    configure_argument_inputs(
        fixture, feedback, argument_colors(), sources);

    fixture.enqueue(producer, 1, std::make_shared<Effects>());
    use_argument_fragment(fixture, array ? "argument_array_step" : "argument_first_step", array);
    fixture.enqueue(feedback, 2, std::make_shared<Effects>());
    const auto last = fixture.enqueue(
        feedback, 3, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(last.completion.snapshot().guest_visible_complete());
    check_block_color(
        *fixture.backend, target_offset,
        {std::byte{192}, std::byte{64}, std::byte{64}, std::byte{255}});
}

void persistent_argument_buffer_completion_lifetime_and_reuse() {
    Fixture fixture;
    auto draw = snapshot();
    const auto colors = argument_colors();
    configure_argument_inputs(
        fixture, draw, colors, argument_sources(), 5);
    use_argument_fragment(fixture, "argument_first");

    constexpr std::size_t output_base = 131072;
    constexpr std::size_t output_stride = 2048;
    std::vector<SubmissionReceipt> receipts;
    receipts.reserve(4);
    for (unsigned index = 0; index != 4; ++index) {
        auto queued = draw;
        configure_target(queued, output_base + index * output_stride,
                         64, 16, 0xd5, 0x1000, 16, 16);
        queued.constant_buffers[4][4].address =
            gpu_base + argument_handle_offset + index * argument_handle_stride;
        receipts.push_back(fixture.enqueue(
            queued, index + 1, std::make_shared<Effects>()));
    }
    CHECK(std::ranges::all_of(receipts, [](const auto& receipt) {
        return receipt.completion.snapshot().phase == SubmissionPhase::Accepted;
    }));
    CHECK(ok(fixture.renderer->drain(deadline())));
    for (unsigned index = 0; index != receipts.size(); ++index) {
        CHECK(receipts[index].completion.snapshot().guest_visible_complete());
        check_pitch_color(*fixture.backend,
                          output_base + index * output_stride,
                          colors[index]);
    }

    auto reused = draw;
    configure_target(reused, output_base + 4 * output_stride,
                     64, 16, 0xd5, 0x1000, 16, 16);
    reused.constant_buffers[4][4].address =
        gpu_base + argument_handle_offset + 4 * argument_handle_stride;
    const auto final = fixture.enqueue(
        reused, 5, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(final.completion.snapshot().guest_visible_complete());
    check_pitch_color(*fixture.backend, output_base + 4 * output_stride,
                      colors[4]);
}

void configure_texture_consumer(Fixture& fixture, RenderStop& stop,
                                const char* entry,
                                platform::TextureFactDimension dimension,
                                std::uint32_t tic_target,
                                std::size_t source_offset,
                                std::uint32_t width,
                                std::uint32_t height,
                                std::uint32_t depth,
                                std::uint32_t format_base,
                                std::array<std::uint32_t, 4> swizzle,
                                std::uint32_t block_height_log2,
                                std::uint32_t block_depth_log2,
                                bool array = false,
                                std::uint32_t levels = 1,
                                std::uint64_t source_base = gpu_base) {
    constexpr std::size_t tic_offset = 24576;
    constexpr std::size_t tsc_offset = tic_offset + 64;
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, uniform_offset,
         tsc_offset + texture_descriptor_size - uniform_offset}));
    write_word(*fixture.backend, uniform_offset + 32, 0);
    std::uint32_t format_word = format_base;
    for (unsigned channel = 0; channel != 4; ++channel) {
        format_word |= 2u << (7 + channel * 3);
        format_word |= swizzle[channel] << (19 + channel * 3);
    }
    write_word(*fixture.backend, tic_offset, format_word);
    const std::uint64_t source_address = source_base + source_offset;
    write_word(*fixture.backend, tic_offset + 4,
               static_cast<std::uint32_t>(source_address));
    write_word(*fixture.backend, tic_offset + 8,
               static_cast<std::uint32_t>(source_address >> 32) | (3u << 21));
    write_word(*fixture.backend, tic_offset + 12,
               (block_height_log2 << 3) | (block_depth_log2 << 6) |
                   ((levels - 1) << 28));
    write_word(*fixture.backend, tic_offset + 16,
               (width - 1) | (tic_target << 23));
    write_word(*fixture.backend, tic_offset + 20,
               (height - 1) | ((depth - 1) << 16) | (1u << 31));
    write_word(*fixture.backend, tic_offset + 28, (levels - 1) << 4);
    write_word(*fixture.backend, tsc_offset,
               2u | (2u << 3) | (2u << 6));
    write_word(*fixture.backend, tsc_offset + 4,
               1u | (1u << 4) | (1u << 6));
    transaction->commit();

    stop.registers[0x982] = 4;
    stop.registers[0x55e] = gpu_base + tic_offset;
    stop.registers[0x558] = gpu_base + tsc_offset;
    stop.constant_buffers[4][4] = {gpu_base + uniform_offset, 36, true};
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    fragment->entry_point = entry;
    fragment->constant_buffer_windows.clear();
    fragment->constant_buffer_support_index.reset();
    fragment->texture_handle_cbuf = 4;
    fragment->textures = {{32, 0, std::uint8_t{0}}};
    platform::DerivedTextureFact fact{4, 32, true, false};
    fact.dimension = dimension;
    fact.array = array;
    fact.scalar = platform::TextureSampledScalar::f32;
    fact.component_format = platform::TextureComponentFormat::floating;
    fragment->texture_facts = {fact};
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = entry;
    stages->admitted.fragment_key.artifact =
        std::string("synthetic-") + entry;
}

void use_custom_border_fragment(Fixture& fixture) {
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    fragment->entry_point = "custom_border_corner";
    fragment->textures.clear();
    fragment->texture_facts.clear();
    fragment->texture_handle_cbuf.reset();
    fragment->resources.clear();
    platform::CompiledShaderResource texture;
    texture.resource_id = 10;
    texture.array_length = 1;
    texture.cbuf_slot = 4;
    texture.handle_byte_offset = 32;
    texture.kind = platform::CompiledShaderResourceKind::SampledTexture;
    texture.texture_descriptor = platform::CompiledShaderTextureDescriptor{};
    texture.texture_descriptor->component_format =
        platform::TextureComponentFormat::unorm;
    texture.location = {
        platform::CompiledShaderResourceLocationKind::Direct,
        platform::CompiledShaderResourceNamespace::Texture, 3, 0, 1, 0};
    platform::CompiledShaderResource sampler = texture;
    sampler.resource_id = 11;
    sampler.kind = platform::CompiledShaderResourceKind::Sampler;
    sampler.texture_descriptor.reset();
    sampler.location = {
        platform::CompiledShaderResourceLocationKind::Direct,
        platform::CompiledShaderResourceNamespace::Sampler, 5, 0, 1, 0};
    fragment->resources = {texture, sampler};
    fragment->custom_border =
        platform::CompiledCustomBorderSupport{2, {{10, 11}}};
    fragment->legacy_texture_bindings_compatible = false;
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact =
        "synthetic-custom-border-corner";
}

void configure_border_sampler(Fixture& fixture,
                              const std::array<float, 4>& color) {
    constexpr std::size_t tsc_offset = 24576 + 64;
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, tsc_offset, texture_descriptor_size}));
    write_word(*fixture.backend, tsc_offset,
               3u | (3u << 3) | (3u << 6));
    write_word(*fixture.backend, tsc_offset + 4,
               2u | (2u << 4) | (1u << 6));
    write_word(*fixture.backend, tsc_offset + 8, 0);
    write_word(*fixture.backend, tsc_offset + 12, 0);
    for (unsigned component = 0; component < color.size(); ++component) {
        write_word(*fixture.backend, tsc_offset + 16 + component * 4,
                   std::bit_cast<std::uint32_t>(color[component]));
    }
    transaction->commit();
}

struct CustomBorderPlannerProbe final : m::BindingPlanner {
    std::shared_ptr<m::BindingPlanner> delegate;
    bool prepared = false;
    bool uniform_support = false;
    bool retained_coverage = false;
    bool host_generated_support = false;
    std::optional<std::uint32_t> support_enabled;

    explicit CustomBorderPlannerProbe(std::shared_ptr<m::BindingPlanner> next)
        : delegate(std::move(next)) {}

    RendererResult<m::Bindings> prepare(
        const RenderStop& stop, const m::AdmittedGraphicsStages& stages,
        const DrawPlan& draw, ResourceMemory& memory, BufferCache& buffers,
        TextureCache& textures, const m::BindingReuse& reuse) override {
        auto result = delegate->prepare(
            stop, stages, draw, memory, buffers, textures, reuse);
        auto* bindings = std::get_if<m::Bindings>(&result);
        if (!bindings) return result;
        prepared = true;
        retained_coverage = bindings->custom_border_textures.size() == 1;
        const auto support = std::ranges::find_if(
            bindings->buffers, [](const auto& binding) {
                return binding.stage == shader_input::Stage::Fragment &&
                       binding.index == 2;
            });
        uniform_support = support != bindings->buffers.end() &&
                          support->use == BufferUse::Uniform &&
                          support->access == ResourceAccess::Read;
        if (support != bindings->buffers.end()) {
            host_generated_support =
                !support->buffer.key.bytes.backing.owner;
            id<MTLBuffer> native =
                m::resident_native_buffer(support->buffer.storage);
            if (native && support->buffer.offset + 52 <= native.length) {
                std::uint32_t enabled = 0;
                std::memcpy(
                    &enabled,
                    static_cast<const std::byte*>(native.contents) +
                        support->buffer.offset + 48,
                    sizeof(enabled));
                support_enabled = enabled;
            }
        }
        return result;
    }
};

std::shared_ptr<CustomBorderPlannerProbe>
observe_custom_border_planner(Fixture& fixture) {
    auto probe = std::make_shared<CustomBorderPlannerProbe>(
        fixture.services.bindings);
    fixture.services.bindings = probe;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64}, fixture.services));
    return probe;
}

void use_explicit_texture_view(Fixture& fixture,
                               std::optional<std::uint32_t> first_level,
                               std::optional<std::uint32_t> first_layer,
                               ResourceAccess access = ResourceAccess::Read) {
    fixture.services.bindings = std::make_shared<ExplicitTexturePlanner>(
        fixture.services.bindings, first_level, first_layer, access);
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64}, fixture.services));
}

void vertex_reads_high_address_storage_descriptor() {
    Fixture fixture;
    CHECK(ok(fixture.memory->register_mapping(
        {{1, 3, 1}, {1, 1, 1},
         {high_storage_alias, fixture.backend->bytes.size(), 0, 0x10000000,
          fixture.backend->bytes.size(), 4096}})));
    const std::array<std::uint32_t, 4> initial{
        0x11223344, std::bit_cast<std::uint32_t>(0.5f),
        0xaabbccdd, 0x55667788};
    write_storage_words(fixture, initial);
    auto draw = snapshot();
    write_storage_descriptor(
        fixture, draw, 0, high_storage_alias + storage_data_offset,
        sizeof(initial));
    use_vertex_storage_reader(fixture);
    const auto receipt = fixture.enqueue(
        draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    fixture.pixels(128, 128, 0);
}

void overlapping_storage_reads_share_one_allocation() {
    Fixture fixture;
    CHECK(ok(fixture.memory->register_mapping(
        {{1, 3, 1}, {1, 1, 1},
         {high_storage_alias, fixture.backend->bytes.size(), 0, 0x10000000,
          fixture.backend->bytes.size(), 4096}})));
    const std::array<std::uint32_t, 6> initial{
        0x11223344, std::bit_cast<std::uint32_t>(0.25f),
        0xaabbccdd, std::bit_cast<std::uint32_t>(0.5f), 0, 0};
    write_storage_words(fixture, initial);
    auto draw = snapshot();
    write_storage_descriptor(fixture, draw, 0,
        high_storage_alias + storage_data_offset, 16);
    auto transaction = take(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, storage_descriptor_cbuf_offset + 32, 12}));
    const auto address = high_storage_alias + storage_data_offset + 8;
    write_word(*fixture.backend, storage_descriptor_cbuf_offset + 32, std::uint32_t(address));
    write_word(*fixture.backend, storage_descriptor_cbuf_offset + 36, std::uint32_t(address >> 32));
    write_word(*fixture.backend, storage_descriptor_cbuf_offset + 40, 16);
    transaction->commit();
    draw.constant_buffers[4][2] = draw.constant_buffers[0][2];
    use_vertex_storage_reader(fixture);
    const auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(
        *stages->admitted.stages.fragment);
    fragment->entry_point = "fragment_storage_overlap";
    fragment->constant_buffer_windows = {{2, 64, 0}};
    fragment->constant_buffer_support_index = 17;
    fragment->storage_buffers = {{2, 32, 3, platform::CompiledShaderResourceAccess::Read, 4}};
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact = "synthetic-overlapping-storage-read";
    const auto receipt = fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    fixture.pixels(191, 191, 0);
}

void fragment_storage_write_orders_cpu_and_gpu_consumers() {
    Fixture fixture;
    CHECK(ok(fixture.memory->register_mapping(
        {{1, 3, 1}, {1, 1, 1},
         {high_storage_alias, fixture.backend->bytes.size(), 0, 0x10000000,
          fixture.backend->bytes.size(), 4096}})));
    const std::array<std::uint32_t, 4> initial{
        0x11223344, std::bit_cast<std::uint32_t>(0.25f),
        0xaabbccdd, 0x55667788};
    write_storage_words(fixture, initial);

    auto producer = snapshot();
    write_storage_descriptor(
        fixture, producer, 4, high_storage_alias + storage_data_offset,
        sizeof(initial));
    use_fragment_storage_update(fixture);
    const auto produced = fixture.enqueue(
        producer, 1, std::make_shared<Effects>());

    auto consumer = snapshot();
    write_storage_descriptor(
        fixture, consumer, 0, high_storage_alias + storage_data_offset,
        sizeof(initial));
    use_vertex_storage_reader(fixture);
    const auto consumed = fixture.enqueue(
        consumer, 2, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(produced.completion.snapshot().guest_visible_complete());
    CHECK(consumed.completion.snapshot().guest_visible_complete());
    fixture.pixels(191, 191, 0);

    auto transaction = take(fixture.memory->prepare_cpu_read(
        {{1, 1, 1}, storage_data_offset, sizeof(initial)},
         ResolveReason::CpuRead));
    std::array<std::uint32_t, 4> actual{};
    std::memcpy(actual.data(),
                fixture.backend->bytes.data() + storage_data_offset,
                sizeof(actual));
    CHECK(actual[0] == initial[0]);
    CHECK(actual[1] == std::bit_cast<std::uint32_t>(0.75f));
    CHECK(actual[2] == initial[2] && actual[3] == initial[3]);
    transaction.reset();
}

void storage_descriptor_bounds_and_zero_size_guard_gpu_access() {
    const std::array<std::uint32_t, 4> guard{
        0x10203040, 0x50607080, 0x90a0b0c0, 0xd0e0f000};
    for (const bool zero_size : {false, true}) {
        Fixture fixture;
        CHECK(ok(fixture.memory->register_mapping(
            {{1, 3, 1}, {1, 1, 1},
             {high_storage_alias, fixture.backend->bytes.size(), 0,
              0x10000000, fixture.backend->bytes.size(), 4096}})));
        write_storage_words(fixture, guard);
        const auto address = zero_size
            ? std::numeric_limits<std::uint64_t>::max()
            : high_storage_alias + storage_data_offset;
        const std::uint32_t size = zero_size ? 0 : 4;
        auto draw = snapshot();
        write_storage_descriptor(fixture, draw, 4, address, size);
        use_fragment_storage_update(fixture);
        const auto receipt = fixture.enqueue(
            draw, 1, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        CHECK(receipt.completion.snapshot().guest_visible_complete());
        fixture.pixels(0, 0, 0);

        auto reader = snapshot();
        write_storage_descriptor(fixture, reader, 0, address, size);
        use_vertex_storage_reader(fixture);
        const auto read = fixture.enqueue(
            reader, 2, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        CHECK(read.completion.snapshot().guest_visible_complete());
        fixture.pixels(0, 0, 0);
        CHECK(std::equal(
            reinterpret_cast<const std::byte*>(guard.data()),
            reinterpret_cast<const std::byte*>(guard.data()) + sizeof(guard),
            fixture.backend->bytes.begin() + storage_data_offset));
    }
}

void malformed_storage_descriptors_fail_renderer_admission() {
    const auto expect_error = [](
        bool enabled, std::uint32_t cbuf_size, std::uint64_t address,
        std::uint32_t size, std::uint32_t alignment,
        CommandStatus status) {
        Fixture fixture;
        auto draw = snapshot();
        write_storage_descriptor(
            fixture, draw, 4, address, size, enabled, cbuf_size);
        use_fragment_storage_update(
            fixture, platform::CompiledShaderResourceAccess::ReadWrite,
            alignment);
        auto effects = std::make_shared<Effects>();
        auto result = fixture.renderer->enqueue(
            {{1, 1}, std::make_shared<const RenderStop>(draw),
             fixture.memory, {}, effects});
        const auto* error = std::get_if<RendererError>(&result);
        CHECK(error != nullptr);
        if (error) CHECK(error->command().status == status);
        CHECK(effects->coherent == 0 && effects->prepared == 0 &&
              effects->published == 0);
        CHECK(m::submission_statistics(*fixture.services.submissions).accepted == 0);
    };
    expect_error(false, 64, gpu_base + storage_data_offset, 16, 4,
                 CommandStatus::MemoryFault);
    expect_error(true, 24, gpu_base + storage_data_offset, 16, 4,
                 CommandStatus::MemoryFault);
    expect_error(true, 64, 0xdead00000000ULL, 16, 4,
                 CommandStatus::MemoryFault);
    expect_error(true, 64, gpu_base + storage_data_offset, 16, 8,
                 CommandStatus::Unsupported);
}

void batched_uniform_versions() {
    Fixture fixture;
    auto first_effects = std::make_shared<Effects>();
    auto second_effects = std::make_shared<Effects>();
    auto first = snapshot();
    first.registers[0x381] = 8u << 16;
    auto a = fixture.enqueue(first, 1, first_effects);
    fixture.uniform(0.25f);
    auto second = snapshot();
    second.registers[0x381] = 8u | (16u << 16);
    auto b = fixture.enqueue(second, 2, second_effects);
    CHECK(a.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(b.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(first_effects->published == 0 && second_effects->published == 0);
    CHECK(fixture.backend->bytes[target_offset] == std::byte{0});
    CHECK(ok(fixture.renderer->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(first_effects->published == 0 && second_effects->published == 0);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(a.completion.snapshot().guest_visible_complete());
    CHECK(b.completion.snapshot().guest_visible_complete());
    CHECK(first_effects->published == 1 && second_effects->published == 1);
    CHECK(fixture.pipelines->keys.size() == 2);
    CHECK(fixture.pipelines->keys[0] == fixture.pipelines->keys[1]);
    const auto statistics = m::submission_statistics(*fixture.services.submissions);
    CHECK(statistics.accepted == 2 && statistics.draws == 2);
    CHECK(statistics.command_buffers == 1 && statistics.render_passes == 1);
    fixture.pixels(255, 64, 64);
}

void buffer_growth_split() {
    Fixture fixture(true);
    auto first_effects = std::make_shared<Effects>();
    auto second_effects = std::make_shared<Effects>();
    auto a = fixture.enqueue(snapshot(true), 1, first_effects);
    auto second = snapshot(true);
    second.draw->count = 6;
    auto b = fixture.enqueue(second, 2, second_effects);
    CHECK(a.completion.snapshot().guest_visible_complete());
    CHECK(first_effects->published == 1);
    CHECK(!b.completion.snapshot().guest_visible_complete());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(second_effects->published == 1);
    CHECK(m::submission_statistics(*fixture.services.submissions).command_buffers == 2);
    fixture.pixels(255, 255, 64);
}

void overlapping_streams_and_normalized_accesses() {
    Fixture fixture(true);
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto vertex = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.vertex);
    vertex->vertex_class_inputs.push_back({1, VertexFormatClass::float_class});
    vertex->interface.generic_inputs.push_back({1, 7});
    stages->admitted.stages.vertex = vertex;
    auto draw = snapshot(true);
    draw.registers[0x459] = 1u | (2u << 21) | (7u << 27);
    draw.registers[0x704] = 0x1000 | 12;
    draw.registers[0x706] = gpu_base + vertex_offset + 12;
    draw.registers[0x7c3] = gpu_base + vertex_offset + 47;
    auto receipt = fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    fixture.pixels(255, 255, 64);
}

void unfinished_snapshot_split() {
    Fixture fixture;
    auto first = fixture.enqueue(snapshot(), 1, std::make_shared<Effects>());
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.fragment);
    fragment->entry_point = "fragment_bytes";
    stages->admitted.stages.fragment = fragment;
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    stages->admitted.fragment_key.artifact = "byte-consumer-artifact";
    auto consumer = snapshot();
    consumer.constant_buffers[4][0].address = gpu_base + target_offset;
    auto second = fixture.enqueue(consumer, 2, std::make_shared<Effects>());
    CHECK(first.completion.snapshot().guest_visible_complete());
    CHECK(!second.completion.snapshot().guest_visible_complete());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(m::submission_statistics(*fixture.services.submissions).command_buffers == 2);
    fixture.pixels(255, 255, 64);
}

void renderer_limits_and_staging_admission() {
    Fixture fixture;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 1, 1}, fixture.services));
    auto first = fixture.enqueue(snapshot(), 1, std::make_shared<Effects>());
    auto second = fixture.enqueue(snapshot(), 2, std::make_shared<Effects>());
    CHECK(first.completion.snapshot().guest_visible_complete());
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(m::submission_statistics(*fixture.services.submissions).command_buffers == 2);
    fixture.renderer = take(m::create_renderer({16 * 1024 * 1024, 16, 1, 1}, fixture.services));
    auto effects = std::make_shared<Effects>();
    auto rejected = fixture.renderer->enqueue({{1, 3}, std::make_shared<const RenderStop>(snapshot()),
        fixture.memory, {}, effects});
    CHECK(std::holds_alternative<RendererError>(rejected));
    CHECK(std::get<RendererError>(rejected).command().status == CommandStatus::BudgetExceeded);
    CHECK(effects->published == 0);
    CHECK(m::submission_statistics(*fixture.services.submissions).accepted == 2);
}

void divided_instance_fetches() {
    Fixture fixture(true);
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto vertex = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.vertex);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.fragment);
    vertex->entry_point = "vertex_instances";
    vertex->vertex_class_inputs.push_back({1, VertexFormatClass::float_class});
    vertex->interface.generic_inputs.push_back({1, 7});
    vertex->interface.generic_outputs.push_back({0, 7});
    fragment->entry_point = "fragment_instances";
    fragment->interface.generic_inputs.push_back({0, 7});
    fragment->constant_buffer_windows.clear();
    fragment->constant_buffer_support_index.reset();
    stages->admitted.stages = {vertex, fragment};
    stages->admitted.vertex_key.entry_point = vertex->entry_point;
    stages->admitted.fragment_key.entry_point = fragment->entry_point;
    const std::array<float, 18> quad{-1,-1,0, 1,-1,0, -1,1,0, -1,1,0, 1,-1,0, 1,1,0};
    std::memcpy(fixture.backend->bytes.data() + vertex_offset, quad.data(), sizeof(quad));
    constexpr auto instances_offset = vertex_offset + 4096;
    const std::array<float, 12> colors{0,0,1, 1,1,1, 1,0,0, 0,1,0};
    std::memcpy(fixture.backend->bytes.data() + instances_offset, colors.data(), sizeof(colors));
    auto draw = snapshot(true);
    draw.draw->count = 6;
    draw.draw->base_instance = 7;
    draw.draw->instance_count = 4;
    draw.registers[0x459] = 1u | (2u << 21) | (7u << 27);
    draw.registers[0x704] = 0x1000 | 12;
    draw.registers[0x706] = gpu_base + instances_offset;
    draw.registers[0x707] = 3;
    draw.registers[0x621] = 1;
    draw.registers[0x7c3] = gpu_base + instances_offset + sizeof(colors) - 1;
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    const auto& key = fixture.pipelines->keys.back();
    CHECK(key.buffers.size() == 2 && key.buffers[1].rate == 1);
    for (unsigned y = 0; y != 16; ++y) for (unsigned x = 0; x != 16; ++x) {
        const auto offset = target_offset + y * 64 + x * 4;
        CHECK(fixture.backend->bytes[offset] == std::byte(x < 8 ? 255 : 0));
        CHECK(fixture.backend->bytes[offset + 1] == std::byte(x < 8 ? 0 : 255));
        CHECK(fixture.backend->bytes[offset + 2] == std::byte{0});
        CHECK(fixture.backend->bytes[offset + 3] == std::byte{255});
    }
}

void rejected_request_and_domain() {
    Fixture fixture;
    auto effects = std::make_shared<Effects>();
    auto bad = snapshot();
    bad.registers[0x671] = 1;
    auto rejected = fixture.renderer->enqueue({{1, 1}, std::make_shared<const RenderStop>(bad),
        fixture.memory, {}, effects});
    CHECK(std::holds_alternative<RendererError>(rejected));
    CHECK(effects->coherent == 0 && effects->prepared == 0 && effects->published == 0);
    CHECK(fixture.backend->bytes[target_offset] == std::byte{0});
    auto good = fixture.enqueue(snapshot(), 1, effects);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(good.completion.snapshot().guest_visible_complete());
    auto duplicate = fixture.renderer->enqueue({{1, 1}, std::make_shared<const RenderStop>(snapshot()),
        fixture.memory, {}, effects});
    CHECK(std::holds_alternative<RendererError>(duplicate));
    auto services = fixture.services;
    services.device = take(m::create_device_context());
    CHECK(std::holds_alternative<RendererError>(m::create_renderer({1,1,1,1}, services)));
}

void guarded_preparation_failure_retries_after_publication() {
    Fixture fixture;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64, true}, fixture.services));
    auto guard = take(fixture.memory->defer_host_store(
        {{1, 1, 1}, uniform_offset, 4}));
    auto invalid = snapshot();
    invalid.registers[0x671] = 1;
    const auto request = [&] {
        return fixture.renderer->enqueue(
            {{1, 1}, std::make_shared<const RenderStop>(invalid),
             fixture.memory, {}, std::make_shared<Effects>()});
    };
    const auto blocked = request();
    const auto* transient = std::get_if<RendererError>(&blocked);
    CHECK(transient != nullptr);
    if (transient)
        CHECK(m::commit_hold_blocked(RendererStatus(*transient)));
    CHECK(m::submission_statistics(*fixture.services.submissions).accepted == 0);
    guard.reset();
    const auto permanent = request();
    const auto* rejected = std::get_if<RendererError>(&permanent);
    CHECK(rejected != nullptr);
    if (rejected) {
        CHECK(!m::commit_hold_blocked(RendererStatus(*rejected)));
        CHECK(rejected->command().status == CommandStatus::Unsupported);
    }
}

void persistent_render_target_layout_draws() {
    constexpr std::uint32_t width = 37;
    constexpr std::uint32_t height = 257;
    constexpr std::byte sentinel{0x5a};
    const std::array<std::byte, 4> color{
        std::byte{64}, std::byte{0}, std::byte{64}, std::byte{255}};
    for (std::uint32_t block_height_log2 = 0; block_height_log2 <= 5;
         ++block_height_log2) {
        Fixture fixture;
        const std::size_t row_bytes = ((width * 4 + 63) / 64) * 64;
        const std::size_t block_rows = std::size_t{8} << block_height_log2;
        const std::size_t storage_rows =
            (height + block_rows - 1) / block_rows * block_rows;
        const std::size_t storage_size = row_bytes * storage_rows;
        std::fill_n(fixture.backend->bytes.begin() + target_offset,
                    storage_size, sentinel);
        auto draw = snapshot();
        configure_target(draw, target_offset, width, height, 0xd5,
                         block_height_log2 << 4, width, height);
        fixture.uniform(0.25f);
        const auto receipt = fixture.enqueue(
            draw, 1, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        CHECK(receipt.completion.snapshot().guest_visible_complete());
        const auto expected = expected_block_linear(
            width, height, 4, block_height_log2, 0, color, sentinel,
            storage_size);
        CHECK(std::equal(expected.begin(), expected.end(),
                         fixture.backend->bytes.begin() + target_offset));
    }
}

void persistent_array_stride_draw() {
    Fixture fixture;
    constexpr std::uint32_t width = 19;
    constexpr std::uint32_t height = 17;
    constexpr std::size_t level_size = 4096;
    constexpr std::size_t layer_stride = 4096 * 4;
    constexpr std::size_t storage_size = 2 * layer_stride + level_size;
    constexpr std::byte sentinel{0x47};
    std::fill_n(fixture.backend->bytes.begin() + target_offset,
                storage_size, sentinel);
    auto draw = snapshot();
    configure_target(draw, target_offset, width, height, 0xd5, 0x10,
                     width, height);
    draw.color_targets[0].depth = 3;
    draw.color_targets[0].array_pitch = 4096;
    draw.color_targets[0].base_layer = 1;
    draw.registers[0x206] = 3;
    draw.registers[0x207] = 4096;
    draw.registers[0x208] = 1;
    fixture.uniform(0.25f);
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    const std::array<std::byte, 4> color{
        std::byte{64}, std::byte{0}, std::byte{64}, std::byte{255}};
    std::vector<std::byte> expected(storage_size, sentinel);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t offset = layer_stride +
                gob_offset(x * 4, y, 1, 0, width * 4);
            std::copy(color.begin(), color.end(), expected.begin() + offset);
        }
    }
    CHECK(std::equal(expected.begin(), expected.end(),
                     fixture.backend->bytes.begin() + target_offset));
    CHECK(std::all_of(
        fixture.backend->bytes.begin() + target_offset,
        fixture.backend->bytes.begin() + target_offset + layer_stride,
        [](std::byte value) { return value == sentinel; }));
    CHECK(std::all_of(
        fixture.backend->bytes.begin() + target_offset + 2 * layer_stride,
        fixture.backend->bytes.begin() + target_offset + storage_size,
        [](std::byte value) { return value == sentinel; }));
}

void persistent_pitch_clear_padding() {
    Fixture fixture;
    constexpr std::uint32_t pitch = 176;
    constexpr std::uint32_t width = 19;
    constexpr std::uint32_t height = 19;
    constexpr std::byte sentinel{0x33};
    constexpr std::size_t storage_size = pitch * height;
    std::fill_n(fixture.backend->bytes.begin() + target_offset,
                storage_size, sentinel);

    auto clear = snapshot();
    clear.kind = RenderKind::Clear;
    clear.original.method = 0x674;
    clear.original.argument = 0xfu << 2;
    clear.registers[0x674] = clear.original.argument;
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.25f);
    clear.registers[0x361] = std::bit_cast<std::uint32_t>(0.5f);
    clear.registers[0x362] = std::bit_cast<std::uint32_t>(0.75f);
    clear.registers[0x363] = std::bit_cast<std::uint32_t>(1.0f);
    configure_target(clear, target_offset, pitch, height, 0xd5, 0x1000,
                     width, height);
    fixture.enqueue(clear, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    const std::array<std::byte, 4> first{
        std::byte{64}, std::byte{128}, std::byte{191}, std::byte{255}};
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const auto offset = target_offset + y * pitch + x * 4;
            CHECK(std::equal(first.begin(), first.end(),
                             fixture.backend->bytes.begin() + offset));
        }
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + target_offset + y * pitch + width * 4,
            fixture.backend->bytes.begin() + target_offset + (y + 1) * pitch,
            [](std::byte value) { return value == sentinel; }));
    }

    clear.original.argument = 1u << 2;
    clear.registers[0x674] = clear.original.argument;
    clear.registers[0x43e] = 1u << 8;
    clear.registers[0x381] = 4u | (12u << 16);
    clear.registers[0x382] = 3u | (10u << 16);
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.5f);
    fixture.enqueue(clear, 2, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            auto expected = first;
            if (x >= 4 && x < 12 && y >= 3 && y < 10) {
                expected[0] = std::byte{128};
            }
            const auto offset = target_offset + y * pitch + x * 4;
            CHECK(std::equal(expected.begin(), expected.end(),
                             fixture.backend->bytes.begin() + offset));
        }
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + target_offset + y * pitch + width * 4,
            fixture.backend->bytes.begin() + target_offset + (y + 1) * pitch,
            [](std::byte value) { return value == sentinel; }));
    }
}

void render_to_real_tic(bool volume) {
    Fixture fixture;
    const std::uint32_t width = volume ? 64 : 16;
    const std::uint32_t height = 16;
    const std::uint32_t bytes_per_texel = volume ? 1 : 4;
    const std::uint32_t format = volume ? 0xf3 : 0xd5;
    const std::uint32_t tile_mode = volume ? 0x10400 : 0;
    const std::uint32_t block_depth_log2 = volume ? 4 : 0;
    const std::size_t storage_size = volume ? 8704 : 1024;
    constexpr std::byte sentinel{0x6d};
    std::fill_n(fixture.backend->bytes.begin() + target_offset,
                storage_size, sentinel);

    auto producer = snapshot();
    configure_target(producer, target_offset, width, height, format,
                     tile_mode, width, height);
    fixture.uniform(0.25f);
    fixture.enqueue(producer, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    const std::array<std::byte, 4> rgba{
        std::byte{64}, std::byte{0}, std::byte{64}, std::byte{255}};
    const std::array<std::byte, 1> red{std::byte{64}};
    const auto produced = expected_block_linear(
        width, height, bytes_per_texel, 0, block_depth_log2,
        volume ? std::span<const std::byte>(red)
               : std::span<const std::byte>(rgba),
        sentinel, storage_size);
    CHECK(std::equal(produced.begin(), produced.end(),
                     fixture.backend->bytes.begin() + target_offset));
    if (volume) {
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + target_offset,
            fixture.backend->bytes.begin() + target_offset + 512,
            [](std::byte value) { return value == std::byte{64}; }));
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + target_offset + 512,
            fixture.backend->bytes.begin() + target_offset + 8192,
            [](std::byte value) { return value == sentinel; }));
        CHECK(fixture.backend->bytes[target_offset + 8192] == std::byte{64});
        CHECK(fixture.backend->bytes[target_offset + 8703] == std::byte{64});

        auto clear = snapshot();
        clear.kind = RenderKind::Clear;
        clear.original.method = 0x674;
        clear.original.argument = 0xfu << 2;
        clear.registers[0x674] = clear.original.argument;
        clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.75f);
        configure_target(clear, target_offset, width, height, format,
                         tile_mode, width, height);
        fixture.enqueue(clear, 2, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        const std::array<std::byte, 1> full_red{std::byte{191}};
        const auto full_cleared = expected_block_linear(
            width, height, 1, 0, block_depth_log2, full_red, sentinel,
            storage_size);
        CHECK(std::equal(full_cleared.begin(), full_cleared.end(),
                         fixture.backend->bytes.begin() + target_offset));

        clear.original.argument = 1u << 2;
        clear.registers[0x674] = clear.original.argument;
        clear.registers[0x43e] = 1u << 8;
        clear.registers[0x381] = 32u | (64u << 16);
        clear.registers[0x382] = 0u | (16u << 16);
        clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.5f);
        fixture.enqueue(clear, 3, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        auto masked = full_cleared;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 32; x < width; ++x) {
                masked[gob_offset(x, y, 0, block_depth_log2, width)] =
                    std::byte{128};
            }
        }
        CHECK(std::equal(masked.begin(), masked.end(),
                         fixture.backend->bytes.begin() + target_offset));
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + target_offset + 512,
            fixture.backend->bytes.begin() + target_offset + 8192,
            [](std::byte value) { return value == sentinel; }));
    }

    const auto before = m::texture_cache_statistics(*fixture.services.textures);
    auto consumer = snapshot();
    configure_target(consumer, sample_output_offset, 64, 16, 0xd5,
                     0x1000, 16, 16);
    configure_texture_consumer(
        fixture, consumer, volume ? "tex_3d" : "tex_2d",
        volume ? platform::TextureFactDimension::d3
               : platform::TextureFactDimension::d2,
        volume ? 2 : 1, target_offset, width, height, 1,
        volume ? 0x1d : 0x08,
        volume ? std::array<std::uint32_t, 4>{2, 0, 0, 7}
               : std::array<std::uint32_t, 4>{2, 3, 4, 5},
        0, block_depth_log2);
    fixture.enqueue(consumer, volume ? 4 : 2, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    const auto after = m::texture_cache_statistics(*fixture.services.textures);
    CHECK(after.uploads == before.uploads + 1);
    CHECK(after.hits > before.hits);
    const std::array<std::byte, 4> sampled = volume
        ? std::array<std::byte, 4>{std::byte{128}, std::byte{0},
                                   std::byte{0}, std::byte{255}}
        : rgba;
    for (std::size_t pixel = 0; pixel < 16 * 16; ++pixel) {
        CHECK(std::equal(sampled.begin(), sampled.end(),
                         fixture.backend->bytes.begin() +
                             sample_output_offset + pixel * 4));
    }
}

void persistent_render_to_texture_sampling() {
    render_to_real_tic(false);
    render_to_real_tic(true);
}

void persistent_gpu_authoritative_texture_sampling(bool incompatible_shape) {
    Fixture fixture;
    fixture.uniform(0.25f);
    constexpr BackingRange source{{1, 1, 1}, target_offset, 1024};
    constexpr std::byte sentinel{0x6d};
    std::fill_n(fixture.backend->bytes.begin() + source.offset,
                source.size, sentinel);

    auto producer = snapshot();
    configure_target(producer, target_offset, 16, 16, 0xd5, 0, 16, 16);
    const auto produced = fixture.enqueue(
        producer, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->flush(SubmissionBoundary::SubmissionEnd)));
    // Wait only for GPU feedback. Poll/drain would publish a submission
    // boundary and resolve the source before the sampling path can inspect it.
    CHECK(ok(fixture.backend->wait_gpu(produced.completion)));
    CHECK(produced.completion.snapshot().phase == SubmissionPhase::GpuComplete);
    const auto before_dirty = take(fixture.memory->inspect(source));
    CHECK(!before_dirty.empty());
    CHECK(std::ranges::all_of(before_dirty,
        [](const DirtyRange& range) { return range.gpu_dirty; }));
    CHECK(std::all_of(
        fixture.backend->bytes.begin() + source.offset,
        fixture.backend->bytes.begin() + source.offset + source.size,
        [](std::byte value) { return value == sentinel; }));

    auto consumer = snapshot();
    configure_target(consumer, sample_output_offset, 64, 16, 0xd5,
                     0x1000, 16, 16);
    configure_texture_consumer(
        fixture, consumer, "tex_2d", platform::TextureFactDimension::d2,
        1, target_offset, incompatible_shape ? 8 : 16, 16, 1,
        0x08, {2, 3, 4, 5}, 0, 0);
    const auto before = m::texture_cache_statistics(*fixture.services.textures);
    const auto writes_before = fixture.backend->writes;
    const auto consumed = fixture.enqueue(
        consumer, 2, std::make_shared<Effects>());
    const auto after = m::texture_cache_statistics(*fixture.services.textures);
    const auto after_dirty = take(fixture.memory->inspect(source));
    CHECK(!after_dirty.empty());
    CHECK(produced.completion.snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(consumed.completion.snapshot().phase == SubmissionPhase::Accepted);
    if (incompatible_shape) {
        // Both shapes cover the same padded block-linear backing extent, but
        // the narrower view requires resolving and uploading the original bytes.
        CHECK(after.readbacks == before.readbacks + 1);
        CHECK(after.uploads == before.uploads + 2);
        CHECK(fixture.backend->writes > writes_before);
        CHECK(std::ranges::none_of(after_dirty,
            [](const DirtyRange& range) { return range.gpu_dirty; }));
        CHECK(fixture.backend->bytes[target_offset] == std::byte{64});
    } else {
        // Only the separate output attachment uploads. The sampled source
        // retains GPU authority and its CPU backing remains untouched.
        CHECK(after.readbacks == before.readbacks);
        CHECK(after.uploads == before.uploads + 1);
        CHECK(after.hits > before.hits);
        CHECK(fixture.backend->writes == writes_before);
        CHECK(std::ranges::all_of(after_dirty,
            [](const DirtyRange& range) { return range.gpu_dirty; }));
        CHECK(std::all_of(
            fixture.backend->bytes.begin() + source.offset,
            fixture.backend->bytes.begin() + source.offset + source.size,
            [](std::byte value) { return value == sentinel; }));
    }

    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(produced.completion.snapshot().guest_visible_complete());
    CHECK(consumed.completion.snapshot().guest_visible_complete());
    check_pitch_color(*fixture.backend, sample_output_offset,
                      {std::byte{64}, std::byte{0}, std::byte{64},
                       std::byte{255}}, 1);
}

void fill_block_linear_rgba(Memory& memory, std::size_t offset,
                            std::uint32_t width, std::uint32_t height,
                            const std::array<std::byte, 4>& color) {
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t texel = offset +
                gob_offset(x * 4, y, 0, 0, width * 4);
            std::copy(color.begin(), color.end(),
                      memory.bytes.begin() + texel);
        }
    }
}

std::array<std::byte, 4> border_corner_reference(
    const std::array<std::byte, 4>& source,
    const std::array<float, 4>& border) {
    constexpr float coverage = 1.0f / 16.0f;
    std::array<std::byte, 4> result{};
    for (unsigned component = 0; component < result.size(); ++component) {
        const float source_value =
            static_cast<float>(std::to_integer<unsigned>(source[component])) /
            255.0f;
        const float value = source_value * coverage +
                            border[component] * (1.0f - coverage);
        result[component] = std::byte{static_cast<unsigned char>(
            std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f))};
    }
    return result;
}

void renderer_custom_border_case(const std::array<float, 4>& border,
                                 bool custom) {
    constexpr std::size_t source_offset = 32768;
    constexpr std::size_t source_extent = 512;
    const std::array<std::byte, 4> source{
        std::byte{51}, std::byte{102}, std::byte{153}, std::byte{204}};
    Fixture fixture;
    {
        auto transaction = take(fixture.memory->prepare_cpu_write(
            {{1, 1, 1}, source_offset, source_extent}));
        std::fill_n(fixture.backend->bytes.begin() + source_offset,
                    source_extent, std::byte{0});
        fill_block_linear_rgba(
            *fixture.backend, source_offset, 4, 4, source);
        transaction->commit();
    }

    auto draw = snapshot();
    configure_target(draw, target_offset, 64, 16, 0xd5, 0x1000, 16, 16);
    configure_texture_consumer(
        fixture, draw, "custom_border_corner",
        platform::TextureFactDimension::d2, 1, source_offset, 4, 4, 1,
        0x08, {2, 3, 4, 5}, 0, 0);
    configure_border_sampler(fixture, border);
    use_custom_border_fragment(fixture);
    auto planner = observe_custom_border_planner(fixture);

    const auto receipt = fixture.enqueue(
        draw, 1, std::make_shared<Effects>());
    CHECK(receipt.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    CHECK(planner->prepared && planner->uniform_support &&
          planner->host_generated_support && planner->retained_coverage);
    CHECK(planner->support_enabled ==
          std::optional<std::uint32_t>{custom ? 1u : 0u});
    const auto statistics =
        m::submission_statistics(*fixture.services.submissions);
    CHECK(statistics.accepted == 1 && statistics.draws == 1 &&
          statistics.command_buffers == 1 && statistics.render_passes == 1);
    check_pitch_color(
        *fixture.backend, target_offset,
        border_corner_reference(source, border), 2);
}

void composed_custom_border_submission() {
    // A native constant still carries the reflected capability and its
    // disabled host-generated support record through submission validation.
    renderer_custom_border_case({1.0f, 1.0f, 1.0f, 1.0f}, false);
    // The arbitrary border exercises real coverage residency and corner
    // footprint correction through the full renderer path.
    renderer_custom_border_case({0.8f, 0.1f, 0.4f, 0.25f}, true);
}

void check_block_linear_rgba(const Memory& memory, std::size_t offset,
                             std::uint32_t width, std::uint32_t height,
                             const std::array<std::byte, 4>& color) {
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t texel = offset +
                gob_offset(x * 4, y, 0, 0, width * 4);
            CHECK(std::equal(color.begin(), color.end(),
                             memory.bytes.begin() + texel));
        }
    }
}

void persistent_same_subresource_feedback() {
    Fixture fixture;
    fixture.uniform(0.25f);
    auto producer = snapshot();
    configure_target(producer, target_offset, 16, 16, 0xd5, 0, 16, 16);
    fixture.enqueue(producer, 1, std::make_shared<Effects>());

    auto feedback = snapshot();
    configure_target(feedback, target_offset, 16, 16, 0xd5, 0, 16, 16);
    configure_texture_consumer(
        fixture, feedback, "tex_2d_step",
        platform::TextureFactDimension::d2, 1, target_offset, 16, 16, 1,
        0x08, {2, 3, 4, 5}, 0, 0);
    fixture.enqueue(feedback, 2, std::make_shared<Effects>());
    fixture.enqueue(feedback, 3, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    // The first feedback draw observes the producer's 64/0/64 result and the
    // second observes the first draw's 128/32/64 result.
    check_block_linear_rgba(
        *fixture.backend, target_offset, 16, 16,
        {std::byte{192}, std::byte{64}, std::byte{64}, std::byte{255}});
}

void persistent_disjoint_mip_feedback() {
    Fixture fixture;
    constexpr std::size_t mip_one_offset = target_offset + 1024;
    const std::array<std::byte, 4> mip_zero{
        std::byte{17}, std::byte{34}, std::byte{51}, std::byte{255}};
    const std::array<std::byte, 4> mip_one{
        std::byte{32}, std::byte{96}, std::byte{160}, std::byte{255}};
    fill_block_linear_rgba(*fixture.backend, target_offset, 16, 16, mip_zero);
    fill_block_linear_rgba(*fixture.backend, mip_one_offset, 8, 8, mip_one);

    auto draw = snapshot();
    configure_target(draw, target_offset, 16, 16, 0xd5, 0, 16, 16);
    configure_texture_consumer(
        fixture, draw, "tex_2d", platform::TextureFactDimension::d2,
        1, target_offset, 16, 16, 1, 0x08, {2, 3, 4, 5}, 0, 0,
        false, 2);
    use_explicit_texture_view(fixture, 1, std::nullopt);
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    check_block_linear_rgba(
        *fixture.backend, target_offset, 16, 16, mip_one);
    check_block_linear_rgba(
        *fixture.backend, mip_one_offset, 8, 8, mip_one);
}

void persistent_disjoint_layer_feedback() {
    Fixture fixture;
    constexpr std::size_t layer_stride = 1024;
    const std::array<std::byte, 4> layer_zero{
        std::byte{21}, std::byte{42}, std::byte{63}, std::byte{255}};
    const std::array<std::byte, 4> layer_one{
        std::byte{48}, std::byte{112}, std::byte{176}, std::byte{255}};
    fill_block_linear_rgba(*fixture.backend, target_offset, 16, 16, layer_zero);
    fill_block_linear_rgba(
        *fixture.backend, target_offset + layer_stride, 16, 16, layer_one);

    auto draw = snapshot();
    configure_target(draw, target_offset, 16, 16, 0xd5, 0, 16, 16);
    draw.color_targets[0].depth = 2;
    draw.color_targets[0].array_pitch = layer_stride / 4;
    draw.color_targets[0].base_layer = 0;
    draw.registers[0x206] = 2;
    draw.registers[0x207] = layer_stride / 4;
    draw.registers[0x208] = 0;
    configure_texture_consumer(
        fixture, draw, "tex_2da_zero",
        platform::TextureFactDimension::d2, 5, target_offset, 16, 16, 2,
        0x08, {2, 3, 4, 5}, 0, 0, true);
    use_explicit_texture_view(fixture, std::nullopt, 1);
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    check_block_linear_rgba(
        *fixture.backend, target_offset, 16, 16, layer_one);
    check_block_linear_rgba(
        *fixture.backend, target_offset + layer_stride, 16, 16, layer_one);
}

// The sampled resource is a two-plane R8 volume while the attachment is a
// single 3D slice with the same block-depth interleave. Their overlapping
// backing ranges have incompatible native shapes and require a pre-draw copy.
void persistent_volume_slice_feedback() {
    Fixture fixture;
    constexpr std::uint32_t width = 16;
    constexpr std::uint32_t height = 16;
    constexpr std::size_t storage_size = 2048;
    constexpr std::byte sentinel{0x5c};
    constexpr std::array<std::byte, 2> planes{
        std::byte{0x24}, std::byte{0xa4}};
    std::fill_n(fixture.backend->bytes.begin() + target_offset,
                storage_size, sentinel);
    for (std::size_t z = 0; z != planes.size(); ++z) {
        for (std::size_t y = 0; y != height; ++y) {
            for (std::size_t x = 0; x != width; ++x) {
                fixture.backend->bytes[target_offset + volume_gob_offset(
                    x, y, z, 0, 1, width, height)] = planes[z];
            }
        }
    }

    auto draw = snapshot();
    configure_target(draw, target_offset, width, height, 0xf3, 0x10100,
                     width, height);
    configure_texture_consumer(
        fixture, draw, "tex_3d_step", platform::TextureFactDimension::d3,
        2, target_offset, width, height, 2, 0x1d, {2, 0, 0, 7}, 0, 1);
    fixture.enqueue(draw, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    std::vector<std::byte> expected(storage_size, sentinel);
    for (std::size_t z = 0; z != planes.size(); ++z) {
        const std::byte value = z == 0 ? std::byte{0xc4} : planes[1];
        for (std::size_t y = 0; y != height; ++y) {
            for (std::size_t x = 0; x != width; ++x) {
                expected[volume_gob_offset(
                    x, y, z, 0, 1, width, height)] = value;
            }
        }
    }
    CHECK(std::equal(expected.begin(), expected.end(),
                     fixture.backend->bytes.begin() + target_offset));
}

void persistent_virtual_alias_feedback() {
    Fixture fixture;
    constexpr std::uint64_t alias_base = 0x200000;
    CHECK(ok(fixture.memory->register_mapping(
        {{1, 2, 1}, {1, 1, 1},
         {alias_base, fixture.backend->bytes.size(), 0, 0x10000000,
          fixture.backend->bytes.size(), 4096}})));
    fixture.uniform(0.25f);
    auto producer = snapshot();
    configure_target(producer, target_offset, 16, 16, 0xd5, 0, 16, 16);
    fixture.enqueue(producer, 1, std::make_shared<Effects>());

    auto feedback = snapshot();
    configure_target(feedback, target_offset, 16, 16, 0xd5, 0, 16, 16);
    configure_texture_consumer(
        fixture, feedback, "tex_2d_step",
        platform::TextureFactDimension::d2, 1, target_offset, 16, 16, 1,
        0x08, {2, 3, 4, 5}, 0, 0, false, 1, alias_base);
    fixture.enqueue(feedback, 2, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));

    check_block_linear_rgba(
        *fixture.backend, target_offset, 16, 16,
        {std::byte{128}, std::byte{32}, std::byte{64}, std::byte{255}});
}

void persistent_storage_write_feedback_rejected() {
    Fixture fixture;
    const auto before = std::vector<std::byte>(
        fixture.backend->bytes.begin() + target_offset,
        fixture.backend->bytes.begin() + target_offset + 1024);
    auto feedback = snapshot();
    configure_target(feedback, target_offset, 16, 16, 0xd5, 0, 16, 16);
    configure_texture_consumer(
        fixture, feedback, "tex_2d",
        platform::TextureFactDimension::d2, 1, target_offset, 16, 16, 1,
        0x08, {2, 3, 4, 5}, 0, 0);
    use_explicit_texture_view(
        fixture, std::nullopt, std::nullopt, ResourceAccess::ReadWrite);
    auto effects = std::make_shared<Effects>();
    auto rejected = fixture.renderer->enqueue(
        {{1, 1}, std::make_shared<const RenderStop>(feedback),
         fixture.memory, {}, effects});
    const auto* error = std::get_if<RendererError>(&rejected);
    CHECK(error != nullptr);
    if (error) {
        CHECK(error->command().status == CommandStatus::Unsupported);
        CHECK(error->command().detail.find("writable") != std::string_view::npos);
    }
    CHECK(effects->coherent == 0 && effects->prepared == 0 &&
          effects->published == 0);
    CHECK(m::submission_statistics(*fixture.services.submissions).accepted == 0);
    CHECK(std::equal(before.begin(), before.end(),
                     fixture.backend->bytes.begin() + target_offset));
}

void persistent_feedback_budget_rejected() {
    Fixture fixture;
    const auto before = std::vector<std::byte>(
        fixture.backend->bytes.begin() + target_offset,
        fixture.backend->bytes.begin() + target_offset + 1024);
    auto feedback = snapshot();
    configure_target(feedback, target_offset, 16, 16, 0xd5, 0, 16, 16);
    configure_texture_consumer(
        fixture, feedback, "tex_2d",
        platform::TextureFactDimension::d2, 1, target_offset, 16, 16, 1,
        0x08, {2, 3, 4, 5}, 0, 0);
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1, 4, 64}, fixture.services));
    auto effects = std::make_shared<Effects>();
    auto rejected = fixture.renderer->enqueue(
        {{1, 1}, std::make_shared<const RenderStop>(feedback),
         fixture.memory, {}, effects});
    // Native preparation runs on the queue's encoding worker: the budget
    // failure surfaces either from enqueue or, sticky, from the next drain.
    std::optional<RendererError> failed;
    if (const auto* error = std::get_if<RendererError>(&rejected)) failed = *error;
    else if (const auto drained = fixture.renderer->drain(deadline());
             const auto* error = std::get_if<RendererError>(&drained)) failed = *error;
    CHECK(failed.has_value());
    if (failed) CHECK(failed->command().status == CommandStatus::BudgetExceeded);
    CHECK(effects->coherent == 0 && effects->published == 0);
    CHECK(std::equal(before.begin(), before.end(),
                     fixture.backend->bytes.begin() + target_offset));
}

// Exercise guest TIC/TSC resolution, typed fact checks, L4 residency, argument
// tables and GPU sampling together. Each array layer, cube face and depth slice
// has distinct data so accidentally binding a plain 2D view cannot pass.
void persistent_texture_targets() {
    using platform::TextureFactDimension;
    using platform::TextureSampledScalar;
    using platform::TextureComponentFormat;
    struct Target {
        unsigned tic_target;
        TextureFactDimension dimension;
        bool array;
        const char* entry;
        unsigned layers, depth, sampled_layer, sampled_z;
        TextureSampledScalar scalar = TextureSampledScalar::f32;
    };
    const Target targets[] = {
        {0, TextureFactDimension::d1, false, "tex_1d", 1, 1, 0, 0},
        {4, TextureFactDimension::d1, true, "tex_1da", 2, 1, 1, 0},
        {1, TextureFactDimension::d2, false, "tex_2d", 1, 1, 0, 0},
        {5, TextureFactDimension::d2, true, "tex_2da", 2, 1, 1, 0},
        {2, TextureFactDimension::d3, false, "tex_3d", 1, 2, 0, 1},
        {3, TextureFactDimension::cube, false, "tex_cube", 6, 1, 5, 0},
        {8, TextureFactDimension::cube, true, "tex_cubea", 12, 1, 11, 0},
        {7, TextureFactDimension::d2, false, "tex_rect", 1, 1, 0, 0},
        {1, TextureFactDimension::d2, false, "tex_uint", 1, 1, 0, 0, TextureSampledScalar::u32},
        {1, TextureFactDimension::d2, false, "tex_sint", 1, 1, 0, 0, TextureSampledScalar::s32},
    };
    constexpr std::size_t tic_offset = 24576, tsc_offset = tic_offset + 64;
    constexpr std::size_t image_offset = 32768;
    for (const auto& target : targets) {
        Fixture fixture;
        auto stop = snapshot();
        const bool rectangle = target.tic_target == 7;
        const bool integer = target.scalar != TextureSampledScalar::f32;
        const bool cube = target.dimension == TextureFactDimension::cube;
        const unsigned height = target.dimension == TextureFactDimension::d1 ? 1 : 2;
        const auto word = [&](std::size_t offset, std::uint32_t value) {
            for (unsigned byte = 0; byte != 4; ++byte)
                fixture.backend->bytes[offset + byte] = std::byte((value >> (byte * 8)) & 255);
        };
        auto transaction = take(fixture.memory->prepare_cpu_write(
            {{1, 1, 1}, uniform_offset, fixture.backend->bytes.size() - uniform_offset}));
        word(uniform_offset + 32, 0);
        const unsigned component = target.scalar == TextureSampledScalar::u32 ? tic_component_uint :
            target.scalar == TextureSampledScalar::s32 ? tic_component_sint : tic_component_unorm;
        std::uint32_t format_word = 8;
        for (unsigned channel = 0; channel != 4; ++channel) {
            format_word |= component << (7 + channel * 3);
            format_word |= (channel + 2u) << (19 + channel * 3);
        }
        word(tic_offset, format_word);
        word(tic_offset + 4, gpu_base + image_offset);
        word(tic_offset + 8, 3u << 21);
        word(tic_offset + 16, 1u | (target.tic_target << 23));
        const unsigned depth_field = target.tic_target == 8 ? target.layers / 6 :
            target.array ? target.layers : target.depth;
        word(tic_offset + 20, (height - 1) | ((depth_field - 1) << 16) |
            (rectangle ? 0 : 1u << 31));
        word(tsc_offset, 2u | (2u << 3) | (2u << 6));
        word(tsc_offset + 4, 1u | (1u << 4) | (1u << 6));
        for (unsigned layer = 0; layer != target.layers; ++layer)
            for (unsigned z = 0; z != target.depth; ++z)
                for (unsigned y = 0; y != height; ++y)
                    for (unsigned x = 0; x != 2; ++x) {
                        const std::array<unsigned, 4> color{
                            32 + layer * 16 + z * 8, 40 + (cube ? 0 : x * 32),
                            60 + (cube ? 0 : y * 32), 255};
                        // A two-pixel row occupies the first eight bytes of
                        // each GOB row; the next row starts sixteen bytes on.
                        const auto offset = image_offset + layer * 512 + z * 512 + y * 16 + x * 4;
                        for (unsigned channel = 0; channel != 4; ++channel)
                            fixture.backend->bytes[offset + channel] = std::byte(
                                (color[channel] + (target.scalar == TextureSampledScalar::s32 ? 128 : 0)) & 255);
                    }
        transaction->commit();
        stop.registers[0x982] = 4;
        stop.registers[0x55e] = gpu_base + tic_offset;
        stop.registers[0x558] = gpu_base + tsc_offset;
        stop.constant_buffers[4][4] = {gpu_base + uniform_offset, 36, true};
        auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
        auto fragment = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.fragment);
        fragment->entry_point = target.entry;
        fragment->constant_buffer_windows.clear();
        fragment->constant_buffer_support_index.reset();
        fragment->texture_handle_cbuf = 4;
        fragment->textures = {{32, 0, rectangle || integer ? gm::TextureBinding::no_sampler : std::uint8_t{0}}};
        platform::DerivedTextureFact fact{4, 32, !rectangle, false};
        fact.dimension = target.dimension;
        fact.array = target.array;
        fact.scalar = target.scalar;
        fact.component_format = target.scalar == TextureSampledScalar::u32 ? TextureComponentFormat::unsigned_integer :
            target.scalar == TextureSampledScalar::s32 ? TextureComponentFormat::signed_integer : TextureComponentFormat::floating;
        fragment->texture_facts = {fact};
        stages->admitted.stages.fragment = fragment;
        stages->admitted.fragment_key.entry_point = target.entry;
        for (unsigned sequence = 1; sequence != 3; ++sequence) {
            auto receipt = fixture.enqueue(stop, sequence, std::make_shared<Effects>());
            CHECK(ok(fixture.renderer->drain(deadline())));
            CHECK(receipt.completion.snapshot().guest_visible_complete());
            const std::array<unsigned, 4> expected{
                32 + target.sampled_layer * 16 + target.sampled_z * 8,
                cube ? 40u : 72u, cube || height == 1 ? 60u : 92u, 255};
            for (unsigned pixel = 0; pixel != 16 * 16; ++pixel)
                for (unsigned channel = 0; channel != 4; ++channel) {
                    const auto actual = std::to_integer<unsigned>(
                        fixture.backend->bytes[target_offset + pixel * 4 + channel]);
                    if (actual != expected[channel]) {
                        std::fprintf(stderr, "%s pixel %u channel %u: %u != %u\n",
                            target.entry, pixel, channel, actual, expected[channel]);
                        CHECK(actual == expected[channel]);
                        break;
                    }
                }
        }
        CHECK(m::texture_cache_statistics(*fixture.services.textures).hits >= 1);
    }
}

RenderStop depth_clear_snapshot(std::uint64_t address, std::uint32_t aspects = 1) {
    auto clear = snapshot();
    clear.kind = RenderKind::Clear;
    clear.original.method = 0x674;
    clear.original.argument = aspects;
    clear.registers[0x674] = aspects;
    clear.registers[0x364] = std::bit_cast<std::uint32_t>(0.5f);
    clear.registers[0x3f8] = static_cast<std::uint32_t>(address >> 32);
    clear.registers[0x3f9] = static_cast<std::uint32_t>(address);
    clear.registers[0x3fa] = 0x0a;
    clear.registers[0x3fb] = 0;
    clear.registers[0x3fc] = 0x1000;
    clear.registers[0x48a] = clear.registers[0x48b] = 16;
    clear.registers[0x48c] = 0x10001;
    clear.registers[0x54e] = 1;
    clear.depth = {address, 0x0a, 0, 0x1000, 16, 16, true};
    return clear;
}

// Record host uploads independently of pipeline compilation and shader behavior.
struct RecordingBuffers final : BufferCache {
    std::shared_ptr<BufferCache> delegate;
    std::vector<std::vector<std::byte>> uploads;
    explicit RecordingBuffers(std::shared_ptr<BufferCache> original) : delegate(std::move(original)) {}
    ResidencyDomain domain() const noexcept override { return delegate->domain(); }
    RendererResult<ResidentBuffer> acquire(const BufferRequest& request) override {
        return delegate->acquire(request);
    }
    RendererResult<ResidentBuffer> upload(std::span<const std::byte> bytes,
                                         std::uint64_t alignment) override {
        uploads.emplace_back(bytes.begin(), bytes.end());
        return delegate->upload(bytes, alignment);
    }
    RendererStatus invalidate(BackingRange bytes) override { return delegate->invalidate(bytes); }
    RendererStatus resolve(BufferKey key, ResolveReason reason) override { return delegate->resolve(key, reason); }
    RendererStatus trim(std::uint64_t budget) override { return delegate->trim(budget); }
};

void depth_bias_support_activation() {
    Fixture fixture;
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    auto fragment = std::make_shared<platform::CompiledGraphicsStage>(*stages->admitted.stages.fragment);
    fragment->emulates_depth_bias = true;
    stages->admitted.stages.fragment = fragment;
    auto buffers = std::make_shared<RecordingBuffers>(fixture.services.buffers);
    fixture.services.buffers = buffers;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64}, fixture.services));
    std::size_t ordinary_uploads = 0;
    for (unsigned variant = 0; variant != 6; ++variant) {
        auto stop = depth_clear_snapshot(gpu_base + sample_output_offset);
        stop.kind = RenderKind::Draw;
        stop.registers[0x372] = variant != 0;
        stop.registers[0x55b] = std::bit_cast<std::uint32_t>(32.0f);
        stop.registers[0x56f] = std::bit_cast<std::uint32_t>(variant == 1 ? 0.0f : 8.0f);
        stop.registers[0x61f] = std::bit_cast<std::uint32_t>(variant == 4 ? 0.01f : 0.0f);
        if (variant == 2) {
            stop.depth.enabled = false;
            stop.registers[0x54e] = 0;
        }
        // Disabled -> slope-only -> no depth -> constant -> combined cap -> disabled.
        if (variant == 5) stop.registers[0x372] = 0;
        buffers->uploads.clear();
        fixture.enqueue(stop, variant + 1, std::make_shared<Effects>());
        CHECK(ok(fixture.renderer->drain(deadline())));
        const bool active = variant == 3 || variant == 4;
        CHECK(fixture.pipelines->keys.back().shader_depth_bias == active);
        if (variant == 0) ordinary_uploads = buffers->uploads.size();
        CHECK(buffers->uploads.size() == ordinary_uploads + (active ? 1 : 0));
        if (active) {
            CHECK(buffers->uploads.back().size() == sizeof(float) * 4);
            std::array<float, 4> support{};
            std::memcpy(support.data(), buffers->uploads.back().data(), sizeof(support));
            CHECK(support[0] == 8.0f / 16777216.0f);
            CHECK(support[1] == (variant == 4 ? 32.0f : 0.0f));
            CHECK(support[2] == (variant == 4 ? 0.01f : 0.0f));
            CHECK(support[3] == 1.0f);
        }
    }
}

void attachment_less_draws_execute_in_order() {
    Fixture fixture(true);
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    {
        auto write = take(fixture.memory->prepare_cpu_write({{1, 1, 1}, index_offset, sizeof(indices)}));
        std::memcpy(fixture.backend->bytes.data() + index_offset, indices.data(), sizeof(indices));
        write->commit();
    }
    {
        auto write = take(fixture.memory->prepare_cpu_write({{1, 1, 1}, empty_vertex_offset, 36}));
        std::memcpy(fixture.backend->bytes.data() + empty_vertex_offset,
                    fixture.backend->bytes.data() + vertex_offset, 36);
        write->commit();
    }
    auto stages = std::dynamic_pointer_cast<Stages>(fixture.services.stages);
    const auto attachment_stages = stages->admitted;
    std::vector<unsigned> order;
    std::vector<SubmissionReceipt> receipts;
    std::vector<std::shared_ptr<Effects>> effects;
    const auto enqueue = [&](RenderStop stop, unsigned sequence) {
        auto actions = std::make_shared<Effects>();
        actions->publication_order = &order;
        actions->publication_id = sequence;
        effects.push_back(actions);
        receipts.push_back(fixture.enqueue(std::move(stop), sequence, actions));
    };

    enqueue(snapshot(true), 1);
    use_outputless_fragment(fixture);
    auto empty_192 = attachment_less_snapshot(192, true);
    const auto empty_plan = take(plan_render_targets(empty_192));
    CHECK(empty_plan.area == (RenderArea{0, 0, 192, 192}));
    CHECK(empty_plan.samples == 1);
    CHECK(!empty_plan.depth.has_value() && !empty_plan.stencil.has_value());
    enqueue(std::move(empty_192), 2);
    enqueue(attachment_less_snapshot(192, true), 3);
    enqueue(attachment_less_snapshot(96, true), 4);
    stages->admitted = attachment_stages;
    enqueue(snapshot(true), 5);

    CHECK(order.empty());
    // Empty passes still reserve vertex reads until GPU completion. Indices
    // use owned snapshots and do not reserve their original guest bytes.
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_write(
        {{1, 1, 1}, empty_vertex_offset, 36})));
    CHECK(std::ranges::all_of(receipts, [](const auto& receipt) {
        return receipt.completion.snapshot().phase == SubmissionPhase::Accepted;
    }));
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK((order == std::vector<unsigned>{1, 2, 3, 4, 5}));
    CHECK(std::ranges::all_of(receipts, [](const auto& receipt) {
        return receipt.completion.snapshot().guest_visible_complete();
    }));
    CHECK(std::ranges::all_of(effects, [](const auto& actions) {
        return actions->coherent == 1 && actions->prepared == 1 &&
               actions->published == 1;
    }));

    CHECK(fixture.pipelines->keys.size() == 5);
    CHECK(fixture.pipelines->depth_stencil_keys.size() == 5);
    for (std::size_t index = 1; index != 4; ++index) {
        const auto& key = fixture.pipelines->keys[index];
        CHECK(key.fragment.entry_point == "fragment_empty");
        CHECK(std::ranges::none_of(key.colors, [](const auto& color) {
            return color.format.has_value();
        }));
        CHECK(!key.depth_format.has_value());
        CHECK(!key.stencil_format.has_value());
        const auto& depth_stencil = fixture.pipelines->depth_stencil_keys[index];
        CHECK(!depth_stencil.depth_test);
        CHECK(!depth_stencil.depth_write);
        CHECK(!depth_stencil.stencil_test);
    }
    const auto statistics = m::submission_statistics(*fixture.services.submissions);
    CHECK(statistics.accepted == 5 && statistics.draws == 5);
    CHECK(statistics.command_buffers == 1 && statistics.render_passes == 4);
    fixture.pixels(255, 255, 64);
    {
        auto write = take(fixture.memory->prepare_cpu_write({{1, 1, 1}, empty_vertex_offset, 36}));
        write->commit();
    }

    use_outputless_fragment(fixture);
    auto unmapped = attachment_less_snapshot(192, true);
    unmapped.depth.address = 0x900000;
    unmapped.registers[0x3f9] = 0x900000;
    auto rejected_effects = std::make_shared<Effects>();
    auto rejected = fixture.renderer->enqueue({{1, 6},
        std::make_shared<const RenderStop>(std::move(unmapped)), fixture.memory,
        {}, rejected_effects});
    const auto* error = std::get_if<RendererError>(&rejected);
    CHECK(error != nullptr);
    if (error) {
        CHECK(error->command().status == CommandStatus::Stopped);
        CHECK(error->command().detail.find("unmapped") != std::string_view::npos);
    }
    CHECK(rejected_effects->coherent == 0 && rejected_effects->prepared == 0 &&
          rejected_effects->published == 0);
    CHECK(m::submission_statistics(*fixture.services.submissions).accepted == 5);

    // A draw with no guest-backed bindings must still execute, without asking
    // the memory registry to create an empty reservation.
    Fixture resource_free;
    use_outputless_fragment(resource_free);
    auto empty = attachment_less_snapshot(192);
    empty.draw->indexed = false;
    auto receipt = resource_free.enqueue(empty, 1, std::make_shared<Effects>());
    CHECK(ok(resource_free.renderer->drain(deadline())));
    CHECK(receipt.completion.snapshot().guest_visible_complete());
    const auto empty_statistics = m::submission_statistics(*resource_free.services.submissions);
    CHECK(empty_statistics.draws == 1 && empty_statistics.render_passes == 1);
    CHECK(empty_statistics.command_buffers == 1);
}

void unbound_depth_clear_after_unmap() {
    Fixture fixture;
    auto real = depth_clear_snapshot(gpu_base + target_offset);
    auto first = fixture.enqueue(real, 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(first.completion.snapshot().guest_visible_complete());
    const std::array removed{MappingId{1, 1, 1}};
    const std::array discarded{BackingId{1, 1, 1}};
    CHECK(ok(fixture.memory->invalidate_mappings(removed, discarded)));
    CHECK(fixture.memory->address_space().mappings().empty());

    const auto bytes = fixture.backend->bytes;
    const auto reads = fixture.backend->reads;
    const auto writes = fixture.backend->writes;
    const auto statistics = m::submission_statistics(*fixture.services.submissions);
    for (std::uint32_t aspects = 1; aspects <= 3; ++aspects) {
        auto effects = std::make_shared<Effects>();
        auto clear = depth_clear_snapshot(0, aspects);
        // This formerly valid color slot now names the removed mapping.
        CHECK(clear.color_targets[0].address == gpu_base + target_offset);
        auto receipt = fixture.enqueue(clear, aspects + 1, effects);
        CHECK(effects->published == 0);
        CHECK(ok(fixture.renderer->drain(deadline())));
        CHECK(receipt.completion.snapshot().guest_visible_complete());
        CHECK(effects->coherent == 1 && effects->prepared == 1 && effects->published == 1);
    }
    CHECK(fixture.backend->reads == reads && fixture.backend->writes == writes);
    CHECK(fixture.backend->bytes == bytes);
    const auto after = m::submission_statistics(*fixture.services.submissions);
    CHECK(after.accepted == statistics.accepted + 3);
    CHECK(after.command_buffers == statistics.command_buffers);
    CHECK(after.render_passes == statistics.render_passes);

    auto rejected_effects = std::make_shared<Effects>();
    auto rejected = fixture.renderer->enqueue({{1, 5}, std::make_shared<const RenderStop>(real),
        fixture.memory, {}, rejected_effects});
    CHECK(std::holds_alternative<RendererError>(rejected));
    CHECK(rejected_effects->published == 0);
    CHECK(m::submission_statistics(*fixture.services.submissions).accepted == after.accepted);

    // The same GPU VA becomes real storage again with a new backing identity.
    CHECK(ok(fixture.memory->register_backing({{1, 2, 1}, fixture.backend->bytes.size(), fixture.backend})));
    CHECK(ok(fixture.memory->register_mapping({{1, 2, 1}, {1, 2, 1},
        {gpu_base, fixture.backend->bytes.size(), 0, 0x20000000, fixture.backend->bytes.size(), 4096}})));
    real.registers[0x364] = std::bit_cast<std::uint32_t>(0.25f);
    auto rebound = fixture.enqueue(real, 5, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(rebound.completion.snapshot().guest_visible_complete());
    for (std::size_t y = 0; y < 16; ++y) {
        for (std::size_t x = 0; x < 16; ++x) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, fixture.backend->bytes.data() + target_offset +
                gob_offset(x * 4, y, 0, 0, 64), sizeof(actual));
            CHECK(actual == std::bit_cast<std::uint32_t>(0.25f));
        }
    }
    CHECK(fixture.backend->writes > writes);
    CHECK(m::submission_statistics(*fixture.services.submissions).command_buffers > after.command_buffers);
}

void unbound_depth_clear_preserves_order() {
    Fixture fixture;
    std::vector<unsigned> order;
    order.reserve(2);
    auto first_effects = std::make_shared<Effects>();
    auto second_effects = std::make_shared<Effects>();
    first_effects->publication_order = second_effects->publication_order = &order;
    first_effects->publication_id = 1;
    second_effects->publication_id = 2;
    auto first = fixture.enqueue(depth_clear_snapshot(gpu_base + target_offset), 1, first_effects);
    auto second = fixture.enqueue(depth_clear_snapshot(0), 2, second_effects);
    CHECK(order.empty());
    CHECK(!second.completion.snapshot().guest_visible_complete());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(first.completion.snapshot().guest_visible_complete());
    CHECK(second.completion.snapshot().guest_visible_complete());
    CHECK((order == std::vector<unsigned>{1, 2}));
    const auto statistics = m::submission_statistics(*fixture.services.submissions);
    CHECK(statistics.accepted == 2 && statistics.command_buffers == 1 && statistics.render_passes == 1);
}

void unbound_depth_clear_honors_failed_dependency() {
    Fixture fixture;
    SubmissionCompletion dependency({2, 1}, {}, std::make_shared<Effects>());
    auto effects = std::make_shared<Effects>();
    const auto reads = fixture.backend->reads;
    const auto writes = fixture.backend->writes;
    auto receipt = take(fixture.renderer->enqueue({{1, 1},
        std::make_shared<const RenderStop>(depth_clear_snapshot(0)), fixture.memory,
        {dependency.token()}, effects}));
    CHECK(std::holds_alternative<RendererError>(fixture.renderer->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(receipt.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(effects->published == 0);
    CHECK(ok(dependency.reject({CommandStatus::Stopped, "synthetic null-clear dependency failure"})));
    CHECK(std::holds_alternative<RendererError>(fixture.renderer->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(receipt.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(std::holds_alternative<RendererError>(fixture.renderer->poll()));
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(effects->published == 0);
    CHECK(fixture.backend->reads == reads && fixture.backend->writes == writes);
    CHECK(m::submission_statistics(*fixture.services.submissions).command_buffers == 0);
    CHECK(m::submission_statistics(*fixture.services.submissions).render_passes == 0);
}

// Observe ownership without extending the lifetime that the real queue must
// provide. All admission, encoding and completion still run on the native queue.
class SnapshotObserver final : public m::SubmissionQueue {
public:
    explicit SnapshotObserver(std::shared_ptr<m::SubmissionQueue> queue)
        : queue_(std::move(queue)) {}
    std::vector<std::weak_ptr<const RenderStop>> snapshots;
    ResidencyDomain domain() const noexcept override { return queue_->domain(); }
    RendererResult<SubmissionReceipt> append(m::PreparedRender prepared) override {
        snapshots.push_back(prepared.request.snapshot);
        return queue_->append(std::move(prepared));
    }
    RendererStatus flush(SubmissionBoundary boundary) override { return queue_->flush(boundary); }
    RendererStatus poll() override { return queue_->poll(); }
    RendererStatus drain(std::uint64_t limit) override { return queue_->drain(limit); }
private:
    std::shared_ptr<m::SubmissionQueue> queue_;
};

void guarded_gpu_admission_retries_after_prefix_retirement() {
    Fixture fixture;
    auto observer = std::make_shared<SnapshotObserver>(fixture.services.submissions);
    fixture.services.submissions = observer;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64, true}, fixture.services));
    auto first = fixture.enqueue(snapshot(), 1, std::make_shared<Effects>());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(first.completion.snapshot().guest_visible_complete());
    CHECK(observer->snapshots.size() == 1);
    auto guard = take(fixture.memory->defer_host_store({{1, 1, 1}, target_offset, 4}));
    const auto request = [&] {
        return fixture.renderer->enqueue({{1, 2}, std::make_shared<const RenderStop>(snapshot()),
            fixture.memory, {}, std::make_shared<Effects>()});
    };
    const auto blocked = request();
    const auto* transient = std::get_if<RendererError>(&blocked);
    CHECK(transient != nullptr);
    if (transient) CHECK(m::commit_hold_blocked(RendererStatus(*transient)));
    // The warm attachment needs no CPU read: exclusion must be honored at
    // GPU admission even though the completed prefix left no pending receipt.
    CHECK(observer->snapshots.size() == 2);
    guard.reset();
    auto accepted = take(request());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(accepted.completion.snapshot().guest_visible_complete());
}

void owned_snapshots_preserve_lifetime_and_caller_isolation() {
    Fixture fixture;
    auto observer = std::make_shared<SnapshotObserver>(fixture.services.submissions);
    fixture.services.submissions = observer;
    fixture.renderer = take(m::create_renderer(
        {16 * 1024 * 1024, 1024 * 1024, 4, 64}, fixture.services));
    std::vector<unsigned> order;
    const auto effects = [&](unsigned id) {
        auto value = std::make_shared<Effects>();
        value->publication_order = &order;
        value->publication_id = id;
        return value;
    };
    // The renderer reads a stop only while preparing it: queued packets carry
    // one shared placeholder, never the caller's object or a copy of it.
    auto owned = std::make_unique<const RenderStop>(snapshot());
    const auto* identity = owned.get();
    const auto first = take(fixture.renderer->enqueue_owned(
        {{1, 1}, {}, fixture.memory, {}, effects(1)}, std::move(owned)));
    CHECK(!owned);
    CHECK(observer->snapshots.size() == 1);
    const auto placeholder = observer->snapshots[0].lock();
    CHECK(placeholder && placeholder.get() != identity);
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Accepted);

    // Ordinary enqueue still accepts a shared mutable caller object by making
    // its own immutable snapshot, including draw fields and constant bindings.
    auto mutable_stop = std::make_shared<RenderStop>(snapshot());
    const auto second = take(fixture.renderer->enqueue(
        {{1, 2}, mutable_stop, fixture.memory, {}, effects(2)}));
    CHECK(observer->snapshots.size() == 2);
    CHECK(observer->snapshots[1].lock() == placeholder);
    CHECK(mutable_stop.use_count() == 1);
    mutable_stop.reset();

    // The same ownership seam must preserve non-draw work and its ordering.
    auto clear = snapshot();
    clear.kind = RenderKind::Clear;
    clear.original.method = 0x674;
    clear.original.argument = 0xfu << 2;
    clear.registers[0x674] = 0xfu << 2;
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.5f);
    clear.registers[0x363] = std::bit_cast<std::uint32_t>(1.0f);
    auto owned_clear = std::make_unique<const RenderStop>(std::move(clear));
    const auto* clear_identity = owned_clear.get();
    const auto third = take(fixture.renderer->enqueue_owned(
        {{1, 3}, {}, fixture.memory, {}, effects(3)}, std::move(owned_clear)));
    CHECK(!owned_clear);
    CHECK(observer->snapshots.size() == 3);
    CHECK(observer->snapshots[2].lock() == placeholder && placeholder.get() != clear_identity);
    CHECK(order.empty());
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(first.completion.snapshot().guest_visible_complete());
    CHECK(second.completion.snapshot().guest_visible_complete());
    CHECK(third.completion.snapshot().guest_visible_complete());
    CHECK((order == std::vector<unsigned>{1, 2, 3}));
    fixture.pixels(128, 128, 0);
}

void clears_preserve_order() {
    Fixture fixture;
    auto effects = std::make_shared<Effects>();
    fixture.enqueue(snapshot(), 1, effects);
    auto clear = snapshot();
    clear.kind = RenderKind::Clear;
    clear.original.method = 0x674;
    clear.original.argument = 0xfu << 2;
    clear.registers[0x674] = 0xfu << 2;
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.5f);
    clear.registers[0x363] = std::bit_cast<std::uint32_t>(1.0f);
    auto cleared = fixture.enqueue(clear, 2, effects);
    CHECK(ok(fixture.renderer->drain(deadline())));
    CHECK(cleared.completion.snapshot().guest_visible_complete());
    fixture.pixels(128, 128, 0);
    // A masked/scissored clear is an operation, preserving the other channels.
    clear.original.argument = 1u << 2;
    clear.registers[0x674] = 1u << 2;
    clear.registers[0x43e] = 1u << 8;
    clear.registers[0x381] = 8u << 16;
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.25f);
    fixture.enqueue(clear, 3, effects);
    CHECK(ok(fixture.renderer->drain(deadline())));
    fixture.pixels(64, 128, 0);
}
} // namespace

int main() {
    // These tests cover the optional attachment feedback snapshot path.
    setenv("ARMIGO_FEEDBACK_SNAPSHOTS", "1", 1);
    @autoreleasepool {
        vertex_reads_high_address_storage_descriptor();
        overlapping_storage_reads_share_one_allocation();
        fragment_storage_write_orders_cpu_and_gpu_consumers();
        storage_descriptor_bounds_and_zero_size_guard_gpu_access();
        malformed_storage_descriptors_fail_renderer_admission();
        batched_uniform_versions();
        buffer_growth_split();
        overlapping_streams_and_normalized_accesses();
        unfinished_snapshot_split();
        renderer_limits_and_staging_admission();
        divided_instance_fetches();
        rejected_request_and_domain();
        guarded_preparation_failure_retries_after_publication();
        persistent_render_target_layout_draws();
        persistent_array_stride_draw();
        persistent_pitch_clear_padding();
        persistent_argument_buffer_many_resources();
        persistent_argument_buffer_each_member();
        persistent_direct_resource_high_indices();
        persistent_argument_buffer_typed_null();
        persistent_argument_buffer_feedback_substitution();
        persistent_argument_buffer_feedback_substitution(true);
        persistent_argument_buffer_completion_lifetime_and_reuse();
        persistent_render_to_texture_sampling();
        persistent_gpu_authoritative_texture_sampling(false);
        persistent_gpu_authoritative_texture_sampling(true);
        composed_custom_border_submission();
        persistent_same_subresource_feedback();
        persistent_disjoint_mip_feedback();
        persistent_disjoint_layer_feedback();
        persistent_volume_slice_feedback();
        persistent_virtual_alias_feedback();
        persistent_storage_write_feedback_rejected();
        persistent_feedback_budget_rejected();
        owned_snapshots_preserve_lifetime_and_caller_isolation();
        guarded_gpu_admission_retries_after_prefix_retirement();
        clears_preserve_order();
        depth_bias_support_activation();
        attachment_less_draws_execute_in_order();
        unbound_depth_clear_after_unmap();
        unbound_depth_clear_preserves_order();
        unbound_depth_clear_honors_failed_dependency();
        persistent_texture_targets();
    }
    std::printf("Metal renderer integration: %u failures\n", failures);
    return failures ? 1 : 0;
}
