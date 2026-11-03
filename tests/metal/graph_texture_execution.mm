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
#import <Metal/Metal.h>

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Executes emitted control-flow graph sampling on the real Metal 4 GPU: a
// runtime constant-buffer word selects one arm of a fragment diamond whose
// taken arm samples a 2x2 texture of distinct texel colours at a texel centre
// under one uniform predicate per draw, so the derivative question does not
// arise, and the join block samples the same texture at another texel centre.
// Both runtime outcomes run to completion and every pixel is compared with the
// CPU evaluator driven by the same nearest-texel sampler callback. This proves
// guarded placement, component extraction, and reflection on real GPU
// execution; it does not establish implicit-LOD equivalence under divergence.

namespace {

namespace gx = armigo::shader::graphics;
namespace gm = armigo::shader::graphics_msl;
using Stage = armigo::shader_input::Stage;

constexpr NSUInteger kWidth = 8;
constexpr NSUInteger kHeight = 8;
constexpr NSUInteger kPixelBytes = 4 * sizeof(float);
constexpr NSUInteger kGuardBytes = 64;
constexpr std::byte kGuard{0xa7};

void put_header(gx::GraphSpec& spec, Stage stage) {
    const auto header = armigo::shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned byte = 0; byte < 4; ++byte) {
        spec.sph[byte] = std::byte(header >> (8 * byte));
    }
}

// Straight-line synthetic vertex graph: three attribute words and a literal
// clip W, exported as position.
std::optional<gx::Program> make_vertex_program() {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_header(spec, Stage::Vertex);
    spec.sph[0x18] = std::byte{0x07};  // generic input vector 0, lanes XYZ
    spec.sph[0x35] = std::byte{0xf0};  // position outputs declared
    spec.nodes = {
        {8, gx::AttributeLoad{32}},
        {16, gx::AttributeLoad{33}},
        {24, gx::AttributeLoad{34}},
        {32, gx::Literal{0x40000000u}},  // 2.0f clip W
    };
    spec.exports = {
        {gx::OutputKind::Position, 0, gx::ScalarId{0}},
        {gx::OutputKind::Position, 1, gx::ScalarId{1}},
        {gx::OutputKind::Position, 2, gx::ScalarId{2}},
        {gx::OutputKind::Position, 3, gx::ScalarId{3}},
    };
    spec.blocks = {{0, 4, gx::Return{}}};
    auto result = gx::make_graph(spec);
    auto* program = std::get_if<gx::Program>(&result);
    if (!program) std::fputs("Vertex graph construction failed.\n", stderr);
    if (!program) return std::nullopt;
    return std::move(*program);
}

// Fragment diamond with sampling: a runtime constant-buffer word drives
// WordNotEqual, the taken arm samples handle offset 32 at the (0.25, 0.25)
// texel centre and extracts red and green, the untaken arm substitutes
// literals, the join block samples handle offset 36 at the (0.75, 0.25) texel
// centre and extracts blue and alpha, and scalar phis carry the arm lanes into
// the exports. One selector word per draw keeps the predicate uniform.
std::optional<gx::Program> make_fragment_program() {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_header(spec, Stage::Fragment);
    spec.sph[0x48] = std::byte{0x0f};  // color target 0 RGBA declared
    spec.nodes = {
        {8, gx::ConstantLoad{0, 0}},
        {16, gx::Literal{0}},
        {24, gx::WordNotEqual{gx::ScalarId{0}, gx::ScalarId{1}}},
        {32, gx::Literal{0x3e800000u}},  // 0.25f u coordinate
        {40, gx::Literal{0x3e800000u}},  // 0.25f v coordinate
        {48, gx::TextureSample{gx::ScalarId{3}, gx::ScalarId{4}, 32, true}},
        {56, gx::SampleComponent{gx::SampleId{5}, 0}},
        {64, gx::SampleComponent{gx::SampleId{5}, 1}},
        {72, gx::Literal{0x3f800000u}},  // untaken arm red 1.0
        {80, gx::Literal{0x3f000000u}},  // untaken arm green 0.5
        {88, gx::ScalarPhi{{
                 {gx::BlockId{1}, gx::ScalarId{6}},
                 {gx::BlockId{2}, gx::ScalarId{8}}}}},
        {96, gx::ScalarPhi{{
                 {gx::BlockId{1}, gx::ScalarId{7}},
                 {gx::BlockId{2}, gx::ScalarId{9}}}}},
        {104, gx::Literal{0x3f400000u}},  // 0.75f u coordinate
        {112, gx::Literal{0x3e800000u}},  // 0.25f v coordinate
        {120, gx::TextureSample{gx::ScalarId{12}, gx::ScalarId{13}, 36, false}},
        {128, gx::SampleComponent{gx::SampleId{14}, 2}},
        {136, gx::SampleComponent{gx::SampleId{14}, 3}},
    };
    spec.exports = {
        {gx::OutputKind::Color, 0, gx::ScalarId{10}},
        {gx::OutputKind::Color, 1, gx::ScalarId{11}},
        {gx::OutputKind::Color, 2, gx::ScalarId{15}},
        {gx::OutputKind::Color, 3, gx::ScalarId{16}},
    };
    spec.blocks = {
        {0, 5, gx::BranchIf{gx::PredicateId{2}, gx::BlockId{1}, gx::BlockId{2}}},
        {5, 8, gx::Jump{gx::BlockId{3}}},
        {8, 10, gx::Jump{gx::BlockId{3}}},
        {10, 17, gx::Return{}},
    };
    auto result = gx::make_graph(spec);
    auto* program = std::get_if<gx::Program>(&result);
    if (!program) std::fputs("Fragment graph construction failed.\n", stderr);
    if (!program) return std::nullopt;
    return std::move(*program);
}

bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}
// True when `statement` appears inside the body of the first guard whose
// opening text is `guard`, tracked by brace depth, so a statement hoisted
// after the body does not satisfy the check. Unit assertion only.
bool guard_contains(const std::string& text, const std::string& guard,
                    const std::string& statement) {
    const auto open = text.find(guard);
    if (open == std::string::npos) return false;
    const auto body_open = text.find('{', open);
    if (body_open == std::string::npos) return false;
    for (std::size_t at = body_open + 1, depth = 1; at < text.size(); ++at) {
        if (text[at] == '{') {
            ++depth;
        } else if (text[at] == '}') {
            if (--depth == 0) return false;
        } else if (text.compare(at, statement.size(), statement) == 0) {
            return true;
        }
    }
    return false;
}

struct Context {
    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4CommandQueue> queue = nil;
};

id<MTLLibrary> library(Context& context, const std::string& source) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> result = [context.device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:options error:&error];
    if (!result) std::fprintf(stderr, "Metal source compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

id<MTLRenderPipelineState> pipeline(Context& context,
                                    id<MTLLibrary> vertex_library,
                                    id<MTLLibrary> fragment_library,
                                    MTLVertexDescriptor* vertex_descriptor) {
    MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = vertex_library;
    vertex.name = @"armigo_vertex_main";
    MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = fragment_library;
    fragment.name = @"armigo_fragment_main";
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = fragment;
    descriptor.vertexDescriptor = vertex_descriptor;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    id<MTLRenderPipelineState> result = [context.compiler
        newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!result) std::fprintf(stderr, "Metal render pipeline compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

struct Readback {
    id<MTLTexture> target = nil;
    id<MTLBuffer> buffer = nil;
};

Readback make_readback(id<MTLDevice> device) {
    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:kWidth height:kHeight mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    Readback result;
    result.target = [device newTextureWithDescriptor:descriptor];
    result.buffer = [device newBufferWithLength:kGuardBytes * 2 + kWidth * kHeight * kPixelBytes
                                        options:MTLResourceStorageModeShared];
    if (result.buffer) std::memset(result.buffer.contents, std::to_integer<int>(kGuard),
                                   result.buffer.length);
    return result;
}

bool encode_readback(id<MTL4CommandBuffer> commands, Readback readback) {
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) return false;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
               visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:readback.target sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:readback.buffer destinationOffset:kGuardBytes
    destinationBytesPerRow:kWidth * kPixelBytes destinationBytesPerImage:0];
    [copy endEncoding];
    return true;
}

bool submit(Context& context, id<MTL4CommandBuffer> commands, NSArray* retained) {
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
    [context.queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed,
            dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
        std::fputs("Metal graph texture execution timed out.\n", stderr);
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "Metal graph texture execution failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

bool guards_intact(const Readback& readback) {
    const auto* bytes = static_cast<const std::byte*>(readback.buffer.contents);
    for (NSUInteger i = 0; i < kGuardBytes; ++i) {
        if (bytes[i] != kGuard || bytes[readback.buffer.length - 1 - i] != kGuard) return false;
    }
    return true;
}

const float* pixels(const Readback& readback) {
    return reinterpret_cast<const float*>(
        static_cast<const std::byte*>(readback.buffer.contents) + kGuardBytes);
}

bool graph_texture_draw(Context& context) {
    auto vertex_program = make_vertex_program();
    auto fragment_program = make_fragment_program();
    if (!vertex_program || !fragment_program) return false;
    gm::EmitOptions vertex_options;
    vertex_options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    vertex_options.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3}};
    gm::EmitOptions fragment_options;
    auto vertex_result = gm::emit(*vertex_program, vertex_options);
    auto fragment_result = gm::emit(*fragment_program, fragment_options);
    const auto* emitted_vertex = std::get_if<gm::EmittedStage>(&vertex_result);
    const auto* emitted_fragment = std::get_if<gm::EmittedStage>(&fragment_result);
    if (!emitted_vertex || !emitted_fragment) {
        std::fputs("Graph texture MSL emission failed.\n", stderr);
        return false;
    }
    // Reflection matches straight-line sampling: one branch buffer and two
    // texture/sampler slots, one per unique handle offset.
    if (emitted_fragment->constant_buffers.size() != 1 ||
        emitted_fragment->constant_buffers[0].guest_binding != 0 ||
        emitted_fragment->constant_buffers[0].metal_buffer_index != 0 ||
        emitted_fragment->constant_buffers[0].minimum_size < 4 ||
        emitted_fragment->textures.size() != 2 ||
        emitted_fragment->textures[0].handle_byte_offset != 32 ||
        emitted_fragment->textures[0].texture_index != 0 ||
        emitted_fragment->textures[0].sampler_index != 0 ||
        emitted_fragment->textures[1].handle_byte_offset != 36 ||
        emitted_fragment->textures[1].texture_index != 1 ||
        emitted_fragment->textures[1].sampler_index != 1) {
        std::fputs("Graph texture reflection does not match the program.\n", stderr);
        return false;
    }
    // The arm sample and its component extraction stay inside the arm guard,
    // the join sample stays inside the join guard, and no eager select
    // replaces the guards.
    const std::string& source = emitted_fragment->source;
    if (!contains(source, "[[maybe_unused]] float4 s5 = float4(0.0f);") ||
        !contains(source, "[[maybe_unused]] float4 s14 = float4(0.0f);") ||
        !guard_contains(source, "if (a1) {",
            "s5 = tex0.sample(samp0, float2(as_type<float>(v3), as_type<float>(v4)));") ||
        !guard_contains(source, "if (a1) {", "v6 = as_type<uint>(s5[0]);") ||
        !guard_contains(source, "if (a1) {", "v7 = as_type<uint>(s5[1]);") ||
        !guard_contains(source, "if (a2) {", "v10 = v8;") ||
        !guard_contains(source, "if (a3) {",
            "s14 = tex1.sample(samp1, float2(as_type<float>(v12), as_type<float>(v13)));") ||
        !guard_contains(source, "if (a3) {", "v15 = as_type<uint>(s14[2]);") ||
        !guard_contains(source, "if (a3) {", "v16 = as_type<uint>(s14[3]);") ||
        contains(source, "select(")) {
        std::fputs("Emitted graph texture source lacks guards or extraction.\n", stderr);
        return false;
    }

    id<MTLLibrary> vertex_library = library(context, emitted_vertex->source);
    id<MTLLibrary> fragment_library = library(context, emitted_fragment->source);
    MTLVertexDescriptor* vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    // Vertex fetch reads argument-table index 1; the fragment selector word is
    // argument-table index 0, matching the emitted reflection.
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    vertex_descriptor.attributes[0].bufferIndex = 1;
    vertex_descriptor.layouts[1].stride = 3 * sizeof(float);
    id<MTLRenderPipelineState> metal_pipeline = vertex_library && fragment_library
        ? pipeline(context, vertex_library, fragment_library, vertex_descriptor) : nil;
    if (!metal_pipeline) return false;

    constexpr std::array<float, 9> vertices{
        -2, -2, 0,
         6, -2, 0,
        -2,  6, 0};
    id<MTLBuffer> vertex_buffer = [context.device newBufferWithBytes:vertices.data()
        length:sizeof(vertices) options:MTLResourceStorageModeShared];
    id<MTLBuffer> control_buffer = [context.device
        newBufferWithLength:sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    // Distinct texels and lanes make coordinate and extraction mistakes
    // observable. Row-major rows below match the Metal upload order.
    const std::array<std::array<float, 4>, 4> texels{{
        {0.125f, 0.25f, 0.375f, 0.5f}, {0.625f, 0.75f, 0.875f, 1.0f},
        {0.5f, 0.375f, 0.25f, 0.125f}, {1.0f, 0.875f, 0.75f, 0.625f}}};
    auto* texture_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:NO];
    texture_descriptor.storageMode = MTLStorageModeShared;
    texture_descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [context.device newTextureWithDescriptor:texture_descriptor];
    std::array<float, 16> texture_data{};
    for (unsigned texel = 0; texel < 4; ++texel) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            texture_data[texel * 4 + lane] = texels[texel][lane];
        }
    }
    [texture replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0
                 withBytes:texture_data.data() bytesPerRow:2 * kPixelBytes];
    MTLSamplerDescriptor* sampler_descriptor = [MTLSamplerDescriptor new];
    // Nearest filtering at texel centres returns exact texel colours, so the
    // GPU result must equal the CPU sampler callback bit for bit.
    sampler_descriptor.minFilter = MTLSamplerMinMagFilterNearest;
    sampler_descriptor.magFilter = MTLSamplerMinMagFilterNearest;
    sampler_descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> sampler = [context.device newSamplerStateWithDescriptor:sampler_descriptor];
    auto readback = make_readback(context.device);
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    table_descriptor.maxTextureBindCount = 2;
    table_descriptor.maxSamplerStateBindCount = 2;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [context.device
        newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 6;
    id<MTLResidencySet> residency = [context.device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!vertex_buffer || !control_buffer || !texture || !sampler || !readback.target ||
        !readback.buffer || !table || !residency) return false;
    [table setAddress:control_buffer.gpuAddress atIndex:0];
    [table setAddress:vertex_buffer.gpuAddress atIndex:1];
    [table setTexture:texture.gpuResourceID atIndex:0];
    [table setTexture:texture.gpuResourceID atIndex:1];
    [table setSamplerState:sampler.gpuResourceID atIndex:0];
    [table setSamplerState:sampler.gpuResourceID atIndex:1];
    [residency addAllocation:vertex_buffer];
    [residency addAllocation:control_buffer];
    [residency addAllocation:texture];
    [residency addAllocation:readback.target];
    [residency addAllocation:readback.buffer];
    [residency commit];
    NSArray* retained = @[context.device, context.compiler, context.queue, vertex_library,
                          fragment_library, metal_pipeline, vertex_buffer, control_buffer,
                          texture, sampler, readback.target, readback.buffer, table, residency];

    // CPU reference: the same nearest-texel lookup the GPU sampler applies.
    gx::Inputs cpu_inputs;
    std::vector<std::uint32_t> sampled_handles;
    cpu_inputs.sample_2d_implicit_lod =
        [&](const gx::TextureRequest& request, std::array<std::uint32_t, 4>& result) {
        if (request.handle_byte_offset != 32 && request.handle_byte_offset != 36) return false;
        sampled_handles.push_back(request.handle_byte_offset);
        const float u = std::bit_cast<float>(request.x_bits);
        const float v = std::bit_cast<float>(request.y_bits);
        const unsigned column = std::clamp(unsigned(u * 2.0f), 0u, 1u);
        const unsigned row = std::clamp(unsigned(v * 2.0f), 0u, 1u);
        for (unsigned lane = 0; lane < 4; ++lane) {
            result[lane] = std::bit_cast<std::uint32_t>(texels[row * 2 + column][lane]);
        }
        return true;
    };

    // The selector word is uniform across each draw, so the sampled arm is
    // quad-uniform and the implicit-LOD derivative question does not arise.
    // The taken arm reads red 0.125 and green 0.25; the other arm supplies
    // red 1 and green 0.5. The join reads blue 0.875 and alpha 1 in both cases.
    const std::array<std::pair<std::uint32_t, std::array<float, 4>>, 2> outcomes{
        {{0x00000001u, {0.125f, 0.25f, 0.875f, 1.0f}},
         {0x00000000u, {1.0f, 0.5f, 0.875f, 1.0f}}}};
    for (const auto& [selector, expected] : outcomes) {
        const std::array<std::uint32_t, 1> words{selector};
        cpu_inputs.constant_buffers[0] = words;
        sampled_handles.clear();
        const auto cpu_result = gx::evaluate(*fragment_program, cpu_inputs);
        const auto* evaluation = std::get_if<gx::Evaluation>(&cpu_result);
        if (!evaluation || evaluation->outputs.size() != 4) {
            std::fputs("CPU graph texture evaluation failed.\n", stderr);
            return false;
        }
        const std::vector<std::uint32_t> expected_handles = selector != 0
            ? std::vector<std::uint32_t>{32, 36} : std::vector<std::uint32_t>{36};
        if (sampled_handles != expected_handles) {
            std::fputs("CPU graph texture sample order disagrees with the selected path.\n", stderr);
            return false;
        }
        for (unsigned lane = 0; lane < 4; ++lane) {
            if (evaluation->outputs[lane].bits != std::bit_cast<std::uint32_t>(expected[lane])) {
                std::fputs("CPU graph texture evaluation disagrees with the texel model.\n",
                           stderr);
                return false;
            }
        }
        std::memcpy(control_buffer.contents, words.data(), sizeof(words));
        id<MTL4CommandAllocator> allocator = [context.device newCommandAllocator];
        id<MTL4CommandBuffer> commands = [context.device newCommandBuffer];
        if (!allocator || !commands) return false;
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = readback.target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(-99.0, -99.0, -99.0, -99.0);
        id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
        if (!render) return false;
        [render setRenderPipelineState:metal_pipeline];
        [render setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [render endEncoding];
        if (!encode_readback(commands, readback)) return false;
        NSArray* draw_retained = [retained arrayByAddingObjectsFromArray:
            @[allocator, commands]];
        if (!submit(context, commands, draw_retained) || !guards_intact(readback)) {
            return false;
        }
        const float* actual = pixels(readback);
        for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
            for (unsigned lane = 0; lane < 4; ++lane) {
                const float gpu = actual[pixel * 4 + lane];
                const float reference =
                    std::bit_cast<float>(evaluation->outputs[lane].bits);
                if (std::bit_cast<std::uint32_t>(gpu) != evaluation->outputs[lane].bits) {
                    std::fprintf(stderr,
                        "Graph texture mismatch at pixel %lu lane %u (selector 0x%08x): "
                        "expected CPU %.8g, got GPU %.8g.\n",
                        pixel, lane, selector, reference, gpu);
                    return false;
                }
            }
        }
    }
    std::printf("Verified both uniform sampling outcomes over %lu pixels each.\n",
                kWidth * kHeight);
    return true;
}

}  // namespace

int main() {
    @autoreleasepool {
        Context context;
        context.device = MTLCreateSystemDefaultDevice();
        if (!context.device || ![context.device supportsFamily:MTLGPUFamilyMetal4] ||
            !context.device.hasUnifiedMemory) {
            std::fputs("Graph texture execution tests require unified-memory Metal 4.\n",
                       stderr);
            return 1;
        }
        NSError* error = nil;
        context.compiler = [context.device
            newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        context.queue = [context.device newMTL4CommandQueue];
        if (!context.compiler || !context.queue) {
            std::fprintf(stderr, "Metal 4 setup failed: %s\n",
                         error.localizedDescription.UTF8String);
            return 1;
        }
        if (!graph_texture_draw(context)) return 1;
        return 0;
    }
}
