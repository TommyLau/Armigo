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

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Executes an emitted control-flow graph on the real Metal 4 GPU: a runtime
// constant-buffer word selects one arm of a fragment diamond whose red channel
// is a predecessor-selected phi value and whose blue channel composes a
// controlled nearest multiply by an authored 2.0 literal with a typed
// Saturate node in each arm (the runtime blue input is loaded only inside its
// branch arm), and the joined predicate phi drives a second real branch whose
// arms select the green channel through a second-level phi in the return
// block. Both outcomes run to completion and must produce their distinct
// expected colors. This proves the composed Armigo CPU/MSL contract on real
// GPU execution, not Maxwell hardware zero-sign parity.

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

// Fragment diamond chained into a second branch: a constant-buffer word
// drives WordNotEqual, each arm loads its own runtime blue input from the
// buffer, multiplies it by an authored 2.0 literal with a controlled nearest
// multiply, and saturates the product with a typed Saturate node; the join
// selects red and blue with scalar phis and joins the arm predicates into a
// phi that drives a second branch, and the second branch's arms select green
// with a phi in the return block. Alpha is defined before the branch.
std::optional<gx::Program> make_fragment_program() {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_header(spec, Stage::Fragment);
    spec.sph[0x48] = std::byte{0x0f};  // color target 0 RGBA declared
    spec.nodes = {
        {8, gx::ConstantLoad{0, 0}},
        {16, gx::Literal{0}},
        {24, gx::WordNotEqual{gx::ScalarId{0}, gx::ScalarId{1}}},
        {32, gx::Literal{0x3f800000u}},  // alpha on every path
        {40, gx::Literal{0x3f800000u}},  // taken arm red
        {48, gx::ConstantLoad{0, 4}},    // taken arm blue input: branch-only load
        {56, gx::Literal{0x40000000u}},  // authored 2.0f multiplier
        {64, gx::FloatArithmetic{gx::Arithmetic::MultiplyNearest,
                                 gx::ScalarId{5}, gx::ScalarId{6}, std::nullopt, false}},
        {72, gx::Saturate{gx::ScalarId{7}}},
        {80, gx::PredicateLiteral{true}},
        {88, gx::Literal{0x00000000u}},  // untaken arm red
        {96, gx::ConstantLoad{0, 8}},    // untaken arm blue input: branch-only load
        {104, gx::Literal{0x40000000u}},  // authored 2.0f multiplier
        {112, gx::FloatArithmetic{gx::Arithmetic::MultiplyNearest,
                                  gx::ScalarId{11}, gx::ScalarId{12}, std::nullopt, false}},
        {120, gx::Saturate{gx::ScalarId{13}}},
        {128, gx::ScalarPhi{{
                  {gx::BlockId{1}, gx::ScalarId{4}},
                  {gx::BlockId{2}, gx::ScalarId{10}}}}},
        {136, gx::ScalarPhi{{
                  {gx::BlockId{1}, gx::ScalarId{8}},
                  {gx::BlockId{2}, gx::ScalarId{14}}}}},
        {144, gx::PredicatePhi{{
                  {gx::BlockId{1}, gx::PredicateId{9}},
                  {gx::BlockId{2}, gx::PredicateId{2}}}}},
        {152, gx::Literal{0x3f800000u}},  // second branch true arm green
        {160, gx::Literal{0x3e800000u}},  // second branch false arm green
        {168, gx::ScalarPhi{{
                  {gx::BlockId{4}, gx::ScalarId{18}},
                  {gx::BlockId{5}, gx::ScalarId{19}}}}},
    };
    spec.exports = {
        {gx::OutputKind::Color, 0, gx::ScalarId{15}},
        {gx::OutputKind::Color, 1, gx::ScalarId{20}},
        {gx::OutputKind::Color, 2, gx::ScalarId{16}},
        {gx::OutputKind::Color, 3, gx::ScalarId{3}},
    };
    spec.blocks = {
        {0, 4, gx::BranchIf{gx::PredicateId{2}, gx::BlockId{1}, gx::BlockId{2}}},
        {4, 10, gx::Jump{gx::BlockId{3}}},
        {10, 15, gx::Jump{gx::BlockId{3}}},
        {15, 18, gx::BranchIf{gx::PredicateId{17}, gx::BlockId{4}, gx::BlockId{5}}},
        {18, 19, gx::Jump{gx::BlockId{6}}},
        {19, 20, gx::Jump{gx::BlockId{6}}},
        {20, 21, gx::Return{}},
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
bool ordered(const std::string& text, const std::string& first,
             const std::string& second) {
    const auto first_at = text.find(first);
    const auto second_at = text.find(second);
    return first_at != std::string::npos && second_at != std::string::npos &&
           first_at < second_at;
}
// True when `statement` appears inside the body of the first guard whose
// opening text is `guard`. The scan tracks braces from that guard's opening
// brace and stops at the one that closes the body, so a statement hoisted
// after the body — including one hidden inside a later guard with the same
// text — does not satisfy the check, while branches nested inside the body
// still count. Generated graph MSL contains no string literals, so brace
// text is unambiguous. Unit assertion only; no source parser.
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
        std::fputs("Metal graph execution timed out.\n", stderr);
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "Metal graph execution failed: %s\n",
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

bool graph_draw(Context& context) {
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
        std::fputs("Graph MSL emission failed.\n", stderr);
        return false;
    }
    // Reflection: the branch words are the only resource, bound at buffer 0
    // with room for all three loaded words; no texture resources and no eager
    // whole-path loads exist.
    if (emitted_fragment->constant_buffers.size() != 1 ||
        emitted_fragment->constant_buffers[0].guest_binding != 0 ||
        emitted_fragment->constant_buffers[0].metal_buffer_index != 0 ||
        emitted_fragment->constant_buffers[0].minimum_size < 12 ||
        !emitted_fragment->textures.empty()) {
        std::fputs("Graph reflection does not match the branch buffer.\n", stderr);
        return false;
    }
    // The condition reads the runtime buffer, both first-level arms are real
    // guards with their own load, controlled multiply, and saturation step
    // plus predicate-phi edge assignments, the joined predicate drives a
    // second real guard, and each second-level phi channel is assigned on its
    // taken edge before the join region runs. Guard checks are scoped: the
    // statement must sit inside that guard's own brace body, so a statement
    // hoisted out of the guard or deferred to a later same-text guard fails.
    // The scoped matcher itself must reject the synthetic shape of an empty
    // first guard followed by the statement and a duplicate guard.
    const std::string synthetic_guards =
        "if (a2) {}\nv11 = cb0[2];\nif (a2) {\n    v11 = cb0[2];\n}\n";
    if (guard_contains(synthetic_guards, "if (a2) {", "v11 = cb0[2];")) {
        std::fputs("Scoped guard matcher accepted an outside-first-guard statement.\n",
                   stderr);
        return false;
    }
    const std::string& source = emitted_fragment->source;
    if (!contains(source, "v0 = cb0[0];") || !contains(source, "p2 = v0 != v1;") ||
        !contains(source, "if (p2) {") || contains(source, "select(") ||
        !guard_contains(source, "if (a1) {", "v5 = cb0[1];") ||
        !guard_contains(source, "if (a2) {", "v11 = cb0[2];") ||
        !guard_contains(source, "if (a1) {", "v7 = armigo_mul_rn_ftz(v5, v6);") ||
        !ordered(source, "v7 = armigo_mul_rn_ftz(v5, v6);",
                 "v8 = armigo_saturate_float32(v7);") ||
        !guard_contains(source, "if (a2) {", "v13 = armigo_mul_rn_ftz(v11, v12);") ||
        !ordered(source, "v13 = armigo_mul_rn_ftz(v11, v12);",
                 "v14 = armigo_saturate_float32(v13);") ||
        !guard_contains(source, "if (a1) {", "p17 = p9;") ||
        !guard_contains(source, "if (a2) {", "p17 = p2;") ||
        !guard_contains(source, "if (a3) {", "if (p17) {") ||
        !guard_contains(source, "if (a4) {", "v20 = v18;") ||
        !guard_contains(source, "if (a5) {", "v20 = v19;") ||
        !contains(source, "out.color.w = as_type<float>(v3);")) {
        std::fputs("Emitted graph source lacks guards or edge phi assignments.\n", stderr);
        return false;
    }

    id<MTLLibrary> vertex_library = library(context, emitted_vertex->source);
    id<MTLLibrary> fragment_library = library(context, emitted_fragment->source);
    MTLVertexDescriptor* vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    // Vertex fetch reads argument-table index 1; the fragment branch word is
    // argument-table index 0, matching the emitted reflection.
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
        newBufferWithLength:3 * sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    auto readback = make_readback(context.device);
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [context.device
        newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 4;
    id<MTLResidencySet> residency = [context.device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!vertex_buffer || !control_buffer || !readback.target || !readback.buffer ||
        !table || !residency) return false;
    [table setAddress:control_buffer.gpuAddress atIndex:0];
    [table setAddress:vertex_buffer.gpuAddress atIndex:1];
    [residency addAllocation:vertex_buffer];
    [residency addAllocation:control_buffer];
    [residency addAllocation:readback.target];
    [residency addAllocation:readback.buffer];
    [residency commit];
    NSArray* retained = @[context.device, context.compiler, context.queue, vertex_library,
                          fragment_library, metal_pipeline, vertex_buffer, control_buffer,
                          readback.target, readback.buffer, table, residency];

    // Distinct runtime blue inputs expose a wrong phi arm, load address, or
    // skipped saturation: the taken arm multiplies 2.0 by 2.0 and saturates
    // 4.0 to 1.0, while the untaken arm multiplies 2.0 by -0.5 and saturates
    // the negative product to positive zero. Source assertions above check
    // that each load, multiply, and saturation stays inside its block guard;
    // readback alone cannot distinguish harmless eager reads of valid words.
    // Forcing the second branch true reports green 1.0 instead of 0.25.
    constexpr std::uint32_t kBlueInputTaken = 0x40000000u;    // 2.0f
    constexpr std::uint32_t kBlueInputUntaken = 0xbf000000u;  // -0.5f
    const std::array<float, 4> taken{1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 4> untaken{0.0f, 0.25f, 0.0f, 1.0f};
    const std::array<std::pair<std::array<std::uint32_t, 3>,
                               const std::array<float, 4>&>, 2> outcomes{
        {{{0x000000a5u, kBlueInputTaken, kBlueInputUntaken}, taken},
         {{0x00000000u, kBlueInputTaken, kBlueInputUntaken}, untaken}}};
    for (const auto& [words, expected] : outcomes) {
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
                if (!std::isfinite(actual[pixel * 4 + lane]) ||
                    std::abs(actual[pixel * 4 + lane] - expected[lane]) > 2.0e-6f) {
                    std::fprintf(stderr,
                        "Graph draw mismatch at pixel %lu lane %u (control 0x%08x): "
                        "expected %.8g, got %.8g.\n",
                        pixel, lane, words[0], expected[lane],
                        actual[pixel * 4 + lane]);
                    return false;
                }
            }
        }
    }
    std::printf("Verified both runtime branch outcomes over %lu pixels each.\n",
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
            std::fputs("Graph execution tests require unified-memory Metal 4.\n", stderr);
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
        if (!graph_draw(context)) return 1;
        return 0;
    }
}
