// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared real-GPU harness for registered pure operations. A case is a
// single-block fragment graph whose runtime inputs are constant-buffer-0
// words; its four Raw32 color exports are evaluated by the CPU reference and
// by the emitted MSL on the Metal 4 GPU, and the raw bits must match. Each
// operation family adds its own executable of cases built on this header.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test::operation_harness {

namespace gx = armigo::shader::graphics;
namespace gm = armigo::shader::graphics_msl;
using Stage = armigo::shader_input::Stage;

struct OperationCase {
    const char* name = "";
    // Fragment nodes; runtime inputs are gx::ConstantLoad{0, 4 * word}.
    std::vector<gx::Node> nodes;
    // Node indices of the four exported color lanes (Raw32 values).
    std::array<std::uint32_t, 4> outputs{};
    // Constant buffer 0 words.
    std::vector<std::uint32_t> constants;
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4CommandQueue> queue = nil;
};

inline constexpr NSUInteger kWidth = 4;
inline constexpr NSUInteger kHeight = 4;
inline constexpr NSUInteger kPixelBytes = 4 * sizeof(std::uint32_t);

inline void put_header(gx::GraphSpec& spec, Stage stage) {
    const auto header = armigo::shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned byte = 0; byte < 4; ++byte) spec.sph[byte] = std::byte(header >> (8 * byte));
}

inline std::optional<Context> make_context() {
    Context context;
    context.device = MTLCreateSystemDefaultDevice();
    if (!context.device) {
        std::fputs("No Metal device.\n", stderr);
        return std::nullopt;
    }
    NSError* error = nil;
    context.compiler = [context.device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                          error:&error];
    context.queue = [context.device newMTL4CommandQueue];
    if (!context.compiler || !context.queue) {
        std::fputs("Metal 4 compiler or queue unavailable.\n", stderr);
        return std::nullopt;
    }
    return context;
}

inline std::optional<gx::Program> vertex_program() {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_header(spec, Stage::Vertex);
    spec.sph[0x18] = std::byte{0x07};
    spec.sph[0x35] = std::byte{0xf0};
    spec.nodes = {{8, gx::AttributeLoad{32}}, {16, gx::AttributeLoad{33}},
                  {24, gx::AttributeLoad{34}}, {32, gx::Literal{0x3f800000u}}};
    spec.exports = {{gx::OutputKind::Position, 0, gx::ScalarId{0}},
                    {gx::OutputKind::Position, 1, gx::ScalarId{1}},
                    {gx::OutputKind::Position, 2, gx::ScalarId{2}},
                    {gx::OutputKind::Position, 3, gx::ScalarId{3}}};
    spec.blocks = {{0, 4, gx::Return{}}};
    auto result = gx::make_graph(spec);
    auto* program = std::get_if<gx::Program>(&result);
    if (!program) return std::nullopt;
    return std::move(*program);
}

inline std::optional<gx::Program> fragment_program(const OperationCase& item) {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_header(spec, Stage::Fragment);
    spec.sph[0x48] = std::byte{0x0f};
    spec.nodes = item.nodes;
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Color, lane, gx::ScalarId{item.outputs[lane]}});
    }
    spec.blocks = {{0, static_cast<std::uint32_t>(item.nodes.size()), gx::Return{}}};
    auto result = gx::make_graph(spec);
    if (const auto* error = std::get_if<gx::Error>(&result)) {
        std::fprintf(stderr, "%s: graph rejected: %s (pc %u detail %u)\n", item.name,
                     error->message(), error->pc, error->detail);
        return std::nullopt;
    }
    return std::move(std::get<gx::Program>(result));
}

inline id<MTLLibrary> library(Context& context, const std::string& source) {
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

inline id<MTLRenderPipelineState> pipeline(Context& context, id<MTLLibrary> vertex_library,
                                           id<MTLLibrary> fragment_library) {
    MTLVertexDescriptor* vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    vertex_descriptor.attributes[0].bufferIndex = 18;
    vertex_descriptor.layouts[18].stride = 3 * sizeof(float);
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
    // The emitted color is the exported word reinterpreted as float; a
    // 32-bit float target stores finite, zero and infinite words bit for bit.
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    id<MTLRenderPipelineState> result = [context.compiler
        newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!result) std::fprintf(stderr, "Metal render pipeline compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

// Runs one case on the CPU and the GPU and compares the four exported words
// bit for bit. A float render target does not promise NaN payloads, so cases
// keep NaN results inside the graph and check them through a later
// operation, or leave payload checks to the CPU tests.
inline bool run_case(Context& context, const OperationCase& item) {
    auto vertex = vertex_program();
    auto fragment = fragment_program(item);
    if (!vertex || !fragment) return false;

    gx::Inputs inputs;
    inputs.constant_buffers[0] = item.constants;
    const auto evaluated = gx::evaluate(*fragment, inputs);
    const auto* reference = std::get_if<gx::Evaluation>(&evaluated);
    if (!reference || reference->outputs.size() != 4) {
        std::fprintf(stderr, "%s: CPU reference failed.\n", item.name);
        return false;
    }
    std::array<std::uint32_t, 4> expected{};
    for (const auto& output : reference->outputs) expected[output.component] = output.bits;

    gm::EmitOptions vertex_options;
    vertex_options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    vertex_options.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3}};
    auto vertex_result = gm::emit(*vertex, vertex_options);
    auto fragment_result = gm::emit(*fragment, gm::EmitOptions{});
    const auto* emitted_vertex = std::get_if<gm::EmittedStage>(&vertex_result);
    const auto* emitted_fragment = std::get_if<gm::EmittedStage>(&fragment_result);
    if (!emitted_vertex || !emitted_fragment) {
        std::fprintf(stderr, "%s: MSL emission failed.\n", item.name);
        return false;
    }
    id<MTLLibrary> vertex_library = library(context, emitted_vertex->source);
    id<MTLLibrary> fragment_library = library(context, emitted_fragment->source);
    id<MTLRenderPipelineState> state = vertex_library && fragment_library
        ? pipeline(context, vertex_library, fragment_library) : nil;
    if (!state) {
        std::fprintf(stderr, "%s: pipeline creation failed.\n%s\n", item.name,
                     emitted_fragment->source.c_str());
        return false;
    }

    constexpr std::array<float, 9> vertices{-2, -2, 0, 6, -2, 0, -2, 6, 0};
    id<MTLBuffer> vertex_buffer = [context.device newBufferWithBytes:vertices.data()
        length:sizeof(vertices) options:MTLResourceStorageModeShared];
    std::vector<std::uint32_t> constants = item.constants;
    if (constants.empty()) constants.push_back(0);
    id<MTLBuffer> constant_buffer = [context.device newBufferWithBytes:constants.data()
        length:constants.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    auto* target_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:kWidth height:kHeight mipmapped:NO];
    target_descriptor.storageMode = MTLStorageModePrivate;
    target_descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [context.device newTextureWithDescriptor:target_descriptor];
    id<MTLBuffer> readback = [context.device newBufferWithLength:kWidth * kHeight * kPixelBytes
                                                         options:MTLResourceStorageModeShared];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 19;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [context.device newArgumentTableWithDescriptor:table_descriptor
                                                                           error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 4;
    id<MTLResidencySet> residency = [context.device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    id<MTL4CommandAllocator> allocator = [context.device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [context.device newCommandBuffer];
    if (!vertex_buffer || !constant_buffer || !target || !readback || !table || !residency ||
        !allocator || !commands) {
        std::fprintf(stderr, "%s: Metal resource creation failed.\n", item.name);
        return false;
    }
    [table setAddress:constant_buffer.gpuAddress atIndex:0];
    [table setAddress:vertex_buffer.gpuAddress atIndex:18];
    for (id<MTLAllocation> allocation in @[vertex_buffer, constant_buffer, target, readback]) {
        [residency addAllocation:allocation];
    }
    [residency commit];

    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return false;
    [render setRenderPipelineState:state];
    [render setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) return false;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
               visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:target sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:readback destinationOffset:0
    destinationBytesPerRow:kWidth * kPixelBytes destinationBytesPerImage:0];
    [copy endEncoding];
    [commands endCommandBuffer];

    NSArray* retained = @[vertex_library, fragment_library, state, vertex_buffer,
                          constant_buffer, target, readback, table, residency, allocator];
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
        std::fprintf(stderr, "%s: Metal execution timed out.\n", item.name);
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "%s: Metal execution failed: %s\n", item.name,
                     execution_error.localizedDescription.UTF8String);
        return false;
    }
    const auto* words = static_cast<const std::uint32_t*>(readback.contents);
    bool matched = true;
    for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            if (words[pixel * 4 + lane] != expected[lane]) {
                std::fprintf(stderr, "%s: pixel %lu lane %u GPU 0x%08x CPU 0x%08x\n", item.name,
                             static_cast<unsigned long>(pixel), lane, words[pixel * 4 + lane],
                             expected[lane]);
                matched = false;
                break;
            }
        }
        if (!matched) break;
    }
    return matched;
}

// Runs every case and reports a process exit status.
inline int run_cases(const std::vector<OperationCase>& cases) {
    @autoreleasepool {
        auto context = make_context();
        if (!context) return 1;
        int failures = 0;
        for (const auto& item : cases) {
            if (!run_case(*context, item)) {
                std::fprintf(stderr, "FAILED %s\n", item.name);
                ++failures;
            }
        }
        std::printf("%zu operation cases, %d failed\n", cases.size(), failures);
        return failures == 0 ? 0 : 1;
    }
}

}  // namespace armigo::shader::test::operation_harness
