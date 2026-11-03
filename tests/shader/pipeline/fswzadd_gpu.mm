// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Offscreen FSWZADD verification: translated fragment stages with the quad
// swizzle add draw over a full-screen triangle on a Metal 4 device, and every
// pixel's four colour components are compared bit-exactly against the CPU
// model of the swizzle-add semantics. The quad lane a pixel observes is
// its position inside the aligned 2x2 quad: (0,0),(1,0),(0,1),(1,1).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/phases.h"

#include <array>
#include <bit>
#include <cstdio>
#include <string>

namespace armigo::shader::test::fswzadd {
std::string gpu_fragment_source(unsigned program);
}

namespace {

constexpr NSUInteger kWidth = 4;
constexpr NSUInteger kHeight = 4;
constexpr NSUInteger kPixelBytes = 4 * sizeof(std::uint32_t);

// Swizzle-add expression identical to the model in fswzadd_tests.
float reference_result(float x, float y, std::uint32_t pn_word, unsigned lane) {
    constexpr float x_table[4] = {1.0F, -1.0F, 1.0F, 0.0F};
    constexpr float y_table[4] = {1.0F, 1.0F, -1.0F, 1.0F};
    const unsigned code = (pn_word >> (lane * 2)) & 3;
    return x * x_table[code] + y * y_table[code];
}

struct Case {
    unsigned program;
    std::uint32_t component_masks[4];
    // Per-component (first source, second source) register words.
    std::uint32_t sources[4][2];
};

// Program zero: ordinary values through every code and lane combination.
// Program one: signed-zero and NaN propagation through the move code.
constexpr Case cases[2] = {{
    0,
    {0xe4U, 0x39U, 0x4eU, 0x93U},
    {{0x3fc00000U, 0xc0100000U}, {0x3fc00000U, 0xc0100000U},
     {0x3fc00000U, 0xc0100000U}, {0x3fc00000U, 0xc0100000U}},
}, {
    1,
    {0xc0U, 0xc0U, 0xc0U, 0xc0U},
    {{0x3f800000U, 0x80000000U}, {0x80000000U, 0x80000000U},
     {0x7fc00000U, 0x3f800000U}, {0x3f800000U, 0x7fc00000U}},
}};

const char *vertex_source =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "vertex float4 armigo_vertex_main(uint vid [[vertex_id]]) {\n"
    "  const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};\n"
    "  return float4(positions[vid], 0.0, 1.0);\n"
    "}\n";

id<MTLLibrary> compile_library(id<MTLDevice> device, const std::string &source,
                               const char *label) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  options.mathMode = MTLMathModeSafe;
  NSError *error = nil;
  id<MTLLibrary> result = [device
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:options
                     error:&error];
  if (!result)
    std::fprintf(stderr, "%s Metal compilation failed: %s\n", label,
                 error.localizedDescription.UTF8String);
  return result;
}

int run_case(id<MTLDevice> device, id<MTL4Compiler> compiler,
             id<MTL4CommandQueue> queue, const Case &item) {
  std::string fragment_msl;
  try {
    fragment_msl = armigo::shader::test::fswzadd::gpu_fragment_source(item.program);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "program %u: %s\n", item.program, error.what());
    return 1;
  }
  const auto vertex_library = compile_library(device, vertex_source, "vertex");
  const auto fragment_library =
      compile_library(device, fragment_msl, "fragment");
  if (!vertex_library || !fragment_library)
    return 1;
  MTL4LibraryFunctionDescriptor *vertex_function =
      [MTL4LibraryFunctionDescriptor new];
  vertex_function.library = vertex_library;
  vertex_function.name = @"armigo_vertex_main";
  MTL4LibraryFunctionDescriptor *fragment_function =
      [MTL4LibraryFunctionDescriptor new];
  fragment_function.library = fragment_library;
  fragment_function.name = @"armigo_fragment_main";
  MTL4RenderPipelineDescriptor *pipeline_descriptor =
      [MTL4RenderPipelineDescriptor new];
  pipeline_descriptor.vertexFunctionDescriptor = vertex_function;
  pipeline_descriptor.fragmentFunctionDescriptor = fragment_function;
  pipeline_descriptor.colorAttachments[0].pixelFormat =
      MTLPixelFormatRGBA32Float;
  NSError *error = nil;
  id<MTLRenderPipelineState> pipeline = [compiler
      newRenderPipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil
                                       error:&error];
  if (!pipeline) {
    std::fprintf(stderr, "program %u pipeline creation failed: %s\n", item.program,
                 error.localizedDescription.UTF8String);
    return 1;
  }
  auto *target_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                  width:kWidth
                                 height:kHeight
                              mipmapped:NO];
  target_descriptor.storageMode = MTLStorageModePrivate;
  target_descriptor.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [device newTextureWithDescriptor:target_descriptor];
  id<MTLBuffer> readback = [device
      newBufferWithLength:kWidth * kHeight * kPixelBytes
                  options:MTLResourceStorageModeShared];
  MTLResidencySetDescriptor *residency_descriptor =
      [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 2;
  id<MTLResidencySet> residency =
      [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  if (!target || !readback || !residency || !allocator || !commands) {
    std::fprintf(stderr, "program %u Metal resource creation failed.\n", item.program);
    return 1;
  }
  for (id<MTLAllocation> allocation in @[ target, readback ])
    [residency addAllocation:allocation];
  [residency commit];
  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
  id<MTL4RenderCommandEncoder> render =
      [commands renderCommandEncoderWithDescriptor:pass];
  if (!render)
    return 1;
  [render setRenderPipelineState:pipeline];
  // One full-screen triangle: every pixel of the target is covered, so every
  // 2x2 quad computes the swizzle add with real (non-helper) lanes.
  [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render endEncoding];
  id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
  if (!copy)
    return 1;
  [copy barrierAfterQueueStages:MTLStageFragment
                   beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
  [copy copyFromTexture:target
                   sourceSlice:0
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(0, 0, 0)
                     sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                       toBuffer:readback
              destinationOffset:0
         destinationBytesPerRow:kWidth * kPixelBytes
       destinationBytesPerImage:0];
  [copy endEncoding];
  [commands endCommandBuffer];
  NSArray *retained = @[ vertex_library, fragment_library, pipeline, target,
                         readback, residency, allocator ];
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  __block NSError *execution_error = nil;
  MTL4CommitOptions *options = [MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    static_cast<void>(retained.count);
    execution_error = feedback.error;
    dispatch_semaphore_signal(completed);
  }];
  id<MTL4CommandBuffer> submitted[] = {commands};
  [queue commit:submitted count:1 options:options];
  if (dispatch_semaphore_wait(
          completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
    std::fprintf(stderr, "program %u execution timed out.\n", item.program);
    return 1;
  }
  if (execution_error) {
    std::fprintf(stderr, "program %u execution failed: %s\n", item.program,
                 execution_error.localizedDescription.UTF8String);
    return 1;
  }
  const auto *pixels = static_cast<const std::uint32_t *>(readback.contents);
  int failures = 0;
  for (NSUInteger row = 0; row < kHeight && !failures; ++row) {
    for (NSUInteger column = 0; column < kWidth && !failures; ++column) {
      // Metal fragment quads align to even coordinates; the lane order inside
      // the quad is (0,0),(1,0),(0,1),(1,1).
      const unsigned lane =
          static_cast<unsigned>(((row % 2) << 1) | (column % 2));
      const auto *pixel = pixels + (row * kWidth + column) * 4;
      for (unsigned component = 0; component < 4; ++component) {
        const float x = std::bit_cast<float>(item.sources[component][0]);
        const float y = std::bit_cast<float>(item.sources[component][1]);
        const std::uint32_t expected = std::bit_cast<std::uint32_t>(
            reference_result(x, y, item.component_masks[component], lane));
        if (pixel[component] != expected) {
          std::fprintf(stderr,
                       "program %u pixel (%lu, %lu) lane %u component %u: "
                       "GPU 0x%08x expected 0x%08x\n",
                       item.program, static_cast<unsigned long>(column),
                       static_cast<unsigned long>(row), lane, component,
                       pixel[component], expected);
          failures = 1;
        }
      }
    }
  }
  return failures;
}

} // namespace

namespace armigo::shader::test::fswzadd {

int run_gpu_cases() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fputs("No Metal device.\n", stderr);
      return 1;
    }
    NSError *error = nil;
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                    error:&error];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (!compiler || !queue) {
      std::fputs("Metal 4 compiler or queue unavailable.\n", stderr);
      return 1;
    }
    int failures = 0;
    for (const auto &item : cases)
      failures += run_case(device, compiler, queue, item);
    std::printf("%zu FSWZADD GPU fragment cases, %d failed\n",
                sizeof(cases) / sizeof(cases[0]), failures);
    return failures == 0 ? 0 : 1;
  }
}

} // namespace armigo::shader::test::fswzadd
