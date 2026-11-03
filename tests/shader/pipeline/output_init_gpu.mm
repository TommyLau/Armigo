// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Offscreen verification of armigo-output-init-v1: synthetic vertex stages
// with early exits run through the real pipeline and draw on a Metal 4
// device, and the render target observes the entry-initialized outputs.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/phases.h"

#include <array>
#include <cstdio>
#include <string>

namespace armigo::shader::test::output_init {
std::string gpu_vertex_source(unsigned program);
}

namespace {

constexpr NSUInteger kWidth = 4;
constexpr NSUInteger kHeight = 4;
constexpr NSUInteger kPixelBytes = 4 * sizeof(std::uint32_t);

struct Case {
    const char *name;
    unsigned program;
    // Exports one observed output component to every colour channel.
    const char *fragment_source;
    std::uint32_t expected;
};

// Program zero takes its early exit at runtime, so the rendered position W is
// the entry value, and program one never executes the guarded generic store,
// so the exported varying is the entry zero. Uncovered pixels would keep the
// -1 clear colour, which differs from both expectations.
const Case cases[] = {
    {"early_exit_position_w", 0,
     "#include <metal_stdlib>\nusing namespace metal;\n"
     "fragment float4 armigo_fragment_main(float4 position [[position]]) "
     "{ return float4(position.w); }\n",
     0x3f800000u},
    {"split_path_generic_zero", 1,
     "#include <metal_stdlib>\nusing namespace metal;\n"
     "struct ProbeInput { float generic_value [[user(locn0_0), flat]]; };\n"
     "fragment float4 armigo_fragment_main(ProbeInput in [[stage_in]]) "
     "{ return float4(in.generic_value); }\n",
     0x00000000u},
};

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
  std::string vertex_source;
  try {
    vertex_source = armigo::shader::test::output_init::gpu_vertex_source(item.program);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s: %s\n", item.name, error.what());
    return 1;
  }
  const auto vertex_library = compile_library(device, vertex_source, item.name);
  const auto fragment_library =
      compile_library(device, item.fragment_source, item.name);
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
  pipeline_descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
  pipeline_descriptor.colorAttachments[0].pixelFormat =
      MTLPixelFormatRGBA32Float;
  NSError *error = nil;
  id<MTLRenderPipelineState> pipeline = [compiler
      newRenderPipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil
                                       error:&error];
  if (!pipeline) {
    std::fprintf(stderr, "%s pipeline creation failed: %s\n", item.name,
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
    std::fprintf(stderr, "%s Metal resource creation failed.\n", item.name);
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
  // One point with vertex id zero: the guest predicate takes the early-exit
  // arm, and the translated point size spreads it over the whole target.
  [render drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
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
    std::fprintf(stderr, "%s execution timed out.\n", item.name);
    return 1;
  }
  if (execution_error) {
    std::fprintf(stderr, "%s execution failed: %s\n", item.name,
                 execution_error.localizedDescription.UTF8String);
    return 1;
  }
  const auto *pixels = static_cast<const std::uint32_t *>(readback.contents);
  int failures = 0;
  for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
    for (unsigned lane = 0; lane < 4; ++lane) {
      if (pixels[pixel * 4 + lane] != item.expected) {
        std::fprintf(stderr, "%s: pixel %lu lane %u GPU 0x%08x expected 0x%08x\n",
                     item.name, static_cast<unsigned long>(pixel), lane,
                     pixels[pixel * 4 + lane], item.expected);
        failures = 1;
      }
    }
    if (failures) break;
  }
  return failures;
}

} // namespace

namespace armigo::shader::test::output_init {

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
    std::printf("%zu output initialization GPU cases, %d failed\n",
                sizeof(cases) / sizeof(cases[0]), failures);
    return failures == 0 ? 0 : 1;
  }
}

} // namespace armigo::shader::test::output_init
