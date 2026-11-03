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

#include "pipeline/resources.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace armigo::shader::test::cbuf_msl {
std::string gpu_fragment_source(unsigned case_index);
std::string gpu_argument_buffer_source();
}

namespace {

using armigo::shader::pipeline::constant_buffer_window_bytes;

constexpr char vertex_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOutput { float4 position [[position]]; };
vertex VertexOutput armigo_test_vertex(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {float2(-2.0f, -2.0f), float2(6.0f, -2.0f), float2(-2.0f, 6.0f)};
    VertexOutput output;
    output.position = float4(positions[vertex_id], 0.0f, 1.0f);
    return output;
}
)";

struct Case {
  const char *name;
  unsigned source_index;
  std::uint32_t size_bytes;
  std::vector<std::uint32_t> backing;
  std::uint32_t dynamic_offset;
  std::array<std::uint32_t, 4> expected;
};

struct CompiledFragment {
  id<MTLLibrary> library{};
  id<MTLRenderPipelineState> pipeline{};
};
using PipelineCache = std::array<CompiledFragment, 13>;

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
             id<MTL4CommandQueue> queue, const Case &item,
             id<MTLLibrary> vertex_library, PipelineCache &cache) {
  auto &compiled = cache.at(item.source_index);
  if (!compiled.pipeline) {
    std::string fragment_source;
    try {
      fragment_source = armigo::shader::test::cbuf_msl::gpu_fragment_source(
          item.source_index);
    } catch (const std::exception &error) {
      std::fprintf(stderr, "%s: %s\n", item.name, error.what());
      return 1;
    }
    compiled.library = compile_library(device, fragment_source, item.name);
    if (!compiled.library)
      return 1;

    MTL4LibraryFunctionDescriptor *vertex_function =
        [MTL4LibraryFunctionDescriptor new];
    vertex_function.library = vertex_library;
    vertex_function.name = @"armigo_test_vertex";
    MTL4LibraryFunctionDescriptor *fragment_function =
        [MTL4LibraryFunctionDescriptor new];
    fragment_function.library = compiled.library;
    fragment_function.name = @"armigo_fragment_main";
    MTL4RenderPipelineDescriptor *pipeline_descriptor =
        [MTL4RenderPipelineDescriptor new];
    pipeline_descriptor.vertexFunctionDescriptor = vertex_function;
    pipeline_descriptor.fragmentFunctionDescriptor = fragment_function;
    pipeline_descriptor.inputPrimitiveTopology =
        MTLPrimitiveTopologyClassTriangle;
    pipeline_descriptor.colorAttachments[0].pixelFormat =
        MTLPixelFormatRGBA32Uint;
    NSError *error = nil;
    compiled.pipeline =
        [compiler newRenderPipelineStateWithDescriptor:pipeline_descriptor
                                   compilerTaskOptions:nil
                                                 error:&error];
    if (!compiled.pipeline) {
      std::fprintf(stderr, "%s pipeline creation failed: %s\n", item.name,
                   error.localizedDescription.UTF8String);
      return 1;
    }
  }
  id<MTLLibrary> fragment_library = compiled.library;
  id<MTLRenderPipelineState> pipeline = compiled.pipeline;
  NSError *error = nil;

  // The binding contract: a bound window holds the guest's whole words below
  // the bound size and zero words past it, up to the window size.
  std::vector<std::uint32_t> window(constant_buffer_window_bytes / sizeof(std::uint32_t), 0u);
  const auto words = std::min<std::size_t>({item.backing.size(), item.size_bytes / sizeof(std::uint32_t),
                                            window.size()});
  std::copy_n(item.backing.begin(), words, window.begin());
  id<MTLBuffer> constants =
      [device newBufferWithBytes:window.data()
                          length:window.size() * sizeof(std::uint32_t)
                         options:MTLResourceStorageModeShared];
  std::array<std::uint32_t, 18> sizes{};
  sizes[0] = item.size_bytes;
  sizes[1] = sizeof(item.dynamic_offset);
  id<MTLBuffer> size_table =
      [device newBufferWithBytes:sizes.data()
                          length:sizes.size() * sizeof(std::uint32_t)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> dynamic_offset =
      [device newBufferWithBytes:&item.dynamic_offset
                          length:sizeof(item.dynamic_offset)
                         options:MTLResourceStorageModeShared];
  auto *target_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                   width:1
                                  height:1
                               mipmapped:NO];
  target_descriptor.storageMode = MTLStorageModePrivate;
  target_descriptor.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [device newTextureWithDescriptor:target_descriptor];
  id<MTLBuffer> readback =
      [device newBufferWithLength:4 * sizeof(std::uint32_t)
                          options:MTLResourceStorageModeShared];
  MTL4ArgumentTableDescriptor *table_descriptor =
      [MTL4ArgumentTableDescriptor new];
  table_descriptor.maxBufferBindCount = 12;
  id<MTL4ArgumentTable> table =
      [device newArgumentTableWithDescriptor:table_descriptor error:&error];
  MTLResidencySetDescriptor *residency_descriptor =
      [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 5;
  id<MTLResidencySet> residency =
      [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  if (!constants || !size_table || !dynamic_offset || !target || !readback ||
      !table || !residency || !allocator || !commands) {
    std::fprintf(stderr, "%s Metal resource creation failed.\n", item.name);
    return 1;
  }
  [table setAddress:constants.gpuAddress atIndex:3];
  [table setAddress:dynamic_offset.gpuAddress atIndex:4];
  [table setAddress:size_table.gpuAddress atIndex:11];
  for (id<MTLAllocation> allocation in
       @[ constants, dynamic_offset, size_table, target, readback ])
    [residency addAllocation:allocation];
  [residency commit];

  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  id<MTL4RenderCommandEncoder> render =
      [commands renderCommandEncoderWithDescriptor:pass];
  if (!render)
    return 1;
  [render setRenderPipelineState:pipeline];
  [render setArgumentTable:table atStages:MTLRenderStageFragment];
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
                    sourceSize:MTLSizeMake(1, 1, 1)
                      toBuffer:readback
             destinationOffset:0
        destinationBytesPerRow:4 * sizeof(std::uint32_t)
      destinationBytesPerImage:0];
  [copy endEncoding];
  [commands endCommandBuffer];

  NSArray *retained = @[
    fragment_library, pipeline, constants, dynamic_offset, size_table, target,
    readback, table, residency, allocator
  ];
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
  const auto *result = static_cast<const std::uint32_t *>(readback.contents);
  for (unsigned lane = 0; lane < 4; ++lane) {
    if (result[lane] != item.expected[lane]) {
      std::fprintf(stderr, "%s lane %u: GPU 0x%08x expected 0x%08x\n",
                   item.name, lane, result[lane], item.expected[lane]);
      return 1;
    }
  }
  return 0;
}

std::vector<Case> make_cases() {
  std::vector<std::uint32_t> full_window(constant_buffer_window_bytes /
                                         sizeof(std::uint32_t));
  full_window.back() = 0xfaceb00cu;
  auto dynamic_full_window = full_window;
  std::vector<std::uint32_t> beyond_window(
      constant_buffer_window_bytes / sizeof(std::uint32_t) + 1);
  beyond_window.back() = 0xdecafbadu;
  return {
      {"short_last_word",
       0,
       8,
       {0x44332211u, 0x89abcdefu},
       0,
       {0x89abcdefu, 0x89abcdefu, 0x89abcdefu, 0x89abcdefu}},
      {"short_boundary", 1, 8, {0x44332211u, 0x89abcdefu}, 0, {0u, 0u, 0u, 0u}},
      {"huge_literal_dynamic", 2, 16, {1u, 2u, 3u, 4u}, 0, {0u, 0u, 0u, 0u}},
      {"unaligned_u32",
       3,
       8,
       {0x44332211u, 0x89abcdefu},
       0,
       {0x89abcdefu, 0x89abcdefu, 0x89abcdefu, 0x89abcdefu}},
      {"partial_tail_half",
       4,
       6,
       {0x44332211u, 0x89abcdefu},
       0,
       {0u, 0u, 0u, 0u}},
      {"narrow_byte", 5, 4, {0x44332211u}, 0, {0x33u, 0x33u, 0x33u, 0x33u}},
      {"word64_partial",
       6,
       8,
       {0x44332211u, 0x89abcdefu},
       0,
       {0x89abcdefu, 0u, 0x89abcdefu, 0u}},
      {"window_64k_last_word",
       7,
       constant_buffer_window_bytes,
       std::move(full_window),
       0,
       {0xfaceb00cu, 0xfaceb00cu, 0xfaceb00cu, 0xfaceb00cu}},
      {"unbound_zero_size", 8, 0, {}, 0, {0u, 0u, 0u, 0u}},
      {"runtime_dynamic_zero_size", 9, 0, {}, 0, {0u, 0u, 0u, 0u}},
      {"runtime_dynamic_one_byte", 9, 1, {0x44332211u}, 0, {0u, 0u, 0u, 0u}},
      {"runtime_dynamic_final_word",
       9,
       8,
       {0x44332211u, 0x89abcdefu},
       4,
       {0x89abcdefu, 0x89abcdefu, 0x89abcdefu, 0x89abcdefu}},
      {"runtime_dynamic_binding_boundary",
       9,
       8,
       {0x44332211u, 0x89abcdefu},
       8,
       {0u, 0u, 0u, 0u}},
      {"runtime_dynamic_larger_binding",
       9,
       16,
       {0x44332211u, 0x89abcdefu, 0x12345678u, 0x76543210u},
       4,
       {0x89abcdefu, 0x89abcdefu, 0x89abcdefu, 0x89abcdefu}},
      {"runtime_dynamic_64k_last_word",
       9,
       constant_buffer_window_bytes,
       std::move(dynamic_full_window),
       constant_buffer_window_bytes - 4,
       {0xfaceb00cu, 0xfaceb00cu, 0xfaceb00cu, 0xfaceb00cu}},
      {"runtime_dynamic_reflected_boundary",
       9,
       constant_buffer_window_bytes + 4,
       std::move(beyond_window),
       constant_buffer_window_bytes,
       {0u, 0u, 0u, 0u}},
      {"runtime_dynamic_uint_max",
       9,
       16,
       {1u, 2u, 3u, 4u},
       UINT32_MAX,
       {0u, 0u, 0u, 0u}},
      {"signed_byte",
       10,
       4,
       {0x80abcdefu},
       0,
       {0xffffff80u, 0xffffff80u, 0xffffff80u, 0xffffff80u}},
      {"signed_half",
       11,
       4,
       {0x80000000u},
       0,
       {0xffff8000u, 0xffff8000u, 0xffff8000u, 0xffff8000u}},
      {"unaligned_half",
       12,
       4,
       {0x89abcdefu},
       0,
       {0x89abu, 0x89abu, 0x89abu, 0x89abu}},
  };
}

} // namespace

namespace armigo::shader::test::cbuf_msl {

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
    id<MTLLibrary> vertex_library =
        compile_library(device, vertex_source, "constant-buffer vertex");
    if (!compiler || !queue || !vertex_library) {
      std::fputs("Metal 4 compiler, queue, or vertex library unavailable.\n",
                 stderr);
      return 1;
    }
    const auto cases = make_cases();
    PipelineCache cache{};
    int failures = 0;
    try {
      if (!compile_library(device, gpu_argument_buffer_source(), "emitted argument-buffer stage")) ++failures;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "Argument-buffer compilation fixture failed: %s\n", error.what());
      ++failures;
    }
    for (const auto &item : cases)
      failures +=
          run_case(device, compiler, queue, item, vertex_library, cache);
    std::printf("%zu constant-buffer GPU cases, %d failed\n", cases.size(),
                failures);
    return failures == 0 ? 0 : 1;
  }
}

} // namespace armigo::shader::test::cbuf_msl
