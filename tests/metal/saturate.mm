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

#include <armigo/shader/saturate.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Link the fixture source with ARMIGO_SATURATE_ORACLE_ONLY to share the raw
// input words without registering the CPU test harness in this executable.
namespace armigo::shader::test {
std::vector<std::uint32_t> saturate_inputs(std::size_t random_count);
}

namespace {

bool run(id<MTLDevice> device, bool safe_math) {
    const auto inputs = armigo::shader::test::saturate_inputs(100000);
    constexpr std::uint32_t guard = 0x5b1e3a92u;
    constexpr std::size_t guard_count = 17;
    std::vector<std::uint32_t> expected;
    expected.reserve(inputs.size());
    for (const auto bits : inputs) expected.push_back(armigo::shader::saturate_float32(bits));
    std::vector<std::uint32_t> initial(inputs.size() + guard_count, guard);
    const std::uint32_t count = static_cast<std::uint32_t>(inputs.size());
    const std::string source =
        "#include <metal_stdlib>\nusing namespace metal;\n" +
        armigo::shader::emit_saturate_msl() + R"msl(
kernel void saturate_test(device const uint* input [[buffer(0)]],
                          device uint* result [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    if (index < count) result[index] = armigo_saturate_float32(input[index]);
}
)msl";
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    if (safe_math) options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:options error:&error];
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    if (!library || !compiler) {
        std::fprintf(stderr, "Saturation Metal compiler setup failed: %s\n",
                     error.localizedDescription.UTF8String);
        return false;
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"saturate_test";
    MTL4ComputePipelineDescriptor* pipeline_descriptor = [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Saturation Metal 4 pipeline compilation failed: %s\n",
                     error.localizedDescription.UTF8String);
        return false;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 3;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 3;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!queue || !allocator || !commands || !table || !residency) {
        std::fputs("Saturation Metal 4 dispatch allocation failed.\n", stderr);
        return false;
    }
    const void* data[] = {inputs.data(), initial.data(), &count};
    const std::size_t lengths[] = {inputs.size() * 4, initial.size() * 4,
                                   sizeof(count)};
    NSMutableArray* retained = [NSMutableArray arrayWithArray:
        @[library, compiler, pipeline, queue, allocator, commands, table, residency]];
    id<MTLBuffer> output = nil;
    for (NSUInteger binding = 0; binding < 3; ++binding) {
        id<MTLBuffer> buffer = [device newBufferWithBytes:data[binding]
                                                 length:lengths[binding]
                                                options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::fputs("Saturation Metal buffer allocation failed.\n", stderr);
            return false;
        }
        [retained addObject:buffer];
        [residency addAllocation:buffer];
        [table setAddress:buffer.gpuAddress atIndex:binding];
        if (binding == 1) output = buffer;
    }
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) {
        [commands endCommandBuffer];
        std::fputs("Saturation Metal compute encoder creation failed.\n", stderr);
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    const auto group_size = std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count + guard_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block bool failed = false;
    MTL4CommitOptions* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        // Retain all submitted allocations through actual completion, even
        // when the bounded host wait returns before the GPU callback.
        static_cast<void>(retained.count);
        failed = feedback.error != nil;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    if (dispatch_semaphore_wait(completed,
            dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0 || failed) {
        std::fputs("Saturation Metal execution failed or timed out.\n", stderr);
        return false;
    }
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "Saturation GPU mismatch at %zu: input %08x, "
                         "expected %08x, got %08x.\n",
                         index, inputs[index], expected[index], actual[index]);
            return false;
        }
    }
    for (std::size_t index = expected.size(); index < initial.size(); ++index) {
        if (actual[index] != guard) {
            std::fputs("Saturation GPU wrote beyond the output count.\n", stderr);
            return false;
        }
    }
    std::printf("Verified %u saturation GPU results (safe math %d).\n",
                count, safe_math);
    return true;
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fputs("Saturation GPU tests require a Metal 4 device.\n", stderr);
            return 1;
        }
        for (const bool safe_math : {false, true}) {
            @autoreleasepool {
                if (!run(device, safe_math)) return 1;
            }
        }
        return 0;
    }
}
