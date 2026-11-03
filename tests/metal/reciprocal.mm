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

#include <armigo/shader/reciprocal.h>

#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// Link the independent oracle source with ARMIGO_RECIPROCAL_ORACLE_ONLY to share
// fixtures without registering the CPU test harness in this executable.
namespace armigo::shader::test {
std::uint32_t reciprocal_oracle(std::uint32_t input);
bool reciprocal_oracle_ready();
std::vector<std::uint32_t> reciprocal_inputs(std::size_t random_count);
}

namespace {

bool run(id<MTLDevice> device, bool safe_math) {
    const auto inputs = armigo::shader::test::reciprocal_inputs(100000);
    constexpr std::uint32_t guard = 0xa6d39c51u;
    constexpr std::size_t guard_count = 17;
    std::vector<std::uint32_t> expected;
    for (const auto input : inputs) {
        expected.push_back(armigo::shader::test::reciprocal_oracle(input));
    }
    std::vector<std::uint32_t> initial(inputs.size() + guard_count, guard);
    const std::uint32_t count = static_cast<std::uint32_t>(inputs.size());
    const std::string source =
        "#include <metal_stdlib>\nusing namespace metal;\n" +
        armigo::shader::emit_reciprocal_msl() + R"msl(
kernel void reciprocal_test(device const uint* input [[buffer(0)]],
                            device uint* result [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < count) result[index] = armigo_reciprocal_rne(input[index]);
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
        std::fprintf(stderr, "Reciprocal Metal compiler setup failed: %s\n",
                     error.localizedDescription.UTF8String);
        return false;
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"reciprocal_test";
    MTL4ComputePipelineDescriptor* pipeline_descriptor = [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Reciprocal Metal 4 pipeline compilation failed: %s\n",
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
        std::fputs("Reciprocal Metal 4 dispatch allocation failed.\n", stderr);
        return false;
    }
    const void* data[] = {inputs.data(), initial.data(), &count};
    const std::size_t lengths[] = {inputs.size() * sizeof(std::uint32_t),
                                   initial.size() * sizeof(std::uint32_t), sizeof(count)};
    NSMutableArray* retained = [NSMutableArray arrayWithArray:
        @[library, compiler, pipeline, queue, allocator, commands, table, residency]];
    id<MTLBuffer> output = nil;
    for (NSUInteger binding = 0; binding < 3; ++binding) {
        id<MTLBuffer> buffer = [device newBufferWithBytes:data[binding]
                                                 length:lengths[binding]
                                                options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::fputs("Reciprocal Metal buffer allocation failed.\n", stderr);
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
        std::fputs("Reciprocal Metal compute encoder creation failed.\n", stderr);
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
        std::fputs("Reciprocal Metal execution failed or timed out.\n", stderr);
        return false;
    }
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "Reciprocal GPU mismatch at %zu: expected %08x, got %08x.\n",
                         index, expected[index], actual[index]);
            return false;
        }
    }
    for (std::size_t index = expected.size(); index < initial.size(); ++index) {
        if (actual[index] != guard) {
            std::fputs("Reciprocal GPU wrote beyond the output count.\n", stderr);
            return false;
        }
    }
    std::printf("Verified %u Reciprocal GPU values (safe math %d).\n", count, safe_math);
    return true;
}

}  // namespace

int main() {
    @autoreleasepool {
        if (std::fesetround(FE_TONEAREST) != 0 ||
            !armigo::shader::test::reciprocal_oracle_ready()) {
            std::fputs("Reciprocal oracle requires native nearest-even division and gradual underflow.\n", stderr);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fputs("Reciprocal GPU tests require a Metal 4 device.\n", stderr);
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
