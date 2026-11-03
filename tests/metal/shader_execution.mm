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

#include "fixtures.h"
#include <armigo/shader/msl_emitter.h>
#include <armigo/shader/reference_evaluator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace armigo::shader;
using namespace armigo::shader::test;
using Words = std::vector<std::uint32_t>;
constexpr std::uint32_t kGuard = 0xcdcdcdcdu;
constexpr std::uint32_t kGuardElements = 16;

bool is_nan(std::uint32_t value) {
    return (value & 0x7f800000u) == 0x7f800000u && (value & 0x007fffffu) != 0;
}

std::vector<Words> make_inputs(const Program& program, std::uint32_t count) {
    constexpr std::array<std::uint32_t, 12> samples = {
        0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x00800000u, 0x00000004u, 0x80000004u, 0x7f800000u,
        0xff800000u, 0x7fc00000u, 0x3f400000u, 0xbf000000u,
    };
    std::vector<Words> inputs(program.inputs.size(), Words(count));
    for (std::size_t binding = 0; binding < inputs.size(); ++binding) {
        for (std::uint32_t index = 0; index < count; ++index) {
            inputs[binding][index] = program.inputs[binding] == ScalarType::f32
                ? samples[(index + binding) % samples.size()]
                : 0xa5a50000u + index;
        }
    }
    if (count != 0 && (program.entry_point == "sum_left" ||
                       program.entry_point == "sum_right")) {
        inputs[0][0] = f32_bits(1e8f);
        inputs[1][0] = f32_bits(-1e8f);
        inputs[2][0] = f32_bits(1.0f);
    }
    if (count != 0 && program.entry_point == "mul_add") {
        inputs[0][0] = 0x3f800800u;
        inputs[1][0] = 0x3f800800u;
        inputs[2][0] = 0xbf801000u;
    }
    return inputs;
}

bool run_case(id<MTLDevice> device, const NamedProgram& fixture,
              std::uint32_t count, bool explicit_safe) {
    const auto& program = fixture.program;
    const auto emitted = emit_msl(program);
    if (!emitted.ok()) {
        std::fputs("Synthetic shader emission failed.\n", stderr);
        return false;
    }
    Words parameters;
    for (std::size_t index = 0; index < program.parameters.size(); ++index) {
        parameters.push_back(program.parameters[index] == ScalarType::f32
            ? f32_bits(index == 0 ? 1.25f : -0.5f) : 0xdeadbeefu);
    }
    const auto inputs = make_inputs(program, count);
    const auto reference = evaluate_reference(
        program, parameters, inputs, count,
        ReferenceOptions{SubnormalMode::flush_to_zero});
    if (!reference.ok()) {
        std::fputs("Synthetic shader reference evaluation failed.\n", stderr);
        return false;
    }

    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    if (explicit_safe) {
        options.mathMode = MTLMathModeSafe;
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:emitted.kernel.source.c_str()]
                     options:options error:&error];
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    if (!library || !compiler) {
        std::fprintf(stderr, "Metal shader compiler setup failed (code %ld).\n",
                     static_cast<long>(error.code));
        return false;
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = [NSString stringWithUTF8String:emitted.kernel.entry_point.c_str()];
    MTL4ComputePipelineDescriptor* pipeline_descriptor = [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Metal 4 pipeline compilation failed (code %ld).\n",
                     static_cast<long>(error.code));
        return false;
    }

    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* arguments_descriptor = [MTL4ArgumentTableDescriptor new];
    arguments_descriptor.maxBufferBindCount = emitted.kernel.bindings.size();
    id<MTL4ArgumentTable> arguments =
        [device newArgumentTableWithDescriptor:arguments_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = emitted.kernel.bindings.size();
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!queue || !allocator || !commands || !arguments || !residency) {
        std::fputs("Metal 4 dispatch allocation failed.\n", stderr);
        return false;
    }
    NSMutableArray* retained_resources =
        [NSMutableArray arrayWithArray:@[queue, allocator, commands, arguments,
                                        residency, pipeline, library, compiler]];
    std::vector<id<MTLBuffer>> outputs(program.outputs.size());
    for (const auto& binding : emitted.kernel.bindings) {
        Words words;
        switch (binding.kind) {
        case MslBindingKind::parameters: words = parameters; break;
        case MslBindingKind::input: words = inputs[binding.ordinal]; break;
        case MslBindingKind::output: words.assign(count + kGuardElements, kGuard); break;
        case MslBindingKind::element_count: words.push_back(count); break;
        }
        // Even a zero-element dispatch needs a valid bound GPU address.
        if (words.empty()) {
            words.push_back(kGuard);
        }
        id<MTLBuffer> buffer = [device newBufferWithBytes:words.data()
                                                  length:words.size() * sizeof(std::uint32_t)
                                                 options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::fputs("Metal shader buffer allocation failed.\n", stderr);
            return false;
        }
        [retained_resources addObject:buffer];
        [residency addAllocation:buffer];
        [arguments setAddress:buffer.gpuAddress atIndex:binding.buffer_index];
        if (binding.kind == MslBindingKind::output) {
            outputs[binding.ordinal] = buffer;
        }
    }
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) {
        [commands endCommandBuffer];
        std::fputs("Metal 4 compute encoder creation failed.\n", stderr);
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:arguments];
    const NSUInteger group_size = std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup);
    // Extra invocations must leave every output guard untouched.
    [encoder dispatchThreads:MTLSizeMake(count + kGuardElements, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];

    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block bool execution_failed = false;
    MTL4CommitOptions* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        // The callback owns resources until GPU completion even after timeout.
        static_cast<void>(retained_resources.count);
        execution_failed = feedback.error != nil;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    if (dispatch_semaphore_wait(completed,
            dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0 || execution_failed) {
        std::fputs("Metal 4 shader execution failed or timed out.\n", stderr);
        return false;
    }
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        const auto* actual = static_cast<const std::uint32_t*>(outputs[output].contents);
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto expected = reference.outputs[output][index];
            const bool equivalent_nan = program.outputs[output] == ScalarType::f32 &&
                                        is_nan(expected) && is_nan(actual[index]);
            if (actual[index] != expected && !equivalent_nan) {
                std::fprintf(stderr, "%s output %zu element %u: expected %08x, got %08x.\n",
                             fixture.name.c_str(), output, index, expected, actual[index]);
                return false;
            }
        }
        for (std::uint32_t index = count; index < count + kGuardElements; ++index) {
            if (actual[index] != kGuard) {
                std::fputs("Shader wrote beyond the declared element count.\n", stderr);
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fputs("Shader execution tests require a Metal 4 GPU.\n", stderr);
            return 1;
        }
        auto fixtures = msl_fixtures();
        ProgramBuilder subnormal("subnormal_arithmetic");
        subnormal.add_input(ScalarType::f32);
        subnormal.add_output(ScalarType::f32);
        subnormal.add_output(ScalarType::f32);
        const auto input = subnormal.load_input(ScalarType::f32, 0);
        subnormal.store_output(ScalarType::f32, 0,
            subnormal.fmul(input, subnormal.constant_f32(0.25f)));
        subnormal.store_output(ScalarType::f32, 1, input);
        fixtures.push_back({"subnormal_arithmetic", subnormal.take()});
        std::size_t passed = 0;
        for (const bool explicit_safe : {false, true}) {
            for (const auto count : {0u, 65u}) {
                for (const auto& fixture : fixtures) {
                    @autoreleasepool {
                        if (!run_case(device, fixture, count, explicit_safe)) {
                            std::fprintf(stderr, "Failed: %s, count %u, safe math %d.\n",
                                         fixture.name.c_str(), count, explicit_safe);
                            return 1;
                        }
                        ++passed;
                    }
                }
            }
        }
        std::printf("Passed %zu Metal 4 shader dispatch comparisons.\n", passed);
        return 0;
    }
}
