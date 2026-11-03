// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic sampling probe: decoded TSC descriptors map
// onto Metal sampler states through the production decode and mapping path,
// and the sampled pixels must match the CPU model of the requested address
// mode and filter on a small synthetic texture.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "texture_format.h"

#include <armigo/graphics/texture.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <variant>

namespace {

using namespace armigo::graphics;
namespace platform = armigo::platform;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (false)

constexpr std::uint32_t kTextureWidth = 4;
constexpr std::uint32_t kTextureHeight = 4;

// Distinct synthetic texel colours; the CPU expectations below read them.
std::array<std::array<std::uint32_t, kTextureWidth>, kTextureHeight> g_pixels;
void fill_pixels() {
    for (unsigned y = 0; y < kTextureHeight; ++y) {
        for (unsigned x = 0; x < kTextureWidth; ++x) {
            const unsigned red = 20 + 60 * x;
            const unsigned green = 20 + 60 * y;
            g_pixels[y][x] = red | (green << 8) | (7u << 16) | (250u << 24);
        }
    }
}

float channel(std::uint32_t pixel, unsigned index) {
    return static_cast<float>((pixel >> (index * 8)) & 0xffu) / 255.0f;
}

void put32(std::array<std::byte, 32>& bytes, std::size_t offset,
           std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
    }
}

// Builds a synthetic TSC: per-axis wrap codes, nearest/linear mag+min filter,
// no mip filter, everything else at the decoded zero defaults.
std::array<std::byte, 32> tsc(unsigned wrap_u, unsigned wrap_v,
                              unsigned filter) {
    std::array<std::byte, 32> bytes{};
    put32(bytes, 0, wrap_u | (wrap_v << 3) | (2u << 6));
    put32(bytes, 4, filter | (filter << 4) | (1u << 6));
    return bytes;
}

enum class Model { Repeat, Mirror, Clamp };

// Independent CPU model of one sample: the UV transforms into texel space
// with the half-texel center offset, the neighbour indices resolve per the
// address mode, and nearest or bilinear weighting produces the colour.
std::array<float, 4> model_sample(float u, float v, Model mode_x,
                                  Model mode_y, bool linear) {
    const auto resolve = [](double index, Model mode) {
        if (mode == Model::Clamp) {
            return std::max(0.0, std::min(3.0, index));
        }
        if (mode == Model::Mirror) {
            const double period = std::fmod(std::fmod(index, 8.0) + 8.0, 8.0);
            return period < 4.0 ? period : 7.0 - period;
        }
        return std::fmod(std::fmod(index, 4.0) + 4.0, 4.0);
    };
    const auto sample_texel = [&](double x, double y) {
        const unsigned pixel = g_pixels[static_cast<unsigned>(std::lround(y))]
                                     [static_cast<unsigned>(std::lround(x))];
        std::array<float, 4> value{};
        for (unsigned component = 0; component < 4; ++component) {
            value[component] = channel(pixel, component);
        }
        return value;
    };
    const double position_x = static_cast<double>(u) * 4.0 - 0.5;
    const double position_y = static_cast<double>(v) * 4.0 - 0.5;
    if (!linear) {
        return sample_texel(resolve(std::floor(position_x + 0.5), mode_x),
                            resolve(std::floor(position_y + 0.5), mode_y));
    }
    const double x0 = std::floor(position_x), fx = position_x - x0;
    const double y0 = std::floor(position_y), fy = position_y - y0;
    const auto at = [&](double x, double y) {
        return sample_texel(resolve(x, mode_x), resolve(y, mode_y));
    };
    std::array<float, 4> value{};
    for (unsigned component = 0; component < 4; ++component) {
        const double low =
            at(x0, y0)[component] * (1.0 - fx) + at(x0 + 1.0, y0)[component] * fx;
        const double high = at(x0, y0 + 1.0)[component] * (1.0 - fx) +
                            at(x0 + 1.0, y0 + 1.0)[component] * fx;
        value[component] = static_cast<float>(low * (1.0 - fy) + high * fy);
    }
    return value;
}

struct Case {
    const char* name;
    std::array<std::byte, 32> tsc;
    float u;
    float v;
    bool filtered;  // Linear probes tolerate the GPU's unorm filter precision.
    float expected[4];
};

std::array<Case, 6> build_cases() {
    const auto clamp_to_edge = 2u;
    const auto nearest = 1u;
    const auto linear = 2u;
    const auto mirror = 1u;
    const auto repeat = 0u;
    const auto expected = [&](float u, float v, Model mode_x, Model mode_y,
                              bool lin) {
        std::array<float, 4> value =
            model_sample(u, v, mode_x, mode_y, lin);
        return value;
    };
    const auto repeat_nearest =
        expected(2.1f, 3.9f, Model::Repeat, Model::Repeat, false);
    const auto mirror_nearest =
        expected(-1.2f, 0.2f, Model::Mirror, Model::Mirror, false);
    const auto clamp_nearest =
        expected(1.7f, -0.3f, Model::Clamp, Model::Clamp, false);
    const auto repeat_linear =
        expected(1.1f, 1.1f, Model::Repeat, Model::Repeat, true);
    const auto mirror_linear =
        expected(1.25f, 0.75f, Model::Mirror, Model::Mirror, true);
    const auto clamp_linear =
        expected(1.1f, 0.9f, Model::Clamp, Model::Clamp, true);
    return {{
        {"repeat-nearest", tsc(repeat, repeat, nearest), 2.1f, 3.9f, false,
         {repeat_nearest[0], repeat_nearest[1], repeat_nearest[2],
          repeat_nearest[3]}},
        {"mirror-nearest", tsc(mirror, mirror, nearest), -1.2f, 0.2f, false,
         {mirror_nearest[0], mirror_nearest[1], mirror_nearest[2],
          mirror_nearest[3]}},
        {"clamp-nearest", tsc(clamp_to_edge, clamp_to_edge, nearest), 1.7f,
         -0.3f, false,
         {clamp_nearest[0], clamp_nearest[1], clamp_nearest[2],
          clamp_nearest[3]}},
        {"repeat-linear", tsc(repeat, repeat, linear), 1.1f, 1.1f, true,
         {repeat_linear[0], repeat_linear[1], repeat_linear[2],
          repeat_linear[3]}},
        {"mirror-linear", tsc(mirror, mirror, linear), 1.25f, 0.75f, true,
         {mirror_linear[0], mirror_linear[1], mirror_linear[2],
          mirror_linear[3]}},
        {"clamp-linear", tsc(clamp_to_edge, clamp_to_edge, linear), 1.1f,
         0.9f, false,
         {clamp_linear[0], clamp_linear[1], clamp_linear[2], clamp_linear[3]}},
    }};
}

const char* kKernel = R"metal(#include <metal_stdlib>
using namespace metal;
kernel void armigo_sampler_probe(
    texture2d<float, access::sample> source [[texture(0)]],
    sampler state [[sampler(0)]],
    constant float2& coordinates [[buffer(0)]],
    device float4* result [[buffer(1)]]) {
    result[0] = source.sample(state, coordinates);
}
)metal";

bool run(id<MTLDevice> device) {
    fill_pixels();
    id<MTLTexture> texture = nil;
    {
        auto* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                         width:kTextureWidth
                                        height:kTextureHeight
                                     mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead;
        texture = [device newTextureWithDescriptor:descriptor];
        CHECK(texture != nil);
        if (texture == nil) return false;
        [texture replaceRegion:MTLRegionMake2D(0, 0, kTextureWidth,
                                               kTextureHeight)
                   mipmapLevel:0
                     withBytes:g_pixels.data()
                   bytesPerRow:kTextureWidth * 4];
    }
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:kKernel]
                     options:options
                       error:&error];
    MTL4CompilerDescriptor* compiler_descriptor =
        [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    if (!library || !compiler) {
        std::fprintf(stderr, "Sampler probe compiler setup failed: %s\n",
                     error.localizedDescription.UTF8String);
        return false;
    }
    MTL4LibraryFunctionDescriptor* function =
        [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"armigo_sampler_probe";
    MTL4ComputePipelineDescriptor* pipeline_descriptor =
        [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil
                                       error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Sampler probe pipeline compilation failed: %s\n",
                     error.localizedDescription.UTF8String);
        return false;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor =
        [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    table_descriptor.maxTextureBindCount = 1;
    table_descriptor.maxSamplerStateBindCount = 1;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor =
        [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 24;
    id<MTLResidencySet> residency = [device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!queue || !allocator || !commands || !table || !residency) {
        std::fputs("Sampler probe dispatch allocation failed.\n", stderr);
        return false;
    }
    NSMutableArray* retained = [NSMutableArray
        arrayWithArray:@[texture, library, compiler, pipeline, queue,
                         allocator, commands, table, residency]];
    const auto retain_resource = [&](id<MTLResource> resource) {
        CHECK(resource != nil);
        [retained addObject:resource];
        [residency addAllocation:resource];
    };
    [table setTexture:texture.gpuResourceID atIndex:0];

    // Full production stack per case: decode the synthetic TSC, map it to
    // Metal sampler properties, and build the real sampler state.
    std::array<id<MTLSamplerState>, 6> states{};
    std::array<id<MTLBuffer>, 6> coordinate_buffers{};
    std::array<id<MTLBuffer>, 6> output_buffers{};
    auto cases = build_cases();
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& probe = cases[index];
        const auto decoded = decode_sampler_descriptor(probe.tsc);
        const auto* sampler = std::get_if<SamplerDescriptor>(&decoded);
        CHECK(sampler != nullptr);
        if (sampler == nullptr) return false;
        const auto mapped = platform::metal_sampler_properties(*sampler);
        auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties != nullptr);
        if (properties == nullptr) return false;
        states[index] = [device
            newSamplerStateWithDescriptor:platform::metal_sampler_descriptor(
                                              *properties)];
        CHECK(states[index] != nil);
        const std::array<float, 2> coordinate{{probe.u, probe.v}};
        coordinate_buffers[index] = [device
            newBufferWithBytes:coordinate.data()
                        length:sizeof(coordinate)
                       options:MTLResourceStorageModeShared];
        output_buffers[index] = [device
            newBufferWithLength:sizeof(float) * 4
                        options:MTLResourceStorageModeShared];
        retain_resource(coordinate_buffers[index]);
        retain_resource(output_buffers[index]);
    }
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    CHECK(encoder != nil);
    if (encoder == nil) return false;
    for (std::size_t index = 0; index < cases.size(); ++index) {
        [table setSamplerState:states[index].gpuResourceID atIndex:0];
        [table setAddress:coordinate_buffers[index].gpuAddress atIndex:0];
        [table setAddress:output_buffers[index].gpuAddress atIndex:1];
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block bool failed = false;
    MTL4CommitOptions* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        failed = feedback.error != nil;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    if (dispatch_semaphore_wait(completed,
            dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0 ||
        failed) {
        std::fputs("Sampler probe execution failed or timed out.\n", stderr);
        return false;
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& probe = cases[index];
        const auto* result =
            static_cast<const float*>(output_buffers[index].contents);
        const float tolerance = probe.filtered ? 2.0f / 255.0f : 1.0f / 255.0f;
        for (unsigned component = 0; component < 4; ++component) {
            if (std::fabs(result[component] - probe.expected[component]) >
                tolerance) {
                std::fprintf(stderr,
                             "Sampler probe %s component %u: expected %.6f, "
                             "got %.6f\n",
                             probe.name, component, probe.expected[component],
                             result[component]);
                ++failures;
            }
        }
    }
    std::printf("Sampled %zu synthetic sampler probes on GPU.\n",
                cases.size());
    return true;
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fputs("Sampler GPU tests require a Metal 4 device.\n",
                       stderr);
            return 1;
        }
        if (!run(device)) {
            std::printf("Metal sampler execution failures: %u\n", failures);
            return 1;
        }
        std::printf("Metal sampler execution failures: %u\n", failures);
        return failures == 0 ? 0 : 1;
    }
}
