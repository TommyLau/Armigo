// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../shader/pipeline/fragment_depth_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using namespace armigo::shader::test::fragment_depth;
constexpr unsigned width = 768, height = 432, triangles = 2000;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
id<MTLLibrary> library(id<MTLDevice> device, const std::string& source) {
    auto options = [MTLCompileOptions new]; options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe; options.preserveInvariance = YES;
    NSError* error = nil;
    auto result = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
    return result;
}
id<MTLRenderPipelineState> pipeline(id<MTL4Compiler> compiler, id<MTLLibrary> vertex,
                                   id<MTLLibrary> fragment, NSString* function, bool depth) {
    auto vs = [MTL4LibraryFunctionDescriptor new]; vs.library = vertex; vs.name = @"synthetic_vertex";
    auto fs = [MTL4LibraryFunctionDescriptor new]; fs.library = fragment; fs.name = function;
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vs; descriptor.fragmentFunctionDescriptor = fs;
    if (!depth) descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    auto result = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
    return result;
}
}
int main() { @autoreleasepool { try {
    auto device = MTLCreateSystemDefaultDevice(); check(device != nil, "Metal device");
    NSError* error = nil;
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    auto queue = [device newMTL4CommandQueue]; check(compiler && queue, "Metal 4 compiler and queue");
    const auto vertex = library(device, R"metal(#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position, invariant]]; };
vertex Vertex synthetic_vertex(uint id [[vertex_id]], const device float4* points [[buffer(0)]], constant float& shift [[buffer(1)]]) {
    Vertex result; result.position = points[id]; result.position.z += shift * result.position.w; return result;
}
fragment void synthetic_depth() {}
)metal");
    const auto artifact = translate(program());
    const auto fragment = library(device, artifact.source);
    const auto depth_pipeline = pipeline(compiler, vertex, vertex, @"synthetic_depth", true);
    const auto read_pipeline = pipeline(compiler, vertex, fragment, @"armigo_fragment_main", false);
    auto depth_state_descriptor = [MTLDepthStencilDescriptor new];
    depth_state_descriptor.depthCompareFunction = MTLCompareFunctionLessEqual; depth_state_descriptor.depthWriteEnabled = YES;
    const auto depth_state = [device newDepthStencilStateWithDescriptor:depth_state_descriptor];
    auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:width height:height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    const auto stored = [device newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    const auto color = [device newTextureWithDescriptor:td];
    const auto readback = [device newBufferWithLength:width * height * 16 options:MTLResourceStorageModeShared];
    auto table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2; table_descriptor.maxTextureBindCount = 128;
    const auto textures = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    unsigned texture_count = 0;
    for (const auto& resource : artifact.reflection.resources) {
        if (std::holds_alternative<TextureUse>(resource.use)) {
            [textures setTexture:stored.gpuResourceID atIndex:std::get<MetalRange>(resource.location).first]; ++texture_count;
        } else throw std::runtime_error("Synthetic reader unexpectedly needs another resource");
    }
    check(texture_count == 1, "one stored-depth texture");
    std::mt19937 rng(0xc0ffeeu); std::uniform_real_distribution<float> unit(0, 1);
    unsigned long covered_total = 0;
    for (unsigned mode = 0; mode < 5; ++mode) {
        std::vector<std::array<float, 4>> points(triangles * 3);
        for (unsigned triangle = 0; triangle < triangles; ++triangle) {
            const float x = -1 + (triangle % 50) * .04f, y = -1 + (triangle / 50) * .05f;
            const float constant_depth = unit(rng);
            for (unsigned corner = 0; corner < 3; ++corner) {
                const float px = x + .04f * (corner == 1 ? .94f : .03f);
                const float py = y + .05f * (corner == 2 ? .94f : .03f);
                const float w = mode == 1 ? 1 : .5f + 3 * unit(rng);
                const float z = mode == 2 ? constant_depth : mode == 3 ? .001f * unit(rng)
                    : mode == 4 ? .999f + .000999f * unit(rng) : unit(rng);
                points[3 * triangle + corner] = {px * w, py * w, z * w, w};
            }
        }
        const auto vertices = [device newBufferWithBytes:points.data() length:points.size() * sizeof(points[0]) options:MTLResourceStorageModeShared];
        for (bool farther : {false, true}) {
            const float zero = 0, shift = farther ? 4e-6f : 0;
            const auto near_parameter = [device newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];
            const auto far_parameter = [device newBufferWithBytes:&shift length:4 options:MTLResourceStorageModeShared];
            const auto near_table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
            const auto far_table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
            [near_table setAddress:vertices.gpuAddress atIndex:0]; [near_table setAddress:near_parameter.gpuAddress atIndex:1];
            [far_table setAddress:vertices.gpuAddress atIndex:0]; [far_table setAddress:far_parameter.gpuAddress atIndex:1];
            const auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
            for (id<MTLAllocation> allocation in @[vertices, near_parameter, far_parameter, stored, color, readback]) [residency addAllocation:allocation];
            [residency commit];
            const auto allocator = [device newCommandAllocator]; const auto commands = [device newCommandBuffer];
            [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
            auto pass = [MTL4RenderPassDescriptor new];
            pass.depthAttachment.texture = stored; pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore; pass.depthAttachment.clearDepth = 1;
            auto render = [commands renderCommandEncoderWithDescriptor:pass];
            [render setRenderPipelineState:depth_pipeline]; [render setDepthStencilState:depth_state];
            [render setArgumentTable:near_table atStages:MTLRenderStageVertex];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:triangles * 3]; [render endEncoding];
            auto barrier = [commands computeCommandEncoder];
            [barrier barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
            [barrier endEncoding];
            pass = [MTL4RenderPassDescriptor new];
            pass.colorAttachments[0].texture = color; pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            render = [commands renderCommandEncoderWithDescriptor:pass];
            [render setRenderPipelineState:read_pipeline]; [render setArgumentTable:far_table atStages:MTLRenderStageVertex];
            [render setArgumentTable:textures atStages:MTLRenderStageFragment];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:triangles * 3]; [render endEncoding];
            auto copy = [commands computeCommandEncoder];
            [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
            [copy copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                sourceSize:MTLSizeMake(width,height,1) toBuffer:readback destinationOffset:0
                destinationBytesPerRow:width*16 destinationBytesPerImage:width*height*16];
            [copy endEncoding]; [commands endCommandBuffer];
            auto done = dispatch_semaphore_create(0); __block NSError* failure = nil;
            auto commit = [MTL4CommitOptions new];
            [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) { failure = feedback.error; dispatch_semaphore_signal(done); }];
            id<MTL4CommandBuffer> batch[]{commands}; [queue commit:batch count:1 options:commit];
            check(dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC)) == 0, "GPU completion timeout");
            if (failure) throw std::runtime_error(failure.localizedDescription.UTF8String);
            const auto* pixels = static_cast<const float*>(readback.contents);
            unsigned long covered = 0, above = 0, below = 0;
            for (unsigned pixel = 0; pixel < width * height; ++pixel) if (pixels[4*pixel] > .5f) {
                ++covered; above += pixels[4*pixel+1] > .5f; below += pixels[4*pixel+2] > .5f;
            }
            std::printf("depth mode=%u farther=%u covered=%lu above=%lu below=%lu\n", mode, unsigned(farther), covered, above, below);
            check(covered > 100000, "Synthetic triangles cover enough pixels");
            check(farther ? above == covered : above == 0 && below == 0,
                farther ? "A distinct farther surface must remain farther" : "A fragment must equal its own surface's stored depth");
            covered_total += covered;
        }
    }
    std::printf("Translated stored-depth comparisons passed (%lu pixels)\n", covered_total); return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; } } }
