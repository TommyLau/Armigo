// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Executes a translated fragment depth export on the GPU. The synthetic guest
// program exports a constant from R0 through the SPH extra-exports depth bit,
// and the cases pin the platform contract: the exported value replaces the
// rasterized depth for the compare and the write, and with no early fragment
// tests the export still governs LESS accepts and rejects.

#include "pipeline/phases.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
using namespace armigo::shader::pipeline;

constexpr NSUInteger extent = 8;
constexpr std::uint32_t exported_bits = 0x3f000000u;  // 0.5f

template <class T>
T take(Result<T> result) {
    if (const auto* error = std::get_if<Diagnostic>(&result))
        throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
}

MslArtifact translated_depth_fragment() {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-fragment-depth"};
            if (key.kind == FactKind::constant_buffer_bound_mask)
                return Known<FactValue>{std::uint32_t{1}, origin};
            if (key.kind == FactKind::constant_buffer_size)
                return Known<FactValue>{std::uint32_t{64}, origin};
            return Unknown{"Dynamic synthetic input"};
        }
    } source;
    StageImage image;
    image.stage = Stage::fragment;
    const auto append = [&](std::uint64_t bits) {
        for (unsigned byte = 0; byte < 8; ++byte)
            image.bytes.push_back(std::byte((bits >> (byte * 8)) & 255));
    };
    const auto instruction = [&](std::uint64_t bits) {
        if (image.bytes.size() % 32 == 0) append(0);
        append(bits);
    };
    // The extra-exports ABI reads the depth export from GPR1 (sample mask
    // would take GPR0), so GPR1 receives the constant word from buffer 0.
    instruction(0xef9400000007ff00ull | 1);
    instruction(0xe30000000007000full);
    image.sph[0x4c] = std::byte{0x02};
    FactReader facts(source);
    const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
    auto decoded = take(decode(image, facts, {}));
    auto registers = take(emit_ir(decoded, facts, {}));
    auto ssa = take(construct_ssa(std::move(registers), {}));
    auto legal = take(legalize(std::move(ssa), facts, profile));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile));
    bool depth_export = false, sample_mask_export = false;
    for (const auto& use : plan.io) {
        if (use.semantic != IoSemantic::fragment_depth && use.semantic != IoSemantic::sample_mask)
            continue;
        if (use.access == Access::read) continue;
        (use.semantic == IoSemantic::fragment_depth ? depth_export : sample_mask_export) = true;
    }
    if (!depth_export)
        throw std::runtime_error("depth export reflection is missing");
    if (sample_mask_export || plan.stage != Stage::fragment)
        throw std::runtime_error("unexpected export reflection");
    return take(emit_msl(structured, plan, {}));
}

id<MTLLibrary> compile(id<MTLDevice> device, const std::string& source) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                       options:options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    return library;
}

// One depth-only pass. The rasterized depth is always 0.0 (the fullscreen
// triangle carries z=0), so every case separates the exported value from the
// rasterized one.
int run_case(id<MTLDevice> device, id<MTL4CommandQueue> queue,
             id<MTLRenderPipelineState> state, id<MTLDepthStencilState> depth_state,
             float cleared, float expected, const char* name) {
    NSError* error = nil;
    auto descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                    width:extent height:extent mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    auto texture = [device newTextureWithDescriptor:descriptor];
    auto readback = [device newBufferWithLength:extent * extent * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    auto input = [device newBufferWithBytes:&exported_bits
                                     length:sizeof(exported_bits)
                                    options:MTLResourceStorageModeShared];
    // The runtime word-zero policy bounds every constant-buffer load through
    // a support table word per slot; slot 0 must cover the exported word.
    constexpr std::array<std::uint32_t, 18> support_sizes{4};
    auto support = [device newBufferWithBytes:support_sizes.data()
                                       length:sizeof(support_sizes)
                                      options:MTLResourceStorageModeShared];
    auto allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    auto table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 5;
    auto residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!texture || !readback || !input || !support || !allocator || !commands || !table || !residency) {
        std::fprintf(stderr, "%s: resource creation failed\n", name);
        return 1;
    }
    [table setAddress:input.gpuAddress atIndex:0];
    [table setAddress:support.gpuAddress atIndex:1];
    [residency addAllocation:texture];
    [residency addAllocation:readback];
    for (id<MTLAllocation> allocation in @[input, support]) [residency addAllocation:allocation];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    auto attachment = pass.depthAttachment;
    attachment.texture = texture;
    attachment.loadAction = MTLLoadActionClear;
    attachment.clearDepth = cleared;
    attachment.storeAction = MTLStoreActionStore;
    auto render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return 1;
    [render setRenderPipelineState:state];
    [render setDepthStencilState:depth_state];
    [render setArgumentTable:table atStages:MTLRenderStageFragment];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    auto copy = [commands computeCommandEncoder];
    if (!copy) return 1;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(extent, extent, 1)
                 toBuffer:readback destinationOffset:0
    destinationBytesPerRow:extent * sizeof(float) destinationBytesPerImage:0];
    [copy endEncoding];
    [commands endCommandBuffer];
    auto retained = @[texture, readback, input, support, table, residency, allocator, state, depth_state];
    auto completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
        std::fprintf(stderr, "%s: GPU execution timed out\n", name);
        return 1;
    }
    if (execution_error) {
        std::fprintf(stderr, "%s: GPU execution failed: %s\n", name,
                     execution_error.localizedDescription.UTF8String);
        return 1;
    }
    const auto* depths = static_cast<const float*>(readback.contents);
    int failures = 0;
    for (unsigned pixel = 0; pixel < extent * extent; ++pixel) {
        if (depths[pixel] != expected) {
            std::fprintf(stderr, "%s: pixel %u: got %.6f, expected %.6f\n", name, pixel,
                         depths[pixel], expected);
            ++failures;
            break;
        }
    }
    return failures;
}

id<MTLDepthStencilState> depth_state(id<MTLDevice> device, MTLCompareFunction compare) {
    auto* descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction = compare;
    descriptor.depthWriteEnabled = YES;
    return [device newDepthStencilStateWithDescriptor:descriptor];
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            auto device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            NSError* error = nil;
            auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
            auto queue = [device newMTL4CommandQueue];
            if (!compiler || !queue) throw std::runtime_error("Metal 4 compiler or queue unavailable");
            auto vertex_library = compile(device,
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "vertex float4 synthetic_vertex(uint id [[vertex_id]]) {\n"
                "  float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};\n"
                "  return float4(p[id], 0, 1);\n}\n");
            const auto fragment = translated_depth_fragment();
            auto fragment_library = compile(device, fragment.source);
            auto vertex = [MTL4LibraryFunctionDescriptor new];
            vertex.library = vertex_library;
            vertex.name = @"synthetic_vertex";
            auto function = [MTL4LibraryFunctionDescriptor new];
            function.library = fragment_library;
            function.name = [NSString stringWithUTF8String:fragment.entry_point.c_str()];
            MTL4RenderPipelineDescriptor* pipeline = [MTL4RenderPipelineDescriptor new];
            pipeline.vertexFunctionDescriptor = vertex;
            pipeline.fragmentFunctionDescriptor = function;
            pipeline.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            pipeline.rasterSampleCount = 1;
            auto state = [compiler newRenderPipelineStateWithDescriptor:pipeline
                                                    compilerTaskOptions:nil error:&error];
            if (!state) {
                throw std::runtime_error("depth export pipeline creation failed: " +
                                         std::string(error.localizedDescription.UTF8String));
            }
            int failures = 0;
            // Depth test ALWAYS with writes on: the exported 0.5 replaces both
            // the cleared value and the rasterized z of 0.0.
            failures += run_case(device, queue, state,
                                 depth_state(device, MTLCompareFunctionAlways),
                                 0.25f, 0.5f, "always write exports 0.5");
            // LESS accepts when the exported value passes and is written.
            failures += run_case(device, queue, state,
                                 depth_state(device, MTLCompareFunctionLess),
                                 0.75f, 0.5f, "less accepts exported 0.5");
            // LESS rejects on the exported value: a rasterized z of 0.0 would
            // have passed and overwritten the cleared 0.25.
            failures += run_case(device, queue, state,
                                 depth_state(device, MTLCompareFunctionLess),
                                 0.25f, 0.25f, "less rejects exported 0.5");
            std::printf("fragment depth export GPU cases, %d failed\n", failures);
            return failures == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Fragment depth GPU test failed: %s\n", error.what());
            return 1;
        }
    }
}
