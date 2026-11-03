// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

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

constexpr NSUInteger width = 4;
constexpr NSUInteger height = 4;
constexpr NSUInteger pixel_bytes = 4 * sizeof(std::uint32_t);
constexpr std::array<unsigned, 3> locations{0, 3, 7};
using ColorWords = std::array<std::uint32_t, 4>;
constexpr ColorWords integer_words{0xffffffffu, 0x80000000u, 0x7fc12345u, 0x01000001u};
constexpr ColorWords float_words{0x3f800000u, 0xbf800000u, 0x3e800000u, 0x40000000u};

template<class T>
T take(Result<T> result) {
    if (const auto* error = std::get_if<Diagnostic>(&result))
        throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
}

MslArtifact translated_fragment(bool implicit_exports) {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-fragment-output"};
            if (key.kind == FactKind::constant_buffer_bound_mask)
                return Known<FactValue>{std::uint32_t{1}, origin};
            if (key.kind == FactKind::constant_buffer_size)
                return Known<FactValue>{std::uint32_t{64}, origin};
            if (key.kind == FactKind::fragment_output_type)
                return Known<FactValue>{IoType{ScalarType::u32, 4}, origin};
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
    // Dynamic LDC input prevents folding from bypassing the guest word path.
    // Sparse MRT locations also check that constant indices are locations,
    // rather than indices into the compact output reflection vector.
    std::uint32_t output_mask = 0;
    for (unsigned target = 0; target < locations.size(); ++target) {
        output_mask |= 15u << (locations[target] * 4);
        for (unsigned lane = 0; lane < 4; ++lane) {
            const unsigned offset = (target * 4 + lane) * 4;
            const unsigned destination = implicit_exports ? target * 4 + lane : 0;
            instruction(0xef9400000007ff00ull | destination | (std::uint64_t{offset} << 20));
            if (!implicit_exports) {
                const unsigned attribute = 0x80 + locations[target] * 16 + lane * 4;
                instruction(0xeff000000007ff00ull | (std::uint64_t{attribute} << 20));
            }
        }
    }
    if (implicit_exports) {
        // EXIT exports R0..R3, R4..R7 and R8..R11 to the active targets.
        // Disabled targets consume no GPR group, despite their location gaps.
        for (unsigned byte = 0; byte < 4; ++byte)
            image.sph[0x48 + byte] = std::byte((output_mask >> (byte * 8)) & 255);
    }
    instruction(0xe30000000007000full);
    FactReader facts(source);
    const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
    auto decoded = take(decode(image, facts, {}));
    auto registers = take(emit_ir(decoded, facts, {}));
    auto ssa = take(construct_ssa(std::move(registers), {}));
    auto legal = take(legalize(std::move(ssa), facts, profile));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile));
    EmitOptions options;
    options.fragment_output_types_from_render_targets = true;
    return take(emit_msl(structured, plan, options));
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

struct Case {
    const char* name;
    std::array<std::uint32_t, 3> classes;
};

int run_case(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue,
             id<MTLLibrary> vertex_library, id<MTLLibrary> fragment_library,
             const Case& item) {
    MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = vertex_library;
    vertex.name = @"synthetic_vertex";
    MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = fragment_library;
    fragment.name = @"armigo_fragment_main";
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (unsigned target = 0; target < locations.size(); ++target)
        [constants setConstantValue:&item.classes[target] type:MTLDataTypeUInt atIndex:locations[target]];
    MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
    specialized.functionDescriptor = fragment;
    specialized.constantValues = constants;
    MTL4RenderPipelineDescriptor* pipeline = [MTL4RenderPipelineDescriptor new];
    pipeline.vertexFunctionDescriptor = vertex;
    pipeline.fragmentFunctionDescriptor = specialized;
    pipeline.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    constexpr std::array<MTLPixelFormat, 3> formats{
        MTLPixelFormatRGBA32Float, MTLPixelFormatRGBA32Uint, MTLPixelFormatRGBA32Sint};
    for (unsigned target = 0; target < locations.size(); ++target)
        pipeline.colorAttachments[locations[target]].pixelFormat = formats[item.classes[target]];
    NSError* error = nil;
    auto state = [compiler newRenderPipelineStateWithDescriptor:pipeline compilerTaskOptions:nil error:&error];
    if (!state) {
        std::fprintf(stderr, "%s: pipeline creation failed: %s\n", item.name,
                     error.localizedDescription.UTF8String);
        return 1;
    }

    std::array<ColorWords, 4> inputs{};
    for (unsigned target = 0; target < locations.size(); ++target) {
        const auto& words = item.classes[target] == 0 ? float_words : integer_words;
        // Distinct lane order per target exposes accidental reuse of another
        // compact register group even when every attachment has the same type.
        for (unsigned lane = 0; lane < 4; ++lane)
            inputs[target][lane] = words[(lane + target) % 4];
    }
    auto input = [device newBufferWithBytes:inputs.data() length:sizeof(inputs)
                                    options:MTLResourceStorageModeShared];
    std::array<std::uint32_t, 18> buffer_sizes{};
    buffer_sizes[0] = sizeof(inputs);
    auto sizes = [device newBufferWithBytes:buffer_sizes.data() length:sizeof(buffer_sizes)
                                    options:MTLResourceStorageModeShared];
    auto readback = [device newBufferWithLength:locations.size() * width * height * pixel_bytes
                                        options:MTLResourceStorageModeShared];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    auto table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 6;
    auto residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    auto allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    if (!input || !sizes || !readback || !table || !residency || !allocator || !commands) {
        std::fprintf(stderr, "%s: resource creation failed\n", item.name);
        return 1;
    }
    [table setAddress:input.gpuAddress atIndex:0];
    [table setAddress:sizes.gpuAddress atIndex:1];
    NSMutableArray<id<MTLTexture>>* targets = [NSMutableArray new];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    for (unsigned target = 0; target < locations.size(); ++target) {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[item.classes[target]]
            width:width height:height mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget;
        auto texture = [device newTextureWithDescriptor:descriptor];
        if (!texture) {
            std::fprintf(stderr, "%s: texture creation failed\n", item.name);
            return 1;
        }
        [targets addObject:texture];
        [residency addAllocation:texture];
        auto attachment = pass.colorAttachments[locations[target]];
        attachment.texture = texture;
        attachment.loadAction = MTLLoadActionClear;
        attachment.storeAction = MTLStoreActionStore;
        attachment.clearColor = MTLClearColorMake(0, 0, 0, 0);
    }
    for (id<MTLAllocation> allocation in @[input, sizes, readback]) [residency addAllocation:allocation];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    auto render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return 1;
    [render setRenderPipelineState:state];
    [render setArgumentTable:table atStages:MTLRenderStageFragment];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    auto copy = [commands computeCommandEncoder];
    if (!copy) return 1;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
    for (unsigned target = 0; target < locations.size(); ++target)
        [copy copyFromTexture:targets[target] sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                 toBuffer:readback destinationOffset:target * width * height * pixel_bytes
    destinationBytesPerRow:width * pixel_bytes destinationBytesPerImage:0];
    [copy endEncoding];
    [commands endCommandBuffer];
    auto retained = @[vertex_library, fragment_library, state, input, sizes, readback,
                      targets, table, residency, allocator];
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
        std::fprintf(stderr, "%s: GPU execution timed out\n", item.name);
        return 1;
    }
    if (execution_error) {
        std::fprintf(stderr, "%s: GPU execution failed: %s\n", item.name,
                     execution_error.localizedDescription.UTF8String);
        return 1;
    }
    const auto* pixels = static_cast<const std::uint32_t*>(readback.contents);
    for (unsigned target = 0; target < locations.size(); ++target) {
        for (unsigned pixel = 0; pixel < width * height; ++pixel) {
            for (unsigned lane = 0; lane < 4; ++lane) {
                const auto actual = pixels[(target * width * height + pixel) * 4 + lane];
                const auto expected = inputs[target][lane];
                if (actual != expected) {
                    std::fprintf(stderr, "%s: target %u pixel %u lane %u: got 0x%08x, expected 0x%08x\n",
                                 item.name, locations[target], pixel, lane, actual, expected);
                    return 1;
                }
            }
        }
    }
    return 0;
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
            // Reuse the identical translated fragment library across numeric
            // classes. Mixed layouts and a return to uint exercise repeated
            // specialization without translating a new shader.
            constexpr std::array cases{
                Case{"uint words", {1, 1, 1}}, Case{"signed words", {2, 2, 2}},
                Case{"float baseline", {0, 0, 0}}, Case{"mixed MRT", {1, 0, 2}},
                Case{"changed MRT classes", {2, 1, 0}}, Case{"uint repeat", {1, 1, 1}}};
            int failures = 0;
            for (const bool implicit_exports : {false, true}) {
                const auto fragment = translated_fragment(implicit_exports);
                auto fragment_library = compile(device, fragment.source);
                for (const auto& item : cases) {
                    const std::string name = std::string(implicit_exports ? "SPH/EXIT " : "AST ") + item.name;
                    const Case named{name.c_str(), item.classes};
                    failures += run_case(device, compiler, queue, vertex_library, fragment_library, named);
                }
            }
            std::printf("%zu translated fragment output GPU cases, %d failed\n", cases.size() * 2, failures);
            return failures == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Fragment output GPU test failed: %s\n", error.what());
            return 1;
        }
    }
}
