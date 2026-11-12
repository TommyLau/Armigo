// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../shader/pipeline/early_fragment_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using namespace armigo::shader::test::early_fragment;
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
                                   id<MTLLibrary> fragment, NSString* function) {
    auto vs = [MTL4LibraryFunctionDescriptor new]; vs.library = vertex; vs.name = @"synthetic_vertex";
    auto fs = [MTL4LibraryFunctionDescriptor new]; fs.library = fragment; fs.name = function;
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vs; descriptor.fragmentFunctionDescriptor = fs;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
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
fragment float4 synthetic_depth() { return float4(0,1,0,1); }
)metal");
    const auto artifact = translate();
    const auto fragment = library(device, artifact.source);
    auto forced_source = artifact.source;
    if (forced_source.find("[[early_fragment_tests]]") == std::string::npos) {
        const auto at = forced_source.find("fragment ArmigoOutput armigo_fragment_main");
        check(at != std::string::npos, "Fragment entry point exists");
        forced_source.insert(at, "[[early_fragment_tests]] ");
    }
    const auto forced = library(device, forced_source);
    const auto depth_pipeline = pipeline(compiler, vertex, vertex, @"synthetic_depth");
    const auto read_pipeline = pipeline(compiler, vertex, fragment, @"armigo_fragment_main");
    const auto forced_pipeline = pipeline(compiler, vertex, forced, @"armigo_fragment_main");
    auto ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionLess; ds.depthWriteEnabled = YES;
    const auto depth_state = [device newDepthStencilStateWithDescriptor:ds];
    ds.depthCompareFunction = MTLCompareFunctionEqual; ds.depthWriteEnabled = NO;
    const auto equal_state = [device newDepthStencilStateWithDescriptor:ds];
    auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:width height:height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    const auto stored = [device newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    const auto mask = [device newTextureWithDescriptor:td], color = [device newTextureWithDescriptor:td];
    const auto readback = [device newBufferWithLength:width*height*16*3 options:MTLResourceStorageModeShared];
    const std::array<std::uint32_t,16> green{0,0x3f800000u,0,0x3f800000u,0,0x3f800000u,0,0x3f800000u,
        0,0x3f800000u,0,0x3f800000u,0,0x3f800000u,0,0x3f800000u};
    const auto payload = [device newBufferWithBytes:green.data() length:sizeof green options:MTLResourceStorageModeShared];
    std::vector<std::uint32_t> table(guest_memory_bytes/4,0); table[0]=2;
    const std::uint64_t base=0x100000000ull, end=base+sizeof green, delta=payload.gpuAddress-base;
    for (auto item : {std::pair{0u,base},std::pair{2u,end},std::pair{4u,delta}}) {
        table[guest_memory_header_words+guest_memory_entry_words+item.first]=std::uint32_t(item.second);
        table[guest_memory_header_words+guest_memory_entry_words+item.first+1]=std::uint32_t(item.second>>32);
    }
    const auto guest = [device newBufferWithBytes:table.data() length:table.size()*4 options:MTLResourceStorageModeShared];
    auto table_descriptor = [MTL4ArgumentTableDescriptor new]; table_descriptor.maxBufferBindCount=31;
    const auto fragment_table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    check(artifact.reflection.resources.size()==1, "Raw load uses exactly one guest-memory table");
    for (const auto& resource : artifact.reflection.resources) {
        const auto* b=std::get_if<BufferUse>(&resource.use);
        check(b && std::holds_alternative<SupportBufferSource>(b->source) &&
            std::get<SupportBufferSource>(b->source).schema==guest_memory_schema,"Raw diagnostic table is reflected");
        [fragment_table setAddress:guest.gpuAddress atIndex:std::get<MetalRange>(resource.location).first];
    }
    std::mt19937 rng(0xc0ffeeu); std::uniform_real_distribution<float> unit(0, 1);
    unsigned long covered_total = 0, holes_total = 0;
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
        const float zero=0;
        const auto shift=[device newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];
        const auto vertex_table=[device newArgumentTableWithDescriptor:table_descriptor error:&error];
        [vertex_table setAddress:vertices.gpuAddress atIndex:0]; [vertex_table setAddress:shift.gpuAddress atIndex:1];
        const auto residency=[device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> a in @[vertices,shift,stored,mask,color,readback,payload,guest]) [residency addAllocation:a];
        [residency commit];
        const auto allocator=[device newCommandAllocator]; const auto commands=[device newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        const auto copy_color = [&](id<MTLTexture> source, unsigned offset) {
            auto copy=[commands computeCommandEncoder];
            [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
            [copy copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                sourceSize:MTLSizeMake(width,height,1) toBuffer:readback destinationOffset:offset
                destinationBytesPerRow:width*16 destinationBytesPerImage:width*height*16];
            [copy barrierAfterQueueStages:MTLStageFragment | MTLStageBlit beforeStages:MTLStageVertex visibilityOptions:MTL4VisibilityOptionDevice];
            [copy endEncoding];
        };
        auto pass=[MTL4RenderPassDescriptor new];
        pass.depthAttachment.texture=stored; pass.depthAttachment.loadAction=MTLLoadActionClear;
        pass.depthAttachment.storeAction=MTLStoreActionStore; pass.depthAttachment.clearDepth=1;
        pass.colorAttachments[0].texture=mask; pass.colorAttachments[0].loadAction=MTLLoadActionClear;
        pass.colorAttachments[0].storeAction=MTLStoreActionStore; pass.colorAttachments[0].clearColor=MTLClearColorMake(1,0,0,1);
        auto render=[commands renderCommandEncoderWithDescriptor:pass];
        [render setRenderPipelineState:depth_pipeline]; [render setDepthStencilState:depth_state];
        [render setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:triangles*3]; [render endEncoding];
        copy_color(mask,0);
        for (unsigned control=0;control<2;++control) {
            pass=[MTL4RenderPassDescriptor new];
            pass.depthAttachment.texture=stored; pass.depthAttachment.loadAction=MTLLoadActionLoad; pass.depthAttachment.storeAction=MTLStoreActionStore;
            pass.colorAttachments[0].texture=color; pass.colorAttachments[0].loadAction=MTLLoadActionClear;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore; pass.colorAttachments[0].clearColor=MTLClearColorMake(1,0,0,1);
            render=[commands renderCommandEncoderWithDescriptor:pass];
            [render setRenderPipelineState:control?forced_pipeline:read_pipeline]; [render setDepthStencilState:equal_state];
            [render setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
            [render setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:triangles*3]; [render endEncoding];
            copy_color(color,(control+1)*width*height*16);
        }
        [commands endCommandBuffer]; auto done=dispatch_semaphore_create(0); __block NSError* failure=nil;
        auto commit=[MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) { failure=feedback.error; dispatch_semaphore_signal(done); }];
        id<MTL4CommandBuffer> batch[]{commands}; [queue commit:batch count:1 options:commit];
        check(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,20*NSEC_PER_SEC))==0,"GPU completion timeout");
        if(failure)throw std::runtime_error(failure.localizedDescription.UTF8String);
        const auto* pixels=static_cast<const float*>(readback.contents); unsigned covered=0,lost[2]{};
        for(unsigned i=0;i<width*height;++i)if(pixels[i*4+1]>.5f){++covered;for(unsigned c=0;c<2;++c)lost[c]+=pixels[(c+1)*width*height*4+i*4+1]<.5f;}
        std::printf("equal mode=%u covered=%u production_holes=%u forced_early_holes=%u guest_misses=%u\n",mode,covered,lost[0],lost[1],static_cast<const unsigned*>(guest.contents)[1]);
        check(covered>100000,"Triangles cover enough pixels"); check(lost[1]==0,"Forced early control preserves the prepass coverage");
        check(static_cast<const unsigned*>(guest.contents)[1]==0,"All guest accesses stay inside their valid pool");
        covered_total+=covered; holes_total+=lost[0];
    }
    check(holes_total==0,"Read-only diagnostic loads must preserve depth-EQUAL prepass coverage");
    check(artifact.source.find("[[early_fragment_tests]]")!=std::string::npos,"Read-only fragments explicitly preserve early tests");
    for (const auto behaviour : {Behaviour::store,Behaviour::atomic,Behaviour::discard,Behaviour::depth,Behaviour::sample_mask})
        check(translate(behaviour).source.find("[[early_fragment_tests]]")==std::string::npos,
            "Guest side effects and depth/sample coverage exports retain their test order");

    std::printf("Early depth-EQUAL comparisons passed (%lu pixels)\n", covered_total); return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; } } }
