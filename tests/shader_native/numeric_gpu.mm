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
#include "fixture.h"
#include <array>
#include <cstdio>
#include <cstring>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
id<MTLLibrary> library(id<MTLDevice> device, const std::string& source) {
    MTLCompileOptions* options=[MTLCompileOptions new]; options.languageVersion=MTLLanguageVersion4_0; options.mathMode=MTLMathModeSafe;
    NSError* error=nil;
    auto result=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if(!result) throw std::runtime_error(error.localizedDescription.UTF8String);
    return result;
}
sn::Instruction load(unsigned destination,unsigned offset) {
    auto i=raw(0xa02);field(i,16,8,destination);field(i,54,5,3);field(i,38,16,offset);i.words[2]=0xf00;return i;
}
void run(id<MTLDevice> device, const Fixture& fixture, std::array<std::uint32_t,3> input, std::uint32_t expected) {
    auto translated=sn::translate_native(fixture.program());
    if(auto error=std::get_if<ir::Diagnostic>(&translated))throw std::runtime_error(error->message);
    const auto& artifact=std::get<ir::MslArtifact>(translated);
    auto fs=library(device,artifact.source);
    auto vs=library(device,R"(#include <metal_stdlib>
using namespace metal;
struct VertexResult { float4 position [[position]]; };
vertex VertexResult native_test_vertex(uint i [[vertex_id]]) {
    const float2 p[3]={float2(-2,-2),float2(6,-2),float2(-2,6)};
    VertexResult out; out.position=float4(p[i],0,1); return out;
})");
    NSError* error=nil;
    auto compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    check(compiler!=nil,"Metal 4 compiler");
    MTL4LibraryFunctionDescriptor* vertex=[MTL4LibraryFunctionDescriptor new];vertex.library=vs;vertex.name=@"native_test_vertex";
    MTL4LibraryFunctionDescriptor* fragment=[MTL4LibraryFunctionDescriptor new];fragment.library=fs;fragment.name=@"armigo_fragment_main";
    MTLFunctionConstantValues* constants=[MTLFunctionConstantValues new];std::uint32_t color_type=1;
    [constants setConstantValue:&color_type type:MTLDataTypeUInt atIndex:0];
    MTL4SpecializedFunctionDescriptor* specialized=[MTL4SpecializedFunctionDescriptor new];specialized.functionDescriptor=fragment;specialized.constantValues=constants;
    MTL4RenderPipelineDescriptor* descriptor=[MTL4RenderPipelineDescriptor new];descriptor.vertexFunctionDescriptor=vertex;descriptor.fragmentFunctionDescriptor=specialized;
    descriptor.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;descriptor.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA32Uint;
    auto pipeline=[compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto data=[device newBufferWithLength:65536 options:MTLResourceStorageModeShared];std::memcpy(data.contents,input.data(),sizeof(input));
    std::array<std::uint32_t,18> sizes{};sizes[3]=sizeof(input);
    auto bounds=[device newBufferWithBytes:sizes.data() length:sizeof(sizes) options:MTLResourceStorageModeShared];
    auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint width:1 height:1 mipmapped:NO];td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
    auto target=[device newTextureWithDescriptor:td];auto readback=[device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    MTL4ArgumentTableDescriptor* at=[MTL4ArgumentTableDescriptor new];at.maxBufferBindCount=31;
    auto table=[device newArgumentTableWithDescriptor:at error:&error];
    for(const auto& resource:artifact.reflection.resources) {
        const auto& range=std::get<ir::MetalRange>(resource.location);
        const auto& buffer=std::get<ir::BufferUse>(resource.use);
        auto address=std::holds_alternative<ir::ConstantBufferSource>(buffer.source)?data.gpuAddress:bounds.gpuAddress;
        [table setAddress:address atIndex:range.first];
    }
    auto residency=[device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    for(id<MTLAllocation> allocation in @[data,bounds,target,readback])[residency addAllocation:allocation];[residency commit];
    auto allocator=[device newCommandAllocator];auto commands=[device newCommandBuffer];auto queue=[device newMTL4CommandQueue];
    check(data&&bounds&&target&&readback&&table&&residency&&allocator&&commands&&queue,"Metal resources");
    [commands beginCommandBufferWithAllocator:allocator];[commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    auto render=[commands renderCommandEncoderWithDescriptor:pass];[render setRenderPipelineState:pipeline];[render setArgumentTable:table atStages:MTLRenderStageFragment];[render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[render endEncoding];
    auto copy=[commands computeCommandEncoder];[copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1,1,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:16 destinationBytesPerImage:0];[copy endEncoding];[commands endCommandBuffer];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);__block NSError* execution=nil;
    MTL4CommitOptions* commit=[MTL4CommitOptions new];[commit addFeedbackHandler:^(id<MTL4CommitFeedback> f){execution=f.error;dispatch_semaphore_signal(done);}];
    id<MTL4CommandBuffer> submitted[]={commands};[queue commit:submitted count:1 options:commit];
    check(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC))==0,"GPU completion");
    if(execution)throw std::runtime_error(execution.localizedDescription.UTF8String);
    const auto got=*static_cast<const std::uint32_t*>(readback.contents);
    if(got!=expected){std::fprintf(stderr,"expected %08x, got %08x\n",expected,got);throw std::runtime_error("native numeric GPU result");}
}
}
int main() { @autoreleasepool { try {
    auto device=MTLCreateSystemDefaultDevice();check(device!=nil,"Metal device");
    auto mul=raw(0x220);field(mul,24,8,1);field(mul,32,8,2);field(mul,84,3,4);
    Fixture rejected({load(1,0),load(2,4),mul,exit_instruction()});
    check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(rejected.program())),"non-FTZ cbuf multiplication must not produce executable MSL");
    field(mul,80,1,1);
    Fixture ftz({load(1,0),load(2,4),mul,exit_instruction()});run(device,ftz,{1,0x40000000,0},0);
    auto add=raw(0x221);field(add,24,8,0);field(add,32,8,3);field(add,80,1,1);
    Fixture separate({load(1,0),load(2,4),load(3,8),mul,add,exit_instruction()});
    run(device,separate,{0x3f800001,0x3f7ffffe,0xbf800000},0);
    auto fma=raw(0x223);field(fma,24,8,1);field(fma,32,8,2);field(fma,64,8,3);field(fma,80,1,1);
    Fixture fused({load(1,0),load(2,4),load(3,8),fma,exit_instruction()});run(device,fused,{0x3f800001,0x3f7ffffe,0xbf800000},0xa8800000);
    for(unsigned selector:{7u,8u}) {
        auto cmp=raw(0x20b);field(cmp,24,8,1);field(cmp,32,8,2);field(cmp,76,4,selector);field(cmp,80,1,1);field(cmp,81,3,0);field(cmp,84,3,7);field(cmp,87,4,7);
        Fixture test({load(1,0),load(2,4),move(0,0),cmp,move(0,1,0),exit_instruction()});
        run(device,test,{0x7fc00001,0,0},selector==8?1:0);run(device,test,{0x3f800000,0,0},selector==7?1:0);
    }
    std::puts("Native numeric Metal execution passed");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;} } }
