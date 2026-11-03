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

#include "draw_encoding.h"
#include <armigo/graphics/resource_memory_registry.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
using namespace armigo::graphics;
using armigo::platform::metal::select_native_draw;
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); } } while (false)

bool ok(const RendererStatus& status) { return std::holds_alternative<std::monostate>(status); }
template<class T> T require(RendererResult<T> result) {
    if (auto* error = std::get_if<RendererError>(&result)) {
        const auto command = error->command();
        std::fprintf(stderr, "Fixture failed: %.*s\n", int(command.detail.size()), command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}
struct Backend final : ResourceMemoryBackend {
    std::vector<std::byte> bytes = std::vector<std::byte>(4096);
    unsigned reads = 0;
    RendererStatus read_cpu(BackingRange range, std::span<std::byte> destination) override {
        ++reads;
        if (range.offset > bytes.size() || destination.size() > bytes.size() - range.offset)
            return RendererError({CommandStatus::MemoryFault, "Synthetic read exceeds allocation"});
        std::copy_n(bytes.data() + range.offset, destination.size(), destination.data());
        return renderer_success();
    }
    RendererStatus write_cpu(BackingRange range, std::span<const std::byte> source) override {
        if (range.offset > bytes.size() || source.size() > bytes.size() - range.offset)
            return RendererError({CommandStatus::MemoryFault, "Synthetic write exceeds allocation"});
        std::copy(source.begin(), source.end(), bytes.begin() + range.offset);
        return renderer_success();
    }
    RendererStatus wait_gpu(CompletionToken) override { return renderer_success(); }
};
struct Fixture {
    std::shared_ptr<Backend> backend = std::make_shared<Backend>();
    std::shared_ptr<ResourceMemoryRegistry> memory = require(make_resource_memory(
        {1, backend, CoherenceMode::SubmissionBoundary, 1}));
    RenderStop stop{};
    Fixture() {
        CHECK(ok(memory->register_backing({{1,1,1},4096,std::make_shared<ResourceLease>()})));
        CHECK(ok(memory->register_mapping({{1,1,1},{1,1,1},
            {0x1000,4096,0,0x8000,4096,0x10}})));
        stop.kind = RenderKind::Draw;
        stop.engine = engine_class::threed;
        stop.registers[0x5f3] = 0x1000;
        stop.registers[0x5f5] = 0x1fff;
    }
    DrawPlan make(std::initializer_list<std::uint32_t> indices, IndexFormat format,
                  std::uint32_t topology = 4, std::int32_t base = 0,
                  bool restart = false, std::uint32_t sentinel = 0xffffffffu) {
        const auto size = format == IndexFormat::U8 ? 1 : format == IndexFormat::U16 ? 2 : 4;
        std::size_t cursor = 0;
        for (auto value : indices) for (int i = 0; i < size; ++i)
            backend->bytes[cursor++] = std::byte((value >> (i*8)) & 255);
        stop.registers[0x5f6] = std::uint32_t(format);
        stop.registers[0x591] = restart;
        stop.registers[0x592] = sentinel;
        stop.draw = DrawDescription{true, topology, 0, std::uint32_t(indices.size()), base, 7, 1};
        return require(assemble_draw(stop, {}, memory->address_space()));
    }
};

std::vector<std::uint32_t> decode(const DrawPlan& plan) {
    const auto size = plan.indices->encoded_format == IndexFormat::U16 ? 2 : 4;
    const auto& bytes = plan.indices->converted;
    std::vector<std::uint32_t> values(bytes.size()/size);
    for (std::size_t i=0; i<values.size(); ++i) for (int j=0; j<size; ++j)
        values[i] |= std::to_integer<std::uint32_t>(bytes[i*size+j]) << (j*8);
    return values;
}

void unit_tests() {
    Fixture fixture;
    auto plan = fixture.make({10,11,12,255,10,11,12},IndexFormat::U8,5,-10,true,255);
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.indices->encoded_format == IndexFormat::U16);
    CHECK(plan.indices->restart_enabled && plan.segments.size()==1);
    CHECK(plan.segments[0].base_vertex==-10);
    CHECK((decode(plan)==std::vector<std::uint32_t>{10,11,12,65535,10,11,12}));

    plan=fixture.make({65533,65534,65535},IndexFormat::U16,4,-65533);
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.indices->encoded_format==IndexFormat::U32);
    CHECK(plan.segments[0].base_vertex==-65533 && decode(plan)[2]==65535);

    plan=fixture.make({0xfffffffd,0xfffffffe,0xffffffff},IndexFormat::U32);
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    const auto shifted=decode(plan);
    CHECK(std::uint64_t(shifted[2])+plan.segments[0].base_vertex==0xffffffffull);
    CHECK(shifted[2]!=0xffffffffu);

    plan=fixture.make({0,1,0xffffffff},IndexFormat::U32);
    const auto old_bytes=plan.indices->converted;
    CHECK(!ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.indices->converted==old_bytes && plan.segments[0].base_vertex==0);

    plan=fixture.make({0xfffffffe,0xffffffff,1},IndexFormat::U32,4,1);
    auto status=select_native_draw(plan,fixture.stop,*fixture.memory);
    CHECK(!ok(status) && std::get<RendererError>(status).command().status==CommandStatus::MemoryFault);

    plan=fixture.make({2,3,4,5},IndexFormat::U16,6,10);
    const auto normalized=plan.indices->converted;
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.indices->converted==normalized && plan.segments[0].base_vertex==0);

    plan=fixture.make({1,2,3,99,4,5,6},IndexFormat::U16,4,0,true,99);
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.segments.size()==2 && !plan.indices->restart_enabled);
    CHECK(plan.segments[1].first==4);

    plan=fixture.make({65535,65535},IndexFormat::U16,5,0,true,65535);
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.segments.empty() && !plan.indices);

    plan=fixture.make({0,1,100},IndexFormat::U16);
    VertexAttributePlan attribute{};
    attribute.format.size_bytes=4;
    plan.vertex.attributes.push_back(attribute);
    VertexStreamPlan stream{};
    stream.address=0x2000; stream.inclusive_limit=0x200f; stream.stride=4;
    plan.vertex.streams.push_back(stream);
    plan.vertex.steps.push_back({0,VertexStep::PerVertex,1});
    // Guest U16 indices whose 0xFFFF sentinel is no in-range vertex stay in
    // guest memory unread; an index past the stream limit is not rejected.
    CHECK(ok(select_native_draw(plan,fixture.stop,*fixture.memory)));
    CHECK(plan.indices->converted.empty() && plan.indices->encoded_format==IndexFormat::U16);

    plan=fixture.make({0,1,2},IndexFormat::U16);
    auto retained=require(fixture.memory->retain(0x1000,6,ResourceAccess::ReadWrite));
    auto write=require(fixture.memory->prepare_cpu_write(retained.bytes));
    status=select_native_draw(plan,fixture.stop,*fixture.memory);
    CHECK(!ok(status) && std::get<RendererError>(status).command().status==CommandStatus::Stopped);
}

bool gpu_draw(id<MTLDevice> device, const DrawPlan& plan, std::uint32_t origin,
              std::uint32_t expected_seen, bool square = false, bool covered = true) {
    static NSString* source=@R"msl(
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; };
vertex Out vertex_main(uint id [[vertex_id]], uint instance [[instance_id]],
                       uint base [[base_instance]], constant uint2& controls [[buffer(0)]],
                       device atomic_uint* seen [[buffer(1)]]) {
    uint local = id-controls.x;
    atomic_fetch_or_explicit(seen, local<8 ? 1u<<local : 0x80000000u, memory_order_relaxed);
    if (instance-base != 0) atomic_fetch_or_explicit(seen,0x40000000u,memory_order_relaxed);
    float2 p = local%3==0 ? float2(-1,-1) : local%3==1 ? float2(3,-1) : float2(-1,3);
    if (controls.y) p = float2(local&1 ? 1 : -1, local&2 ? 1 : -1);
    return {float4(p,0,1)};
}
fragment float4 fragment_main() { return float4(1,0,0,1); }
)msl";
    NSError* error=nil;
    MTLCompileOptions* compile=[MTLCompileOptions new];
    compile.languageVersion=MTLLanguageVersion4_0;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:compile error:&error];
    id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    MTL4LibraryFunctionDescriptor* vertex=[MTL4LibraryFunctionDescriptor new];
    vertex.library=library; vertex.name=@"vertex_main";
    MTL4LibraryFunctionDescriptor* fragment=[MTL4LibraryFunctionDescriptor new];
    fragment.library=library; fragment.name=@"fragment_main";
    MTL4RenderPipelineDescriptor* pipeline_desc=[MTL4RenderPipelineDescriptor new];
    pipeline_desc.vertexFunctionDescriptor=vertex; pipeline_desc.fragmentFunctionDescriptor=fragment;
    pipeline_desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
    pipeline_desc.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;
    id<MTLRenderPipelineState> pipeline=[compiler newRenderPipelineStateWithDescriptor:pipeline_desc
        compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr,"Pipeline failed: %s\n",error.localizedDescription.UTF8String); return false;
    }
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget;
    id<MTLTexture> target=[device newTextureWithDescriptor:td];
    id<MTLBuffer> input=[device newBufferWithBytes:plan.indices->converted.data()
        length:plan.indices->converted.size() options:MTLResourceStorageModeShared];
    const std::array<std::uint32_t,2> controls{origin,std::uint32_t(square)};
    id<MTLBuffer> origin_buffer=[device newBufferWithBytes:controls.data() length:8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> seen=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    std::memset(seen.contents,0,4);
    MTL4ArgumentTableDescriptor* table_desc=[MTL4ArgumentTableDescriptor new];
    table_desc.maxBufferBindCount=2;
    id<MTL4ArgumentTable> table=[device newArgumentTableWithDescriptor:table_desc error:&error];
    [table setAddress:origin_buffer.gpuAddress atIndex:0];
    [table setAddress:seen.gpuAddress atIndex:1];
    id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    [residency addAllocation:input]; [residency addAllocation:origin_buffer];
    [residency addAllocation:seen]; [residency addAllocation:target]; [residency commit];
    id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
    id<MTL4CommandBuffer> commands=[device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass=[MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture=target;
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    pass.colorAttachments[0].clearColor=MTLClearColorMake(0,0,0,0);
    id<MTL4RenderCommandEncoder> encoder=[commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline]; [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
    if (square) { [encoder setCullMode:MTLCullModeBack]; [encoder setFrontFacingWinding:MTLWindingCounterClockwise]; }
    const auto size=plan.indices->encoded_format==IndexFormat::U16?2u:4u;
    for (const auto& segment:plan.segments) {
        [encoder drawIndexedPrimitives:plan.topology==PrimitiveTopology::TriangleStrip?
            MTLPrimitiveTypeTriangleStrip:MTLPrimitiveTypeTriangle indexCount:segment.count
            indexType:size==2?MTLIndexTypeUInt16:MTLIndexTypeUInt32
            indexBuffer:input.gpuAddress+segment.first*size indexBufferLength:segment.count*size
            instanceCount:segment.instance_count baseVertex:segment.base_vertex baseInstance:segment.base_instance];
    }
    [encoder endEncoding]; [commands endCommandBuffer];
    NSArray* retained=@[library,compiler,pipeline,input,origin_buffer,seen,target,table,residency,queue,allocator,commands];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block NSError* feedback_error=nil;
    MTL4CommitOptions* options=[MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        (void)retained.count; feedback_error=feedback.error; dispatch_semaphore_signal(done);
    }];
    id<MTL4CommandBuffer> buffers[]={commands}; [queue commit:buffers count:1 options:options];
    if (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC))) {
        std::fputs("Native index test timed out\n",stderr); std::abort();
    }
    if (feedback_error) {
        std::fprintf(stderr,"GPU failed: %s\n",feedback_error.localizedDescription.UTF8String); return false;
    }
    std::array<std::uint8_t,64> pixels{};
    [target getBytes:pixels.data() bytesPerRow:16 fromRegion:MTLRegionMake2D(0,0,4,4) mipmapLevel:0];
    const auto observed=*static_cast<const std::uint32_t*>(seen.contents);
    if (observed!=expected_seen) std::fprintf(stderr,"Observed vertex IDs: %08x, expected %08x\n",observed,expected_seen);
    bool coverage=true;
    for (std::size_t i=0; i<pixels.size(); i+=4)
        coverage &= pixels[i]==(covered?255:0) && pixels[i+1]==0 && pixels[i+3]==(covered?255:0);
    return observed==expected_seen && coverage;
}

void gpu_instance_step(id<MTLDevice> device) {
    static NSString* source=@R"msl(
#include <metal_stdlib>
using namespace metal;
struct Input { uint fetched [[attribute(0)]]; };
struct Output { float4 position [[position]]; };
vertex Output instance_vertex(Input in [[stage_in]], uint vertex_index [[vertex_id]],
                              uint instance [[instance_id]], uint base [[base_instance]],
                              device atomic_uint* observed [[buffer(0)]]) {
    atomic_store_explicit(observed + instance-base, in.fetched, memory_order_relaxed);
    float2 p = vertex_index==0 ? float2(-1,-1) : vertex_index==1 ? float2(3,-1) : float2(-1,3);
    return {float4(p,0,1)};
}
fragment float4 instance_fragment() { return float4(1,0,0,1); }
)msl";
    NSError* error=nil;
    MTLCompileOptions* options=[MTLCompileOptions new];
    options.languageVersion=MTLLanguageVersion4_0;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        std::fprintf(stderr,"Instance probe shader failed: %s\n",error.localizedDescription.UTF8String);
        ++failures; return;
    }
    id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    MTL4LibraryFunctionDescriptor* vertex=[MTL4LibraryFunctionDescriptor new];
    vertex.library=library; vertex.name=@"instance_vertex";
    MTL4LibraryFunctionDescriptor* fragment=[MTL4LibraryFunctionDescriptor new];
    fragment.library=library; fragment.name=@"instance_fragment";
    MTLVertexDescriptor* layout=[MTLVertexDescriptor vertexDescriptor];
    layout.attributes[0].format=MTLVertexFormatUInt;
    layout.attributes[0].bufferIndex=1;
    layout.layouts[1].stride=sizeof(std::uint32_t);
    layout.layouts[1].stepFunction=MTLVertexStepFunctionPerInstance;
    layout.layouts[1].stepRate=3;
    MTL4RenderPipelineDescriptor* descriptor=[MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor=vertex; descriptor.fragmentFunctionDescriptor=fragment;
    descriptor.vertexDescriptor=layout;
    descriptor.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;
    descriptor.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline=[compiler newRenderPipelineStateWithDescriptor:descriptor
        compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr,"Instance probe pipeline failed: %s\n",error.localizedDescription.UTF8String);
        ++failures; return;
    }
    std::array<std::uint32_t,32> elements{};
    for (std::uint32_t i=0; i<elements.size(); ++i) elements[i]=i;
    id<MTLBuffer> input=[device newBufferWithBytes:elements.data() length:sizeof(elements)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:4*sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    std::memset(output.contents,255,output.length);
    MTL4ArgumentTableDescriptor* table_descriptor=[MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount=2; table_descriptor.supportAttributeStrides=YES;
    id<MTL4ArgumentTable> table=[device newArgumentTableWithDescriptor:table_descriptor error:&error];
    [table setAddress:output.gpuAddress atIndex:0];
    [table setAddress:input.gpuAddress attributeStride:sizeof(std::uint32_t) atIndex:1];
    auto* texture_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    texture_descriptor.storageMode=MTLStorageModeShared;
    texture_descriptor.usage=MTLTextureUsageRenderTarget;
    id<MTLTexture> target=[device newTextureWithDescriptor:texture_descriptor];
    id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    [residency addAllocation:input]; [residency addAllocation:output];
    [residency addAllocation:target]; [residency commit];
    id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
    id<MTL4CommandBuffer> commands=[device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass=[MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture=target;
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    id<MTL4RenderCommandEncoder> encoder=[commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:4 baseInstance:7];
    [encoder endEncoding]; [commands endCommandBuffer];
    NSArray* retained=@[library,compiler,pipeline,input,output,table,target,residency,queue,allocator,commands];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block NSError* feedback_error=nil;
    MTL4CommitOptions* commit=[MTL4CommitOptions new];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        (void)retained.count; feedback_error=feedback.error; dispatch_semaphore_signal(done);
    }];
    id<MTL4CommandBuffer> buffers[]={commands}; [queue commit:buffers count:1 options:commit];
    if (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC))) {
        std::fputs("Native instance-step probe timed out\n",stderr); std::abort();
    }
    CHECK(feedback_error==nil);
    const auto* actual=static_cast<const std::uint32_t*>(output.contents);
    std::printf("Metal4 instance fetches (base 7, rate 3): %u %u %u %u\n",
        actual[0],actual[1],actual[2],actual[3]);
    // Metal adds baseInstance after dividing the draw-local instance ordinal.
    // This differs from L8's guest fetch contract, which yields 2, 2, 3, 3.
    CHECK(actual[0]==7 && actual[1]==7 && actual[2]==7 && actual[3]==8);
}

void inline_tests() {
    Fixture fixture;
    fixture.stop.registers[0x5f3] = fixture.stop.registers[0x5f5] = 0;
    fixture.stop.registers[0x5f6] = 99;
    fixture.stop.registers[0x591] = 1;
    fixture.stop.registers[0x592] = 99;
    fixture.stop.draw = DrawDescription{true, 5, 0, 7, -10, 7, 1};
    fixture.stop.draw->instance_index = 2;
    std::vector<std::uint32_t> source{10, 11, 12, 99, 10, 11, 12};
    auto plan = require(assemble_draw(fixture.stop, {}, fixture.memory->address_space(), source));
    source.clear();
    const auto before_reads = fixture.backend->reads;
    CHECK(ok(select_native_draw(plan, fixture.stop, *fixture.memory)));
    CHECK(fixture.backend->reads == before_reads);
    CHECK(plan.indices && plan.indices->restart_enabled);
    CHECK(plan.segments.size() == 1 && plan.segments[0].base_vertex == -10);
    CHECK(plan.source.base_instance == 7 && plan.source.instance_index == 2);
    CHECK(plan.segments.size() == 1 && plan.segments[0].base_instance == 9 &&
          plan.segments[0].instance_index == 2);
    CHECK(decode(plan) == (std::vector<std::uint32_t>{10, 11, 12, 0xffffffffu, 10, 11, 12}));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    CHECK(gpu_draw(device, plan, 0, 7));

    auto inconsistent = plan;
    inconsistent.indices->inline_source.pop_back();
    CHECK(!ok(select_native_draw(inconsistent, fixture.stop, *fixture.memory)));
    CHECK(fixture.backend->reads == before_reads);

    // Rebase the live U32 sentinel without losing its guest vertex ID.
    fixture.stop.registers[0x591] = 0;
    fixture.stop.draw = DrawDescription{true, 4, 0, 3, 0, 7, 1};
    source = {0xfffffffd, 0xfffffffe, 0xffffffff};
    plan = require(assemble_draw(fixture.stop, {}, fixture.memory->address_space(), source));
    CHECK(ok(select_native_draw(plan, fixture.stop, *fixture.memory)));
    CHECK(gpu_draw(device, plan, 0xfffffffd, 7));
    CHECK(fixture.backend->reads == before_reads);
}

void gpu_tests() {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::fputs("Metal 4 GPU required for native index tests\n",stderr); ++failures; return;
    }
    Fixture f;
    auto p=f.make({10,11,12,255,10,11,12},IndexFormat::U8,5,-10,true,255);
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0,7));
    p=f.make({65533,65534,65535},IndexFormat::U16,4,-65533);
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0,7));
    p=f.make({0xfffffffd,0xfffffffe,0xffffffff},IndexFormat::U32);
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0xfffffffd,7));
    p=f.make({1,2,3,99,1,2,3},IndexFormat::U16,4,-1,true,99);
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0,7));

    // L8's two independent strips and the selected native restart strip must
    // both cover the square with back-face culling enabled.
    p=f.make({10,11,12,13,99,10,11,12,13},IndexFormat::U16,5,-10,true,99);
    CHECK(gpu_draw(device,p,0,15,true));
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0,15,true));
    // One leading repeated vertex reverses the parity of every subsequent
    // live triangle. Preserving that degenerate must preserve the culled image.
    p=f.make({10,10,11,12,13},IndexFormat::U16,5,-10);
    CHECK(gpu_draw(device,p,0,15,true,false));
    CHECK(ok(select_native_draw(p,f.stop,*f.memory))); CHECK(gpu_draw(device,p,0,15,true,false));
    gpu_instance_step(device);
}
} // namespace

int main() {
    @autoreleasepool { unit_tests(); gpu_tests(); inline_tests(); }
    std::printf("Native draw encoding: %u failures\n",failures);
    return failures?1:0;
}
