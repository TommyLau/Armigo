// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "fixture.h"
#include "pipeline/handle_program.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <optional>
#include <set>
#include <vector>

namespace armigo::shader_native::test {
// Executes a translated native fragment program over one pixel of an
// RGBA32Uint target. Constant bank 3 holds the caller's input words; the
// result is the target pixel, so up to four exported registers are visible.
class MetalHarness {
public:
    struct Compiled { id<MTLRenderPipelineState> pipeline; ir::MslArtifact artifact; };
    // Guest memory behind one window: guest [base, base + 4 * words.size())
    // backed by a shared buffer. After a run, words holds the memory and
    // table the window table, including its miss record.
    struct GuestMemory {
        std::uint64_t base{};
        std::vector<std::uint32_t> words, table;
        // The pool is reached through its chunk's partial hint (word 7)
        // instead of the chunk hint, behind an out-of-order decoy entry that
        // makes the binary search miss: only the hinted scan finds it.
        bool partial_hint = false;
        // When set, builds the table (header and entries) for the window's
        // device address instead: several pools over the one buffer.
        std::function<std::vector<std::uint32_t>(std::uint64_t)> build_table;
    };
    MetalHarness() {
        device_ = MTLCreateSystemDefaultDevice(); check(device_ != nil, "Metal device");
        NSError* error = nil;
        compiler_ = [device_ newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error]; check(compiler_ != nil, "Metal 4 compiler");
        queue_ = [device_ newMTL4CommandQueue]; check(queue_ != nil, "Metal 4 queue");
        vertex_ = library(R"(#include <metal_stdlib>
using namespace metal;
struct VertexResult { float4 position [[position]]; };
vertex VertexResult native_test_vertex(uint i [[vertex_id]]) {
    const float2 p[3]={float2(-2,-2),float2(6,-2),float2(-2,6)};
    VertexResult out; out.position=float4(p[i],0,1); return out;
})");
    }
    // y_negate specializes the guest Y sign (function constant 9).
    Compiled compile(const Fixture& fixture, bool y_negate = false) {
        return compile(fixture, FrontendOptions{}, y_negate);
    }
    Compiled compile(const Fixture& fixture, const FrontendOptions& options, bool y_negate = false) {
        auto translated = translate_native(fixture.program(), options);
        if (auto error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        Compiled result{nil, std::get<ir::MslArtifact>(std::move(translated))};
        auto fragment_library = library(result.artifact.source);
        MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new]; vertex.library = vertex_; vertex.name = @"native_test_vertex";
        MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new]; fragment.library = fragment_library; fragment.name = @"armigo_fragment_main";
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new]; std::uint32_t color_type = 1;
        [constants setConstantValue:&color_type type:MTLDataTypeUInt atIndex:0];
        [constants setConstantValue:&y_negate type:MTLDataTypeBool atIndex:9];
        MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
        specialized.functionDescriptor = fragment; specialized.constantValues = constants;
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vertex; descriptor.fragmentFunctionDescriptor = specialized;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle; descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
        NSError* error = nil;
        result.pipeline = [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!result.pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        return result;
    }
    // Builds a rasterization-disabled render pipeline from a translated
    // vertex stage and returns the translation; a failure throws.
    ir::MslArtifact discard_pipeline(const Fixture& fixture, const FrontendOptions& options) {
        auto translated = translate_native(fixture.program(), options);
        if (auto error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        auto artifact = std::get<ir::MslArtifact>(std::move(translated));
        MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = library(artifact.source);
        vertex.name = [NSString stringWithUTF8String:(artifact.reflection.discard_entry_point.empty()
            ? artifact.entry_point : artifact.reflection.discard_entry_point).c_str()];
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vertex; descriptor.rasterizationEnabled = NO;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
        NSError* error = nil;
        if (![compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error])
            throw std::runtime_error(error.localizedDescription.UTF8String);
        return artifact;
    }
    // Executes an output-less native vertex through either static entry.
    // Pixel Z observes the depth adapter; guest memory observes side effects.
    std::vector<std::array<float, 4>> outputless_vertex(const ir::MslArtifact& artifact,
            bool discard, bool triangles, const std::vector<std::uint32_t>& input, GuestMemory& guest) {
        NSError* error = nil;
        auto vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = library(artifact.source);
        const auto& entry = discard ? artifact.reflection.discard_entry_point : artifact.entry_point;
        check(!entry.empty(), "output-less vertex entry is declared");
        vertex.name = [NSString stringWithUTF8String:entry.c_str()];
        auto descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vertex;
        descriptor.rasterizationEnabled = !discard;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassUnspecified;
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
        if (!discard) {
            auto fragment = [MTL4LibraryFunctionDescriptor new];
            fragment.library = library("#include <metal_stdlib>\nusing namespace metal; fragment float4 probe(float4 p [[position]]) { return float4(p.z,1,2,3); }");
            fragment.name = @"probe";
            descriptor.fragmentFunctionDescriptor = fragment;
        }
        auto pipeline = [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        id<MTLBuffer> words=nil, guest_table=nil, data=nil, bounds=nil;
        guest_buffers(&guest, words, guest_table);
        constant_buffers(input, data, bounds);
        auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
        td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
        auto target = [device_ newTextureWithDescriptor:td];
        auto pixels = [device_ newBufferWithLength:256 options:MTLResourceStorageModeShared];
        NSMutableArray<id<MTLAllocation>>* allocations = [NSMutableArray arrayWithArray:@[words,guest_table,data,bounds,target,pixels]];
        auto table = bind(artifact, allocations, data, bounds, guest_table, nil);
        auto residency = [device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
        [residency commit];
        auto allocator = [device_ newCommandAllocator];
        auto commands = [device_ newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        auto pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(-1,-1,-1,-1);
        auto render = [commands renderCommandEncoderWithDescriptor:pass];
        [render setRenderPipelineState:pipeline];
        [render setArgumentTable:table atStages:MTLRenderStageVertex];
        [render setViewport:MTLViewport{0,0,4,4,0,1}];
        [render drawPrimitives:triangles ? MTLPrimitiveTypeTriangle : MTLPrimitiveTypePoint
                  vertexStart:0 vertexCount:triangles ? 3 : 1];
        [render endEncoding];
        auto copy = [commands computeCommandEncoder];
        [copy barrierAfterQueueStages:MTLStageVertex | MTLStageFragment beforeStages:MTLStageBlit
                   visibilityOptions:MTL4VisibilityOptionDevice];
        [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(4,4,1) toBuffer:pixels destinationOffset:0
            destinationBytesPerRow:64 destinationBytesPerImage:256];
        [copy endEncoding]; [commands endCommandBuffer];
        submit(commands);
        read_back(&guest, words, guest_table);
        std::vector<std::array<float,4>> result(16);
        std::memcpy(result.data(), pixels.contents, 256);
        return result;
    }
    // Front-facing winding of the harness triangle's draws.
    MTLWinding front_winding = MTLWindingClockwise;
    // Mipmap levels of a 2D pattern texture; above one, the sampler picks the
    // nearest level.
    unsigned mip_levels = 1;
    // Ordinary sampled textures may exercise wrapping and border gathers.
    MTLSamplerAddressMode texture_address_mode = MTLSamplerAddressModeClampToEdge;
    MTLSamplerBorderColor texture_border_color = MTLSamplerBorderColorTransparentBlack;
    // Read-only 2D images become R32Float, each texel holding its w.
    bool red_pattern = false;
    // A texture keyed to a handle program gets the pattern only when the
    // program, evaluated over the input words (every constant bank) and the
    // run's guest memory, names this texture and sampler ID; otherwise a
    // zero texture.
    std::uint32_t handle_texture = 0, handle_sampler = 0;
    // The texels (row by row, RGBA) of the 2D images the last draw or
    // dispatch wrote, in reflection order. A 3D image is a 4x4x4 volume;
    // images3d() reads all 64 of its texels. An atomic image is a separate
    // 4x4 R32Uint texture; atomic_images() reads its sixteen words.
    static std::uint32_t atomic_texel(unsigned x, unsigned y) { return 0x01010000u | (y << 8) | x; }
    std::vector<std::array<std::uint32_t, 16>> atomic_images() const {
        std::vector<std::array<std::uint32_t, 16>> out;
        for (id<MTLTexture> image in atomic_images_) {
            std::array<std::uint32_t, 16> texels{};
            [image getBytes:texels.data() bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
            out.push_back(texels);
        }
        return out;
    }
    std::vector<std::array<float, 64>> images() const {
        std::vector<std::array<float, 64>> out;
        for (id<MTLTexture> image in images_) {
            std::array<float, 64> texels{};
            [image getBytes:texels.data() bytesPerRow:64 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
            out.push_back(texels);
        }
        return out;
    }
    // 4x4x4 RGBA, x fastest, then y, then z, of each 3D image the last draw
    // or dispatch wrote, in reflection order.
    std::vector<std::array<float, 256>> images3d() const {
        std::vector<std::array<float, 256>> out;
        for (id<MTLTexture> image in images3d_) {
            std::array<float, 256> texels{};
            [image getBytes:texels.data() bytesPerRow:64 bytesPerImage:256
                  fromRegion:MTLRegionMake3D(0, 0, 0, 4, 4, 4) mipmapLevel:0 slice:0];
            out.push_back(texels);
        }
        return out;
    }
    // The clear value marks a discarded fragment.
    static constexpr std::uint32_t cleared = 0xdeadbeefu;
    // With depth_target, draws carry a Depth32Float attachment cleared to 1,
    // every fragment passing and writing it (a Metal 4 pipeline names no
    // depth format); the last draw's depths land in depths, row by row.
    bool depth_target = false;
    std::vector<float> depths;
    // One constant-color texel, or a 2D image whose column x stores
    // red = column_base + x (the address-mode test).
    struct PoolTexture {
        std::array<float, 4> color{0.f, 0.f, 0.f, 1.f};
        unsigned width = 1, height = 1;
        bool column_red = false;
        float column_base = 0.f;
    };
    struct PoolSampler {
        MTLSamplerMinMagFilter filter = MTLSamplerMinMagFilterNearest;
        MTLSamplerAddressMode address = MTLSamplerAddressModeClampToEdge;
    };
    // n IDs and n + 1 entries (the project documentation). Entry 0 and
    // entry n are the dummies the test stored. A texture pool keeps a zeroed
    // bitmap of ceil(n / 32) words; mark publishes its GPU address, and a
    // cleared mark stores zero so the stage marks nothing. dimension, array,
    // multisample and comparison are the element type a reflected pool matches
    // when several tables are bound.
    struct TexturePool {
        std::uint32_t n = 1;
        std::vector<PoolTexture> entries;
        bool mark = true;
        ir::TextureDimension dimension = ir::TextureDimension::d2;
        bool array = false;
        bool multisample = false;
        bool comparison = false;
    };
    struct SamplerPool {
        std::uint32_t n = 1;
        std::vector<PoolSampler> entries;
    };
    std::optional<TexturePool> texture_pool;
    // Several texture pools, one bitmap each, matched to reflected resources
    // by element type. Used instead of texture_pool when non-empty.
    std::vector<TexturePool> texture_pools;
    std::optional<SamplerPool> sampler_pool;
    // The texture-pool bitmap after the last draw, one word per ceil(n / 32).
    // Empty when that draw had no singular texture pool.
    std::vector<std::uint32_t> pool_bitmap;
    // One bitmap per bound entry of texture_pools, in bind order, keyed by
    // the element type that selected the table.
    struct PoolBitmap {
        ir::TextureDimension dimension = ir::TextureDimension::d2;
        bool array = false;
        bool comparison = false;
        std::vector<std::uint32_t> words;
    };
    std::vector<PoolBitmap> pool_bitmaps;
    // When non-zero, grid first draws an occluder that writes depth over the
    // first this many columns, then the stage with a less-than depth test.
    // Zero keeps the single draw. stage_writes_depth gives that second test
    // a depth state that writes; otherwise the test is read-only.
    unsigned occlude_columns = 0;
    bool stage_writes_depth = false;
    // height is the render-target height the draw-parameter record carries.
    std::array<std::uint32_t, 4> run(const Compiled& compiled, const std::vector<std::uint32_t>& input, GuestMemory* guest = nullptr,
                                     std::uint32_t height = 1) {
        return grid(compiled, input, 1, 1, guest, height).front();
    }
    // Draws a width x height target, every pixel covered, and returns the
    // pixels row by row from the first memory row.
    std::vector<std::array<std::uint32_t, 4>> grid(const Compiled& compiled, const std::vector<std::uint32_t>& input, unsigned width, unsigned rows,
                                                   GuestMemory* guest = nullptr, std::uint32_t height = 1) {
        NSError* error = nil;
        const std::array<std::uint32_t, 2> parameters{0, height};
        auto draw = [device_ newBufferWithBytes:parameters.data() length:sizeof(parameters) options:MTLResourceStorageModeShared];
        id<MTLBuffer> guest_words = nil, guest_table = nil;
        guest_buffers(guest, guest_words, guest_table);
        id<MTLBuffer> data = nil, bounds = nil;
        constant_buffers(input, data, bounds);
        auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint width:width height:rows mipmapped:NO];
        td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
        auto target = [device_ newTextureWithDescriptor:td];
        auto readback = [device_ newBufferWithLength:16 * width * rows options:MTLResourceStorageModeShared];
        NSMutableArray<id<MTLAllocation>>* allocations = [NSMutableArray arrayWithArray:@[data, bounds, target, readback, draw]];
        id<MTLTexture> depth_texture = nil;
        id<MTLBuffer> depth_readback = nil;
        id<MTLDepthStencilState> depth_state = nil;
        if (depth_target) {
            auto dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:width height:rows mipmapped:NO];
            dd.storageMode = MTLStorageModePrivate; dd.usage = MTLTextureUsageRenderTarget;
            depth_texture = [device_ newTextureWithDescriptor:dd];
            depth_readback = [device_ newBufferWithLength:4 * width * rows options:MTLResourceStorageModeShared];
            MTLDepthStencilDescriptor* state = [MTLDepthStencilDescriptor new];
            state.depthCompareFunction = MTLCompareFunctionAlways; state.depthWriteEnabled = YES;
            depth_state = [device_ newDepthStencilStateWithDescriptor:state];
            check(depth_texture && depth_readback && depth_state, "Metal depth resources");
            [allocations addObject:depth_texture]; [allocations addObject:depth_readback];
        }
        if (guest) { [allocations addObject:guest_words]; [allocations addObject:guest_table]; }
        bound_guest_ = guest;
        auto table = bind(compiled.artifact, allocations, data, bounds, guest_table, draw);
        bound_guest_ = nullptr;
        auto residency = [device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
        [residency commit];
        auto allocator = [device_ newCommandAllocator]; auto commands = [device_ newCommandBuffer];
        check(data && bounds && target && readback && table && residency && allocator && commands, "Metal resources");
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(double(cleared), double(cleared), double(cleared), double(cleared));
        if (depth_texture) {
            pass.depthAttachment.texture = depth_texture; pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore; pass.depthAttachment.clearDepth = 1.0;
        }
        auto render = [commands renderCommandEncoderWithDescriptor:pass];
        if (occlude_columns) {
            check(depth_state != nil && occlude_columns <= width, "occlusion needs a depth target");
            [render setRenderPipelineState:occluder_pipeline()];
            [render setDepthStencilState:depth_state];
            [render setFrontFacingWinding:front_winding];
            [render setScissorRect:MTLScissorRect{0, 0, occlude_columns, rows}];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [render setScissorRect:MTLScissorRect{0, 0, width, rows}];
        }
        [render setRenderPipelineState:compiled.pipeline]; [render setArgumentTable:table atStages:MTLRenderStageFragment];
        if (occlude_columns) [render setDepthStencilState:stage_writes_depth ? depth_write_less() : depth_less()];
        else if (depth_state) [render setDepthStencilState:depth_state];
        [render setFrontFacingWinding:front_winding];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [render endEncoding];
        auto copy = [commands computeCommandEncoder];
        [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, rows, 1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:16 * width destinationBytesPerImage:0];
        if (depth_texture)
            [copy copyFromTexture:depth_texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, rows, 1)
                         toBuffer:depth_readback destinationOffset:0 destinationBytesPerRow:4 * width destinationBytesPerImage:0];
        [copy endEncoding]; [commands endCommandBuffer];
        submit(commands);
        std::vector<std::array<std::uint32_t, 4>> pixels(width * rows);
        std::memcpy(pixels.data(), readback.contents, 16 * width * rows);
        depths.assign(depth_texture ? width * rows : 0, 0.0f);
        if (depth_texture) std::memcpy(depths.data(), depth_readback.contents, 4 * width * rows);
        read_back(guest, guest_words, guest_table);
        pool_bitmap.clear();
        if (pool_bitmap_buffer_) {
            pool_bitmap.resize(pool_bitmap_buffer_.length / 4);
            std::memcpy(pool_bitmap.data(), pool_bitmap_buffer_.contents, pool_bitmap_buffer_.length);
        }
        pool_bitmaps.clear();
        for (NSUInteger i = 0; i < extra_pool_bitmaps_.count; ++i) {
            id<MTLBuffer> buffer = extra_pool_bitmaps_[i];
            PoolBitmap bitmap = pending_bitmap_keys_[i];
            bitmap.words.resize(buffer.length / 4);
            std::memcpy(bitmap.words.data(), buffer.contents, buffer.length);
            pool_bitmaps.push_back(std::move(bitmap));
        }
        return pixels;
    }
    struct CompiledCompute { id<MTLComputePipelineState> pipeline; ir::MslArtifact artifact; };
    CompiledCompute compile_compute(const Fixture& fixture) {
        auto translated = translate_native(fixture.program());
        if (auto error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        CompiledCompute result{nil, std::get<ir::MslArtifact>(std::move(translated))};
        MTL4LibraryFunctionDescriptor* kernel = [MTL4LibraryFunctionDescriptor new];
        kernel.library = library(result.artifact.source); kernel.name = @"armigo_compute_main";
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = kernel;
        NSError* error = nil;
        result.pipeline = [compiler_ newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!result.pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        return result;
    }
    // Dispatches groups x 1 x 1 thread groups of the fixture's launch size.
    // Constant banks hold input as for draws; results come back through the
    // guest memory, which the program stores to.
    void dispatch(const CompiledCompute& compiled, const Fixture& fixture, const std::vector<std::uint32_t>& input, GuestMemory& guest,
                  unsigned groups = 1) {
        id<MTLBuffer> guest_words = nil, guest_table = nil;
        guest_buffers(&guest, guest_words, guest_table);
        id<MTLBuffer> data = nil, bounds = nil;
        constant_buffers(input, data, bounds);
        NSMutableArray<id<MTLAllocation>>* allocations = [NSMutableArray arrayWithArray:@[data, bounds, guest_words, guest_table]];
        auto table = bind(compiled.artifact, allocations, data, bounds, guest_table, nil);
        NSError* error = nil;
        auto residency = [device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
        [residency commit];
        auto allocator = [device_ newCommandAllocator]; auto commands = [device_ newCommandBuffer];
        check(table && residency && allocator && commands, "Metal compute resources");
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        auto encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:compiled.pipeline]; [encoder setArgumentTable:table];
        const auto& size = fixture.local_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(size[0], size[1], size[2])];
        [encoder endEncoding]; [commands endCommandBuffer];
        submit(commands);
        read_back(&guest, guest_words, guest_table);
    }
    // The mesh stage's paired fragment reads, and the source of the
    // fragment function mesh_test_fragment instead of the default, when set.
    std::optional<std::array<std::uint8_t, 32>> mesh_fragment_reads;
    std::string mesh_fragment_source;
    // The last mesh_grid's mesh stage source.
    std::string mesh_source;
    // Draws a mesh stage, under a task stage when one is given, into a width
    // x rows RGBA32Uint target cleared to zero, launching groups task (or
    // mesh) workgroups; the draw's first workgroup is first. The fragment
    // returns (generic0.x bits, 1, 0, 0), flat. Returns the pixels row by row;
    // mesh_status holds the task stage's status words after the draw.
    std::vector<std::array<std::uint32_t, 4>> mesh_grid(const Fixture* task, const Fixture& mesh, std::uint32_t first, unsigned groups,
                                                        unsigned width, unsigned rows, const std::vector<std::uint32_t>& input = {}) {
        FrontendOptions options; options.vertex_depth = ir::DepthConvention::zero_to_one;
        options.mesh_fragment_reads = mesh_fragment_reads;
        const auto translate = [&](const Fixture& fixture) {
            auto translated = translate_native(fixture.program(), options);
            if (auto error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
            return std::get<ir::MslArtifact>(std::move(translated));
        };
        const auto function = [&](const std::string& source, const std::string& name) {
            MTL4LibraryFunctionDescriptor* f = [MTL4LibraryFunctionDescriptor new];
            f.library = library(source); f.name = [NSString stringWithUTF8String:name.c_str()];
            return f;
        };
        std::optional<ir::MslArtifact> object;
        MTL4MeshRenderPipelineDescriptor* descriptor = [MTL4MeshRenderPipelineDescriptor new];
        if (task) {
            object = translate(*task);
            const auto program = task->program();
            descriptor.objectFunctionDescriptor = function(object->source, object->entry_point);
            descriptor.payloadMemoryLength = (program.mesh_launch->payload_bytes + 15) & ~15u;
            descriptor.maxTotalThreadsPerObjectThreadgroup = program.mesh_launch->threads;
            descriptor.maxTotalThreadgroupsPerMeshGrid = 1024;  // Metal's limit, which the task stage clamps to
        }
        const auto drawn = translate(mesh);
        const auto mesh_threads = mesh.program().mesh_launch->threads;
        descriptor.meshFunctionDescriptor = function(drawn.source, drawn.entry_point);
        descriptor.maxTotalThreadsPerMeshThreadgroup = mesh_threads;
        mesh_source = drawn.source;
        descriptor.fragmentFunctionDescriptor = function(!mesh_fragment_source.empty() ? mesh_fragment_source : R"(#include <metal_stdlib>
using namespace metal;
struct MeshTestInput { float4 position [[position]]; float generic0_0 [[user(locn0_0), flat]]; };
fragment uint4 mesh_test_fragment(MeshTestInput in [[stage_in]]) { return uint4(as_type<uint>(in.generic0_0), 1u, 0u, 0u); })",
                                                         "mesh_test_fragment");
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
        NSError* error = nil;
        id<MTLRenderPipelineState> pipeline = [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        const std::array<std::uint32_t, 2> parameters{first, 0};
        auto draw = [device_ newBufferWithBytes:parameters.data() length:sizeof(parameters) options:MTLResourceStorageModeShared];
        id<MTLBuffer> data = nil, bounds = nil;
        constant_buffers(input, data, bounds);
        auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint width:width height:rows mipmapped:NO];
        td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
        auto target = [device_ newTextureWithDescriptor:td];
        auto readback = [device_ newBufferWithLength:16 * width * rows options:MTLResourceStorageModeShared];
        NSMutableArray<id<MTLAllocation>>* allocations = [NSMutableArray arrayWithArray:@[data, bounds, target, readback, draw]];
        auto mesh_table = bind(drawn, allocations, data, bounds, nil, draw);
        auto object_table = object ? bind(*object, allocations, data, bounds, nil, draw) : nil;
        const std::array<std::uint32_t, 2> zero{};
        auto status = [device_ newBufferWithBytes:zero.data() length:sizeof(zero) options:MTLResourceStorageModeShared];
        [allocations addObject:status];
        if (object) [object_table setAddress:status.gpuAddress atIndex:*object->reflection.draw_support_buffer_index + 1];
        auto residency = [device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
        [residency commit];
        auto allocator = [device_ newCommandAllocator]; auto commands = [device_ newCommandBuffer];
        check(target && readback && mesh_table && residency && allocator && commands, "Metal mesh resources");
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore; pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        auto render = [commands renderCommandEncoderWithDescriptor:pass];
        [render setRenderPipelineState:pipeline];
        [render setArgumentTable:mesh_table atStages:MTLRenderStageMesh];
        if (object_table) [render setArgumentTable:object_table atStages:MTLRenderStageObject];
        const auto object_threads = task ? task->program().mesh_launch->threads : 1u;
        [render drawMeshThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(object_threads, 1, 1)
              threadsPerMeshThreadgroup:MTLSizeMake(mesh_threads, 1, 1)];
        [render endEncoding];
        auto copy = [commands computeCommandEncoder];
        [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, rows, 1)
                     toBuffer:readback destinationOffset:0 destinationBytesPerRow:16 * width destinationBytesPerImage:0];
        [copy endEncoding]; [commands endCommandBuffer];
        submit(commands);
        std::vector<std::array<std::uint32_t, 4>> pixels(width * rows);
        std::memcpy(pixels.data(), readback.contents, 16 * width * rows);
        std::memcpy(mesh_status.data(), status.contents, sizeof(mesh_status));
        return pixels;
    }
    std::array<std::uint32_t, 2> mesh_status{};
    // Texel (x, y, z or layer) of a level holds (x, y, z, 10000 level + 1000 +
    // 100 z + 10 y + x) as RGBA32Float on a 4x4 (x4) texture; level L of a
    // mipmapped 2D texture is 4 >> L texels wide.
    static std::array<float, 4> texel(unsigned x, unsigned y, unsigned z, unsigned level = 0) {
        return {float(x), float(y), float(z), float(10000 * level + 1000 + 100 * z + 10 * y + x)};
    }
private:
    const GuestMemory* bound_guest_ = nullptr;
    // One pool (entry 1); the hint names it for its first 2 MiB chunk, and
    // accesses in a following chunk take the binary search.
    void guest_buffers(GuestMemory* guest, __strong id<MTLBuffer>& words, __strong id<MTLBuffer>& table) {
        if (!guest) return;
        words = [device_ newBufferWithBytes:guest->words.data() length:guest->words.size() * 4 options:MTLResourceStorageModeShared];
        if (guest->build_table) {
            guest->table = guest->build_table(words.gpuAddress);
            guest->table.resize(ir::guest_memory_bytes / 4, 0);
            table = [device_ newBufferWithBytes:guest->table.data() length:guest->table.size() * 4 options:MTLResourceStorageModeShared];
            return;
        }
        guest->table.assign(ir::guest_memory_bytes / 4, 0);
        const auto end = guest->base + guest->words.size() * 4;
        const auto entry = ir::guest_memory_header_words + ir::guest_memory_entry_words;
        const std::uint64_t delta = words.gpuAddress - guest->base, chunk = guest->base >> 21;
        guest->table[0] = 2;
        guest->table[entry] = std::uint32_t(guest->base); guest->table[entry + 1] = std::uint32_t(guest->base >> 32);
        guest->table[entry + 2] = std::uint32_t(end); guest->table[entry + 3] = std::uint32_t(end >> 32);
        guest->table[entry + 4] = std::uint32_t(delta); guest->table[entry + 5] = std::uint32_t(delta >> 32);
        const auto slot = ir::guest_memory_header_words + (chunk & 8191) * ir::guest_memory_entry_words;
        if (guest->partial_hint) {
            guest->table[0] = 3;
            guest->table[entry + ir::guest_memory_entry_words + 2] = 16;  // decoy entry 2: guest [0, 16), delta 0
            // Both chunks the pool starts and ends in name it.
            for (const std::uint64_t at : {guest->base, end - 1})
                guest->table[ir::guest_memory_header_words + ((at >> 21) & 8191) * ir::guest_memory_entry_words + 7] =
                    std::uint32_t((at >> 21) << 13) | 1;
        } else {
            guest->table[slot + 6] = std::uint32_t(chunk << 13) | 1;
        }
        table = [device_ newBufferWithBytes:guest->table.data() length:guest->table.size() * 4 options:MTLResourceStorageModeShared];
    }
    void constant_buffers(const std::vector<std::uint32_t>& input, __strong id<MTLBuffer>& data, __strong id<MTLBuffer>& bounds) {
        data = [device_ newBufferWithLength:65536 options:MTLResourceStorageModeShared];
        std::memcpy(data.contents, input.data(), input.size() * 4);
        std::array<std::uint32_t, 18> sizes{}; sizes[3] = static_cast<std::uint32_t>(input.size() * 4);
        bounds = [device_ newBufferWithBytes:sizes.data() length:sizeof(sizes) options:MTLResourceStorageModeShared];
    }
    // Binds every reflected resource: constant banks read data, the bounds
    // buffer holds bank sizes, guest memory reads the pool table, textures
    // read pattern textures through nearest samplers.
    id<MTL4ArgumentTable> bind(const ir::MslArtifact& artifact, NSMutableArray<id<MTLAllocation>>* allocations, id<MTLBuffer> data,
                               id<MTLBuffer> bounds, id<MTLBuffer> guest_table, id<MTLBuffer> draw) {
        NSError* error = nil;
        [images_ removeAllObjects];
        [images3d_ removeAllObjects];
        [pool_textures_ removeAllObjects];
        [pool_samplers_ removeAllObjects];
        [pool_tables_ removeAllObjects];
        [extra_pool_bitmaps_ removeAllObjects];
        pending_bitmap_keys_.clear();
        pool_bitmap_buffer_ = nil;
        [atomic_images_ removeAllObjects];
        MTL4ArgumentTableDescriptor* at = [MTL4ArgumentTableDescriptor new];
        at.maxBufferBindCount = 31; at.maxTextureBindCount = 128; at.maxSamplerStateBindCount = 16;
        auto table = [device_ newArgumentTableWithDescriptor:at error:&error];
        std::set<std::uint32_t> named;
        for (std::size_t i = 0; i < artifact.reflection.handle_programs.size(); ++i) {
            const auto* words = data ? static_cast<const std::uint32_t*>(data.contents) : nullptr;
            const std::size_t count = data ? data.length / 4 : 0;
            const auto handle = ir::evaluate_handle_program(artifact.reflection.handle_programs[i],
                [&](std::uint32_t, std::uint32_t offset) { return offset / 4 < count ? words[offset / 4] : 0u; },
                [&](std::uint64_t address) -> std::uint32_t {
                    if (!bound_guest_ || address < bound_guest_->base || address % 4) return 0;
                    const auto index = (address - bound_guest_->base) / 4;
                    return index < bound_guest_->words.size() ? bound_guest_->words[index] : 0u;
                });
            if (handle && (*handle & 0xfffffu) == handle_texture && (*handle >> 20) == handle_sampler)
                named.insert(static_cast<std::uint32_t>(i * 4));
        }
        if (artifact.reflection.draw_support_buffer_index) {
            check(draw != nil, "draw parameters for a program that reads them");
            [table setAddress:draw.gpuAddress atIndex:*artifact.reflection.draw_support_buffer_index];
        }
        for (const auto& resource : artifact.reflection.resources) {
            const auto& range = std::get<ir::MetalRange>(resource.location);
            if (const auto* buffer = std::get_if<ir::BufferUse>(&resource.use)) {
                const auto* support = std::get_if<ir::SupportBufferSource>(&buffer->source);
                if (support && support->schema == ir::guest_memory_schema) {
                    check(guest_table != nil, "guest memory for a program that reads it");
                    [table setAddress:guest_table.gpuAddress atIndex:range.first];
                    continue;
                }
                if (const auto* storage = std::get_if<ir::StorageBufferSource>(&buffer->source)) {
                    check(data && guest_table, "storage descriptor data and guest backing");
                    const auto key = storage->descriptor;
                    check(key.handle_byte_offset <= data.length - 12, "storage descriptor within test data");
                    const auto* descriptor = static_cast<const std::uint32_t*>(data.contents) + key.handle_byte_offset / 4;
                    const auto address = std::uint64_t(descriptor[0]) | (std::uint64_t(descriptor[1]) << 32);
                    const auto* words = static_cast<const std::uint32_t*>(guest_table.contents);
                    std::uint64_t device_address = 0;
                    for (unsigned entry = 1; entry < words[0]; ++entry) {
                        const auto* row = words + ir::guest_memory_header_words + entry * ir::guest_memory_entry_words;
                        const auto start = std::uint64_t(row[0]) | (std::uint64_t(row[1]) << 32);
                        const auto end = std::uint64_t(row[2]) | (std::uint64_t(row[3]) << 32);
                        if (address >= start && address < end && descriptor[2] <= end - address) {
                            device_address = address + (std::uint64_t(row[4]) | (std::uint64_t(row[5]) << 32));
                            break;
                        }
                    }
                    check(device_address != 0, "test storage descriptor fits its guest backing");
                    [table setAddress:device_address atIndex:range.first];
                    continue;
                }
                [table setAddress:(std::holds_alternative<ir::ConstantBufferSource>(buffer->source) ? data.gpuAddress : bounds.gpuAddress) atIndex:range.first];
            } else if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use); texture && texture->access != ir::Access::read) {
                // A U32 atomic image is a seeded 4x4 R32Uint texture.
                // atomic_images() reads it back after the work. A float image
                // stays a zeroed RGBA32Float texture, or a 4x4x4 volume.
                if (texture->atomic) {
                    check(texture->descriptor.shape.dimension == ir::TextureDimension::d2 &&
                          !texture->descriptor.shape.array && !texture->descriptor.shape.multisample &&
                          texture->descriptor.scalar == ir::ScalarType::u32 &&
                          texture->descriptor.format == ir::ComponentFormat::unsigned_integer,
                          "atomic test images are 2D R32Uint");
                    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                                  width:4 height:4 mipmapped:NO];
                    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageShaderAtomic;
                    td.storageMode = MTLStorageModeShared;
                    auto image = [device_ newTextureWithDescriptor:td];
                    check(image != nil, "Metal atomic texture");
                    std::array<std::uint32_t, 16> seed{};
                    for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) seed[y * 4 + x] = atomic_texel(x, y);
                    [image replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:seed.data() bytesPerRow:16];
                    [allocations addObject:image]; [atomic_images_ addObject:image];
                    [table setTexture:image.gpuResourceID atIndex:range.first];
                    continue;
                }
                const bool volume = texture->descriptor.shape.dimension == ir::TextureDimension::d3;
                MTLTextureDescriptor* td = volume ? [MTLTextureDescriptor new]
                    : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
                if (volume) {
                    td.textureType = MTLTextureType3D;
                    td.pixelFormat = MTLPixelFormatRGBA32Float;
                    td.width = 4; td.height = 4; td.depth = 4;
                }
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; td.storageMode = MTLStorageModeShared;
                auto image = [device_ newTextureWithDescriptor:td];
                check(image != nil, "Metal storage texture");
                if (volume) {
                    const std::array<float, 256> zero{};
                    [image replaceRegion:MTLRegionMake3D(0, 0, 0, 4, 4, 4) mipmapLevel:0 slice:0 withBytes:zero.data() bytesPerRow:64 bytesPerImage:256];
                    [allocations addObject:image]; [images3d_ addObject:image];
                } else {
                    const std::array<float, 64> zero{};
                    [image replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:zero.data() bytesPerRow:64];
                    [allocations addObject:image]; [images_ addObject:image];
                }
                [table setTexture:image.gpuResourceID atIndex:range.first];
            } else if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use); texture && texture->pool) {
                check(range.space == ir::MetalNamespace::buffer && range.count == 1, "a texture pool binds one buffer");
                if (!texture_pools.empty()) {
                    const auto& shape = texture->descriptor.shape;
                    int found = -1;
                    for (std::size_t i = 0; i < texture_pools.size(); ++i) {
                        const auto& pool = texture_pools[i];
                        if (pool.dimension != shape.dimension || pool.array != shape.array ||
                            pool.multisample != shape.multisample || pool.comparison != texture->descriptor.comparison)
                            continue;
                        check(found < 0, "two texture pools share an element type");
                        found = static_cast<int>(i);
                    }
                    check(found >= 0, "reflected texture pool has no table");
                    id<MTLBuffer> bitmap = nil;
                    [table setAddress:texture_pool_table(texture_pools[static_cast<std::size_t>(found)], allocations, bitmap).gpuAddress atIndex:range.first];
                    [extra_pool_bitmaps_ addObject:bitmap];
                    PoolBitmap key;
                    key.dimension = shape.dimension;
                    key.array = shape.array;
                    key.comparison = texture->descriptor.comparison;
                    pending_bitmap_keys_.push_back(key);
                } else {
                    check(!pool_bitmap_buffer_, "a stage has one texture pool");
                    check(texture_pool.has_value(), "a reflected texture pool needs a table");
                    [table setAddress:texture_pool_table(*texture_pool, allocations, pool_bitmap_buffer_).gpuAddress atIndex:range.first];
                }
            } else if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use)) {
                auto bound = texture->key.cbuf_slot == ir::handle_program_slot && !named.count(texture->key.handle_byte_offset)
                    ? blank(texture->descriptor.shape) : pattern(texture->descriptor.shape);
                [allocations addObject:bound];
                [table setTexture:bound.gpuResourceID atIndex:range.first];
            } else if (const auto* sampler = std::get_if<ir::SamplerUse>(&resource.use); sampler && sampler->pool) {
                check(range.space == ir::MetalNamespace::buffer && range.count == 1, "a sampler pool binds one buffer");
                check(pool_samplers_.count == 0, "a stage has one sampler pool");
                [table setAddress:sampler_pool_table(allocations).gpuAddress atIndex:range.first];
            } else if (std::holds_alternative<ir::SamplerUse>(resource.use)) {
                MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
                sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest; sd.supportArgumentBuffers = YES;
                if (mip_levels > 1) sd.mipFilter = MTLSamplerMipFilterNearest;
                sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = texture_address_mode;
                sd.borderColor = texture_border_color;
                auto sampler = [device_ newSamplerStateWithDescriptor:sd];
                check(sampler != nil, "Metal sampler");
                [table setSamplerState:sampler.gpuResourceID atIndex:range.first];
            } else throw std::runtime_error("unsupported reflected test resource");
        }
        return table;
    }
    void submit(id<MTL4CommandBuffer> commands) {
        dispatch_semaphore_t done = dispatch_semaphore_create(0); __block NSError* execution = nil;
        MTL4CommitOptions* commit = [MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> f) { execution = f.error; dispatch_semaphore_signal(done); }];
        id<MTL4CommandBuffer> submitted[] = {commands}; [queue_ commit:submitted count:1 options:commit];
        check(dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0, "GPU completion");
        if (execution) throw std::runtime_error(execution.localizedDescription.UTF8String);
    }
    void read_back(GuestMemory* guest, id<MTLBuffer> words, id<MTLBuffer> table) {
        if (!guest) return;
        std::memcpy(guest->words.data(), words.contents, guest->words.size() * 4);
        std::memcpy(guest->table.data(), table.contents, guest->table.size() * 4);
    }
    id<MTLTexture> blank(const ir::TextureShape& shape) {
        if (shape.dimension != ir::TextureDimension::d2 || shape.array) throw std::runtime_error("zero test textures are 2D");
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
        td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageShaderRead;
        auto texture = [device_ newTextureWithDescriptor:td];
        check(texture != nil, "Metal zero texture");
        const std::array<float, 64> zero{};
        [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:zero.data() bytesPerRow:64];
        return texture;
    }
    id<MTLTexture> pattern(const ir::TextureShape& shape) {
        if (red_pattern) {
            // A 4x4 R32Float image holding each texel's w: a format without
            // alpha, which Metal reads as alpha 1.
            if (shape.dimension != ir::TextureDimension::d2 || shape.array || mip_levels > 1)
                throw std::runtime_error("red test textures are single-level 2D");
            MTLTextureDescriptor* rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:4 height:4 mipmapped:NO];
            rd.storageMode = MTLStorageModeShared; rd.usage = MTLTextureUsageShaderRead;
            auto texture = [device_ newTextureWithDescriptor:rd];
            check(texture != nil, "Metal red pattern texture");
            std::array<float, 16> texels{};
            for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) texels[y * 4 + x] = texel(x, y, 0)[3];
            [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
            return texture;
        }
        MTLTextureDescriptor* td = [MTLTextureDescriptor new];
        td.pixelFormat = MTLPixelFormatRGBA32Float; td.width = 4; td.height = 4; td.storageMode = MTLStorageModeShared;
        td.usage = MTLTextureUsageShaderRead;
        unsigned depth = 1, layers = 1;
        if (shape.dimension == ir::TextureDimension::d3) { td.textureType = MTLTextureType3D; td.depth = depth = 4; }
        else if (shape.dimension == ir::TextureDimension::d2 && shape.array) { td.textureType = MTLTextureType2DArray; td.arrayLength = layers = 4; }
        else if (shape.dimension == ir::TextureDimension::d2) td.textureType = MTLTextureType2D;
        else throw std::runtime_error("unsupported test texture shape");
        if (mip_levels > 1) {
            if (td.textureType != MTLTextureType2D) throw std::runtime_error("mipmapped test textures are 2D");
            td.mipmapLevelCount = mip_levels;
        }
        auto texture = [device_ newTextureWithDescriptor:td];
        check(texture != nil, "Metal pattern texture");
        for (unsigned level = 1; level < mip_levels; ++level) {
            const unsigned size = 4 >> level;
            std::vector<float> texels(size * size * 4);
            for (unsigned y = 0; y < size; ++y) for (unsigned x = 0; x < size; ++x) {
                const auto t = texel(x, y, 0, level);
                std::copy(t.begin(), t.end(), texels.begin() + (y * size + x) * 4);
            }
            [texture replaceRegion:MTLRegionMake2D(0, 0, size, size) mipmapLevel:level withBytes:texels.data() bytesPerRow:16 * size];
        }
        for (unsigned z = 0; z < std::max(depth, layers); ++z) {
            std::array<float, 64> texels{};
            for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
                const auto t = texel(x, y, z);
                std::copy(t.begin(), t.end(), texels.begin() + (y * 4 + x) * 4);
            }
            [texture replaceRegion:MTLRegionMake3D(0, 0, depth > 1 ? z : 0, 4, 4, 1) mipmapLevel:0 slice:layers > 1 ? z : 0
                         withBytes:texels.data() bytesPerRow:64 bytesPerImage:256];
        }
        return texture;
    }
    // A 16 + 8 (n + 1) byte table: n, a zero word, the bitmap address, then
    // one Metal resource ID per entry (the project documentation).
    id<MTLBuffer> pool_table(std::uint32_t n, std::uint64_t seen, const std::vector<std::uint64_t>& ids) {
        check(n >= 1 && ids.size() == static_cast<std::size_t>(n) + 1, "a pool table holds n + 1 entries");
        const std::size_t bytes = 16 + ids.size() * 8;
        std::vector<std::uint8_t> raw(bytes);
        std::memcpy(raw.data(), &n, 4);
        std::memcpy(raw.data() + 8, &seen, 8);
        for (std::size_t i = 0; i < ids.size(); ++i) std::memcpy(raw.data() + 16 + i * 8, &ids[i], 8);
        auto buffer = [device_ newBufferWithBytes:raw.data() length:bytes options:MTLResourceStorageModeShared];
        check(buffer != nil, "pool table");
        [pool_tables_ addObject:buffer];
        return buffer;
    }
    id<MTLTexture> pool_texture(const PoolTexture& entry, bool array) {
        check(entry.width && entry.height, "pool texture size");
        MTLTextureDescriptor* td = array ? [MTLTextureDescriptor new]
            : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                width:entry.width height:entry.height mipmapped:NO];
        if (array) {
            td.textureType = MTLTextureType2DArray;
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            td.width = entry.width; td.height = entry.height; td.arrayLength = 1;
        }
        td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageShaderRead;
        auto texture = [device_ newTextureWithDescriptor:td];
        check(texture != nil, "pool texture");
        std::vector<float> texels(entry.width * entry.height * 4);
        for (unsigned y = 0; y < entry.height; ++y) for (unsigned x = 0; x < entry.width; ++x) {
            const auto at = (y * entry.width + x) * 4;
            texels[at] = entry.column_red ? entry.column_base + float(x) : entry.color[0];
            texels[at + 1] = entry.color[1]; texels[at + 2] = entry.color[2]; texels[at + 3] = entry.color[3];
        }
        if (array)
            [texture replaceRegion:MTLRegionMake2D(0, 0, entry.width, entry.height) mipmapLevel:0 slice:0
                         withBytes:texels.data() bytesPerRow:entry.width * 16 bytesPerImage:entry.width * entry.height * 16];
        else
            [texture replaceRegion:MTLRegionMake2D(0, 0, entry.width, entry.height) mipmapLevel:0
                         withBytes:texels.data() bytesPerRow:entry.width * 16];
        return texture;
    }
    id<MTLBuffer> texture_pool_table(const TexturePool& pool, NSMutableArray<id<MTLAllocation>>* allocations, id<MTLBuffer> __strong & bitmap_out) {
        check(pool.n >= 1 && pool.entries.size() == pool.n + 1u, "texture pool entries cover 0 through n");
        const auto words = (pool.n + 31u) / 32u;
        auto bitmap = [device_ newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
        check(bitmap != nil, "texture pool bitmap");
        std::memset(bitmap.contents, 0, words * 4);
        bitmap_out = bitmap;
        [allocations addObject:bitmap];
        std::vector<std::uint64_t> ids;
        ids.reserve(pool.entries.size());
        for (const auto& entry : pool.entries) {
            auto texture = pool_texture(entry, pool.array);
            [pool_textures_ addObject:texture];
            [allocations addObject:texture];
            ids.push_back(texture.gpuResourceID._impl);
        }
        const std::uint64_t seen = pool.mark ? bitmap.gpuAddress : 0;
        auto table = pool_table(pool.n, seen, ids);
        [allocations addObject:table];
        return table;
    }
    id<MTLBuffer> sampler_pool_table(NSMutableArray<id<MTLAllocation>>* allocations) {
        check(sampler_pool && sampler_pool->n >= 1 && sampler_pool->entries.size() == sampler_pool->n + 1u,
              "sampler pool entries cover 0 through n");
        std::vector<std::uint64_t> ids;
        ids.reserve(sampler_pool->entries.size());
        for (const auto& entry : sampler_pool->entries) {
            MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = entry.filter;
            sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
            sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = entry.address;
            sd.supportArgumentBuffers = YES;
            sd.normalizedCoordinates = YES;
            auto sampler = [device_ newSamplerStateWithDescriptor:sd];
            check(sampler != nil, "pool sampler");
            [pool_samplers_ addObject:sampler];
            ids.push_back(sampler.gpuResourceID._impl);
        }
        // A sampler pool carries no read bitmap.
        auto table = pool_table(sampler_pool->n, 0, ids);
        [allocations addObject:table];
        return table;
    }
    id<MTLRenderPipelineState> occluder_pipeline() {
        if (occluder_) return occluder_;
        auto fragment_library = library(R"(#include <metal_stdlib>
using namespace metal;
struct VertexResult { float4 position [[position]]; };
fragment uint4 pool_occluder_fragment(VertexResult in [[stage_in]]) {
    return uint4(as_type<uint>(in.position.z), 0u, 0u, 0u);
})");
        MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = vertex_; vertex.name = @"native_test_vertex";
        MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
        fragment.library = fragment_library; fragment.name = @"pool_occluder_fragment";
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vertex; descriptor.fragmentFunctionDescriptor = fragment;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
        NSError* error = nil;
        occluder_ = [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!occluder_) throw std::runtime_error(error.localizedDescription.UTF8String);
        return occluder_;
    }
    id<MTLDepthStencilState> depth_less() {
        if (depth_less_) return depth_less_;
        MTLDepthStencilDescriptor* state = [MTLDepthStencilDescriptor new];
        state.depthCompareFunction = MTLCompareFunctionLess; state.depthWriteEnabled = NO;
        depth_less_ = [device_ newDepthStencilStateWithDescriptor:state];
        check(depth_less_ != nil, "less-than depth state");
        return depth_less_;
    }
    // Less-than with writes, for a pool stage that exports depth. The
    // read-only state stays the one early fragment tests use.
    id<MTLDepthStencilState> depth_write_less() {
        if (depth_write_less_) return depth_write_less_;
        MTLDepthStencilDescriptor* state = [MTLDepthStencilDescriptor new];
        state.depthCompareFunction = MTLCompareFunctionLess; state.depthWriteEnabled = YES;
        depth_write_less_ = [device_ newDepthStencilStateWithDescriptor:state];
        check(depth_write_less_ != nil, "less-than depth-write state");
        return depth_write_less_;
    }
    id<MTLLibrary> library(const std::string& source) {
        MTLCompileOptions* options = [MTLCompileOptions new]; options.languageVersion = MTLLanguageVersion4_0; options.mathMode = MTLMathModeSafe;
        options.preserveInvariance = YES;
        NSError* error = nil;
        auto result = [device_ newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
        if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
        return result;
    }
    id<MTLDevice> device_;
    id<MTL4Compiler> compiler_;
    id<MTL4CommandQueue> queue_;
    id<MTLLibrary> vertex_;
    NSMutableArray<id<MTLTexture>>* images_ = [NSMutableArray array];
    NSMutableArray<id<MTLTexture>>* images3d_ = [NSMutableArray array];
    NSMutableArray<id<MTLTexture>>* pool_textures_ = [NSMutableArray array];
    NSMutableArray<id<MTLSamplerState>>* pool_samplers_ = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>>* pool_tables_ = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>>* extra_pool_bitmaps_ = [NSMutableArray array];
    std::vector<PoolBitmap> pending_bitmap_keys_;
    id<MTLBuffer> pool_bitmap_buffer_ = nil;
    id<MTLRenderPipelineState> occluder_ = nil;
    id<MTLDepthStencilState> depth_less_ = nil;
    NSMutableArray<id<MTLTexture>>* atomic_images_ = [NSMutableArray array];
    id<MTLDepthStencilState> depth_write_less_ = nil;
};
} // namespace armigo::shader_native::test
