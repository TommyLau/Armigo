// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Thirty-two threads per GOB move 16-byte segments between block-linear and
// linear storage. Metal 4 argument tables bind the buffers, and both directions
// use the validated extents defined by graphics/texture_layout.cpp.

#include "texture_swizzle_gpu.h"

#include <armigo/graphics/channel.h>

#include <algorithm>
#include <cstring>

namespace armigo::platform::metal {
namespace {

using graphics::CommandStatus;

graphics::RendererError failure(const char* detail,
                                CommandStatus status = CommandStatus::Unsupported) {
    return graphics::RendererError({status, detail});
}

// Per-dispatch geometry written into the shared parameters buffer. The field
// order must match SwizzleParams in the shader source below.
struct SwizzleParameters {
    std::uint64_t source_offset = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t pitch = 0;
    std::uint32_t row_bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t slices = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
    std::uint32_t gobs_x = 0;
};

static_assert(sizeof(SwizzleParameters) == 48);

constexpr std::uint32_t max_swizzle_log2 = 5;
constexpr std::uint32_t max_swizzle_dimension = 1u << 26;
constexpr std::size_t min_parameter_slots = 16;

// One GOB holds 8 rows of 64 bytes; 32 threads each own one 16-byte segment.
const char* const k_swizzle_source = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct SwizzleParams {
        ulong source_offset;
        ulong destination_offset;
        ulong pitch;
        uint row_bytes;
        uint rows;
        uint slices;
        uint gob_block_height_log2;
        uint gob_block_depth_log2;
        uint gobs_x;
    };

    // Byte offset of one 16-byte segment inside a level's block-linear
    // storage: blocks-of-GOBs per depth superplane and GOB row, then the
    // depth/row pair inside the block of GOBs, then the segment within the
    // GOB. Thread id bits enumerate the segments exactly as the layout's
    // intra-GOB address bits.
    inline ulong gob_segment_offset(constant SwizzleParams& p, uint gob_x,
                                    uint gob_y, uint z, uint tid) {
        const uint bh = 1u << p.gob_block_height_log2;
        const uint bd = 1u << p.gob_block_depth_log2;
        const uint blocks_y =
            (p.rows + (8u * bh) - 1u) >> (3u + p.gob_block_height_log2);
        const ulong block = 512ul * bh * bd;
        const ulong gob_in_block =
            ((ulong)(z & (bd - 1u)) << p.gob_block_height_log2) +
            (gob_y & (bh - 1u));
        return ((ulong)(z >> p.gob_block_depth_log2) * p.gobs_x * blocks_y +
                (ulong)(gob_y >> p.gob_block_height_log2) * p.gobs_x + gob_x) *
                   block +
               gob_in_block * 512ul + (ulong)tid * 16ul;
    }

    // Block-linear source bytes into a tightly packed linear image: one
    // thread copies the segment's 16-byte window clipped to the row payload.
    kernel void armigo_texture_deswizzle(
        device const uchar* source [[buffer(0)]],
        device uchar* destination [[buffer(1)]],
        constant SwizzleParams& p [[buffer(2)]],
        uint3 id [[thread_position_in_grid]]) {
        const uint tid = id.x & 31u;
        const uint gob_x = id.x >> 5;
        const uint xseg = ((tid >> 4) << 1) | ((tid >> 1) & 1u);
        const uint y = (id.y << 3) | (((tid >> 2) & 3u) << 1) | (tid & 1u);
        const uint x = (gob_x << 6) | (xseg << 4);
        if (x >= p.row_bytes || y >= p.rows || id.z >= p.slices) return;
        const uint n = min(16u, p.row_bytes - x);
        const ulong so = p.source_offset + gob_segment_offset(p, gob_x, id.y, id.z, tid);
        const ulong doff = p.destination_offset + ((ulong)id.z * p.rows + y) * p.pitch + x;
        if (n == 16u && ((so | doff) & 15ul) == 0ul)
            *(device uint4*)(destination + doff) = *(device const uint4*)(source + so);
        else
            for (uint i = 0; i < n; i++) destination[doff + i] = source[so + i];
    }

    // Inverse: tightly packed linear image into block-linear guest bytes.
    kernel void armigo_texture_swizzle(
        device const uchar* source [[buffer(0)]],
        device uchar* destination [[buffer(1)]],
        constant SwizzleParams& p [[buffer(2)]],
        uint3 id [[thread_position_in_grid]]) {
        const uint tid = id.x & 31u;
        const uint gob_x = id.x >> 5;
        const uint xseg = ((tid >> 4) << 1) | ((tid >> 1) & 1u);
        const uint y = (id.y << 3) | (((tid >> 2) & 3u) << 1) | (tid & 1u);
        const uint x = (gob_x << 6) | (xseg << 4);
        if (x >= p.row_bytes || y >= p.rows || id.z >= p.slices) return;
        const uint n = min(16u, p.row_bytes - x);
        const ulong so = p.source_offset + ((ulong)id.z * p.rows + y) * p.pitch + x;
        const ulong doff = p.destination_offset + gob_segment_offset(p, gob_x, id.y, id.z, tid);
        if (n == 16u && ((so | doff) & 15ul) == 0ul)
            *(device uint4*)(destination + doff) = *(device const uint4*)(source + so);
        else
            for (uint i = 0; i < n; i++) destination[doff + i] = source[so + i];
    }
)";

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, NSString* name) {
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:@(k_swizzle_source) options:nil error:&error];
    if (!library) return nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    return [device newComputePipelineStateWithFunction:function error:&error];
}

// Exact bytes the kernel touches on the linear side: the payload window of
// every row of the last depth slice.
std::uint64_t linear_footprint(const TextureSwizzleExtent& extent) noexcept {
    const std::uint64_t pitch = extent.linear_pitch;
    return (std::uint64_t)(extent.slices - 1) * extent.rows * pitch +
           (std::uint64_t)(extent.rows - 1) * pitch + extent.row_bytes;
}

graphics::RendererStatus encode_dispatch(
    id<MTLComputePipelineState> pipeline, id<MTL4ArgumentTable> arguments,
    id<MTLBuffer> parameters, std::uint32_t slot, id<MTLBuffer> source,
    id<MTLBuffer> destination, bool to_linear,
    const TextureSwizzleExtent& extent) {
    if (!pipeline || !arguments || !parameters || !source || !destination)
        return failure("texture swizzle transfer lacks its native objects");
    if (!extent.rows || !extent.slices || !extent.row_bytes || !extent.gobs_x ||
        !extent.guest_size || extent.linear_pitch < extent.row_bytes)
        return failure("texture swizzle extent has an empty transfer region");
    if (extent.gob_block_height_log2 > max_swizzle_log2 ||
        extent.gob_block_depth_log2 > max_swizzle_log2)
        return failure("texture swizzle extent exceeds GOB block limits");
    if (extent.rows > max_swizzle_dimension || extent.slices > max_swizzle_dimension ||
        extent.gobs_x > max_swizzle_dimension || extent.row_bytes > max_swizzle_dimension ||
        extent.linear_pitch > max_swizzle_dimension)
        return failure("texture swizzle extent exceeds dimension limits");
    // The guest side reads or writes the level's storage extent; the linear
    // side touches exactly the rows the kernel covers.
    const id<MTLBuffer> guest_buffer = to_linear ? source : destination;
    const id<MTLBuffer> linear_buffer = to_linear ? destination : source;
    if (extent.guest_offset > std::uint64_t(guest_buffer.length) ||
        extent.guest_size > guest_buffer.length - extent.guest_offset)
        return failure("texture swizzle guest extent exceeds the staging buffer");
    if (extent.linear_offset > std::uint64_t(linear_buffer.length) ||
        linear_footprint(extent) > linear_buffer.length - extent.linear_offset)
        return failure("texture swizzle linear extent exceeds the staging buffer");

    SwizzleParameters parameters_entry;
    parameters_entry.row_bytes = extent.row_bytes;
    parameters_entry.rows = extent.rows;
    parameters_entry.slices = extent.slices;
    parameters_entry.gob_block_height_log2 = extent.gob_block_height_log2;
    parameters_entry.gob_block_depth_log2 = extent.gob_block_depth_log2;
    parameters_entry.gobs_x = extent.gobs_x;
    if (to_linear) {
        parameters_entry.source_offset = extent.guest_offset;
        parameters_entry.destination_offset = extent.linear_offset;
        parameters_entry.pitch = extent.linear_pitch;
    } else {
        parameters_entry.source_offset = extent.linear_offset;
        parameters_entry.destination_offset = extent.guest_offset;
        parameters_entry.pitch = extent.linear_pitch;
    }
    const std::size_t slot_offset = std::size_t{slot} * sizeof(SwizzleParameters);
    if (slot_offset + sizeof(SwizzleParameters) > std::size_t(parameters.length))
        return failure("texture swizzle parameter slot exceeds the staging buffer");
    std::memcpy(static_cast<std::byte*>(parameters.contents) + slot_offset,
                &parameters_entry, sizeof(parameters_entry));

    // Buffer 0 reads the source image, buffer 1 the destination, buffer 2 the
    // parameters slot written above. Addresses are re-set per dispatch so
    // every level's entry is consumed in stream order.
    [arguments setAddress:source.gpuAddress atIndex:0];
    [arguments setAddress:destination.gpuAddress atIndex:1];
    [arguments setAddress:parameters.gpuAddress + slot_offset atIndex:2];
    // One threadgroup per (GOB column, GOB row, depth slice); 32 threads
    // cover one GOB's 16-byte segments.
    return graphics::renderer_success();
}

} // namespace

GpuTextureSwizzle::~GpuTextureSwizzle() = default;

graphics::RendererResult<std::shared_ptr<GpuTextureSwizzle>>
GpuTextureSwizzle::create(id<MTLDevice> device) {
    if (!device)
        return failure("texture swizzle requires a retained Metal device");
    auto swizzle = std::shared_ptr<GpuTextureSwizzle>(new GpuTextureSwizzle());
    swizzle->device_ = device;
    swizzle->deswizzle_pipeline_ = make_pipeline(device, @"armigo_texture_deswizzle");
    swizzle->swizzle_pipeline_ = make_pipeline(device, @"armigo_texture_swizzle");
    if (!swizzle->deswizzle_pipeline_ || !swizzle->swizzle_pipeline_)
        return failure("texture swizzle pipeline compilation failed");
    auto* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 3;
    descriptor.maxTextureBindCount = 0;
    NSError* error = nil;
    swizzle->arguments_ = [device newArgumentTableWithDescriptor:descriptor error:&error];
    if (!swizzle->arguments_)
        return failure("texture swizzle argument table allocation failed");
    swizzle->parameters_ = [device
        newBufferWithLength:min_parameter_slots * sizeof(SwizzleParameters)
                    options:MTLResourceStorageModeShared];
    if (!swizzle->parameters_)
        return failure("texture swizzle parameter allocation failed");
    return swizzle;
}

id<MTLBuffer> GpuTextureSwizzle::parameters_buffer() const noexcept {
    return parameters_;
}

graphics::RendererStatus GpuTextureSwizzle::reserve(std::size_t slots) {
    const std::size_t needed =
        std::max(slots, min_parameter_slots) * sizeof(SwizzleParameters);
    if (!parameters_ || std::size_t(parameters_.length) < needed) {
        parameters_ = [device_ newBufferWithLength:needed
                                           options:MTLResourceStorageModeShared];
        if (!parameters_) return failure("texture swizzle parameter allocation failed");
    }
    return graphics::renderer_success();
}

graphics::RendererStatus GpuTextureSwizzle::encode(
    id<MTLComputePipelineState> pipeline, id<MTL4ComputeCommandEncoder> encoder,
    id<MTLBuffer> source, id<MTLBuffer> destination, std::uint32_t slot,
    bool to_linear, const TextureSwizzleExtent& extent) {
    if (!encoder) return failure("texture swizzle requires a compute encoder");
    if (auto grown = reserve(std::size_t{slot} + 1);
        std::holds_alternative<graphics::RendererError>(grown))
        return grown;
    auto status = encode_dispatch(pipeline, arguments_, parameters_, slot,
                                  source, destination, to_linear, extent);
    if (std::holds_alternative<graphics::RendererError>(status)) return status;
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:arguments_];
    // One threadgroup per (GOB column, GOB row, depth slice); 32 threads
    // cover one GOB's 16-byte segments, as required by the GOB geometry.
    [encoder dispatchThreadgroups:MTLSizeMake(extent.gobs_x, (extent.rows + 7) >> 3, extent.slices)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return graphics::renderer_success();
}

graphics::RendererStatus GpuTextureSwizzle::encode_deswizzle(
    id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> source,
    id<MTLBuffer> destination, std::uint32_t slot,
    const TextureSwizzleExtent& extent) {
    return encode(deswizzle_pipeline_, encoder, source, destination, slot,
                  true, extent);
}

graphics::RendererStatus GpuTextureSwizzle::encode_swizzle(
    id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> source,
    id<MTLBuffer> destination, std::uint32_t slot,
    const TextureSwizzleExtent& extent) {
    return encode(swizzle_pipeline_, encoder, source, destination, slot,
                  false, extent);
}

} // namespace armigo::platform::metal
