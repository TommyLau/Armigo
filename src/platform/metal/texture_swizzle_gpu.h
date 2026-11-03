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
// linear storage through Metal 4 argument tables and validated layout extents.

#pragma once

#import <Metal/Metal.h>

#include <armigo/graphics/renderer_types.h>

#include <memory>
#include <cstdint>

namespace armigo::platform::metal {

// Storage geometry of one mip level of one layer for a GPU swizzle transfer,
// derived from a calculated TextureLayoutDescription (never from descriptors).
// The guest side addresses the level's block-linear bytes inside the staging
// buffer; the linear side addresses its tightly packed image with rows of
// row_bytes payload at linear_pitch stride. gobs_x is the level's GOB column
// count including the sparse-tile column alignment, matching the level's
// slice_size addressing in texture_layout.cpp.
struct TextureSwizzleExtent {
    std::uint64_t guest_offset = 0;
    std::uint64_t linear_offset = 0;
    std::size_t guest_size = 0;
    std::uint32_t linear_pitch = 0;
    std::uint32_t row_bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t slices = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
    std::uint32_t gobs_x = 0;
};

// Metal 4 compute deswizzle/swizzle pipelines built once per device. One
// argument table and a growable parameters buffer are shared by every
// dispatch: each encoded level references its own parameters slot.
class GpuTextureSwizzle {
public:
    ~GpuTextureSwizzle();
    GpuTextureSwizzle(const GpuTextureSwizzle&) = delete;
    GpuTextureSwizzle& operator=(const GpuTextureSwizzle&) = delete;

    [[nodiscard]] static graphics::RendererResult<std::shared_ptr<GpuTextureSwizzle>>
    create(id<MTLDevice> device);

    // The transfer residency set must retain the parameters buffer while GPU
    // work referencing it is pending.
    [[nodiscard]] id<MTLBuffer> parameters_buffer() const noexcept;

    // Ensures the parameters buffer addresses the requested slot count.
    // Callers that retain parameters_buffer() in a residency set must reserve
    // before capturing it: encoding may otherwise replace the buffer.
    [[nodiscard]] graphics::RendererStatus reserve(std::size_t slots);

    // Deswizzle: reads the block-linear guest bytes and writes the tightly
    // packed linear image. Swizzle: the inverse. Slot selects the parameters
    // entry written for this dispatch; slots are reused across transfers.
    [[nodiscard]] graphics::RendererStatus encode_deswizzle(
        id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> source,
        id<MTLBuffer> destination, std::uint32_t slot,
        const TextureSwizzleExtent& extent);
    [[nodiscard]] graphics::RendererStatus encode_swizzle(
        id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> source,
        id<MTLBuffer> destination, std::uint32_t slot,
        const TextureSwizzleExtent& extent);

private:
    GpuTextureSwizzle() = default;
    // Retains the device that owns the pipelines, the argument table and the
    // parameters buffer, and allocates parameters growth.
    id<MTLDevice> device_ = nil;
    id<MTLComputePipelineState> deswizzle_pipeline_ = nil;
    id<MTLComputePipelineState> swizzle_pipeline_ = nil;
    id<MTL4ArgumentTable> arguments_ = nil;
    id<MTLBuffer> parameters_ = nil;

    // Grows the parameters buffer so the slot is addressable, then encodes
    // one dispatch of the named pipeline for the extent.
    [[nodiscard]] graphics::RendererStatus encode(
        id<MTLComputePipelineState> pipeline,
        id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> source,
        id<MTLBuffer> destination, std::uint32_t slot, bool to_linear,
        const TextureSwizzleExtent& extent);
};

} // namespace armigo::platform::metal
