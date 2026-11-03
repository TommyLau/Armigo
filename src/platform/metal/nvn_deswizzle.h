// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture uploads deswizzle block-linear guest storage on the GPU: one compute
// dispatch per level and layer reads the swizzled bytes straight from the
// memory pool's buffer and writes linear rows into the upload staging buffer.

#pragma once

#import <Metal/Metal.h>

#include <armigo/graphics/texture_layout.h>

#include <cstdint>
#include <string>

namespace armigo::platform::metal::nvn_driver {

// The compiled deswizzle kernel of one device, or why it is unavailable.
struct DeswizzleKernel {
  id<MTLComputePipelineState> pipeline = nil;
  std::string error;
};

[[nodiscard]] DeswizzleKernel make_deswizzle_kernel(id<MTLDevice> device);

// Whether a dispatch can copy the level into rows of `pitch` bytes: the
// extents fit the kernel's 32-bit coordinates and the rows fit the pitch.
[[nodiscard]] bool deswizzle_supported(
    const graphics::BlockLinearLevelAddressing &level, std::uint64_t pitch);

// Encodes one dispatch that copies the level, whose storage starts at
// `source_offset` in `source` (guest_base is relative to it), into
// `destination` from `destination_offset`: block row y of depth slice z goes
// to destination_offset + (z * rows + y) * pitch, and the row's bytes past
// row_bytes are left untouched. One threadgroup of 32 threads covers one GOB;
// each thread copies one 16-byte GOB segment. The caller checks
// deswizzle_supported first.
void encode_deswizzle(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> pipeline, id<MTLBuffer> source,
                      std::uint64_t source_offset, id<MTLBuffer> destination,
                      std::uint64_t destination_offset, std::uint64_t pitch,
                      const graphics::BlockLinearLevelAddressing &level);

} // namespace armigo::platform::metal::nvn_driver
