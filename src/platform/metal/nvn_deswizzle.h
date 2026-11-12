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

#include <cstddef>
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

inline constexpr std::size_t deswizzle_parameter_bytes = 64;

// Metal 4 keeps dispatch parameters in caller-owned storage through GPU
// completion. A batch can suballocate this storage with 256-byte alignment.
bool encode_deswizzle_metal4(id<MTL4ComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            id<MTL4ArgumentTable> table,
                            id<MTLBuffer> parameters, std::uint64_t parameter_offset,
                            id<MTLBuffer> source, std::uint64_t source_offset,
                            id<MTLBuffer> destination, std::uint64_t destination_offset,
                            std::uint64_t pitch,
                            const graphics::BlockLinearLevelAddressing& level);

} // namespace armigo::platform::metal::nvn_driver
