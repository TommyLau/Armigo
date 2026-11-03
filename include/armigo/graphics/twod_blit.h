// ============================================================================
// Armigo Graphics - 2D Blit State Decode
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The 2D engine decodes the PixelsFromMemorySrcY0Int trigger, fixed-point
// source rectangle arithmetic, corner-origin adjustment, out-of-bounds linear
// pitch wrapping, surface fields, and the accelerated blit configuration.

#pragma once

#include <armigo/graphics/channel.h>
#include <armigo/graphics/texture_format.h>
#include <array>
#include <cstdint>
#include <variant>

namespace armigo::graphics {

// One 2D engine surface description (G80_2D SetDst*/SetSrc* register block).
struct TwodSurface {
    std::uint64_t address = 0;
    // Raw SetSrcFormatV/SetDstFormatV encoding; decode_twod_format maps it.
    std::uint32_t format = 0;
    // SetMemoryLayout: false is block-linear, true is pitch-linear.
    bool pitch = false;
    // Row stride in bytes (SetPitch); meaningful for pitch layouts, and the
    // wrap correction uses it for block-linear widths too.
    std::uint32_t pitch_bytes = 0;
    std::uint8_t block_height_log2 = 0;
    std::uint8_t block_depth_log2 = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // SetDepth zero means the engine's power-on default of one slice.
    std::uint32_t depth = 0;
    std::uint32_t layer = 0;
};

// SetOperationV. Only SrcCopy has a defined whole-texel meaning here; the
// remaining raster-operation/blend encodings return a structured error.
enum class TwodOperation : std::uint32_t {
    SrcCopyAnd = 0,
    RopAnd = 1,
    BlendAnd = 2,
    SrcCopy = 3,
    Rop = 4,
    SrcCopyPremult = 5,
    BlendPremult = 6,
};

// Decoded PixelsFromMemory state of one 0x237 trigger, ready for backend
// execution. Source coordinates are 32.32 fixed point; rectangles are texel
// units with exclusive ends.
struct TwodBlit {
    TwodSurface source{};
    TwodSurface destination{};
    TwodOperation operation = TwodOperation::SrcCopy;
    // SetPixelsFromMemorySampleMode: bit 0 origin (corner centering), bit 4
    // filter (bilinear).
    bool corner_origin = false;
    bool linear_filter = false;
    // SetPixelsFromMemoryDirection: true reverses the walked axis. The
    // supported execution paths keep the default direction; nonzero
    // directions decode but the backend rejects them by name.
    bool flip_x = false;
    bool flip_y = false;
    std::int32_t source_x1 = 0;
    std::int32_t source_y1 = 0;
    std::int32_t source_x2 = 0;
    std::int32_t source_y2 = 0;
    std::int32_t destination_x1 = 0;
    std::int32_t destination_y1 = 0;
    std::uint32_t destination_width = 0;
    std::uint32_t destination_height = 0;
    // Out-of-bounds linear-pitch wrap: bytes added to the source address after
    // the engine skips a first-line gap.
    std::uint64_t source_address_offset = 0;
    // SetClip rectangle. Enabled clips are a structured backend rejection.
    std::uint32_t clip_x0 = 0;
    std::uint32_t clip_y0 = 0;
    std::uint32_t clip_width = 0;
    std::uint32_t clip_height = 0;
    bool clip_enable = false;
};

// Decode outcome: a blit ready for backend admission, or the structured
// CommandResult that names the rejected state.
using TwodBlitResult = std::variant<TwodBlit, CommandResult>;

// Decodes the 2D register file captured at a 0x237 trigger into a TwodBlit.
// Reserved operation and direction encodings return a structured error naming
// them; every representable blit otherwise decodes.
[[nodiscard]] TwodBlitResult decode_twod_blit(
    const std::array<std::uint32_t, 0x1000>& registers) noexcept;

// Maps one SetSrcFormatV/SetDstFormatV encoding onto the shared texture format
// table. Channel
// reversed (BGR) encodings have no table entry and return false.
[[nodiscard]] bool decode_twod_format(std::uint32_t encoded,
                                      TextureFormat& out) noexcept;

}  // namespace armigo::graphics
