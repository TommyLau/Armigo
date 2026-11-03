// ============================================================================
// Armigo Graphics - GPU Color Clear Planning
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/engines.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace armigo::graphics {

inline constexpr std::size_t max_color_clear_storage_size = 256U << 20;

enum class ColorTargetFormat : std::uint32_t {
    A2B10G10R10Unorm = 0xd1,
    R8G8B8A8Unorm = 0xd5,
    R8G8B8A8Srgb = 0xd6,
};

enum class ColorTargetLayout : std::uint8_t {
    Pitch,
    BlockLinear,
};

// Logical texel rectangle inside the full target surface. Always populated:
// the full surface when no clear scissor applies, otherwise the validated
// scissor extent. Storage geometry (width, height, row_bytes, storage_size,
// and every pixel_byte_offset answer) keeps describing the full surface.
struct ColorClearRegion {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ColorClearPlan {
    std::uint64_t target_address = 0;
    std::uint32_t target_index = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Physical row stride, including any extra GOB columns for block-linear
    // surfaces. Storage size is the stride times the layout-aligned height;
    // allocation padding beyond that geometry is not included.
    std::size_t row_bytes = 0;
    std::size_t storage_size = 0;
    ColorTargetFormat format = ColorTargetFormat::R8G8B8A8Unorm;
    ColorTargetLayout layout = ColorTargetLayout::Pitch;
    std::uint32_t pitch_bytes = 0;
    std::uint32_t block_height_log2 = 0;
    std::array<float, 4> color{};
    ColorClearRegion region{};
};

using ColorClearResult = std::variant<ColorClearPlan, CommandResult>;
using PixelByteOffsetResult = std::variant<std::size_t, CommandResult>;

[[nodiscard]] ColorClearResult plan_color_clear(const RenderStop& stop) noexcept;

[[nodiscard]] PixelByteOffsetResult
pixel_byte_offset(const ColorClearPlan& plan, std::uint32_t x, std::uint32_t y) noexcept;

}  // namespace armigo::graphics
