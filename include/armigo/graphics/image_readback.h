// ============================================================================
// Armigo Graphics - CPU Image Readback
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/color_clear.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::graphics {

struct Rgba8Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::byte> pixels;
};

struct ImageReadbackError {
    enum class Code {
        InvalidCrop,
        InvalidLayout,
        InputTooSmall,
        ExtentLimit,
        AllocationFailure,
    };

    Code code = Code::InvalidLayout;
    std::string_view detail;
};

using Rgba8ReadbackResult = std::variant<Rgba8Image, ImageReadbackError>;

// Copies an already completed, CPU-readable color target into tightly packed
// row-major RGBA8 bytes. The caller owns synchronization: it must wait for the
// producer fence and retain the backing storage for the duration of the call.
// The all-zero crop selects the full surface as a host capture policy; it is
// not a recovered Horizon display meaning. A nonempty crop is left, top, right,
// bottom in target pixels.
[[nodiscard]] Rgba8ReadbackResult
readback_rgba8(const ColorClearPlan& plan,
               std::span<const std::byte> storage,
               std::array<std::int32_t, 4> crop = {}) noexcept;

}  // namespace armigo::graphics
