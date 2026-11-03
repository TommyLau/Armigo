// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Address arithmetic shared by the decoder, encoders, and relocation. Nothing
// here wraps: results outside the 64-bit address space are reported as absent.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace armigo::aarch64::detail {

// base + displacement, or nullopt when the sum leaves [0, 2^64).
constexpr std::optional<std::uint64_t> offset_address(std::uint64_t base,
                                                      std::int64_t displacement) noexcept {
    if (displacement >= 0) {
        const auto magnitude = static_cast<std::uint64_t>(displacement);
        if (base > std::numeric_limits<std::uint64_t>::max() - magnitude) {
            return std::nullopt;
        }
        return base + magnitude;
    }
    const std::uint64_t magnitude = std::uint64_t{0} - static_cast<std::uint64_t>(displacement);
    if (base < magnitude) {
        return std::nullopt;
    }
    return base - magnitude;
}

// target - origin as a signed value, or nullopt when it does not fit int64.
constexpr std::optional<std::int64_t> distance(std::uint64_t origin, std::uint64_t target) noexcept {
    constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (target >= origin) {
        const std::uint64_t magnitude = target - origin;
        if (magnitude > limit) {
            return std::nullopt;
        }
        return static_cast<std::int64_t>(magnitude);
    }
    const std::uint64_t magnitude = origin - target;
    if (magnitude > limit + 1) {
        return std::nullopt;
    }
    return magnitude == limit + 1 ? std::numeric_limits<std::int64_t>::min()
                                  : -static_cast<std::int64_t>(magnitude);
}

constexpr bool fits(std::int64_t value, std::int64_t minimum, std::int64_t maximum) noexcept {
    return value >= minimum && value <= maximum;
}

} // namespace armigo::aarch64::detail
