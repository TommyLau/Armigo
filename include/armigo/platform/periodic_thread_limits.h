// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <algorithm>
#include <chrono>

namespace armigo::platform {

inline constexpr std::chrono::nanoseconds minimum_periodic_computation{50'000};
inline constexpr std::chrono::nanoseconds maximum_periodic_computation{50'000'000};

// Keep the requested CPU budget within both the frame allowance and the host's
// absolute computation limit. These are request values, not policy readback.
inline constexpr std::chrono::nanoseconds bounded_periodic_computation(
    std::chrono::nanoseconds period, std::chrono::nanoseconds computation) noexcept {
    using Duration = std::chrono::nanoseconds;
    if (period <= Duration::zero()) return Duration::zero();
    // Divide before multiplying so even an invalid enormous period cannot wrap.
    const auto count = period.count();
    const Duration frame_limit(count / 10 * 9 + count % 10 * 9 / 10);
    return std::min({std::max(computation, Duration(1'000'000)),
                     frame_limit, maximum_periodic_computation});
}

} // namespace armigo::platform
