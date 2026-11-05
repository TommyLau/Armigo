// ============================================================================
// Armigo Graphics - Presentation Scheduling Hints
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace armigo::graphics {
namespace detail {
inline std::atomic<std::uint32_t> maximum_presentation_rate{0};
}

// Process-wide host scheduling hint, independent of guest display timing.
// Zero means the application has not supplied a maximum presentation rate.
inline void set_maximum_presentation_rate(std::uint32_t rate) noexcept {
    detail::maximum_presentation_rate.store(rate, std::memory_order_relaxed);
}

struct PresentationScheduling {
    std::chrono::nanoseconds period{};
    // A guest cadence supplies a deadline without changing the unpaced
    // encoder's waiting strategy.
    bool guest_cadence = false;
};

inline PresentationScheduling presentation_scheduling(
    std::chrono::nanoseconds paced_period,
    std::chrono::nanoseconds guest_period = {}) noexcept {
    // Known host pacing supplies a deadline independently of a maximum-rate hint.
    if (paced_period > std::chrono::nanoseconds::zero()) return {paced_period, false};
    const auto rate = detail::maximum_presentation_rate.load(std::memory_order_relaxed);
    if (rate) return {std::chrono::nanoseconds(1'000'000'000ull / rate), false};
    if (guest_period > std::chrono::nanoseconds::zero()) return {guest_period, true};
    return {};
}

inline std::chrono::nanoseconds presentation_scheduling_period(
    std::chrono::nanoseconds paced_period) noexcept {
    return presentation_scheduling(paced_period).period;
}

} // namespace armigo::graphics
