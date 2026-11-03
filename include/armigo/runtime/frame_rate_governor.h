// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace armigo::runtime {

// Host-side Auto frame-rate policy. The scene check uses actual presented
// frames, not guest submissions; thermal states follow NSProcessInfo values
// (0 nominal, 1 fair, 2 serious, 3 critical), with negative meaning unknown.
class AutomaticFrameRateGovernor {
public:
    std::optional<unsigned> observe(std::chrono::steady_clock::time_point now,
                                    std::uint64_t presented_count, int thermal_state) noexcept;
    unsigned rate() const noexcept { return rate_; }

private:
    std::optional<unsigned> change_rate(unsigned rate,
        std::chrono::steady_clock::time_point now, std::uint64_t presented_count) noexcept;

    unsigned rate_ = 60;
    double up_wait_seconds_ = 300;
    std::optional<std::chrono::steady_clock::time_point> started_;
    std::optional<std::chrono::steady_clock::time_point> window_started_;
    std::optional<std::chrono::steady_clock::time_point> last_evaluation_;
    std::optional<std::chrono::steady_clock::time_point> nominal_since_;
    std::optional<std::chrono::steady_clock::time_point> hold_until_;
    std::optional<std::chrono::steady_clock::time_point> last_up_;
    std::uint64_t window_presented_ = 0;
};

} // namespace armigo::runtime
