// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/frame_rate_governor.h>

#include <algorithm>

namespace armigo::runtime {
namespace {

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

} // namespace

std::optional<unsigned> AutomaticFrameRateGovernor::change_rate(
    unsigned rate, std::chrono::steady_clock::time_point now, std::uint64_t presented_count) noexcept {
    if (rate == rate_) return std::nullopt;
    if (rate == 30 && last_up_ && seconds(now - *last_up_) < 300) {
        up_wait_seconds_ = std::min(1800.0, up_wait_seconds_ * 2);
    }
    rate_ = rate;
    window_started_ = now;
    window_presented_ = presented_count;
    last_evaluation_ = now;
    if (rate == 60) last_up_ = now;
    return rate;
}

std::optional<unsigned> AutomaticFrameRateGovernor::observe(
    std::chrono::steady_clock::time_point now, std::uint64_t presented_count, int thermal_state) noexcept {
    if (!started_) {
        started_ = now;
        window_started_ = now;
        last_evaluation_ = now;
        window_presented_ = presented_count;
        return std::nullopt;
    }
    if (presented_count < window_presented_) {
        window_started_ = now;
        window_presented_ = presented_count;
        return std::nullopt;
    }
    if (last_evaluation_ && seconds(now - *last_evaluation_) < 1) return std::nullopt;
    last_evaluation_ = now;

    const double window = seconds(now - *window_started_);
    if (rate_ > 30 && window >= 10 && seconds(now - *started_) >= 60) {
        const std::uint64_t shown = presented_count - window_presented_;
        const double expected = window * rate_;
        const double missed_share = expected > shown ? (expected - shown) / expected : 0;
        if (shown >= 100 && missed_share > 0.04) return change_rate(30, now, presented_count);
        window_started_ = now;
        window_presented_ = presented_count;
    }

    if (thermal_state >= 2) {
        nominal_since_.reset();
        hold_until_ = now + std::chrono::seconds(120);
        return change_rate(30, now, presented_count);
    }
    if (thermal_state == 1) {
        nominal_since_.reset();
        return change_rate(30, now, presented_count);
    }
    if (thermal_state < 0) return std::nullopt;
    if (!nominal_since_) nominal_since_ = now;
    if (rate_ == 30 && (!hold_until_ || now >= *hold_until_) &&
        seconds(now - *nominal_since_) >= up_wait_seconds_) {
        return change_rate(60, now, presented_count);
    }
    return std::nullopt;
}

} // namespace armigo::runtime
