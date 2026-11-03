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
#include <array>
#include <chrono>
#include <cstddef>

namespace armigo::platform {

// Estimates a periodic thread's CPU budget from its own CPU clock. The policy
// callback runs on that thread and must leave its old policy intact on failure.
class PeriodicThreadBudget {
public:
    using Duration = std::chrono::nanoseconds;
    enum class Result { unchanged, applied, rejected };

    Duration period() const noexcept { return applied_period_; }
    Duration computation() const noexcept { return applied_computation_; }

    // A zero period requests ordinary scheduling. Failed changes preserve the
    // last accepted declaration and retry after one sampling window, without
    // treating an unaccepted budget as active.
    template <class Apply>
    Result tick(Duration period, Duration cpu, Apply&& apply) {
        if (period < Duration::zero()) period = Duration::zero();
        if (period != requested_period_) {
            requested_period_ = period;
            requested_computation_ = period == Duration::zero()
                ? Duration::zero() : bounded(period, Duration(2'000'000));
            count_ = 0;
            last_cpu_ = Duration::zero();
            retry_ = 0;
            return declare(apply);
        }
        if (retry_) {
            if (--retry_) return Result::unchanged;
            return declare(apply);
        }
        if (period == Duration::zero() || applied_period_ != period)
            return Result::unchanged;
        const auto previous = last_cpu_;
        last_cpu_ = cpu;
        if (previous == Duration::zero() || cpu < previous)
            return Result::unchanged;
        samples_[count_++] = cpu - previous;
        if (count_ != samples_.size()) return Result::unchanged;
        count_ = 0;
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        const auto computation = bounded(period, sorted[107] * 5 / 4);
        if (computation <= applied_computation_ * 115 / 100 &&
            computation >= applied_computation_ * 85 / 100)
            return Result::unchanged;
        requested_computation_ = computation;
        return declare(apply);
    }

private:
    static Duration bounded(Duration period, Duration computation) noexcept {
        return std::min(std::max(computation, Duration(1'000'000)), period * 9 / 10);
    }

    template <class Apply> Result declare(Apply& apply) {
        if (!apply(requested_period_, requested_computation_)) {
            retry_ = samples_.size();
            count_ = 0;
            last_cpu_ = Duration::zero();
            return Result::rejected;
        }
        applied_period_ = requested_period_;
        applied_computation_ = requested_computation_;
        retry_ = 0;
        return Result::applied;
    }

    Duration requested_period_{}, requested_computation_{};
    Duration applied_period_{}, applied_computation_{}, last_cpu_{};
    std::array<Duration, 120> samples_{};
    std::size_t count_ = 0, retry_ = 0;
};

} // namespace armigo::platform
