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

#include <armigo/platform/periodic_thread_limits.h>
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

    // Last successful request; the native scheduler may adjust its computation.
    Duration period() const noexcept { return applied_period_; }
    Duration computation() const noexcept { return applied_computation_; }

    // A zero period requests ordinary scheduling. Failed changes preserve the
    // last successful request and retry after a fresh sampling window, without
    // treating a rejected budget as active.
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
            return declare(apply, cpu);
        }
        if (retry_) {
            if (period != Duration::zero()) {
                const auto previous = last_cpu_;
                last_cpu_ = cpu;
                if (cpu < previous) {
                    count_ = 0;
                    retry_ = samples_.size();
                    return Result::unchanged;
                }
                samples_[count_++] = cpu - previous;
            }
            if (--retry_) return Result::unchanged;
            if (period != Duration::zero()) requested_computation_ = estimate(period);
            count_ = 0;
            return declare(apply, cpu);
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
        const auto computation = estimate(period);
        if (computation <= applied_computation_ * 115 / 100 &&
            computation >= applied_computation_ * 85 / 100)
            return Result::unchanged;
        requested_computation_ = computation;
        return declare(apply, cpu);
    }

private:
    static Duration bounded(Duration period, Duration computation) noexcept {
        return bounded_periodic_computation(period, computation);
    }

    Duration estimate(Duration period) const {
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        // Bound before scaling so a very long observed CPU interval cannot wrap.
        return bounded(period, std::min(sorted[107], maximum_periodic_computation) * 5 / 4);
    }

    template <class Apply> Result declare(Apply& apply, Duration cpu) {
        if (!apply(requested_period_, requested_computation_)) {
            retry_ = samples_.size();
            count_ = 0;
            last_cpu_ = cpu;
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
