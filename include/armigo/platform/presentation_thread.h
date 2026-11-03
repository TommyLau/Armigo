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

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

// Scheduling for the thread that paces presentation to the display period.
// A periodic time-constraint declaration keeps its wake-ups at the period's
// deadlines; an ordinary thread's timed sleeps are coalesced by milliseconds,
// which moves presents past their slot.

namespace armigo::platform {

// Declares the calling thread periodic: it needs up to `computation` of CPU
// in every `period`. Returns whether the host accepted the declaration; the
// thread keeps its previous policy otherwise.
inline bool declare_periodic_thread(std::chrono::nanoseconds period,
                                    std::chrono::nanoseconds computation) noexcept {
#if defined(__APPLE__)
    if (period <= std::chrono::nanoseconds::zero()) return false;
    mach_timebase_info_data_t timebase{};
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || !timebase.numer) return false;
    const auto ticks = [&](std::chrono::nanoseconds value) {
        return static_cast<std::uint32_t>(static_cast<std::uint64_t>(value.count()) *
                                          timebase.denom / timebase.numer);
    };
    const auto minimum = std::chrono::nanoseconds(1'000'000);
    const auto maximum = period * 9 / 10;
    if (computation < minimum) computation = minimum;
    if (computation > maximum) computation = maximum;
    thread_time_constraint_policy_data_t policy{};
    policy.period = ticks(period);
    policy.computation = ticks(computation);
    policy.constraint = ticks(period);
    policy.preemptible = 1;
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                             reinterpret_cast<thread_policy_t>(&policy),
                             THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS;
#else
    static_cast<void>(period);
    static_cast<void>(computation);
    return false;
#endif
}

// Waits until `deadline` on the steady clock with the host's absolute wait.
inline void wait_until(std::chrono::steady_clock::time_point deadline) noexcept {
#if defined(__APPLE__)
    const auto now = std::chrono::steady_clock::now();
    if (deadline <= now) return;
    mach_timebase_info_data_t timebase{};
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || !timebase.numer) {
        std::this_thread::sleep_until(deadline);
        return;
    }
    const auto delta = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count());
    mach_wait_until(mach_absolute_time() + delta * timebase.denom / timebase.numer);
#else
    std::this_thread::sleep_until(deadline);
#endif
}

} // namespace armigo::platform
