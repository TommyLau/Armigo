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

#include <armigo/execution/guest_dispatch.h>
#include <armigo/runtime/launch_result.h>
#include <armigo/services/shared_clock.h>
#include <chrono>
#include <optional>
#include <functional>
#include <memory>
#include <stop_token>

namespace armigo::runtime {

// Lane E supplies one injected monotonic epoch to every guest thread, time
// service, shared clock and timeout conversion. The existing 20.5.0 contract
// uses 19.2 MHz; frequency changes and clock regression
// are unsupported. Wall time never drives deadlines. Overflow is an error,
// not wraparound; ns->ticks deadlines round up, sampled ticks->ns round down.
using ClockInstant = std::chrono::steady_clock::time_point;
struct ClockSample {
    ClockInstant instant{};
    std::uint64_t ticks = 0;
    std::chrono::nanoseconds counter_time{};
};
enum class ClockWait { reached, cancelled };
class GuestClock {
public:
    virtual ~GuestClock() = default;
    virtual LaunchResult<ClockSample> sample() const = 0;
    // Borrowed callback/context; clock lifetime covers every dispatch scope.
    virtual execution::GuestCounter counter() noexcept = 0;
    virtual LaunchResult<std::chrono::nanoseconds> ticks_to_nanoseconds(
        std::uint64_t ticks) const = 0;
    virtual LaunchResult<std::uint64_t> nanoseconds_to_ticks(
        std::chrono::nanoseconds duration) const = 0;
    virtual LaunchResult<ClockWait> wait_until(ClockInstant deadline,
                                              std::stop_token stop) = 0;
};

struct GuestClockSources {
    // Either both empty (production monotonic host clock), or both provided
    // (deterministic tests). They share one time-point domain and are thread-safe.
    std::function<ClockInstant()> now;
    std::function<LaunchResult<ClockWait>(ClockInstant, std::stop_token)> wait;
};
// Samples one origin; counter starts at zero. All conversions and shared-clock
// initialization use that origin. Source errors latch a structured clock failure;
// the no-throw GuestCounter adapter must not fabricate advancing valid ticks.
[[nodiscard]] LaunchResult<std::shared_ptr<GuestClock>> create_guest_clock(
    GuestClockSources sources = {});

} // namespace armigo::runtime
