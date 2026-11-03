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

#include <armigo/runtime/guest_clock.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

namespace armigo::runtime {

// Lane E implements a refresh grid on the SAME injected monotonic timeline.
// Lane G uses it for both VsyncSource signaling and queue pacing, with distinct
// events. This is host display policy, not a claim of recovered firmware timing.
// There must be no independent sleep in H.
// Zero interval is immediate; positive intervals target the first grid point
// at/after now and at least interval refreshes after the last completed target.
// Late frames skip missed grid points without accumulating drift. Negative
// intervals, invalid periods and arithmetic overflow return structured errors.
struct PresentationSchedule {
    ClockInstant deadline{};
    std::uint64_t refresh_index = 0;
    bool immediate = false;
};
struct PresentationClockConfiguration {
    ClockInstant epoch{};
    // Explicit host display mode; zero is invalid, not an assumed 60 Hz.
    std::chrono::nanoseconds refresh_period{};
};
class PresentationClock {
public:
    virtual ~PresentationClock() = default;
    virtual std::shared_ptr<GuestClock> clock() const noexcept = 0;
    virtual PresentationClockConfiguration configuration() const noexcept = 0;
    // Rebase the grid at the current completed refresh with a new period.
    // Refresh indices remain continuous for streams retaining a last index.
    // Clocks without runtime rate control return false and stay unchanged.
    virtual bool set_refresh_period(std::chrono::nanoseconds period) noexcept {
        static_cast<void>(period);
        return false;
    }
    // Stateless planning per stream. Caller updates last_refresh only after a
    // completed presentation. An immediate completion records its current index.
    virtual LaunchResult<PresentationSchedule> schedule(
        std::optional<std::uint64_t> last_refresh, std::int32_t swap_interval) const = 0;
    // Next strictly future refresh, for display events independent of buffers.
    virtual LaunchResult<PresentationSchedule> next_refresh() const = 0;
    // A refresh of the display the presents are shown on: the vblank time on
    // the steady clock and the display's current refresh period. When the
    // display's period divides the grid's evenly, the grid's phase follows the
    // display so grid points sit just after a vblank and every present lands
    // on the same vblank (60 Hz content on a 60, 120 or 240 Hz display); the
    // grid's period never changes. Other rates are ignored. Thread-safe
    // against the planning calls; optional.
    virtual void observe_display_refresh(ClockInstant vblank, std::chrono::nanoseconds display_period) noexcept {
        static_cast<void>(vblank);
        static_cast<void>(display_period);
    }
    // Once the grid follows a display: the grid point nearest the instant, for
    // a producer that paces presents on its own cadence. Empty before that
    // (and always without a display), so such a cadence is left as it is.
    virtual std::optional<ClockInstant> display_refresh_near(ClockInstant instant) const noexcept {
        static_cast<void>(instant);
        return std::nullopt;
    }
};

[[nodiscard]] LaunchResult<std::shared_ptr<PresentationClock>> create_presentation_clock(
    std::shared_ptr<GuestClock> clock, PresentationClockConfiguration configuration);

} // namespace armigo::runtime
