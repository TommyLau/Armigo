// ============================================================================
// Armigo Services - Display Refresh Source
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/event.h>
#include <armigo/runtime/presentation_clock.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>

namespace armigo::services::vi {

// The display refresh source behind `GetDisplayVsyncEvent` (vi command 5202).
// Each display has one readable event that the source signals once per refresh
// interval; the application waits on it to pace frame presentation.
//
// The refresh interval is host display policy. The vi display mode this runtime
// reports is 1280x720 at 60 Hz, so the default interval is one 60 Hz period.
class VsyncSource final {
public:
    explicit VsyncSource(std::chrono::nanoseconds interval = std::chrono::nanoseconds(16'666'667));
    explicit VsyncSource(std::shared_ptr<runtime::PresentationClock> clock);
    ~VsyncSource();

    VsyncSource(const VsyncSource&) = delete;
    VsyncSource& operator=(const VsyncSource&) = delete;

    // Returns the display's event, creating it on the first request. The thread
    // starts with the first event and signals every registered event until the
    // source is destroyed.
    std::shared_ptr<horizon::Event> acquire(std::uint64_t display_id, bool& created);

    // Live host policy: the presentation clock can change its refresh period
    // after this source is created, for example when the display rate changes.
    std::chrono::nanoseconds interval() const noexcept {
        return presentation_clock_ ? presentation_clock_->configuration().refresh_period : interval_;
    }
    // Once the refresh grid follows the window's display, the grid point
    // nearest `slot`, so a present paced on its own cadence lands on the
    // display's vblanks too; empty otherwise (and always headless).
    std::optional<std::chrono::steady_clock::time_point> display_slot(
        std::chrono::steady_clock::time_point slot) const noexcept {
        return presentation_clock_ ? presentation_clock_->display_refresh_near(slot) : std::nullopt;
    }

    std::shared_ptr<runtime::PresentationClock> imposed_clock() const noexcept {
        return presentation_clock_ && presentation_clock_->imposed_pacing() ? presentation_clock_ : nullptr;
    }

    // The sequence advances once for every refresh without consuming or
    // resetting any display event. Waiting returns false if the source fails.
    std::uint64_t sequence() const;
    bool wait_sequence(std::uint64_t target);
    // Waits for the same refresh sequence but returns false as soon as the
    // caller's stop token is set, so a presentation thread can be shut down
    // without waiting for the next refresh.
    bool wait_sequence(std::uint64_t target, std::stop_token stop);

    // Reports a construction or asynchronous clock failure. The first causal
    // failure is retained so an invalid or stopped source cannot look healthy.
    std::optional<runtime::LaunchFailure> failure() const;

private:
    void run(std::stop_token stop);
    void latch_failure(runtime::LaunchFailure failure);

    std::shared_ptr<runtime::PresentationClock> presentation_clock_;
    std::shared_ptr<runtime::GuestClock> guest_clock_;
    std::chrono::nanoseconds interval_{};
    mutable std::mutex mutex_;
    std::condition_variable sequence_changed_;
    std::map<std::uint64_t, std::shared_ptr<horizon::Event>> events_;
    std::jthread thread_;
    std::optional<runtime::LaunchFailure> failure_;
    std::uint64_t sequence_ = 0;
    bool running_ = false;
};

// The process-wide display refresh source. One panel serves every display session,
// so repeated sessions share the same per-display events.
std::shared_ptr<VsyncSource> shared_vsync_source();

}  // namespace armigo::services::vi
