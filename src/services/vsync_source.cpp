// ============================================================================
// Armigo Services - Display Refresh Source
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/vsync_source.h>

#include <armigo/platform/display_capabilities.h>

#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace armigo::services::vi {
namespace {

runtime::LaunchFailure make_failure(runtime::LaunchError code,
                                    const char* operation,
                                    std::string detail) {
    runtime::LaunchFailure failure;
    failure.stage = runtime::LaunchStage::presentation;
    failure.code = code;
    failure.operation = operation;
    failure.detail = std::move(detail);
    return failure;
}

}  // namespace

VsyncSource::VsyncSource(std::chrono::nanoseconds interval) : interval_(interval) {
    auto created_clock = runtime::create_guest_clock();
    if (auto* failure = std::get_if<runtime::LaunchFailure>(&created_clock)) {
        latch_failure(std::move(*failure));
        return;
    }
    guest_clock_ = std::get<std::shared_ptr<runtime::GuestClock>>(
        std::move(created_clock));
    auto sampled = guest_clock_->sample();
    if (auto* failure = std::get_if<runtime::LaunchFailure>(&sampled)) {
        latch_failure(std::move(*failure));
        return;
    }
    runtime::PresentationClockConfiguration configuration;
    configuration.epoch = std::get<runtime::ClockSample>(sampled).instant;
    configuration.refresh_period = interval_;
    auto created_presentation = runtime::create_presentation_clock(
        guest_clock_, configuration);
    if (auto* failure =
            std::get_if<runtime::LaunchFailure>(&created_presentation)) {
        latch_failure(std::move(*failure));
        return;
    }
    presentation_clock_ =
        std::get<std::shared_ptr<runtime::PresentationClock>>(
            std::move(created_presentation));
}

VsyncSource::VsyncSource(std::shared_ptr<runtime::PresentationClock> clock)
    : presentation_clock_(std::move(clock)) {
    if (!presentation_clock_) {
        latch_failure(make_failure(runtime::LaunchError::invalid_input,
                                   "vsync_source.create",
                                   "a presentation clock is required"));
        return;
    }
    interval_ = presentation_clock_->configuration().refresh_period;
    guest_clock_ = presentation_clock_->clock();
    if (interval_ <= std::chrono::nanoseconds::zero()) {
        latch_failure(make_failure(runtime::LaunchError::invalid_input,
                                   "vsync_source.create",
                                   "the refresh period must be positive"));
        return;
    }
    if (!guest_clock_) {
        latch_failure(make_failure(runtime::LaunchError::invalid_input,
                                   "vsync_source.create",
                                   "the presentation clock has no guest clock"));
    }
}

VsyncSource::~VsyncSource() {
    std::jthread thread;
    {
        const std::lock_guard lock(mutex_);
        running_ = false;
        thread = std::move(thread_);
    }
    sequence_changed_.notify_all();
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
}

std::shared_ptr<horizon::Event> VsyncSource::acquire(std::uint64_t display_id, bool& created) {
    const std::lock_guard lock(mutex_);
    created = false;
    auto& event = events_[display_id];
    if (!event) {
        event = std::make_shared<horizon::Event>(false);
        created = true;
    }
    if (!running_ && !failure_) {
        running_ = true;
        try {
            thread_ = std::jthread(
                [this](std::stop_token stop) { run(std::move(stop)); });
        } catch (const std::exception& exception) {
            running_ = false;
            failure_ = make_failure(runtime::LaunchError::host_failure,
                                    "vsync_source.start", exception.what());
        } catch (...) {
            running_ = false;
            failure_ = make_failure(
                runtime::LaunchError::host_failure, "vsync_source.start",
                "the refresh worker could not be started");
        }
        if (failure_) sequence_changed_.notify_all();
    }
    return event;
}

std::uint64_t VsyncSource::sequence() const {
    const std::lock_guard lock(mutex_);
    return sequence_;
}

bool VsyncSource::wait_sequence(std::uint64_t target) {
    std::unique_lock lock(mutex_);
    sequence_changed_.wait(lock, [this, target] {
        return sequence_ >= target || failure_.has_value() || !running_;
    });
    return sequence_ >= target && !failure_;
}

bool VsyncSource::wait_sequence(std::uint64_t target, std::stop_token stop) {
    std::unique_lock lock(mutex_);
    std::stop_callback wake(stop, [this] { sequence_changed_.notify_all(); });
    sequence_changed_.wait(lock, [this, target, &stop] {
        return sequence_ >= target || failure_.has_value() || !running_ ||
               stop.stop_requested();
    });
    return sequence_ >= target && !failure_ && !stop.stop_requested();
}

std::optional<runtime::LaunchFailure> VsyncSource::failure() const {
    const std::lock_guard lock(mutex_);
    return failure_;
}

void VsyncSource::latch_failure(runtime::LaunchFailure failure) {
    {
        const std::lock_guard lock(mutex_);
        if (!failure_) failure_ = std::move(failure);
        running_ = false;
    }
    sequence_changed_.notify_all();
}

void VsyncSource::run(std::stop_token stop) {
    // The guest clock ends each refresh wait at its deadline, so the refresh
    // is signalled then rather than at a coalesced moment after it.
    try {
        // Each refresh index is signalled once: when the grid's phase follows
        // the display, a refresh may move a little later after it was
        // signalled, and asking only for the next future refresh would
        // signal it again.
        std::optional<std::uint64_t> last_index;
        while (!stop.stop_requested()) {
            auto scheduled = last_index ? presentation_clock_->schedule(last_index, 1)
                                        : presentation_clock_->next_refresh();
            if (auto* failure =
                    std::get_if<runtime::LaunchFailure>(&scheduled)) {
                latch_failure(std::move(*failure));
                return;
            }
            const auto deadline =
                std::get<runtime::PresentationSchedule>(scheduled).deadline;
            const auto index =
                std::get<runtime::PresentationSchedule>(scheduled).refresh_index;
            auto waited = guest_clock_->wait_until(deadline, stop);
            if (auto* failure = std::get_if<runtime::LaunchFailure>(&waited)) {
                latch_failure(std::move(*failure));
                return;
            }
            if (std::get<runtime::ClockWait>(waited) ==
                runtime::ClockWait::cancelled) {
                if (!stop.stop_requested()) {
                    latch_failure(make_failure(
                        runtime::LaunchError::cancelled, "vsync_source.wait",
                        "the refresh wait was cancelled without a stop request"));
                }
                return;
            }
            {
                const std::lock_guard lock(mutex_);
                if (stop.stop_requested()) {
                    running_ = false;
                    sequence_changed_.notify_all();
                    return;
                }
                if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
                    failure_ = make_failure(runtime::LaunchError::resource_limit,
                                            "vsync_source.sequence",
                                            "the refresh sequence overflowed");
                    running_ = false;
                    sequence_changed_.notify_all();
                    return;
                }
                ++sequence_;
                for (auto& [display, event] : events_) event->signal();
                presentation_clock_->observe_vsync(std::get<runtime::PresentationSchedule>(scheduled));
            }
            last_index = index;
            sequence_changed_.notify_all();
        }
        {
            const std::lock_guard lock(mutex_);
            running_ = false;
        }
        sequence_changed_.notify_all();
    } catch (const std::exception& exception) {
        latch_failure(make_failure(runtime::LaunchError::host_failure,
                                   "vsync_source.run", exception.what()));
    } catch (...) {
        latch_failure(make_failure(runtime::LaunchError::host_failure,
                                   "vsync_source.run",
                                   "the refresh worker raised an exception"));
    }
}

std::shared_ptr<VsyncSource> shared_vsync_source() {
    // The display's own refresh rate, not an assumed panel: the vsync event and
    // the presentation pacing of an application both follow it.
    static const auto source = [] {
        const auto capabilities = platform::query_display_capabilities();
        return std::make_shared<VsyncSource>(
            std::chrono::nanoseconds(capabilities.refresh_period_ns()));
    }();
    return source;
}

}  // namespace armigo::services::vi
