// ============================================================================
// Armigo Tests - Display Refresh Source
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/vsync_source.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace std::chrono_literals;
using armigo::runtime::ClockInstant;
using armigo::runtime::ClockWait;
using armigo::runtime::GuestClock;
using armigo::runtime::GuestClockSources;
using armigo::runtime::LaunchError;
using armigo::runtime::LaunchFailure;
using armigo::runtime::LaunchResult;
using armigo::runtime::LaunchStage;
using armigo::runtime::PresentationClock;
using armigo::runtime::PresentationClockConfiguration;
using armigo::services::vi::VsyncSource;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

struct ManualTimeline {
    ClockInstant now() const {
        const std::lock_guard lock(mutex);
        return instant_locked();
    }

    LaunchResult<ClockWait> wait_until(ClockInstant deadline,
                                       std::stop_token stop) {
        std::unique_lock lock(mutex);
        deadlines.push_back(deadline);
        changed.notify_all();
        if (fail_wait) {
            LaunchFailure failure;
            failure.stage = LaunchStage::presentation;
            failure.code = LaunchError::host_failure;
            failure.operation = "manual_timeline.wait";
            failure.detail = "injected refresh wait failure";
            return failure;
        }
        const bool reached = changed.wait(
            lock, stop, [&] { return instant_locked() >= deadline; });
        if (!reached) {
            cancelled = true;
            changed.notify_all();
            return ClockWait::cancelled;
        }
        return ClockWait::reached;
    }

    void advance_to(std::int64_t nanoseconds) {
        const std::lock_guard lock(mutex);
        now_ns = nanoseconds;
        changed.notify_all();
    }

    bool wait_for_deadlines(std::size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s,
                                [&] { return deadlines.size() >= count; });
    }

    bool wait_for_cancellation() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s, [&] { return cancelled; });
    }

    std::optional<std::int64_t> deadline(std::size_t index) const {
        const std::lock_guard lock(mutex);
        if (index >= deadlines.size()) return std::nullopt;
        return deadlines[index].time_since_epoch().count();
    }

    mutable std::mutex mutex;
    std::condition_variable_any changed;
    std::int64_t now_ns = 0;
    std::vector<ClockInstant> deadlines;
    bool fail_wait = false;
    bool cancelled = false;

private:
    ClockInstant instant_locked() const {
        return ClockInstant{std::chrono::nanoseconds{now_ns}};
    }
};

std::shared_ptr<PresentationClock> presentation_clock(
    const std::shared_ptr<ManualTimeline>& timeline,
    std::chrono::nanoseconds period = 10ns) {
    GuestClockSources sources;
    sources.now = [timeline] { return timeline->now(); };
    sources.wait = [timeline](ClockInstant deadline, std::stop_token stop) {
        return timeline->wait_until(deadline, stop);
    };
    auto created_guest = armigo::runtime::create_guest_clock(std::move(sources));
    CHECK(std::holds_alternative<std::shared_ptr<GuestClock>>(created_guest));
    if (auto* guest =
            std::get_if<std::shared_ptr<GuestClock>>(&created_guest)) {
        PresentationClockConfiguration configuration;
        configuration.epoch = ClockInstant{};
        configuration.refresh_period = period;
        auto created = armigo::runtime::create_presentation_clock(
            *guest, configuration);
        CHECK(std::holds_alternative<std::shared_ptr<PresentationClock>>(created));
        if (auto* clock =
                std::get_if<std::shared_ptr<PresentationClock>>(&created)) {
            return *clock;
        }
    }
    return {};
}

std::optional<LaunchFailure> wait_for_failure(VsyncSource& source) {
    const auto end = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < end) {
        if (auto failure = source.failure()) return failure;
        std::this_thread::yield();
    }
    return std::nullopt;
}

void test_shared_grid_events_skip_missed_refreshes() {
    auto timeline = std::make_shared<ManualTimeline>();
    std::shared_ptr<armigo::horizon::Event> first;
    {
        VsyncSource source(presentation_clock(timeline));
        CHECK(!source.failure());
        CHECK(source.interval() == 10ns);

        bool created = false;
        first = source.acquire(3, created);
        CHECK(first && created);
        bool repeated_created = true;
        auto repeated = source.acquire(3, repeated_created);
        CHECK(repeated == first && !repeated_created);

        CHECK(timeline->wait_for_deadlines(1));
        CHECK(timeline->deadline(0) == 10);
        timeline->advance_to(55);
        CHECK(timeline->wait_for_deadlines(2));
        CHECK(timeline->deadline(1) == 60);
        CHECK(first->is_signaled());
        CHECK(first->reset());

        bool second_created = false;
        auto second = source.acquire(9, second_created);
        CHECK(second && second_created && second != first);
        CHECK(!second->is_signaled());
        timeline->advance_to(60);
        CHECK(timeline->wait_for_deadlines(3));
        CHECK(timeline->deadline(2) == 70);
        CHECK(first->is_signaled());
        CHECK(second->is_signaled());
        CHECK(!source.failure());
    }
    CHECK(timeline->wait_for_cancellation());
}

// When the grid's phase follows the display, a refresh can move later after
// it was signalled; the source never signals the same refresh twice.
void test_refresh_moved_later_is_not_signalled_again() {
    auto timeline = std::make_shared<ManualTimeline>();
    constexpr std::int64_t period = 16'666'667;
    auto clock = presentation_clock(timeline, std::chrono::nanoseconds{period});
    {
        VsyncSource source(clock);
        bool created = false;
        auto event = source.acquire(1, created);
        CHECK(event && created);
        CHECK(timeline->wait_for_deadlines(1));
        CHECK(timeline->deadline(0) == period);
        // Not yet following a display: no slot is offered to a producer.
        CHECK(!source.display_slot(std::chrono::steady_clock::time_point{std::chrono::nanoseconds{period}}));
        // A 60 Hz display whose refreshes want the grid 1 ms later (a vblank
        // half a millisecond after the first refresh): one step of 0.2 ms.
        clock->observe_display_refresh(ClockInstant{std::chrono::nanoseconds{500'000 + period}},
                                       std::chrono::nanoseconds{period});
        timeline->advance_to(period);
        CHECK(timeline->wait_for_deadlines(2));
        CHECK(timeline->deadline(1) == 2 * period + 200'000);
        CHECK(event && event->is_signaled());
        // Now following it: a producer's slot snaps to the nearest refresh.
        const auto slot = source.display_slot(
            std::chrono::steady_clock::time_point{std::chrono::nanoseconds{3 * period + 5'000'000}});
        CHECK(slot && slot->time_since_epoch().count() == 3 * period + 200'000);
    }
    CHECK(timeline->wait_for_cancellation());
}

void test_invalid_sources_report_failures_without_starting() {
    VsyncSource missing(std::shared_ptr<PresentationClock>{});
    auto missing_failure = missing.failure();
    CHECK(missing_failure &&
          missing_failure->code == LaunchError::invalid_input &&
          missing_failure->operation == "vsync_source.create");
    bool created = false;
    auto event = missing.acquire(1, created);
    CHECK(event && created && !event->is_signaled());

    VsyncSource invalid_period(0ns);
    auto period_failure = invalid_period.failure();
    CHECK(period_failure &&
          period_failure->code == LaunchError::invalid_input &&
          period_failure->operation == "create_presentation_clock");
}

void test_asynchronous_clock_failure_is_latched() {
    auto timeline = std::make_shared<ManualTimeline>();
    timeline->fail_wait = true;
    VsyncSource source(presentation_clock(timeline));
    bool created = false;
    auto event = source.acquire(4, created);
    CHECK(event && created);
    CHECK(timeline->wait_for_deadlines(1));
    auto failure = wait_for_failure(source);
    CHECK(failure && failure->code == LaunchError::host_failure &&
          failure->operation == "manual_timeline.wait");
    CHECK(!event->is_signaled());
}

}  // namespace

int main() {
    test_shared_grid_events_skip_missed_refreshes();
    test_refresh_moved_later_is_not_signalled_again();
    test_invalid_sources_report_failures_without_starting();
    test_asynchronous_clock_failure_is_latched();
    if (failures) {
        std::fprintf(stderr, "%u VsyncSource test(s) failed\n", failures);
        return 1;
    }
    std::puts("All VsyncSource tests passed");
    return 0;
}
