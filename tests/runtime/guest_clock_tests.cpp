// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/guest_clock.h>

#include <armigo/execution/guest_dispatch.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stop_token>
#include <thread>
#include <utility>
#include <variant>

namespace {
using namespace armigo;
using namespace armigo::runtime;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

// Deterministic injected source: the test moves the timeline by hand and
// scripts wait outcomes. Production waits are exercised only by the small
// dedicated tests at the end.
struct ManualClock {
    std::atomic<std::int64_t> now_ns{0};
    std::atomic<int> fail_waits{0};
    std::atomic<int> wait_calls{0};

    ClockInstant instant() const {
        return ClockInstant{std::chrono::nanoseconds{
            now_ns.load(std::memory_order_relaxed)}};
    }
};

GuestClockSources manual_sources(const std::shared_ptr<ManualClock>& manual) {
    GuestClockSources sources;
    sources.now = [manual] { return manual->instant(); };
    sources.wait = [manual](ClockInstant, std::stop_token stop)
        -> LaunchResult<ClockWait> {
        manual->wait_calls.fetch_add(1, std::memory_order_relaxed);
        if (manual->fail_waits.load(std::memory_order_relaxed) > 0) {
            LaunchFailure failure;
            failure.stage = LaunchStage::services;
            failure.code = LaunchError::host_failure;
            failure.operation = "manual_wait";
            failure.detail = "scripted wait failure";
            return failure;
        }
        return stop.stop_requested() ? ClockWait::cancelled
                                     : ClockWait::reached;
    };
    return sources;
}

std::shared_ptr<GuestClock> manual_clock(ManualClock& manual) {
    auto created = create_guest_clock(manual_sources(
        std::shared_ptr<ManualClock>(&manual, [](ManualClock*) {})));
    // Non-owning view: the tests keep the ManualClock alive on the stack.
    return std::get<std::shared_ptr<GuestClock>>(std::move(created));
}

bool is_failure(const LaunchResult<ClockSample>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && failure->code == code;
}

void test_creation_rejects_half_configured_sources() {
    GuestClockSources only_now;
    only_now.now = [] { return ClockInstant{}; };
    auto created = create_guest_clock(std::move(only_now));
    const auto* failure = std::get_if<LaunchFailure>(&created);
    CHECK(failure && failure->code == LaunchError::invalid_input &&
          failure->stage == LaunchStage::inputs &&
          failure->operation == "create_guest_clock");

    GuestClockSources only_wait;
    only_wait.wait = [](ClockInstant, std::stop_token)
        -> LaunchResult<ClockWait> { return ClockWait::reached; };
    auto created_wait = create_guest_clock(std::move(only_wait));
    CHECK(std::holds_alternative<LaunchFailure>(created_wait));
}

void test_origin_and_sample_rounding() {
    ManualClock manual;
    manual.now_ns.store(1'000);
    auto clock = manual_clock(manual);
    // The origin is sampled at creation: the same source reading still
    // reports zero ticks.
    auto first = clock->sample();
    const auto& baseline = std::get<ClockSample>(first);
    CHECK(baseline.ticks == 0 && baseline.counter_time.count() == 0 &&
          baseline.instant == manual.instant());

    manual.now_ns.store(1'000 + 52);
    auto at52 = std::get<ClockSample>(clock->sample());
    CHECK(at52.ticks == 0 && at52.counter_time.count() == 0);

    manual.now_ns.store(1'000 + 53);
    auto at53 = std::get<ClockSample>(clock->sample());
    CHECK(at53.ticks == 1 && at53.counter_time.count() == 52);

    manual.now_ns.store(1'000 + 625);
    auto at625 = std::get<ClockSample>(clock->sample());
    CHECK(at625.ticks == 12 && at625.counter_time.count() == 625);

    manual.now_ns.store(1'000 + 626);
    auto at626 = std::get<ClockSample>(clock->sample());
    CHECK(at626.ticks == 12 && at626.counter_time.count() == 625);

    // Sampled ticks convert downward: the nanosecond report never runs ahead
    // of the instant it describes.
    CHECK(at626.counter_time <= at626.instant.time_since_epoch());
}

void test_deadline_conversion_rounds_up() {
    ManualClock manual;
    auto clock = manual_clock(manual);
    const std::pair<std::chrono::nanoseconds, std::uint64_t> cases[] = {
        {std::chrono::nanoseconds{0}, 0},
        {std::chrono::nanoseconds{1}, 1},
        {std::chrono::nanoseconds{52}, 1},
        {std::chrono::nanoseconds{53}, 2},
        {std::chrono::nanoseconds{624}, 12},
        {std::chrono::nanoseconds{625}, 12},
        {std::chrono::nanoseconds{626}, 13},
    };
    for (const auto& [duration, ticks] : cases) {
        auto converted = clock->nanoseconds_to_ticks(duration);
        CHECK(std::get<std::uint64_t>(converted) == ticks);
    }
    // The whole signed nanoseconds domain converts; the deadline lands at or
    // after the requested time.
    auto largest = clock->nanoseconds_to_ticks(
        std::chrono::nanoseconds{std::numeric_limits<std::int64_t>::max()});
    CHECK(std::get<std::uint64_t>(largest) == 177'088'743'107'611'696ULL);
}

void test_sample_conversion_rounds_down() {
    ManualClock manual;
    auto clock = manual_clock(manual);
    const std::pair<std::uint64_t, std::int64_t> cases[] = {
        {0, 0},
        {1, 52},
        {2, 104},
        {11, 572},
        {12, 625},
        {13, 677},
        {24, 1250},
    };
    for (const auto& [ticks, nanoseconds] : cases) {
        auto converted = clock->ticks_to_nanoseconds(ticks);
        CHECK(std::get<std::chrono::nanoseconds>(converted).count() ==
              nanoseconds);
    }
    // Rounding identity: ceil(floor(ticks -> ns)) returns the same ticks.
    for (const std::uint64_t ticks : {1ULL, 2ULL, 11ULL, 12ULL, 13ULL, 100ULL,
                                      123'456ULL}) {
        const auto nanoseconds = std::get<std::chrono::nanoseconds>(
            clock->ticks_to_nanoseconds(ticks));
        CHECK(std::get<std::uint64_t>(clock->nanoseconds_to_ticks(
                  nanoseconds)) == ticks);
    }
}

void test_conversion_overflow_is_per_call() {
    ManualClock manual;
    auto clock = manual_clock(manual);
    // Largest tick count whose nanosecond value stays inside int64.
    auto largest = clock->ticks_to_nanoseconds(177'088'743'107'611'695ULL);
    CHECK(std::get<std::chrono::nanoseconds>(largest).count() ==
          9'223'372'036'854'775'781LL);
    auto beyond = clock->ticks_to_nanoseconds(177'088'743'107'611'696ULL);
    const auto* failure = std::get_if<LaunchFailure>(&beyond);
    CHECK(failure && failure->code == LaunchError::resource_limit &&
          failure->operation == "guest_clock.ticks_to_nanoseconds");
    // A conversion overflow is a caller-input outcome: the clock itself stays
    // healthy and keeps sampling.
    auto still_healthy = clock->sample();
    CHECK(std::holds_alternative<ClockSample>(still_healthy));

    auto negative = clock->nanoseconds_to_ticks(std::chrono::nanoseconds{-1});
    const auto* invalid = std::get_if<LaunchFailure>(&negative);
    CHECK(invalid && invalid->code == LaunchError::invalid_input);
    CHECK(std::holds_alternative<ClockSample>(clock->sample()));
}

void test_regression_latches_and_freezes_the_counter() {
    ManualClock manual;
    manual.now_ns.store(100);
    auto clock = manual_clock(manual);
    manual.now_ns.store(153);
    auto healthy = std::get<ClockSample>(clock->sample());
    CHECK(healthy.ticks == 1 && healthy.counter_time.count() == 52);

    manual.now_ns.store(50);
    auto regressed = clock->sample();
    const auto* failure = std::get_if<LaunchFailure>(&regressed);
    CHECK(failure && failure->code == LaunchError::host_failure &&
          failure->stage == LaunchStage::services &&
          failure->operation == "guest_clock.sample");

    // Every operation now reports the latched failure; nothing fabricates a
    // healthy clock.
    CHECK(is_failure(clock->sample(), LaunchError::host_failure));
    CHECK(std::holds_alternative<LaunchFailure>(
        clock->ticks_to_nanoseconds(12)));
    CHECK(std::holds_alternative<LaunchFailure>(
        clock->nanoseconds_to_ticks(std::chrono::nanoseconds{625})));
    CHECK(std::holds_alternative<LaunchFailure>(clock->wait_until(
        manual.instant(), std::stop_token{})));

    // The no-throw counter adapter freezes at the last valid value and never
    // advances again, even while the injected source keeps moving.
    auto counter = clock->counter();
    CHECK(execution::validate(counter) == execution::GuestCounterError::none);
    CHECK(counter.frequency == 19'200'000);
    CHECK(counter.read_ticks(counter.context) == 1);
    manual.now_ns.store(1'000'000);
    CHECK(counter.read_ticks(counter.context) == 1);
    CHECK(counter.read_ticks(counter.context) == 1);
}

void test_wait_failure_latches_and_the_counter_stays_at_zero() {
    ManualClock manual;
    manual.fail_waits.store(1);
    auto clock = manual_clock(manual);
    auto failed = clock->wait_until(manual.instant(), std::stop_token{});
    const auto* failure = std::get_if<LaunchFailure>(&failed);
    CHECK(failure && failure->code == LaunchError::host_failure &&
          failure->operation == "manual_wait");
    manual.fail_waits.store(0);
    // The latched failure is the first causal one, preserved in full.
    auto after = clock->sample();
    const auto* latched = std::get_if<LaunchFailure>(&after);
    CHECK(latched && latched->code == LaunchError::host_failure &&
          latched->operation == "manual_wait");

    // No valid sample was ever taken: the adapter stays frozen at zero.
    auto counter = clock->counter();
    manual.now_ns.store(500);
    CHECK(counter.read_ticks(counter.context) == 0);
    CHECK(counter.read_ticks(counter.context) == 0);
}

void test_injected_wait_passthrough() {
    ManualClock manual;
    auto clock = manual_clock(manual);
    std::stop_source cancelled;
    cancelled.request_stop();
    CHECK(std::get<ClockWait>(clock->wait_until(
              manual.instant() + std::chrono::seconds{1},
              cancelled.get_token())) == ClockWait::cancelled);
    CHECK(std::get<ClockWait>(clock->wait_until(
              manual.instant(), std::stop_token{})) == ClockWait::reached);
    CHECK(manual.wait_calls.load() == 2);
}

void test_production_sources() {
    auto created = create_guest_clock();
    auto clock = std::get<std::shared_ptr<GuestClock>>(std::move(created));
    const auto first = std::get<ClockSample>(clock->sample());
    const auto second = std::get<ClockSample>(clock->sample());
    CHECK(second.instant >= first.instant && second.ticks >= first.ticks);

    // An expired deadline is reached without waiting; an already-stopped
    // token cancels without waiting.
    CHECK(std::get<ClockWait>(clock->wait_until(
              std::chrono::steady_clock::now(), std::stop_token{})) ==
          ClockWait::reached);
    std::stop_source stopped;
    stopped.request_stop();
    CHECK(std::get<ClockWait>(clock->wait_until(
              std::chrono::steady_clock::now() + std::chrono::hours{1},
              stopped.get_token())) == ClockWait::cancelled);

    // Cancellation wakes a real waiter.
    std::stop_source stop;
    LaunchResult<ClockWait> outcome{LaunchFailure{}};
    std::thread waiter([&] {
        outcome = clock->wait_until(
            std::chrono::steady_clock::now() + std::chrono::hours{1},
            stop.get_token());
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    stop.request_stop();
    waiter.join();
    CHECK(std::holds_alternative<ClockWait>(outcome) &&
          std::get<ClockWait>(outcome) == ClockWait::cancelled);

    // A short real deadline is reached.
    CHECK(std::get<ClockWait>(clock->wait_until(
              std::chrono::steady_clock::now() + std::chrono::milliseconds{1},
              std::stop_token{})) == ClockWait::reached);
}

} // namespace

int main() {
    test_creation_rejects_half_configured_sources();
    test_origin_and_sample_rounding();
    test_deadline_conversion_rounds_up();
    test_sample_conversion_rounds_down();
    test_conversion_overflow_is_per_call();
    test_regression_latches_and_freezes_the_counter();
    test_wait_failure_latches_and_the_counter_stays_at_zero();
    test_injected_wait_passthrough();
    test_production_sources();
    return failures == 0 ? 0 : 1;
}
