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
#include <armigo/execution/guest_timeline.h>
#include <armigo/platform/presentation_thread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

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

void test_platform_counter_rates() {
    ManualClock manual;
    manual.now_ns.store(1'000);
    auto clock = manual_clock(manual);
    manual.now_ns.store(1'000 + 1'000);
    auto switch1 = clock->counter();
    auto rated1 = clock->counter(horizon::switch1_counter);
    CHECK(execution::validate(switch1) == execution::GuestCounterError::none);
    CHECK(execution::validate(rated1) == execution::GuestCounterError::none);
    CHECK(switch1.frequency == 19'200'000);
    CHECK(rated1.frequency == 19'200'000);
    const auto sampled = std::get<ClockSample>(clock->sample());
    CHECK(switch1.read_ticks(switch1.context) == sampled.ticks);
    CHECK(rated1.read_ticks(rated1.context) == sampled.ticks);

    auto rated2 = clock->counter(horizon::switch2_counter);
    CHECK(execution::validate(rated2) == execution::GuestCounterError::none);
    CHECK(rated2.frequency == 31'250'000);
    CHECK(rated2.read_ticks(rated2.context) == 1'000 / 32);

    manual.now_ns.store(1'000 + 32'000);
    CHECK(rated2.read_ticks(rated2.context) == 32'000 / 32);
    const auto at_32us = std::get<ClockSample>(clock->sample());
    CHECK(switch1.read_ticks(switch1.context) == at_32us.ticks);
    CHECK(at_32us.ticks != 32'000 / 32);

    horizon::CounterRate other{1, 1, 1};
    auto empty = clock->counter(other);
    CHECK(execution::validate(empty) == execution::GuestCounterError::missing_tick_source);

    const auto frozen2 = rated2.read_ticks(rated2.context);
    const auto frozen19 = switch1.read_ticks(switch1.context);
    manual.now_ns.store(500);
    CHECK(rated2.read_ticks(rated2.context) == frozen2);
    CHECK(switch1.read_ticks(switch1.context) == frozen19);
    CHECK(std::holds_alternative<LaunchFailure>(clock->sample()));
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

#if defined(__APPLE__)
// The period of the calling thread's time-constraint policy, if it has one.
std::optional<std::uint32_t> constraint_period() {
    thread_time_constraint_policy_data_t policy{};
    mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
    boolean_t get_default = FALSE;
    if (thread_policy_get(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                          reinterpret_cast<thread_policy_t>(&policy), &count, &get_default) != KERN_SUCCESS ||
        get_default)
        return std::nullopt;
    return policy.period;
}
#endif

#if defined(__APPLE__)
// How many threads of this process carry the clock alarm's name.
int alarm_threads() {
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) return -1;
    int found = 0;
    for (mach_msg_type_number_t i = 0; i < count; ++i) {
        char name[64] = {};
        if (const pthread_t thread = pthread_from_mach_thread_np(threads[i]);
            thread && pthread_getname_np(thread, name, sizeof name) == 0 &&
            std::string_view(name) == "armigo-clock-alarm")
            ++found;
        mach_port_deallocate(mach_task_self(), threads[i]);
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), count * sizeof(thread_act_t));
    return found;
}
#endif

bool reached(const LaunchResult<ClockWait>& result) {
    return std::holds_alternative<ClockWait>(result) && std::get<ClockWait>(result) == ClockWait::reached;
}

// The waiting thread's scheduling is never changed by a wait: an interactive
// thread keeps its QoS class without a time constraint, and a thread that
// holds its own constraint (the guest-present thread) keeps that one.
void test_production_waits_leave_the_waiter_scheduling() {
#if defined(__APPLE__)
    auto clock = std::get<std::shared_ptr<GuestClock>>(create_guest_clock());
    std::thread([&] {
        CHECK(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0);
        for (int i = 0; i < 3; ++i)
            CHECK(reached(clock->wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds{2},
                                            std::stop_token{})));
        CHECK(qos_class_self() == QOS_CLASS_USER_INTERACTIVE);
        CHECK(!constraint_period());
    }).join();
    std::thread([&] {
        CHECK(platform::declare_periodic_thread(std::chrono::nanoseconds{8'333'333},
                                                std::chrono::nanoseconds{1'000'000}));
        const auto before = constraint_period();
        CHECK(before.has_value());
        CHECK(reached(clock->wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds{2},
                                        std::stop_token{})));
        CHECK(constraint_period() == before);
    }).join();
#endif
}

// Concurrent waiters are each woken at or after their own deadline, never
// before; a cancelled waiter's far deadline does not hold the clock.
void test_production_deadlines_wake_each_waiter() {
    auto clock = std::get<std::shared_ptr<GuestClock>>(create_guest_clock());
    std::atomic<int> early{0};
    std::atomic<int> done{0};
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> waiters;
    for (int i = 4; i >= 1; --i)
        waiters.emplace_back([&, i] {
            const auto deadline = start + std::chrono::milliseconds{5 * i};
            const auto result = clock->wait_until(deadline, std::stop_token{});
            if (std::chrono::steady_clock::now() < deadline) early.fetch_add(1);
            if (reached(result)) done.fetch_add(1);
        });
    for (auto& waiter : waiters) waiter.join();
    CHECK(early.load() == 0);
    CHECK(done.load() == 4);

    std::stop_source stop;
    std::thread far([&] {
        CHECK(std::get<ClockWait>(clock->wait_until(std::chrono::steady_clock::now() + std::chrono::hours{1},
                                                    stop.get_token())) == ClockWait::cancelled);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    stop.request_stop();
    far.join();
    const auto destroyed = std::chrono::steady_clock::now();
    clock.reset();
    CHECK(std::chrono::steady_clock::now() - destroyed < std::chrono::seconds{1});
}

// Waits of at most 500 us end at the host's own wake-up and never start the
// alarm; a longer wait starts one alarm thread, which ends with its clock.
void test_short_waits_leave_the_alarm_idle() {
#if defined(__APPLE__)
    CHECK(alarm_threads() == 0);
    auto clock = std::get<std::shared_ptr<GuestClock>>(create_guest_clock());
    for (int i = 0; i < 20; ++i)
        CHECK(reached(clock->wait_until(std::chrono::steady_clock::now() + std::chrono::microseconds{200},
                                        std::stop_token{})));
    CHECK(alarm_threads() == 0);
    CHECK(reached(clock->wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds{5},
                                    std::stop_token{})));
    CHECK(alarm_threads() == 1);
    clock.reset();
    CHECK(alarm_threads() == 0);
#endif
}

// A frame-length wait on an interactive thread ends at its deadline. Without
// the alarm the host ended such waits 5 to 8 ms late (median above 5 ms).
void test_production_waits_end_on_time() {
#if defined(__APPLE__)
    auto clock = std::get<std::shared_ptr<GuestClock>>(create_guest_clock());
    std::vector<double> late;
    std::thread([&] {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        for (int i = 0; i < 12; ++i) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds{16'666'667};
            CHECK(reached(clock->wait_until(deadline, std::stop_token{})));
            late.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - deadline).count());
        }
    }).join();
    std::sort(late.begin(), late.end());
    std::fprintf(stderr, "16.7 ms waits ended late by median %.3f ms, max %.3f ms\n", late[late.size() / 2], late.back());
    CHECK(late[late.size() / 2] < 3.0);
#endif
}

} // namespace

// The published guest timeline answers from its source until that source is
// withdrawn; withdrawing another context leaves it in place.
void test_guest_timeline_publication() {
    using armigo::execution::GuestTimelineSource;
    CHECK(!armigo::execution::guest_timeline_now() && !armigo::execution::guest_timeline_origin());
    static std::int64_t elapsed = 5000;
    int first_context = 0, second_context = 0;
    const auto origin = std::chrono::steady_clock::time_point(std::chrono::seconds(7));
    const auto read = [](void* context) noexcept -> std::optional<std::chrono::nanoseconds> {
        return std::chrono::nanoseconds(elapsed + (context ? 0 : 1));
    };
    armigo::execution::publish_guest_timeline(GuestTimelineSource{read, &first_context, origin});
    CHECK(armigo::execution::guest_timeline_now() == std::chrono::nanoseconds(5000));
    CHECK(armigo::execution::guest_timeline_origin() == origin);
    armigo::execution::publish_guest_timeline(GuestTimelineSource{read, &second_context, origin + std::chrono::seconds(1)});
    armigo::execution::withdraw_guest_timeline(&first_context);
    CHECK(armigo::execution::guest_timeline_origin() == origin + std::chrono::seconds(1));
    elapsed = 6000;
    CHECK(armigo::execution::guest_timeline_now() == std::chrono::nanoseconds(6000));
    armigo::execution::withdraw_guest_timeline(&second_context);
    CHECK(!armigo::execution::guest_timeline_now() && !armigo::execution::guest_timeline_origin());
}

int main() {
    test_guest_timeline_publication();
    test_creation_rejects_half_configured_sources();
    test_origin_and_sample_rounding();
    test_deadline_conversion_rounds_up();
    test_sample_conversion_rounds_down();
    test_conversion_overflow_is_per_call();
    test_regression_latches_and_freezes_the_counter();
    test_platform_counter_rates();
    test_wait_failure_latches_and_the_counter_stays_at_zero();
    test_injected_wait_passthrough();
    test_production_sources();
    test_production_waits_leave_the_waiter_scheduling();
    test_production_deadlines_wake_each_waiter();
    test_short_waits_leave_the_alarm_idle();
    test_production_waits_end_on_time();
    return failures == 0 ? 0 : 1;
}
