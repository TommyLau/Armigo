// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/periodic_thread_budget.h>
#include <armigo/platform/presentation_thread.h>
#include <armigo/graphics/presentation_hints.h>

#include <chrono>
#include <cstdio>
#include <limits>

using namespace std::chrono_literals;
using Budget = armigo::platform::PeriodicThreadBudget;

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #value); ++failures; \
} } while (false)
}

int main() {
    using namespace armigo::graphics;
    set_maximum_presentation_rate(0);
    CHECK(presentation_scheduling(0ns, 0ns).period == 0ns);
    CHECK(!presentation_scheduling(0ns, -1ns).guest_cadence);
    const auto two_refreshes = 16'666'666ns * 2;
    CHECK(presentation_scheduling(0ns, two_refreshes).period == two_refreshes);
    CHECK(presentation_scheduling(0ns, two_refreshes).guest_cadence);
    set_maximum_presentation_rate(300);
    CHECK(presentation_scheduling(0ns, two_refreshes).period == 3'333'333ns);
    CHECK(!presentation_scheduling(0ns, two_refreshes).guest_cadence);
    CHECK(presentation_scheduling(20ms, two_refreshes).period == 20ms);
    CHECK(!presentation_scheduling(20ms, two_refreshes).guest_cadence);
    set_maximum_presentation_rate(0);
    CHECK(presentation_scheduling(0ns, 16'666'666ns * 3).period == 49'999'998ns);

    Budget budget;
    Budget::Duration cpu = 1ms;
    unsigned calls = 0;
    bool accept = true;
    const auto apply = [&](Budget::Duration, Budget::Duration) { ++calls; return accept; };
    CHECK(budget.tick(0ns, cpu, apply) == Budget::Result::unchanged);
    CHECK(calls == 0);
    CHECK(budget.tick(10ms, cpu, apply) == Budget::Result::applied);
    CHECK(budget.computation() == 2ms);
    // The initial observation primes the CPU clock; a matching estimate stays
    // inside the hysteresis band and does not make another policy request.
    for (unsigned i = 0; i < 121; ++i) budget.tick(10ms, cpu += 1600us, apply);
    CHECK(calls == 1);
    for (unsigned i = 0; i < 120; ++i) budget.tick(10ms, cpu += 1ms, apply);
    CHECK(budget.computation() == 1250us);
    CHECK(calls == 2);
    // Twelve outliers are above the p90 sample and must not inflate its budget.
    for (unsigned i = 0; i < 120; ++i)
        budget.tick(10ms, cpu += i < 12 ? 8ms : 1ms, apply);
    CHECK(calls == 2);
    for (unsigned i = 0; i < 120; ++i) budget.tick(10ms, cpu += 100us, apply);
    CHECK(budget.computation() == 1ms);

    accept = false;
    CHECK(budget.tick(3ms, cpu, apply) == Budget::Result::rejected);
    CHECK(budget.period() == 10ms && budget.computation() == 1ms);
    const auto rejected_calls = calls;
    accept = true;
    for (unsigned i = 0; i < 119; ++i) budget.tick(3ms, cpu += 1ms, apply);
    CHECK(calls == rejected_calls);
    CHECK(budget.tick(3ms, cpu += 1ms, apply) == Budget::Result::applied);
    CHECK(budget.period() == 3ms && budget.computation() == 1250us);
    for (unsigned i = 0; i < 121; ++i) budget.tick(3ms, cpu += 4ms, apply);
    CHECK(budget.computation() == 2700us);

    accept = false;
    CHECK(budget.tick(0ns, cpu, apply) == Budget::Result::rejected);
    CHECK(budget.period() == 3ms);
    accept = true;
    for (unsigned i = 0; i < 120; ++i) budget.tick(0ns, cpu += 1ms, apply);
    CHECK(budget.period() == 0ns && budget.computation() == 0ns);
    CHECK(budget.tick(500us, cpu, apply) == Budget::Result::applied);
    CHECK(budget.computation() == 450us);
    // A two-refresh deadline admits a long CPU frame without squeezing its
    // budget into a single-refresh constraint.
    Budget cadence_budget;
    CHECK(cadence_budget.tick(two_refreshes, cpu, apply) == Budget::Result::applied);
    CHECK(cadence_budget.computation() == 2ms);
    for (unsigned i = 0; i < 121; ++i)
        cadence_budget.tick(two_refreshes, cpu += 30ms, apply);
    CHECK(cadence_budget.period() == two_refreshes);
    CHECK(cadence_budget.computation() == two_refreshes * 9 / 10);

    Budget long_period;
    CHECK(long_period.tick(83'333'330ns, cpu, apply) == Budget::Result::applied);
    for (unsigned i = 0; i < 121; ++i)
        long_period.tick(83'333'330ns, cpu += 45ms, apply);
    CHECK(long_period.computation() == 50ms);

    Budget recovery;
    Budget::Duration last_request{};
    const auto reject_large = [&](Budget::Duration, Budget::Duration computation) {
        last_request = computation;
        return computation <= 20ms;
    };
    CHECK(recovery.tick(80ms, cpu, reject_large) == Budget::Result::applied);
    for (unsigned i = 0; i < 121; ++i)
        recovery.tick(80ms, cpu += 45ms, reject_large);
    CHECK(last_request == 50ms && recovery.computation() == 2ms);
    // Work falls while the request is rejected: the retry must use these new
    // samples rather than repeatedly resubmitting the rejected high budget.
    for (unsigned i = 0; i < 120; ++i)
        recovery.tick(80ms, cpu += 1ms, reject_large);
    CHECK(last_request == 1250us && recovery.computation() == 1250us);

#if defined(__APPLE__)
    std::thread native_limits([&] {
        CHECK(!armigo::platform::declare_periodic_thread(0ns, 2ms));
        const bool installed = armigo::platform::declare_periodic_thread(83'333'330ns, 56ms);
        CHECK(installed);
        if (installed) {
            thread_time_constraint_policy_data_t observed{};
            mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
            boolean_t is_default = false;
            CHECK(thread_policy_get(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                reinterpret_cast<thread_policy_t>(&observed), &count, &is_default) == KERN_SUCCESS);
            mach_timebase_info_data_t timebase{};
            CHECK(mach_timebase_info(&timebase) == KERN_SUCCESS);
            CHECK(observed.period == 83'333'330ull * timebase.denom / timebase.numer);
            CHECK(!is_default && observed.computation <= 50'000'000ull * timebase.denom / timebase.numer);
        }
        CHECK(!armigo::platform::declare_periodic_thread(Budget::Duration::max(), 2ms));
    });
    native_limits.join();
#endif
    return failures ? 1 : 0;
}
