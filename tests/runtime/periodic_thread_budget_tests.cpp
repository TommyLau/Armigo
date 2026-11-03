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

#include <chrono>
#include <cstdio>

using namespace std::chrono_literals;
using Budget = armigo::platform::PeriodicThreadBudget;

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #value); ++failures; \
} } while (false)
}

int main() {
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
    CHECK(budget.period() == 3ms && budget.computation() == 2ms);
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
    return failures ? 1 : 0;
}
