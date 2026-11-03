// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/frame_rate_governor.h>

#include <chrono>
#include <cstdio>

namespace {

using armigo::runtime::AutomaticFrameRateGovernor;
using Clock = std::chrono::steady_clock;
unsigned failures = 0;

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

Clock::time_point at(double seconds) {
    return Clock::time_point(std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(seconds)));
}

void test_scene_and_startup_grace() {
    AutomaticFrameRateGovernor governor;
    CHECK(!governor.observe(at(0), 0, 0));
    CHECK(!governor.observe(at(30), 900, 0));
    CHECK(governor.rate() == 60);
    CHECK(!governor.observe(at(61), 3660, 0));
    const auto changed = governor.observe(at(71), 3660 + 180, 0);
    CHECK(changed && *changed == 30);
    CHECK(governor.rate() == 30);
}

void test_thermal_fallback_and_recovery() {
    AutomaticFrameRateGovernor governor;
    governor.observe(at(0), 0, 0);
    const auto fair = governor.observe(at(2), 120, 1);
    CHECK(fair && *fair == 30);
    CHECK(!governor.observe(at(3), 150, 0));
    CHECK(!governor.observe(at(302), 120 + 300 * 30, 0));
    CHECK(governor.rate() == 30);
    const auto up = governor.observe(at(303), 120 + 301 * 30, 0);
    CHECK(up && *up == 60);
    const auto down = governor.observe(at(310), 120 + 301 * 30 + 7 * 60, 1);
    CHECK(down && *down == 30);
    CHECK(!governor.observe(at(311), 120 + 301 * 30 + 7 * 60 + 30, 0));
    CHECK(!governor.observe(at(909), 120 + 301 * 30 + 7 * 60 + 599 * 30, 0));
    CHECK(!governor.observe(at(910), 120 + 301 * 30 + 7 * 60 + 600 * 30, 0));
    const auto second_up = governor.observe(at(911), 120 + 301 * 30 + 7 * 60 + 601 * 30, 0);
    CHECK(second_up && *second_up == 60);
}

void test_serious_hold() {
    AutomaticFrameRateGovernor governor;
    governor.observe(at(0), 0, 0);
    CHECK(governor.observe(at(1), 60, 2) == 30);
    CHECK(!governor.observe(at(120), 60 + 119 * 30, 0));
    CHECK(!governor.observe(at(419), 60 + 418 * 30, 0));
    const auto up = governor.observe(at(421), 60 + 420 * 30, 0);
    CHECK(up && *up == 60);
}

} // namespace

int main() {
    test_scene_and_startup_grace();
    test_thermal_fallback_and_recovery();
    test_serious_hold();
    return failures == 0 ? 0 : 1;
}
