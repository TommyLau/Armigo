// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Conditional smoke test for poll_apple_input. It links the real GameController
// framework through the polled library but requires no physical hardware. It
// checks only snapshot-local invariants that hold on every machine: a
// disconnected or unsupported state carries no inputs, a supported state is
// finite and in range, and the flags agree with each other. It deliberately
// never compares a returned snapshot against a later framework enumeration,
// because a device hot-plug between the poll and the enumeration would race.

#import <armigo/platform/apple_input.h>
#import <armigo/platform/host_input.h>

#include <array>
#include <cstdio>

namespace {
using namespace armigo::platform;

int g_failed = 0;
int g_total = 0;

#define CHECK(condition)                                                                \
    do {                                                                                \
        ++g_total;                                                                      \
        if (!(condition)) {                                                             \
            ++g_failed;                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
        }                                                                               \
    } while (false)

bool stick_in_range(const std::array<float, 2>& stick) {
    // A NaN component fails the comparison, so NaN is rejected too.
    for (const float component : stick) {
        if (!(component >= -1.0f && component <= 1.0f)) return false;
    }
    return true;
}

bool trigger_in_range(float value) {
    return value >= 0.0f && value <= 1.0f;
}

void check_state_consistent(const HostInputDeviceState& state) {
    if (!state.connected) {
        CHECK(!state.supported);
        CHECK(state.buttons == 0);
        CHECK(stick_in_range(state.left_stick) && stick_in_range(state.right_stick));
        CHECK(state.left_trigger == 0.0f && state.right_trigger == 0.0f);
    }
    // Connected without the required profile reports no inputs, never a
    // phantom working state.
    if (state.connected && !state.supported) {
        CHECK(state.buttons == 0);
        CHECK(state.left_trigger == 0.0f && state.right_trigger == 0.0f);
    }
    if (state.supported) {
        CHECK(state.connected);
        CHECK(stick_in_range(state.left_stick));
        CHECK(stick_in_range(state.right_stick));
        CHECK(trigger_in_range(state.left_trigger));
        CHECK(trigger_in_range(state.right_trigger));
    }
}

void test_poll_smoke() {
    // A short-lived outer pool bounds the smoke test's framework temporaries;
    // each poll also owns its internal pool and needs no application lifecycle.
    @autoreleasepool {
        const HostInputSnapshot snapshot = poll_apple_input();
        check_state_consistent(snapshot.controller);
        check_state_consistent(snapshot.keyboard);
    }
    // A second call resamples physical devices and reads the current virtual
    // touch state. Its result may differ if either changed in between, so only
    // the same snapshot-local invariants are checked.
    @autoreleasepool {
        const HostInputSnapshot again = poll_apple_input();
        check_state_consistent(again.controller);
        check_state_consistent(again.keyboard);
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep reports if a defect crashes the process
    test_poll_smoke();
    std::printf("%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
