// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored coverage for the display refresh link, driven by the
// main screen's display link (no window is created): host times map onto the
// steady clock; refreshes arrive on a thread of their own, on the steady
// clock, with a plausible refresh period, and none arrive after the link is
// destroyed. The refresh part skips without a display session or while the
// display sleeps (no refreshes are produced then).

#include "display_refresh.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

unsigned failures = 0;

#define CHECK(...)                                                          \
    do {                                                                    \
        if (!(__VA_ARGS__)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); \
            ++failures;                                                     \
        }                                                                   \
    } while (false)

struct Refresh {
    std::chrono::steady_clock::time_point vblank;
    std::chrono::nanoseconds period;
    std::chrono::steady_clock::time_point received;
    bool main_thread;
};

} // namespace

int main() {
    @autoreleasepool {
        // Host time (which stops while the Mac sleeps) maps onto the steady
        // clock (which does not): now is now, and offsets are kept.
        const auto before = std::chrono::steady_clock::now();
        const double host = CACurrentMediaTime();
        const auto mapped = armigo::platform::steady_from_host_time(host);
        const auto mapped_later = armigo::platform::steady_from_host_time(host + 0.25);
        const auto after = std::chrono::steady_clock::now();
        CHECK(mapped >= before - std::chrono::milliseconds(1) && mapped <= after + std::chrono::milliseconds(1));
        CHECK(std::llabs((mapped_later - mapped - std::chrono::milliseconds(250)).count()) < 1'000'000);

        NSScreen* screen = [NSScreen mainScreen];
        if (!screen) {
            std::printf("SKIP: no display session\n");
            return failures ? 1 : 77;
        }
        std::mutex mutex;
        std::vector<Refresh> refreshes;
        auto link = armigo::platform::start_display_refresh(
            screen, [&](std::chrono::steady_clock::time_point vblank, std::chrono::nanoseconds period) {
                const std::lock_guard lock(mutex);
                refreshes.push_back({vblank, period, std::chrono::steady_clock::now(), [NSThread isMainThread] == YES});
            });
        CHECK(link != nullptr);
        if (!link) return 1;
        // The main thread never pumps a run loop here: refreshes must arrive
        // on the link's own thread.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                const std::lock_guard lock(mutex);
                if (refreshes.size() >= 10) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        link.reset();
        std::vector<Refresh> seen;
        {
            const std::lock_guard lock(mutex);
            seen = refreshes;
        }
        if (seen.empty()) {
            // A session without an active display (screen asleep) delivers none.
            std::printf("SKIP: the display delivered no refreshes\n");
            return failures ? 1 : 77;
        }
        CHECK(seen.size() >= 10);
        for (std::size_t i = 0; i < seen.size(); ++i) {
            const auto& refresh = seen[i];
            CHECK(!refresh.main_thread);
            // A refresh period between 400 Hz and 20 Hz.
            CHECK(refresh.period >= std::chrono::microseconds(2'500) && refresh.period <= std::chrono::milliseconds(50));
            // The vblank is on the steady clock: recent, and not in the future
            // by more than one refresh.
            const auto lag = refresh.received - refresh.vblank;
            CHECK(lag > -refresh.period - std::chrono::milliseconds(1) && lag < std::chrono::milliseconds(100));
            if (i) CHECK(refresh.vblank > seen[i - 1].vblank);
        }
        // Nothing is delivered once the link is destroyed.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        {
            const std::lock_guard lock(mutex);
            CHECK(refreshes.size() == seen.size());
        }
    }
    if (failures) {
        std::fprintf(stderr, "%u failure(s)\n", failures);
        return 1;
    }
    std::puts("display refresh tests passed");
    return 0;
}
