// ============================================================================
// Armigo Tests - Guest Address Arbitration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/arbitration.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)
constexpr std::uint64_t address = 0x7030001234;
// Every blocking wait in this test is bounded. A lost signal then fails the assertion
// instead of stalling the test process.
constexpr std::int64_t bounded_wait = 5'000'000'000;
}  // namespace

int main() {
    using armigo::horizon::ArbitrateOutcome;
    using armigo::horizon::Arbitration;

    // Guest timeouts round up to whole 19.2 MHz ticks plus two, saturating at the
    // largest host duration.
    using armigo::horizon::kernel_timeout;
    CHECK(kernel_timeout(1) == std::chrono::nanoseconds(156));
    CHECK(kernel_timeout(625) == std::chrono::nanoseconds(729));
    CHECK(kernel_timeout(1'000'000) == std::chrono::nanoseconds(1'000'104));
    CHECK(kernel_timeout(INT64_MAX) == std::chrono::nanoseconds::max());
    using armigo::horizon::switch1_counter;
    using armigo::horizon::switch2_counter;
    CHECK(kernel_timeout(1, switch1_counter) == std::chrono::nanoseconds(156));
    CHECK(kernel_timeout(625, switch1_counter) == std::chrono::nanoseconds(729));
    CHECK(kernel_timeout(1'000'000, switch1_counter) == std::chrono::nanoseconds(1'000'104));
    CHECK(kernel_timeout(INT64_MAX, switch1_counter) == std::chrono::nanoseconds::max());
    CHECK(kernel_timeout(1, switch2_counter) == std::chrono::nanoseconds(96));
    CHECK(kernel_timeout(32, switch2_counter) == std::chrono::nanoseconds(96));
    CHECK(kernel_timeout(33, switch2_counter) == std::chrono::nanoseconds(128));
    CHECK(kernel_timeout(1'000'000, switch2_counter) == std::chrono::nanoseconds(1'000'064));
    CHECK(kernel_timeout(INT64_MAX, switch2_counter) == std::chrono::nanoseconds::max());
    // A second conversion is a different duration, so the SVC layer must round once.
    CHECK(kernel_timeout(kernel_timeout(1, switch1_counter).count(), switch1_counter) ==
          std::chrono::nanoseconds(260));
    CHECK(kernel_timeout(kernel_timeout(1, switch2_counter).count(), switch2_counter) ==
          std::chrono::nanoseconds(160));

    Arbitration arbitration;
    // A wait whose observed word differs from the expected value returns immediately.
    CHECK(arbitration.wait(address, 1, 0, bounded_wait) == ArbitrateOutcome::value_mismatch);
    CHECK(arbitration.waiters(address) == 0);
    // A zero timeout does not block and reports the timeout.
    CHECK(arbitration.wait(address, 1, 1, 0) == ArbitrateOutcome::timed_out);
    CHECK(arbitration.waiters(address) == 0);
    // A short timeout expires on its own.
    CHECK(arbitration.wait(address, 1, 1, 1'000'000) == ArbitrateOutcome::timed_out);
    // Signalling an address with no waiters wakes nobody.
    CHECK(arbitration.signal(address, 1) == 0);

    // A waiter blocks until another thread signals the address.
    std::atomic<bool> entered{false};
    std::thread waiter([&] {
        entered.store(true);
        CHECK(arbitration.wait(address, 1, 1, bounded_wait) == ArbitrateOutcome::succeeded);
    });
    while (!entered.load()) std::this_thread::yield();
    for (unsigned spin = 0; spin != 5000 && arbitration.waiters(address) == 0; ++spin)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(arbitration.waiters(address) == 1);
    CHECK(arbitration.signal(address, 1) == 1);
    waiter.join();
    CHECK(arbitration.waiters(address) == 0);

    // wait_checked compares the word under the queue lock: a release that happened after the
    // caller's first read is a mismatch instead of a lost wakeup, and an unreadable word is
    // refused rather than guessed.
    {
        Arbitration checked;
        std::int32_t word = 0;
        const auto read = [](void* context, std::uint64_t, std::int32_t& value) noexcept {
            value = *static_cast<std::int32_t*>(context);
            return true;
        };
        const auto unreadable = [](void*, std::uint64_t, std::int32_t&) noexcept {
            return false;
        };
        CHECK(checked.wait_checked(address, 1, bounded_wait, read, &word) ==
              ArbitrateOutcome::value_mismatch);
        CHECK(checked.wait_checked(address, 1, bounded_wait, unreadable, &word) ==
              ArbitrateOutcome::unreadable);
        word = 1;
        CHECK(checked.wait_checked(address, 1, 1'000'000, read, &word) ==
              ArbitrateOutcome::timed_out);
        std::atomic<bool> entered{false};
        std::thread waiter([&] {
            entered.store(true);
            CHECK(checked.wait_checked(address, 1, bounded_wait, read, &word) ==
                  ArbitrateOutcome::succeeded);
        });
        while (!entered.load()) std::this_thread::yield();
        for (unsigned spin = 0; spin != 5000 && checked.waiters(address) == 0; ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(checked.waiters(address) == 1);
        word = 0;
        CHECK(checked.signal(address, 1) == 1);
        waiter.join();
        // A word that is already free is a mismatch even with an indefinite timeout.
        CHECK(checked.wait_checked(address, 1, bounded_wait, read, &word) ==
              ArbitrateOutcome::value_mismatch);
    }

    // A signal that arrives while two threads wait wakes both.
    std::atomic<int> ready{0};
    std::atomic<int> released{0};
    const auto body = [&] {
        ready.fetch_add(1);
        CHECK(arbitration.wait(address, 2, 2, bounded_wait) == ArbitrateOutcome::succeeded);
        released.fetch_add(1);
    };
    std::thread first(body), second(body);
    while (ready.load() != 2) std::this_thread::yield();
    for (unsigned spin = 0; spin != 5000 && arbitration.waiters(address) != 2; ++spin)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(arbitration.waiters(address) == 2);
    CHECK(arbitration.signal(address, 2) == 2);
    first.join();
    second.join();
    CHECK(released.load() == 2);

    if (failures == 0) std::fprintf(stderr, "Arbitration failures: 0\n");
    return failures == 0 ? 0 : 1;
}
