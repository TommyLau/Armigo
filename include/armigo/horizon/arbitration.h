// ============================================================================
// Armigo Horizon - Guest Address Arbitration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/common/guest_platform.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <stop_token>

namespace armigo::horizon {

// Guest counter scale. Switch 1 is 19.2 MHz (625 ns per 12 ticks). Switch 2 is
// 31.25 MHz (exactly 32 ns per tick). The SVC layer rounds a positive timeout
// once with the process rate; waits below that layer do not convert again.
struct CounterRate {
    std::uint64_t frequency = 0;
    std::uint64_t nanoseconds = 0;
    std::uint64_t ticks = 0;
};

inline constexpr CounterRate switch1_counter{19'200'000, 625, 12};
inline constexpr CounterRate switch2_counter{31'250'000, 32, 1};

constexpr CounterRate counter_rate(GuestPlatform platform) noexcept {
    return platform == GuestPlatform::switch2 ? switch2_counter : switch1_counter;
}

// A positive guest timeout as the kernel waits it: the nanoseconds rounded up
// to whole ticks of `rate`, plus two ticks, converted back exactly. The result
// saturates when the host duration cannot represent it. The default rate is
// the Switch 1 counter.
std::chrono::nanoseconds kernel_timeout(std::int64_t nanoseconds,
                                        CounterRate rate = switch1_counter) noexcept;

// The arbitration types the guest passes to svcArbitrateAddress. Only the wait and signal
// forms are implemented; anything else is refused instead of reported as success.
enum class ArbitrateType : std::uint32_t {
    wait = 0,
    signal = 1,
    signal_and_increment_if_equal = 2,
    signal_and_modify_by = 3,
};

enum class ArbitrateOutcome {
    succeeded,
    // The word at the guest address did not match the expected value, so the wait returned
    // immediately; that is a normal result, not an error.
    value_mismatch,
    timed_out,
    cancelled,
    unsupported_type,
    // The guest word could not be read.
    unreadable,
};

// Serializes guest word checks/updates, enrollment, selection and timeout removal.
// Each enrolled waiter has its own completion predicate; notifications do not grant peers.
class Arbitration {
public:
    // Reads one signed guest word under the queue lock. Returning false means the address is
    // not readable and the wait is refused rather than guessed.
    using WordReader = bool (*)(void* context, std::uint64_t address,
                                std::int32_t& value) noexcept;

    // Callbacks run under the arbitration lock and must not reenter this object. Guest
    // memory updates must use host atomics on the same backing as native guest atomics;
    // the queue lock alone cannot serialize against instructions outside the kernel.
    using WaitCheck = ArbitrateOutcome (*)(void* context) noexcept;
    using SignalCheck = ArbitrateOutcome (*)(void* context, std::size_t waiting) noexcept;
    // The timeout arrives already converted by the SVC layer. A host caller passes
    // raw nanoseconds; this wait does not convert again.
    ArbitrateOutcome wait_transaction(std::uint64_t address, std::int64_t timeout_nanoseconds,
                                      WaitCheck check, void* context,
                                      std::uint32_t priority = 32);
    // Cancellation removes an enrolled waiter under the queue lock. A signal that selects
    // the waiter first still completes normally; cancellation does not consume that signal.
    ArbitrateOutcome wait_transaction(std::uint64_t address, std::int64_t timeout_nanoseconds,
                                      WaitCheck check, void* context, std::stop_token stop,
                                      std::uint32_t priority = 32);
    ArbitrateOutcome signal_transaction(std::uint64_t address, std::size_t count,
                                        SignalCheck check, void* context);

    // Unconditional queue primitive after comparing two supplied constants. Guest memory
    // conditions must use wait_transaction or wait_checked instead of a stale observation.
    ArbitrateOutcome wait(std::uint64_t address, std::int32_t value, std::int32_t observed,
                          std::int64_t timeout_nanoseconds);
    ArbitrateOutcome wait(std::uint64_t address, std::int32_t value, std::int32_t observed,
                          std::int64_t timeout_nanoseconds, std::stop_token stop);
    // Blocks only while the guest word still reads `expected`, checked under the same lock
    // that a signal uses to select waiters. This is how a contended
    // wait avoids blocking after the owner has already released the word: a release between
    // the caller's first read and this call is observed as a value mismatch.
    ArbitrateOutcome wait_checked(std::uint64_t address, std::int32_t expected,
                                  std::int64_t timeout_nanoseconds, WordReader read,
                                  void* context);
    ArbitrateOutcome wait_checked(std::uint64_t address, std::int32_t expected,
                                  std::int64_t timeout_nanoseconds, WordReader read,
                                  void* context, std::stop_token stop);
    // Wakes up to `count` waiters of the address. Returns how many were woken.
    std::size_t signal(std::uint64_t address, std::size_t count);
    // Number of threads currently waiting on the address, for diagnostics and tests.
    std::size_t waiters(std::uint64_t address) const;

private:
    struct Waiter {
        std::condition_variable changed;
        std::uint32_t priority;
        bool queued = false;
        bool signaled = false;
        bool cancelled = false;
    };
    using Queue = std::list<Waiter*>;
    std::size_t signal_locked(std::uint64_t address, std::size_t count);
    mutable std::mutex mutex_;
    std::map<std::uint64_t, Queue> queues_;
};

}  // namespace armigo::horizon
