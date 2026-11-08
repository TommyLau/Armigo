// ============================================================================
// Armigo Horizon - Guest Address Arbitration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/arbitration.h>

#include <algorithm>
#include <limits>

namespace armigo::horizon {

std::chrono::nanoseconds kernel_timeout(std::int64_t nanoseconds, CounterRate rate) noexcept {
    const auto count = static_cast<std::uint64_t>(nanoseconds);
    const auto whole = count / rate.nanoseconds * rate.ticks;
    const auto remainder = count % rate.nanoseconds;
    const auto frac = (remainder * rate.ticks + rate.nanoseconds - 1) / rate.nanoseconds;
    const auto ticks = whole + frac + 2;
    const auto groups = ticks / rate.ticks;
    const auto part = (ticks % rate.ticks) * rate.nanoseconds / rate.ticks;
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (groups > (limit - part) / rate.nanoseconds) return std::chrono::nanoseconds::max();
    return std::chrono::nanoseconds(groups * rate.nanoseconds + part);
}

ArbitrateOutcome Arbitration::wait_transaction(std::uint64_t address,
                                                std::int64_t timeout_nanoseconds,
                                                WaitCheck check, void* context,
                                                std::uint32_t priority) {
    return wait_transaction(address, timeout_nanoseconds, check, context, {}, priority);
}

ArbitrateOutcome Arbitration::wait_transaction(std::uint64_t address,
                                                std::int64_t timeout_nanoseconds,
                                                WaitCheck check, void* context,
                                                std::stop_token stop,
                                                std::uint32_t priority) {
    // The timeout covers the whole operation, including contention for the kernel lock.
    const auto now = std::chrono::steady_clock::now();
    const auto available = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::time_point::max() - now).count();
    const auto timeout = timeout_nanoseconds > 0 ? timeout_nanoseconds : 0;
    const auto deadline = timeout > 0 && timeout < available
        ? now + std::chrono::nanoseconds(timeout)
        : std::chrono::steady_clock::time_point::max();
    std::unique_lock lock(mutex_);
    if (stop.stop_requested()) return ArbitrateOutcome::cancelled;
    if (check == nullptr) return ArbitrateOutcome::unreadable;
    const auto outcome = check(context);
    if (outcome != ArbitrateOutcome::succeeded) return outcome;
    if (timeout_nanoseconds == 0) return ArbitrateOutcome::timed_out;

    Waiter waiter{{}, priority, false, false, false};
    auto& queue = queues_[address];
    // Lower numerical priorities run first; equal priorities retain arrival order.
    queue.insert(std::find_if(queue.begin(), queue.end(), [&](const auto* other) {
        return other->priority > priority;
    }), &waiter);
    waiter.queued = true;
    lock.unlock();
    std::stop_callback cancelled(stop, [&] {
        const std::lock_guard callback_lock(mutex_);
        if (!waiter.queued) return;
        const auto found = queues_.find(address);
        if (found != queues_.end()) {
            found->second.remove(&waiter);
            if (found->second.empty()) queues_.erase(found);
        }
        waiter.queued = false;
        waiter.cancelled = true;
        waiter.changed.notify_one();
    });
    lock.lock();
    if (timeout_nanoseconds < 0) {
        waiter.changed.wait(lock, [&] { return waiter.signaled || waiter.cancelled; });
    } else if (!waiter.changed.wait_until(lock, deadline, [&] {
                   return waiter.signaled || waiter.cancelled;
               })) {
        // A signal removes its selected nodes before notifying. Timeout removal uses the
        // same lock, so a waiter can consume at most one signal and never a later credit.
        const auto found = queues_.find(address);
        if (found != queues_.end()) {
            found->second.remove(&waiter);
            if (found->second.empty()) queues_.erase(found);
        }
        waiter.queued = false;
        lock.unlock();
        return ArbitrateOutcome::timed_out;
    }
    const auto result = waiter.cancelled ? ArbitrateOutcome::cancelled
                                         : ArbitrateOutcome::succeeded;
    lock.unlock();
    return result;
}

ArbitrateOutcome Arbitration::wait(std::uint64_t address, std::int32_t value,
                                   std::int32_t observed, std::int64_t timeout_nanoseconds) {
    return wait(address, value, observed, timeout_nanoseconds, {});
}

ArbitrateOutcome Arbitration::wait(std::uint64_t address, std::int32_t value,
                                   std::int32_t observed, std::int64_t timeout_nanoseconds,
                                   std::stop_token stop) {
    struct Check { std::int32_t value; std::int32_t observed; } context{value, observed};
    return wait_transaction(address, timeout_nanoseconds, [](void* raw) noexcept {
        const auto& check = *static_cast<Check*>(raw);
        return check.observed == check.value ? ArbitrateOutcome::succeeded
                                             : ArbitrateOutcome::value_mismatch;
    }, &context, stop);
}

ArbitrateOutcome Arbitration::wait_checked(std::uint64_t address, std::int32_t expected,
                                           std::int64_t timeout_nanoseconds, WordReader read,
                                           void* context) {
    return wait_checked(address, expected, timeout_nanoseconds, read, context, {});
}

ArbitrateOutcome Arbitration::wait_checked(std::uint64_t address, std::int32_t expected,
                                           std::int64_t timeout_nanoseconds, WordReader read,
                                           void* context, std::stop_token stop) {
    struct Check {
        std::uint64_t address;
        std::int32_t expected;
        WordReader read;
        void* context;
    } check{address, expected, read, context};
    return wait_transaction(address, timeout_nanoseconds, [](void* raw) noexcept {
        const auto& check = *static_cast<Check*>(raw);
        std::int32_t observed = 0;
        if (check.read == nullptr || !check.read(check.context, check.address, observed))
            return ArbitrateOutcome::unreadable;
        return observed == check.expected ? ArbitrateOutcome::succeeded
                                           : ArbitrateOutcome::value_mismatch;
    }, &check, stop);
}

std::size_t Arbitration::signal_locked(std::uint64_t address, std::size_t count) {
    const auto found = queues_.find(address);
    if (found == queues_.end()) return 0;
    auto& queue = found->second;
    std::size_t woken = 0;
    while (!queue.empty() && woken < count) {
        auto* waiter = queue.front();
        queue.pop_front();
        waiter->queued = false;
        waiter->signaled = true;
        waiter->changed.notify_one();
        ++woken;
    }
    if (queue.empty()) queues_.erase(found);
    return woken;
}

ArbitrateOutcome Arbitration::signal_transaction(std::uint64_t address, std::size_t count,
                                                  SignalCheck check, void* context) {
    const std::lock_guard lock(mutex_);
    const auto found = queues_.find(address);
    const auto waiting = found == queues_.end() ? 0 : found->second.size();
    if (check != nullptr) {
        const auto outcome = check(context, waiting);
        if (outcome != ArbitrateOutcome::succeeded) return outcome;
    }
    signal_locked(address, count);
    return ArbitrateOutcome::succeeded;
}

std::size_t Arbitration::signal(std::uint64_t address, std::size_t count) {
    const std::lock_guard lock(mutex_);
    return signal_locked(address, count);
}

std::size_t Arbitration::waiters(std::uint64_t address) const {
    const std::lock_guard lock(mutex_);
    const auto found = queues_.find(address);
    return found == queues_.end() ? 0 : found->second.size();
}

}  // namespace armigo::horizon
