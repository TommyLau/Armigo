// ============================================================================
// Armigo Horizon - Process-Wide Condition Variable
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/process_wide_key.h>

#include <algorithm>
#include <chrono>

namespace armigo::horizon {
namespace {
// The mutex word's waiters-present marker. The SDK sets it when it enqueues a kernel lock
// waiter and its user-mode unlock fast path only succeeds when the word equals its own
// handle, so a queued handoff must leave the marker in the word.
constexpr std::uint32_t waiters_present = 0x40000000u;
}  // namespace

ProcessWideKey::ProcessWideKey(std::shared_ptr<Arbitration>) {}

bool ProcessWideKey::read_word_locked(GuestWords words, std::uint64_t address,
                                      std::uint32_t& value) {
    if (words.read == nullptr) return false;
    return words.read(words.context, address, value);
}

bool ProcessWideKey::write_word_locked(GuestWords words, std::uint64_t address,
                                       std::uint32_t value) {
    if (words.write == nullptr) return false;
    return words.write(words.context, address, value);
}

void ProcessWideKey::grant_locked(const std::shared_ptr<Waiter>& waiter) {
    waiter->granted = true;
    waiter->changed.notify_all();
}

void ProcessWideKey::cancel_locked(std::uint64_t key,
                                   const std::shared_ptr<Waiter>& waiter) {
    if (waiter->granted) return;
    if (waiter->in_key_queue) {
        const auto found = queues_.find(key);
        if (found != queues_.end()) {
            found->second.remove(waiter);
            if (found->second.empty()) queues_.erase(found);
        }
        waiter->in_key_queue = false;
    }
    if (waiter->in_mutex_queue) {
        const auto found = mutex_waiters_.find(waiter->mutex_address);
        if (found != mutex_waiters_.end()) {
            found->second.remove(waiter);
            if (found->second.empty()) mutex_waiters_.erase(found);
        }
        waiter->in_mutex_queue = false;
    }
    waiter->result = WaitOutcome::cancelled;
    grant_locked(waiter);
}

ProcessWideKey::LockOutcome ProcessWideKey::wait_for_lock_locked(
    std::unique_lock<std::mutex>& lock, const std::shared_ptr<Waiter>& waiter,
    std::stop_token stop) {
    lock.unlock();
    std::stop_callback cancelled(stop, [&] {
        const std::lock_guard callback_lock(mutex_);
        cancel_locked(0, waiter);
    });
    lock.lock();
    if (!waiter->granted) waiter->changed.wait(lock, [&] { return waiter->granted; });
    const auto result = waiter->result == WaitOutcome::invalid_address
                          ? LockOutcome::invalid_address
                          : waiter->result == WaitOutcome::cancelled
                              ? LockOutcome::cancelled
                              : LockOutcome::succeeded;
    lock.unlock();
    return result;
}

void ProcessWideKey::enqueue_key_waiter_locked(std::uint64_t key,
                                               const std::shared_ptr<Waiter>& waiter) {
    auto& queue = queues_[key];
    const auto position = std::find_if(queue.begin(), queue.end(), [&](const auto& queued) {
        return waiter->priority < queued->priority;
    });
    queue.insert(position, waiter);
    waiter->in_key_queue = true;
}

void ProcessWideKey::enqueue_mutex_waiter_locked(const std::shared_ptr<Waiter>& waiter) {
    auto& queue = mutex_waiters_[waiter->mutex_address];
    const auto position = std::find_if(queue.begin(), queue.end(), [&](const auto& queued) {
        return waiter->priority < queued->priority;
    });
    queue.insert(position, waiter);
    waiter->in_mutex_queue = true;
}

bool ProcessWideKey::release_locked(GuestWords words, std::uint64_t mutex_address) {
    const auto found = mutex_waiters_.find(mutex_address);
    if (found != mutex_waiters_.end() && !found->second.empty()) {
        const auto next = found->second.front();
        found->second.pop_front();
        next->in_mutex_queue = false;
        const auto more = !found->second.empty();
        if (!more) mutex_waiters_.erase(found);
        // The marker is based on the unified queue. It forces the selected owner's unlock
        // through the kernel whenever any ordinary or signalled condition waiter remains.
        const auto word = more ? (next->mutex_tag | waiters_present) : next->mutex_tag;
        if (!write_word_locked(words, mutex_address, word)) {
            next->result = WaitOutcome::invalid_address;
            grant_locked(next);
            return false;
        }
        grant_locked(next);
        return true;
    }
    if (!write_word_locked(words, mutex_address, 0)) return false;
    return true;
}

ProcessWideKey::LockOutcome ProcessWideKey::lock(GuestWords words,
                                                 std::uint64_t mutex_address,
                                                 std::uint32_t owner,
                                                 std::uint32_t tag,
                                                 std::int32_t priority) {
    return lock(words, mutex_address, owner, tag, {}, priority);
}

ProcessWideKey::LockOutcome ProcessWideKey::lock(GuestWords words,
                                                 std::uint64_t mutex_address,
                                                 std::uint32_t owner,
                                                 std::uint32_t tag, std::stop_token stop,
                                                 std::int32_t priority) {
    if (stop.stop_requested()) return LockOutcome::cancelled;
    std::unique_lock lock(mutex_);
    std::uint32_t observed = 0;
    if (!read_word_locked(words, mutex_address, observed)) return LockOutcome::invalid_address;
    if (observed != (owner | waiters_present)) return LockOutcome::succeeded;
    if (words.validate_owner != nullptr && !words.validate_owner(words.context, owner))
        return LockOutcome::invalid_owner;

    auto waiter = std::make_shared<Waiter>(mutex_address, tag, priority);
    enqueue_mutex_waiter_locked(waiter);
    return wait_for_lock_locked(lock, waiter, stop);
}

ProcessWideKey::WaitOutcome ProcessWideKey::wait(GuestWords words, std::uint64_t mutex_address,
                                                 std::uint64_t key, std::uint32_t tag,
                                                 std::int64_t timeout_nanoseconds,
                                                 std::int32_t priority) {
    return wait(words, mutex_address, key, tag, timeout_nanoseconds, {}, priority);
}

ProcessWideKey::WaitOutcome ProcessWideKey::wait(GuestWords words,
                                                 std::uint64_t mutex_address,
                                                 std::uint64_t key, std::uint32_t tag,
                                                 std::int64_t timeout_nanoseconds,
                                                 std::stop_token stop,
                                                 std::int32_t priority) {
    key &= ~std::uint64_t{3};
    auto deadline = std::chrono::steady_clock::time_point::max();
    if (timeout_nanoseconds > 0) {
        const auto now = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(timeout_nanoseconds));
        deadline = duration > std::chrono::steady_clock::time_point::max() - now
                       ? std::chrono::steady_clock::time_point::max()
                       : now + duration;
    }
    std::unique_lock lock(mutex_);
    // Arm the condition word before releasing the mutex. The guest's signaller reads this
    // word in user mode and skips the signal call when it reads zero. The kernel ignores
    // an inaccessible condition word, but a failed mutex release fails the wait.
    write_word_locked(words, key, 1);
    if (!release_locked(words, mutex_address)) return WaitOutcome::invalid_address;

    auto waiter = std::make_shared<Waiter>(mutex_address, tag, priority);
    enqueue_key_waiter_locked(key, waiter);
    if (stop.stop_requested()) {
        cancel_locked(key, waiter);
        lock.unlock();
        return WaitOutcome::cancelled;
    }
    if (timeout_nanoseconds == 0) {
        // Enrollment and removal stay under one lock; a signal cannot grant this poll.
        const auto found = queues_.find(key);
        if (found != queues_.end()) {
            found->second.remove(waiter);
            if (found->second.empty()) queues_.erase(found);
        }
        waiter->in_key_queue = false;
        waiter->result = WaitOutcome::timed_out;
        waiter->granted = true;
        return WaitOutcome::timed_out;
    }
    lock.unlock();
    std::stop_callback cancelled(stop, [&] {
        const std::lock_guard callback_lock(mutex_);
        cancel_locked(key, waiter);
    });
    lock.lock();
    if (timeout_nanoseconds < 0) {
        waiter->changed.wait(lock, [&] { return waiter->granted; });
        const auto result = waiter->result;
        lock.unlock();
        return result;
    }
    while (!waiter->granted) {
        if (waiter->changed.wait_until(lock, deadline) != std::cv_status::timeout) continue;
        if (waiter->granted) break;
        if (waiter->in_key_queue) {
            const auto key_queue = queues_.find(key);
            if (key_queue != queues_.end()) {
                key_queue->second.remove(waiter);
                if (key_queue->second.empty()) queues_.erase(key_queue);
            }
            waiter->in_key_queue = false;
        }
        if (waiter->in_mutex_queue) {
            const auto mutex_queue = mutex_waiters_.find(mutex_address);
            if (mutex_queue != mutex_waiters_.end()) {
                mutex_queue->second.remove(waiter);
                if (mutex_queue->second.empty()) mutex_waiters_.erase(mutex_queue);
            }
            waiter->in_mutex_queue = false;
        }
        waiter->result = WaitOutcome::timed_out;
        waiter->granted = true;
        lock.unlock();
        return WaitOutcome::timed_out;
    }
    const auto result = waiter->result;
    lock.unlock();
    return result;
}

ProcessWideKey::SignalOutcome ProcessWideKey::signal(GuestWords words, std::uint64_t key,
                                                     std::int64_t count) {
    key &= ~std::uint64_t{3};
    // A separate read and write could overwrite a mutex owner the guest claimed by its own
    // compare-and-swap between the two. Without the atomic handoff the signal must not
    // touch a queue or a guest word at all.
    if (words.compare_exchange == nullptr) return SignalOutcome::unsupported_atomic_access;
    std::lock_guard lock(mutex_);
    bool remaining = false;
    const auto found = queues_.find(key);
    if (found != queues_.end()) {
        auto& queue = found->second;
        std::int64_t taken = 0;
        while (!queue.empty() && (count <= 0 || taken < count)) {
            const auto waiter = queue.front();
            queue.pop_front();
            waiter->in_key_queue = false;
            ++taken;
            // The firmware's atomic helper claims the word with one compare-exchange that
            // writes the waiter's tag over a zero word, or only the waiters-present marker
            // over an owned one, retrying on contention. Matching that exact transition
            // makes this the linearization point against a native guest compare-exchange
            // on the same word: an owner claimed in between is observed, never overwritten.
            std::uint32_t expected = 0;
            auto result = ExchangeResult::mismatch;
            while (result == ExchangeResult::mismatch) {
                const auto desired = expected == 0 ? waiter->mutex_tag
                                                   : expected | waiters_present;
                result = words.compare_exchange(words.context, waiter->mutex_address,
                                                expected, desired);
            }
            if (result == ExchangeResult::exchanged) {
                // expected still holds the prior word: zero means this waiter owns the
                // mutex now. An owned word defers the handoff only if its owner is valid.
                if (expected == 0)
                    grant_locked(waiter);
                else if (words.validate_owner != nullptr &&
                         !words.validate_owner(words.context, expected & ~waiters_present)) {
                    // The marker written by the exchange remains, but this waiter must
                    // finish instead of waiting for an owner that cannot release it.
                    waiter->result = WaitOutcome::invalid_state;
                    grant_locked(waiter);
                } else {
                    enqueue_mutex_waiter_locked(waiter);
                }
            } else {
                // An inaccessible mutex word fails only this waiter's wait; the signal
                // itself stays completed, matching the firmware's per-target fault. The
                // waiter is finished instead of parked in the mutex queue.
                waiter->result = WaitOutcome::invalid_address;
                grant_locked(waiter);
            }
        }
        if (queue.empty())
            queues_.erase(found);
        else
            remaining = true;
    }
    // Horizon clears the condition word when the last waiter of the key leaves it.
    if (!remaining) write_word_locked(words, key, 0);
    return SignalOutcome::signaled;
}

bool ProcessWideKey::unlock(GuestWords words, std::uint64_t mutex_address) {
    std::lock_guard lock(mutex_);
    return release_locked(words, mutex_address);
}

std::size_t ProcessWideKey::waiters(std::uint64_t key) const {
    key &= ~std::uint64_t{3};
    std::lock_guard lock(mutex_);
    const auto found = queues_.find(key);
    return found == queues_.end() ? 0 : found->second.size();
}

std::size_t ProcessWideKey::mutex_waiters(std::uint64_t mutex_address) const {
    std::lock_guard lock(mutex_);
    const auto found = mutex_waiters_.find(mutex_address);
    return found == mutex_waiters_.end() ? 0 : found->second.size();
}

}  // namespace armigo::horizon
