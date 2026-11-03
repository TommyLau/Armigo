// ============================================================================
// Armigo Horizon - Process-Wide Condition Variable
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/arbitration.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stop_token>

namespace armigo::horizon {

// The kernel object behind svcWaitProcessWideKeyAtomic (0x1c) and
// svcSignalProcessWideKey (0x1d): a condition variable every thread of one
// process shares, keyed by a guest address.
//
// Horizon's wait takes a mutex word and the tag the address arbiter knows for
// the calling thread. The kernel releases the mutex while the thread waits and,
// when the condition is signalled, hands the mutex to the woken thread by
// writing that thread's tag into the word. A signalled thread whose mutex is
// still owned does not return early: the handoff is deferred until the current
// owner releases it. A timed-out wait never takes the mutex back.
//
// The object holds only the process-wide state. The caller supplies the guest-word
// accessors on every call, because each guest thread carries its own checked accessor
// context while all of them share this one condition-variable table. All reads and
// writes happen while the object holds its lock. The condition key is rounded down to
// a word boundary and its word is armed before the mutex is released. If the condition
// word is inaccessible, its failed write is ignored and the queue still exists.
class ProcessWideKey final {
public:
    // Guest word access. Both return false for an address the process does not map. Reads
    // use acquire ordering and writes use release ordering with respect to native guest
    // accesses of the same aligned word.
    using GuestWordReader = bool (*)(void* context, std::uint64_t address,
                                     std::uint32_t& value) noexcept;
    using GuestWordWriter = bool (*)(void* context, std::uint64_t address,
                                     std::uint32_t value) noexcept;

    // One strong atomic compare-and-swap of a guest word; spurious mismatches are forbidden.
    // On mismatch the callback stores the
    // observed word in expected and reports mismatch; on exchange the word holds desired
    // and expected keeps the prior matching value. invalid_address means the word is not
    // accessible. Like the reader and writer, the callback must be atomic with respect to
    // every other guest-word access of the same address. A successful exchange uses at
    // least acquire-release ordering, and a failed exchange uses at least acquire ordering.
    enum class ExchangeResult { exchanged, mismatch, invalid_address };
    using GuestWordCompareExchange = ExchangeResult (*)(void* context, std::uint64_t address,
                                                        std::uint32_t& expected,
                                                        std::uint32_t desired) noexcept;
    using OwnerValidator = bool (*)(void* context, std::uint32_t owner) noexcept;

    struct GuestWords {
        GuestWordReader read = nullptr;
        GuestWordWriter write = nullptr;
        void* context = nullptr;
        // The condition-variable handoff swaps the waiter's tag into the mutex word with
        // this operation. Without it a signal cannot touch the queues, because a separate
        // read and write could overwrite a mutex owner the guest claimed by its own
        // compare-and-swap in between.
        GuestWordCompareExchange compare_exchange = nullptr;
        // Called under the queue lock after a lock's mutex word matches the supplied
        // owner, or after a condition signal marks an owned mutex for a later handoff.
        // A changed lock word and an immediately granted condition waiter skip validation.
        // The callback must not reenter this ProcessWideKey instance.
        OwnerValidator validate_owner = nullptr;
    };

    enum class WaitOutcome {
        signaled,
        timed_out,
        cancelled,
        // The mutex word could not be accessed. Condition-word faults are ignored.
        invalid_address,
        // A signalled waiter encountered an owned mutex without a valid owner thread.
        invalid_state,
    };

    enum class SignalOutcome {
        signaled,
        // The supplied guest words cannot perform the atomic handoff. Nothing was taken
        // from a queue and no guest word changed.
        unsupported_atomic_access,
    };

    enum class LockOutcome {
        succeeded,
        cancelled,
        // The mutex word could not be read or the selected handoff could not be written.
        invalid_address,
        invalid_owner,
    };

    // A zero timeout expires before blocking, a negative one waits without a deadline,
    // and a positive one is the guest duration in nanoseconds. The optional arbitration
    // argument is retained for source compatibility; mutex and address waits have separate
    // queues.
    explicit ProcessWideKey(std::shared_ptr<Arbitration> arbitration = {});

    // svcArbitrateLock(owner, mutex, tag): if the mutex still names owner with the
    // waiters-present marker, queues tag for an exact ownership handoff and blocks until
    // selected. A changed mutex means the guest lock loop can retry immediately.
    LockOutcome lock(GuestWords words, std::uint64_t mutex_address, std::uint32_t owner,
                     std::uint32_t tag, std::int32_t priority = 32);
    // A stop that removes the queued waiter returns cancelled without taking ownership. If
    // handoff wins first, this returns succeeded and the caller owns the guest mutex.
    LockOutcome lock(GuestWords words, std::uint64_t mutex_address, std::uint32_t owner,
                     std::uint32_t tag, std::stop_token stop,
                     std::int32_t priority = 32);

    // svcWaitProcessWideKeyAtomic(mutex, key, tag, timeout). On return the mutex has been
    // released, and on signaled it has also been handed to this waiter.
    WaitOutcome wait(GuestWords words, std::uint64_t mutex_address, std::uint64_t key,
                     std::uint32_t tag, std::int64_t timeout_nanoseconds,
                     std::int32_t priority = 32);
    // Host cancellation releases an initially owned mutex and removes the waiter from
    // its current condition or mutex queue. A completed handoff wins the race and returns
    // signaled with mutex ownership intact.
    WaitOutcome wait(GuestWords words, std::uint64_t mutex_address, std::uint64_t key,
                     std::uint32_t tag, std::int64_t timeout_nanoseconds,
                     std::stop_token stop, std::int32_t priority = 32);

    // svcSignalProcessWideKey(key, count): hands up to count waiters (all when count is
    // nonpositive) their mutex. Waiters whose mutex is still owned stay
    // pending until its owner unlocks. An invalid owner finishes that waiter with
    // invalid_state while the signal succeeds. The condition word is cleared when no waiter
    // of the key is left. The handoff claims each mutex word with the compare-exchange callback,
    // so a signal without one reports unsupported_atomic_access and changes nothing.
    SignalOutcome signal(GuestWords words, std::uint64_t key, std::int64_t count);

    // svcArbitrateUnlock(mutex): hands the mutex to the next ordinary or signalled waiter,
    // or clears the word. Returns false for an unwritable word.
    bool unlock(GuestWords words, std::uint64_t mutex_address);

    // Threads still blocked on the key; signalled waiters whose handoff is pending are not
    // counted because they have left the condition variable.
    std::size_t waiters(std::uint64_t key) const;
    // Ordinary and signalled condition waiters awaiting ownership of this mutex.
    std::size_t mutex_waiters(std::uint64_t mutex_address) const;

private:
    struct Waiter final {
        Waiter(std::uint64_t address, std::uint32_t value,
               std::int32_t waiter_priority) noexcept
            : mutex_address(address), mutex_tag(value), priority(waiter_priority) {}
        std::uint64_t mutex_address;
        std::uint32_t mutex_tag;
        std::int32_t priority;
        bool in_key_queue = false;
        bool in_mutex_queue = false;
        bool granted = false;
        // The outcome the waiting thread observes once granted is set. A failed handoff
        // access finishes the waiter with invalid_address instead of parking it forever.
        WaitOutcome result = WaitOutcome::signaled;
        std::condition_variable changed;
    };

    static bool write_word_locked(GuestWords words, std::uint64_t address,
                                  std::uint32_t value);
    static bool read_word_locked(GuestWords words, std::uint64_t address, std::uint32_t& value);
    // Lower numeric priorities are selected first; equal priorities retain arrival order.
    void enqueue_key_waiter_locked(std::uint64_t key, const std::shared_ptr<Waiter>& waiter);
    void enqueue_mutex_waiter_locked(const std::shared_ptr<Waiter>& waiter);
    // Releases the mutex while the lock is held: unified handoff first, then a free word.
    bool release_locked(GuestWords words, std::uint64_t mutex_address);
    // Marks a signalled waiter as granted and wakes it.
    void grant_locked(const std::shared_ptr<Waiter>& waiter);
    void cancel_locked(std::uint64_t key, const std::shared_ptr<Waiter>& waiter);
    LockOutcome wait_for_lock_locked(std::unique_lock<std::mutex>& lock,
                                     const std::shared_ptr<Waiter>& waiter,
                                     std::stop_token stop);

    mutable std::mutex mutex_;
    std::map<std::uint64_t, std::list<std::shared_ptr<Waiter>>> queues_;
    std::map<std::uint64_t, std::list<std::shared_ptr<Waiter>>> mutex_waiters_;
};

}  // namespace armigo::horizon
