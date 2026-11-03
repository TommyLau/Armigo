// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/handle_table.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <vector>

namespace armigo::horizon {

struct EventResult;
class ThreadRecord;

// A readable manual-reset event object. The owning host service retains this
// object and signals it; guest handles name the readable endpoint. This does
// not implement writable event handles or CreateEvent/SignalEvent SVCs.
// A wait on several objects at once. Each object lists the waiters registered
// with it and wakes them, under its own lock, when it signals or exits.
struct MultiWaiter {
    std::mutex mutex;
    std::condition_variable changed;
    bool woken = false;
    // The handle index of the object that woke the wait first, which the
    // wait reports, as the kernel reports the object it synchronized on.
    std::uint32_t woken_by = 0;
    void wake(std::uint32_t index) noexcept {
        {
            const std::lock_guard lock(mutex);
            if (!woken) woken_by = index;
            woken = true;
        }
        changed.notify_all();
    }
};
// An object's registration of a multiple-object waiter and the handle index
// the waiter knows it by.
struct MultiWaiterEntry {
    MultiWaiter* waiter;
    std::uint32_t index;
};

class Event final : public KernelObject {
public:
    explicit Event(bool signaled = false) : signaled_(signaled) {}
    void signal() noexcept;
    bool is_signaled() const noexcept;
    // Atomically clears the state and reports whether it was signaled.
    bool reset() noexcept;

private:
    friend class SynchronizationWait;
    friend EventResult wait_event(HandleTable&, std::span<const Handle>, std::int64_t);
    friend EventResult wait_event(HandleTable&, std::span<const Handle>, std::int64_t,
                                  std::stop_token);
    enum class WaitOutcome { signaled, timed_out, cancelled };
    bool wait(std::int64_t timeout_nanoseconds);
    WaitOutcome wait(std::int64_t timeout_nanoseconds, std::stop_token stop);
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    bool signaled_;
    // A reset after signaling must not retract completion from existing waiters.
    std::uint64_t signal_generation_ = 0;
    std::vector<MultiWaiterEntry> waiters_;
};

enum class EventDisposition { handled, unsupported, cancelled };

struct EventResult {
    EventDisposition disposition = EventDisposition::unsupported;
    std::uint32_t result = 0;
    // Success writes the selected index. Caller-aware queued waits write -1 on
    // guest timeout/cancellation; validation and immediate failures preserve X1.
    // Legacy waits leave the index empty on every failure.
    std::optional<std::uint32_t> index;
    const char* detail = nullptr;
};

// Supports exactly one readable Event or ThreadRecord without consuming an event signal.
// Zero polls; negative waits indefinitely; positive waits use a monotonic timeout. The
// stop-token overload reports host cancellation separately from every guest result.
// Other object types are unsupported.
// Lookup pins the object through completion, independently of concurrent close.
EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds);
EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds, std::stop_token stop);
// Caller-aware WaitSynchronization supports zero or one readable event/thread.
// Counts above 64 return 0xee01; every handle/type is checked before waiting.
// Valid multi-object waits remain explicitly unsupported. A ready object or a
// zero-timeout poll wins over pending guest cancellation without consuming it.
// Guest cancellation returns handled/0xec01; host stop remains cancelled.
// The caller must stay alive and have only one concurrent synchronization wait.
EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds, ThreadRecord& caller,
                       std::stop_token stop = {});
// Current-thread pseudo handles resolve to caller; all other handles require a
// ThreadRecord. Successful requests return zero even when cancellation is deferred.
EventResult cancel_synchronization(HandleTable& table, Handle handle,
                                   ThreadRecord& caller);
// Clears a signaled readable event; an already-clear event returns 0xfa01.
// Missing handles return 0xe401. No wait index is written.
EventResult reset_event(HandleTable& table, Handle handle);

}  // namespace armigo::horizon
