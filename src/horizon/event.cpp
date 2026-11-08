// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>

#include <armigo/horizon/startup_queries.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

namespace armigo::horizon {
namespace {
EventResult handled(std::uint32_t result, std::optional<std::uint32_t> index = std::nullopt) {
    return {EventDisposition::handled, result, index, nullptr};
}
EventResult unsupported(const char* detail) {
    return {EventDisposition::unsupported, 0, std::nullopt, detail};
}
EventResult cancelled() {
    return {EventDisposition::cancelled, 0, std::nullopt, "Wait cancelled"};
}
}  // namespace


// Registration and cancellation use caller -> object lock order. Signal/exit and
// host stop take only the object lock. This makes readiness, cancellation, and
// timeout a single decision while preserving a signal followed by an immediate reset.
class SynchronizationWait {
    struct Registration {
        ThreadRecord& caller;
        void* context = nullptr;
        ~Registration() {
            if (!context) return;
            const std::lock_guard lock(caller.guest_wait_mutex_);
            caller.guest_wait_cancel_ = nullptr;
            caller.guest_wait_context_ = nullptr;
        }
    };

    template<class Ready>
    static EventResult wait(ThreadRecord& caller, std::mutex& mutex,
                            std::condition_variable& changed, Ready ready,
                            std::int64_t timeout_nanoseconds, std::stop_token stop) {
        struct State {
            std::mutex& mutex;
            std::condition_variable& changed;
            Ready& ready;
            bool guest_cancelled = false;
            bool host_cancelled = false;
            bool completed = false;
        } state{mutex, changed, ready};
        // Destruction removes registration only after releasing both state locks,
        // and before destroying the callback's stack context.
        Registration registration{caller};
        std::unique_lock caller_lock(caller.guest_wait_mutex_);
        std::unique_lock lock(mutex);
        if (ready(true)) return handled(0, 0);
        if (stop.stop_requested()) return cancelled();
        if (timeout_nanoseconds == 0) return handled(0xea01);
        if (caller.guest_cancellation_pending_) {
            caller.guest_cancellation_pending_ = false;
            return handled(0xec01);
        }
        if (caller.guest_wait_cancel_)
            return unsupported("A thread cannot enter concurrent synchronization waits");
        caller.guest_wait_context_ = &state;
        caller.guest_wait_cancel_ = [](void* context) noexcept {
            auto& wait = *static_cast<State*>(context);
            const std::lock_guard callback_lock(wait.mutex);
            if (wait.completed || wait.guest_cancelled || wait.host_cancelled ||
                wait.ready(false)) return false;
            wait.guest_cancelled = true;
            wait.changed.notify_all();
            return true;
        };
        registration.context = &state;
        caller_lock.unlock();
        lock.unlock();
        std::stop_callback host_cancellation(stop, [&] {
            const std::lock_guard callback_lock(mutex);
            if (state.completed || state.guest_cancelled || state.host_cancelled ||
                ready(false)) return;
            state.host_cancelled = true;
            changed.notify_all();
        });
        lock.lock();
        const auto finished = [&] {
            return state.guest_cancelled || state.host_cancelled || ready(false);
        };
        bool woke = true;
        if (timeout_nanoseconds < 0) {
            changed.wait(lock, finished);
        } else {
            using Clock = std::chrono::steady_clock;
            const auto now = Clock::now();
            const auto timeout = std::chrono::nanoseconds(timeout_nanoseconds);
            const auto deadline = timeout < Clock::time_point::max() - now
                ? now + std::chrono::duration_cast<Clock::duration>(timeout)
                : Clock::time_point::max();
            woke = changed.wait_until(lock, deadline, finished);
        }
        state.completed = true;
        const auto result = state.guest_cancelled ? handled(0xec01, 0xffffffffu)
            : state.host_cancelled ? cancelled()
            : woke ? handled(0, 0) : handled(0xea01, 0xffffffffu);
        lock.unlock();
        return result;
    }

public:
    static EventResult event(Event& event, ThreadRecord& caller,
                             std::int64_t timeout_nanoseconds, std::stop_token stop) {
        std::uint64_t generation = 0;
        return wait(caller, event.mutex_, event.changed_, [&](bool initial) {
            if (initial) generation = event.signal_generation_;
            return event.signaled_ || event.signal_generation_ != generation;
        }, timeout_nanoseconds, stop);
    }

    static EventResult thread(ThreadRecord& thread, ThreadRecord& caller,
                              std::int64_t timeout_nanoseconds, std::stop_token stop) {
        return wait(caller, thread.exit_mutex_, thread.exit_changed_,
                    [&](bool) { return thread.exited_; }, timeout_nanoseconds, stop);
    }

    // Waits until any of the objects is signaled (an event) or has exited (a
    // thread) and reports the index of the first ready one in handle order.
    // The waiter registers with every object (caller -> object lock order, as
    // for one object); an object already ready completes the wait at once.
    // Cancellation by the guest or the host wins over a later signal.
    static EventResult any(std::span<const std::shared_ptr<KernelObject>> objects, ThreadRecord& caller,
                           std::int64_t timeout_nanoseconds, std::stop_token stop) {
        struct Entry {
            Event* event = nullptr;
            ThreadRecord* thread = nullptr;
            std::uint64_t generation = 0;
            bool registered = false;
        };
        std::array<Entry, 64> entries{};
        for (std::size_t index = 0; index != objects.size(); ++index) {
            entries[index].event = dynamic_cast<Event*>(objects[index].get());
            entries[index].thread = dynamic_cast<ThreadRecord*>(objects[index].get());
            if (!entries[index].event && !entries[index].thread)
                return unsupported("Waiting for process exit is not implemented");
        }
        MultiWaiter waiter;
        struct State {
            MultiWaiter& waiter;
            bool guest_cancelled = false;
            bool host_cancelled = false;
            bool completed = false;
        } state{waiter};
        // Each check holds the object's own lock.
        const auto ready = [&](Entry& entry) {
            if (entry.event)
                return entry.event->signaled_ || entry.event->signal_generation_ != entry.generation;
            return entry.thread->exited_;
        };
        const auto locked = [](Entry& entry) -> std::mutex& {
            return entry.event ? entry.event->mutex_ : entry.thread->exit_mutex_;
        };
        const auto unregister = [&] {
            for (std::size_t index = 0; index != objects.size(); ++index) {
                auto& entry = entries[index];
                if (!entry.registered) continue;
                const std::lock_guard lock(locked(entry));
                auto& list = entry.event ? entry.event->waiters_ : entry.thread->exit_waiters_;
                // A handle listed twice registered twice: this removes both.
                std::erase_if(list, [&](const MultiWaiterEntry& registered) {
                    return registered.waiter == &waiter;
                });
                entry.registered = false;
            }
        };
        const auto first_ready = [&]() -> std::optional<std::uint32_t> {
            for (std::size_t index = 0; index != objects.size(); ++index) {
                const std::lock_guard lock(locked(entries[index]));
                if (ready(entries[index])) return static_cast<std::uint32_t>(index);
            }
            return std::nullopt;
        };
        Registration registration{caller};
        std::unique_lock caller_lock(caller.guest_wait_mutex_);
        for (std::size_t index = 0; index != objects.size(); ++index) {
            auto& entry = entries[index];
            const std::lock_guard lock(locked(entry));
            if (entry.event) entry.generation = entry.event->signal_generation_;
            if (ready(entry)) {
                caller_lock.unlock();
                unregister();
                return handled(0, static_cast<std::uint32_t>(index));
            }
            try {
                (entry.event ? entry.event->waiters_ : entry.thread->exit_waiters_)
                    .push_back({&waiter, static_cast<std::uint32_t>(index)});
            } catch (const std::bad_alloc&) {
                caller_lock.unlock();
                unregister();
                return unsupported("The multiple-object wait could not register its waiter");
            }
            entry.registered = true;
        }
        const auto leave = [&](EventResult result) {
            caller_lock.unlock();
            unregister();
            return result;
        };
        if (stop.stop_requested()) return leave(cancelled());
        if (timeout_nanoseconds == 0) return leave(handled(0xea01));
        if (caller.guest_cancellation_pending_) {
            caller.guest_cancellation_pending_ = false;
            return leave(handled(0xec01));
        }
        if (caller.guest_wait_cancel_)
            return leave(unsupported("A thread cannot enter concurrent synchronization waits"));
        caller.guest_wait_context_ = &state;
        caller.guest_wait_cancel_ = [](void* context) noexcept {
            auto& wait = *static_cast<State*>(context);
            const std::lock_guard callback_lock(wait.waiter.mutex);
            if (wait.completed || wait.guest_cancelled || wait.host_cancelled || wait.waiter.woken)
                return false;
            wait.guest_cancelled = true;
            wait.waiter.changed.notify_all();
            return true;
        };
        registration.context = &state;
        caller_lock.unlock();
        {
            std::stop_callback host_cancellation(stop, [&] {
                const std::lock_guard callback_lock(waiter.mutex);
                if (state.completed || state.guest_cancelled || state.host_cancelled || waiter.woken) return;
                state.host_cancelled = true;
                waiter.changed.notify_all();
            });
            std::unique_lock lock(waiter.mutex);
            const auto finished = [&] {
                return state.guest_cancelled || state.host_cancelled || waiter.woken;
            };
            if (timeout_nanoseconds < 0) {
                waiter.changed.wait(lock, finished);
            } else {
                using Clock = std::chrono::steady_clock;
                const auto now = Clock::now();
                const auto timeout = std::chrono::nanoseconds(timeout_nanoseconds);
                const auto deadline = timeout < Clock::time_point::max() - now
                    ? now + std::chrono::duration_cast<Clock::duration>(timeout)
                    : Clock::time_point::max();
                waiter.changed.wait_until(lock, deadline, finished);
            }
            state.completed = true;
        }
        unregister();
        if (state.guest_cancelled) return handled(0xec01, 0xffffffffu);
        if (state.host_cancelled) return cancelled();
        if (waiter.woken) return handled(0, waiter.woken_by);
        if (const auto index = first_ready()) return handled(0, *index);
        return handled(0xea01, 0xffffffffu);
    }

    static EventResult empty(ThreadRecord& caller, std::int64_t timeout_nanoseconds,
                             std::stop_token stop) {
        std::mutex mutex;
        std::condition_variable changed;
        return wait(caller, mutex, changed, [](bool) { return false; },
                    timeout_nanoseconds, stop);
    }
};

void Event::signal() noexcept {
    {
        const std::lock_guard lock(mutex_);
        if (signaled_) return;
        signaled_ = true;
        ++signal_generation_;
        for (const auto& entry : waiters_) entry.waiter->wake(entry.index);
    }
    changed_.notify_all();
}

bool Event::is_signaled() const noexcept {
    const std::lock_guard lock(mutex_);
    return signaled_;
}

bool Event::reset() noexcept {
    const std::lock_guard lock(mutex_);
    const bool was_signaled = signaled_;
    signaled_ = false;
    return was_signaled;
}

WritableEvent::WritableEvent(std::shared_ptr<Event> event, std::function<void()> observer)
    : event_(std::move(event)), observer_(std::move(observer)) {
    if (!event_) throw std::invalid_argument("A writable event needs its readable end.");
}

void WritableEvent::signal() {
    event_->signal();
    if (observer_) observer_();
}

void WritableEvent::clear() noexcept { (void)event_->reset(); }

bool Event::wait(std::int64_t timeout_nanoseconds) {
    return wait(timeout_nanoseconds, {}) == WaitOutcome::signaled;
}

Event::WaitOutcome Event::wait(std::int64_t timeout_nanoseconds, std::stop_token stop) {
    std::unique_lock lock(mutex_);
    if (signaled_) return WaitOutcome::signaled;
    if (stop.stop_requested()) return WaitOutcome::cancelled;
    if (timeout_nanoseconds == 0) return WaitOutcome::timed_out;
    const auto observed = signal_generation_;
    bool was_cancelled = false;
    bool completed = false;
    lock.unlock();
    std::stop_callback cancellation(stop, [&] {
        const std::lock_guard callback_lock(mutex_);
        if (signaled_ || signal_generation_ != observed || completed) return;
        was_cancelled = true;
        changed_.notify_all();
    });
    lock.lock();
    const auto ready = [&] {
        return was_cancelled || signaled_ || signal_generation_ != observed;
    };
    bool finished = true;
    if (timeout_nanoseconds < 0) {
        changed_.wait(lock, ready);
    } else {
        using Clock = std::chrono::steady_clock;
        const auto now = Clock::now();
        const auto timeout = std::chrono::nanoseconds(timeout_nanoseconds);
        const auto deadline = timeout < Clock::time_point::max() - now
            ? now + std::chrono::duration_cast<Clock::duration>(timeout)
            : Clock::time_point::max();
        finished = changed_.wait_until(lock, deadline, ready);
    }
    const auto result = !finished ? WaitOutcome::timed_out
                                 : was_cancelled ? WaitOutcome::cancelled
                                                 : WaitOutcome::signaled;
    completed = true;
    lock.unlock();
    return result;
}

EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds) {
    return wait_event(table, handles, timeout_nanoseconds, {});
}

EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds, std::stop_token stop) {
    if (handles.size() != 1)
        return unsupported("Only a single readable event wait is implemented");
    const auto object = table.lookup(handles.front());
    if (!object) return handled(0xe401);
    if (const auto event = std::dynamic_pointer_cast<Event>(object)) {
        switch (event->wait(timeout_nanoseconds, stop)) {
        case Event::WaitOutcome::signaled:
            return handled(0, 0);
        case Event::WaitOutcome::timed_out:
            return handled(0xea01);
        case Event::WaitOutcome::cancelled:
            return cancelled();
        }
    }
    // A thread handle waits for the thread to finish, which is how the guest joins a
    // worker after starting it.
    if (const auto thread = std::dynamic_pointer_cast<ThreadRecord>(object)) {
        switch (thread->wait_for_exit(timeout_nanoseconds, stop)) {
        case ThreadWaitOutcome::exited:
            return handled(0, 0);
        case ThreadWaitOutcome::timed_out:
            return handled(0xea01);
        case ThreadWaitOutcome::cancelled:
            return cancelled();
        }
    }
    // A writable event end is not a synchronization object.
    if (std::dynamic_pointer_cast<WritableEvent>(object)) return handled(0xe401);
    return unsupported("Waiting on this kernel object type is not implemented");
}


EventResult wait_event(HandleTable& table, std::span<const Handle> handles,
                       std::int64_t timeout_nanoseconds, ThreadRecord& caller,
                       std::stop_token stop) {
    if (handles.size() > 64) return handled(0xee01);
    std::array<std::shared_ptr<KernelObject>, 64> objects;
    for (std::size_t index = 0; index != handles.size(); ++index) {
        objects[index] = table.lookup(handles[index]);
        if (!objects[index] ||
            (!std::dynamic_pointer_cast<Event>(objects[index]) &&
             !std::dynamic_pointer_cast<ThreadRecord>(objects[index]) &&
             !std::dynamic_pointer_cast<ProcessRecord>(objects[index])))
            return handled(0xe401);
    }
    if (handles.size() > 1)
        return SynchronizationWait::any(std::span(objects.data(), handles.size()), caller,
                                        timeout_nanoseconds, stop);
    if (handles.empty()) return SynchronizationWait::empty(caller, timeout_nanoseconds, stop);
    if (const auto event = std::dynamic_pointer_cast<Event>(objects[0]))
        return SynchronizationWait::event(*event, caller, timeout_nanoseconds, stop);
    if (const auto thread = std::dynamic_pointer_cast<ThreadRecord>(objects[0]))
        return SynchronizationWait::thread(*thread, caller, timeout_nanoseconds, stop);
    return unsupported("Waiting for process exit is not implemented");
}

EventResult cancel_synchronization(HandleTable& table, Handle handle, ThreadRecord& caller) {
    if (handle == current_thread_handle) {
        caller.cancel_synchronization();
        return handled(0);
    }
    const auto thread = table.lookup_as<ThreadRecord>(handle);
    if (!thread) return handled(0xe401);
    thread->cancel_synchronization();
    return handled(0);
}

EventResult reset_event(HandleTable& table, Handle handle) {
    // The kernel looks for a readable event, then for a process (including the
    // current-process pseudo-handle). Anything else is an invalid handle.
    if (handle == current_process_handle)
        return unsupported("ResetSignal for a process is not implemented");
    const auto object = table.lookup(handle);
    if (!object) return handled(0xe401);
    if (const auto event = std::dynamic_pointer_cast<Event>(object))
        return handled(event->reset() ? 0 : 0xfa01);
    if (std::dynamic_pointer_cast<ProcessRecord>(object))
        return unsupported("ResetSignal for a process is not implemented");
    return handled(0xe401);
}

}  // namespace armigo::horizon
