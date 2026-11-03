// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/guest_entry.h>
#include <armigo/horizon/startup_queries.h>
#include <armigo/runtime/guest_clock.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/launch_plan.h>
#include <functional>
#include <stop_token>
#include <memory>
#include <optional>

namespace armigo::runtime {

// Lane D composes existing ThreadRecord, exact-SP GuestThreadEntryRequest,
// GuestThreadContext, dispatch/slot bindings and shared process waits. Create
// publishes a suspended thread, start executes at most once, exit notification
// occurs only after leaving the guest stack and destroying thread-local bindings.
// Started guests run concurrently on separate host threads, without an execution
// admission lock, including when their virtual processor bindings match. Core
// masks and bindings are coherent query/per-core-lane state, not host CPU pinning
// or mutual exclusion. Priority scheduling and native preemption are not provided.
// The entry bridge calls prepared code on an owned stack; it is not the full
// 20.5.0 launch ABI. Handles are slot-and-serial values. Startup queries report
// configured state and do not allocate memory or start threads. Special
// core values and unsupported scheduling/termination requests need evidenced
// handling or structured errors, never fabricated scheduling success.
struct ThreadLaunch {
    execution::GuestThreadEntryRequest entry;
    GuestMemoryPin stack;
    std::uint32_t priority = 0;
    std::int32_t ideal_core = 0;
    std::uint64_t affinity_mask = 0;
    // Selects the one initial process entry and its x0/x1 semantics. Every
    // thread in an original plan uses the process register bridge internally;
    // callers neither select that bridge mode nor its trusted landing.
    bool original_process = false;
    // host_startup plans: this thread runs the launch's startup calls in order
    // instead of entry.function. host_arguments is the guest argv pointer.
    bool host_startup = false;
    std::uint64_t host_arguments = 0;
};
struct CreatedThread {
    horizon::Handle handle = 0;
    std::shared_ptr<horizon::ThreadRecord> record;
};
enum class ThreadState { created, running, exiting, exited };
struct ThreadSnapshot {
    ThreadState state = ThreadState::created;
    std::uint32_t priority = 0;
    std::int32_t ideal_core = 0;
    std::uint64_t affinity_mask = 0;
    std::optional<LaunchFailure> failure;
};
class ThreadRuntime {
public:
    virtual ~ThreadRuntime() = default;
    // Uses admitted process permissions; validates executable ownership, full
    // pinned stack/SP and exclusive stack use. Failure publishes no handle.
    virtual LaunchResult<CreatedThread> create(const ThreadLaunch& launch) = 0;
    virtual LaunchStatus start(horizon::Handle thread) = 0;
    virtual std::optional<LaunchFailure> failure() const = 0;
    virtual LaunchResult<ThreadSnapshot> snapshot(horizon::Handle thread) const = 0;
    virtual LaunchStatus set_priority(horizon::Handle thread, std::uint32_t priority) = 0;
    virtual LaunchStatus set_core_mask(horizon::Handle thread,
        std::int32_t ideal_core, std::uint64_t mask) = 0;
    // Called on the invoking thread's supervisor boundary. Success authorizes
    // GateStatus::exit_thread; it must not mark_exited while still on guest SP.
    virtual LaunchStatus exit_current() = 0;
    // Idempotent host cancellation: wake owned waits, refuse new starts and
    // request safe gate exits. Never detach, asynchronously free a stack, or
    // claim arbitrary native code can be preempted by this interface.
    virtual void request_stop() noexcept = 0;
    // Once a stop has been requested, ends every guest thread that is executing
    // guest code without a supervisor call in flight, as process termination
    // does; threads in host code are left to reach their boundary. Returns how
    // many started threads have not exited yet (zero when none is left). Does
    // nothing before a stop, and from a worker or cleanup callback.
    virtual std::size_t preempt_guest_threads() noexcept { return 0; }
    virtual LaunchStatus join(std::optional<ClockInstant> deadline) = 0;
};
struct ThreadServiceBinding {
    execution::GuestSupervisor supervisor;
    // Owns per-thread query/IPC adapters and their pinned thread-local storage.
    // Destroyed on the host thread after dispatch/slot bindings, before exit
    // notification. The binder initializes the context's guest thread pointers
    // from the admitted ABI; no SDK-private structure is allocated here.
    std::shared_ptr<void> owner;
    // Runtime-owned scheduling SVCs consult this hook before dispatch. Resume
    // permits the operation, including a cancellation-raced grant that must
    // complete; exit_thread or failed prevents it. With no hook, a requested
    // stop exits at the next scheduling boundary.
    execution::GuestSupervisor scheduling_boundary;
};
struct ThreadRuntimeDependencies {
    LaunchProfile profile;
    PreparedLaunch launch;
    std::shared_ptr<GuestMemory> memory;
    std::shared_ptr<horizon::HandleTable> handles;
    std::shared_ptr<GuestClock> clock;
    // F supplies per-thread services; D invokes on the new host thread before
    // entry. The shared stop token cancels gate/IPC/wait work. Binding failure
    // records the cause and exits without executing guest code. Capture lifetime
    // must not form a runtime->callback->runtime shared-ownership cycle.
    std::function<LaunchResult<ThreadServiceBinding>(
        std::shared_ptr<horizon::ThreadRecord>, execution::GuestThreadContext&,
        std::stop_token)> bind_services;
};
[[nodiscard]] LaunchResult<std::shared_ptr<ThreadRuntime>> create_thread_runtime(
    ThreadRuntimeDependencies dependencies);

// ThreadRuntime polls for exited workers whose handles have closed and reclaims
// them without global shutdown. Handle closure has no notification path, so
// reclamation may lag by one short polling interval. ThreadRecord lifetime still
// follows ordinary handle and shared_ptr ownership. A failed or timed-out join
// retains all dependencies; owner must not destroy it until joined.
// One authoritative ideal-core/priority/mask state also feeds SVC query results.
// D owns cancellation-aware wait helpers; F propagates this same token to every
// blocking service/wait. Cancellation removes wait registrations and preserves
// mutex/CV ownership before safe gate exit, never resumes with fabricated success.
// Repeated short guest waits are not a substitute for cancellation support.

} // namespace armigo::runtime
