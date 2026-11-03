// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Thread-local dispatcher for the AArch64 host gate.
//
// The assembly wrapper selects an invocation's host callback stack when one
// is supplied; ordinary gates retain their caller stack. The function it calls
// is noexcept and catches everything a handler can throw, so no exception
// ever unwinds into handwritten code.

#include <armigo/execution/gate.h>

#include "gate_exit.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include <pthread.h>
#include <unistd.h>

namespace armigo::execution {
namespace {

// Per-thread dispatcher state. Ordinary thread_local storage: the gate makes
// no assumption about the host's thread-specific-data layout.
struct ThreadGateState {
    GateScope* innermost_scope = nullptr;
    GateCall* innermost_call = nullptr;
    std::uint32_t depth = 0;
    detail::GuestExitScope* guest_entry = nullptr;
    // Default virtual guest x18, used while no scope selects other storage.
    std::uint64_t guest_x18 = 0;
    // Word selected by the innermost scope; null selects `guest_x18`.
    std::uint64_t* selected_guest_x18 = nullptr;

    std::uint64_t* current_guest_x18() noexcept {
        return selected_guest_x18 != nullptr ? selected_guest_x18 : &guest_x18;
    }
};

thread_local ThreadGateState t_gate_state;
// Slot of the guest preemption binding of this thread, if any.
thread_local GuestPreemptSlot* t_preempt_slot = nullptr;

std::atomic<GateTrapHook> g_trap_hook{nullptr};

// Reports the failure and ends the process. Formatting uses a stack buffer
// so a trap never depends on the heap.
[[noreturn]] void trap(const GateFailure& failure) noexcept {
    if (const GateTrapHook hook = g_trap_hook.load(std::memory_order_acquire)) {
        hook(failure);
    }
    char text[256];
    const int length = std::snprintf(
        text, sizeof text,
        "armigo host gate: fatal %s (%s) at return address 0x%llx depth %u\n",
        describe(failure.kind), failure.message ? failure.message : "no detail",
        static_cast<unsigned long long>(failure.return_address),
        static_cast<unsigned>(failure.depth));
    if (length > 0) {
        const auto count = static_cast<std::size_t>(length) < sizeof text
                               ? static_cast<std::size_t>(length)
                               : sizeof text - 1;
        (void)::write(STDERR_FILENO, text, count);
    }
    std::abort();
}

}  // namespace

const char* describe(GateFailureKind kind) noexcept {
    switch (kind) {
    case GateFailureKind::none:
        return "no failure";
    case GateFailureKind::handler_reported:
        return "handler reported failure";
    case GateFailureKind::handler_threw:
        return "handler threw an exception";
    case GateFailureKind::host_reserved_slot_modified:
        return "handler modified the host-reserved x18 slot";
    case GateFailureKind::stack_pointer_modified:
        return "handler modified the saved stack pointer";
    case GateFailureKind::no_scope:
        return "gate entered without an active GateScope on this thread";
    case GateFailureKind::invalid_frame:
        return "dispatcher received a frame without the gate marker";
    case GateFailureKind::scope_misuse:
        return "GateScope destroyed out of order, on another thread, or while in use";
    case GateFailureKind::invalid_policy:
        return "unsupported gate failure policy";
    case GateFailureKind::exit_unavailable:
        return "thread exit requires an innermost guest entry without an active outer callback";
    }
    return "unknown failure";
}

GuestPreemptBinding::GuestPreemptBinding(GuestPreemptSlot& slot) noexcept
    : slot_(slot), previous_(t_preempt_slot) {
    const std::lock_guard lock(slot.mutex);
    slot.thread.store(pthread_mach_thread_np(pthread_self()), std::memory_order_release);
    slot.record.store(nullptr, std::memory_order_release);
    t_preempt_slot = &slot;
}

GuestPreemptBinding::~GuestPreemptBinding() {
    {
        const std::lock_guard lock(slot_.mutex);
        slot_.record.store(nullptr, std::memory_order_release);
        slot_.thread.store(0, std::memory_order_release);
    }
    t_preempt_slot = previous_;
}

namespace detail {

PreemptPublication publish_invocation(void* record, std::uint64_t stack_low,
                                      std::uint64_t stack_high) noexcept {
    auto* slot = t_preempt_slot;
    if (slot == nullptr) return {};
    const PreemptPublication previous{slot->record.load(std::memory_order_relaxed),
                                      slot->stack_low.load(std::memory_order_relaxed),
                                      slot->stack_high.load(std::memory_order_relaxed)};
    // The range goes first: a stopper reads the record last.
    slot->record.store(nullptr, std::memory_order_release);
    slot->stack_low.store(stack_low, std::memory_order_release);
    slot->stack_high.store(stack_high, std::memory_order_release);
    slot->record.store(stack_high != 0 ? record : nullptr, std::memory_order_release);
    return previous;
}

void restore_invocation(const PreemptPublication& previous) noexcept {
    auto* slot = t_preempt_slot;
    if (slot == nullptr) return;
    slot->record.store(nullptr, std::memory_order_release);
    slot->stack_low.store(previous.stack_low, std::memory_order_release);
    slot->stack_high.store(previous.stack_high, std::memory_order_release);
    slot->record.store(previous.record, std::memory_order_release);
}

}  // namespace detail

GuestPreemptHold::GuestPreemptHold() noexcept {
    const auto previous = detail::publish_invocation(nullptr, 0, 0);
    saved_ = {previous.record, previous.stack_low, previous.stack_high};
}

GuestPreemptHold::~GuestPreemptHold() {
    detail::restore_invocation({saved_.record, saved_.stack_low, saved_.stack_high});
}

namespace detail {

static_assert(offsetof(GateExitTarget, continuation) == 0);
static_assert(offsetof(GateExitTarget, record) == 8);
static_assert(sizeof(GateExitTarget) == 16);

GuestExitScope::GuestExitScope(std::uintptr_t continuation, void* record,
                               const std::uint64_t* saved_host_sp,
                               std::uint64_t stack_low, std::uint64_t stack_high) noexcept
    : target_{continuation, record}, outer_(t_gate_state.guest_entry),
      owner_(&t_gate_state), entry_depth_(t_gate_state.depth), saved_host_sp_(saved_host_sp),
      outer_publication_(publish_invocation(record, stack_low, stack_high)) {
    t_gate_state.guest_entry = this;
}

GuestExitScope::~GuestExitScope() {
    restore_invocation(outer_publication_);
    if (owner_ != &t_gate_state || t_gate_state.guest_entry != this ||
        t_gate_state.depth != entry_depth_) {
        trap({GateFailureKind::scope_misuse, "guest entry destroyed while a callback is active",
              0, t_gate_state.depth});
    }
    t_gate_state.guest_entry = outer_;
}

const GateExitTarget* GuestExitScope::target(std::uint32_t depth) noexcept {
    const auto* entry = t_gate_state.guest_entry;
    return entry != nullptr && depth != 0 && depth - 1 == entry->entry_depth_
               ? &entry->target_ : nullptr;
}

std::uintptr_t GuestExitScope::callback_stack() noexcept {
    const auto* entry = t_gate_state.guest_entry;
    if (entry == nullptr || entry->saved_host_sp_ == nullptr ||
        t_gate_state.depth != entry->entry_depth_) return 0;
    // An active callback already runs below this saved SP. Keeping its current
    // stack on direct reentry avoids overwriting live callback frames. A nested
    // guest invocation instead installs its own scope at the current depth.
    const auto saved = static_cast<std::uintptr_t>(*entry->saved_host_sp_);
    return saved != 0 ? saved : 1;
}

} // namespace detail

bool GateCall::can_exit_thread() const noexcept {
    return t_gate_state.innermost_call == this && t_gate_state.innermost_scope == scope_ &&
           detail::GuestExitScope::target(depth_) != nullptr;
}

// ---------------------------------------------------------------------------
// GateCall
// ---------------------------------------------------------------------------

std::uint64_t GateCall::x(unsigned index) const noexcept {
    if (index == kGateHostReservedRegister) {
        return *guest_x18_;
    }
    return index < kGateGeneralRegisterCount ? frame_->x[index] : 0;
}

bool GateCall::set_x(unsigned index, std::uint64_t value) noexcept {
    if (index == kGateHostReservedRegister) {
        *guest_x18_ = value;
        return true;
    }
    if (index >= kGateGeneralRegisterCount) {
        return false;
    }
    frame_->x[index] = value;
    return true;
}

GateVector GateCall::v(unsigned index) const noexcept {
    return index < kGateVectorRegisterCount ? frame_->v[index] : GateVector{{0, 0}};
}

bool GateCall::set_v(unsigned index, GateVector value) noexcept {
    if (index >= kGateVectorRegisterCount) {
        return false;
    }
    frame_->v[index] = value;
    return true;
}

// ---------------------------------------------------------------------------
// GateScope
// ---------------------------------------------------------------------------

GateScope::GateScope(GateHandler handler, void* context, GateFailurePolicy policy,
                     std::uint64_t* guest_x18_storage) noexcept
    : handler_(handler),
      context_(context),
      policy_(policy),
      outer_(t_gate_state.innermost_scope),
      owner_thread_state_(&t_gate_state),
      guest_x18_(guest_x18_storage != nullptr ? guest_x18_storage
                                              : t_gate_state.current_guest_x18()) {
    if (policy != GateFailurePolicy::trap && policy != GateFailurePolicy::resume_unchanged) {
        trap(GateFailure{GateFailureKind::invalid_policy, "invalid scope failure policy",
                         0, t_gate_state.depth});
    }
    t_gate_state.innermost_scope = this;
    t_gate_state.selected_guest_x18 = guest_x18_;
}

GateScope::~GateScope() {
    if (owner_thread_state_ != &t_gate_state || t_gate_state.innermost_scope != this) {
        trap(GateFailure{GateFailureKind::scope_misuse,
                         owner_thread_state_ != &t_gate_state
                             ? "scope destroyed on a different thread"
                             : "scope destroyed while an inner scope is still active",
                         0, t_gate_state.depth});
    }
    if (active_dispatches_ != 0) {
        trap(GateFailure{GateFailureKind::scope_misuse,
                         "scope destroyed while its callback is active", 0, t_gate_state.depth});
    }
    t_gate_state.innermost_scope = outer_;
    t_gate_state.selected_guest_x18 = outer_ != nullptr ? outer_->guest_x18_ : nullptr;
}

void GateScope::clear_failures() noexcept {
    failure_count_ = 0;
    first_failure_ = GateFailure{};
}

void GateScope::record_failure(const GateFailure& failure) noexcept {
    if (failure_count_ == 0) {
        first_failure_ = failure;
    }
    ++failure_count_;
}

bool GateScope::thread_has_scope() noexcept {
    return t_gate_state.innermost_scope != nullptr;
}

std::uint32_t GateScope::thread_gate_depth() noexcept {
    return t_gate_state.depth;
}

std::uint64_t thread_guest_x18() noexcept {
    return *t_gate_state.current_guest_x18();
}

void set_thread_guest_x18(std::uint64_t value) noexcept {
    *t_gate_state.current_guest_x18() = value;
}

GateTrapHook set_gate_trap_hook(GateTrapHook hook) noexcept {
    return g_trap_hook.exchange(hook, std::memory_order_acq_rel);
}

}  // namespace armigo::execution

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

// This small pre-dispatch accessor is called before the assembly switches SP.
// GuestExitScope construction has already initialized this thread's TLS state.
extern "C" std::uintptr_t armigo_execution_gate_callback_stack() noexcept {
    return armigo::execution::detail::GuestExitScope::callback_stack();
}

extern "C" const armigo::execution::detail::GateExitTarget*
armigo_execution_gate_dispatch(armigo::execution::GateFrame* frame) noexcept {
    using namespace armigo::execution;

    ThreadGateState& state = t_gate_state;
    const std::uint64_t return_address =
        frame != nullptr ? frame->x[kGateLinkRegister] : 0;

    if (frame == nullptr || frame->magic != kGateFrameMagic ||
        frame->sp != reinterpret_cast<std::uintptr_t>(frame) + sizeof(GateFrame)) {
        trap(GateFailure{GateFailureKind::invalid_frame, "frame marker or sp mismatch",
                         return_address, state.depth});
    }

    GateScope* const scope = state.innermost_scope;
    if (scope == nullptr) {
        trap(GateFailure{GateFailureKind::no_scope, "install a GateScope before entering the gate",
                         return_address, state.depth});
    }

    // The call keeps the x18 word selected at entry for its whole duration;
    // scopes installed by the handler cannot redirect it or its rollback.
    std::uint64_t* const guest_x18 = scope->guest_x18_;
    GateCall call(frame, state.innermost_call, state.depth + 1, guest_x18, scope);
    state.innermost_call = &call;
    state.depth = call.depth();

    // Snapshot so a failed handler cannot leave partial changes behind.
    const GateFrame snapshot = *frame;
    const std::uint64_t guest_x18_snapshot = *guest_x18;

    GateFailure failure{GateFailureKind::none, nullptr, return_address, call.depth()};
    ++scope->active_dispatches_;
    GateStatus status = GateStatus::failed;
    try {
        status = scope->handler_ != nullptr ? scope->handler_(call, scope->context_)
                                           : GateStatus::failed;
        if (scope->handler_ == nullptr) {
            failure.kind = GateFailureKind::handler_reported;
            failure.message = "the innermost GateScope has no handler";
        } else if (status != GateStatus::resume && status != GateStatus::exit_thread) {
            failure.kind = GateFailureKind::handler_reported;
            failure.message = call.failure_message();
        }
    } catch (const std::exception& error) {
        failure.kind = GateFailureKind::handler_threw;
        failure.message = "std::exception escaped the handler";
        (void)error;
    } catch (...) {
        failure.kind = GateFailureKind::handler_threw;
        failure.message = "non-standard exception escaped the handler";
    }

    if (failure.kind == GateFailureKind::none) {
        if (frame->x[kGateHostReservedRegister] != 0) {
            failure.kind = GateFailureKind::host_reserved_slot_modified;
            failure.message = "use GateCall::set_guest_x18 for the virtual guest x18";
        } else if (frame->sp != snapshot.sp) {
            failure.kind = GateFailureKind::stack_pointer_modified;
            failure.message = "the gate cannot relocate the caller's stack";
        }
    }

    const detail::GateExitTarget* exit = nullptr;
    if (failure.kind == GateFailureKind::none && status == GateStatus::exit_thread) {
        if (!call.can_exit_thread() || state.innermost_scope != scope) {
            failure.kind = GateFailureKind::exit_unavailable;
            failure.message = "thread exit would abandon a host callback or scope";
        } else {
            exit = detail::GuestExitScope::target(call.depth());
        }
    }

    state.innermost_call = const_cast<GateCall*>(call.outer());
    state.depth = call.depth() - 1;
    --scope->active_dispatches_;

    if (failure.kind == GateFailureKind::none) {
        return exit;
    }

    if (scope->policy_ == GateFailurePolicy::trap) {
        trap(failure);
    }

    scope->record_failure(failure);
    *frame = snapshot;
    *guest_x18 = guest_x18_snapshot;
    return nullptr;
}
