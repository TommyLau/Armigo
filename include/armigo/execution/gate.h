// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// AArch64 host gate: the thread-local dispatcher behind
// `armigo_execution_gate_entry`.
//
// Guest-like code enters the gate with a direct `bl`. The assembly saves the
// observable register state into a `GateFrame` on the caller's stack and
// calls the C++ dispatcher, which invokes the handler installed by the
// innermost `GateScope` on the current thread. When the handler returns
// `GateStatus::resume`, the assembly reloads the frame and returns to the
// address in `x[30]`, so every explicit change the handler made to the frame
// becomes visible to the caller. When the handler fails, the scope's failure
// policy decides between trapping the process and resuming the caller with
// the exact pre-gate state; failures are never silent.
#pragma once

#include <armigo/execution/gate_frame.h>

#include <cstdint>

namespace armigo::execution::detail { struct GateExitTarget; }

extern "C" {

// Gate entry. Enter with a direct `bl` from code that follows the AAPCS64
// stack discipline (sp 16-byte aligned, no live data below sp). Not callable
// as a C function: it has no C prototype and clobbers nothing observable.
void armigo_execution_gate_entry();

// Dispatcher called by the gate with x0 = frame. Exposed for the assembly and
// for frame-validation tests only; never call it with a frame that was not
// produced by the gate. Null resumes normally; the opaque non-null target is
// consumed by gate assembly after all C++ dispatcher frames have returned.
const armigo::execution::detail::GateExitTarget*
armigo_execution_gate_dispatch(armigo::execution::GateFrame* frame) noexcept;

}  // extern "C"

namespace armigo::execution {

enum class GateStatus : std::uint32_t {
    resume = 0,  // Apply the frame changes and return to the caller.
    failed = 1,  // Report a failure; the scope policy decides what happens.
    // Finish the innermost invoke_guest_entry after all handler frames return.
    // No guest continuation runs. Requires can_exit_thread().
    exit_thread = 2,
};

enum class GateFailureKind : std::uint32_t {
    none = 0,
    handler_reported,             // The handler returned GateStatus::failed.
    handler_threw,                // The handler let an exception escape.
    host_reserved_slot_modified,  // The handler wrote the frame's x18 slot.
    stack_pointer_modified,       // The handler changed the frame's sp field.
    no_scope,                     // The gate was entered without a GateScope.
    invalid_frame,                // The dispatcher received a frame without the gate marker.
    scope_misuse,                 // A scope was destroyed out of order, on another thread, or in use.
    invalid_policy,               // The requested failure policy is not supported.
    exit_unavailable,             // No entry, or exit would abandon a host callback.
};

const char* describe(GateFailureKind kind) noexcept;

struct GateFailure {
    GateFailureKind kind = GateFailureKind::none;
    const char* message = nullptr;  // Static text; never owned.
    std::uint64_t return_address = 0;
    std::uint32_t depth = 0;
};

enum class GateFailurePolicy : std::uint32_t {
    trap = 0,              // Report through the trap hook and abort the process.
    resume_unchanged = 1,  // Record the failure on the scope and resume with the pre-gate state.
};

class GateScope;

// The typed view handed to a handler for one gate entry.
class GateCall {
public:
    GateCall(const GateCall&) = delete;
    GateCall& operator=(const GateCall&) = delete;

    // Raw frame. `x[18]` must stay zero and `sp` must stay unchanged; use
    // `set_guest_x18` for the virtual guest x18.
    GateFrame& frame() noexcept { return *frame_; }
    const GateFrame& frame() const noexcept { return *frame_; }

    // x30 as captured at entry: the address after the caller's `bl`, which
    // identifies the gate site. Changing `x(30)` redirects the return.
    std::uint64_t return_address() const noexcept { return return_address_; }

    // 1 for a gate entered from the thread's outermost guest-like code; one
    // more for each gate entered from inside a handler.
    std::uint32_t depth() const noexcept { return depth_; }
    const GateCall* outer() const noexcept { return outer_; }

    // General registers 0..30. Index 18 reads and writes this call's
    // virtual guest x18 storage (see guest_x18_storage), never the host
    // register.
    std::uint64_t x(unsigned index) const noexcept;
    bool set_x(unsigned index, std::uint64_t value) noexcept;

    // The caller's sp at entry. Read-only.
    std::uint64_t sp() const noexcept { return frame_->sp; }

    GateVector v(unsigned index) const noexcept;
    bool set_v(unsigned index, GateVector value) noexcept;

    std::uint64_t nzcv() const noexcept { return frame_->nzcv; }
    void set_nzcv(std::uint64_t value) noexcept { frame_->nzcv = value; }
    std::uint64_t fpsr() const noexcept { return frame_->fpsr; }
    void set_fpsr(std::uint64_t value) noexcept { frame_->fpsr = value; }
    std::uint64_t fpcr() const noexcept { return frame_->fpcr; }
    void set_fpcr(std::uint64_t value) noexcept { frame_->fpcr = value; }

    // The virtual guest x18 of this entry. The storage is the one selected
    // on the thread when the gate was entered and stays fixed for the whole
    // call: scopes a handler installs for nested entries, including ones that
    // select different storage, never redirect this call's reads, writes or
    // failure rollback.
    std::uint64_t guest_x18() const noexcept { return *guest_x18_; }
    void set_guest_x18(std::uint64_t value) noexcept { *guest_x18_ = value; }
    const std::uint64_t* guest_x18_storage() const noexcept { return guest_x18_; }

    // True only when exit can leave the innermost guest invocation without
    // abandoning another C++ gate callback. Return GateStatus::exit_thread
    // normally; do not longjmp, throw, or pthread_exit across guest frames.
    bool can_exit_thread() const noexcept;

    // Optional static text attached to a GateStatus::failed result.
    void set_failure_message(const char* message) noexcept { failure_message_ = message; }
    const char* failure_message() const noexcept { return failure_message_; }

private:
    friend const detail::GateExitTarget* ::armigo_execution_gate_dispatch(GateFrame* frame) noexcept;

    GateCall(GateFrame* frame, const GateCall* outer, std::uint32_t depth,
             std::uint64_t* guest_x18, const GateScope* scope) noexcept
        : frame_(frame),
          guest_x18_(guest_x18),
          scope_(scope),
          outer_(outer),
          depth_(depth),
          return_address_(frame->x[kGateLinkRegister]) {}

    GateFrame* frame_;
    std::uint64_t* guest_x18_;
    const GateScope* scope_;
    const GateCall* outer_;
    std::uint32_t depth_;
    std::uint64_t return_address_;
    const char* failure_message_ = nullptr;
};

// Handler invoked for every gate entry while its scope is the innermost one
// on the thread. Exceptions are caught by the dispatcher and reported as
// GateFailureKind::handler_threw; they never reach the assembly.
// The assembly clears the local exclusive monitor before every guest resume,
// including redirected and failure-policy returns, after all host frames exit.
using GateHandler = GateStatus (*)(GateCall& call, void* context);

// Per-thread, lexically scoped handler ownership.
//
// Scopes stack in strict LIFO order on the thread that constructed them. The
// innermost scope receives every gate entry, including entries made from a
// handler that calls back into guest-like code. Destroying a scope that is
// not the innermost one, belongs to another thread, or still has an active
// callback is a programming error that traps the process. Unsupported
// failure policies are rejected at construction.
//
// Virtual guest x18 storage selection: a scope constructed with a non-null
// `guest_x18_storage` selects that word as the thread's virtual guest x18
// while it is the innermost scope; a scope constructed without one keeps the
// selection that was current when it was constructed (an outer scope's, or
// the thread's own default word). Destroying a scope restores the previous
// selection with it, so no selected word is ever lost or reassigned. The
// caller owns explicit storage: it must stay valid for the scope's whole
// lifetime and must not be accessed concurrently by another thread while
// gate entries or thread_guest_x18 may use it on this one.
class GateScope {
public:
    explicit GateScope(GateHandler handler, void* context = nullptr,
                       GateFailurePolicy policy = GateFailurePolicy::trap,
                       std::uint64_t* guest_x18_storage = nullptr) noexcept;
    ~GateScope();

    GateScope(const GateScope&) = delete;
    GateScope& operator=(const GateScope&) = delete;

    GateHandler handler() const noexcept { return handler_; }
    void* context() const noexcept { return context_; }
    GateFailurePolicy policy() const noexcept { return policy_; }
    // The virtual guest x18 word gate entries use while this scope is the
    // innermost one: its own storage or the inherited selection.
    const std::uint64_t* guest_x18_storage() const noexcept { return guest_x18_; }

    // Failures recorded under GateFailurePolicy::resume_unchanged. The first
    // failure is kept in full; later failures only increment the count.
    std::uint64_t failure_count() const noexcept { return failure_count_; }
    const GateFailure* first_failure() const noexcept {
        return failure_count_ == 0 ? nullptr : &first_failure_;
    }
    void clear_failures() noexcept;

    // True when the calling thread has at least one active scope.
    static bool thread_has_scope() noexcept;
    // Number of gate entries currently in progress on the calling thread.
    static std::uint32_t thread_gate_depth() noexcept;

private:
    friend const detail::GateExitTarget* ::armigo_execution_gate_dispatch(GateFrame* frame) noexcept;

    void record_failure(const GateFailure& failure) noexcept;

    GateHandler handler_;
    void* context_;
    GateFailurePolicy policy_;
    GateScope* outer_;
    void* owner_thread_state_;
    std::uint64_t* guest_x18_;
    GateFailure first_failure_{};
    std::uint64_t failure_count_ = 0;
    std::uint32_t active_dispatches_ = 0;
};

// The virtual guest x18 currently selected on the calling thread: the word
// selected by the innermost GateScope (see GateScope), or the thread's own
// default word when no scope selects one. The default is ordinary
// thread-local storage owned by the dispatcher; the hardware x18 is never
// written.
std::uint64_t thread_guest_x18() noexcept;
void set_thread_guest_x18(std::uint64_t value) noexcept;

// Process-wide hook consulted before the dispatcher aborts the process. The
// hook must not return; if it does, the dispatcher still aborts. Returns the
// previous hook.
using GateTrapHook = void (*)(const GateFailure& failure) noexcept;
GateTrapHook set_gate_trap_hook(GateTrapHook hook) noexcept;

}  // namespace armigo::execution
