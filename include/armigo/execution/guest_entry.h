// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Native guest entry: call one prepared AAPCS64 function on an owned
// GuardedStack and come back to the host with its x0/x1 results.
//
// invoke_guest_entry validates the request, claims the stack exclusively, and
// hands a GuestEntryRecord to the assembly bridge. The bridge saves the host
// state it must preserve on the caller's current stack, switches sp to the
// guest stack, calls the function with a fully defined register state,
// verifies that the function returned with sp restored, switches back, and
// restores the saved host state from its own save area.
//
// Call-mode guest entry state:
//   pc        the function; 4-byte aligned, in an executable, non-writable
//             mapping
//   x0, x1    the two request arguments
//   x2..x15,
//   x17,
//   x19..x28  zero
//   x16       the function address (the bridge branches through it)
//   x18       untouched; it is the Apple platform register, not guest state
//   x29       frame record at top - 16 linking to the bridge's host frame
//   x30       bridge return address
//   sp        top - 32, 16-byte aligned; [top - 32, top) belongs to the bridge
//   FPCR      the requested value, or the host's current value
//   FPSR      the requested value, or zero
//   exclusive monitor cleared immediately before entry
//   v0..v31,
//   NZCV      unspecified
//
// The function must return to x30 with sp equal to its entry value and must
// not write the bridge's 32-byte record. A readable record mismatch executes
// BRK #ARMIGO_GUEST_ENTRY_TRAP_IMMEDIATE (SIGTRAP). An unreadable returned sp
// or record can fault before that check. The cookie diagnoses ordinary
// corruption; it cannot authenticate fabricated pointers or sandbox memory.
// Host x19..x30, sp,
// d8..d15, FPCR and FPSR are restored from the bridge's own save area, never
// from guest registers. TPIDR_EL0, TPIDRRO_EL0 and x18 are never read for
// the guest or written by the bridge.
//
// An explicit GateStatus::exit_thread can terminate at a prepared gate without
// returning through guest frames. Host callbacks must first return normally.
// No exception, longjmp, or unwind may cross the bridge: it carries no
// unwind information. Gate handlers are noexcept at the gate boundary
// (gate.h); registered invocations run their C++ callbacks on the saved host
// stack while the native gate frame remains on the guest stack.
//
// This is a call primitive for prepared code. It does not make arbitrary
// guest code safe, schedule threads, or emulate any Horizon service.

#pragma once

// Raw bridge record layout, shared with the assembly.
#define ARMIGO_GUEST_ENTRY_RECORD_FUNCTION_OFFSET 0
#define ARMIGO_GUEST_ENTRY_RECORD_STACK_TOP_OFFSET 8
#define ARMIGO_GUEST_ENTRY_RECORD_X0_OFFSET 16
#define ARMIGO_GUEST_ENTRY_RECORD_X1_OFFSET 24
#define ARMIGO_GUEST_ENTRY_RECORD_FPCR_OFFSET 32
#define ARMIGO_GUEST_ENTRY_RECORD_RESULT_X0_OFFSET 40
#define ARMIGO_GUEST_ENTRY_RECORD_RESULT_X1_OFFSET 48
#define ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPCR_OFFSET 56
#define ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPSR_OFFSET 64
#define ARMIGO_GUEST_ENTRY_RECORD_HOST_SP_OFFSET 72
#define ARMIGO_GUEST_ENTRY_RECORD_COMPLETION_OFFSET 80
#define ARMIGO_GUEST_ENTRY_RECORD_MODE_OFFSET 88
#define ARMIGO_GUEST_ENTRY_RECORD_FPSR_OFFSET 96
#define ARMIGO_GUEST_ENTRY_RECORD_SIZE 104
#define ARMIGO_GUEST_ENTRY_RETURNED 1
#define ARMIGO_GUEST_ENTRY_THREAD_EXITED 2
#define ARMIGO_GUEST_ENTRY_MODE_CALL 0
#define ARMIGO_GUEST_ENTRY_MODE_THREAD 1
#define ARMIGO_GUEST_ENTRY_MODE_PROCESS 2

// Bytes the bridge reserves below the supplied stack top for its return
// record (host sp and check word) and the guest-side frame record.
#define ARMIGO_GUEST_ENTRY_STACK_RESERVE 32

// Bytes the bridge allocates on the caller's (host) stack.
#define ARMIGO_GUEST_ENTRY_HOST_FRAME_SIZE 192

// Return-record check constant, as four 16-bit halfwords (low first).
#define ARMIGO_GUEST_ENTRY_MAGIC_HW0 0x5347
#define ARMIGO_GUEST_ENTRY_MAGIC_HW1 0x4b54
#define ARMIGO_GUEST_ENTRY_MAGIC_HW2 0x4e45
#define ARMIGO_GUEST_ENTRY_MAGIC_HW3 0x4152

// BRK immediate executed when a guest return breaks the entry contract.
#define ARMIGO_GUEST_ENTRY_TRAP_IMMEDIATE 0xae10

#ifndef __ASSEMBLER__

#include <armigo/platform/guest_stack.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace armigo::execution {

// One bridge call. Inputs are read before the switch; results are written
// after a verified return.
struct GuestEntryRecord {
    std::uint64_t function = 0;
    // Call mode: entry sp = stack_top - 32. Thread mode: exact entry sp.
    std::uint64_t stack_top = 0;
    std::uint64_t x0 = 0;
    std::uint64_t x1 = 0;
    std::uint64_t fpcr = 0;  // installed for the guest
    std::uint64_t result_x0 = 0;
    std::uint64_t result_x1 = 0;
    std::uint64_t result_fpcr = 0;  // guest values at return
    std::uint64_t result_fpsr = 0;
    // Owned by the bridge; never supplied by the guest.
    std::uint64_t host_sp = 0;
    std::uint64_t completion = 0;
    // Formerly reserved; this field keeps its original offset. Owned by the host caller.
    std::uint64_t mode = ARMIGO_GUEST_ENTRY_MODE_CALL;
    std::uint64_t fpsr = 0;  // installed for the guest
};

}  // namespace armigo::execution

extern "C" {

// Raw bridge. Performs no validation: the caller must hold exclusive use of
// a live stack. Call mode requires [stack_top - 32, stack_top) for its return
// record; thread mode treats stack_top as exact SP and must exit through a
// registered gate. Use invoke_guest_entry or invoke_guest_thread.
void armigo_execution_guest_entry_bridge(armigo::execution::GuestEntryRecord* record) noexcept;

}  // extern "C"

namespace armigo::execution {

// FPCR bits a request may set explicitly: AHP, DN, FZ, RMode and FZ16.
// Trap-enable and reserved bits are refused.
inline constexpr std::uint64_t kGuestEntryFpcrMask =
    (std::uint64_t{1} << 26) | (std::uint64_t{1} << 25) | (std::uint64_t{1} << 24) |
    (std::uint64_t{3} << 22) | (std::uint64_t{1} << 19);

// Default space that must remain between the entry sp and the usable base.
// Gate entries made by the guest place their native gate frame on this stack.
// Registered invocations run C++ callbacks on their saved host stack.
inline constexpr std::size_t kGuestEntryDefaultMinimumStack = std::size_t{16} << 10;

struct GuestEntryRequest {
    // Entry address of prepared code: nonzero, 4-byte aligned, and inside an
    // executable mapping that is not writable (for example a sealed module).
    std::uint64_t function = 0;
    std::uint64_t x0 = 0;
    std::uint64_t x1 = 0;
    // Stack top inside the guarded stack; defaults to its aligned top. Must be
    // 16-byte aligned. The entry sp is 32 bytes below it.
    std::optional<std::uint64_t> stack_top;
    // Floating-point state for the guest. FPCR defaults to the host's current
    // value; explicit FPCR values may only use kGuestEntryFpcrMask bits. FPSR
    // defaults to zero.
    std::optional<std::uint64_t> fpcr;
    std::uint64_t fpsr = 0;
    // Refuse to enter unless the calling thread has an active GateScope, so a
    // prepared gate site in the guest cannot reach the gate unserved.
    bool require_gate_scope = true;
    // Minimum bytes between the entry sp and the stack's usable base.
    std::size_t minimum_stack_bytes = kGuestEntryDefaultMinimumStack;
};

enum class GuestEntryError : std::uint8_t {
    none,
    null_function,
    misaligned_function,
    function_not_executable,  // unmapped, not executable, or writable
    misaligned_stack_top,
    stack_top_out_of_range,   // top above the aligned top or below the base
    insufficient_stack,       // less than minimum_stack_bytes below the entry sp
    invalid_fpcr,
    no_gate_scope,
    stack_busy,               // the stack is in use on this or another thread
    stack_not_writable,       // borrowed thread stack is not entirely readable/writable
    invalid_process_landing,  // missing, misaligned, nonexecutable, or used outside process mode
};

const char* describe(GuestEntryError error) noexcept;

enum class GuestEntryCompletion : std::uint8_t {
    not_entered = 0,
    returned = ARMIGO_GUEST_ENTRY_RETURNED,
    thread_exited = ARMIGO_GUEST_ENTRY_THREAD_EXITED,
};

struct GuestEntryResult {
    GuestEntryError error = GuestEntryError::none;
    std::uint64_t x0 = 0;
    std::uint64_t x1 = 0;
    std::uint64_t fpcr = 0;  // guest FPCR at return; host FPCR is restored
    std::uint64_t fpsr = 0;  // guest FPSR at return; host FPSR is restored
    GuestEntryCompletion completion = GuestEntryCompletion::not_entered;
    // Both a normal return and an explicit exit are successful completions.
    bool ok() const noexcept { return error == GuestEntryError::none; }
    bool returned() const noexcept {
        return ok() && completion == GuestEntryCompletion::returned;
    }
    bool exited() const noexcept {
        return ok() && completion == GuestEntryCompletion::thread_exited;
    }
};

// Calls request.function on `stack`. A normal return supplies its x0/x1;
// GateStatus::exit_thread instead supplies thread_exited with x0/x1 zero.
// Handler C++ frames return normally before the gate transfers to this bridge.
// Guest frames are abandoned without unwinding: SDK/thread teardown must have
// already finished. Only the innermost invocation terminates.
//
// The calling thread blocks until the function returns; the stack is held
// exclusively for that whole time, so the call nests (a gate handler may
// enter guest code on a different stack) but never reuses a stack that is
// already in use. The stack must outlive the call. Errors are reported
// before any guest instruction executes. Normal return requires the stated
// ABI and memory-lifetime contract; native faults are not recovered here.
GuestEntryResult invoke_guest_entry(platform::GuardedStack& stack,
                                    const GuestEntryRequest& request) noexcept;

struct GuestThreadEntryRequest {
    // Logical prepared entry root. In original-process mode this remains the
    // admission authority even though the bridge first enters transfer_function.
    std::uint64_t function = 0;
    std::uint64_t x0 = 0;
    std::uint64_t x1 = 0;
    // Exact, 16-byte-aligned SP inside or one past the pinned writable span.
    std::uint64_t stack_pointer = 0;
    std::optional<std::uint64_t> fpcr;
    std::uint64_t fpsr = 0;
    bool require_gate_scope = true;
    // Bridge-only landing used by original-process entry. Prepared code must
    // clear x16 and directly branch to function so the real entry does not see
    // the bridge's dynamic branch destination. ThreadRuntime validates the
    // exact two-instruction landing contract for process launches.
    std::uint64_t transfer_function = 0;
    bool original_process = false;
    // Call the function and return when it returns, as invoke_guest_entry
    // does; its 32-byte return record sits just below stack_pointer.
    bool returns = false;
};

// Enters a non-returning prepared thread function at the exact supplied SP.
// No guest return/frame record is reserved, written or cleared. Entry x29 is
// zero; x30 points to a fatal ordinary-return trap. Other entry registers and
// host ABI restoration follow invoke_guest_entry. Completion must come through
// GateStatus::exit_thread after all host callbacks have returned normally.
//
// Original-process mode instead enters transfer_function with x29/x30,
// v0..v31 and NZCV zero. FPCR and FPSR use the request values: process main
// entry supplies zero, while CreateThread workers supply the creator snapshot.
// The bridge must hold transfer_function in x16;
// the required landing clears x16 and directly branches to the logical function.
// An ordinary return therefore follows the firmware LR=0 state rather than the
// synthetic fatal-return path. x18 and thread-pointer system registers remain
// the dispatch/binder's responsibility.
//
// The caller pins the span's mapping/permissions and ensures exclusive use
// until this synchronous call completes. This API does not acquire a
// GuardedStack token, allocate a stack, or impose a 16 KiB minimum. The span
// still must have enough real headroom for guest frames and the gate frame;
// native overflow is not recovered. A current RW mapping check is not a pin.
GuestEntryResult invoke_guest_thread(std::span<std::byte> pinned_stack,
                                     const GuestThreadEntryRequest& request) noexcept;

}  // namespace armigo::execution

#endif  // __ASSEMBLER__
