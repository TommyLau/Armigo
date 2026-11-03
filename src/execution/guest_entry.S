// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Guest-entry bridge for Apple arm64 (see guest_entry.h).
//
// void armigo_execution_guest_entry_bridge(GuestEntryRecord* record)
//
// Host frame, allocated on the caller's stack (offsets from the host sp):
//     0   x29, x30          frame record
//    16   x19 .. x28
//    96   d8 .. d15
//   160   FPCR, FPSR
//   176   record pointer, expected guest entry sp
//
// Call-mode guest stack, below the supplied top T (16-byte aligned):
//   T-32  host sp, check word = host sp ^ guest sp ^ magic
//   T-16  x29, x30 frame record linking to the host frame
//   The guest starts with sp = T-32 and x29 = T-16.
//
// After the guest returns, the record is read at the guest's sp. Its check
// word binds both stack pointers, and the host frame's copy of the expected
// entry sp must match the returned sp. Any mismatch executes BRK instead of
// adopting an untrusted stack pointer. The record is cleared before the
// switch back so a stale copy can never validate a later return.
//
// Thread and process modes use T as exact SP and write no guest return record.
// Synthetic thread ordinary return traps. Process mode branches with LR zero;
// its transfer landing clears the dynamic x16 target before the real entry.
// An explicit gate exit restores the saved host frame without clearing any
// bytes at the requested guest SP.
//
// x18, TPIDR_EL0 and TPIDRRO_EL0 are never touched. No unwind information
// is emitted: nothing may unwind through this function.
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/gate_frame_layout.h>

#define FRAME ARMIGO_GUEST_ENTRY_HOST_FRAME_SIZE

    .text
    .globl _armigo_execution_guest_entry_bridge
    .p2align 2
_armigo_execution_guest_entry_bridge:
    stp x29, x30, [sp, #-FRAME]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mrs x9, fpcr
    mrs x10, fpsr
    stp x9, x10, [sp, #160]

    ldr x11, [x0, #ARMIGO_GUEST_ENTRY_RECORD_STACK_TOP_OFFSET]
    ldr x10, [x0, #ARMIGO_GUEST_ENTRY_RECORD_MODE_OFFSET]
    cbnz x10, L_armigo_guest_entry_exact_sp
    sub x11, x11, #ARMIGO_GUEST_ENTRY_STACK_RESERVE
L_armigo_guest_entry_exact_sp:
    stp x0, x11, [sp, #176]
    mov x12, sp
    str x12, [x0, #ARMIGO_GUEST_ENTRY_RECORD_HOST_SP_OFFSET]
    str xzr, [x0, #ARMIGO_GUEST_ENTRY_RECORD_COMPLETION_OFFSET]

    cbnz x10, L_armigo_guest_thread_setup
    // Return record and guest-side frame record.
    mov x12, sp
    movz x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW0
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW1, lsl #16
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW2, lsl #32
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW3, lsl #48
    eor x13, x13, x12
    eor x13, x13, x11
    stp x12, x13, [x11]
    adr x14, L_armigo_guest_entry_return
    stp x29, x14, [x11, #16]
    add x29, x11, #16
    mov x30, #0
    b L_armigo_guest_entry_setup

L_armigo_guest_thread_setup:
    // The host-owned mode selects a distinct transfer after register setup.
    mov x29, #0
    mov x30, x10
L_armigo_guest_entry_setup:
    // Guest floating-point control and status.
    ldr x9, [x0, #ARMIGO_GUEST_ENTRY_RECORD_FPCR_OFFSET]
    ldr x10, [x0, #ARMIGO_GUEST_ENTRY_RECORD_FPSR_OFFSET]
    msr fpcr, x9
    msr fpsr, x10

    ldr x16, [x0, #ARMIGO_GUEST_ENTRY_RECORD_FUNCTION_OFFSET]
    ldr x1, [x0, #ARMIGO_GUEST_ENTRY_RECORD_X1_OFFSET]
    ldr x0, [x0, #ARMIGO_GUEST_ENTRY_RECORD_X0_OFFSET]
    mov sp, x11
    mov x2, #0
    mov x3, #0
    mov x4, #0
    mov x5, #0
    mov x6, #0
    mov x7, #0
    mov x8, #0
    mov x9, #0
    mov x10, #0
    mov x11, #0
    mov x12, #0
    mov x13, #0
    mov x14, #0
    mov x15, #0
    mov x17, #0
    mov x19, #0
    mov x20, #0
    mov x21, #0
    mov x22, #0
    mov x23, #0
    mov x24, #0
    mov x25, #0
    mov x26, #0
    mov x27, #0
    mov x28, #0
    // A new guest invocation must not inherit a host exclusive reservation.
    clrex
    cbnz x30, L_armigo_guest_noncall_invoke
    blr x16
L_armigo_guest_entry_return:
    // x0/x1 are the results. Nothing but sp is trusted from here on.
    mrs x2, fpcr
    mrs x3, fpsr
    mov x12, sp
    ldp x9, x10, [x12]
    movz x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW0
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW1, lsl #16
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW2, lsl #32
    movk x13, #ARMIGO_GUEST_ENTRY_MAGIC_HW3, lsl #48
    eor x13, x13, x9
    eor x13, x13, x12
    cmp x10, x13
    b.ne L_armigo_guest_entry_violation
    ldr x14, [x9, #184]
    cmp x14, x12
    b.ne L_armigo_guest_entry_violation
    stp xzr, xzr, [x12]
    stp xzr, xzr, [x12, #16]
    mov sp, x9

    ldr x14, [sp, #176]
    stp x0, x1, [x14, #ARMIGO_GUEST_ENTRY_RECORD_RESULT_X0_OFFSET]
    stp x2, x3, [x14, #ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPCR_OFFSET]

    mov x10, #ARMIGO_GUEST_ENTRY_RETURNED
    str x10, [x14, #ARMIGO_GUEST_ENTRY_RECORD_COMPLETION_OFFSET]
L_armigo_guest_entry_restore_host:
    str xzr, [x14, #ARMIGO_GUEST_ENTRY_RECORD_HOST_SP_OFFSET]
    ldp x9, x10, [sp, #160]
    msr fpcr, x9
    msr fpsr, x10
    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #FRAME
    ret

// Called only by gate assembly after its dispatcher and handlers have returned.
// x0 is the innermost invocation's host-owned record; x1 is the gate frame.
// The saved host sp does not depend on the guest's current frame depth or LR.
    .globl _armigo_execution_guest_entry_exit
    .private_extern _armigo_execution_guest_entry_exit
    .p2align 2
_armigo_execution_guest_entry_exit:
    ldr x9, [x0, #ARMIGO_GUEST_ENTRY_RECORD_HOST_SP_OFFSET]
    cbz x9, L_armigo_guest_entry_violation
    ldr x10, [x9, #176]
    cmp x10, x0
    b.ne L_armigo_guest_entry_violation
    ldr x10, [x0, #ARMIGO_GUEST_ENTRY_RECORD_MODE_OFFSET]
    cbnz x10, L_armigo_guest_entry_exit_snapshot
    ldr x12, [x9, #184]
    stp xzr, xzr, [x12]
    stp xzr, xzr, [x12, #16]
L_armigo_guest_entry_exit_snapshot:
    ldr x2, [x1, #ARMIGO_GATE_FRAME_FPCR_OFFSET]
    ldr x3, [x1, #ARMIGO_GATE_FRAME_FPSR_OFFSET]
    mov x14, x0
    mov sp, x9
    stp xzr, xzr, [x14, #ARMIGO_GUEST_ENTRY_RECORD_RESULT_X0_OFFSET]
    stp x2, x3, [x14, #ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPCR_OFFSET]
    mov x10, #ARMIGO_GUEST_ENTRY_THREAD_EXITED
    str x10, [x14, #ARMIGO_GUEST_ENTRY_RECORD_COMPLETION_OFFSET]
    clrex
    b L_armigo_guest_entry_restore_host

L_armigo_guest_noncall_invoke:
    cmp x30, #ARMIGO_GUEST_ENTRY_MODE_PROCESS
    b.eq L_armigo_guest_process_invoke
    blr x16
    // A thread entry has no guest return record to inspect or adopt.
    b L_armigo_guest_entry_violation

L_armigo_guest_process_invoke:
    mov x30, xzr
    movi v0.2d, #0
    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0
    movi v5.2d, #0
    movi v6.2d, #0
    movi v7.2d, #0
    movi v8.2d, #0
    movi v9.2d, #0
    movi v10.2d, #0
    movi v11.2d, #0
    movi v12.2d, #0
    movi v13.2d, #0
    movi v14.2d, #0
    movi v15.2d, #0
    movi v16.2d, #0
    movi v17.2d, #0
    movi v18.2d, #0
    movi v19.2d, #0
    movi v20.2d, #0
    movi v21.2d, #0
    movi v22.2d, #0
    movi v23.2d, #0
    movi v24.2d, #0
    movi v25.2d, #0
    movi v26.2d, #0
    movi v27.2d, #0
    movi v28.2d, #0
    movi v29.2d, #0
    movi v30.2d, #0
    movi v31.2d, #0
    msr nzcv, xzr
    br x16

L_armigo_guest_entry_violation:
    brk #ARMIGO_GUEST_ENTRY_TRAP_IMMEDIATE
    b L_armigo_guest_entry_violation
