// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared assembly body for the linked entry, local gate, and embedded gate.
// ARMIGO_GATE_RELOCATABLE selects the call sequence; X_SLOT and V_SLOT are
// defined by gate_entry.S. This header is not a C++ interface.

    sub sp, sp, #ARMIGO_GATE_FRAME_SIZE
    stp x0, x1, [sp, #X_SLOT(0)]
    stp x2, x3, [sp, #X_SLOT(2)]
    stp x4, x5, [sp, #X_SLOT(4)]
    stp x6, x7, [sp, #X_SLOT(6)]
    stp x8, x9, [sp, #X_SLOT(8)]
    stp x10, x11, [sp, #X_SLOT(10)]
    stp x12, x13, [sp, #X_SLOT(12)]
    stp x14, x15, [sp, #X_SLOT(14)]
    stp x16, x17, [sp, #X_SLOT(16)]
    // Slot 18 receives zero: the platform register is not captured.
    stp xzr, x19, [sp, #X_SLOT(18)]
    stp x20, x21, [sp, #X_SLOT(20)]
    stp x22, x23, [sp, #X_SLOT(22)]
    stp x24, x25, [sp, #X_SLOT(24)]
    stp x26, x27, [sp, #X_SLOT(26)]
    stp x28, x29, [sp, #X_SLOT(28)]
    str x30, [sp, #X_SLOT(30)]

    // Frame record for host backtraces, then x29 points at it. The record
    // sits beyond the reach of a pair store from sp, so go through x9.
    add x9, sp, #ARMIGO_GATE_FRAME_RECORD_OFFSET
    stp x29, x30, [x9]
    mov x29, x9

    // The caller's sp before the frame was allocated.
    add x9, sp, #ARMIGO_GATE_FRAME_SIZE
    str x9, [sp, #ARMIGO_GATE_FRAME_SP_OFFSET]

    stp q0, q1, [sp, #V_SLOT(0)]
    stp q2, q3, [sp, #V_SLOT(2)]
    stp q4, q5, [sp, #V_SLOT(4)]
    stp q6, q7, [sp, #V_SLOT(6)]
    stp q8, q9, [sp, #V_SLOT(8)]
    stp q10, q11, [sp, #V_SLOT(10)]
    stp q12, q13, [sp, #V_SLOT(12)]
    stp q14, q15, [sp, #V_SLOT(14)]
    stp q16, q17, [sp, #V_SLOT(16)]
    stp q18, q19, [sp, #V_SLOT(18)]
    stp q20, q21, [sp, #V_SLOT(20)]
    stp q22, q23, [sp, #V_SLOT(22)]
    stp q24, q25, [sp, #V_SLOT(24)]
    stp q26, q27, [sp, #V_SLOT(26)]
    stp q28, q29, [sp, #V_SLOT(28)]
    stp q30, q31, [sp, #V_SLOT(30)]

    mrs x9, nzcv
    str x9, [sp, #ARMIGO_GATE_FRAME_NZCV_OFFSET]
    mrs x9, fpsr
    str x9, [sp, #ARMIGO_GATE_FRAME_FPSR_OFFSET]
    mrs x9, fpcr
    str x9, [sp, #ARMIGO_GATE_FRAME_FPCR_OFFSET]

    movz x9, #ARMIGO_GATE_FRAME_MAGIC_HW0
    movk x9, #ARMIGO_GATE_FRAME_MAGIC_HW1, lsl #16
    movk x9, #ARMIGO_GATE_FRAME_MAGIC_HW2, lsl #32
    movk x9, #ARMIGO_GATE_FRAME_MAGIC_HW3, lsl #48
    str x9, [sp, #ARMIGO_GATE_FRAME_MAGIC_OFFSET]

    mov x0, sp
#if ARMIGO_GATE_RELOCATABLE == 2
_armigo_execution_embedded_gate_load:
    adrp x16, 0
    ldr x16, [x16]
    blr x16
#elif ARMIGO_GATE_RELOCATABLE
    ldr x16, L_armigo_gate_dispatch_address
    blr x16
#else
    bl _armigo_execution_gate_dispatch_on_host_stack
#endif

    // The dispatcher has returned through all C++ frames. A non-null target
    // abandons only guest frames and restores the active entry's host stack.
    cbz x0, 1f
    mov x1, sp
    ldp x16, x0, [x0]
    br x16
1:

    ldr x9, [sp, #ARMIGO_GATE_FRAME_FPCR_OFFSET]
    msr fpcr, x9
    ldr x9, [sp, #ARMIGO_GATE_FRAME_FPSR_OFFSET]
    msr fpsr, x9
    ldr x9, [sp, #ARMIGO_GATE_FRAME_NZCV_OFFSET]
    msr nzcv, x9

    ldp q0, q1, [sp, #V_SLOT(0)]
    ldp q2, q3, [sp, #V_SLOT(2)]
    ldp q4, q5, [sp, #V_SLOT(4)]
    ldp q6, q7, [sp, #V_SLOT(6)]
    ldp q8, q9, [sp, #V_SLOT(8)]
    ldp q10, q11, [sp, #V_SLOT(10)]
    ldp q12, q13, [sp, #V_SLOT(12)]
    ldp q14, q15, [sp, #V_SLOT(14)]
    ldp q16, q17, [sp, #V_SLOT(16)]
    ldp q18, q19, [sp, #V_SLOT(18)]
    ldp q20, q21, [sp, #V_SLOT(20)]
    ldp q22, q23, [sp, #V_SLOT(22)]
    ldp q24, q25, [sp, #V_SLOT(24)]
    ldp q26, q27, [sp, #V_SLOT(26)]
    ldp q28, q29, [sp, #V_SLOT(28)]
    ldp q30, q31, [sp, #V_SLOT(30)]

    // Every general register except x18 is reloaded so that handler changes,
    // including a redirected x30, take effect. The remaining loads and the
    // final `add` do not touch the flags restored above.
    ldp x0, x1, [sp, #X_SLOT(0)]
    ldp x2, x3, [sp, #X_SLOT(2)]
    ldp x4, x5, [sp, #X_SLOT(4)]
    ldp x6, x7, [sp, #X_SLOT(6)]
    ldp x8, x9, [sp, #X_SLOT(8)]
    ldp x10, x11, [sp, #X_SLOT(10)]
    ldp x12, x13, [sp, #X_SLOT(12)]
    ldp x14, x15, [sp, #X_SLOT(14)]
    ldp x16, x17, [sp, #X_SLOT(16)]
    ldr x19, [sp, #X_SLOT(19)]
    ldp x20, x21, [sp, #X_SLOT(20)]
    ldp x22, x23, [sp, #X_SLOT(22)]
    ldp x24, x25, [sp, #X_SLOT(24)]
    ldp x26, x27, [sp, #X_SLOT(26)]
    ldp x28, x29, [sp, #X_SLOT(28)]
    ldr x30, [sp, #X_SLOT(30)]
    add sp, sp, #ARMIGO_GATE_FRAME_SIZE
    // Host supervision may have opened a reservation. Model exception return
    // after every host frame has returned, including redirected guest resumes.
    clrex
    ret
