// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Authored native fixture: publish three values through the probed thread
// slot, call one prepared SVC site, and read the same cell after the callback.
// void armigo_test_cell_round_trip(uint64_t offset, uint64_t seed, uint64_t* out)
// out[0] is the SVC result; out[1..3] are x18, TPIDR and TPIDRRO guest values.
// Hardware x18 and both hardware thread-pointer registers are never written.
    .text
    .p2align 2
    .globl _armigo_test_cell_round_trip
_armigo_test_cell_round_trip:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    mov x20, x2
    mrs x9, TPIDRRO_EL0
    ldr x9, [x9, x19]
    str x1, [x9]
    add x1, x1, #1
    str x1, [x9, #8]
    add x1, x1, #1
    str x1, [x9, #16]
    bl _armigo_execution_gate_entry
    .globl _armigo_test_cell_gate_return
_armigo_test_cell_gate_return:
    str x0, [x20]
    mrs x9, TPIDRRO_EL0
    ldr x9, [x9, x19]
    ldp x10, x11, [x9]
    stp x10, x11, [x20, #8]
    ldr x10, [x9, #16]
    str x10, [x20, #24]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// Two-argument entry wrapper for the guarded-stack bridge.
// x0 = native slot offset; x1 points to {seed, output buffer pointer}.
    .globl _armigo_test_cell_guarded_entry
_armigo_test_cell_guarded_entry:
    ldr x2, [x1, #8]
    ldr x1, [x1]
    b _armigo_test_cell_round_trip
