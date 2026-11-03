// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Test-only host continuation around a four-KiB logical guest stack.
// Record: guest top +0, gate +8, saved host SP +16, exited frame +24.
    .text
    .p2align 2
    .globl _armigo_test_callback_stack_enter
_armigo_test_callback_stack_enter:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, x0
    mov x20, sp
    str x20, [x19, #16]
    ldr x9, [x19]
    ldr x16, [x19, #8]
    mov sp, x9
    mov x0, #7
    mov x1, #11
    blr x16
    mov sp, x20
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

    .globl _armigo_test_callback_stack_exit
_armigo_test_callback_stack_exit:
    str x1, [x0, #24]
    ldr x9, [x0, #16]
    mov sp, x9
    mov x0, #99
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

    .globl _armigo_test_callback_stack_reenter
_armigo_test_callback_stack_reenter:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl _armigo_execution_gate_entry
    ldp x29, x30, [sp], #16
    ret
