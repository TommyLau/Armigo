// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Authored input for the reachable-code collector. It is only read as data
// and must never be executed: it contains a supervisor call and unresolved
// indirect transfers. Regions A and B model two linked modules; A reaches B
// directly and through a linker-relocated GOT thunk. Region C holds real
// assembler encodings of an indirect call and branch.
    .text
    .p2align 2
    .globl _armigo_test_reachable_a_begin
    .globl _armigo_test_reachable_a_end
    .globl _armigo_test_reachable_b_begin
    .globl _armigo_test_reachable_b_function
    .globl _armigo_test_reachable_b_target
    .globl _armigo_test_reachable_b_end
    .globl _armigo_test_reachable_c_begin
    .globl _armigo_test_reachable_c_end
    .globl _armigo_test_reachable_got

_armigo_test_reachable_a_begin:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
1:
    subs x0, x0, #1
    b.ne 1b
    cbz x1, 2f
    bl _armigo_test_reachable_b_function
2:
    bl Lthunk
    ldp x29, x30, [sp], #16
    ret
Lthunk:
    adrp x16, _armigo_test_reachable_got@PAGE
    ldr x17, [x16, _armigo_test_reachable_got@PAGEOFF]
    add x16, x16, _armigo_test_reachable_got@PAGEOFF
    br x17
_armigo_test_reachable_a_end:

_armigo_test_reachable_b_begin:
_armigo_test_reachable_b_function:
    tbz w0, #0, 3f
    svc #0x26
3:
    ret
_armigo_test_reachable_b_target:
    cmp x0, #4
    b.hi 4f
    mov x0, #0
4:
    ret
_armigo_test_reachable_b_end:

_armigo_test_reachable_c_begin:
    adrp x8, _armigo_test_reachable_got@PAGE
    ldr x9, [x8, _armigo_test_reachable_got@PAGEOFF]
    blr x9
    br x9
_armigo_test_reachable_c_end:

    .data
    .p2align 3
_armigo_test_reachable_got:
    .quad _armigo_test_reachable_b_target
    .quad 0
