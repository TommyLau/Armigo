// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// uint64_t fixture(void* gate, uint64_t input): the indirect call is only a
// test entry. Production sites use a branch to a gate placed in their module.
    .text
    .p2align 2
    .globl _armigo_test_local_gate
    .globl _armigo_test_local_gate_return
_armigo_test_local_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x16, x0
    mov x0, x1
    mov x2, sp
    mov x9, #123
    blr x16
_armigo_test_local_gate_return:
    // Verify x16 survived the template's own indirect dispatcher call.
    cmp x16, x3
    b.ne L_local_gate_failure
    cmp x9, #321
    b.ne L_local_gate_failure
    ldp x29, x30, [sp], #16
    ret
L_local_gate_failure:
    mov x0, #-1
    ldp x29, x30, [sp], #16
    ret
