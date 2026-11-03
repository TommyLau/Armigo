// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

    .text
    .p2align 3
    .globl _armigo_test_call_hook_begin
_armigo_test_call_hook_begin:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x1, [sp, #16]
    .globl _armigo_test_call_hook_site
_armigo_test_call_hook_site:
    bl _armigo_test_call_hook_target
    ldr x1, [sp, #16]
    mov x9, #2
    str x9, [x1, #24]
    ldp x29, x30, [sp], #32
    ret

    .globl _armigo_test_call_hook_target
_armigo_test_call_hook_target:
    str x30, [x1]
    mov x9, sp
    str x9, [x1, #8]
    add x0, x0, #7
    ret
    .globl _armigo_test_call_hook_end
_armigo_test_call_hook_end:

    .globl _armigo_test_call_hook_stack_words
_armigo_test_call_hook_stack_words:
    str x30, [sp, #-16]!
    ldr x30, [sp], #16
