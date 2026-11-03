// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic register capture around a prepared x18 conditional branch.
// Both destinations return to the unchanged architectural link. The host wrapper
// owns its stack frame and saves its AAPCS64 callee-saved state before loading
// the test register image. Hardware x18 is never changed.
    .text
    .p2align 2
    .globl _armigo_test_conditional_x18_run
_armigo_test_conditional_x18_run:
    stp x29, x30, [sp, #-128]!
    adr x9, Lconditional_x18_return
    stp x9, x1, [sp, #16]
    stp x2, x0, [sp, #32]
    stp x19, x20, [sp, #48]
    stp x21, x22, [sp, #64]
    stp x23, x24, [sp, #80]
    stp x25, x26, [sp, #96]
    stp x27, x28, [sp, #112]
    str x18, [x1, #272]
    mov x9, sp
    str x9, [x1, #256]
    adr x9, Lconditional_x18_return
    str x9, [x1, #288]
    ldr x9, [x0, #248]
    msr nzcv, x9
    ldr q0, [x0, #304]
    ldr q31, [x0, #320]
    ldp x2, x3, [x0, #16]
    ldp x4, x5, [x0, #32]
    ldp x6, x7, [x0, #48]
    ldp x8, x9, [x0, #64]
    ldp x10, x11, [x0, #80]
    ldp x12, x13, [x0, #96]
    ldp x14, x15, [x0, #112]
    ldp x16, x17, [x0, #128]
    ldr x19, [x0, #152]
    ldp x20, x21, [x0, #160]
    ldp x22, x23, [x0, #176]
    ldp x24, x25, [x0, #192]
    ldp x26, x27, [x0, #208]
    ldp x28, x29, [x0, #224]
    ldr x1, [x0, #8]
    ldr x17, [sp, #32]
    ldr x30, [sp, #16]
    ldr x0, [sp, #24]
    br x17
Lconditional_x18_return:
    ldr x0, [sp, #24]
    mov x9, sp
    str x9, [x0, #264]
    str x18, [x0, #280]
    ldp x19, x20, [sp, #48]
    ldp x21, x22, [sp, #64]
    ldp x23, x24, [sp, #80]
    ldp x25, x26, [sp, #96]
    ldp x27, x28, [sp, #112]
    ldp x29, x30, [sp], #128
    ret

// Only this block is copied; the x18 instruction is replaced before execution.
    .p2align 3
    .globl _armigo_test_conditional_x18_begin
_armigo_test_conditional_x18_begin:
    cbz x18, Lconditional_taken
    str xzr, [sp, #40]
    b Lconditional_capture
    .globl _armigo_test_conditional_x18_taken
_armigo_test_conditional_x18_taken:
Lconditional_taken:
    str x30, [sp, #40]
Lconditional_capture:
    str x0, [x0]
    str x1, [x0, #8]
    stp x2, x3, [x0, #16]
    stp x4, x5, [x0, #32]
    stp x6, x7, [x0, #48]
    stp x8, x9, [x0, #64]
    stp x10, x11, [x0, #80]
    stp x12, x13, [x0, #96]
    stp x14, x15, [x0, #112]
    stp x16, x17, [x0, #128]
    str x19, [x0, #152]
    stp x20, x21, [x0, #160]
    stp x22, x23, [x0, #176]
    stp x24, x25, [x0, #192]
    stp x26, x27, [x0, #208]
    stp x28, x29, [x0, #224]
    str x30, [x0, #240]
    mrs x1, nzcv
    str x1, [x0, #248]
    str q0, [x0, #304]
    str q31, [x0, #320]
    mov x1, sp
    str x1, [x0, #336]
    ldr x1, [sp, #40]
    str x1, [x0, #296]
    ret
    .globl _armigo_test_conditional_x18_end
_armigo_test_conditional_x18_end:

// Assembler oracle for the fixed pair-save and pair-restore instructions.
    .globl _armigo_test_conditional_x18_stack_words
_armigo_test_conditional_x18_stack_words:
    stp x0, x1, [sp, #-16]!
    ldp x0, x1, [sp], #16
