// ============================================================================
// Armigo NVN - AArch64 Direct-Call ABI Thunks
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#if defined(__APPLE__)
#define ARMIGO_C_SYMBOL_INNER(name) _##name
#define ARMIGO_C_SYMBOL(name) ARMIGO_C_SYMBOL_INNER(name)
#else
#define ARMIGO_C_SYMBOL(name) name
#endif

// Each fixed-size thunk derives its entry index from its address, atomically
// increments that entry's counter, and tail-branches to the current target.
// x16 and x17 are AAPCS64 intra-procedure-call temporaries; argument registers,
// SIMD registers, the link register, and the guest stack remain untouched.
    .arch_extension lse
    .text
    .p2align 6
    .globl ARMIGO_C_SYMBOL(armigo_nvn_count_thunks)
ARMIGO_C_SYMBOL(armigo_nvn_count_thunks):
    .rept 541
    adr x16, .
    adrp x17, ARMIGO_C_SYMBOL(armigo_nvn_count_thunks)@PAGE
    add x17, x17, ARMIGO_C_SYMBOL(armigo_nvn_count_thunks)@PAGEOFF
    sub x16, x16, x17
    lsr x16, x16, #6
    adrp x17, ARMIGO_C_SYMBOL(armigo_nvn_entry_slots)@PAGE
    add x17, x17, ARMIGO_C_SYMBOL(armigo_nvn_entry_slots)@PAGEOFF
    add x16, x17, x16, lsl #4
    mov x17, #1
    stadd x17, [x16]
    add x16, x16, #8
    ldar x17, [x16]
    br x17
    .p2align 6
    .endr

// Test support for making a call with raw guest register contents. The second
// argument points to x0-x7 values and the third to d0-d7 values; in particular,
// narrow inputs may retain arbitrary high bits exactly as they do at a guest
// BLR site.
    .p2align 2
    .globl ARMIGO_C_SYMBOL(armigo_nvn_guest_blr)
ARMIGO_C_SYMBOL(armigo_nvn_guest_blr):
    stp x29, x30, [sp, #-48]!
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]
    mov x29, sp
    mov x19, x0
    mov x20, x1
    mov x21, x2
    ldp d0, d1, [x21]
    ldp d2, d3, [x21, #16]
    ldp d4, d5, [x21, #32]
    ldp d6, d7, [x21, #48]
    ldp x0, x1, [x20]
    ldp x2, x3, [x20, #16]
    ldp x4, x5, [x20, #32]
    ldp x6, x7, [x20, #48]
    blr x19
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// A guest-style loop used by the synthetic call-cost test. Each iteration calls
// the loader-returned address with BLR and a null object argument.
    .p2align 2
    .globl ARMIGO_C_SYMBOL(armigo_nvn_guest_call_loop)
ARMIGO_C_SYMBOL(armigo_nvn_guest_call_loop):
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, x0
    mov x20, x1
    cbz x20, 2f
1:
    mov x0, xzr
    blr x19
    subs x20, x20, #1
    b.ne 1b
2:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// Resolve through a mapped bootstrap entry, then call GetInteger using guest
// registers. This exercises both indirect transfers without a host gate.
    .p2align 2
    .globl ARMIGO_C_SYMBOL(armigo_nvn_guest_resolve_integer)
ARMIGO_C_SYMBOL(armigo_nvn_guest_resolve_integer):
    stp x29, x30, [sp, #-48]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x29, sp
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x0, x20
    blr x19
    cbz x0, 3f
    mov x16, x0
    mov x0, xzr
    mov w1, w21
    mov x2, x22
    blr x16
    mov w0, #1
3:
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// Calls function(context) with SP at stack_top, then returns on the original
// stack. x19 keeps the original SP across the call; the frame record stays on
// the original stack so unwinding walks back into the caller.
    .p2align 2
    .globl ARMIGO_C_SYMBOL(armigo_nvn_call_on_stack)
ARMIGO_C_SYMBOL(armigo_nvn_call_on_stack):
    stp x29, x30, [sp, #-32]!
    str x19, [sp, #16]
    mov x29, sp
    mov x19, sp
    mov sp, x2
    blr x1
    mov sp, x19
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
