// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// This authored input must never run directly: it uses Apple's reserved
// register. Tests copy it, rewrite every marked instruction, seal the copy,
// and only then execute that copy with a bound guest register cell.
    .text
    .p2align 2
    .globl _armigo_test_scalar_x18_begin
    .globl _armigo_test_scalar_x18_end
_armigo_test_scalar_x18_begin:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x2, #0x1234
    mov x3, #0x5678
Lsite0:
    mov x18, x1
    cmp x1, x1
Lsite1:
    csinc x18, x18, x1, ne
Lsite2:
    ubfiz x18, x18, #1, #31
Lsite3:
    str x18, [x0]
Lsite4:
    str x1, [x0, x18]
Lsite5:
    ldr x18, [x0, #8]
Lsite6:
    movk x18, #0xface, lsl #48
Lsite7:
    adds w18, w18, #1
Lsite8:
    str x18, [x0, #16]
    stp x2, x3, [x0, #24]
    mrs x4, NZCV
    str x4, [x0, #40]
Lsite9:
    mov x0, x18
    ldp x29, x30, [sp], #16
    ret
_armigo_test_scalar_x18_end:

    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_scalar_x18_offsets
_armigo_test_scalar_x18_offsets:
    .long Lsite0 - _armigo_test_scalar_x18_begin
    .long Lsite1 - _armigo_test_scalar_x18_begin
    .long Lsite2 - _armigo_test_scalar_x18_begin
    .long Lsite3 - _armigo_test_scalar_x18_begin
    .long Lsite4 - _armigo_test_scalar_x18_begin
    .long Lsite5 - _armigo_test_scalar_x18_begin
    .long Lsite6 - _armigo_test_scalar_x18_begin
    .long Lsite7 - _armigo_test_scalar_x18_begin
    .long Lsite8 - _armigo_test_scalar_x18_begin
    .long Lsite9 - _armigo_test_scalar_x18_begin

    .text
    .p2align 2
    .globl _armigo_test_adrp_x18_begin
    .globl _armigo_test_adrp_x18_end
_armigo_test_adrp_x18_begin:
    mov x1, #0x3456
    mov x2, sp
    mov x3, x30
    mov x4, x18
    cmp x1, x1
Ladrp_site:
    .long 0x90000012 // ADRP X18, own page; replaced before any execution.
    str x0, [x0]
    stp x1, x2, [x0, #8]
    mov x5, sp
    stp x3, x5, [x0, #24]
    mov x5, x30
    stp x4, x5, [x0, #40]
    mov x5, x18
    mrs x6, NZCV
    stp x5, x6, [x0, #56]
    ret
_armigo_test_adrp_x18_end:

    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_adrp_x18_offset
_armigo_test_adrp_x18_offset:
    .long Ladrp_site - _armigo_test_adrp_x18_begin

    .text
    .p2align 2
    .globl _armigo_test_sp_x18_begin
    .globl _armigo_test_sp_x18_end
_armigo_test_sp_x18_begin:
    sub sp, sp, #96
    stp x29, x30, [sp, #80]
    mov x29, sp
    mov x2, sp
    mov x3, x18
    mov x4, x30
    mov x5, #0x7080
    movk x5, #0x5060, lsl #16
    movk x5, #0x3040, lsl #32
    movk x5, #0x1020, lsl #48
    str x5, [sp, #16]
    str x1, [sp, #40]
    cmp x1, x1
Lsp_site0:
    ldr x18, [sp, #40]
Lsp_site1:
    str w18, [sp, #20]
Lsp_site2:
    str x18, [sp, #24]
Lsp_site3:
    ldr w18, [sp, #40]
Lsp_site4:
    str x18, [sp]
Lsp_site5:
    ldr x18, [sp]
    stp x0, x1, [x0]
    ldp x5, x6, [sp, #16]
    stp x5, x6, [x0, #16]
    ldr x5, [sp]
    mrs x6, NZCV
    stp x5, x6, [x0, #32]
    mov x5, sp
    stp x2, x5, [x0, #48]
    mov x5, x18
    stp x3, x5, [x0, #64]
    mov x5, x30
    stp x4, x5, [x0, #80]
    ldp x29, x30, [sp, #80]
    add sp, sp, #96
    ret
_armigo_test_sp_x18_end:

    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_sp_x18_offsets
_armigo_test_sp_x18_offsets:
    .long Lsp_site0 - _armigo_test_sp_x18_begin
    .long Lsp_site1 - _armigo_test_sp_x18_begin
    .long Lsp_site2 - _armigo_test_sp_x18_begin
    .long Lsp_site3 - _armigo_test_sp_x18_begin
    .long Lsp_site4 - _armigo_test_sp_x18_begin
    .long Lsp_site5 - _armigo_test_sp_x18_begin

// Stack accesses whose address arithmetic must be re-derived because the
// trampoline spill already moved SP down by 16 bytes. The frame below is
// zeroed so every loaded value is deterministic.
    .text
    .p2align 2
    .globl _armigo_test_sp_pair_x18_begin
    .globl _armigo_test_sp_pair_x18_end
_armigo_test_sp_pair_x18_begin:
    sub sp, sp, #128
    mov x2, sp
    stp xzr, xzr, [sp, #16]
    stp xzr, xzr, [sp, #32]
    stp xzr, xzr, [sp, #48]
    stp xzr, xzr, [sp, #64]
    stp xzr, xzr, [sp, #80]
    stp xzr, xzr, [sp, #96]
    stp x29, x30, [sp, #112]
    mov x29, sp
    mov x3, x18
    mov x4, x30
    mov x5, #0x1234
    movk x5, #0x5678, lsl #16
    movk x5, #0x9abc, lsl #32
    movk x5, #0xdef0, lsl #48
    str x5, [sp, #24]
    cmp x1, x1
Lpair_site0:
    ldp x18, x9, [sp, #24]
Lpair_site1:
    stur x18, [sp, #40]
Lpair_site2:
    stp x18, x9, [sp, #-32]!
Lpair_site3:
    ldp x18, x10, [sp], #32
Lpair_site4:
    strb w18, [sp, #48]
Lpair_site5:
    ldrh w18, [sp, #48]
Lpair_site6:
    stp x18, x9, [sp, #56]
Lpair_site7:
    ldp x11, x18, [sp, #56]
    ldp x12, x13, [sp, #56]
    stp x12, x13, [x0, #0]
    ldr x12, [sp, #48]
    str x12, [x0, #16]
    str x18, [x0, #24]
    mov x5, sp
    str x5, [x0, #32]
    str x2, [x0, #40]
    mrs x6, NZCV
    str x6, [x0, #48]
    mov x5, x30
    str x5, [x0, #56]
    str x3, [x0, #64]
    str x4, [x0, #72]
    ldp x29, x30, [sp, #112]
    add sp, sp, #128
    ret
_armigo_test_sp_pair_x18_end:

// SP pairs at the top of imm7 that the 16-byte spill pushes out of range: the
// trampoline transfers each register on its own. x0 receives the values read
// back, the old SP and the SP after the sites; guest x18 ends sign-extended.
    .globl _armigo_test_sp_far_pair_x18_begin
    .globl _armigo_test_sp_far_pair_x18_end
_armigo_test_sp_far_pair_x18_begin:
    sub sp, sp, #0x220
    stp x29, x30, [sp]
    mov x16, sp
    mov x5, #0x0002
    movk x5, #0x8000, lsl #16
    movk x5, #0x0001, lsl #32
    movk x5, #0x8000, lsl #48
    mov w9, #0xfffe
    movk w9, #0x7fff, lsl #16
Lfar_site0:
    stp x5, x18, [sp, #0x1f0]
    ldp x6, x7, [sp, #0x1f0]
Lfar_site1:
    ldp x18, x11, [sp, #0x1f0]
Lfar_site2:
    stp w18, w9, [sp, #0xf0]
    ldp w12, w13, [sp, #0xf0]
Lfar_site3:
    ldpsw x14, x18, [sp, #0xf0]
    stp x6, x7, [x0, #0]
    stp x11, x12, [x0, #16]
    stp x13, x14, [x0, #32]
    mov x15, sp
    stp x16, x15, [x0, #48]
    ldp x29, x30, [sp]
    add sp, sp, #0x220
    ret
_armigo_test_sp_far_pair_x18_end:

    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_sp_pair_x18_offsets
_armigo_test_sp_pair_x18_offsets:
    .long Lpair_site0 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site1 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site2 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site3 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site4 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site5 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site6 - _armigo_test_sp_pair_x18_begin
    .long Lpair_site7 - _armigo_test_sp_pair_x18_begin
    .globl _armigo_test_sp_far_pair_x18_offsets
_armigo_test_sp_far_pair_x18_offsets:
    .long Lfar_site0 - _armigo_test_sp_far_pair_x18_begin
    .long Lfar_site1 - _armigo_test_sp_far_pair_x18_begin
    .long Lfar_site2 - _armigo_test_sp_far_pair_x18_begin
    .long Lfar_site3 - _armigo_test_sp_far_pair_x18_begin

// ADR carries a byte displacement from its own address, so preparation has to
// move the address calculation into the trampoline together with the result.
    .text
    .p2align 2
    .globl _armigo_test_adr_x18_begin
    .globl _armigo_test_adr_x18_end
_armigo_test_adr_x18_begin:
Ladr_read:
    mov x4, x18
    mov x3, x30
    cmp x1, x1
Ladr_site:
    adr x18, Ladr_target
Ladr_store:
    str x18, [x0]
    str x4, [x0, #8]
    mov x5, x30
    str x5, [x0, #16]
    mrs x6, NZCV
    str x6, [x0, #24]
    str x3, [x0, #32]
    ret
Ladr_target:
    nop
_armigo_test_adr_x18_end:

    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_adr_x18_offsets
_armigo_test_adr_x18_offsets:
    .long Ladr_read - _armigo_test_adr_x18_begin
    .long Ladr_site - _armigo_test_adr_x18_begin
    .long Ladr_store - _armigo_test_adr_x18_begin
    .globl _armigo_test_adr_x18_target
_armigo_test_adr_x18_target:
    .long Ladr_target - _armigo_test_adr_x18_begin
