// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic prepared-code fixtures for guest gate-site dispatch.
//
// _armigo_test_guest_sites is a block of 16 gate sites. Site i occupies 8
// bytes at _armigo_test_guest_sites + 8 * i: a direct `bl` into the host gate
// followed by a branch to the common capture code, so its gate return address
// is _armigo_test_guest_sites + 8 * i + 4.
//
// armigo_test_guest_run_site(const GuestImage* in, GuestImage* out,
//                            uint64_t site)
// loads x0..x17, x19..x29 and NZCV from the input image, branches to the site
// address with x30 as the only scratch register, and after the gate returns
// stores x0..x17, x19..x30, NZCV and host facts (hardware x18 and sp before
// and after) into the output image. It is an AAPCS64 function and never
// writes the Apple platform register x18.
//
// armigo_test_guest_sequence(uint64_t value, uint64_t* out) is straight-line
// prepared code whose six gate sites stand for, in order:
//     msr tpidr_el0, x1        (x1 = value)
//     mrs x2, tpidr_el0
//     mrs x3, cntvct_el0
//     mrs x4, cntpct_el0
//     mrs x5, cntfrq_el0
//     svc #imm                 (x0 = 7 on entry)
// and then stores x2, x3, x4, x5 and x0 to out[0..4]. Each site's return
// address is exported as _armigo_test_guest_sequence_return_<n>.

#define IMG_X(n) ((n) * 8)
#define IMG_NZCV 248
#define IMG_X18_BEFORE 256
#define IMG_X18_AFTER 264
#define IMG_SP_BEFORE 272
#define IMG_SP_AFTER 280

#define FRAME_SIZE 128
#define FRAME_X19 16
#define FRAME_IN 96
#define FRAME_OUT 104
#define FRAME_SITE 112
#define FRAME_X0_TEMP 120

    .text
    .globl _armigo_test_guest_run_site
    .p2align 2
_armigo_test_guest_run_site:
    stp x29, x30, [sp, #-FRAME_SIZE]!
    mov x29, sp
    stp x19, x20, [sp, #(FRAME_X19 + 0)]
    stp x21, x22, [sp, #(FRAME_X19 + 16)]
    stp x23, x24, [sp, #(FRAME_X19 + 32)]
    stp x25, x26, [sp, #(FRAME_X19 + 48)]
    stp x27, x28, [sp, #(FRAME_X19 + 64)]
    stp x0, x1, [sp, #FRAME_IN]
    str x2, [sp, #FRAME_SITE]

    // Host facts immediately before the site.
    str x18, [x1, #IMG_X18_BEFORE]
    mov x9, sp
    str x9, [x1, #IMG_SP_BEFORE]

    ldr x9, [x0, #IMG_NZCV]
    msr nzcv, x9

    // x0 last because it is the base; x18 never; x30 carries the site.
    ldp x2, x3, [x0, #IMG_X(2)]
    ldp x4, x5, [x0, #IMG_X(4)]
    ldp x6, x7, [x0, #IMG_X(6)]
    ldp x8, x9, [x0, #IMG_X(8)]
    ldp x10, x11, [x0, #IMG_X(10)]
    ldp x12, x13, [x0, #IMG_X(12)]
    ldp x14, x15, [x0, #IMG_X(14)]
    ldp x16, x17, [x0, #IMG_X(16)]
    ldr x19, [x0, #IMG_X(19)]
    ldp x20, x21, [x0, #IMG_X(20)]
    ldp x22, x23, [x0, #IMG_X(22)]
    ldp x24, x25, [x0, #IMG_X(24)]
    ldp x26, x27, [x0, #IMG_X(26)]
    ldp x28, x29, [x0, #IMG_X(28)]
    ldr x30, [sp, #FRAME_SITE]
    ldr x1, [x0, #IMG_X(1)]
    ldr x0, [x0, #IMG_X(0)]
    br x30

    .p2align 3
    .globl _armigo_test_guest_sites
_armigo_test_guest_sites:
    .rept 16
    bl _armigo_execution_gate_entry
    b Lguest_capture
    .endr

Lguest_capture:
    // Stores, loads and `mov` leave NZCV untouched until it is read below.
    str x0, [sp, #FRAME_X0_TEMP]
    ldr x0, [sp, #FRAME_OUT]
    str x1, [x0, #IMG_X(1)]
    stp x2, x3, [x0, #IMG_X(2)]
    stp x4, x5, [x0, #IMG_X(4)]
    stp x6, x7, [x0, #IMG_X(6)]
    stp x8, x9, [x0, #IMG_X(8)]
    stp x10, x11, [x0, #IMG_X(10)]
    stp x12, x13, [x0, #IMG_X(12)]
    stp x14, x15, [x0, #IMG_X(14)]
    stp x16, x17, [x0, #IMG_X(16)]
    str x19, [x0, #IMG_X(19)]
    stp x20, x21, [x0, #IMG_X(20)]
    stp x22, x23, [x0, #IMG_X(22)]
    stp x24, x25, [x0, #IMG_X(24)]
    stp x26, x27, [x0, #IMG_X(26)]
    stp x28, x29, [x0, #IMG_X(28)]
    str x30, [x0, #IMG_X(30)]
    str x18, [x0, #IMG_X18_AFTER]
    mrs x9, nzcv
    str x9, [x0, #IMG_NZCV]
    mov x9, sp
    str x9, [x0, #IMG_SP_AFTER]
    ldr x9, [sp, #FRAME_X0_TEMP]
    str x9, [x0, #IMG_X(0)]

    ldp x19, x20, [sp, #(FRAME_X19 + 0)]
    ldp x21, x22, [sp, #(FRAME_X19 + 16)]
    ldp x23, x24, [sp, #(FRAME_X19 + 32)]
    ldp x25, x26, [sp, #(FRAME_X19 + 48)]
    ldp x27, x28, [sp, #(FRAME_X19 + 64)]
    ldp x29, x30, [sp], #FRAME_SIZE
    ret

    .globl _armigo_test_guest_sequence
    .p2align 2
_armigo_test_guest_sequence:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x1
    mov x1, x0
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_0
_armigo_test_guest_sequence_return_0:
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_1
_armigo_test_guest_sequence_return_1:
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_2
_armigo_test_guest_sequence_return_2:
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_3
_armigo_test_guest_sequence_return_3:
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_4
_armigo_test_guest_sequence_return_4:
    mov x0, #7
    bl _armigo_execution_gate_entry
    .globl _armigo_test_guest_sequence_return_5
_armigo_test_guest_sequence_return_5:
    stp x2, x3, [x19, #0]
    stp x4, x5, [x19, #16]
    str x0, [x19, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
