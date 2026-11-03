// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic guest routines for the guest-entry bridge, and one host harness
// that plants recognizable callee-saved state around the raw bridge. None of
// these routines writes x18, TPIDR_EL0 or TPIDRRO_EL0.

    .text
    .p2align 2

// Guest: report the entry state.
//   x0 = output block (host memory, 32 words), x1 = value
//   block: [0] sp, [8] x29, [16] x30, [24] fpcr, [32] fpsr, [40] x1,
//          [48..168] x2..x17, [176..248] x19..x28
//   returns x0 = entry sp, x1 = ~value
    .globl _armigo_test_guest_report
_armigo_test_guest_report:
    stp x2, x3, [x0, #48]
    stp x4, x5, [x0, #64]
    stp x6, x7, [x0, #80]
    stp x8, x9, [x0, #96]
    stp x10, x11, [x0, #112]
    stp x12, x13, [x0, #128]
    stp x14, x15, [x0, #144]
    stp x16, x17, [x0, #160]
    stp x19, x20, [x0, #176]
    stp x21, x22, [x0, #192]
    stp x23, x24, [x0, #208]
    stp x25, x26, [x0, #224]
    stp x27, x28, [x0, #240]
    mov x9, sp
    str x9, [x0, #0]
    stp x29, x30, [x0, #8]
    mrs x9, fpcr
    mrs x10, fpsr
    stp x9, x10, [x0, #24]
    str x1, [x0, #40]
    mov x0, sp
    mvn x1, x1
    ret

// Guest: deliberately clobber every register the bridge must not rely on,
// including callee-saved GPRs, x29, all of v0..v31, NZCV, FPCR and FPSR, then
// return with sp intact. Returns x0 = a + b, x1 = a ^ b.
    .globl _armigo_test_guest_clobber
_armigo_test_guest_clobber:
    add x9, x0, x1
    eor x10, x0, x1
    movz x11, #0xdead
    mov x2, x11
    mov x3, x11
    mov x4, x11
    mov x5, x11
    mov x6, x11
    mov x7, x11
    mov x8, x11
    mov x12, x11
    mov x13, x11
    mov x14, x11
    mov x15, x11
    mov x16, x11
    mov x17, x11
    mov x19, x11
    mov x20, x11
    mov x21, x11
    mov x22, x11
    mov x23, x11
    mov x24, x11
    mov x25, x11
    mov x26, x11
    mov x27, x11
    mov x28, x11
    mov x29, x11
    dup v0.2d, x11
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b
    mov v8.16b, v0.16b
    mov v9.16b, v0.16b
    mov v10.16b, v0.16b
    mov v11.16b, v0.16b
    mov v12.16b, v0.16b
    mov v13.16b, v0.16b
    mov v14.16b, v0.16b
    mov v15.16b, v0.16b
    mov v16.16b, v0.16b
    mov v17.16b, v0.16b
    mov v18.16b, v0.16b
    mov v19.16b, v0.16b
    mov v20.16b, v0.16b
    mov v21.16b, v0.16b
    mov v22.16b, v0.16b
    mov v23.16b, v0.16b
    mov v24.16b, v0.16b
    mov v25.16b, v0.16b
    mov v26.16b, v0.16b
    mov v27.16b, v0.16b
    mov v28.16b, v0.16b
    mov v29.16b, v0.16b
    mov v30.16b, v0.16b
    mov v31.16b, v0.16b
    // FPCR: DN | RMode = round toward zero. FPSR: IOC | DZC | QC.
    mov x11, #0x02c00000
    msr fpcr, x11
    movz x11, #0x0800, lsl #16
    movk x11, #0x0003
    msr fpsr, x11
    cmp x11, #1
    mov x0, x9
    mov x1, x10
    mov x9, #0xdead
    mov x10, x9
    mov x11, x9
    ret

// Original-process transfer veneer. The bridge necessarily branches through
// x16; this exact landing clears it before a direct branch to the real entry.
    .globl _armigo_test_guest_process_landing
_armigo_test_guest_process_landing:
    mov x16, xzr
    b _armigo_test_guest_process_entry

// Guest: snapshot original-process entry state into the block in x1, then
// terminate through the gate. The block holds x0..x30, sp, FPCR, FPSR, NZCV,
// one padding word, then all 32 128-bit vector registers.
    .globl _armigo_test_guest_process_entry
_armigo_test_guest_process_entry:
    stp x0, x1, [x1, #0]
    stp x2, x3, [x1, #16]
    stp x4, x5, [x1, #32]
    stp x6, x7, [x1, #48]
    stp x8, x9, [x1, #64]
    stp x10, x11, [x1, #80]
    stp x12, x13, [x1, #96]
    stp x14, x15, [x1, #112]
    stp x16, x17, [x1, #128]
    stp x18, x19, [x1, #144]
    stp x20, x21, [x1, #160]
    stp x22, x23, [x1, #176]
    stp x24, x25, [x1, #192]
    stp x26, x27, [x1, #208]
    stp x28, x29, [x1, #224]
    str x30, [x1, #240]
    mov x9, sp
    str x9, [x1, #248]
    mrs x9, fpcr
    str x9, [x1, #256]
    mrs x9, fpsr
    str x9, [x1, #264]
    mrs x9, nzcv
    str x9, [x1, #272]
    stp q0, q1, [x1, #288]
    stp q2, q3, [x1, #320]
    stp q4, q5, [x1, #352]
    stp q6, q7, [x1, #384]
    stp q8, q9, [x1, #416]
    stp q10, q11, [x1, #448]
    stp q12, q13, [x1, #480]
    stp q14, q15, [x1, #512]
    stp q16, q17, [x1, #544]
    stp q18, q19, [x1, #576]
    stp q20, q21, [x1, #608]
    stp q22, q23, [x1, #640]
    stp q24, q25, [x1, #672]
    stp q26, q27, [x1, #704]
    stp q28, q29, [x1, #736]
    stp q30, q31, [x1, #768]
    bl _armigo_execution_gate_entry
    brk #0

// Guest: enter the host gate once from a proper frame.
//   x0 = value; the gate handler sees x1 = sp at the gate site.
//   returns the x0 the handler left; x1 = sp at the gate site.
    .globl _armigo_test_guest_gate
_armigo_test_guest_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x1, sp
    bl _armigo_execution_gate_entry
    mov x1, sp
    ldp x29, x30, [sp], #16
    ret

// Guest: call a host function directly. x0 = function, x1 = argument.
// Returns the function's x0/x1.
    .globl _armigo_test_guest_call_host
_armigo_test_guest_call_host:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x16, x0
    mov x0, x1
    blr x16
    ldp x29, x30, [sp], #16
    ret

// Guest: return with sp 16 bytes lower than at entry.
    .globl _armigo_test_guest_bad_sp
_armigo_test_guest_bad_sp:
    sub sp, sp, #16
    ret

// Guest: overwrite the bridge's return record above the entry sp.
    .globl _armigo_test_guest_corrupt_record
_armigo_test_guest_corrupt_record:
    str xzr, [sp, #8]
    ret

// Guest: touch the stack downwards one 4 KiB step at a time, x0 bytes deep.
    .globl _armigo_test_guest_descend
_armigo_test_guest_descend:
    mov x9, sp
    sub x10, x9, x0
1:  sub x9, x9, #4096
    cmp x9, x10
    b.lo 2f
    str xzr, [x9]
    b 1b
2:  mov x0, #0
    ret

// Host harness:
// void armigo_test_bridge_harness(GuestEntryRecord* record, HarnessReport* report)
//   Plants patterns in x19..x28, d8..d15, FPCR and FPSR, calls the raw bridge,
//   and records what it sees afterwards:
//   report: [0..72] x19..x28, [80..136] d8..d15, [144] fpcr, [152] fpsr,
//           [160] sp before, [168] sp after, [176] x18 before, [184] x18 after,
//           [192] x29 before, [200] x29 after
//   The harness restores its caller's callee-saved state, FPCR and FPSR.
    .globl _armigo_test_bridge_harness
_armigo_test_bridge_harness:
    stp x29, x30, [sp, #-176]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mrs x9, fpcr
    mrs x10, fpsr
    stp x9, x10, [sp, #160]
    // The report pointer lives in the frame: no register survives by trust.
    sub sp, sp, #16
    str x1, [sp]

    movz x19, #0x1919
    movz x20, #0x2020
    movz x21, #0x2121
    movz x22, #0x2222
    movz x23, #0x2323
    movz x24, #0x2424
    movz x25, #0x2525
    movz x26, #0x2626
    movz x27, #0x2727
    movz x28, #0x2828
    movz x9, #0x0808
    fmov d8, x9
    movz x9, #0x0909
    fmov d9, x9
    movz x9, #0x1010
    fmov d10, x9
    movz x9, #0x1111
    fmov d11, x9
    movz x9, #0x1212
    fmov d12, x9
    movz x9, #0x1313
    fmov d13, x9
    movz x9, #0x1414
    fmov d14, x9
    movz x9, #0x1515
    fmov d15, x9
    // Host FPCR: FZ | RMode = round toward plus infinity. FPSR: IXC | OFC.
    mov x9, #0x01400000
    msr fpcr, x9
    mov x9, #0x14
    msr fpsr, x9

    mov x9, sp
    str x9, [x1, #160]
    mov x9, x18
    str x9, [x1, #176]
    str x29, [x1, #192]

    bl _armigo_execution_guest_entry_bridge

    mov x9, sp
    mov x10, x18
    mrs x11, fpcr
    mrs x12, fpsr
    mov x13, x29
    ldr x1, [sp]
    stp x19, x20, [x1, #0]
    stp x21, x22, [x1, #16]
    stp x23, x24, [x1, #32]
    stp x25, x26, [x1, #48]
    stp x27, x28, [x1, #64]
    stp d8, d9, [x1, #80]
    stp d10, d11, [x1, #96]
    stp d12, d13, [x1, #112]
    stp d14, d15, [x1, #128]
    stp x11, x12, [x1, #144]
    str x9, [x1, #168]
    str x10, [x1, #184]
    str x13, [x1, #200]

    add sp, sp, #16
    ldp x9, x10, [sp, #160]
    msr fpcr, x9
    msr fpsr, x10
    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
