// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Authored input containing SVCs. Only a fully patched copy is executed.
    .text
    .p2align 2
    .globl _armigo_test_exit_begin
    .globl _armigo_test_exit_end
    .globl _armigo_test_exit_normal
    .globl _armigo_test_exit_svc
    .globl _armigo_test_exit_nested_svc
    .globl _armigo_test_exit_thread
_armigo_test_exit_begin:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x9, #1
    str x9, [x0]
    bl L_exit_nested
    mov x9, #1
    str x9, [x0, #8]
    ldp x29, x30, [sp], #16
    ret
L_exit_nested:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    mov x19, #0xdead
    mov x20, #0xdead
    mov x21, #0xdead
    mov x22, #0xdead
    mov x23, #0xdead
    mov x24, #0xdead
    mov x25, #0xdead
    mov x26, #0xdead
    mov x27, #0xdead
    mov x28, #0xdead
    mov x29, #0xdead
    mov x9, #0
    msr fpcr, x9
    mov x9, #1
    msr fpsr, x9
    fmov d8, x9
    fmov d9, x9
    fmov d10, x9
    fmov d11, x9
    fmov d12, x9
    fmov d13, x9
    fmov d14, x9
    fmov d15, x9
_armigo_test_exit_svc:
    svc #0xa
    mov x9, #1
    str x9, [x0, #8]
    ldp x29, x30, [sp], #32
    ret
_armigo_test_exit_normal:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
_armigo_test_exit_nested_svc:
    svc #0x7f
    mov x0, #42
    mov x1, #43
    ldp x29, x30, [sp], #16
    ret
// Exact-SP thread entry. Observe the entry registers and all 64 bytes around
// SP before making any guest stack writes, then use the existing exit fixture.
// x0 points to twelve report words; x1 is an opaque initial argument.
_armigo_test_exit_thread:
    mov x9, sp
    stp x9, x29, [x0]
    stp x30, x1, [x0, #16]
    ldp x9, x10, [sp, #-32]
    stp x9, x10, [x0, #32]
    ldp x9, x10, [sp, #-16]
    stp x9, x10, [x0, #48]
    ldp x9, x10, [sp]
    stp x9, x10, [x0, #64]
    ldp x9, x10, [sp, #16]
    stp x9, x10, [x0, #80]
    b L_exit_nested
_armigo_test_exit_end:

// Linked host-callable gate used to check rejection when no entry exists or
// when abandoning this gate would skip an outer C++ callback.
    .globl _armigo_test_exit_direct_gate
    .globl _armigo_test_exit_direct_return
_armigo_test_exit_direct_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl _armigo_execution_gate_entry
_armigo_test_exit_direct_return:
    ldp x29, x30, [sp], #16
    ret

// Host ABI probe: call x0(x1), capture host callee-saved/FP state into x2.
// This calls C++ invoke_guest_entry through a callback, so the real entry
// registration and cleanup are exercised as well as the raw bridge.
    .globl _armigo_test_exit_host_probe
_armigo_test_exit_host_probe:
    stp x29, x30, [sp, #-192]!
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
    mrs x10, fpcr
    mrs x11, fpsr
    stp x10, x11, [sp, #160]
    str x2, [sp, #176]
    mov x9, x0
    mov x0, x1
    mov x19, #19
    mov x20, #20
    mov x21, #21
    mov x22, #22
    mov x23, #23
    mov x24, #24
    mov x25, #25
    mov x26, #26
    mov x27, #27
    mov x28, #28
    mov x10, #8
    fmov d8, x10
    mov x10, #9
    fmov d9, x10
    mov x10, #10
    fmov d10, x10
    mov x10, #11
    fmov d11, x10
    mov x10, #12
    fmov d12, x10
    mov x10, #13
    fmov d13, x10
    mov x10, #14
    fmov d14, x10
    mov x10, #15
    fmov d15, x10
    mov x10, #0x400000
    msr fpcr, x10
    mov x10, #0x11
    msr fpsr, x10
    blr x9
    ldr x9, [sp, #176]
    stp x19, x20, [x9, #0]
    stp x21, x22, [x9, #16]
    stp x23, x24, [x9, #32]
    stp x25, x26, [x9, #48]
    stp x27, x28, [x9, #64]
    stp d8, d9, [x9, #80]
    stp d10, d11, [x9, #96]
    stp d12, d13, [x9, #112]
    stp d14, d15, [x9, #128]
    mov x10, sp
    cmp x29, x10
    cset x10, eq
    str x10, [x9, #144]
    mrs x10, fpcr
    mrs x11, fpsr
    stp x10, x11, [x9, #152]
    ldp x10, x11, [sp, #160]
    msr fpcr, x10
    msr fpsr, x11
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #192
    ret
