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
    .p2align 2
    .globl _armigo_test_monitor_gate
    .globl _armigo_test_monitor_redirect
// uint64_t fixture(void* gate, uint64_t* word): no guest LDXR precedes STXR.
_armigo_test_monitor_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x16, x0
    mov x0, x1
    mov x1, #99
    blr x16
    stxr w2, x1, [x0]
    mov x0, x2
    ldp x29, x30, [sp], #16
    ret
_armigo_test_monitor_redirect:
    stxr w2, x1, [x0]
    mov x0, x2
    ldp x29, x30, [sp], #16
    ret

    .globl _armigo_test_monitor_gate_loop
// void fixture(void* gate, uint64_t count): identical warm gate workload.
_armigo_test_monitor_gate_loop:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, x0
    mov x20, x1
1:
    blr x19
    subs x20, x20, #1
    b.ne 1b
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

    .globl _armigo_test_monitor_signal
// The signal handler opens the reservation and releases this guest spin loop.
_armigo_test_monitor_signal:
    mov w3, #1
    stlr w3, [x1]
1:
    ldar w3, [x2]
    cbz w3, 1b
    mov x1, #99
    stxr w2, x1, [x0]
    mov x0, x2
    ret

    .globl _armigo_test_monitor_entry
// Seed the monitor immediately before the real entry bridge, with no C++
// validation or host calls between LDXR and the bridge under test.
_armigo_test_monitor_entry:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    ldxr x9, [x1]
    bl _armigo_execution_guest_entry_bridge
    ldp x29, x30, [sp], #16
    ret

    .globl _armigo_test_monitor_store
_armigo_test_monitor_store:
    stxr w2, x1, [x0]
    mov x0, x2
    ret
