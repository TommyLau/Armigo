// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic process entry. The test places this code in
// an NSO and runs only its sealed offline-prepared copy.
    .text
    .p2align 2
    .globl _armigo_process_outcome_begin
    .globl _armigo_process_outcome_end
    .globl _armigo_process_outcome_main
    .globl _armigo_process_outcome_main_end
    .globl _armigo_process_outcome_heap

_armigo_process_outcome_begin:
_armigo_process_outcome_main:
    cbnz x0, 90f
    cbnz x1, 90f
_armigo_process_outcome_heap:
    movz x19, #0
    movk x19, #0, lsl #16
    movk x19, #0, lsl #32
    movk x19, #0, lsl #48
    mov x1, #0x200000
    svc #0x01
    cbnz x0, 90f
    cmp x1, x19
    b.ne 90f
    mov w9, #1
    stlr w9, [x19]

10:
    add x10, x19, #4
    ldar w9, [x10]
    cbz w9, 10b
    cmp w9, #2
    b.eq 30f
    cmp w9, #3
    b.eq 40f
    cmp w9, #1
    b.ne 90f

    // Copy the host-authored HIPC QueueBuffer packet into this thread's TLS.
    mrs x20, TPIDR_EL0
    add x10, x19, #1, lsl #12
    mov w11, #32
20:
    ldr x12, [x10], #8
    str x12, [x20], #8
    subs w11, w11, #1
    b.ne 20b
    ldr w0, [x19, #8]
    svc #0x21
    cbnz x0, 90f
    mov w9, #1
    add x10, x19, #12
    stlr w9, [x10]
    b 30f

40:
    adr x1, 50f
    svc #0x1f
    b 90f
50:
    .asciz "sm:"
    .p2align 2
30:
    svc #0x07
    b .
90:
    svc #0x40
    b .
_armigo_process_outcome_main_end:
_armigo_process_outcome_end:
