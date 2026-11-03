// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Source bytes only. This function is copied and all seven system instructions
// are replaced before any execution of the copy.
    .text
    .p2align 2
    .globl _armigo_test_patch_begin
    .globl _armigo_test_patch_write_tls
    .globl _armigo_test_patch_read_tls
    .globl _armigo_test_patch_read_frame_register
    .globl _armigo_test_patch_read_tlr
    .globl _armigo_test_patch_read_frequency
    .globl _armigo_test_patch_read_ticks
    .globl _armigo_test_patch_svc
    .globl _armigo_test_patch_end
_armigo_test_patch_begin:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
_armigo_test_patch_write_tls:
    msr tpidr_el0, x0
_armigo_test_patch_read_tls:
    mrs x0, tpidr_el0
_armigo_test_patch_read_frame_register:
    mrs x29, tpidr_el0
    mov x5, x29
    mov x29, sp
_armigo_test_patch_read_tlr:
    mrs x1, tpidrro_el0
_armigo_test_patch_read_frequency:
    mrs x2, cntfrq_el0
_armigo_test_patch_read_ticks:
    mrs x3, cntpct_el0
    mov x4, sp
_armigo_test_patch_svc:
    svc #0x1e
    ldp x29, x30, [sp], #16
    ret
_armigo_test_patch_end:
