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
    .globl _armigo_test_import_begin
    .globl _armigo_test_import_end
_armigo_test_import_begin:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    // The test places this function at image offset 0x100 and the import slot
    // at 0x8000. ADR remains correct when the entire module is relocated.
    adr x16, _armigo_test_import_begin + 0x7f00
    ldr x16, [x16]
    blr x16
    add w0, w0, #5
    ldp x29, x30, [sp], #16
    ret
_armigo_test_import_end:

    .p2align 2
    .globl _armigo_test_provider_begin
    .globl _armigo_test_provider_end
_armigo_test_provider_begin:
    mov w0, #37
    ret
_armigo_test_provider_end:

    .p2align 2
    .globl _armigo_test_prepared_begin
    .globl _armigo_test_prepared_end
_armigo_test_prepared_begin:
    mov w0, #72
    ret
_armigo_test_prepared_end:
