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
    .globl _armigo_test_mapping_code_begin
    .globl _armigo_test_mapping_code_end
_armigo_test_mapping_code_begin:
    mov w0, #42
    ret
_armigo_test_mapping_code_end:
