// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored instructions encoded by the system assembler and
// resolved by the linker. The code is never executed; the tests read the words
// and compare decoded fields and referenced addresses with the address table,
// whose entries the assembler and linker compute from the same labels.
//
// Entry order is mirrored by the expectation arrays in
// assembler_fixture_tests.cpp.

    // Armv8.6-A includes FEAT_ECV (CNTVCTSS_EL0); BC.cond needs FEAT_HBC.
    .arch armv8.6-a
    .arch_extension hbc

    .section __TEXT,__text,regular,pure_instructions
    .p2align 4
    .globl _armigo_aarch64_fixture_code
_armigo_aarch64_fixture_code:
Lback:
    nop
L0:  b      Lforward
L1:  bl     Lback
L2:  b.hi   Lback
L3:  bc.eq  Lforward
L4:  cbz    x18, Lforward
L5:  cbnz   w3, Lback
L6:  tbz    w7, #31, Lforward
L7:  tbnz   x18, #63, Lback
L8:  adr    x18, Lforward + 3
L9:  adr    x1, Lback
L10: adrp   x2, _armigo_aarch64_fixture_data@PAGE
L11: ldr    w1, Lliteral
L12: ldr    x2, Lliteral
L13: ldrsw  x18, Lliteral
L14: prfm   pldl1keep, Lliteral
L15: ldr    s4, Lliteral
L16: ldr    d5, Lliteral
L17: ldr    q6, Lliteral
L18: br     x18
L19: blr    x9
L20: ret
L21: svc    #0x7f
L22: brk    #0x18
L23: mrs    x18, tpidr_el0
L24: msr    tpidr_el0, x1
L25: mrs    x2, tpidrro_el0
L26: mrs    x3, cntpct_el0
L27: mrs    x4, cntvct_el0
L28: mrs    x5, cntfrq_el0
L29: mrs    x6, cntvctss_el0
L30: mrs    x7, nzcv
L31: msr    fpcr, x8
L32: mrs    x9, ctr_el0

// Maximum and minimum displacements, written as label expressions so the
// assembler, not the test, encodes the endpoint.
E0:  b      E0 + 0x7fffffc
E1:  bl     E1 - 0x8000000
E2:  b.ne   E2 + 0xffffc
E3:  cbz    w0, E3 - 0x100000
E4:  tbnz   x1, #40, E4 + 0x7ffc
E5:  tbz    w2, #1, E5 - 0x8000
E6:  adr    x3, E6 + 0xfffff
E7:  adr    x4, E7 - 0x100000
E8:  ldr    x5, E8 + 0xffffc
E9:  ldr    q6, E9 - 0x100000

    .p2align 4
Lforward:
    nop
    nop
    nop
    nop
Lliteral:
    .quad 0x0123456789abcdef
    .quad 0xfedcba9876543210

    .section __DATA,__const
    .p2align 3
    .globl _armigo_aarch64_fixture_table
_armigo_aarch64_fixture_table:
    .quad L0, Lforward
    .quad L1, Lback
    .quad L2, Lback
    .quad L3, Lforward
    .quad L4, Lforward
    .quad L5, Lback
    .quad L6, Lforward
    .quad L7, Lback
    .quad L8, Lforward + 3
    .quad L9, Lback
    .quad L10, _armigo_aarch64_fixture_data
    .quad L11, Lliteral
    .quad L12, Lliteral
    .quad L13, Lliteral
    .quad L14, Lliteral
    .quad L15, Lliteral
    .quad L16, Lliteral
    .quad L17, Lliteral
    .quad L18, 0
    .quad L19, 0
    .quad L20, 0
    .quad L21, 0
    .quad L22, 0
    .quad L23, 0
    .quad L24, 0
    .quad L25, 0
    .quad L26, 0
    .quad L27, 0
    .quad L28, 0
    .quad L29, 0
    .quad L30, 0
    .quad L31, 0
    .quad L32, 0
    .globl _armigo_aarch64_fixture_table_end
_armigo_aarch64_fixture_table_end:

    .globl _armigo_aarch64_fixture_endpoints
_armigo_aarch64_fixture_endpoints:
    .quad E0
    .quad E1
    .quad E2
    .quad E3
    .quad E4
    .quad E5
    .quad E6
    .quad E7
    .quad E8
    .quad E9
    .globl _armigo_aarch64_fixture_endpoints_end
_armigo_aarch64_fixture_endpoints_end:

    .section __DATA,__data
    .p2align 12
    .space 0x18  // keep the low 12 bits nonzero so ADRP's page truncation is observable
    .globl _armigo_aarch64_fixture_data
_armigo_aarch64_fixture_data:
    .quad 0
