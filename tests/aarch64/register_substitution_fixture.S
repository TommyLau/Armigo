// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently written substitution expectations encoded by the system
// assembler from mnemonics.
//
// The first table is never executed. Each entry is two words: an original
// instruction and the instruction expected after renaming the register given
// in register_substitution_fixture_tests.cpp (usually x18 to x9). Refused
// entries repeat the original. Entry order is mirrored by the expectation
// table; the numbers are entry indices.
//
// The harnesses below execute natively. Each case exists twice: once naming
// x10 and once with x10 renamed to x12. Only x9..x15 and v10 are used; x18 is
// the Apple platform register and is never touched.

    .arch armv8.3-a
    .arch_extension lse

    .section __TEXT,__text,regular,pure_instructions
    .p2align 2
    .globl _armigo_substitution_pairs
_armigo_substitution_pairs:
    // Destination, source and read-modify-write operands, W and X views.
    add     x18, x1, x2                     // 0
    add     x9, x1, x2
    add     w1, w18, w2                     // 1
    add     w1, w9, w2
    add     x18, x18, #1                    // 2
    add     x9, x9, #1
    movk    w18, #0x12, lsl #16             // 3
    movk    w9, #0x12, lsl #16
    eor     x18, x18, x18                   // 4
    eor     x9, x9, x9
    // Immediates and prefetch operations whose bits equal 18 in a field
    // position: nothing to rename.
    add     x1, x2, #18                     // 5
    add     x1, x2, #18
    movz    x1, #0x12                       // 6
    movz    x1, #0x12
    ccmp    x1, #18, #0, eq                 // 7
    ccmp    x1, #18, #0, eq
    ubfx    x1, x2, #18, #4                 // 8
    ubfx    x1, x2, #18, #4
    prfm    pstl2keep, [x1]                 // 9
    prfm    pstl2keep, [x1]
    tbz     x1, #18, .+8                    // 10
    tbz     x1, #18, .+8
    lsr     w18, w1, #18                    // 11
    lsr     w9, w1, #18
    // Aliases and register 31.
    mov     x18, sp                         // 12
    mov     x9, sp
    mov     sp, x18                         // 13
    mov     sp, x9
    mov     w18, wzr                        // 14
    mov     w9, wzr
    cmp     x18, #0                         // 15
    cmp     x9, #0
    csel    x18, x1, x18, ne                // 16
    csel    x9, x1, x9, ne
    mul     x1, x18, x18                    // 17
    mul     x1, x9, x9
    add     x18, sp, w1, uxtw               // 18
    add     x9, sp, w1, uxtw
    // Memory: writeback, index, pairs, exclusive status.
    ldr     x1, [x18, #8]!                  // 19
    ldr     x1, [x9, #8]!
    ldr     x1, [x18, #8]!                  // 20 to x1: transfer collision
    ldr     x1, [x18, #8]!
    str     x18, [x1], #16                  // 21
    str     x9, [x1], #16
    ldp     x18, x1, [sp, #16]              // 22
    ldp     x9, x1, [sp, #16]
    ldp     x18, x1, [sp]                   // 23 to x1: pair collision
    ldp     x18, x1, [sp]
    stp     x18, x18, [sp, #-16]!           // 24
    stp     x9, x9, [sp, #-16]!
    ldpsw   x1, x2, [x18], #8               // 25 to x1: writeback collision
    ldpsw   x1, x2, [x18], #8
    ldr     x1, [x2, x18, lsl #3]           // 26
    ldr     x1, [x2, x9, lsl #3]
    ldr     w1, [x2, w18, sxtw #2]          // 27
    ldr     w1, [x2, w9, sxtw #2]
    ldr     x18, [x18]                      // 28
    ldr     x9, [x9]
    ldxr    x1, [x18]                       // 29
    ldxr    x1, [x9]
    stxr    w18, x1, [x2]                   // 30
    stxr    w9, x1, [x2]
    stxr    w1, x18, [x2]                   // 31 to x1: status collision
    stxr    w1, x18, [x2]
    stlxr   w1, x2, [x18]                   // 32 to x1: status and base collision
    stlxr   w1, x2, [x18]
    // SIMD&FP transfer registers are never renamed.
    ldr     q18, [x18, #32]                 // 33
    ldr     q18, [x9, #32]
    str     d9, [x18]                       // 34
    str     d9, [x9]
    ldr     s18, [x1]                       // 35
    ldr     s18, [x1]
    // Branches, the implicit link register and system moves.
    bl      .+8                             // 36 x30: implicit
    bl      .+8
    blr     x18                             // 37
    blr     x9
    blr     x18                             // 38 to x30: link collision
    blr     x18
    blr     x30                             // 39 x30: implicit
    blr     x30
    ret                                     // 40 x30 to x9
    ret     x9
    br      x18                             // 41
    br      x9
    cbnz    w18, .+8                        // 42
    cbnz    w9, .+8
    adr     x18, .+16                       // 43
    adr     x9, .+16
    mrs     x18, tpidr_el0                  // 44
    mrs     x9, tpidr_el0
    msr     nzcv, x18                       // 45
    msr     nzcv, x9
    // Collisions with ordinary operands and the same-register policy.
    add     x18, x9, x2                     // 46 to x9: collision
    add     x18, x9, x2
    add     x18, x1, x2                     // 47 x18 to x18
    add     x18, x1, x2
    // Outside decoded coverage, or handler-defined registers.
    ldadd   x1, x18, [x2]                   // 48
    ldadd   x1, x9, [x2]
    cas     x1, x18, [x2]                   // 49
    cas     x1, x9, [x2]
    paciasp                                 // 50 x30
    paciasp
    fmov    d18, x18                        // 51
    fmov    d18, x9
    svc     #0x80                           // 52
    svc     #0x80
    brk     #18                             // 53
    brk     #18
    .globl _armigo_substitution_pairs_end
_armigo_substitution_pairs_end:

// Native harness. x0 points to the state: x9..x15 at offsets 0..48, NZCV at
// 56. The words between \symbol_words and \symbol_end are the case.
.macro HARNESS_ENTRY symbol
    .p2align 2
    .globl \symbol
\symbol:
    str     d10, [sp, #-16]!
    ldp     x9, x10, [x0]
    ldp     x11, x12, [x0, #16]
    ldp     x13, x14, [x0, #32]
    ldp     x15, x1, [x0, #48]
    msr     nzcv, x1
    .globl \symbol\()_words
\symbol\()_words:
.endm

.macro HARNESS_EXIT symbol
    .globl \symbol\()_end
\symbol\()_end:
    mrs     x1, nzcv
    stp     x9, x10, [x0]
    stp     x11, x12, [x0, #16]
    stp     x13, x14, [x0, #32]
    stp     x15, x1, [x0, #48]
    ldr     d10, [sp], #16
    ret
.endm

HARNESS_ENTRY _armigo_native_add_source_original
    add     x9, x10, x11
HARNESS_EXIT _armigo_native_add_source_original
HARNESS_ENTRY _armigo_native_add_source_renamed
    add     x9, x12, x11
HARNESS_EXIT _armigo_native_add_source_renamed

HARNESS_ENTRY _armigo_native_add_destination_original
    add     x10, x9, x11
HARNESS_EXIT _armigo_native_add_destination_original
HARNESS_ENTRY _armigo_native_add_destination_renamed
    add     x12, x9, x11
HARNESS_EXIT _armigo_native_add_destination_renamed

// The immediate equals the renamed register number in the Ra position.
HARNESS_ENTRY _armigo_native_w_read_write_original
    add     w10, w10, #10
HARNESS_EXIT _armigo_native_w_read_write_original
HARNESS_ENTRY _armigo_native_w_read_write_renamed
    add     w12, w12, #10
HARNESS_EXIT _armigo_native_w_read_write_renamed

HARNESS_ENTRY _armigo_native_movk_original
    movk    x10, #0x1234, lsl #16
HARNESS_EXIT _armigo_native_movk_original
HARNESS_ENTRY _armigo_native_movk_renamed
    movk    x12, #0x1234, lsl #16
HARNESS_EXIT _armigo_native_movk_renamed

HARNESS_ENTRY _armigo_native_flags_original
    adds    x9, x10, x11
    csel    x13, x10, x11, mi
HARNESS_EXIT _armigo_native_flags_original
HARNESS_ENTRY _armigo_native_flags_renamed
    adds    x9, x12, x11
    csel    x13, x12, x11, mi
HARNESS_EXIT _armigo_native_flags_renamed

HARNESS_ENTRY _armigo_native_multiply_original
    madd    x9, x10, x11, x10
HARNESS_EXIT _armigo_native_multiply_original
HARNESS_ENTRY _armigo_native_multiply_renamed
    madd    x9, x12, x11, x12
HARNESS_EXIT _armigo_native_multiply_renamed

// immr and the conditional-compare immediate equal 10 in the Rm position.
HARNESS_ENTRY _armigo_native_immediate_fields_original
    lsr     x9, x10, #10
    ccmp    x10, #10, #4, ne
    bfi     x11, x10, #8, #16
HARNESS_EXIT _armigo_native_immediate_fields_original
HARNESS_ENTRY _armigo_native_immediate_fields_renamed
    lsr     x9, x12, #10
    ccmp    x12, #10, #4, ne
    bfi     x11, x12, #8, #16
HARNESS_EXIT _armigo_native_immediate_fields_renamed

HARNESS_ENTRY _armigo_native_load_writeback_original
    ldr     x9, [x10, #8]!
HARNESS_EXIT _armigo_native_load_writeback_original
HARNESS_ENTRY _armigo_native_load_writeback_renamed
    ldr     x9, [x12, #8]!
HARNESS_EXIT _armigo_native_load_writeback_renamed

HARNESS_ENTRY _armigo_native_store_post_index_original
    str     x10, [x9], #8
HARNESS_EXIT _armigo_native_store_post_index_original
HARNESS_ENTRY _armigo_native_store_post_index_renamed
    str     x12, [x9], #8
HARNESS_EXIT _armigo_native_store_post_index_renamed

HARNESS_ENTRY _armigo_native_pair_original
    ldp     x9, x11, [x10]
    stp     x10, x11, [x13, #16]
HARNESS_EXIT _armigo_native_pair_original
HARNESS_ENTRY _armigo_native_pair_renamed
    ldp     x9, x11, [x12]
    stp     x12, x11, [x13, #16]
HARNESS_EXIT _armigo_native_pair_renamed

HARNESS_ENTRY _armigo_native_index_original
    ldr     x9, [x11, x10, lsl #3]
HARNESS_EXIT _armigo_native_index_original
HARNESS_ENTRY _armigo_native_index_renamed
    ldr     x9, [x11, x12, lsl #3]
HARNESS_EXIT _armigo_native_index_renamed

HARNESS_ENTRY _armigo_native_branch_original
    cbz     x10, 1f
    add     x9, x9, #1
1:
HARNESS_EXIT _armigo_native_branch_original
HARNESS_ENTRY _armigo_native_branch_renamed
    cbz     x12, 1f
    add     x9, x9, #1
1:
HARNESS_EXIT _armigo_native_branch_renamed

// Every word of the exclusive window is renamed in place; nothing is inserted.
HARNESS_ENTRY _armigo_native_exclusive_original
1:
    ldxr    x9, [x10]
    add     x9, x9, #1
    stxr    w11, x9, [x10]
    cbnz    w11, 1b
HARNESS_EXIT _armigo_native_exclusive_original
HARNESS_ENTRY _armigo_native_exclusive_renamed
1:
    ldxr    x9, [x12]
    add     x9, x9, #1
    stxr    w11, x9, [x12]
    cbnz    w11, 1b
HARNESS_EXIT _armigo_native_exclusive_renamed

// V10 shares the renamed register's number and keeps it.
HARNESS_ENTRY _armigo_native_vector_original
    ldr     d10, [x10]
    str     d10, [x13]
HARNESS_EXIT _armigo_native_vector_original
HARNESS_ENTRY _armigo_native_vector_renamed
    ldr     d10, [x12]
    str     d10, [x13]
HARNESS_EXIT _armigo_native_vector_renamed
