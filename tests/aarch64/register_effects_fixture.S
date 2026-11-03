// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently written instructions encoded by the system assembler from
// mnemonics, including preferred aliases, SP and zero-register operands, and
// x18 in every register position. The code is never executed; the tests read
// the words and compare decoded register effects with expectations written
// from the mnemonics.
//
// One instruction per entry. Entry order is mirrored by the expectation table
// in register_effects_fixture_tests.cpp; the numbers are entry indices.

    .arch armv8.2-a
    .arch_extension crc

    .section __TEXT,__text,regular,pure_instructions
    .p2align 2
    .globl _armigo_register_effects_fixture
_armigo_register_effects_fixture:
    // Add/subtract (immediate).
    mov     x18, sp                          // 0
    mov     sp, x18                          // 1
    add     w18, wsp, #4095                  // 2
    adds    x18, x18, #1, lsl #12            // 3
    cmn     w18, #7                          // 4
    sub     sp, sp, #0x40                    // 5
    cmp     x18, #0                          // 6
    // Logical (immediate).
    and     x18, x18, #0xfffffffffffffffe    // 7
    orr     sp, x1, #0x10                    // 8
    orr     w18, wzr, #0xff00ff00            // 9
    eor     x1, x18, #0x5555555555555555     // 10
    tst     w18, #0x1                        // 11
    // Move wide.
    mov     x18, #-1                         // 12
    movz    w18, #0xffff, lsl #16            // 13
    movk    x18, #0xbeef, lsl #32            // 14
    // Bitfield and extract.
    sxtw    x18, w1                          // 15
    asr     x1, x18, #63                     // 16
    bfxil   x18, x1, #8, #8                  // 17
    bfi     w1, w18, #31, #1                 // 18
    bfc     x18, #4, #8                      // 19
    lsr     w18, w18, #1                     // 20
    uxtb    w18, w1                          // 21
    extr    x18, x18, x1, #12                // 22
    extr    x1, x2, x18, #3                  // 23
    // Logical (shifted register).
    and     x18, x1, x2                      // 24
    bic     w1, w18, w2, lsl #3              // 25
    orr     x1, x2, x18, ror #7              // 26
    mov     x1, x18                          // 27
    mvn     w18, w18                         // 28
    eor     x18, x18, x18                    // 29
    eon     x1, x2, x3                       // 30
    tst     x18, x1, lsr #2                  // 31
    bics    x18, x1, x2                      // 32
    // Add/subtract (shifted register).
    add     x18, x1, x18, lsl #3             // 33
    adds    w1, w18, w2, asr #2              // 34
    neg     x18, x1                          // 35
    negs    w1, w18                          // 36
    cmp     x1, x18, lsl #1                  // 37
    // Add/subtract (extended register).
    add     x18, sp, x1                      // 38
    add     sp, x18, w1, uxtw #2             // 39
    add     x1, x2, w18, sxth                // 40
    adds    x18, sp, w1, uxtb #1             // 41
    sub     sp, sp, x18                      // 42
    cmp     sp, x18                          // 43
    cmn     w18, w1, uxtb                    // 44
    // Add/subtract with carry.
    adc     x18, x1, x2                      // 45
    adcs    w1, w18, w2                      // 46
    sbc     x1, x2, x18                      // 47
    ngcs    w18, w1                          // 48
    // Conditional compare and select.
    ccmp    x18, x1, #0, eq                  // 49
    ccmn    w1, w18, #15, lt                 // 50
    ccmp    x18, #18, #2, hs                 // 51
    ccmn    x1, #18, #0, ne                  // 52
    csel    x18, x18, x1, gt                 // 53
    csinc   w1, w2, w18, ls                  // 54
    cinc    x18, x1, eq                      // 55
    csetm   w18, ne                          // 56
    cinv    x1, x18, ge                      // 57
    csneg   x18, x1, x2, vs                  // 58
    cneg    w1, w18, mi                      // 59
    // Data processing (2 source).
    udiv    w18, w1, w2                      // 60
    sdiv    x1, x18, x2                      // 61
    lsl     x1, x2, x18                      // 62
    lsr     w18, w18, w1                     // 63
    asr     x1, x18, x18                     // 64
    ror     w1, w2, w18                      // 65
    crc32h  w18, w1, w2                      // 66
    crc32w  w1, w18, w2                      // 67
    crc32x  w1, w2, x18                      // 68
    crc32cb w1, w2, w18                      // 69
    crc32ch w18, w18, w18                    // 70
    crc32cw w1, w2, w3                       // 71
    crc32cx w18, w1, x2                      // 72
    // Data processing (1 source).
    rbit    w18, w1                          // 73
    rev16   x1, x18                          // 74
    rev32   x18, x18                         // 75
    rev     x18, x1                          // 76
    rev     w1, w18                          // 77
    clz     w18, w18                         // 78
    cls     x1, x18                          // 79
    // Data processing (3 source).
    madd    x18, x1, x2, x3                  // 80
    msub    w1, w18, w2, w3                  // 81
    mul     x1, x2, x18                      // 82
    mneg    x18, x1, x2                      // 83
    madd    x1, x2, x3, x18                  // 84
    smaddl  x18, w1, w2, x3                  // 85
    smull   x1, w18, w2                      // 86
    smsubl  x1, w2, w3, x18                  // 87
    smnegl  x1, w2, w18                      // 88
    umaddl  x1, w2, w3, x18                  // 89
    umull   x18, w1, w2                      // 90
    umsubl  x1, w18, w2, x3                  // 91
    umnegl  x18, w18, w18                    // 92
    smulh   x18, x1, x2                      // 93
    umulh   x1, x2, x18                      // 94
    // Hints and barriers.
    nop                                      // 95
    dmb     ish                              // 96
    dsb     sy                               // 97
    isb                                      // 98
    clrex                                    // 99
    // Load/store register (unsigned offset).
    ldr     x18, [sp, #32760]                // 100
    ldr     w1, [x18]                        // 101
    str     x18, [x18, #8]                   // 102
    strb    w18, [sp, #4095]                 // 103
    ldrh    w18, [x1, #2]                    // 104
    ldrsh   w1, [x18, #4094]                 // 105
    ldrsb   x18, [x1, #1]                    // 106
    ldrsw   x1, [x18, #16]                   // 107
    strh    wzr, [x18, #2]                   // 108
    prfm    pldl1keep, [x18]                 // 109
    prfm    pstl2keep, [x1, #8]              // 110: prefetch operation 18
    // Unscaled and unprivileged.
    ldur    x18, [x1, #-1]                   // 111
    stur    w1, [x18, #255]                  // 112
    ldurb   w18, [sp, #-256]                 // 113
    ldursw  x18, [x18, #4]                   // 114
    prfum   pstl2keep, [x18, #1]             // 115
    ldtr    x18, [x1]                        // 116
    sttrh   w1, [x18, #2]                    // 117
    ldtrsb  w18, [x1, #-1]                   // 118
    // Pre- and post-index.
    ldr     x18, [x1, #8]!                   // 119
    str     w1, [x18], #-4                   // 120
    stp     x29, x30, [sp, #-16]!            // 121
    ldp     x29, x30, [sp], #16              // 122
    str     x18, [sp, #-16]!                 // 123
    ldr     x18, [sp], #16                   // 124
    ldrb    w1, [x18, #1]!                   // 125
    // Register offset.
    ldr     x1, [x2, x18]                    // 126
    ldr     w18, [x1, w2, uxtw #2]           // 127
    str     x1, [x18, x2, lsl #3]            // 128
    ldrsw   x1, [sp, w18, sxtw]              // 129
    ldrb    w1, [x2, x18, lsl #0]            // 130
    strh    w18, [x1, x2, sxtx #1]           // 131
    prfm    pstl2keep, [x1, x18]             // 132
    // SIMD&FP transfers through general-register addresses.
    ldr     q18, [x1]                        // 133
    str     d1, [x18, #8]                    // 134
    ldr     b1, [x18, x2]                    // 135
    str     q1, [x2, x18, lsl #4]            // 136
    ldr     s18, [x18], #4                   // 137
    ldur    h1, [x18, #-2]                   // 138
    // Register pairs.
    ldp     x18, x1, [x2]                    // 139
    ldp     x1, x18, [x2, #8]                // 140
    stp     x1, x2, [x18, #16]               // 141
    stp     w18, w18, [x1]                   // 142
    ldp     w1, w2, [x18, #-8]!              // 143
    ldpsw   x18, x1, [sp, #8]                // 144
    stnp    x18, x1, [x2]                    // 145
    ldnp    w1, w2, [x18, #252]              // 146
    ldp     q18, q1, [x18]                   // 147
    stp     d1, d2, [x18, #-512]!            // 148
    ldnp    s1, s18, [x1]                    // 149
    stnp    q1, q2, [x18, #1008]             // 150
    // Instructions classified by instruction.h.
    cbz     x18, _armigo_register_effects_fixture     // 151
    br      x18                              // 152
    blr     x18                              // 153
    ret                                      // 154
    mrs     x18, tpidr_el0                   // 155
    adrp    x18, _armigo_register_effects_fixture@PAGE  // 156
    svc     #0                               // 157
    // Exclusive and ordered accesses.
    ldxr    x18, [x1]                        // 158
    ldaxrb  w1, [x18]                        // 159
    stxr    w18, x1, [x2]                    // 160
    stlxrh  w1, w18, [sp]                    // 161
    ldxp    x18, x1, [x2]                    // 162
    stlxp   w1, x2, x18, [x3]                // 163
    stxp    w2, w3, w4, [x18]                // 164
    ldar    w18, [x1]                        // 165
    stlr    x1, [x18]                        // 166
    ldarb   w1, [sp]                         // 167
    .globl _armigo_register_effects_fixture_end
_armigo_register_effects_fixture_end:

// Independently assembled originals and x18-to-x9 substitutions.
    .arch armv8.2-a+fp16
    .globl _armigo_simd_effect_pairs
_armigo_simd_effect_pairs:
    st1 { v18.1d }, [x18]
    st1 { v18.1d }, [x9]
    st1 { v18.1d }, [x18], #8
    st1 { v18.1d }, [x9], #8
    st1 { v18.1d }, [x1], x18
    st1 { v18.1d }, [x1], x9
    st1 { v18.2d, v19.2d }, [x18]
    st1 { v18.2d, v19.2d }, [x9]
    st1 { v18.2d, v19.2d }, [x18], #32
    st1 { v18.2d, v19.2d }, [x9], #32
    st1 { v18.2d, v19.2d }, [x1], x18
    st1 { v18.2d, v19.2d }, [x1], x9
    st1 { v18.4h, v19.4h, v20.4h }, [x18]
    st1 { v18.4h, v19.4h, v20.4h }, [x9]
    st1 { v18.4h, v19.4h, v20.4h }, [x18], #24
    st1 { v18.4h, v19.4h, v20.4h }, [x9], #24
    st1 { v18.4h, v19.4h, v20.4h }, [x1], x18
    st1 { v18.4h, v19.4h, v20.4h }, [x1], x9
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18]
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x9]
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18], #64
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x9], #64
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x18
    st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x9
    st2 { v18.2s, v19.2s }, [x18]
    st2 { v18.2s, v19.2s }, [x9]
    st2 { v18.2s, v19.2s }, [x18], #16
    st2 { v18.2s, v19.2s }, [x9], #16
    st2 { v18.2s, v19.2s }, [x1], x18
    st2 { v18.2s, v19.2s }, [x1], x9
    st3 { v18.8h, v19.8h, v20.8h }, [x18]
    st3 { v18.8h, v19.8h, v20.8h }, [x9]
    st3 { v18.8h, v19.8h, v20.8h }, [x18], #48
    st3 { v18.8h, v19.8h, v20.8h }, [x9], #48
    st3 { v18.8h, v19.8h, v20.8h }, [x1], x18
    st3 { v18.8h, v19.8h, v20.8h }, [x1], x9
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18]
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x9]
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18], #64
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x9], #64
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x18
    st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x9
    st1 { v18.b }[15], [x18]
    st1 { v18.b }[15], [x9]
    st1 { v18.b }[15], [x18], #1
    st1 { v18.b }[15], [x9], #1
    st1 { v18.b }[15], [x1], x18
    st1 { v18.b }[15], [x1], x9
    st2 { v18.h, v19.h }[6], [x18]
    st2 { v18.h, v19.h }[6], [x9]
    st2 { v18.h, v19.h }[6], [x18], #4
    st2 { v18.h, v19.h }[6], [x9], #4
    st2 { v18.h, v19.h }[6], [x1], x18
    st2 { v18.h, v19.h }[6], [x1], x9
    st3 { v18.s, v19.s, v20.s }[3], [x18]
    st3 { v18.s, v19.s, v20.s }[3], [x9]
    st3 { v18.s, v19.s, v20.s }[3], [x18], #12
    st3 { v18.s, v19.s, v20.s }[3], [x9], #12
    st3 { v18.s, v19.s, v20.s }[3], [x1], x18
    st3 { v18.s, v19.s, v20.s }[3], [x1], x9
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x18]
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x9]
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x18], #32
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x9], #32
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x18
    st4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x9
    ld1 { v18.1d }, [x18]
    ld1 { v18.1d }, [x9]
    ld1 { v18.1d }, [x18], #8
    ld1 { v18.1d }, [x9], #8
    ld1 { v18.1d }, [x1], x18
    ld1 { v18.1d }, [x1], x9
    ld1 { v18.2d, v19.2d }, [x18]
    ld1 { v18.2d, v19.2d }, [x9]
    ld1 { v18.2d, v19.2d }, [x18], #32
    ld1 { v18.2d, v19.2d }, [x9], #32
    ld1 { v18.2d, v19.2d }, [x1], x18
    ld1 { v18.2d, v19.2d }, [x1], x9
    ld1 { v18.4h, v19.4h, v20.4h }, [x18]
    ld1 { v18.4h, v19.4h, v20.4h }, [x9]
    ld1 { v18.4h, v19.4h, v20.4h }, [x18], #24
    ld1 { v18.4h, v19.4h, v20.4h }, [x9], #24
    ld1 { v18.4h, v19.4h, v20.4h }, [x1], x18
    ld1 { v18.4h, v19.4h, v20.4h }, [x1], x9
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18]
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x9]
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18], #64
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x9], #64
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x18
    ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x9
    ld2 { v18.2s, v19.2s }, [x18]
    ld2 { v18.2s, v19.2s }, [x9]
    ld2 { v18.2s, v19.2s }, [x18], #16
    ld2 { v18.2s, v19.2s }, [x9], #16
    ld2 { v18.2s, v19.2s }, [x1], x18
    ld2 { v18.2s, v19.2s }, [x1], x9
    ld3 { v18.8h, v19.8h, v20.8h }, [x18]
    ld3 { v18.8h, v19.8h, v20.8h }, [x9]
    ld3 { v18.8h, v19.8h, v20.8h }, [x18], #48
    ld3 { v18.8h, v19.8h, v20.8h }, [x9], #48
    ld3 { v18.8h, v19.8h, v20.8h }, [x1], x18
    ld3 { v18.8h, v19.8h, v20.8h }, [x1], x9
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18]
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x9]
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18], #64
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x9], #64
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x18
    ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x9
    ld1 { v18.b }[15], [x18]
    ld1 { v18.b }[15], [x9]
    ld1 { v18.b }[15], [x18], #1
    ld1 { v18.b }[15], [x9], #1
    ld1 { v18.b }[15], [x1], x18
    ld1 { v18.b }[15], [x1], x9
    ld2 { v18.h, v19.h }[6], [x18]
    ld2 { v18.h, v19.h }[6], [x9]
    ld2 { v18.h, v19.h }[6], [x18], #4
    ld2 { v18.h, v19.h }[6], [x9], #4
    ld2 { v18.h, v19.h }[6], [x1], x18
    ld2 { v18.h, v19.h }[6], [x1], x9
    ld3 { v18.s, v19.s, v20.s }[3], [x18]
    ld3 { v18.s, v19.s, v20.s }[3], [x9]
    ld3 { v18.s, v19.s, v20.s }[3], [x18], #12
    ld3 { v18.s, v19.s, v20.s }[3], [x9], #12
    ld3 { v18.s, v19.s, v20.s }[3], [x1], x18
    ld3 { v18.s, v19.s, v20.s }[3], [x1], x9
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x18]
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x9]
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x18], #32
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x9], #32
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x18
    ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x9
    ld1r { v18.16b }, [x18]
    ld1r { v18.16b }, [x9]
    ld1r { v18.16b }, [x18], #1
    ld1r { v18.16b }, [x9], #1
    ld1r { v18.16b }, [x1], x18
    ld1r { v18.16b }, [x1], x9
    ld2r { v18.4h, v19.4h }, [x18]
    ld2r { v18.4h, v19.4h }, [x9]
    ld2r { v18.4h, v19.4h }, [x18], #4
    ld2r { v18.4h, v19.4h }, [x9], #4
    ld2r { v18.4h, v19.4h }, [x1], x18
    ld2r { v18.4h, v19.4h }, [x1], x9
    ld3r { v18.4s, v19.4s, v20.4s }, [x18]
    ld3r { v18.4s, v19.4s, v20.4s }, [x9]
    ld3r { v18.4s, v19.4s, v20.4s }, [x18], #12
    ld3r { v18.4s, v19.4s, v20.4s }, [x9], #12
    ld3r { v18.4s, v19.4s, v20.4s }, [x1], x18
    ld3r { v18.4s, v19.4s, v20.4s }, [x1], x9
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x18]
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x9]
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x18], #32
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x9], #32
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x1], x18
    ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x1], x9
    ld4 { v31.16b, v0.16b, v1.16b, v2.16b }, [x18], x18
    ld4 { v31.16b, v0.16b, v1.16b, v2.16b }, [x9], x9
    st1 { v18.16b }, [sp], x18
    st1 { v18.16b }, [sp], x9
    ld1 { v18.b }[9], [sp], #1
    ld1 { v18.b }[9], [sp], #1
    st2 { v18.d, v19.d }[0], [sp]
    st2 { v18.d, v19.d }[0], [sp]
    smov w18, v18.b[15]
    smov w9, v18.b[15]
    smov x18, v18.h[4]
    smov x9, v18.h[4]
    smov x18, v18.s[0]
    smov x9, v18.s[0]
    umov w18, v18.b[0]
    umov w9, v18.b[0]
    umov w18, v18.h[7]
    umov w9, v18.h[7]
    umov w18, v18.s[2]
    umov w9, v18.s[2]
    umov x18, v18.d[1]
    umov x9, v18.d[1]
    ins v18.b[15], w18
    ins v18.b[15], w9
    ins v18.h[4], w18
    ins v18.h[4], w9
    ins v18.s[0], w18
    ins v18.s[0], w9
    ins v18.d[1], x18
    ins v18.d[1], x9
    dup v18.8b, w18
    dup v18.8b, w9
    dup v18.8h, w18
    dup v18.8h, w9
    dup v18.2s, w18
    dup v18.2s, w9
    dup v18.2d, x18
    dup v18.2d, x9
    fcvtns w18, s18
    fcvtns w9, s18
    fcvtnu x18, d18
    fcvtnu x9, d18
    fcvtas w18, h18
    fcvtas w9, h18
    fcvtau x18, s18
    fcvtau x9, s18
    fcvtps w18, d18
    fcvtps w9, d18
    fcvtpu x18, h18
    fcvtpu x9, h18
    fcvtms w18, s18
    fcvtms w9, s18
    fcvtmu x18, d18
    fcvtmu x9, d18
    fcvtzs w18, h18
    fcvtzs w9, h18
    fcvtzu x18, s18
    fcvtzu x9, s18
    fcvtzs w18, s18
    fcvtzs w9, s18
    fcvtzs w18, s18, #32
    fcvtzs w9, s18, #32
    fcvtzs x18, d18
    fcvtzs x9, d18
    fcvtzs x18, d18, #64
    fcvtzs x9, d18, #64
    fcvtzs x18, h18
    fcvtzs x9, h18
    fcvtzs x18, h18, #1
    fcvtzs x9, h18, #1
    fcvtzu w18, s18
    fcvtzu w9, s18
    fcvtzu w18, s18, #32
    fcvtzu w9, s18, #32
    fcvtzu x18, d18
    fcvtzu x9, d18
    fcvtzu x18, d18, #64
    fcvtzu x9, d18, #64
    fcvtzu x18, h18
    fcvtzu x9, h18
    fcvtzu x18, h18, #1
    fcvtzu x9, h18, #1
    scvtf s18, w18
    scvtf s18, w9
    scvtf s18, w18, #32
    scvtf s18, w9, #32
    scvtf d18, x18
    scvtf d18, x9
    scvtf d18, x18, #64
    scvtf d18, x9, #64
    scvtf h18, x18
    scvtf h18, x9
    scvtf h18, x18, #1
    scvtf h18, x9, #1
    ucvtf s18, w18
    ucvtf s18, w9
    ucvtf s18, w18, #32
    ucvtf s18, w9, #32
    ucvtf d18, x18
    ucvtf d18, x9
    ucvtf d18, x18, #64
    ucvtf d18, x9, #64
    ucvtf h18, x18
    ucvtf h18, x9
    ucvtf h18, x18, #1
    ucvtf h18, x9, #1
    fmov w18, h18
    fmov w9, h18
    fmov h18, w18
    fmov h18, w9
    fmov x18, h18
    fmov x9, h18
    fmov h18, x18
    fmov h18, x9
    fmov w18, s18
    fmov w9, s18
    fmov s18, w18
    fmov s18, w9
    fmov x18, d18
    fmov x9, d18
    fmov d18, x18
    fmov d18, x9
    fmov x18, v18.d[1]
    fmov x9, v18.d[1]
    fmov v18.d[1], x18
    fmov v18.d[1], x9
    .globl _armigo_simd_effect_pairs_end
_armigo_simd_effect_pairs_end:

// Advanced SIMD data processing with no general registers, encoded by the
// system assembler. One instruction per entry; entry order is mirrored by
// the expectation table in register_effects_fixture_tests.cpp.
    .arch armv8.2-a+fp16+dotprod+aes+sha2
    .text
    .p2align 2
    .globl _armigo_simd_data_fixture
_armigo_simd_data_fixture:
    add v0.4s, v1.4s, v2.4s                 // 0 three same
    sqadd v0.8h, v1.8h, v2.8h               // 1
    addp v0.2d, v1.2d, v2.2d                // 2
    mla v0.4s, v1.4s, v2.4s                 // 3 accumulate
    and v0.16b, v1.16b, v2.16b              // 4 bitwise
    bsl v0.16b, v1.16b, v2.16b              // 5 bitwise keep
    fadd v0.4s, v1.4s, v2.4s                // 6 FP three same
    fmla v0.4s, v1.4s, v2.4s                // 7 FP accumulate
    fabd v0.8h, v1.8h, v2.8h                // 8 FP16 three same
    fmls v0.8h, v1.8h, v2.8h                // 9 FP16 accumulate
    saddl v0.8h, v1.8b, v2.8b               // 10 long
    addhn v0.8b, v1.8h, v2.8h               // 11 narrow
    smlal v0.4s, v1.4h, v2.4h               // 12 long accumulate
    saddw v0.8h, v1.8h, v2.8b               // 13 wide
    rev64 v0.8h, v1.8h                      // 14 misc
    suqadd v0.4s, v1.4s                     // 15 misc accumulate
    cmgt v0.4s, v1.4s, #0                   // 16 compare zero
    abs v0.8h, v1.8h                        // 17
    fcmgt v0.4s, v1.4s, #0                  // 18 FP compare zero
    fneg v0.4s, v1.4s                       // 19 FP misc
    addv s0, v1.4s                          // 20 across lanes
    smaxv h0, v1.8h                         // 21
    fmaxv s0, v1.4s                         // 22 FP across
    uzp1 v0.8h, v1.8h, v2.8h                // 23 permute
    zip2 v0.16b, v1.16b, v2.16b             // 24
    ext v0.16b, v1.16b, v2.16b, #3          // 25 extract
    tbl v0.16b, {v1.16b, v2.16b}, v3.16b    // 26 table two registers
    tbx v0.16b, {v1.16b}, v2.16b            // 27 table keep
    dup v0.4s, v1.s[2]                      // 28 copy element
    ins v0.s[1], v1.s[2]                    // 29 insert element
    sshr v0.4s, v1.4s, #7                   // 30 shift immediate
    ssra v0.4s, v1.4s, #7                   // 31 shift accumulate
    sshll v0.8h, v1.8b, #3                  // 32 widen
    shrn v0.8b, v1.8h, #7                   // 33 narrow
    sqshrn v0.2s, v1.2d, #16                // 34 narrow signed
    movi v0.4s, #8                          // 35 modified immediate
    mvni v0.4s, #8                          // 36
    orr v0.4h, #0x5a                        // 37 logical immediate
    fmov v0.2s, #1.0                        // 38 FP immediate
    mla v0.4s, v1.4s, v2.s[1]               // 39 vector x indexed
    smull v0.2d, v1.2s, v2.s[1]             // 40 indexed long
    sqdmulh v0.4s, v1.4s, v2.s[1]           // 41 indexed saturating
    fmla v0.4s, v1.4s, v2.s[1]              // 42 FP indexed
    fmul v0.8h, v1.8h, v2.h[2]              // 43 FP16 indexed
    sdot v0.4s, v1.16b, v2.16b              // 44 dot product
    aese v0.16b, v1.16b                     // 45 AES
    sha1c q0, s1, v2.4s                     // 46 SHA three register
    // Scalar forms.
.word 0x5E228420                           // 47 ADD S0, S1, S2 (scalar int three same)
    fadd s0, s1, s2                         // 48 scalar FP two source
    fdiv d0, d1, d2                         // 49
    fmulx s0, s1, s2                        // 50
    fabs s0, s1                             // 51 one source
    fcvt d0, s1                             // 52 precision change
    fcvt h0, s1                             // 53
    fcvt s0, h1                             // 54
    fcmp s0, s1                             // 55 compare
    fcmpe d0, #0.0                          // 56 compare zero extended
    fccmp s0, s1, #2, eq                    // 57 conditional compare
    fcsel s0, s1, s2, mi                    // 58 conditional select
    fmov s0, #1.0                           // 59 immediate
    fmadd s0, s1, s2, s3                    // 60 three source
    sshr d0, d1, #3                         // 61 scalar shift
    sri d0, d1, #3                          // 62 scalar shift insert
    sqshrun h0, s1, #16                     // 63 scalar narrow shift
    sqxtn s0, d1                            // 64 scalar narrow
    frecpe s0, s1                           // 65 scalar reciprocal estimate
    frecps s0, s1, s2                       // 66 scalar reciprocal step
    frecpx d0, d1                           // 68 scalar reciprocal exact
    uqadd b0, b1, b2                        // 69 scalar byte saturating
    fcmgt s0, s1, #0                        // 70 scalar FP compare zero
    addp d0, v1.2d                          // 71 ADDP D (scalar pairwise)
    sha1h s0, s1                            // 72 SHA one register
    .globl _armigo_simd_data_fixture_end
_armigo_simd_data_fixture_end:
