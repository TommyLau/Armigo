// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// This wrapper executes only prepared instructions or assembler-authored X28
// reference instructions. It captures the complete GPR/vector state and owns
// a frame separate from the trampoline's sixteen scratch bytes.
    .text
    .p2align 2
    .globl _armigo_test_x18_forms_run
_armigo_test_x18_forms_run:
    sub sp, sp, #256
    stp x29, x30, [sp]
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp x0, x1, [sp, #96]
    stp d8, d9, [sp, #128]
    stp d10, d11, [sp, #144]
    stp d12, d13, [sp, #160]
    stp d14, d15, [sp, #176]
    mrs x9, fpcr
    mrs x10, fpsr
    stp x9, x10, [sp, #192]
    str x18, [x0, #264]
    ldr x9, [x0, #248]
    msr nzcv, x9
    ldr x9, [x0, #784]
    ldr x10, [x0, #792]
    msr fpcr, x9
    msr fpsr, x10
    ldp q0, q1, [x0, #272]
    ldp q2, q3, [x0, #304]
    ldp q4, q5, [x0, #336]
    ldp q6, q7, [x0, #368]
    ldp q8, q9, [x0, #400]
    ldp q10, q11, [x0, #432]
    ldp q12, q13, [x0, #464]
    ldp q14, q15, [x0, #496]
    ldp q16, q17, [x0, #528]
    ldp q18, q19, [x0, #560]
    ldp q20, q21, [x0, #592]
    ldp q22, q23, [x0, #624]
    ldp q24, q25, [x0, #656]
    ldp q26, q27, [x0, #688]
    ldp q28, q29, [x0, #720]
    ldp q30, q31, [x0, #752]
    ldp x2, x3, [x0, #16]
    ldp x4, x5, [x0, #32]
    ldp x6, x7, [x0, #48]
    ldp x8, x9, [x0, #64]
    ldp x10, x11, [x0, #80]
    ldp x12, x13, [x0, #96]
    ldp x14, x15, [x0, #112]
    ldp x16, x17, [x0, #128]
    ldr x19, [x0, #152]
    ldp x20, x21, [x0, #160]
    ldp x22, x23, [x0, #176]
    ldp x24, x25, [x0, #192]
    ldp x26, x27, [x0, #208]
    ldp x28, x29, [x0, #224]
    ldr x1, [x0, #8]
    ldr x30, [sp, #104]
    ldr x0, [x0]
    blr x30
    stp x0, x1, [sp, #112]
    ldr x0, [sp, #96]
    stp x2, x3, [x0, #16]
    stp x4, x5, [x0, #32]
    stp x6, x7, [x0, #48]
    stp x8, x9, [x0, #64]
    stp x10, x11, [x0, #80]
    stp x12, x13, [x0, #96]
    stp x14, x15, [x0, #112]
    stp x16, x17, [x0, #128]
    stp x18, x19, [x0, #144]
    stp x20, x21, [x0, #160]
    stp x22, x23, [x0, #176]
    stp x24, x25, [x0, #192]
    stp x26, x27, [x0, #208]
    stp x28, x29, [x0, #224]
    str x30, [x0, #240]
    mrs x9, nzcv
    str x9, [x0, #248]
    mov x9, sp
    str x9, [x0, #256]
    ldp x9, x10, [sp, #112]
    stp x9, x10, [x0]
    mrs x9, fpcr
    mrs x10, fpsr
    str x9, [x0, #784]
    str x10, [x0, #792]
    stp q0, q1, [x0, #272]
    stp q2, q3, [x0, #304]
    stp q4, q5, [x0, #336]
    stp q6, q7, [x0, #368]
    stp q8, q9, [x0, #400]
    stp q10, q11, [x0, #432]
    stp q12, q13, [x0, #464]
    stp q14, q15, [x0, #496]
    stp q16, q17, [x0, #528]
    stp q18, q19, [x0, #560]
    stp q20, q21, [x0, #592]
    stp q22, q23, [x0, #624]
    stp q24, q25, [x0, #656]
    stp q26, q27, [x0, #688]
    stp q28, q29, [x0, #720]
    stp q30, q31, [x0, #752]
    ldp x9, x10, [sp, #192]
    msr fpcr, x9
    msr fpsr, x10
    ldp d8, d9, [sp, #128]
    ldp d10, d11, [sp, #144]
    ldp d12, d13, [sp, #160]
    ldp d14, d15, [sp, #176]
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp]
    add sp, sp, #256
    ret

// Each table record has an input kind and two independently assembled words:
// virtual X18 input and its safe X28 reference. The table is never executed.
    .section __TEXT,__const
    .p2align 2
    .globl _armigo_test_x18_forms
_armigo_test_x18_forms:

    .macro structure stride, op, regs:vararg
    .long 1
    \op \regs, [x18]
    \op \regs, [x28]
    .long 1
    \op \regs, [x18], #\stride
    \op \regs, [x28], #\stride
    .long 1
    \op \regs, [x18], x2
    \op \regs, [x28], x2
    .endm

// Multiple structures: every opcode, vector width and element arrangement.
    structure 8, ld1, { v30.8b }
    structure 8, st1, { v30.8b }
    structure 16, ld1, { v30.16b }
    structure 16, st1, { v30.16b }
    structure 8, ld1, { v30.4h }
    structure 8, st1, { v30.4h }
    structure 16, ld1, { v30.8h }
    structure 16, st1, { v30.8h }
    structure 8, ld1, { v30.2s }
    structure 8, st1, { v30.2s }
    structure 16, ld1, { v30.4s }
    structure 16, st1, { v30.4s }
    structure 8, ld1, { v30.1d }
    structure 8, st1, { v30.1d }
    structure 16, ld1, { v30.2d }
    structure 16, st1, { v30.2d }
    structure 16, ld2, { v30.8b, v31.8b }
    structure 16, st2, { v30.8b, v31.8b }
    structure 32, ld2, { v30.16b, v31.16b }
    structure 32, st2, { v30.16b, v31.16b }
    structure 16, ld2, { v30.4h, v31.4h }
    structure 16, st2, { v30.4h, v31.4h }
    structure 32, ld2, { v30.8h, v31.8h }
    structure 32, st2, { v30.8h, v31.8h }
    structure 16, ld2, { v30.2s, v31.2s }
    structure 16, st2, { v30.2s, v31.2s }
    structure 32, ld2, { v30.4s, v31.4s }
    structure 32, st2, { v30.4s, v31.4s }
    structure 32, ld2, { v30.2d, v31.2d }
    structure 32, st2, { v30.2d, v31.2d }
    structure 24, ld3, { v30.8b, v31.8b, v0.8b }
    structure 24, st3, { v30.8b, v31.8b, v0.8b }
    structure 48, ld3, { v30.16b, v31.16b, v0.16b }
    structure 48, st3, { v30.16b, v31.16b, v0.16b }
    structure 24, ld3, { v30.4h, v31.4h, v0.4h }
    structure 24, st3, { v30.4h, v31.4h, v0.4h }
    structure 48, ld3, { v30.8h, v31.8h, v0.8h }
    structure 48, st3, { v30.8h, v31.8h, v0.8h }
    structure 24, ld3, { v30.2s, v31.2s, v0.2s }
    structure 24, st3, { v30.2s, v31.2s, v0.2s }
    structure 48, ld3, { v30.4s, v31.4s, v0.4s }
    structure 48, st3, { v30.4s, v31.4s, v0.4s }
    structure 48, ld3, { v30.2d, v31.2d, v0.2d }
    structure 48, st3, { v30.2d, v31.2d, v0.2d }
    structure 32, ld4, { v30.8b, v31.8b, v0.8b, v1.8b }
    structure 32, st4, { v30.8b, v31.8b, v0.8b, v1.8b }
    structure 64, ld4, { v30.16b, v31.16b, v0.16b, v1.16b }
    structure 64, st4, { v30.16b, v31.16b, v0.16b, v1.16b }
    structure 32, ld4, { v30.4h, v31.4h, v0.4h, v1.4h }
    structure 32, st4, { v30.4h, v31.4h, v0.4h, v1.4h }
    structure 64, ld4, { v30.8h, v31.8h, v0.8h, v1.8h }
    structure 64, st4, { v30.8h, v31.8h, v0.8h, v1.8h }
    structure 32, ld4, { v30.2s, v31.2s, v0.2s, v1.2s }
    structure 32, st4, { v30.2s, v31.2s, v0.2s, v1.2s }
    structure 64, ld4, { v30.4s, v31.4s, v0.4s, v1.4s }
    structure 64, st4, { v30.4s, v31.4s, v0.4s, v1.4s }
    structure 64, ld4, { v30.2d, v31.2d, v0.2d, v1.2d }
    structure 64, st4, { v30.2d, v31.2d, v0.2d, v1.2d }
    structure 16, ld1, { v30.8b, v31.8b }
    structure 16, st1, { v30.8b, v31.8b }
    structure 32, ld1, { v30.16b, v31.16b }
    structure 32, st1, { v30.16b, v31.16b }
    structure 16, ld1, { v30.1d, v31.1d }
    structure 16, st1, { v30.1d, v31.1d }
    structure 32, ld1, { v30.2d, v31.2d }
    structure 32, st1, { v30.2d, v31.2d }
    structure 24, ld1, { v30.8b, v31.8b, v0.8b }
    structure 24, st1, { v30.8b, v31.8b, v0.8b }
    structure 48, ld1, { v30.16b, v31.16b, v0.16b }
    structure 48, st1, { v30.16b, v31.16b, v0.16b }
    structure 24, ld1, { v30.1d, v31.1d, v0.1d }
    structure 24, st1, { v30.1d, v31.1d, v0.1d }
    structure 48, ld1, { v30.2d, v31.2d, v0.2d }
    structure 48, st1, { v30.2d, v31.2d, v0.2d }
    structure 32, ld1, { v30.8b, v31.8b, v0.8b, v1.8b }
    structure 32, st1, { v30.8b, v31.8b, v0.8b, v1.8b }
    structure 64, ld1, { v30.16b, v31.16b, v0.16b, v1.16b }
    structure 64, st1, { v30.16b, v31.16b, v0.16b, v1.16b }
    structure 32, ld1, { v30.1d, v31.1d, v0.1d, v1.1d }
    structure 32, st1, { v30.1d, v31.1d, v0.1d, v1.1d }
    structure 64, ld1, { v30.2d, v31.2d, v0.2d, v1.2d }
    structure 64, st1, { v30.2d, v31.2d, v0.2d, v1.2d }

// Single structures: every structure count, lane width, and first/last lane.
    structure 1, ld1, { v30.b }[0]
    structure 1, st1, { v30.b }[0]
    structure 1, ld1, { v30.b }[15]
    structure 1, st1, { v30.b }[15]
    structure 2, ld1, { v30.h }[0]
    structure 2, st1, { v30.h }[0]
    structure 2, ld1, { v30.h }[7]
    structure 2, st1, { v30.h }[7]
    structure 4, ld1, { v30.s }[0]
    structure 4, st1, { v30.s }[0]
    structure 4, ld1, { v30.s }[3]
    structure 4, st1, { v30.s }[3]
    structure 8, ld1, { v30.d }[0]
    structure 8, st1, { v30.d }[0]
    structure 8, ld1, { v30.d }[1]
    structure 8, st1, { v30.d }[1]
    structure 2, ld2, { v30.b, v31.b }[0]
    structure 2, st2, { v30.b, v31.b }[0]
    structure 2, ld2, { v30.b, v31.b }[15]
    structure 2, st2, { v30.b, v31.b }[15]
    structure 4, ld2, { v30.h, v31.h }[0]
    structure 4, st2, { v30.h, v31.h }[0]
    structure 4, ld2, { v30.h, v31.h }[7]
    structure 4, st2, { v30.h, v31.h }[7]
    structure 8, ld2, { v30.s, v31.s }[0]
    structure 8, st2, { v30.s, v31.s }[0]
    structure 8, ld2, { v30.s, v31.s }[3]
    structure 8, st2, { v30.s, v31.s }[3]
    structure 16, ld2, { v30.d, v31.d }[0]
    structure 16, st2, { v30.d, v31.d }[0]
    structure 16, ld2, { v30.d, v31.d }[1]
    structure 16, st2, { v30.d, v31.d }[1]
    structure 3, ld3, { v30.b, v31.b, v0.b }[0]
    structure 3, st3, { v30.b, v31.b, v0.b }[0]
    structure 3, ld3, { v30.b, v31.b, v0.b }[15]
    structure 3, st3, { v30.b, v31.b, v0.b }[15]
    structure 6, ld3, { v30.h, v31.h, v0.h }[0]
    structure 6, st3, { v30.h, v31.h, v0.h }[0]
    structure 6, ld3, { v30.h, v31.h, v0.h }[7]
    structure 6, st3, { v30.h, v31.h, v0.h }[7]
    structure 12, ld3, { v30.s, v31.s, v0.s }[0]
    structure 12, st3, { v30.s, v31.s, v0.s }[0]
    structure 12, ld3, { v30.s, v31.s, v0.s }[3]
    structure 12, st3, { v30.s, v31.s, v0.s }[3]
    structure 24, ld3, { v30.d, v31.d, v0.d }[0]
    structure 24, st3, { v30.d, v31.d, v0.d }[0]
    structure 24, ld3, { v30.d, v31.d, v0.d }[1]
    structure 24, st3, { v30.d, v31.d, v0.d }[1]
    structure 4, ld4, { v30.b, v31.b, v0.b, v1.b }[0]
    structure 4, st4, { v30.b, v31.b, v0.b, v1.b }[0]
    structure 4, ld4, { v30.b, v31.b, v0.b, v1.b }[15]
    structure 4, st4, { v30.b, v31.b, v0.b, v1.b }[15]
    structure 8, ld4, { v30.h, v31.h, v0.h, v1.h }[0]
    structure 8, st4, { v30.h, v31.h, v0.h, v1.h }[0]
    structure 8, ld4, { v30.h, v31.h, v0.h, v1.h }[7]
    structure 8, st4, { v30.h, v31.h, v0.h, v1.h }[7]
    structure 16, ld4, { v30.s, v31.s, v0.s, v1.s }[0]
    structure 16, st4, { v30.s, v31.s, v0.s, v1.s }[0]
    structure 16, ld4, { v30.s, v31.s, v0.s, v1.s }[3]
    structure 16, st4, { v30.s, v31.s, v0.s, v1.s }[3]
    structure 32, ld4, { v30.d, v31.d, v0.d, v1.d }[0]
    structure 32, st4, { v30.d, v31.d, v0.d, v1.d }[0]
    structure 32, ld4, { v30.d, v31.d, v0.d, v1.d }[1]
    structure 32, st4, { v30.d, v31.d, v0.d, v1.d }[1]

// Replicating loads cover both vector widths and every element size.
    structure 1, ld1r, { v30.8b }
    structure 1, ld1r, { v30.16b }
    structure 2, ld1r, { v30.4h }
    structure 2, ld1r, { v30.8h }
    structure 4, ld1r, { v30.2s }
    structure 4, ld1r, { v30.4s }
    structure 8, ld1r, { v30.1d }
    structure 8, ld1r, { v30.2d }
    structure 2, ld2r, { v30.8b, v31.8b }
    structure 2, ld2r, { v30.16b, v31.16b }
    structure 4, ld2r, { v30.4h, v31.4h }
    structure 4, ld2r, { v30.8h, v31.8h }
    structure 8, ld2r, { v30.2s, v31.2s }
    structure 8, ld2r, { v30.4s, v31.4s }
    structure 16, ld2r, { v30.1d, v31.1d }
    structure 16, ld2r, { v30.2d, v31.2d }
    structure 3, ld3r, { v30.8b, v31.8b, v0.8b }
    structure 3, ld3r, { v30.16b, v31.16b, v0.16b }
    structure 6, ld3r, { v30.4h, v31.4h, v0.4h }
    structure 6, ld3r, { v30.8h, v31.8h, v0.8h }
    structure 12, ld3r, { v30.2s, v31.2s, v0.2s }
    structure 12, ld3r, { v30.4s, v31.4s, v0.4s }
    structure 24, ld3r, { v30.1d, v31.1d, v0.1d }
    structure 24, ld3r, { v30.2d, v31.2d, v0.2d }
    structure 4, ld4r, { v30.8b, v31.8b, v0.8b, v1.8b }
    structure 4, ld4r, { v30.16b, v31.16b, v0.16b, v1.16b }
    structure 8, ld4r, { v30.4h, v31.4h, v0.4h, v1.4h }
    structure 8, ld4r, { v30.8h, v31.8h, v0.8h, v1.8h }
    structure 16, ld4r, { v30.2s, v31.2s, v0.2s, v1.2s }
    structure 16, ld4r, { v30.4s, v31.4s, v0.4s, v1.4s }
    structure 32, ld4r, { v30.1d, v31.1d, v0.1d, v1.1d }
    structure 32, ld4r, { v30.2d, v31.2d, v0.2d, v1.2d }

// Register post-index can read virtual X18 while another register is base.
    .long 2
    ld1 { v4.16b }, [x3], x18
    ld1 { v4.16b }, [x3], x28
    .long 2
    st2 { v4.s, v5.s }[3], [x3], x18
    st2 { v4.s, v5.s }[3], [x3], x28
    .long 2
    ld4r { v4.2d, v5.2d, v6.2d, v7.2d }, [x3], x18
    ld4r { v4.2d, v5.2d, v6.2d, v7.2d }, [x3], x28

// Base and post-index can name the same guest register: both read its old
// value, then the final base is committed to the virtual register cell.
    .long 1
    ld1 { v5.16b }, [x18], x18
    ld1 { v5.16b }, [x28], x28
    .long 1
    st3 { v5.h, v6.h, v7.h }[7], [x18], x18
    st3 { v5.h, v6.h, v7.h }[7], [x28], x28
    .long 1
    ld2r { v5.2d, v6.2d }, [x18], x18
    ld2r { v5.2d, v6.2d }, [x28], x28

// General/vector copies, sign extension and vector lane preservation.
    .long 0
    ins v18.b[0], w18
    ins v18.b[0], w28
    .long 0
    umov w18, v18.b[0]
    umov w28, v18.b[0]
    .long 0
    smov x18, v18.b[0]
    smov x28, v18.b[0]
    .long 0
    smov w18, v18.b[0]
    smov w28, v18.b[0]
    .long 0
    ins v18.b[15], w18
    ins v18.b[15], w28
    .long 0
    umov w18, v18.b[15]
    umov w28, v18.b[15]
    .long 0
    smov x18, v18.b[15]
    smov x28, v18.b[15]
    .long 0
    smov w18, v18.b[15]
    smov w28, v18.b[15]
    .long 0
    ins v18.h[0], w18
    ins v18.h[0], w28
    .long 0
    umov w18, v18.h[0]
    umov w28, v18.h[0]
    .long 0
    smov x18, v18.h[0]
    smov x28, v18.h[0]
    .long 0
    smov w18, v18.h[0]
    smov w28, v18.h[0]
    .long 0
    ins v18.h[7], w18
    ins v18.h[7], w28
    .long 0
    umov w18, v18.h[7]
    umov w28, v18.h[7]
    .long 0
    smov x18, v18.h[7]
    smov x28, v18.h[7]
    .long 0
    smov w18, v18.h[7]
    smov w28, v18.h[7]
    .long 0
    ins v18.s[0], w18
    ins v18.s[0], w28
    .long 0
    umov w18, v18.s[0]
    umov w28, v18.s[0]
    .long 0
    smov x18, v18.s[0]
    smov x28, v18.s[0]
    .long 0
    ins v18.s[3], w18
    ins v18.s[3], w28
    .long 0
    umov w18, v18.s[3]
    umov w28, v18.s[3]
    .long 0
    smov x18, v18.s[3]
    smov x28, v18.s[3]
    .long 0
    ins v18.d[0], x18
    ins v18.d[0], x28
    .long 0
    umov x18, v18.d[0]
    umov x28, v18.d[0]
    .long 0
    ins v18.d[1], x18
    ins v18.d[1], x28
    .long 0
    umov x18, v18.d[1]
    umov x28, v18.d[1]
    .long 0
    dup v18.8b, w18
    dup v18.8b, w28
    .long 0
    dup v18.16b, w18
    dup v18.16b, w28
    .long 0
    dup v18.4h, w18
    dup v18.4h, w28
    .long 0
    dup v18.8h, w18
    dup v18.8h, w28
    .long 0
    dup v18.2s, w18
    dup v18.2s, w28
    .long 0
    dup v18.4s, w18
    dup v18.4s, w28
    .long 0
    dup v18.2d, x18
    dup v18.2d, x28

// Integer/FP conversion directions, widths, rounding modes and fixed-point limits.
    .long 0
    scvtf h18, w18
    scvtf h18, w28
    .long 0
    scvtf h18, w18, #1
    scvtf h18, w28, #1
    .long 0
    scvtf h18, w18, #32
    scvtf h18, w28, #32
    .long 0
    ucvtf h18, w18
    ucvtf h18, w28
    .long 0
    ucvtf h18, w18, #1
    ucvtf h18, w28, #1
    .long 0
    ucvtf h18, w18, #32
    ucvtf h18, w28, #32
    .long 0
    fcvtns w18, h18
    fcvtns w28, h18
    .long 0
    fcvtnu w18, h18
    fcvtnu w28, h18
    .long 0
    fcvtas w18, h18
    fcvtas w28, h18
    .long 0
    fcvtau w18, h18
    fcvtau w28, h18
    .long 0
    fcvtps w18, h18
    fcvtps w28, h18
    .long 0
    fcvtpu w18, h18
    fcvtpu w28, h18
    .long 0
    fcvtms w18, h18
    fcvtms w28, h18
    .long 0
    fcvtmu w18, h18
    fcvtmu w28, h18
    .long 0
    fcvtzs w18, h18
    fcvtzs w28, h18
    .long 0
    fcvtzu w18, h18
    fcvtzu w28, h18
    .long 0
    fcvtzs w18, h18, #1
    fcvtzs w28, h18, #1
    .long 0
    fcvtzs w18, h18, #32
    fcvtzs w28, h18, #32
    .long 0
    fcvtzu w18, h18, #1
    fcvtzu w28, h18, #1
    .long 0
    fcvtzu w18, h18, #32
    fcvtzu w28, h18, #32
    .long 0
    scvtf h18, x18
    scvtf h18, x28
    .long 0
    scvtf h18, x18, #1
    scvtf h18, x28, #1
    .long 0
    scvtf h18, x18, #64
    scvtf h18, x28, #64
    .long 0
    ucvtf h18, x18
    ucvtf h18, x28
    .long 0
    ucvtf h18, x18, #1
    ucvtf h18, x28, #1
    .long 0
    ucvtf h18, x18, #64
    ucvtf h18, x28, #64
    .long 0
    fcvtns x18, h18
    fcvtns x28, h18
    .long 0
    fcvtnu x18, h18
    fcvtnu x28, h18
    .long 0
    fcvtas x18, h18
    fcvtas x28, h18
    .long 0
    fcvtau x18, h18
    fcvtau x28, h18
    .long 0
    fcvtps x18, h18
    fcvtps x28, h18
    .long 0
    fcvtpu x18, h18
    fcvtpu x28, h18
    .long 0
    fcvtms x18, h18
    fcvtms x28, h18
    .long 0
    fcvtmu x18, h18
    fcvtmu x28, h18
    .long 0
    fcvtzs x18, h18
    fcvtzs x28, h18
    .long 0
    fcvtzu x18, h18
    fcvtzu x28, h18
    .long 0
    fcvtzs x18, h18, #1
    fcvtzs x28, h18, #1
    .long 0
    fcvtzs x18, h18, #64
    fcvtzs x28, h18, #64
    .long 0
    fcvtzu x18, h18, #1
    fcvtzu x28, h18, #1
    .long 0
    fcvtzu x18, h18, #64
    fcvtzu x28, h18, #64
    .long 0
    scvtf s18, w18
    scvtf s18, w28
    .long 0
    scvtf s18, w18, #1
    scvtf s18, w28, #1
    .long 0
    scvtf s18, w18, #32
    scvtf s18, w28, #32
    .long 0
    ucvtf s18, w18
    ucvtf s18, w28
    .long 0
    ucvtf s18, w18, #1
    ucvtf s18, w28, #1
    .long 0
    ucvtf s18, w18, #32
    ucvtf s18, w28, #32
    .long 0
    fcvtns w18, s18
    fcvtns w28, s18
    .long 0
    fcvtnu w18, s18
    fcvtnu w28, s18
    .long 0
    fcvtas w18, s18
    fcvtas w28, s18
    .long 0
    fcvtau w18, s18
    fcvtau w28, s18
    .long 0
    fcvtps w18, s18
    fcvtps w28, s18
    .long 0
    fcvtpu w18, s18
    fcvtpu w28, s18
    .long 0
    fcvtms w18, s18
    fcvtms w28, s18
    .long 0
    fcvtmu w18, s18
    fcvtmu w28, s18
    .long 0
    fcvtzs w18, s18
    fcvtzs w28, s18
    .long 0
    fcvtzu w18, s18
    fcvtzu w28, s18
    .long 0
    fcvtzs w18, s18, #1
    fcvtzs w28, s18, #1
    .long 0
    fcvtzs w18, s18, #32
    fcvtzs w28, s18, #32
    .long 0
    fcvtzu w18, s18, #1
    fcvtzu w28, s18, #1
    .long 0
    fcvtzu w18, s18, #32
    fcvtzu w28, s18, #32
    .long 0
    scvtf s18, x18
    scvtf s18, x28
    .long 0
    scvtf s18, x18, #1
    scvtf s18, x28, #1
    .long 0
    scvtf s18, x18, #64
    scvtf s18, x28, #64
    .long 0
    ucvtf s18, x18
    ucvtf s18, x28
    .long 0
    ucvtf s18, x18, #1
    ucvtf s18, x28, #1
    .long 0
    ucvtf s18, x18, #64
    ucvtf s18, x28, #64
    .long 0
    fcvtns x18, s18
    fcvtns x28, s18
    .long 0
    fcvtnu x18, s18
    fcvtnu x28, s18
    .long 0
    fcvtas x18, s18
    fcvtas x28, s18
    .long 0
    fcvtau x18, s18
    fcvtau x28, s18
    .long 0
    fcvtps x18, s18
    fcvtps x28, s18
    .long 0
    fcvtpu x18, s18
    fcvtpu x28, s18
    .long 0
    fcvtms x18, s18
    fcvtms x28, s18
    .long 0
    fcvtmu x18, s18
    fcvtmu x28, s18
    .long 0
    fcvtzs x18, s18
    fcvtzs x28, s18
    .long 0
    fcvtzu x18, s18
    fcvtzu x28, s18
    .long 0
    fcvtzs x18, s18, #1
    fcvtzs x28, s18, #1
    .long 0
    fcvtzs x18, s18, #64
    fcvtzs x28, s18, #64
    .long 0
    fcvtzu x18, s18, #1
    fcvtzu x28, s18, #1
    .long 0
    fcvtzu x18, s18, #64
    fcvtzu x28, s18, #64
    .long 0
    scvtf d18, w18
    scvtf d18, w28
    .long 0
    scvtf d18, w18, #1
    scvtf d18, w28, #1
    .long 0
    scvtf d18, w18, #32
    scvtf d18, w28, #32
    .long 0
    ucvtf d18, w18
    ucvtf d18, w28
    .long 0
    ucvtf d18, w18, #1
    ucvtf d18, w28, #1
    .long 0
    ucvtf d18, w18, #32
    ucvtf d18, w28, #32
    .long 0
    fcvtns w18, d18
    fcvtns w28, d18
    .long 0
    fcvtnu w18, d18
    fcvtnu w28, d18
    .long 0
    fcvtas w18, d18
    fcvtas w28, d18
    .long 0
    fcvtau w18, d18
    fcvtau w28, d18
    .long 0
    fcvtps w18, d18
    fcvtps w28, d18
    .long 0
    fcvtpu w18, d18
    fcvtpu w28, d18
    .long 0
    fcvtms w18, d18
    fcvtms w28, d18
    .long 0
    fcvtmu w18, d18
    fcvtmu w28, d18
    .long 0
    fcvtzs w18, d18
    fcvtzs w28, d18
    .long 0
    fcvtzu w18, d18
    fcvtzu w28, d18
    .long 0
    fcvtzs w18, d18, #1
    fcvtzs w28, d18, #1
    .long 0
    fcvtzs w18, d18, #32
    fcvtzs w28, d18, #32
    .long 0
    fcvtzu w18, d18, #1
    fcvtzu w28, d18, #1
    .long 0
    fcvtzu w18, d18, #32
    fcvtzu w28, d18, #32
    .long 0
    scvtf d18, x18
    scvtf d18, x28
    .long 0
    scvtf d18, x18, #1
    scvtf d18, x28, #1
    .long 0
    scvtf d18, x18, #64
    scvtf d18, x28, #64
    .long 0
    ucvtf d18, x18
    ucvtf d18, x28
    .long 0
    ucvtf d18, x18, #1
    ucvtf d18, x28, #1
    .long 0
    ucvtf d18, x18, #64
    ucvtf d18, x28, #64
    .long 0
    fcvtns x18, d18
    fcvtns x28, d18
    .long 0
    fcvtnu x18, d18
    fcvtnu x28, d18
    .long 0
    fcvtas x18, d18
    fcvtas x28, d18
    .long 0
    fcvtau x18, d18
    fcvtau x28, d18
    .long 0
    fcvtps x18, d18
    fcvtps x28, d18
    .long 0
    fcvtpu x18, d18
    fcvtpu x28, d18
    .long 0
    fcvtms x18, d18
    fcvtms x28, d18
    .long 0
    fcvtmu x18, d18
    fcvtmu x28, d18
    .long 0
    fcvtzs x18, d18
    fcvtzs x28, d18
    .long 0
    fcvtzu x18, d18
    fcvtzu x28, d18
    .long 0
    fcvtzs x18, d18, #1
    fcvtzs x28, d18, #1
    .long 0
    fcvtzs x18, d18, #64
    fcvtzs x28, d18, #64
    .long 0
    fcvtzu x18, d18, #1
    fcvtzu x28, d18, #1
    .long 0
    fcvtzu x18, d18, #64
    fcvtzu x28, d18, #64
    .long 0
    fmov w18, h18
    fmov w28, h18
    .long 0
    fmov h18, w18
    fmov h18, w28
    .long 0
    fmov x18, h18
    fmov x28, h18
    .long 0
    fmov h18, x18
    fmov h18, x28
    .long 0
    fmov w18, s18
    fmov w28, s18
    .long 0
    fmov s18, w18
    fmov s18, w28
    .long 0
    fmov x18, d18
    fmov x28, d18
    .long 0
    fmov d18, x18
    fmov d18, x28
    .long 0
    fmov x18, v18.d[1]
    fmov x28, v18.d[1]
    .long 0
    fmov v18.d[1], x18
    fmov v18.d[1], x28


// Ordered non-exclusive accesses preserve acquire/release semantics.
    .long 0
    ldarb w18, [x3]
    ldarb w28, [x3]
    .long 1
    ldarb w4, [x18]
    ldarb w4, [x28]
    .long 0
    stlrb w18, [x3]
    stlrb w28, [x3]
    .long 1
    stlrb w4, [x18]
    stlrb w4, [x28]
    .long 0
    ldarh w18, [x3]
    ldarh w28, [x3]
    .long 1
    ldarh w4, [x18]
    ldarh w4, [x28]
    .long 0
    stlrh w18, [x3]
    stlrh w28, [x3]
    .long 1
    stlrh w4, [x18]
    stlrh w4, [x28]
    .long 0
    ldar w18, [x3]
    ldar w28, [x3]
    .long 1
    ldar w4, [x18]
    ldar w4, [x28]
    .long 0
    stlr w18, [x3]
    stlr w28, [x3]
    .long 1
    stlr w4, [x18]
    stlr w4, [x28]
    .long 0
    ldar x18, [x3]
    ldar x28, [x3]
    .long 1
    ldar x4, [x18]
    ldar x4, [x28]
    .long 0
    stlr x18, [x3]
    stlr x28, [x3]
    .long 1
    stlr x4, [x18]
    stlr x4, [x28]

// Comparison aliases still read virtual x18 and only update NZCV.
    .long 0
    cmp w18, #0
    cmp w28, #0
    .long 0
    cmp w18, #4095
    cmp w28, #4095
    .long 0
    cmp w18, #4095, lsl #12
    cmp w28, #4095, lsl #12
    .long 0
    cmp w18, w2
    cmp w28, w2
    .long 0
    ccmp w18, #7, #9, eq
    ccmp w28, #7, #9, eq
    .long 0
    ccmp w18, #7, #9, ne
    ccmp w28, #7, #9, ne
    .long 0
    cmp x18, #0
    cmp x28, #0
    .long 0
    cmp x18, #4095
    cmp x28, #4095
    .long 0
    cmp x18, #4095, lsl #12
    cmp x28, #4095, lsl #12
    .long 0
    cmp x18, x2
    cmp x28, x2
    .long 0
    ccmp x18, #7, #9, eq
    ccmp x28, #7, #9, eq
    .long 0
    ccmp x18, #7, #9, ne
    ccmp x28, #7, #9, ne

    .globl _armigo_test_x18_forms_end
_armigo_test_x18_forms_end:

// Decodable forms whose spill or exclusive contract remains unsupported.
    .globl _armigo_test_x18_refusals
_armigo_test_x18_refusals:
    .long 0
    ld1 { v4.16b }, [sp], x18
    .long 0
    st2 { v4.s, v5.s }[3], [sp], x18
    .long 0
    ld4r { v4.2d, v5.2d, v6.2d, v7.2d }, [sp], x18
    .long 0
    ldar x18, [sp]
    .long 1
    ldxr x18, [x3]
    .long 1
    ldaxr w18, [x3]
    .long 1
    stxr w18, x4, [x3]
    .long 1
    stlxr w4, x18, [x3]
    .long 1
    ldxp x18, x4, [x3]
    .long 1
    ldaxp x4, x18, [x3]
    .long 1
    stxp w4, x5, x6, [x18]
    .long 1
    stlxp w18, x5, x6, [x3]
    .globl _armigo_test_x18_refusals_end
_armigo_test_x18_refusals_end:
