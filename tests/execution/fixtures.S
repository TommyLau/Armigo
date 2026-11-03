// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic guest-like fixture for exercising the AArch64 host gate.
//
// armigo_test_fixture_run_gate(const FixtureImage* in, FixtureImage* out)
// loads x0..x17, x19..x29, q0..q31, NZCV, FPSR and FPCR from the input
// image, enters the gate with a direct `bl`, and after the gate returns
// records every general register except x18, every vector register, the
// status registers and host facts (hardware x18 and sp before and after) into
// the output image. The fixture is an AAPCS64 function: it preserves the
// caller's callee-saved registers and restores the host FPCR/FPSR before
// returning. It never writes the Apple platform register x18.
//
// The image is a GateFrame followed by the fixture-specific fields below;
// gate_tests.cpp static_asserts the offsets.
#include <armigo/execution/gate_frame_layout.h>

#define X_SLOT(n) (ARMIGO_GATE_FRAME_X_OFFSET + (n) * 8)
#define V_SLOT(n) (ARMIGO_GATE_FRAME_V_OFFSET + (n) * 16)

#define FIXTURE_X18_BEFORE (ARMIGO_GATE_FRAME_SIZE + 0)
#define FIXTURE_SP_BEFORE (ARMIGO_GATE_FRAME_SIZE + 8)
#define FIXTURE_SP_AFTER (ARMIGO_GATE_FRAME_SIZE + 16)
#define FIXTURE_RETURN_LABEL (ARMIGO_GATE_FRAME_SIZE + 24)
#define FIXTURE_ALTERNATE_MARKER (ARMIGO_GATE_FRAME_SIZE + 32)
#define FIXTURE_X18_AFTER (ARMIGO_GATE_FRAME_SIZE + 40)

// Fixture stack frame.
#define FRAME_SIZE 208
#define FRAME_X19 16
#define FRAME_D8 96
#define FRAME_IN 160
#define FRAME_OUT 168
#define FRAME_X0_TEMP 176
#define FRAME_HOST_FPCR 184
#define FRAME_HOST_FPSR 192

    .text
    .globl _armigo_test_fixture_run_gate
    .p2align 2
_armigo_test_fixture_run_gate:
    stp x29, x30, [sp, #-FRAME_SIZE]!
    mov x29, sp
    stp x19, x20, [sp, #(FRAME_X19 + 0)]
    stp x21, x22, [sp, #(FRAME_X19 + 16)]
    stp x23, x24, [sp, #(FRAME_X19 + 32)]
    stp x25, x26, [sp, #(FRAME_X19 + 48)]
    stp x27, x28, [sp, #(FRAME_X19 + 64)]
    stp d8, d9, [sp, #(FRAME_D8 + 0)]
    stp d10, d11, [sp, #(FRAME_D8 + 16)]
    stp d12, d13, [sp, #(FRAME_D8 + 32)]
    stp d14, d15, [sp, #(FRAME_D8 + 48)]
    stp x0, x1, [sp, #FRAME_IN]
    mrs x9, fpcr
    str x9, [sp, #FRAME_HOST_FPCR]
    mrs x9, fpsr
    str x9, [sp, #FRAME_HOST_FPSR]

    // Host facts observed immediately before the gate.
    str x18, [x1, #FIXTURE_X18_BEFORE]
    mov x9, sp
    str x9, [x1, #FIXTURE_SP_BEFORE]

    // Status registers from the input image.
    ldr x9, [x0, #ARMIGO_GATE_FRAME_FPCR_OFFSET]
    msr fpcr, x9
    ldr x9, [x0, #ARMIGO_GATE_FRAME_FPSR_OFFSET]
    msr fpsr, x9
    ldr x9, [x0, #ARMIGO_GATE_FRAME_NZCV_OFFSET]
    msr nzcv, x9

    ldp q0, q1, [x0, #V_SLOT(0)]
    ldp q2, q3, [x0, #V_SLOT(2)]
    ldp q4, q5, [x0, #V_SLOT(4)]
    ldp q6, q7, [x0, #V_SLOT(6)]
    ldp q8, q9, [x0, #V_SLOT(8)]
    ldp q10, q11, [x0, #V_SLOT(10)]
    ldp q12, q13, [x0, #V_SLOT(12)]
    ldp q14, q15, [x0, #V_SLOT(14)]
    ldp q16, q17, [x0, #V_SLOT(16)]
    ldp q18, q19, [x0, #V_SLOT(18)]
    ldp q20, q21, [x0, #V_SLOT(20)]
    ldp q22, q23, [x0, #V_SLOT(22)]
    ldp q24, q25, [x0, #V_SLOT(24)]
    ldp q26, q27, [x0, #V_SLOT(26)]
    ldp q28, q29, [x0, #V_SLOT(28)]
    ldp q30, q31, [x0, #V_SLOT(30)]

    // General registers from the image: x0 last because it is the base,
    // x18 never (platform register), x30 is produced by the `bl` below.
    ldp x2, x3, [x0, #X_SLOT(2)]
    ldp x4, x5, [x0, #X_SLOT(4)]
    ldp x6, x7, [x0, #X_SLOT(6)]
    ldp x8, x9, [x0, #X_SLOT(8)]
    ldp x10, x11, [x0, #X_SLOT(10)]
    ldp x12, x13, [x0, #X_SLOT(12)]
    ldp x14, x15, [x0, #X_SLOT(14)]
    ldp x16, x17, [x0, #X_SLOT(16)]
    ldr x19, [x0, #X_SLOT(19)]
    ldp x20, x21, [x0, #X_SLOT(20)]
    ldp x22, x23, [x0, #X_SLOT(22)]
    ldp x24, x25, [x0, #X_SLOT(24)]
    ldp x26, x27, [x0, #X_SLOT(26)]
    ldp x28, x29, [x0, #X_SLOT(28)]
    ldr x1, [x0, #X_SLOT(1)]
    ldr x0, [x0, #X_SLOT(0)]

    bl _armigo_execution_gate_entry
    .globl _armigo_test_fixture_gate_return
_armigo_test_fixture_gate_return:
    str x0, [sp, #FRAME_X0_TEMP]
    ldr x0, [sp, #FRAME_OUT]
    b Lcapture

    // Alternate return point. A handler that writes this address into x30
    // redirects the gate's return here; the marker proves the path was taken.
    .globl _armigo_test_fixture_alternate_return
_armigo_test_fixture_alternate_return:
    str x0, [sp, #FRAME_X0_TEMP]
    ldr x0, [sp, #FRAME_OUT]
    str x0, [x0, #FIXTURE_ALTERNATE_MARKER]

Lcapture:
    // Nothing between the gate's `ret` and the `mrs nzcv` below writes the
    // flags: stores, loads, `mov` and `adr` leave NZCV untouched.
    str x1, [x0, #X_SLOT(1)]
    stp x2, x3, [x0, #X_SLOT(2)]
    stp x4, x5, [x0, #X_SLOT(4)]
    stp x6, x7, [x0, #X_SLOT(6)]
    stp x8, x9, [x0, #X_SLOT(8)]
    stp x10, x11, [x0, #X_SLOT(10)]
    stp x12, x13, [x0, #X_SLOT(12)]
    stp x14, x15, [x0, #X_SLOT(14)]
    stp x16, x17, [x0, #X_SLOT(16)]
    str x19, [x0, #X_SLOT(19)]
    stp x20, x21, [x0, #X_SLOT(20)]
    stp x22, x23, [x0, #X_SLOT(22)]
    stp x24, x25, [x0, #X_SLOT(24)]
    stp x26, x27, [x0, #X_SLOT(26)]
    stp x28, x29, [x0, #X_SLOT(28)]
    str x30, [x0, #X_SLOT(30)]
    str x18, [x0, #FIXTURE_X18_AFTER]
    mrs x9, nzcv
    str x9, [x0, #ARMIGO_GATE_FRAME_NZCV_OFFSET]
    mrs x9, fpsr
    str x9, [x0, #ARMIGO_GATE_FRAME_FPSR_OFFSET]
    mrs x9, fpcr
    str x9, [x0, #ARMIGO_GATE_FRAME_FPCR_OFFSET]
    mov x9, sp
    str x9, [x0, #FIXTURE_SP_AFTER]

    stp q0, q1, [x0, #V_SLOT(0)]
    stp q2, q3, [x0, #V_SLOT(2)]
    stp q4, q5, [x0, #V_SLOT(4)]
    stp q6, q7, [x0, #V_SLOT(6)]
    stp q8, q9, [x0, #V_SLOT(8)]
    stp q10, q11, [x0, #V_SLOT(10)]
    stp q12, q13, [x0, #V_SLOT(12)]
    stp q14, q15, [x0, #V_SLOT(14)]
    stp q16, q17, [x0, #V_SLOT(16)]
    stp q18, q19, [x0, #V_SLOT(18)]
    stp q20, q21, [x0, #V_SLOT(20)]
    stp q22, q23, [x0, #V_SLOT(22)]
    stp q24, q25, [x0, #V_SLOT(24)]
    stp q26, q27, [x0, #V_SLOT(26)]
    stp q28, q29, [x0, #V_SLOT(28)]
    stp q30, q31, [x0, #V_SLOT(30)]

    ldr x9, [sp, #FRAME_X0_TEMP]
    str x9, [x0, #X_SLOT(0)]
    adr x9, _armigo_test_fixture_gate_return
    str x9, [x0, #FIXTURE_RETURN_LABEL]

    // Restore the host's floating-point control state and callee-saved
    // registers.
    ldr x9, [sp, #FRAME_HOST_FPCR]
    msr fpcr, x9
    ldr x9, [sp, #FRAME_HOST_FPSR]
    msr fpsr, x9
    ldp x19, x20, [sp, #(FRAME_X19 + 0)]
    ldp x21, x22, [sp, #(FRAME_X19 + 16)]
    ldp x23, x24, [sp, #(FRAME_X19 + 32)]
    ldp x25, x26, [sp, #(FRAME_X19 + 48)]
    ldp x27, x28, [sp, #(FRAME_X19 + 64)]
    ldp d8, d9, [sp, #(FRAME_D8 + 0)]
    ldp d10, d11, [sp, #(FRAME_D8 + 16)]
    ldp d12, d13, [sp, #(FRAME_D8 + 32)]
    ldp d14, d15, [sp, #(FRAME_D8 + 48)]
    ldp x29, x30, [sp], #FRAME_SIZE
    ret
