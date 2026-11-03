// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// AArch64 host-gate entry for Apple platforms.
//
// Contract:
//   * Entered with a direct `bl`; x30 holds the return address into the
//     caller and identifies the gate site.
//   * sp must be 16-byte aligned and the caller must have no live data below
//     sp. The frame is allocated on the caller's stack.
//   * Saves x0..x30, the entry sp, q0..q31, NZCV, FPSR and FPCR, then calls
//     the host-stack dispatcher wrapper with a valid frame record.
//   * x18 is the Apple platform register. It is never stored (its slot is
//     written as zero) and never reloaded, so neither the guest nor a handler
//     can observe or change the host's x18 through the gate.
//   * After the dispatcher returns, reloads every other saved register from
//     the frame, so handler changes take effect, and returns to x[30]. An
//     accepted thread exit instead transfers to the registered entry bridge
//     after all C++ dispatcher frames have returned.
//   * Every return to guest code clears the local exclusive monitor after
//     dispatch, so a host reservation is not visible to a guest store-exclusive.
//   * The dispatcher is noexcept; no exception can unwind through this code.
#include <armigo/execution/gate_frame_layout.h>

#define X_SLOT(n) (ARMIGO_GATE_FRAME_X_OFFSET + (n) * 8)
#define V_SLOT(n) (ARMIGO_GATE_FRAME_V_OFFSET + (n) * 16)

    .text
    .globl _armigo_execution_gate_entry
    .p2align 2
_armigo_execution_gate_entry:
#define ARMIGO_GATE_RELOCATABLE 0
#include "gate_body.h"
#undef ARMIGO_GATE_RELOCATABLE

// The copied template contains no external instruction relocations. Its only
// binding is the aligned dispatcher pointer loaded after x16 has been saved.
    .globl _armigo_execution_gate_template_begin
    .globl _armigo_execution_gate_template_dispatcher
    .globl _armigo_execution_gate_template_end
    .private_extern _armigo_execution_gate_template_begin
    .private_extern _armigo_execution_gate_template_dispatcher
    .private_extern _armigo_execution_gate_template_end
    .p2align 3
_armigo_execution_gate_template_begin:
#define ARMIGO_GATE_RELOCATABLE 1
#include "gate_body.h"
#undef ARMIGO_GATE_RELOCATABLE
    .p2align 3
_armigo_execution_gate_template_dispatcher:
L_armigo_gate_dispatch_address:
    .quad 0
_armigo_execution_gate_template_end:

// The embedded template loads its dispatcher from a separately writable key
// page. Only its two PC-relative address words are encoded at pack time.
    .globl _armigo_execution_embedded_gate_begin
    .globl _armigo_execution_embedded_gate_load
    .globl _armigo_execution_embedded_gate_end
    .private_extern _armigo_execution_embedded_gate_begin
    .private_extern _armigo_execution_embedded_gate_load
    .private_extern _armigo_execution_embedded_gate_end
    .p2align 3
_armigo_execution_embedded_gate_begin:
#define ARMIGO_GATE_RELOCATABLE 2
#include "gate_body.h"
#undef ARMIGO_GATE_RELOCATABLE
    .p2align 3
_armigo_execution_embedded_gate_end:

// The gate frame remains on the guest stack; only this bounded save area and
// the small TLS accessor run below it. C++ dispatch runs below the entry
// bridge's saved host frame. x19 retains the guest SP across that C++ call.
    .globl _armigo_execution_gate_dispatch_on_host_stack
    .private_extern _armigo_execution_gate_dispatch_on_host_stack
    .p2align 2
_armigo_execution_gate_dispatch_on_host_stack:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, sp
    mov x20, x0
    bl _armigo_execution_gate_callback_stack
    cbz x0, L_armigo_callback_stack_selected
    tst x0, #15
    b.ne L_armigo_callback_stack_invalid
    mov sp, x0
L_armigo_callback_stack_selected:
    mov x0, x20
    bl _armigo_execution_gate_dispatch
    mov sp, x19
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
L_armigo_callback_stack_invalid:
    brk #0xae15
    b L_armigo_callback_stack_invalid
