// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic exact-SP thread entry. The test copies this
// code into an NSO, replaces every privileged instruction with the local gate,
// and executes only the sealed prepared copy.
    .text
    .p2align 2
    .globl _armigo_thread_runtime_code_begin
    .globl _armigo_thread_runtime_code_end
    .globl _armigo_thread_runtime_entry
    .globl _armigo_thread_runtime_tpidr
    .globl _armigo_thread_runtime_tpidrro
    .globl _armigo_thread_runtime_first_svc
    .globl _armigo_thread_runtime_second_svc
    .globl _armigo_thread_runtime_exit_svc
    .globl _armigo_thread_runtime_processor_loop
    .globl _armigo_thread_runtime_processor_svc
    .globl _armigo_thread_runtime_scheduling_entry
    .globl _armigo_thread_runtime_get_priority_svc
    .globl _armigo_thread_runtime_set_priority_svc
    .globl _armigo_thread_runtime_get_core_svc
    .globl _armigo_thread_runtime_set_core_svc
    .globl _armigo_thread_runtime_current_processor_svc
    .globl _armigo_thread_runtime_current_priority_svc
    .globl _armigo_thread_runtime_original_entry
    .globl _armigo_thread_runtime_original_first_svc
    .globl _armigo_thread_runtime_original_exit_svc
    .globl _armigo_thread_runtime_original_worker_entry
    .globl _armigo_thread_runtime_original_worker_svc

_armigo_thread_runtime_code_begin:
_armigo_thread_runtime_entry:
    mov x19, x0
    // Optional native handshake, using acquire/release atomics. A peer guest
    // can open the release flag without either thread crossing an SVC boundary.
    ldr x20, [x19, #40]
    cbz x20, 2f
    mov w9, #1
    stlr w9, [x20]
    add x21, x20, #4
1:
    ldar w9, [x21]
    cbz w9, 1b
    ldr x21, [x20, #8]
    cbz x21, 2f
    mov w9, #1
    stlr w9, [x21]
2:
    mov x9, sp
    str x9, [x19]
_armigo_thread_runtime_tpidr:
    mrs x9, TPIDR_EL0
    str x9, [x19, #8]
_armigo_thread_runtime_tpidrro:
    mrs x9, TPIDRRO_EL0
    str x9, [x19, #16]
_armigo_thread_runtime_first_svc:
    svc #0x31
    str x0, [x19, #24]
_armigo_thread_runtime_second_svc:
    svc #0x32
    str x0, [x19, #32]
_armigo_thread_runtime_exit_svc:
    svc #0xa
    brk #0

_armigo_thread_runtime_processor_loop:
    mov x19, x0
    ldr x20, [x19, #40]
    mov w9, #1
    stlr w9, [x20]
1:
_armigo_thread_runtime_processor_svc:
    svc #0x10
    b 1b

_armigo_thread_runtime_scheduling_entry:
    mov x19, x0
    ldr x1, [x19]
_armigo_thread_runtime_get_priority_svc:
    svc #0x0c
    str x0, [x19, #32]
    str x1, [x19, #40]
    ldr x0, [x19]
    ldr x1, [x19, #8]
_armigo_thread_runtime_set_priority_svc:
    svc #0x0d
    str x0, [x19, #48]
    ldr x1, [x19, #16]
    ldr x2, [x19]
_armigo_thread_runtime_get_core_svc:
    svc #0x0e
    str x0, [x19, #56]
    str x1, [x19, #64]
    str x2, [x19, #72]
    ldr x0, [x19]
    ldr x1, [x19, #16]
    ldr x2, [x19, #24]
_armigo_thread_runtime_set_core_svc:
    svc #0x0f
    str x0, [x19, #80]
_armigo_thread_runtime_current_processor_svc:
    svc #0x10
    str x0, [x19, #88]
    mov w1, #0x8000
    movk w1, #0xffff, lsl #16
_armigo_thread_runtime_current_priority_svc:
    svc #0x0c
    str x0, [x19, #96]
    str x1, [x19, #104]
    b _armigo_thread_runtime_exit_svc

// Original-process entry smoke test. Lane D must replace caller-supplied x0/x1
// with zero and the allocated nonzero main-thread handle before execution.
_armigo_thread_runtime_original_entry:
_armigo_thread_runtime_original_first_svc:
    svc #0x31
_armigo_thread_runtime_original_exit_svc:
    svc #0xa
    brk #0

// Preserve the firmware LR and exact SP in the worker's x0 report before the
// first gate. The stores do not alter the remaining observed register state.
_armigo_thread_runtime_original_worker_entry:
    str x30, [x0]
    mov x30, sp
    str x30, [x0, #8]
_armigo_thread_runtime_original_worker_svc:
    svc #0x33
    b _armigo_thread_runtime_original_exit_svc
_armigo_thread_runtime_code_end:
