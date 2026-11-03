// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

    .text
    .p2align 2
    .globl _armigo_gate_site_template_begin
    .globl _armigo_gate_site_template_call
    .globl _armigo_gate_site_template_resume
    .globl _armigo_gate_site_template_end
    .private_extern _armigo_gate_site_template_begin
    .private_extern _armigo_gate_site_template_call
    .private_extern _armigo_gate_site_template_resume
    .private_extern _armigo_gate_site_template_end
_armigo_gate_site_template_begin:
    str x30, [sp, #-16]!
_armigo_gate_site_template_call:
    bl L_gate_site_call_placeholder
L_gate_site_call_placeholder:
    ldr x30, [sp], #16
_armigo_gate_site_template_resume:
    b L_gate_site_resume_placeholder
L_gate_site_resume_placeholder:
_armigo_gate_site_template_end:
