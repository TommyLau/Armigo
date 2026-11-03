// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Native readers in the shape generated code uses: read TPIDRRO_EL0 unmasked,
// load the slot, then access caller storage through the loaded pointer. The
// slot offset arrives in a register only because it is chosen at run time.
// None of these routines writes x18, TPIDR_EL0, TPIDRRO_EL0, or the slot.

    .text
    .p2align 2

// uint64_t armigo_test_thread_slot_load(uint64_t tsd_byte_offset)
    .globl _armigo_test_thread_slot_load
_armigo_test_thread_slot_load:
    mrs x16, TPIDRRO_EL0
    ldr x0, [x16, x0]
    ret

// uint64_t armigo_test_thread_slot_load_word(uint64_t tsd_byte_offset, uint64_t index)
    .globl _armigo_test_thread_slot_load_word
_armigo_test_thread_slot_load_word:
    mrs x16, TPIDRRO_EL0
    ldr x16, [x16, x0]
    ldr x0, [x16, x1, lsl #3]
    ret

// void armigo_test_thread_slot_store_word(uint64_t tsd_byte_offset, uint64_t index,
//                                         uint64_t value)
    .globl _armigo_test_thread_slot_store_word
_armigo_test_thread_slot_store_word:
    mrs x16, TPIDRRO_EL0
    ldr x16, [x16, x0]
    str x2, [x16, x1, lsl #3]
    ret
