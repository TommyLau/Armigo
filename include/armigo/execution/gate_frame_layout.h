// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Saved-frame layout shared by the AArch64 host-gate assembly and the C++
// dispatcher. This header must stay preprocessor-only so that the assembler
// (`.S` sources are run through the C preprocessor) and C++ see one source of
// truth. `gate_frame.h` static_asserts the C++ structure against these
// values, and `gate_entry.S` uses them for every store and load.
//
// All offsets are in bytes from the frame base, which is the stack pointer
// after the gate has allocated the frame. The frame base is 16-byte aligned.
#ifndef ARMIGO_EXECUTION_GATE_FRAME_LAYOUT_H
#define ARMIGO_EXECUTION_GATE_FRAME_LAYOUT_H

// x0..x30 as 31 consecutive 64-bit slots. Slot 18 is the Apple platform
// register: the gate stores zero there and never reads the slot back.
#define ARMIGO_GATE_FRAME_X_OFFSET 0
#define ARMIGO_GATE_FRAME_X_COUNT 31

// The caller's stack pointer at the moment of the `bl` (frame base + size).
#define ARMIGO_GATE_FRAME_SP_OFFSET 248

// q0..q31 as 32 consecutive 128-bit slots.
#define ARMIGO_GATE_FRAME_V_OFFSET 256
#define ARMIGO_GATE_FRAME_V_COUNT 32

// Status registers, each stored as a 64-bit value from `mrs`.
#define ARMIGO_GATE_FRAME_NZCV_OFFSET 768
#define ARMIGO_GATE_FRAME_FPSR_OFFSET 776
#define ARMIGO_GATE_FRAME_FPCR_OFFSET 784

// A marker written by the assembly so the dispatcher can reject frames that
// did not come through the gate entry.
#define ARMIGO_GATE_FRAME_MAGIC_OFFSET 792
#define ARMIGO_GATE_FRAME_MAGIC_HW0 0x5445
#define ARMIGO_GATE_FRAME_MAGIC_HW1 0x4741
#define ARMIGO_GATE_FRAME_MAGIC_HW2 0x4D49
#define ARMIGO_GATE_FRAME_MAGIC_HW3 0x4152
#define ARMIGO_GATE_FRAME_MAGIC 0x41524D4947415445

// An AAPCS64 frame record {x29, x30} copied from the caller so host
// backtraces can walk through the gate while the dispatcher runs.
#define ARMIGO_GATE_FRAME_RECORD_OFFSET 800

#define ARMIGO_GATE_FRAME_SIZE 816

#endif
