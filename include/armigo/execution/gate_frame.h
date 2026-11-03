// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Typed view of the register frame saved by the AArch64 host gate.
#pragma once

#include <armigo/execution/gate_frame_layout.h>

#include <cstddef>
#include <cstdint>

namespace armigo::execution {

// One 128-bit SIMD/FP register (q0..q31) stored as two little-endian lanes.
struct alignas(16) GateVector {
    std::uint64_t lane[2];

    friend constexpr bool operator==(const GateVector& left, const GateVector& right) noexcept {
        return left.lane[0] == right.lane[0] && left.lane[1] == right.lane[1];
    }
};

// Register indices with a fixed meaning for the gate.
inline constexpr unsigned kGateHostReservedRegister = 18;  // Apple platform register.
inline constexpr unsigned kGateFramePointerRegister = 29;
inline constexpr unsigned kGateLinkRegister = 30;
inline constexpr unsigned kGateGeneralRegisterCount = ARMIGO_GATE_FRAME_X_COUNT;
inline constexpr unsigned kGateVectorRegisterCount = ARMIGO_GATE_FRAME_V_COUNT;

// The frame written by `armigo_execution_gate_entry` on the caller's stack.
//
// The assembly writes every field; the dispatcher reads them and the
// assembly reloads x0..x17, x19..x30, v0..v31 and the status registers from
// the frame before returning. `x[18]` is always zero and is never reloaded:
// the host's x18 is neither captured nor writable through this frame. `sp`
// and `frame_record` are informational and are not reloaded either.
struct alignas(16) GateFrame {
    std::uint64_t x[kGateGeneralRegisterCount];
    std::uint64_t sp;
    GateVector v[kGateVectorRegisterCount];
    std::uint64_t nzcv;
    std::uint64_t fpsr;
    std::uint64_t fpcr;
    std::uint64_t magic;
    std::uint64_t frame_record[2];
};

inline constexpr std::uint64_t kGateFrameMagic = ARMIGO_GATE_FRAME_MAGIC;

static_assert(sizeof(GateVector) == 16 && alignof(GateVector) == 16);
static_assert(sizeof(GateFrame) == ARMIGO_GATE_FRAME_SIZE);
static_assert(alignof(GateFrame) == 16);
static_assert(ARMIGO_GATE_FRAME_SIZE % 16 == 0, "The gate frame must keep sp 16-byte aligned.");
static_assert(offsetof(GateFrame, x) == ARMIGO_GATE_FRAME_X_OFFSET);
static_assert(sizeof(GateFrame::x) == 8 * ARMIGO_GATE_FRAME_X_COUNT);
static_assert(offsetof(GateFrame, sp) == ARMIGO_GATE_FRAME_SP_OFFSET);
static_assert(offsetof(GateFrame, v) == ARMIGO_GATE_FRAME_V_OFFSET);
static_assert(sizeof(GateFrame::v) == 16 * ARMIGO_GATE_FRAME_V_COUNT);
static_assert(offsetof(GateFrame, nzcv) == ARMIGO_GATE_FRAME_NZCV_OFFSET);
static_assert(offsetof(GateFrame, fpsr) == ARMIGO_GATE_FRAME_FPSR_OFFSET);
static_assert(offsetof(GateFrame, fpcr) == ARMIGO_GATE_FRAME_FPCR_OFFSET);
static_assert(offsetof(GateFrame, magic) == ARMIGO_GATE_FRAME_MAGIC_OFFSET);
static_assert(offsetof(GateFrame, frame_record) == ARMIGO_GATE_FRAME_RECORD_OFFSET);
static_assert(kGateFrameMagic ==
              (std::uint64_t{ARMIGO_GATE_FRAME_MAGIC_HW3} << 48 |
               std::uint64_t{ARMIGO_GATE_FRAME_MAGIC_HW2} << 32 |
               std::uint64_t{ARMIGO_GATE_FRAME_MAGIC_HW1} << 16 |
               std::uint64_t{ARMIGO_GATE_FRAME_MAGIC_HW0}),
              "The assembly builds the magic from these four halfwords.");

}  // namespace armigo::execution
