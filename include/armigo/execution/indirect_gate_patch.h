// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/gate_patch.h>

namespace armigo::execution {

// Builds a checked native BR/BLR without changing or executing any code.
// The 16-byte stub saves x16/x30, calls the local gate, restores x16/x30 and
// branches through the original target register. A BLR site becomes BL stub,
// preserving its site+4 link; a BR site becomes B stub, preserving its old link.
// The dispatch return address is stub+8 and its instruction is the original word.
//
// Target declarations are not evidence of preparation. The caller must prepare
// every declared target and native continuation, validate the instruction
// preimage, own the injected storage and gate, and seal code after writing.
// All addresses (including targets) use one namespace; rebase table addresses
// and targets together before dispatch. Target registers x18/x30/x31 are refused.
// The site and its next instruction must not overlap either injected region.
//
// The stub requires 16 bytes of writable stack headroom plus the gate's stack
// requirement. Use only an owned guarded stack and never split an exclusive
// window. GateCall::sp() is the original sp minus 16. A table containing these
// sites requires GuestDispatchScope's trap failure policy; failure cannot resume
// to the native branch. Allocation failure propagates to the caller.
GatePatchResult make_indirect_gate_site_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, GateCodeRegion gate,
    std::span<const std::uint64_t> allowed_indirect_targets);

} // namespace armigo::execution
