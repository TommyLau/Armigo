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

#include <armigo/execution/guest_dispatch.h>

#include <array>
#include <optional>

namespace armigo::execution {

struct GateCodeRegion {
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

enum class GatePatchError {
    none,
    invalid_layout,
    overlapping_regions,
    unsupported_site,
    unreachable_branch,
};

struct GateSitePatch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    std::array<std::uint32_t, 4> trampoline{};
    GuestGateSite dispatch_site;

    // The wrapper saves guest x30 on the stack while the gate uses x30 for
    // its own return address. The original guest sp is GateCall::sp() + 16.
    static constexpr std::uint64_t stack_adjustment = 16;
};

struct GatePatchResult {
    GatePatchError error = GatePatchError::none;
    GuestSiteTableError site_error;
    std::optional<GateSitePatch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Builds one system-instruction replacement without changing code or executing
// it. Addresses must all use the same namespace: absolute addresses, or offsets
// in one image that will retain the same relative layout when mapped.
//
// The caller owns unused trampoline storage and a valid module-local gate
// region. The original instruction and its continuation word must be disjoint
// from both injected regions, which must also be disjoint. Before applying, verify the
// instruction preimage and the complete module preparation plan; write only
// to nonexecutable memory and seal afterward. This operation neither finds
// sites nor proves other instructions are host-compatible.
//
// The trampoline preserves the original x30 and sp, with no other scratch
// register. It needs 16 bytes of stack headroom in addition to the gate.
// Dispatch handlers must retain the gate return address. Guest x30 operands
// remain unsupported, and the wrapper may not split an exclusive sequence.
// Allocation failure from site validation propagates to the caller.
GatePatchResult make_gate_site_patch(std::uint32_t instruction,
                                    std::uint64_t instruction_address,
                                    std::uint64_t trampoline_address,
                                    GateCodeRegion gate,
                                    const GuestSiteTableOptions& options = {});

} // namespace armigo::execution
