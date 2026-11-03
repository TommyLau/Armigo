// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/indirect_gate_patch.h>

#include <limits>

namespace armigo::execution {
namespace {

bool valid_range(GateCodeRegion region) noexcept {
    return region.size != 0 && region.size - 1 <=
        std::numeric_limits<std::uint64_t>::max() - region.address;
}

bool overlap(GateCodeRegion left, GateCodeRegion right) noexcept {
    return left.address <= right.address + (right.size - 1) &&
           right.address <= left.address + (left.size - 1);
}

} // namespace

GatePatchResult make_indirect_gate_site_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    GateCodeRegion gate, std::span<const std::uint64_t> allowed_targets) {
    const GateCodeRegion site{address, 8};
    const GateCodeRegion wrapper{trampoline, 16};
    if (address % 4 != 0 || trampoline % 4 != 0 || gate.address % 8 != 0 ||
        gate.size % 8 != 0 || !valid_range(site) || !valid_range(wrapper) ||
        !valid_range(gate)) {
        return {.error = GatePatchError::invalid_layout};
    }
    if (overlap(site, wrapper) || overlap(site, gate) || overlap(wrapper, gate)) {
        return {.error = GatePatchError::overlapping_regions};
    }
    const auto decoded = aarch64::decode(instruction);
    if (decoded.opcode != aarch64::Opcode::br && decoded.opcode != aarch64::Opcode::blr) {
        return {.error = GatePatchError::unsupported_site,
                .site_error = {.kind = GuestSiteTableErrorKind::unsupported_instruction,
                               .return_address = trampoline + 8, .instruction = instruction}};
    }
    const std::array sites = {GuestGateSite{trampoline + 8, instruction,
        std::vector<std::uint64_t>(allowed_targets.begin(), allowed_targets.end())}};
    const auto validated = GuestSiteTable::build(sites);
    if (!validated.ok()) {
        return {.error = GatePatchError::unsupported_site, .site_error = validated.error};
    }
    const auto redirect = decoded.opcode == aarch64::Opcode::blr
        ? aarch64::encode_bl(address, trampoline) : aarch64::encode_b(address, trampoline);
    const auto call = aarch64::encode_bl(trampoline + 4, gate.address);
    const auto transfer = aarch64::encode_br(decoded.reg);
    if (!redirect.ok() || !call.ok() || !transfer.ok()) {
        return {.error = GatePatchError::unreachable_branch};
    }
    // STP x16,x30,[sp,#-16]!; BL gate; LDP x16,x30,[sp],#16; BR original register.
    // x16 stays unchanged until captured by the gate, including when it is the target.
    GateSitePatch patch{address, trampoline, instruction, redirect.word,
                       {0xa9bf7bf0u, call.word, 0xa8c17bf0u, transfer.word}, sites[0]};
    return {.patch = std::move(patch)};
}

} // namespace armigo::execution
