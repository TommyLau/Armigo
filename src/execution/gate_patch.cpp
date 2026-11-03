// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate_patch.h>

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

extern "C" {
extern const std::byte armigo_gate_site_template_begin[];
extern const std::byte armigo_gate_site_template_call[];
extern const std::byte armigo_gate_site_template_resume[];
extern const std::byte armigo_gate_site_template_end[];
}

namespace armigo::execution {
namespace {

bool valid_range(GateCodeRegion region) {
    return region.size != 0 && region.size - 1 <=
        std::numeric_limits<std::uint64_t>::max() - region.address;
}

bool overlap(GateCodeRegion left, GateCodeRegion right) {
    return left.address <= right.address + (right.size - 1) &&
           right.address <= left.address + (left.size - 1);
}

} // namespace

GatePatchResult make_gate_site_patch(std::uint32_t instruction,
                                    std::uint64_t address,
                                    std::uint64_t trampoline,
                                    GateCodeRegion gate,
                                    const GuestSiteTableOptions& options) {
    static_assert(std::endian::native == std::endian::little);
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    // The next instruction must remain original code, not injected storage.
    const GateCodeRegion site{address, 8};
    const GateCodeRegion wrapper{trampoline, 16};
    if (address % 4 != 0 || trampoline % 4 != 0 || gate.address % 8 != 0 ||
        gate.size % 8 != 0 || !valid_range(gate) || !valid_range(wrapper) ||
        address > maximum - 4) {
        return {.error = GatePatchError::invalid_layout};
    }
    if (overlap(site, wrapper) || overlap(site, gate) || overlap(wrapper, gate)) {
        return {.error = GatePatchError::overlapping_regions};
    }
    const std::array dispatch_sites = {GuestGateSite{trampoline + 8, instruction}};
    const auto validated = GuestSiteTable::build(dispatch_sites, options);
    if (!validated.ok()) {
        return {.error = GatePatchError::unsupported_site, .site_error = validated.error};
    }
    const auto redirect = aarch64::encode_b(address, trampoline);
    const auto call = aarch64::encode_bl(trampoline + 4, gate.address);
    const auto resume = aarch64::encode_b(trampoline + 12, address + 4);
    if (!redirect.ok() || !call.ok() || !resume.ok()) {
        return {.error = GatePatchError::unreachable_branch};
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_gate_site_template_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_gate_site_template_end);
    if (end < begin || end - begin != 16 ||
        reinterpret_cast<std::uintptr_t>(armigo_gate_site_template_call) - begin != 4 ||
        reinterpret_cast<std::uintptr_t>(armigo_gate_site_template_resume) - begin != 12) {
        throw std::logic_error("The gate-site template has an invalid linked layout.");
    }
    GateSitePatch patch{address, trampoline, instruction, redirect.word, {}, dispatch_sites[0]};
    std::memcpy(patch.trampoline.data(), armigo_gate_site_template_begin, 16);
    patch.trampoline[1] = call.word;
    patch.trampoline[3] = resume.word;
    return {.patch = patch};
}

} // namespace armigo::execution
