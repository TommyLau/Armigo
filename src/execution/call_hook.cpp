// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/call_hook.h>

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/relocation.h>

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace armigo::execution {
namespace {

bool valid_range(CallHookCodeRegion region) noexcept {
    return region.size != 0 && region.size - 1 <=
        std::numeric_limits<std::uint64_t>::max() - region.address;
}

bool overlap(CallHookCodeRegion left, CallHookCodeRegion right) noexcept {
    return left.address <= right.address + (right.size - 1) &&
           right.address <= left.address + (left.size - 1);
}

} // namespace

const char* describe(CallHookTableErrorKind kind) noexcept {
    switch (kind) {
    case CallHookTableErrorKind::none: return "no call-hook table error";
    case CallHookTableErrorKind::too_many_sites: return "too many call-hook sites";
    case CallHookTableErrorKind::null_return_address: return "call-hook return address is null";
    case CallHookTableErrorKind::misaligned_return_address: return "call-hook return address is misaligned";
    case CallHookTableErrorKind::invalid_call_return_address: return "call-hook guest return address is invalid";
    case CallHookTableErrorKind::invalid_target_address: return "call-hook target address is invalid";
    case CallHookTableErrorKind::duplicate_return_address: return "duplicate call-hook return address";
    }
    return "unknown call-hook table error";
}

CallHookTableBuild CallHookTable::build(std::span<const CallHookSite> sites,
                                        std::size_t max_sites) {
    CallHookTableBuild result;
    if (sites.size() > max_sites) {
        result.error = {CallHookTableErrorKind::too_many_sites, max_sites, 0};
        return result;
    }
    std::vector<std::pair<CallHookSite, std::size_t>> sorted;
    try {
        sorted.reserve(sites.size());
        for (std::size_t index = 0; index < sites.size(); ++index) {
            const auto& site = sites[index];
            CallHookTableErrorKind error = CallHookTableErrorKind::none;
            if (site.return_address < 4) {
                error = CallHookTableErrorKind::null_return_address;
            } else if ((site.return_address & 3u) != 0) {
                error = CallHookTableErrorKind::misaligned_return_address;
            } else if (site.call_return_address < 4 || (site.call_return_address & 3u) != 0) {
                error = CallHookTableErrorKind::invalid_call_return_address;
            } else if (site.target_address == 0 || (site.target_address & 3u) != 0) {
                error = CallHookTableErrorKind::invalid_target_address;
            }
            if (error != CallHookTableErrorKind::none) {
                result.error = {error, index, site.return_address};
                return result;
            }
            sorted.emplace_back(site, index);
        }
        std::stable_sort(sorted.begin(), sorted.end(), [](const auto& left, const auto& right) {
            return left.first.return_address < right.first.return_address;
        });
        for (std::size_t index = 1; index < sorted.size(); ++index) {
            if (sorted[index - 1].first.return_address == sorted[index].first.return_address) {
                result.error = {CallHookTableErrorKind::duplicate_return_address,
                                sorted[index].second, sorted[index].first.return_address};
                return result;
            }
        }
        std::vector<CallHookSite> owned;
        owned.reserve(sorted.size());
        for (const auto& entry : sorted) owned.push_back(entry.first);
        result.table = std::shared_ptr<const CallHookTable>(new CallHookTable(std::move(owned)));
    } catch (const std::bad_alloc&) {
        throw;
    }
    return result;
}

const CallHookSite* CallHookTable::find(std::uint64_t return_address) const noexcept {
    const auto found = std::lower_bound(sites_.begin(), sites_.end(), return_address,
        [](const CallHookSite& site, std::uint64_t address) {
            return site.return_address < address;
        });
    return found != sites_.end() && found->return_address == return_address ? &*found : nullptr;
}

const char* describe(CallHookPatchError error) noexcept {
    switch (error) {
    case CallHookPatchError::none: return "no call-hook patch error";
    case CallHookPatchError::invalid_layout: return "invalid call-hook patch layout";
    case CallHookPatchError::overlapping_regions: return "overlapping call-hook patch regions";
    case CallHookPatchError::unsupported_site: return "call-hook site is not a direct BL";
    case CallHookPatchError::unreachable_branch: return "call-hook branch is out of range";
    }
    return "unknown call-hook patch error";
}

CallHookPatchResult make_call_hook_patch(std::uint32_t instruction, std::uint64_t address,
                                        std::uint64_t trampoline,
                                        CallHookCodeRegion gate) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const CallHookCodeRegion site{address, 8};
    const CallHookCodeRegion wrapper{trampoline, 16};
    if (address % 4 != 0 || trampoline % 4 != 0 || gate.address % 8 != 0 ||
        gate.size % 8 != 0 || address > maximum - 4 || !valid_range(site) ||
        !valid_range(wrapper) || !valid_range(gate)) {
        return {.error = CallHookPatchError::invalid_layout};
    }
    if (overlap(site, wrapper) || overlap(site, gate) || overlap(wrapper, gate)) {
        return {.error = CallHookPatchError::overlapping_regions};
    }
    const auto decoded = aarch64::decode(instruction);
    const auto target = aarch64::referenced_address(decoded, address);
    if (decoded.opcode != aarch64::Opcode::bl || !target) {
        return {.error = CallHookPatchError::unsupported_site};
    }
    const auto redirect = aarch64::encode_bl(address, trampoline);
    const auto call = aarch64::encode_bl(trampoline + 4, gate.address);
    const auto transfer = aarch64::encode_b(trampoline + 12, *target);
    if (!redirect.ok() || !call.ok() || !transfer.ok()) {
        return {.error = CallHookPatchError::unreachable_branch};
    }
    // STR X30,[SP,#-16]!; BL gate; LDR X30,[SP],#16; B original target.
    CallHookPatch patch{address, trampoline, instruction, redirect.word,
                        {0xf81f0ffeu, call.word, 0xf84107feu, transfer.word},
                        {trampoline + 8, address + 4, *target}};
    return {.patch = patch};
}

} // namespace armigo::execution
