// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>
#include <optional>

namespace armigo::platform {

struct HostLimits {
    // The process VM map bounds, with an exclusive upper bound.
    std::uint64_t address_begin = 0;
    std::uint64_t address_end = 0;
    std::uint64_t native_page_bytes = 0;
    // Remaining dirty-memory allowance; absent if the host has no app limit.
    // This is a snapshot, not a promise that a future allocation will succeed.
    std::optional<std::uint64_t> available_memory_bytes;

    constexpr bool contains(std::uint64_t address, std::uint64_t size) const noexcept {
        return size && address >= address_begin && address < address_end && size <= address_end - address;
    }
    constexpr std::uint64_t guest_address_end(std::uint64_t guest_limit) const noexcept {
        return address_end < guest_limit ? address_end : guest_limit;
    }
    constexpr std::uint64_t guest_address_bytes(std::uint64_t guest_limit) const noexcept {
        const auto end = guest_address_end(guest_limit);
        return end > address_begin ? end - address_begin : 0;
    }
};

// Queries actual process limits, including entitlement-dependent iOS VM bounds.
// No framework types cross this interface. Query again to refresh the budget.
std::optional<HostLimits> query_host_limits() noexcept;

// Returns the native virtual-memory page size, or zero when it is unavailable.
std::uint64_t native_page_size() noexcept;

} // namespace armigo::platform
