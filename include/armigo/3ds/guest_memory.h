// ============================================================================
// Armigo Memory - Guest Memory Management
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace armigo::n3ds {

struct HostMemoryMapping {
    std::uint32_t address{};
    std::int32_t error{};
};

using MapGuestMemory = HostMemoryMapping (*)(void* context, std::uint32_t address,
                                           std::uint32_t size,
                                           std::uint32_t permissions) noexcept;

struct GuestMemoryRange {
    std::uint32_t address{};
    std::uint32_t size{};
    std::uint32_t permissions{};
};

// The process scheduler serializes access to this preallocated mapping state.
// A live instance and its mapper context have one owner; copying or reinitializing
// the object does not release the underlying host mappings.
class GuestMemory {
public:
    bool Initialize(std::uint64_t application_budget, MapGuestMemory mapper,
                    void* context) noexcept;

    // Selected by process metadata and the system memory layout, before allocation.
    bool ConfigureLinearHeap(std::uint32_t address, std::uint32_t size) noexcept;
    std::uint32_t LinearCommittedBytes() const noexcept { return linear_used_; }

    // The caller has already established this mapping and its permissions.
    bool RegisterExisting(GuestMemoryRange range, bool charge_application) noexcept;

    // Normal Commit and automatic linear Commit are admitted. Other operations remain explicit.
    std::uint32_t Commit(std::uint32_t operation, std::uint32_t address,
                         std::uint32_t alias_address, std::uint32_t size,
                         std::uint32_t permissions, std::uint32_t& output_address) noexcept;

    std::uint64_t CommittedBytes() const noexcept { return committed_; }
    std::uint64_t ApplicationBudget() const noexcept { return budget_; }
    std::size_t MappingCount() const noexcept { return count_; }
    std::uint32_t CommitCount() const noexcept { return commits_; }

    // A host failure must be diagnosed by execution rather than returned to the guest.
    std::int32_t LastHostError() const noexcept { return host_error_; }

private:
    bool Available(std::uint32_t address, std::uint32_t size) const noexcept;

    std::array<GuestMemoryRange, 64> ranges_{};
    MapGuestMemory mapper_{};
    void* context_{};
    std::uint64_t budget_{};
    std::uint64_t committed_{};
    std::size_t count_{};
    std::uint32_t commits_{};
    std::uint32_t linear_base_{};
    std::uint32_t linear_size_{};
    std::uint32_t linear_used_{};
    std::int32_t host_error_{};
};

} // namespace armigo::n3ds
