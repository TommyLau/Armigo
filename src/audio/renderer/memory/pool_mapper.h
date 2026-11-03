// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <span>

#include "renderer/behavior/behavior_info.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/common/types.h"
#include "renderer/common/result.h"

namespace armigo::audio::renderer {
class AddressInfo;
class GuestMemory;

// Utility functions for managing MemoryPoolInfos
class PoolMapper {
public:
    explicit PoolMapper(GuestMemory* memory, bool force_map);
    explicit PoolMapper(GuestMemory* memory, std::span<MemoryPoolInfo> pool_infos,
                        u32 pool_count, bool force_map);

    // Clear the usage state for all given pools.
    static void clear_use_state(std::span<MemoryPoolInfo> pools, u32 count);

    // Find the memory pool containing the given address and size from a given list of pools.
    MemoryPoolInfo* find_memory_pool(MemoryPoolInfo* pools, u64 count, CpuAddr address,
                                   u64 size) const;

    // Find the memory pool containing the given address and size from the PoolMapper's memory pool.
    MemoryPoolInfo* find_memory_pool(CpuAddr address, u64 size) const;

    // Set the PoolMapper's memory pool to one in the given list of pools, which contains
    // address_info.
    bool fill_dsp_addr(AddressInfo& address_info, MemoryPoolInfo* pools, u32 count) const;

    // Set the PoolMapper's memory pool to the one containing address_info.
    bool fill_dsp_addr(AddressInfo& address_info) const;

    // Try to attach a {address, size} region to the given address_info, and map it. Fills in the
    // given error_info and address_info.
    bool try_attach_buffer(BehaviorInfo::ErrorInfo& error_info, AddressInfo& address_info,
                         CpuAddr address, u64 size) const;

    // Return whether force mapping is enabled.
    bool is_force_map_enabled() const;

    // Map the given memory pool.
    bool map(MemoryPoolInfo& pool) const;

    // Unmap the given memory pool.
    bool unmap(MemoryPoolInfo& pool) const;

    // Forcibly unmap the given region.
    void force_unmap_pointer(const AddressInfo& address_info) const;

    // Force-map a buffer that lies outside every attached pool.
    void force_map_pointer(AddressInfo& address_info) const;

    // Update the given memory pool.
    MemoryPoolInfo::ResultState update(MemoryPoolInfo& pool,
                                       const MemoryPoolInfo::InParameter& in_params,
                                       MemoryPoolInfo::OutStatus& out_params) const;

    // Initialize the PoolMapper's memory pool.
    bool initialize_system_pool(MemoryPoolInfo& pool, const u8* memory, u64 size) const;

private:
    // Guest memory the pools pin into
    GuestMemory* memory{};
    // List of memory pools assigned to this mapper
    MemoryPoolInfo* pool_infos{};
    // The number of pools
    u64 pool_count{};
    // Is forced mapping enabled
    bool force_map;
};

} // namespace armigo::audio::renderer
