// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/memory/address_info.h"
#include "renderer/memory/pool_mapper.h"
#include "renderer/common/guest_memory.h"
#include "renderer/common/support.h"

#include <atomic>

namespace armigo::audio::renderer {

namespace {
// Horizon's memory pool granularity.
constexpr u64 PoolPageSize = 0x1000;
}  // namespace

PoolMapper::PoolMapper(GuestMemory* memory_, bool force_map_)
    : memory{memory_}, force_map{force_map_} {}

PoolMapper::PoolMapper(GuestMemory* memory_, std::span<MemoryPoolInfo> pool_infos_,
                       u32 pool_count_, bool force_map_)
    : memory{memory_}, pool_infos{pool_infos_.data()},
      pool_count{pool_count_}, force_map{force_map_} {}

void PoolMapper::clear_use_state(std::span<MemoryPoolInfo> pools, const u32 count) {
    for (u32 i = 0; i < count; i++) {
        pools[i].set_used(false);
    }
}

MemoryPoolInfo* PoolMapper::find_memory_pool(MemoryPoolInfo* pools, const u64 count,
                                           const CpuAddr address, const u64 size) const {
    auto pool{pools};
    for (u64 i = 0; i < count; i++, pool++) {
        if (pool->contains(address, size)) {
            return pool;
        }
    }
    return nullptr;
}

MemoryPoolInfo* PoolMapper::find_memory_pool(const CpuAddr address, const u64 size) const {
    auto pool{pool_infos};
    for (u64 i = 0; i < pool_count; i++, pool++) {
        if (pool->contains(address, size)) {
            return pool;
        }
    }
    return nullptr;
}

bool PoolMapper::fill_dsp_addr(AddressInfo& address_info, MemoryPoolInfo* pools,
                             const u32 count) const {
    if (address_info.get_cpu_addr() == 0) {
        address_info.set_pool(nullptr);
        return false;
    }

    auto found_pool{
        find_memory_pool(pools, count, address_info.get_cpu_addr(), address_info.get_size())};
    if (found_pool != nullptr) {
        address_info.set_pool(found_pool);
        return true;
    }

    if (force_map) {
        force_map_pointer(address_info);
    } else {
        address_info.set_pool(nullptr);
    }

    return false;
}

bool PoolMapper::fill_dsp_addr(AddressInfo& address_info) const {
    if (address_info.get_cpu_addr() == 0) {
        address_info.set_pool(nullptr);
        return false;
    }

    auto found_pool{find_memory_pool(address_info.get_cpu_addr(), address_info.get_size())};
    if (found_pool != nullptr) {
        address_info.set_pool(found_pool);
        return true;
    }

    if (force_map) {
        force_map_pointer(address_info);
    } else {
        address_info.set_pool(nullptr);
    }

    return false;
}

bool PoolMapper::try_attach_buffer(BehaviorInfo::ErrorInfo& error_info, AddressInfo& address_info,
                                 const CpuAddr address, const u64 size) const {
    address_info.setup(address, size);

    if (!fill_dsp_addr(address_info)) {
        error_info.error_code = ResultInvalidAddressInfo;
        error_info.address = address;
        return force_map;
    }

    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
    return true;
}

bool PoolMapper::is_force_map_enabled() const {
    return force_map;
}

void PoolMapper::force_map_pointer(AddressInfo& address_info) const {
    // A forced mapping makes a buffer outside every attached pool directly
    // reachable by the DSP. Guest memory is only reachable through pinned
    // pools here, so a buffer outside them stays unmapped: its commands are
    // skipped instead of reading unvalidated memory.
    address_info.set_pool(nullptr);
    if (memory != nullptr && memory->contains(address_info.get_cpu_addr(), address_info.get_size())) {
        address_info.set_force_mapped_dsp_addr(address_info.get_cpu_addr());
        return;
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
        AUDIO_LOG_WARNING("forced mapping of {:#x} bytes at {:#x} outside every attached memory pool is not supported; the buffer stays unmapped",
                          address_info.get_size(), address_info.get_cpu_addr());
    }
}

bool PoolMapper::map(MemoryPoolInfo& pool) const {
    // Pin the guest range so it stays mapped while the pool is attached.
    if (memory == nullptr || !memory->map(pool.get_cpu_address(), pool.get_size())) {
        pool.set_dsp_address(0);
        return false;
    }
    pool.set_dsp_address(pool.get_cpu_address());
    return true;
}

bool PoolMapper::unmap(MemoryPoolInfo& pool) const {
    if (memory != nullptr && pool.is_mapped()) {
        memory->unmap(pool.get_cpu_address(), pool.get_size());
    }
    pool.set_cpu_address(0, 0);
    pool.set_dsp_address(0);
    return true;
}

void PoolMapper::force_unmap_pointer([[maybe_unused]] const AddressInfo& address_info) const {
    // Forced mappings never pin anything of their own (see force_map_pointer).
}

MemoryPoolInfo::ResultState PoolMapper::update(MemoryPoolInfo& pool,
                                               const MemoryPoolInfo::InParameter& in_params,
                                               MemoryPoolInfo::OutStatus& out_params) const {
    if (in_params.state != MemoryPoolInfo::State::RequestAttach &&
        in_params.state != MemoryPoolInfo::State::RequestDetach) {
        return MemoryPoolInfo::ResultState::Success;
    }

    if (in_params.address == 0 || in_params.size == 0
    || !is_aligned(in_params.address, PoolPageSize)
    || !is_aligned(in_params.size, PoolPageSize)) {
        return MemoryPoolInfo::ResultState::BadParam;
    }

    switch (in_params.state) {
    case MemoryPoolInfo::State::RequestAttach:
        // Re-attaching releases the previous pin before the new range is pinned.
        if (pool.is_mapped()) {
            unmap(pool);
        }
        pool.set_cpu_address(in_params.address, in_params.size);

        map(pool);

        if (pool.is_mapped()) {
            out_params.state = MemoryPoolInfo::State::Attached;
            return MemoryPoolInfo::ResultState::Success;
        }
        pool.set_cpu_address(0, 0);
        return MemoryPoolInfo::ResultState::MapFailed;

    case MemoryPoolInfo::State::RequestDetach:
        if (pool.get_cpu_address() != in_params.address || pool.get_size() != in_params.size) {
            return MemoryPoolInfo::ResultState::BadParam;
        }

        if (pool.is_used()) {
            return MemoryPoolInfo::ResultState::InUse;
        }

        unmap(pool);

        pool.set_cpu_address(0, 0);
        pool.set_dsp_address(0);
        out_params.state = MemoryPoolInfo::State::Detached;
        return MemoryPoolInfo::ResultState::Success;

    default:
        AUDIO_LOG_ERROR("Invalid MemoryPoolInfo::State!");
        break;
    }

    return MemoryPoolInfo::ResultState::Success;
}

bool PoolMapper::initialize_system_pool(MemoryPoolInfo& pool, const u8* workbuffer,
                                      const u64 size_) const {
    switch (pool.get_location()) {
    case MemoryPoolInfo::Location::CPU:
        return false;
    case MemoryPoolInfo::Location::DSP:
        // The renderer's own host workbuffer: always reachable, never pinned.
        pool.set_cpu_address(reinterpret_cast<u64>(workbuffer), size_);
        pool.set_dsp_address(pool.get_cpu_address());
        return true;
    default:
        AUDIO_LOG_WARNING("Invalid MemoryPoolInfo location={}!",
                    static_cast<u32>(pool.get_location()));
        return false;
    }
}

} // namespace armigo::audio::renderer
