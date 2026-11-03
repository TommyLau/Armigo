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

#include "renderer/memory/memory_pool_info.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

// Represents a region of mapped or unmapped memory.
class AddressInfo {
public:
    AddressInfo() = default;
    AddressInfo(CpuAddr cpu_address_, u64 size_) : cpu_address{cpu_address_}, size{size_} {}

    // Setup a new AddressInfo.
    void setup(CpuAddr cpu_address_, u64 size_) {
        cpu_address = cpu_address_;
        size = size_;
        memory_pool = nullptr;
        dsp_address = 0;
    }

    // Get the CPU address.
    CpuAddr get_cpu_addr() const {
        return cpu_address;
    }

    // Assign this region to a memory pool.
    void set_pool(MemoryPoolInfo* memory_pool_) {
        memory_pool = memory_pool_;
    }

    // Get the size of this region.
    u64 get_size() const {
        return size;
    }

    // Get the ADSP address for this region.
    CpuAddr get_force_mapped_dsp_addr() const {
        return dsp_address;
    }

    // Set the ADSP address for this region.
    void set_force_mapped_dsp_addr(CpuAddr dsp_addr) {
        dsp_address = dsp_addr;
    }

    // Check whether this region has an active memory pool.
    bool has_mapped_memory_pool() const {
        return memory_pool != nullptr && memory_pool->get_dsp_address() != 0;
    }

    // Check whether this region is mapped to the ADSP.
    bool is_mapped() const {
        return has_mapped_memory_pool() || dsp_address != 0;
    }

    // Get a usable reference to this region of memory.
    CpuAddr get_reference(bool mark_in_use) {
        if (!has_mapped_memory_pool()) {
            return dsp_address;
        }

        if (mark_in_use) {
            memory_pool->set_used(true);
        }

        return memory_pool->translate(cpu_address, size);
    }

private:
    // CPU address of this region
    CpuAddr cpu_address;
    // Size of this region
    u64 size;
    // The memory this region is mapped to
    MemoryPoolInfo* memory_pool;
    // ADSP address of this region
    CpuAddr dsp_address;
};

} // namespace armigo::audio::renderer
