// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/memory/memory_pool_info.h"

namespace armigo::audio::renderer {

CpuAddr MemoryPoolInfo::get_cpu_address() const {
    return cpu_address;
}

CpuAddr MemoryPoolInfo::get_dsp_address() const {
    return dsp_address;
}

u64 MemoryPoolInfo::get_size() const {
    return size;
}

MemoryPoolInfo::Location MemoryPoolInfo::get_location() const {
    return location;
}

void MemoryPoolInfo::set_cpu_address(const CpuAddr address, const u64 size_) {
    cpu_address = address;
    size = size_;
}

void MemoryPoolInfo::set_dsp_address(const CpuAddr address) {
    dsp_address = address;
}

bool MemoryPoolInfo::contains(const CpuAddr address_, const u64 size_) const {
    return cpu_address <= address_ && (address_ + size_) <= (cpu_address + size);
}

bool MemoryPoolInfo::is_mapped() const {
    return dsp_address != 0;
}

CpuAddr MemoryPoolInfo::translate(const CpuAddr address, const u64 size_) const {
    if (!contains(address, size_)) {
        return 0;
    }

    if (!is_mapped()) {
        return 0;
    }

    return dsp_address + (address - cpu_address);
}

void MemoryPoolInfo::set_used(const bool used) {
    in_use = used;
}

bool MemoryPoolInfo::is_used() const {
    return in_use;
}

} // namespace armigo::audio::renderer
