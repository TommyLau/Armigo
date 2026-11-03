// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/common/guest_memory.h"

#include "renderer/common/support.h"

#include <cstring>
#include <limits>

namespace armigo::audio::renderer {

GuestMemory::~GuestMemory() { unmap_all(); }

bool GuestMemory::map(CpuAddr address, u64 size) {
    if (address == 0 || size == 0 || size > std::numeric_limits<CpuAddr>::max() - address)
        return false;
    const auto key = std::make_pair(address, size);
    if (auto found = ranges_.find(key); found != ranges_.end()) {
        ++found->second.references;
        return true;
    }
    if (!pins_ || !pins_->pin(address, size)) return false;
    ranges_.emplace(key, Range{size, 1});
    if (size > largest_range_) largest_range_ = size;
    return true;
}

void GuestMemory::unmap(CpuAddr address, u64 size) {
    const auto found = ranges_.find(std::make_pair(address, size));
    if (found == ranges_.end()) return;
    if (--found->second.references != 0) return;
    ranges_.erase(found);
    if (pins_) pins_->unpin(address, size);
}

void GuestMemory::unmap_all() {
    for (const auto& [key, range] : ranges_)
        if (pins_) pins_->unpin(key.first, key.second);
    ranges_.clear();
    largest_range_ = 0;
}

bool GuestMemory::contains(CpuAddr address, u64 size) const {
    if (size > std::numeric_limits<CpuAddr>::max() - address || ranges_.empty()) return false;
    // Only ranges starting at or before `address`, and no further back than the
    // largest pinned size, can contain it.
    auto it = ranges_.upper_bound(std::make_pair(address, std::numeric_limits<u64>::max()));
    while (it != ranges_.begin()) {
        --it;
        const auto start = it->first.first;
        if (address - start > largest_range_) break;
        const auto length = it->first.second;
        if (address - start <= length && size <= length - (address - start)) return true;
    }
    return false;
}

void GuestMemory::reject(CpuAddr address, u64 size, const char* operation) const {
    const auto count = rejected_.fetch_add(1, std::memory_order_relaxed);
    if (count < 16 || (count & 0xfff) == 0)
        AUDIO_LOG_WARNING("guest {} of {:#x} bytes at {:#x} lies outside every attached memory pool ({} rejected)",
                          operation, size, address, count + 1);
}

bool GuestMemory::read_block(CpuAddr address, void* out, u64 size) const {
    if (size == 0) return true;
    if (!contains(address, size)) {
        std::memset(out, 0, size);
        reject(address, size, "read");
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    return true;
}

bool GuestMemory::write_block(CpuAddr address, const void* in, u64 size) {
    if (size == 0) return true;
    if (!contains(address, size)) {
        reject(address, size, "write");
        return false;
    }
    std::memcpy(reinterpret_cast<void*>(address), in, size);
    return true;
}

}  // namespace armigo::audio::renderer
