// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Writable pins prepare their host pages through the page write tracker.
// A syscall writing into a protected tracked page fails with EFAULT instead
// of faulting, so every host write
// into guest memory prepares tracked pages first. Bounded byte copies use a
// registered writable alias when available and publish after the copy.

#pragma once

#include "../horizon/memory_interval_map.h"
#include "../horizon/process_memory.h"

namespace armigo::runtime {

// A borrowed legacy view or a bounded private-tree snapshot. Never persistent.
class MemoryMapView {
public:
    explicit MemoryMapView(std::span<const horizon::MemoryBlock> borrowed) : borrowed_(borrowed) {}
    explicit MemoryMapView(std::vector<horizon::MemoryBlock> owned) : owned_(std::move(owned)), owns_(true) {}
    std::span<const horizon::MemoryBlock> elements() const {
        return owns_ ? std::span<const horizon::MemoryBlock>(owned_) : borrowed_;
    }
    operator std::span<const horizon::MemoryBlock>() const { return elements(); }
    auto begin() const { return elements().begin(); }
    auto end() const { return elements().end(); }
    std::size_t size() const { return elements().size(); }
private:
    std::span<const horizon::MemoryBlock> borrowed_;
    std::vector<horizon::MemoryBlock> owned_;
    bool owns_ = false;
};

struct MemoryMapPlan {
    std::vector<horizon::MemoryBlock> legacy;
    std::vector<horizon::MemoryMapUpdate> updates;
    std::shared_ptr<horizon::MemoryIntervalMap> intervals;
    void replace(horizon::AddressRange range, std::optional<horizon::MemoryBlock> replacement,
        std::uint8_t set_merge = 0, std::uint8_t clear_merge = 0) {
        if (intervals) updates.push_back({range, replacement, set_merge, clear_merge});
        else legacy = horizon::process_memory::replace(legacy, range, replacement, set_merge, clear_merge);
    }
    MemoryMapView view(horizon::AddressRange range) const {
        if (!intervals) return MemoryMapView(std::span<const horizon::MemoryBlock>(legacy));
        auto preview = intervals->snapshot(range);
        for (const auto& update : updates) {
            if (!horizon::process_memory::overlaps(range, update.range)) continue;
            preview = horizon::process_memory::replace(preview, update.range, update.replacement,
                update.set_merge, update.clear_merge);
        }
        return MemoryMapView(std::move(preview));
    }
};
} // namespace armigo::runtime
