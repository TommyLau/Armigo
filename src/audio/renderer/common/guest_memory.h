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

#include "renderer/common/types.h"

#include <armigo/audio/audio_renderer.h>

#include <atomic>
#include <map>

namespace armigo::audio::renderer {

// The renderer's only route to guest memory. A range becomes accessible when
// the guest attaches a memory pool covering it: the pool mapper pins the range
// through the runtime (so it stays mapped until it is detached) and records it
// here. Every DSP read or write of guest sample, context or effect memory is
// checked against the recorded ranges before the native address is touched;
// an unchecked guest pointer is never dereferenced. A rejected read yields
// zeros and a rejected write is dropped, both counted.
class GuestMemory {
public:
    GuestMemory() = default;
    explicit GuestMemory(std::shared_ptr<GuestMemoryPins> pins) : pins_(std::move(pins)) {}
    ~GuestMemory();
    GuestMemory(const GuestMemory&) = delete;
    GuestMemory& operator=(const GuestMemory&) = delete;

    // Releases every pin and adopts a new pinning backend.
    void reset(std::shared_ptr<GuestMemoryPins> pins) {
        unmap_all();
        pins_ = std::move(pins);
    }

    // Pins [address, address + size). Repeated pins of the same range nest.
    bool map(CpuAddr address, u64 size);
    void unmap(CpuAddr address, u64 size);
    void unmap_all();

    // True when [address, address + size) lies inside one pinned range.
    bool contains(CpuAddr address, u64 size) const;

    bool read_block(CpuAddr address, void* out, u64 size) const;
    bool write_block(CpuAddr address, const void* in, u64 size);
    bool write32(CpuAddr address, u32 value) { return write_block(address, &value, sizeof value); }

    u64 rejected_accesses() const noexcept { return rejected_.load(std::memory_order_relaxed); }

private:
    struct Range {
        u64 size = 0;
        u32 references = 0;
    };
    void reject(CpuAddr address, u64 size, const char* operation) const;

    std::shared_ptr<GuestMemoryPins> pins_;
    // Pinned ranges keyed by start address, then size; overlapping ranges are allowed.
    std::map<std::pair<CpuAddr, u64>, Range> ranges_;
    u64 largest_range_ = 0;
    mutable std::atomic<u64> rejected_{0};
};

}  // namespace armigo::audio::renderer
