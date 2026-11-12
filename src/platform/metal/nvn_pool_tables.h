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

#import <Metal/Metal.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

class ReplayResources;

// One shared read-bitmap word count: 2^20 texture IDs, 32 IDs per word.
inline constexpr unsigned pool_seen_words = (1u << 20) / 32;

enum class PoolComponent : std::uint8_t { Float, Sint, Uint, Depth };

// A table is one element type. Every sampler binding shares one table
// (2D, float, normalized coordinates, compare never): each entry is the
// registered sampler's own state. Texture tables force both flags false.
// The entry check is the Metal texture type.
struct PoolTableKey {
    bool sampler = false;
    MTLTextureType type = MTLTextureType2D;
    PoolComponent component = PoolComponent::Float;
    bool normalized_coordinates = false;
    bool comparison = false;
    bool operator==(const PoolTableKey&) const = default;
};

// Guest storage identity for a depth texture a pool read. 0 when the texture
// has no guest memory. A registered texture forces a nonzero key.
inline std::uint64_t pool_storage_key(std::uint64_t memory, std::uint64_t size,
                                      const void* description, std::size_t bytes) {
    if (!memory) return 0;
    std::uint64_t hash = memory ^ (size * 0x9e3779b97f4a7c15ull);
    const auto* raw = static_cast<const unsigned char*>(description);
    for (std::size_t index = 0; index < bytes; ++index)
        hash = (hash ^ raw[index]) * 0x100000001b3ull;
    return hash ? hash : 1;
}

struct PoolSpareStore;

// Encoded table versions keep their bitmap alive through GPU completion.
// Creation epochs distinguish a late old read from a read of a refilled ID.
struct PoolReadBitmap {
    id<MTLBuffer> buffer = nil;
    std::uint64_t pool_generation = 0;
    std::uint64_t empty_generation = 0;
};

// One published table. An encoded submission holds a shared pointer, so the
// buffer and the residency set stay unchanged until that submission retires
// and drops the pointer. The buffer returns to the spare list then.
struct PoolTableVersion {
    id<MTLBuffer> buffer = nil;
    std::shared_ptr<PoolReadBitmap> read_bitmap;
    std::size_t capacity = 0;
    std::uint64_t gpu = 0;
    std::uint32_t n = 0;
    id<MTLResidencySet> residency = nil;
    std::vector<id<MTLTexture>> textures;
    std::vector<id<MTLSamplerState>> samplers;
    std::vector<id<MTLBuffer>> extra_buffers;
    std::vector<id<MTLAllocation>> allocations;
    // The members of allocations, so adding a texture to a patched table
    // does not scan the whole list.
    std::unordered_set<const void*> allocation_keys;
    std::atomic<std::uint64_t> retire_fence{0};
    std::shared_ptr<PoolSpareStore> spares;
    ~PoolTableVersion();
};

struct PoolLookup {
    bool ok = false;
    std::shared_ptr<PoolTableVersion> version;
    id<MTLBuffer> bitmap = nil;
    std::string diagnostic;
};

struct PoolTableStatistics {
    std::uint64_t builds = 0;
    std::uint64_t patches = 0;
    std::uint64_t kept = 0;
    std::uint64_t ids_read = 0;
    std::uint64_t textures_made = 0;
    std::uint64_t textures_left = 0;
    std::uint64_t walk_us = 0;
};

// Current table version per element-type key, plus the registration and
// storage generations it was built from. lookup follows the pool walk:
// reuse, copy-and-patch, or rebuild, inside the frame and request budgets.
class PoolTableCache {
public:
    explicit PoolTableCache(id<MTLDevice> device);
    ~PoolTableCache();

    PoolTableCache(const PoolTableCache&) = delete;
    PoolTableCache& operator=(const PoolTableCache&) = delete;

    // submission is the fence of the submission being encoded. completed_fence
    // is the newest fence whose GPU work has retired. frame is the replay's
    // present count.
    PoolLookup lookup(const PoolTableKey& key, ReplayResources& resources,
                      std::uint64_t submission, std::uint64_t completed_fence,
                      std::uint64_t frame);

    void note_present();
    [[nodiscard]] std::uint64_t present_count() const;
    [[nodiscard]] std::uint64_t generation() const;
    [[nodiscard]] PoolTableStatistics statistics() const;
    [[nodiscard]] id<MTLBuffer> current_texture_buffer() const;
    void set_allocation_failure(bool fail);

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace armigo::platform::metal::nvn_driver
