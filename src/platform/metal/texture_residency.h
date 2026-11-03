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
#include "texture_cache.h"
#include <armigo/graphics/resource_memory_registry.h>

namespace armigo::platform::metal {

// Native binding seam. Retain this lease through GPU completion and give it
// to ResourceMemory::reserve_accesses().commit() for writable requests.
// Register allocation() with the consuming Metal residency set. Sampling uses
// texture(); attachment encoders must honor the requested aspect and view.
class TextureResource : public graphics::ResourceMemoryStorage {
public:
    virtual id<MTLTexture> texture() const noexcept = 0;
    virtual id<MTLTexture> allocation() const noexcept = 0;
    bool guest_boundary_writeback() const noexcept override { return false; }
};

// One rendered 2D producer copied into a layer/level unit of a larger storage
// (a cube face or mip level drawn through its own render target). The cache
// copies these targets into the parent storage
// in queue order. The first packet carrying the copy encodes it.
class SubresourceCopy final : public graphics::ResourceLease {
public:
    id<MTLTexture> source = nil, destination = nil;
    std::uint32_t layer = 0, level = 0, width = 0, height = 0;
    // Present when the two textures encode the same texel bits in different
    // pixel formats (depth and colour), which Metal copies only via a buffer.
    id<MTLBuffer> staging = nil;
    std::uint64_t bytes_per_row = 0;
    mutable bool encoded = false;
};

struct TextureCacheStatistics {
    std::uint64_t uploads = 0, upload_bytes = 0, readbacks = 0;
    std::uint64_t hits = 0, resident_bytes = 0;
    // CPU revalidation outcomes: proven unchanged without a backing read by a
    // registry write stamp, proven unchanged by per-layer content hashes, and
    // re-uploads restricted to the changed layer/level units.
    std::uint64_t revalidations_skipped = 0, revalidations_hashed = 0;
    std::uint64_t reuploads = 0, reupload_bytes = 0;
};
[[nodiscard]] TextureCacheStatistics texture_cache_statistics(
    const graphics::TextureCache& cache) noexcept;

} // namespace armigo::platform::metal
