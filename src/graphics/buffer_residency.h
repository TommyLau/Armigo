// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/buffer_cache.h>
#include <armigo/graphics/resource_memory_registry.h>

namespace armigo::graphics {

// Native adapters own allocation and CPU-visible storage only. Coherence,
// slicing, admission and lifetime accounting all live in buffer_cache.cpp.
class BufferAllocation : public ResourceLease {
public:
    virtual std::span<std::byte> contents() const noexcept = 0;
    virtual std::uint64_t cost() const noexcept = 0;
};
class BufferAllocator : public ResourceLease {
public:
    virtual std::uint64_t device_identity() const noexcept = 0;
    virtual RendererResult<std::shared_ptr<BufferAllocation>> allocate(
        std::uint64_t size) = 0;
};
class BufferStorage : public ResourceMemoryStorage {
public:
    virtual std::shared_ptr<BufferAllocation> allocation() const noexcept = 0;
};

// CPU revalidation outcomes counted since creation: proofs by registry write
// stamp without any guest read, proofs by page content hashes, and re-uploads
// restricted to changed 4096-byte page ranges. Certificates combine a
// registry write stamp with page content hashes.
struct BufferCacheStatistics {
    std::uint64_t revalidations_skipped = 0, revalidations_hashed = 0;
    std::uint64_t reuploads = 0, reupload_bytes = 0;
};
[[nodiscard]] BufferCacheStatistics buffer_cache_statistics(const BufferCache& cache) noexcept;

[[nodiscard]] RendererResult<std::shared_ptr<BufferCache>> make_resident_buffer_cache(
    std::shared_ptr<BufferAllocator> allocator,
    std::shared_ptr<ResourceMemory> memory, std::uint64_t budget);

} // namespace armigo::graphics
