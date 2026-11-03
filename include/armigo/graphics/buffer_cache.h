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

#include <armigo/graphics/resource_memory.h>

namespace armigo::graphics {

// Use is access policy, not backing identity: overlapping vertex/index/uniform/
// storage requests MUST share coherence. A key is an exact interval; the cache
// also finds partial overlaps through the registry rather than key equality.
struct BufferKey {
    BackingRange bytes;
    bool operator==(const BufferKey&) const = default;
};
enum class BufferUse : std::uint8_t { Vertex, Index, Uniform, Storage, Transfer };
struct BufferRequest {
    RetainedRange range;
    BufferUse use = BufferUse::Vertex;
    ResourceAccess access = ResourceAccess::Read;
    std::uint64_t alignment = 1;
    // Native bytes that must exist from the request start, when larger than
    // the range: a GPU mapping rounded to its big page reaches past the
    // memory handle's guest backing. The tail reads zero and GPU writes to
    // it are discarded, never written back to guest memory.
    std::uint64_t native_extent = 0;
};
struct ResidentBuffer {
    BufferKey key;
    ResourceVersion uploaded_version;
    RetainedResource storage;
    std::uint64_t offset = 0, size = 0;
    std::vector<CompletionToken> dependencies;
};

enum class BufferSnapshotPolicy : std::uint8_t {
    RequireMapped,
    ZeroMissingConstantBufferBytes,
};
// Stage a read-only GPU-VA range in virtual order. Each backed fragment retains
// its mapping and resolves GPU authority before copying. Only the explicit
// constant-buffer policy synthesizes zeros for holes/padding; real read errors
// always fail. Descriptors, vertex data and writable storage remain strict.
[[nodiscard]] RendererResult<std::vector<std::byte>> snapshot_buffer_range(
    ResourceMemory& memory, std::uint64_t address, std::uint64_t size,
    BufferSnapshotPolicy policy = BufferSnapshotPolicy::RequireMapped);
// Same snapshot written into caller storage of exactly the snapshot size.
[[nodiscard]] RendererStatus snapshot_buffer_into(
    ResourceMemory& memory, std::uint64_t address, std::span<std::byte> output,
    BufferSnapshotPolicy policy);

// L5: immutable slices retain contents for their consuming completion token.
// Reusing an upload ring slot by frame count or overwriting draw A's uniform
// bytes for draw B is forbidden. Read/write hazards and partial overlaps share
// L2 versions. Unsupported alignment, bounds, aliasing or budgets stop with an
// error; trim may not evict authoritative dirty or in-flight storage.
class BufferCache {
public:
    virtual ~BufferCache() = default;
    virtual ResidencyDomain domain() const noexcept = 0;
    virtual RendererResult<ResidentBuffer> acquire(const BufferRequest& request) = 0;
    // Host-generated data (zero-filled cbuf windows, support buffers, converted
    // indices) is copied before return; it is not tied to temporary input spans.
    virtual RendererResult<ResidentBuffer> upload(std::span<const std::byte> bytes,
                                                 std::uint64_t alignment) = 0;
    // upload() whose bytes the caller writes in place: fill receives the new
    // slice's writable bytes and initialises all of them (no staging copy).
    using UploadFill = RendererStatus (*)(void* context, std::span<std::byte> output);
    virtual RendererResult<ResidentBuffer> upload_filled(std::uint64_t size, std::uint64_t alignment,
                                                        UploadFill fill, void* context) {
        std::vector<std::byte> bytes(size);
        auto status = fill(context, bytes);
        if (auto* failure = std::get_if<RendererError>(&status)) return *failure;
        return upload(bytes, alignment);
    }
    virtual RendererStatus invalidate(BackingRange bytes) = 0;
    virtual RendererStatus resolve(BufferKey key, ResolveReason reason) = 0;
    virtual RendererStatus trim(std::uint64_t budget_bytes) = 0;
};

} // namespace armigo::graphics
