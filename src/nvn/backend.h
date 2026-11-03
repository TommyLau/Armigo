// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Boundaries between the NVN driver's API layer (src/nvn, portable C++) and
// its host graphics implementation (src/platform/metal, Objective-C++):
//   ResourceBackend - memory pools, textures and descriptor registration;
//   ReplayBackend   - command replay, GPU submission and fences.
// The API layer validates arguments and keeps object state in place; the
// backends own host objects, found by object serials and guest addresses.
// Implementations must be callable from any guest thread.

#pragma once

#include "objects.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::nvn {

// A texel region of one level, with byte strides of the CPU-side image.
struct TexelTransfer {
    std::uint32_t level = 0;
    std::uint32_t first_layer = 0;
    NVNcopyRegion region{};
    std::size_t row_stride = 0;
    std::size_t image_stride = 0;
};

class ResourceBackend {
public:
    virtual ~ResourceBackend() = default;

    // A pool's memory becomes GPU-visible at its guest address.
    virtual bool register_pool(const MemoryPoolState& pool) = 0;
    virtual void release_pool(const MemoryPoolState& pool) = 0;
    // The CPU wrote [offset, offset + size) of the pool.
    virtual void flush_pool(const MemoryPoolState& pool, std::uint64_t offset, std::uint64_t size) = 0;
    // The CPU is about to read [offset, offset + size): GPU results written
    // there must reach guest memory first.
    virtual void invalidate_pool(const MemoryPoolState& pool, std::uint64_t offset, std::uint64_t size) = 0;

    // A texture object took a new serial (initialization) or gave it up.
    virtual void register_texture(const TextureState& texture) = 0;
    virtual void release_texture(const TextureState& texture) = 0;

    // Descriptor-pool lifetime hooks let an ID-only backend reject unsupported
    // pool switching instead of silently aliasing slots from different pools.
    virtual bool register_texture_pool(const DescriptorPoolState&) { return true; }
    virtual bool register_sampler_pool(const DescriptorPoolState&) { return true; }
    virtual void release_texture_pool(const DescriptorPoolState&) {}
    virtual void release_sampler_pool(const DescriptorPoolState&) {}

    // Descriptor pool slots as the game writes them.
    virtual void register_texture_descriptor(std::int32_t id, const TextureState& texture,
                                             const TextureViewState* view) = 0;
    virtual void register_sampler_descriptor(std::int32_t id, const SamplerParameters& sampler) = 0;

    // CPU texel access (nvnTextureReadTexels / WriteTexels and their strided forms).
    virtual bool read_texels(const TextureState& texture, const TexelTransfer& transfer, std::byte* out) = 0;
    virtual bool write_texels(const TextureState& texture, const TexelTransfer& transfer, const std::byte* in) = 0;
};

class ReplayBackend {
public:
    virtual ~ReplayBackend() = default;

    // Optional host scheduling hint, captured by each submit. It does not pace
    // execution or change guest fences; zero requests ordinary scheduling.
    virtual void set_scheduling_period(std::uint64_t period_ns) { (void)period_ns; }
    // Marks one presented frame after its submissions. Translation CPU budgets
    // are sampled at frame boundaries; encode-thread budgets are per submission.
    virtual void finish_frame(std::uint64_t period_ns) { set_scheduling_period(period_ns); }
    // Called on the guest thread that acquires window textures, once per
    // acquire: a host scheduling hint for the thread that paces the frame. A
    // zero period (no known presentation rate) leaves its policy as it was.
    virtual void acquire_tick(std::uint64_t period_ns) { (void)period_ns; }

    // Replays the recorded commands of the given command handles, in order,
    // and returns the fence value the GPU reaches once they complete.
    virtual std::uint64_t submit(std::span<const NVNcommandHandle> handles) = 0;
    // Starts submitted work without waiting (nvnQueueFlush).
    virtual void flush() = 0;
    // Highest fence value the GPU completed.
    [[nodiscard]] virtual std::uint64_t completed_fence() const = 0;
    // Blocks the calling thread until the fence completes or the timeout
    // (nanoseconds, UINT64_MAX for none) passes; false on timeout.
    virtual bool wait_fence(std::uint64_t fence, std::uint64_t timeout_ns) = 0;
};

// The backends the driver uses; set once before the game runs.
struct Backends {
    ResourceBackend* resources = nullptr;
    ReplayBackend* replay = nullptr;
};
void set_backends(Backends backends) noexcept;
[[nodiscard]] const Backends& backends() noexcept;

} // namespace armigo::nvn
