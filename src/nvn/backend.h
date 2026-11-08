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
#include <optional>
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
    // The driver is about to store into [offset, offset + size) of the pool:
    // backend GPU work that still reads pool bytes finishes first.
    virtual void prepare_pool_write(const MemoryPoolState&, std::uint64_t, std::uint64_t) {}
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
    // A cadence inferred while unpaced retains the encoder's existing spin.
    virtual void set_scheduling_period(std::uint64_t period_ns, bool preserve_encode_spin = false) {
        (void)period_ns; (void)preserve_encode_spin;
    }
    // Marks one presented frame after its submissions. Translation CPU budgets
    // are sampled at frame boundaries; encode-thread budgets are per submission.
    virtual void finish_frame(std::uint64_t period_ns, bool preserve_encode_spin = false) {
        set_scheduling_period(period_ns, preserve_encode_spin);
    }
    // Called on the guest thread that acquires window textures, once per
    // acquire: a host scheduling hint for the thread that paces the frame. A
    // zero period (no known presentation rate) leaves its policy as it was.
    virtual void acquire_tick(std::uint64_t period_ns) { (void)period_ns; }

    // Optional host feedback: the latest completed presented frame's sum of
    // GPU execution intervals, excluding gaps between command buffers. Zero
    // means unavailable. Querying enables collection; it never changes GPU
    // counters, submission order or guest fence completion.
    [[nodiscard]] virtual std::uint64_t gpu_frame_work_ns() const noexcept { return 0; }

    // Replays the recorded commands of the given command handles, in order,
    // and returns the fence value the GPU reaches once they complete.
    virtual std::uint64_t submit(std::span<const NVNcommandHandle> handles) = 0;
    // The same, naming the NVN queue that submits them; a backend that orders
    // queues independently overrides it (see "Queues and event waits" in
    // the project documentation).
    virtual std::uint64_t submit_from(std::uintptr_t queue, std::span<const NVNcommandHandle> handles) {
        static_cast<void>(queue);
        return submit(handles);
    }
    // The queue `queue` names waits for `fence` to complete before its later
    // work runs (nvnQueueWaitSync). Returns the fence that completes once the
    // wait passed, which becomes the queue's latest fence (0 when nothing is
    // left to wait for), or nothing if `fence` was never submitted. A backend
    // that orders queues independently does not block the caller.
    virtual std::optional<std::uint64_t> queue_wait_fence(std::uintptr_t queue, std::uint64_t fence) {
        static_cast<void>(queue);
        if (!wait_fence(fence, UINT64_MAX)) return std::nullopt;
        return 0;
    }
    // The queue `queue` named is finalized (nvnQueueFinalize): its submitted
    // work still runs in order, and nothing is submitted under it again.
    virtual void forget_queue(std::uintptr_t queue) { static_cast<void>(queue); }
    // Starts submitted work without waiting (nvnQueueFlush).
    virtual void flush() = 0;
    // Highest fence value the GPU completed, with every fence before it.
    [[nodiscard]] virtual std::uint64_t completed_fence() const = 0;
    // Whether the work of `fence` completed: a backend that holds a queue's
    // work while others run completes a later fence before the held one.
    [[nodiscard]] virtual bool fence_completed(std::uint64_t fence) const { return completed_fence() >= fence; }
    // Blocks the calling thread until the fence completes (fence_completed)
    // or the timeout (nanoseconds, UINT64_MAX for none) passes; false on
    // timeout.
    virtual bool wait_fence(std::uint64_t fence, std::uint64_t timeout_ns) = 0;
};

// The backends the driver uses; set once before the game runs.
struct Backends {
    ResourceBackend* resources = nullptr;
    ReplayBackend* replay = nullptr;
};
void set_backends(Backends backends) noexcept;
[[nodiscard]] const Backends& backends() noexcept;

// Called right after each CPU nvnEventSignal store, on the signalling thread,
// with the event and the value that store left, so replay can evaluate the
// event waits it holds against that value even if a later store already
// replaced it. One observer at a time; clearing passes the context that
// registered it.
using EventSignalObserver = void (*)(void* context, const volatile std::uint32_t* event,
                                     std::uint32_t value) noexcept;
void set_event_signal_observer(EventSignalObserver observer, void* context) noexcept;
void clear_event_signal_observer(void* context) noexcept;
// Delivers one CPU signal to the observer; nvnEventSignal calls it.
void notify_event_signal_observer(const volatile std::uint32_t* event, std::uint32_t value) noexcept;

} // namespace armigo::nvn
