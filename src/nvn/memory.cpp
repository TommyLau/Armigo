// ============================================================================
// Armigo NVN - Memory Pool and Buffer Entry Points
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
// Memory pool and buffer entry points. GPU addresses equal guest addresses:
// a pool's GPU base is the storage the game set, made GPU-visible through
// ResourceBackend::register_pool; the pool state lives in place (objects.h).
// FlushMappedRange publishes the flushed range into the pool state (with a
// serial, end 0 while a publication is in progress) so texture creation can
// order itself against CPU writes, and tells the backend about the write.
// Only CPU-cached pools flush or invalidate; the native driver treats them
// as no-ops for every other pool. Virtual pools are accepted as objects
// (no backing, no GPU address) and every mapping request is reported as
// unsupported. Buffers are views into a pool: the native driver accepts
// storage outside the pool (only the accessed ranges matter), rejects
// physical-only pools, and hands buffer addresses out as pool addresses.

#include "backend.h"
#include "entry_table.h"
#include "host_stack.h"
#include "objects.h"

#include <atomic>
#include <cstdio>

namespace armigo::nvn {
namespace {

using namespace api;

constexpr std::uint32_t live_pool_tag = object_tag('P', 'O', 'O', 'L');
constexpr std::uint32_t live_buffer_tag = object_tag('B', 'U', 'F', 0);

// Reports the first occurrence of a misuse or unsupported case.
template <std::size_t Site>
void report_once(const char* text) noexcept {
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true, std::memory_order_relaxed))
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: %s\n", text);
}

// ---------------------------------------------------------------- buffers

// Buffer states are private to this file: nothing outside reads them, and
// both fit the storage the API reserves (the buffer state fills it exactly).
struct BufferBuilderState {
    NVNdevice* device;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint64_t size;
};
static_assert(sizeof(BufferBuilderState) <= sizeof(NVNbufferBuilder),
              "BufferBuilderState exceeds the storage of NVNbufferBuilder");

struct BufferState {
    std::uint32_t tag, reserved;  // reserved is zero: no uninitialized bytes in the object
    NVNdevice* device;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint64_t size;
    std::uint64_t serial;
};
static_assert(sizeof(BufferState) <= sizeof(NVNbuffer), "BufferState exceeds the storage of NVNbuffer");

// ---------------------------------------------------------------- pool helpers

MemoryPoolBuilderState* builder_of(NVNmemoryPoolBuilder* builder) noexcept {
    return reinterpret_cast<MemoryPoolBuilderState*>(builder);
}
const MemoryPoolBuilderState* builder_of(const NVNmemoryPoolBuilder* builder) noexcept {
    return reinterpret_cast<const MemoryPoolBuilderState*>(builder);
}

MemoryPoolState* state_of(NVNmemoryPool* pool) noexcept { return reinterpret_cast<MemoryPoolState*>(pool); }
const MemoryPoolState* state_of(const NVNmemoryPool* pool) noexcept {
    return reinterpret_cast<const MemoryPoolState*>(pool);
}

BufferBuilderState* builder_of(NVNbufferBuilder* builder) noexcept {
    return reinterpret_cast<BufferBuilderState*>(builder);
}
const BufferBuilderState* builder_of(const NVNbufferBuilder* builder) noexcept {
    return reinterpret_cast<const BufferBuilderState*>(builder);
}
BufferState* state_of(NVNbuffer* buffer) noexcept { return reinterpret_cast<BufferState*>(buffer); }
const BufferState* state_of(const NVNbuffer* buffer) noexcept {
    return reinterpret_cast<const BufferState*>(buffer);
}

// Releases a pool's backend registration exactly when it has one: a pool is
// registered only after initialization succeeded with CPU storage.
void release_registration(MemoryPoolState& pool) noexcept {
    retire_pool_textures(&pool);
    if (pool.memory && backends().resources) on_host_stack([&] { backends().resources->release_pool(pool); });
}

// Publishes the latest flushed range: end 0 marks a publication in progress,
// and the serial orders the publication against texture creation serials.
void publish_flush(MemoryPoolState& pool, std::uint64_t begin, std::uint64_t size) noexcept {
    std::atomic_ref<std::uint64_t> flush_end(pool.flush_end);
    std::atomic_ref<std::uint64_t> flush_begin(pool.flush_begin);
    std::atomic_ref<std::uint64_t> flush_serial(pool.flush_serial);
    flush_end.store(0, std::memory_order_release);
    flush_begin.store(begin, std::memory_order_relaxed);
    flush_serial.store(next_resource_serial(), std::memory_order_relaxed);
    flush_end.store(begin + size, std::memory_order_release);
}

// [offset, offset + size) inside [0, pool_size), without overflow.
bool range_inside(std::uint64_t offset, std::uint64_t size, std::uint64_t pool_size) noexcept {
    return offset <= pool_size && size <= pool_size - offset;
}

// ---------------------------------------------------------------- memory pool builder

void pool_builder_set_device(NVNmemoryPoolBuilder* builder, NVNdevice* device) noexcept {
    builder_of(builder)->device = device;
}

void pool_builder_set_defaults(NVNmemoryPoolBuilder* builder) noexcept {
    *builder_of(builder) = MemoryPoolBuilderState{};
}

void pool_builder_set_storage(NVNmemoryPoolBuilder* builder, void* memory, std::size_t size) noexcept {
    MemoryPoolBuilderState* state = builder_of(builder);
    state->memory = memory;
    state->size = size;
}

void pool_builder_set_flags(NVNmemoryPoolBuilder* builder, int flags) noexcept {
    builder_of(builder)->flags = static_cast<std::uint32_t>(flags);
}

void* pool_builder_get_memory(const NVNmemoryPoolBuilder* builder) noexcept {
    return builder_of(builder)->memory;
}

std::size_t pool_builder_get_size(const NVNmemoryPoolBuilder* builder) noexcept {
    return static_cast<std::size_t>(builder_of(builder)->size);
}

NVNmemoryPoolFlags pool_builder_get_flags(const NVNmemoryPoolBuilder* builder) noexcept {
    return static_cast<NVNmemoryPoolFlags>(builder_of(builder)->flags);
}

NVNdevice* pool_builder_get_device(const NVNmemoryPoolBuilder* builder) noexcept {
    return builder_of(builder)->device;
}

// ---------------------------------------------------------------- memory pool

NVNboolean pool_initialize(NVNmemoryPool* pool, const NVNmemoryPoolBuilder* builder) noexcept {
    if (!pool || !builder) {
        report_once<__LINE__>("nvnMemoryPoolInitialize requires a pool and a builder");
        return 0;
    }
    MemoryPoolState* state = state_of(pool);
    // Re-initializing a live pool releases the previous registration first,
    // exactly as Finalize would, so no backend registration is leaked.
    if (state->tag == live_pool_tag) {
        report_once<__LINE__>("nvnMemoryPoolInitialize on a live pool without Finalize; the previous "
                              "registration was released");
        release_registration(*state);
        state->tag = 0;
    }
    const MemoryPoolBuilderState* source = builder_of(builder);
    state->flags = source->flags;
    state->memory = static_cast<std::uint8_t*>(source->memory);
    state->size = source->size;
    state->flush_begin = 0;
    state->flush_end = 0;
    state->flush_serial = 0;
    if (!state->memory && !(state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL))
        report_once<__LINE__>("memory pool without storage and without the virtual flag; initialized with no "
                              "backing as the native driver does");
    if (state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)
        report_once<__LINE__>("virtual memory pool created; virtual mappings are not implemented");
    state->tag = live_pool_tag;
    if (state->memory && backends().resources) {
        if (!on_host_stack([&] { return backends().resources->register_pool(*state); })) {
            report_once<__LINE__>("the resource backend rejected a memory pool registration");
            state->tag = 0;
            return 0;
        }
    }
    return 1;
}

void pool_finalize(NVNmemoryPool* pool) noexcept {
    MemoryPoolState* state = state_of(pool);
    if (!state || state->tag != live_pool_tag) return;
    release_registration(*state);
    state->tag = 0;
}

void pool_set_debug_label(NVNmemoryPool*, const char*) noexcept {}

void* pool_map(const NVNmemoryPool* pool) noexcept {
    const MemoryPoolState* state = state_of(pool);
    if (!state || (state->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)))
        return nullptr;
    return state->memory;
}

void pool_flush_mapped_range(const NVNmemoryPool* pool, std::ptrdiff_t offset, std::size_t size) noexcept {
    const MemoryPoolState* state = state_of(pool);
    if (!state || state->tag != live_pool_tag) {
        report_once<__LINE__>("nvnMemoryPoolFlushMappedRange on a pool that is not initialized");
        return;
    }
    // The native driver flushes the CPU data cache only for CPU-cached pools.
    if (!(state->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) || offset < 0 || size == 0) return;
    const std::uint64_t begin = static_cast<std::uint64_t>(offset);
    if (!range_inside(begin, size, state->size)) {
        report_once<__LINE__>("nvnMemoryPoolFlushMappedRange outside the pool; ignored");
        return;
    }
    if (!state->memory) {
        report_once<__LINE__>("nvnMemoryPoolFlushMappedRange on a pool without CPU storage; ignored");
        return;
    }
    publish_flush(*const_cast<MemoryPoolState*>(state), begin, size);
    if (backends().resources) on_host_stack([&] { backends().resources->flush_pool(*state, begin, size); });
}

void pool_invalidate_mapped_range(const NVNmemoryPool* pool, std::ptrdiff_t offset, std::size_t size) noexcept {
    const MemoryPoolState* state = state_of(pool);
    if (!state || state->tag != live_pool_tag) {
        report_once<__LINE__>("nvnMemoryPoolInvalidateMappedRange on a pool that is not initialized; fenced");
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }
    // The native driver maintains CPU caches only for CPU-cached pools.
    if (!(state->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) || size == 0) return;
    if (!range_inside(static_cast<std::uint64_t>(offset), size, state->size)) {
        report_once<__LINE__>("nvnMemoryPoolInvalidateMappedRange outside the pool; fenced");
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }
    if (!state->memory || (state->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL))) {
        report_once<__LINE__>("nvnMemoryPoolInvalidateMappedRange on a pool without CPU-accessible backing; "
                              "fenced and skipped");
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (backends().resources)
        on_host_stack([&] { backends().resources->invalidate_pool(*state, static_cast<std::uint64_t>(offset), size); });
}

// Whole-pool addresses name the pool: the pool's buffer address, a buffer's
// address, and buffer initialization. The same pool is marked once on this
// thread until a different pool is marked. A texture range and a virtual
// alias are marked at the call, so they do not skip a later whole-pool mark.
void mark_pool_addressed(const MemoryPoolState* pool) noexcept {
    if (!pool || !pool->memory || !pool->size || !backends().resources) return;
    static thread_local const std::uint8_t* last = nullptr;
    if (pool->memory == last) return;
    last = pool->memory;
    const auto address = reinterpret_cast<std::uint64_t>(pool->memory);
    const auto size = pool->size;
    on_host_stack([&] { backends().resources->note_addressed(address, size); });
}

NVNbufferAddress pool_get_buffer_address(const NVNmemoryPool* pool) noexcept {
    const MemoryPoolState* state = state_of(pool);
    if (!state || (state->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL)) return 0;
    mark_pool_addressed(state);
    return reinterpret_cast<NVNbufferAddress>(state->memory);
}

std::size_t pool_get_size(const NVNmemoryPool* pool) noexcept {
    const MemoryPoolState* state = state_of(pool);
    return state ? static_cast<std::size_t>(state->size) : 0;
}

NVNmemoryPoolFlags pool_get_flags(const NVNmemoryPool* pool) noexcept {
    const MemoryPoolState* state = state_of(pool);
    return state ? static_cast<NVNmemoryPoolFlags>(state->flags) : static_cast<NVNmemoryPoolFlags>(0);
}

// ---------------------------------------------------------------- buffer builder

void buffer_builder_set_device(NVNbufferBuilder* builder, NVNdevice* device) noexcept {
    builder_of(builder)->device = device;
}

// Defaults keep the device, like the other builders that carry one.
void buffer_builder_set_defaults(NVNbufferBuilder* builder) noexcept {
    BufferBuilderState* state = builder_of(builder);
    state->pool = nullptr;
    state->offset = 0;
    state->size = 0;
}

void buffer_builder_set_storage(NVNbufferBuilder* builder, NVNmemoryPool* pool, std::ptrdiff_t offset,
                                std::size_t size) noexcept {
    BufferBuilderState* state = builder_of(builder);
    state->pool = state_of(pool);
    state->offset = offset;
    state->size = size;
}

NVNdevice* buffer_builder_get_device(const NVNbufferBuilder* builder) noexcept {
    return builder_of(builder)->device;
}

NVNmemoryPool* buffer_builder_get_memory_pool(const NVNbufferBuilder* builder) noexcept {
    return reinterpret_cast<NVNmemoryPool*>(builder_of(builder)->pool);
}

std::ptrdiff_t buffer_builder_get_memory_offset(const NVNbufferBuilder* builder) noexcept {
    return static_cast<std::ptrdiff_t>(builder_of(builder)->offset);
}

std::size_t buffer_builder_get_size(const NVNbufferBuilder* builder) noexcept {
    return static_cast<std::size_t>(builder_of(builder)->size);
}

// ---------------------------------------------------------------- buffer

NVNboolean buffer_initialize(NVNbuffer* buffer, const NVNbufferBuilder* builder) noexcept {
    if (!buffer || !builder) {
        report_once<__LINE__>("nvnBufferInitialize requires a buffer and a builder");
        return 0;
    }
    const BufferBuilderState* source = builder_of(builder);
    if (!source->pool || source->pool->tag != live_pool_tag) {
        report_once<__LINE__>("nvnBufferInitialize without an initialized memory pool");
        return 0;
    }
    // The native driver does not check the storage against the pool: only the
    // ranges actually accessed matter (the acceptance workload reads back a
    // buffer larger than its pool). Report the misuse, accept the storage.
    if (source->offset < 0 || !range_inside(static_cast<std::uint64_t>(source->offset), source->size,
                                            source->pool->size))
        report_once<__LINE__>("buffer storage outside its memory pool; accepted as the native driver does");
    if (source->pool->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        report_once<__LINE__>("buffers cannot use a physical-only memory pool");
        return 0;
    }
    if (source->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)
        report_once<__LINE__>("buffer in a virtual memory pool; virtual mappings are not implemented");
    BufferState* state = state_of(buffer);
    *state = {live_buffer_tag, 0,           source->device, source->pool,
              source->offset,  source->size, next_resource_serial()};
    mark_pool_addressed(source->pool);
    return 1;
}

void buffer_set_debug_label(NVNbuffer*, const char*) noexcept {}

void buffer_finalize(NVNbuffer* buffer) noexcept {
    if (BufferState* state = state_of(buffer)) state->tag = 0;
}

void* buffer_map(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    if (!state || state->tag != live_buffer_tag || !state->pool->memory ||
        (state->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)))
        return nullptr;
    return state->pool->memory + state->offset;
}

NVNbufferAddress buffer_get_address(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    if (!state || state->tag != live_buffer_tag || !state->pool->memory) return 0;
    mark_pool_addressed(state->pool);
    return reinterpret_cast<NVNbufferAddress>(state->pool->memory + state->offset);
}

// Buffer cache operations follow pool semantics: CPU-cached pools only, and
// the backend is told about the pool range the buffer range covers. The
// texture-ordering flush state is not written: it orders pool flushes against
// texture creation, not buffer traffic.
void buffer_flush_mapped_range(const NVNbuffer* buffer, std::ptrdiff_t offset, std::size_t size) noexcept {
    const BufferState* state = state_of(buffer);
    if (!state || state->tag != live_buffer_tag || offset < 0 ||
        !range_inside(static_cast<std::uint64_t>(offset), size, state->size)) {
        report_once<__LINE__>("nvnBufferFlushMappedRange out of bounds; ignored");
        return;
    }
    if (!(state->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) || !state->pool->memory) return;
    if (backends().resources)
        on_host_stack([&] {
            backends().resources->flush_pool(
                *state->pool, static_cast<std::uint64_t>(state->offset) + static_cast<std::uint64_t>(offset), size);
        });
}

void buffer_invalidate_mapped_range(const NVNbuffer* buffer, std::ptrdiff_t offset, std::size_t size) noexcept {
    const BufferState* state = state_of(buffer);
    if (!state || state->tag != live_buffer_tag || offset < 0 ||
        !range_inside(static_cast<std::uint64_t>(offset), size, state->size)) {
        report_once<__LINE__>("nvnBufferInvalidateMappedRange out of bounds; ignored");
        return;
    }
    if (!(state->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) || !state->pool->memory) return;
    if (backends().resources)
        on_host_stack([&] {
            backends().resources->invalidate_pool(
                *state->pool, static_cast<std::uint64_t>(state->offset) + static_cast<std::uint64_t>(offset), size);
        });
}

NVNmemoryPool* buffer_get_memory_pool(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    return state ? reinterpret_cast<NVNmemoryPool*>(state->pool) : nullptr;
}

std::ptrdiff_t buffer_get_memory_offset(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    return state ? static_cast<std::ptrdiff_t>(state->offset) : 0;
}

std::size_t buffer_get_size(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    return state ? static_cast<std::size_t>(state->size) : 0;
}

std::uint64_t buffer_get_debug_id(const NVNbuffer* buffer) noexcept {
    const BufferState* state = state_of(buffer);
    return state ? state->serial : 0;
}

} // namespace

void register_memory_entries() noexcept {
    install<Entry::nvnMemoryPoolBuilderGetDevice>(&pool_builder_get_device);
    install<Entry::nvnMemoryPoolBuilderSetDevice>(&pool_builder_set_device);
    install<Entry::nvnMemoryPoolBuilderSetDefaults>(&pool_builder_set_defaults);
    install<Entry::nvnMemoryPoolBuilderSetStorage>(&pool_builder_set_storage);
    install<Entry::nvnMemoryPoolBuilderSetFlags>(&pool_builder_set_flags);
    install<Entry::nvnMemoryPoolBuilderGetMemory>(&pool_builder_get_memory);
    install<Entry::nvnMemoryPoolBuilderGetSize>(&pool_builder_get_size);
    install<Entry::nvnMemoryPoolBuilderGetFlags>(&pool_builder_get_flags);
    install<Entry::nvnMemoryPoolInitialize>(&pool_initialize);
    install<Entry::nvnMemoryPoolSetDebugLabel>(&pool_set_debug_label);
    install<Entry::nvnMemoryPoolFinalize>(&pool_finalize);
    install<Entry::nvnMemoryPoolMap>(&pool_map);
    install<Entry::nvnMemoryPoolFlushMappedRange>(&pool_flush_mapped_range);
    install<Entry::nvnMemoryPoolInvalidateMappedRange>(&pool_invalidate_mapped_range);
    install<Entry::nvnMemoryPoolGetBufferAddress>(&pool_get_buffer_address);
    install<Entry::nvnMemoryPoolGetSize>(&pool_get_size);
    install<Entry::nvnMemoryPoolGetFlags>(&pool_get_flags);
    install<Entry::nvnBufferBuilderSetDevice>(&buffer_builder_set_device);
    install<Entry::nvnBufferBuilderSetDefaults>(&buffer_builder_set_defaults);
    install<Entry::nvnBufferBuilderSetStorage>(&buffer_builder_set_storage);
    install<Entry::nvnBufferBuilderGetDevice>(&buffer_builder_get_device);
    install<Entry::nvnBufferBuilderGetMemoryPool>(&buffer_builder_get_memory_pool);
    install<Entry::nvnBufferBuilderGetMemoryOffset>(&buffer_builder_get_memory_offset);
    install<Entry::nvnBufferBuilderGetSize>(&buffer_builder_get_size);
    install<Entry::nvnBufferInitialize>(&buffer_initialize);
    install<Entry::nvnBufferSetDebugLabel>(&buffer_set_debug_label);
    install<Entry::nvnBufferFinalize>(&buffer_finalize);
    install<Entry::nvnBufferMap>(&buffer_map);
    install<Entry::nvnBufferGetAddress>(&buffer_get_address);
    install<Entry::nvnBufferFlushMappedRange>(&buffer_flush_mapped_range);
    install<Entry::nvnBufferInvalidateMappedRange>(&buffer_invalidate_mapped_range);
    install<Entry::nvnBufferGetMemoryPool>(&buffer_get_memory_pool);
    install<Entry::nvnBufferGetMemoryOffset>(&buffer_get_memory_offset);
    install<Entry::nvnBufferGetSize>(&buffer_get_size);
    install<Entry::nvnBufferGetDebugID>(&buffer_get_debug_id);
}

} // namespace armigo::nvn
