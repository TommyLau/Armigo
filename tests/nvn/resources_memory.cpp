// ============================================================================
// Armigo Tests - NVN Device and Memory Resources
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic verification of the device and memory pool/buffer entry points,
// driven through the bootstrap loader exactly as the game reaches them, with
// a recording mock in place of the resource backend. Object states are
// inspected through the in-place layouts of objects.h.

#include "backend.h"
#include "entry_table.h"
#include "objects.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

using namespace armigo::nvn;
using namespace api;

} // namespace

// Installed by the device and memory subsystems; the shared registration
// entry that calls both belongs to the integrator's file.
namespace armigo::nvn {
void register_device_entries() noexcept;
void register_memory_entries() noexcept;
}

namespace {

unsigned failures = 0;
#define CHECK(condition)                                                  \
    do {                                                                  \
        if (!(condition)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
            ++failures;                                                   \
        }                                                                 \
    } while (false)

// ---------------------------------------------------------------- mock backend

struct PoolRange {
    const MemoryPoolState* pool;
    std::uint64_t offset, size;
};

struct MockResources final : ResourceBackend {
    std::vector<const MemoryPoolState*> registered, released;
    std::vector<PoolRange> flushes, invalidates;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> addressed;
    bool accept_pools = true;

    void note_addressed(std::uint64_t address, std::uint64_t size) override {
        addressed.push_back({address, size});
    }

    bool register_pool(const MemoryPoolState& pool) override {
        if (!accept_pools) return false;
        registered.push_back(&pool);
        return true;
    }
    void release_pool(const MemoryPoolState& pool) override { released.push_back(&pool); }
    void flush_pool(const MemoryPoolState& pool, std::uint64_t offset, std::uint64_t size) override {
        flushes.push_back({&pool, offset, size});
    }
    void invalidate_pool(const MemoryPoolState& pool, std::uint64_t offset, std::uint64_t size) override {
        invalidates.push_back({&pool, offset, size});
    }
    void register_texture(const TextureState&) override {}
    void release_texture(const TextureState&) override {}
    void clear_texture_descriptor(std::int32_t) override { CHECK(false); }
    void register_texture_descriptor(std::int32_t, const TextureState&, const TextureViewState*) override {}
    void register_sampler_descriptor(std::int32_t, const SamplerParameters&) override {}
    bool read_texels(const TextureState&, const TexelTransfer&, std::byte*) override { return false; }
    bool write_texels(const TextureState&, const TexelTransfer&, const std::byte*) override { return false; }
};

template <typename Signature>
Signature load(const char* name) {
    return reinterpret_cast<Signature>(armigo_nvn_bootstrap_loader(name));
}

void debug_callback(NVNdebugCallbackSource, NVNdebugCallbackType, int, NVNdebugCallbackSeverity,
                    const unsigned char*, void*) {}

} // namespace

int main() {
    MockResources mock;
    set_backends(Backends{&mock, nullptr});
    register_device_entries();
    register_memory_entries();
    select_api_version({55, 15});

    // Every entry point this suite touches must be installed, not a stub.
    for (const char* name : {"nvnDeviceInitialize", "nvnDeviceGetInteger", "nvnDeviceGetTextureHandle",
                             "nvnMemoryPoolInitialize", "nvnMemoryPoolFlushMappedRange", "nvnBufferInitialize",
                             "nvnBufferGetAddress"})
        CHECK(load<void (*)()>(name) != nullptr);

    // ---------------------------------------------------------------- device

    const auto builder_set_defaults = load<void (*)(NVNdeviceBuilder*)>("nvnDeviceBuilderSetDefaults");
    const auto builder_set_flags = load<void (*)(NVNdeviceBuilder*, int)>("nvnDeviceBuilderSetFlags");
    const auto device_initialize =
        load<NVNboolean (*)(NVNdevice*, const NVNdeviceBuilder*)>("nvnDeviceInitialize");
    const auto device_finalize = load<void (*)(NVNdevice*)>("nvnDeviceFinalize");
    const auto device_get_proc_address =
        load<PFNNVNGENERICFUNCPTRPROC (*)(const NVNdevice*, const char*)>("nvnDeviceGetProcAddress");
    const auto device_get_integer = load<void (*)(const NVNdevice*, NVNdeviceInfo, int*)>("nvnDeviceGetInteger");
    const auto get_texture_handle =
        load<NVNtextureHandle (*)(const NVNdevice*, int, int)>("nvnDeviceGetTextureHandle");
    const auto get_texel_fetch_handle =
        load<NVNtextureHandle (*)(const NVNdevice*, int)>("nvnDeviceGetTexelFetchHandle");
    const auto get_image_handle = load<NVNimageHandle (*)(const NVNdevice*, int)>("nvnDeviceGetImageHandle");
    const auto get_separate_texture =
        load<NVNseparateTextureHandle (*)(const NVNdevice*, int)>("nvnDeviceGetSeparateTextureHandle");
    const auto get_separate_sampler =
        load<NVNseparateSamplerHandle (*)(const NVNdevice*, int)>("nvnDeviceGetSeparateSamplerHandle");
    const auto install_debug_callback = load<void (*)(NVNdevice*, const PFNNVNDEBUGCALLBACKPROC, void*,
                                                      std::uint32_t)>("nvnDeviceInstallDebugCallback");
    const auto set_origin = load<void (*)(NVNdevice*, NVNwindowOriginMode)>("nvnDeviceSetWindowOriginMode");
    const auto get_origin = load<NVNwindowOriginMode (*)(const NVNdevice*)>("nvnDeviceGetWindowOriginMode");
    const auto set_depth = load<void (*)(NVNdevice*, NVNdepthMode)>("nvnDeviceSetDepthMode");
    const auto get_depth = load<NVNdepthMode (*)(const NVNdevice*)>("nvnDeviceGetDepthMode");

    NVNdeviceBuilder builder;
    builder_set_defaults(&builder);
    CHECK(reinterpret_cast<DeviceBuilderState*>(&builder)->flags == 0);
    builder_set_flags(&builder, 0x1234);
    CHECK(load<int (*)(const NVNdeviceBuilder*)>("nvnDeviceBuilderGetFlags")(&builder) == 0x1234);

    NVNdevice device_storage;
    NVNdevice* const device = &device_storage;
    std::memset(device, 0xab, sizeof *device);  // Initialize must not depend on prior contents
    CHECK(device_initialize(device, &builder) == 1);
    const DeviceState* device_state = reinterpret_cast<const DeviceState*>(device);
    CHECK(device_state->tag == object_tag('D', 'E', 'V', 0));
    CHECK(device_state->flags == 0x1234);
    CHECK(device_state->window_origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT);
    CHECK(device_state->depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W);
    CHECK(device_state->debug_callback == nullptr && device_state->debug_data == nullptr);

    set_origin(device, NVN_WINDOW_ORIGIN_MODE_UPPER_LEFT);
    CHECK(get_origin(device) == NVN_WINDOW_ORIGIN_MODE_UPPER_LEFT);
    set_depth(device, NVN_DEPTH_MODE_NEAR_IS_ZERO);
    CHECK(get_depth(device) == NVN_DEPTH_MODE_NEAR_IS_ZERO);

    install_debug_callback(device, &debug_callback, device, 1);
    CHECK(device_state->debug_callback == reinterpret_cast<const void*>(&debug_callback));
    CHECK(device_state->debug_data == device);
    install_debug_callback(device, nullptr, nullptr, 0x100);  // only the low bits are defined
    CHECK(device_state->debug_callback == nullptr);

    CHECK(device_get_proc_address(device, "nvnDeviceGetInteger") != nullptr);
    CHECK(device_get_proc_address(device, "nvnNotAnEntryPoint") == nullptr);

    const struct {
        NVNdeviceInfo info;
        int value;
    } expected[] = {
        {NVN_DEVICE_INFO_API_MAJOR_VERSION, 55},
        {NVN_DEVICE_INFO_API_MINOR_VERSION, 15},
        {NVN_DEVICE_INFO_MAX_UNIFORM_BUFFER_SIZE, 65536},
        {NVN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, 32},
        {NVN_DEVICE_INFO_MAX_TEXTURE_POOL_SIZE, 1048576},
        {NVN_DEVICE_INFO_MAX_TEXTURE_LEVELS, 15},
        {NVN_DEVICE_INFO_MEMORY_POOL_PAGE_SIZE, 65536},
        {NVN_DEVICE_INFO_MAX_TEXTURE_SIZE, 16384},
        {NVN_DEVICE_INFO_MAX_TEXTURES_PER_WINDOW, 4},
        {NVN_DEVICE_INFO_SEPARATE_TEXTURE_BINDINGS_PER_STAGE, 128},
        {NVN_DEVICE_INFO_SEPARATE_SAMPLER_BINDINGS_PER_STAGE, 32},
        {NVN_DEVICE_INFO_DEBUG_GROUPS_MAX_DOMAIN_ID, 65535},
        {NVN_DEVICE_INFO_EVENTS_SUPPORT_REDUCTION_OPERATIONS, 1},
    };
    for (const auto& item : expected) {
        int value = -1;
        device_get_integer(device, item.info, &value);
        CHECK(value == item.value);
    }
    int value = -1;
    device_get_integer(device, static_cast<NVNdeviceInfo>(0x62), &value);
    CHECK(value == 0);
    device_get_integer(device, NVN_DEVICE_INFO_LARGE, &value);
    CHECK(value == 0);

    const NVNtextureHandle handle = get_texture_handle(device, 0x12345, 0x678);
    CHECK(handle == combined_handle(0x12345, 0x678));
    CHECK(handle_texture_id(handle) == 0x12345 && handle_sampler_id(handle) == 0x678);
    CHECK(get_texel_fetch_handle(device, 7) == texture_handle(7));
    CHECK(get_separate_texture(device, 9).value == texture_handle(9));
    CHECK(get_separate_sampler(device, 5).value == sampler_handle(5));
    CHECK(get_image_handle(device, 3) == 3);

    // The device reports the selected API version; other properties stay.
    select_api_version({55, 12});
    device_get_integer(device, NVN_DEVICE_INFO_API_MINOR_VERSION, &value);
    CHECK(value == 12);
    device_get_integer(device, NVN_DEVICE_INFO_API_MAJOR_VERSION, &value);
    CHECK(value == 55);
    device_get_integer(device, NVN_DEVICE_INFO_MAX_TEXTURE_LEVELS, &value);
    CHECK(value == 15);
    select_api_version({55, 15});
    device_get_integer(device, NVN_DEVICE_INFO_API_MINOR_VERSION, &value);
    CHECK(value == 15);

    // Fast-clear registration succeeds on a live device and changes nothing
    // else; a missing value or device is refused.
    const auto fast_color = reinterpret_cast<NVNboolean (*)(NVNdevice*, const float*, NVNformat)>(
        armigo_nvn_bootstrap_loader("nvnDeviceRegisterFastClearColor"));
    const auto fast_colori = reinterpret_cast<NVNboolean (*)(NVNdevice*, const int*, NVNformat)>(
        armigo_nvn_bootstrap_loader("nvnDeviceRegisterFastClearColori"));
    const auto fast_colorui = reinterpret_cast<NVNboolean (*)(NVNdevice*, const std::uint32_t*, NVNformat)>(
        armigo_nvn_bootstrap_loader("nvnDeviceRegisterFastClearColorui"));
    const auto fast_depth = reinterpret_cast<NVNboolean (*)(NVNdevice*, float)>(
        armigo_nvn_bootstrap_loader("nvnDeviceRegisterFastClearDepth"));
    const float zero[4] = {};
    const int zero_i[4] = {};
    const std::uint32_t zero_u[4] = {};
    CHECK(fast_color && fast_color(device, zero, NVN_FORMAT_RGBA8) == 1);
    CHECK(fast_colori && fast_colori(device, zero_i, NVN_FORMAT_RGBA8I) == 1);
    CHECK(fast_colorui && fast_colorui(device, zero_u, NVN_FORMAT_RGBA8UI) == 1);
    CHECK(fast_depth && fast_depth(device, 1.0f) == 1);
    CHECK(fast_color && fast_color(device, nullptr, NVN_FORMAT_RGBA8) == 0);
    CHECK(fast_depth && fast_depth(nullptr, 1.0f) == 0);

    device_finalize(device);
    CHECK(fast_depth && fast_depth(device, 1.0f) == 0);
    CHECK(device_state->tag == 0);

    // ---------------------------------------------------------------- memory pool

    const auto pool_set_device = load<void (*)(NVNmemoryPoolBuilder*, NVNdevice*)>("nvnMemoryPoolBuilderSetDevice");
    const auto pool_set_defaults = load<void (*)(NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderSetDefaults");
    const auto pool_set_storage =
        load<void (*)(NVNmemoryPoolBuilder*, void*, std::size_t)>("nvnMemoryPoolBuilderSetStorage");
    const auto pool_set_flags = load<void (*)(NVNmemoryPoolBuilder*, int)>("nvnMemoryPoolBuilderSetFlags");
    const auto pool_builder_get_memory = load<void* (*)(const NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderGetMemory");
    const auto pool_builder_get_size =
        load<std::size_t (*)(const NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderGetSize");
    const auto pool_builder_get_flags =
        load<NVNmemoryPoolFlags (*)(const NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderGetFlags");
    const auto pool_initialize =
        load<NVNboolean (*)(NVNmemoryPool*, const NVNmemoryPoolBuilder*)>("nvnMemoryPoolInitialize");
    const auto pool_finalize = load<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize");
    const auto pool_map = load<void* (*)(const NVNmemoryPool*)>("nvnMemoryPoolMap");
    const auto pool_flush =
        load<void (*)(const NVNmemoryPool*, std::ptrdiff_t, std::size_t)>("nvnMemoryPoolFlushMappedRange");
    const auto pool_invalidate =
        load<void (*)(const NVNmemoryPool*, std::ptrdiff_t, std::size_t)>("nvnMemoryPoolInvalidateMappedRange");
    const auto pool_get_address =
        load<NVNbufferAddress (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetBufferAddress");
    const auto pool_get_size = load<std::size_t (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetSize");
    const auto pool_get_flags = load<NVNmemoryPoolFlags (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetFlags");

    void* pool_memory = std::aligned_alloc(0x1000, 0x3000);
    CHECK(pool_memory != nullptr);

    NVNmemoryPoolBuilder pool_builder_storage;
    NVNmemoryPoolBuilder* const pool_builder = &pool_builder_storage;
    pool_set_defaults(pool_builder);
    CHECK(reinterpret_cast<MemoryPoolBuilderState*>(pool_builder)->device == nullptr);
    pool_set_device(pool_builder, device);
    pool_set_storage(pool_builder, pool_memory, 0x3000);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_builder_get_memory(pool_builder) == pool_memory);
    CHECK(pool_builder_get_size(pool_builder) == 0x3000);
    CHECK(pool_builder_get_flags(pool_builder) ==
          (NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED));

    NVNmemoryPool pool_storage;
    NVNmemoryPool* const pool = &pool_storage;
    std::memset(pool, 0xcd, sizeof *pool);
    CHECK(pool_initialize(pool, pool_builder) == 1);
    const MemoryPoolState* pool_state = reinterpret_cast<const MemoryPoolState*>(pool);
    CHECK(pool_state->tag == object_tag('P', 'O', 'O', 'L'));
    CHECK(mock.registered.size() == 1 && mock.registered[0] == pool_state);
    CHECK(pool_map(pool) == pool_memory);
    CHECK(pool_get_address(pool) == reinterpret_cast<NVNbufferAddress>(pool_memory));
    CHECK(mock.addressed.size() == 1 &&
          mock.addressed[0].first == reinterpret_cast<std::uint64_t>(pool_memory) &&
          mock.addressed[0].second == 0x3000);
    CHECK(pool_get_address(pool) == reinterpret_cast<NVNbufferAddress>(pool_memory));
    CHECK(mock.addressed.size() == 1);
    CHECK(pool_get_size(pool) == 0x3000);
    CHECK(pool_get_flags(pool) == (NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED));
    CHECK(pool_state->flush_end == 0 && pool_state->flush_serial == 0);

    // A flush publishes the range (with a serial) and reaches the backend.
    pool_flush(pool, 0x1000, 0x800);
    CHECK(mock.flushes.size() == 1 && mock.flushes[0].pool == pool_state);
    CHECK(mock.flushes[0].offset == 0x1000 && mock.flushes[0].size == 0x800);
    CHECK(pool_state->flush_begin == 0x1000 && pool_state->flush_end == 0x1800);
    const std::uint64_t first_flush_serial = pool_state->flush_serial;
    CHECK(first_flush_serial > 0);
    pool_flush(pool, 0, 0x100);
    CHECK(mock.flushes.size() == 2);
    CHECK(pool_state->flush_begin == 0 && pool_state->flush_end == 0x100);
    CHECK(pool_state->flush_serial > first_flush_serial);

    // Out-of-range and empty flushes are ignored; an uncached pool never flushes.
    pool_flush(pool, 0x2000, 0x2000);
    pool_flush(pool, 0, 0);
    pool_flush(pool, -1, 0x10);
    CHECK(mock.flushes.size() == 2);

    pool_invalidate(pool, 0x800, 0x100);
    CHECK(mock.invalidates.size() == 1 && mock.invalidates[0].pool == pool_state);
    CHECK(mock.invalidates[0].offset == 0x800 && mock.invalidates[0].size == 0x100);
    pool_invalidate(pool, 0x2f00, 0x200);
    pool_invalidate(pool, 0, 0);
    CHECK(mock.invalidates.size() == 1);

    // A second buffer initialized before the next flush takes a smaller serial
    // (buffer debug IDs and pool flush serials share one counter).

    // Re-initializing a live pool releases the old registration first.
    CHECK(pool_initialize(pool, pool_builder) == 1);
    CHECK(mock.released.size() == 1 && mock.released[0] == pool_state);
    CHECK(mock.registered.size() == 2);
    CHECK(pool_state->flush_end == 0 && pool_state->flush_serial == 0);

    pool_finalize(pool);
    CHECK(mock.released.size() == 2);
    CHECK(pool_state->tag == 0);

    // Virtual pools exist as objects but expose no backing and no GPU address.
    pool_set_defaults(pool_builder);
    pool_set_storage(pool_builder, nullptr, 0);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_initialize(pool, pool_builder) == 1);
    CHECK(pool_map(pool) == nullptr);
    CHECK(pool_get_address(pool) == 0);
    CHECK(mock.addressed.size() == 1);
    CHECK(pool_get_flags(pool) == (NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_GPU_CACHED));
    CHECK(mock.registered.size() == 2);  // no backing: nothing registered
    pool_flush(pool, 0, 0x100);
    CHECK(mock.flushes.size() == 2);
    pool_finalize(pool);

    // Physical-only pools report no buffer address; the backend may also
    // reject a registration outright, which fails initialization.
    pool_set_defaults(pool_builder);
    pool_set_storage(pool_builder, pool_memory, 0x1000);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_PHYSICAL | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_initialize(pool, pool_builder) == 1);
    CHECK(pool_get_address(pool) == 0);
    CHECK(mock.addressed.size() == 1);
    pool_finalize(pool);
    mock.accept_pools = false;
    CHECK(pool_initialize(pool, pool_builder) == 0);
    CHECK(reinterpret_cast<MemoryPoolState*>(pool)->tag == 0);
    mock.accept_pools = true;

    // ---------------------------------------------------------------- buffers

    const auto buffer_set_device = load<void (*)(NVNbufferBuilder*, NVNdevice*)>("nvnBufferBuilderSetDevice");
    const auto buffer_set_defaults = load<void (*)(NVNbufferBuilder*)>("nvnBufferBuilderSetDefaults");
    const auto buffer_set_storage = load<void (*)(NVNbufferBuilder*, NVNmemoryPool*, std::ptrdiff_t, std::size_t)>(
        "nvnBufferBuilderSetStorage");
    const auto buffer_builder_get_device = load<NVNdevice* (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetDevice");
    const auto buffer_builder_get_pool =
        load<NVNmemoryPool* (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetMemoryPool");
    const auto buffer_builder_get_offset =
        load<std::ptrdiff_t (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetMemoryOffset");
    const auto buffer_builder_get_size = load<std::size_t (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetSize");
    const auto buffer_initialize =
        load<NVNboolean (*)(NVNbuffer*, const NVNbufferBuilder*)>("nvnBufferInitialize");
    const auto buffer_finalize = load<void (*)(NVNbuffer*)>("nvnBufferFinalize");
    const auto buffer_map = load<void* (*)(const NVNbuffer*)>("nvnBufferMap");
    const auto buffer_get_address = load<NVNbufferAddress (*)(const NVNbuffer*)>("nvnBufferGetAddress");
    const auto buffer_flush =
        load<void (*)(const NVNbuffer*, std::ptrdiff_t, std::size_t)>("nvnBufferFlushMappedRange");
    const auto buffer_invalidate =
        load<void (*)(const NVNbuffer*, std::ptrdiff_t, std::size_t)>("nvnBufferInvalidateMappedRange");
    const auto buffer_get_pool = load<NVNmemoryPool* (*)(const NVNbuffer*)>("nvnBufferGetMemoryPool");
    const auto buffer_get_offset = load<std::ptrdiff_t (*)(const NVNbuffer*)>("nvnBufferGetMemoryOffset");
    const auto buffer_get_size = load<std::size_t (*)(const NVNbuffer*)>("nvnBufferGetSize");
    const auto buffer_get_debug_id = load<std::uint64_t (*)(const NVNbuffer*)>("nvnBufferGetDebugID");

    pool_set_defaults(pool_builder);
    pool_set_device(pool_builder, device);
    pool_set_storage(pool_builder, pool_memory, 0x3000);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_initialize(pool, pool_builder) == 1);

    NVNbufferBuilder buffer_builder_storage;
    NVNbufferBuilder* const buffer_builder = &buffer_builder_storage;
    std::memset(buffer_builder, 0xcd, sizeof *buffer_builder);
    buffer_set_defaults(buffer_builder);
    buffer_set_device(buffer_builder, device);
    buffer_set_defaults(buffer_builder);  // defaults keep the device
    CHECK(buffer_builder_get_device(buffer_builder) == device);
    CHECK(buffer_builder_get_pool(buffer_builder) == nullptr);
    CHECK(buffer_builder_get_offset(buffer_builder) == 0);
    CHECK(buffer_builder_get_size(buffer_builder) == 0);

    buffer_set_storage(buffer_builder, pool, 0x1000, 0x800);
    CHECK(buffer_builder_get_pool(buffer_builder) == pool);
    CHECK(buffer_builder_get_offset(buffer_builder) == 0x1000);
    CHECK(buffer_builder_get_size(buffer_builder) == 0x800);

    NVNbuffer buffer_storage;
    NVNbuffer* const buffer = &buffer_storage;
    std::memset(buffer, 0xcd, sizeof *buffer);
    CHECK(buffer_initialize(buffer, buffer_builder) == 1);
    NVNbuffer probe_two_storage;
    CHECK(buffer_initialize(&probe_two_storage, buffer_builder) == 1);
    // An initialized object's bytes, serial aside, do not depend on what the storage held before.
    {
        NVNbuffer zero_filled, one_filled;
        std::memset(&zero_filled, 0x00, sizeof zero_filled);
        std::memset(&one_filled, 0xff, sizeof one_filled);
        CHECK(buffer_initialize(&zero_filled, buffer_builder) == 1);
        CHECK(buffer_initialize(&one_filled, buffer_builder) == 1);
        CHECK(std::memcmp(&zero_filled, &one_filled, 0x28) == 0);
    }
    CHECK(buffer_get_debug_id(&probe_two_storage) > buffer_get_debug_id(buffer));

    CHECK(buffer_map(buffer) == static_cast<std::uint8_t*>(pool_memory) + 0x1000);
    CHECK(buffer_get_address(buffer) == reinterpret_cast<NVNbufferAddress>(static_cast<std::uint8_t*>(pool_memory) + 0x1000));
    // Initialization and the buffer address name this pool, which the pool
    // address above already marked on this thread.
    CHECK(mock.addressed.size() == 1);
    CHECK(buffer_get_pool(buffer) == pool);
    CHECK(buffer_get_offset(buffer) == 0x1000);
    CHECK(buffer_get_size(buffer) == 0x800);

    // Buffer cache operations forward the covering pool range to the backend.
    const std::size_t flushes_before = mock.flushes.size();
    buffer_flush(buffer, 0x100, 0x400);
    CHECK(mock.flushes.size() == flushes_before + 1);
    CHECK(mock.flushes.back().pool == pool_state && mock.flushes.back().offset == 0x1100 &&
          mock.flushes.back().size == 0x400);
    buffer_invalidate(buffer, 0x100, 0x400);
    CHECK(mock.invalidates.back().pool == pool_state && mock.invalidates.back().offset == 0x1100 &&
          mock.invalidates.back().size == 0x400);
    buffer_flush(buffer, 0, 0x1000);  // beyond the buffer
    buffer_flush(buffer, -8, 0x10);
    buffer_invalidate(buffer, 0x900, 1);
    CHECK(mock.flushes.size() == flushes_before + 1);

    // Storage outside the pool is accepted, as the native driver accepts it.
    buffer_set_storage(buffer_builder, pool, 0x2000, 0x8000);
    NVNbuffer beyond_storage;
    NVNbuffer* const beyond = &beyond_storage;
    CHECK(buffer_initialize(beyond, buffer_builder) == 1);
    CHECK(buffer_map(beyond) == static_cast<std::uint8_t*>(pool_memory) + 0x2000);
    buffer_finalize(beyond);

    // A physical-only pool cannot back a buffer.
    pool_set_defaults(pool_builder);
    pool_set_storage(pool_builder, pool_memory, 0x1000);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_PHYSICAL | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_initialize(pool, pool_builder) == 1);
    buffer_set_storage(buffer_builder, pool, 0, 0x100);
    NVNbuffer physical_storage;
    CHECK(buffer_initialize(&physical_storage, buffer_builder) == 0);
    pool_finalize(pool);

    // A buffer without an initialized pool fails; finalize clears the tag;
    // the flush serial shares the counter with buffer debug IDs.
    pool_set_defaults(pool_builder);
    pool_set_storage(pool_builder, pool_memory, 0x3000);
    pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(pool_initialize(pool, pool_builder) == 1);
    pool_flush(pool, 0, 0x40);
    CHECK(pool_state->flush_serial > buffer_get_debug_id(&probe_two_storage));
    buffer_finalize(buffer);
    CHECK(buffer_map(buffer) == nullptr);  // finalized: no live buffer state
    pool_finalize(pool);

    // A fresh pool is marked whole by initialization, and again by its buffer
    // address once another pool has been marked.
    void* left_memory = std::aligned_alloc(0x1000, 0x2000);
    void* right_memory = std::aligned_alloc(0x1000, 0x1000);
    CHECK(left_memory && right_memory);
    if (left_memory && right_memory) {
        const auto addressed = mock.addressed.size();
        NVNmemoryPool left_pool_storage{}, right_pool_storage{};
        pool_set_defaults(pool_builder);
        pool_set_device(pool_builder, device);
        pool_set_storage(pool_builder, left_memory, 0x2000);
        pool_set_flags(pool_builder, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
        CHECK(pool_initialize(&left_pool_storage, pool_builder) == 1);
        pool_set_storage(pool_builder, right_memory, 0x1000);
        CHECK(pool_initialize(&right_pool_storage, pool_builder) == 1);
        buffer_set_storage(buffer_builder, &left_pool_storage, 0x100, 0x40);
        NVNbuffer left_buffer{};
        CHECK(buffer_initialize(&left_buffer, buffer_builder) == 1);
        CHECK(mock.addressed.size() == addressed + 1);
        CHECK(mock.addressed.back().first == reinterpret_cast<std::uint64_t>(left_memory) &&
              mock.addressed.back().second == 0x2000);
        buffer_set_storage(buffer_builder, &right_pool_storage, 0, 0x20);
        NVNbuffer right_buffer{};
        CHECK(buffer_initialize(&right_buffer, buffer_builder) == 1);
        CHECK(mock.addressed.size() == addressed + 2);
        CHECK(mock.addressed.back().first == reinterpret_cast<std::uint64_t>(right_memory) &&
              mock.addressed.back().second == 0x1000);
        CHECK(buffer_get_address(&left_buffer) ==
              reinterpret_cast<NVNbufferAddress>(static_cast<std::uint8_t*>(left_memory) + 0x100));
        CHECK(mock.addressed.size() == addressed + 3);
        CHECK(mock.addressed.back().first == reinterpret_cast<std::uint64_t>(left_memory) &&
              mock.addressed.back().second == 0x2000);
        CHECK(buffer_get_address(&left_buffer) ==
              reinterpret_cast<NVNbufferAddress>(static_cast<std::uint8_t*>(left_memory) + 0x100));
        CHECK(mock.addressed.size() == addressed + 3);
        buffer_finalize(&left_buffer);
        buffer_finalize(&right_buffer);
        pool_finalize(&left_pool_storage);
        pool_finalize(&right_pool_storage);
    }
    std::free(left_memory);
    std::free(right_memory);

    std::free(pool_memory);
    std::printf("NVN device and memory resource failures: %u\n", failures);
    return failures ? 1 : 0;
}
