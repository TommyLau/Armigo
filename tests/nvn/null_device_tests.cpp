// ============================================================================
// Armigo Tests - NVN Null Device
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Stage-A null device behavior: device creation and properties, memory pools,
// buffers, texture builders with console storage layouts, queues, empty
// command recording, immediate syncs, cycling windows, program acceptance and
// the boot entry points. Entry points are resolved exactly as the game
// resolves them, through the redirected bootstrap loader.

#include "entry_table.h"
#include "redirect.h"
#include "objects.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                  \
    do {                                                                  \
        if (!(condition)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
            ++failures;                                                   \
        }                                                                 \
    } while (false)

template <typename Signature>
Signature entry(const char* name) {
    return reinterpret_cast<Signature>(armigo_nvn_bootstrap_loader(name));
}

using device_initialize_t = NVNboolean (*)(NVNdevice*, const NVNdeviceBuilder*);
using device_get_integer_t = void (*)(const NVNdevice*, NVNdeviceInfo, int*);

int query(device_get_integer_t get, const NVNdevice* device, int info) {
    int value = -1;
    get(device, static_cast<NVNdeviceInfo>(info), &value);
    return value;
}

// A pool backed by suitably aligned host memory, as the game's allocator
// would provide it.
struct TestPool {
    alignas(512) std::byte storage[0x4000];
    NVNmemoryPool pool{};
    NVNmemoryPoolBuilder builder{};

    TestPool() {
        auto set_device = entry<void (*)(NVNmemoryPoolBuilder*, NVNdevice*)>("nvnMemoryPoolBuilderSetDevice");
        auto set_storage = entry<void (*)(NVNmemoryPoolBuilder*, void*, std::size_t)>("nvnMemoryPoolBuilderSetStorage");
        auto set_flags = entry<void (*)(NVNmemoryPoolBuilder*, int)>("nvnMemoryPoolBuilderSetFlags");
        auto initialize = entry<NVNboolean (*)(NVNmemoryPool*, const NVNmemoryPoolBuilder*)>("nvnMemoryPoolInitialize");
        set_device(&builder, nullptr);
        set_storage(&builder, storage, sizeof storage);
        set_flags(&builder, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
        CHECK(initialize(&pool, &builder) == 1);
    }
};

void test_boot_and_device() {
    NVNdeviceBuilder builder{};
    NVNdevice device{};
    auto builder_defaults = entry<void (*)(NVNdeviceBuilder*)>("nvnDeviceBuilderSetDefaults");
    auto builder_flags = entry<void (*)(NVNdeviceBuilder*, int)>("nvnDeviceBuilderSetFlags");
    const auto initialize = entry<device_initialize_t>("nvnDeviceInitialize");
    auto finalize = entry<void (*)(NVNdevice*)>("nvnDeviceFinalize");
    const auto get_integer = entry<device_get_integer_t>("nvnDeviceGetInteger");

    builder_defaults(&builder);
    builder_flags(&builder, 0x8);
    CHECK(initialize(&device, &builder) == 1);
    const DeviceState* state = reinterpret_cast<const DeviceState*>(&device);
    CHECK(state->tag == object_tag('D', 'E', 'V', 0) && state->flags == 0x8);

    // API version of the null device and spot-checked console properties.
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_API_MAJOR_VERSION) == 55);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_API_MINOR_VERSION) == 15);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_MAX_TEXTURE_SIZE) == 16384);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_DEFAULT_SIZE) == 262144);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_MAX_TEXTURES_PER_WINDOW) == 4);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_SEPARATE_SAMPLER_BINDINGS_PER_STAGE) == 32);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_DEBUG_GROUPS_MAX_DOMAIN_ID) == 65535);
    CHECK(query(get_integer, &device, NVN_DEVICE_INFO_SUPPORTS_DEBUG_LAYER) == 0);
    CHECK(query(get_integer, &device, 0x100) == 0);  // outside the table
    CHECK(query(get_integer, &device, -1) == 0);

    // Modes default as the API defines and round-trip.
    auto set_origin = entry<void (*)(NVNdevice*, NVNwindowOriginMode)>("nvnDeviceSetWindowOriginMode");
    auto get_origin = entry<NVNwindowOriginMode (*)(const NVNdevice*)>("nvnDeviceGetWindowOriginMode");
    auto set_depth = entry<void (*)(NVNdevice*, NVNdepthMode)>("nvnDeviceSetDepthMode");
    auto get_depth = entry<NVNdepthMode (*)(const NVNdevice*)>("nvnDeviceGetDepthMode");
    CHECK(get_origin(&device) == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT);
    CHECK(get_depth(&device) == NVN_DEPTH_MODE_NEAR_IS_MINUS_W);
    set_origin(&device, NVN_WINDOW_ORIGIN_MODE_UPPER_LEFT);
    set_depth(&device, NVN_DEPTH_MODE_NEAR_IS_ZERO);
    CHECK(get_origin(&device) == NVN_WINDOW_ORIGIN_MODE_UPPER_LEFT);
    CHECK(get_depth(&device) == NVN_DEPTH_MODE_NEAR_IS_ZERO);

    // Handle packing keeps the descriptor ID layout of the API.
    auto texture_handle_of = entry<NVNtextureHandle (*)(const NVNdevice*, int, int)>("nvnDeviceGetTextureHandle");
    auto texel_handle_of = entry<NVNtextureHandle (*)(const NVNdevice*, int)>("nvnDeviceGetTexelFetchHandle");
    CHECK(texture_handle_of(&device, 0x12345, 0x678) == combined_handle(0x12345, 0x678));
    CHECK(texel_handle_of(&device, 7) == texture_handle(7));
    auto image_handle_of = entry<NVNimageHandle (*)(const NVNdevice*, int)>("nvnDeviceGetImageHandle");
    CHECK(image_handle_of(&device, 9) == 9);
    auto separate_texture = entry<NVNseparateTextureHandle (*)(const NVNdevice*, int)>("nvnDeviceGetSeparateTextureHandle");
    auto separate_sampler = entry<NVNseparateSamplerHandle (*)(const NVNdevice*, int)>("nvnDeviceGetSeparateSamplerHandle");
    CHECK(separate_texture(&device, 3).value == texture_handle(3));
    CHECK(separate_sampler(&device, 4).value == sampler_handle(4));

    // Debug callbacks install and clear; the enable flag is a low-bits boolean.
    auto install_callback =
        entry<void (*)(NVNdevice*, const PFNNVNDEBUGCALLBACKPROC, void*, std::uint32_t)>("nvnDeviceInstallDebugCallback");
    const auto callback = [](NVNdebugCallbackSource, NVNdebugCallbackType, int, NVNdebugCallbackSeverity,
                             const unsigned char*, void*) {};
    install_callback(&device, callback, &device, 1);
    CHECK(state->debug_callback != nullptr && state->debug_data == &device);
    install_callback(&device, callback, &device, 0);
    CHECK(state->debug_callback == nullptr);

    // Fast-clear registration is declined without failing the caller's frame.
    const auto register_color = entry<NVNboolean (*)(NVNdevice*, const float*, NVNformat)>("nvnDeviceRegisterFastClearColor");
    const auto register_depth = entry<NVNboolean (*)(NVNdevice*, float)>("nvnDeviceRegisterFastClearDepth");
    CHECK(register_color(&device, nullptr, NVN_FORMAT_RGBA8) == 0);
    CHECK(register_depth(&device, 0.5f) == 0);

    // The loader entry itself resolves through the table like any other.
    const auto bootstrap = entry<PFNNVNGENERICFUNCPTRPROC (*)(const char*)>("nvnBootstrapLoader");
    CHECK(bootstrap != nullptr && bootstrap("nvnDeviceGetInteger") == armigo_nvn_bootstrap_loader("nvnDeviceGetInteger"));
    auto get_proc_address = entry<PFNNVNGENERICFUNCPTRPROC (*)(const NVNdevice*, const char*)>("nvnDeviceGetProcAddress");
    CHECK(get_proc_address(&device, "nvnQueueSubmitCommands") ==
          armigo_nvn_bootstrap_loader("nvnQueueSubmitCommands"));

    // An entry the null device does not cover resolves to the reporting stub.
    const auto event_initialize = entry<NVNboolean (*)(NVNevent*, const NVNeventBuilder*)>("nvnEventInitialize");
    CHECK(event_initialize != nullptr && event_initialize(nullptr, nullptr) == 0);

    finalize(&device);
    CHECK(state->tag == 0);
}

void test_memory_pools() {
    TestPool pool;

    auto map = entry<void* (*)(const NVNmemoryPool*)>("nvnMemoryPoolMap");
    auto get_size = entry<std::size_t (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetSize");
    auto get_flags = entry<NVNmemoryPoolFlags (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetFlags");
    auto get_address = entry<NVNbufferAddress (*)(const NVNmemoryPool*)>("nvnMemoryPoolGetBufferAddress");
    auto flush = entry<void (*)(const NVNmemoryPool*, ptrdiff_t, std::size_t)>("nvnMemoryPoolFlushMappedRange");
    auto invalidate = entry<void (*)(const NVNmemoryPool*, ptrdiff_t, std::size_t)>("nvnMemoryPoolInvalidateMappedRange");
    auto finalize = entry<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize");

    CHECK(map(&pool.pool) == pool.storage);
    CHECK(get_size(&pool.pool) == sizeof pool.storage);
    CHECK(get_flags(&pool.pool) == (NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED));
    CHECK(get_address(&pool.pool) == reinterpret_cast<NVNbufferAddress>(pool.storage));

    // The latest CPU flush of a cached pool is kept, ordered by serial.
    MemoryPoolState* state = reinterpret_cast<MemoryPoolState*>(&pool.pool);
    flush(&pool.pool, 256, 512);
    CHECK(state->flush_begin == 256 && state->flush_end == 256 + 512 && state->flush_serial != 0);
    flush(&pool.pool, 1024, 128);
    CHECK(state->flush_begin == 1024 && state->flush_end == 1024 + 128);
    invalidate(&pool.pool, 0, sizeof pool.storage);  // accepted, no effect on the flush record
    CHECK(state->flush_end == 1024 + 128);

    // Re-initialization over a live pool resets the flush record.
    auto initialize = entry<NVNboolean (*)(NVNmemoryPool*, const NVNmemoryPoolBuilder*)>("nvnMemoryPoolInitialize");
    CHECK(initialize(&pool.pool, &pool.builder) == 1);
    CHECK(state->tag == object_tag('P', 'O', 'O', 'L') && state->flush_serial == 0 && state->flush_end == 0);
    finalize(&pool.pool);
    CHECK(state->tag == 0);

    // Builder getters answer what was stored.
    auto builder_get_size = entry<std::size_t (*)(const NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderGetSize");
    auto builder_get_flags = entry<NVNmemoryPoolFlags (*)(const NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderGetFlags");
    CHECK(builder_get_size(&pool.builder) == sizeof pool.storage);
    CHECK(builder_get_flags(&pool.builder) == (NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED));

    // A virtual pool maps to nothing and reports no buffer address.
    NVNmemoryPoolBuilder builder{};
    NVNmemoryPool virtual_pool{};
    auto set_storage = entry<void (*)(NVNmemoryPoolBuilder*, void*, std::size_t)>("nvnMemoryPoolBuilderSetStorage");
    auto set_flags = entry<void (*)(NVNmemoryPoolBuilder*, int)>("nvnMemoryPoolBuilderSetFlags");
    set_storage(&builder, nullptr, 0x10000);
    set_flags(&builder, NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    CHECK(initialize(&virtual_pool, &builder) == 1);
    CHECK(map(&virtual_pool) == nullptr);
    CHECK(get_address(&virtual_pool) == 0);
}

void test_descriptor_pools() {
    TestPool pool;
    NVNtexturePool textures{};
    NVNsamplerPool samplers{};
    NVNtexture texture{};
    NVNsampler sampler{};

    auto texture_pool_init =
        entry<NVNboolean (*)(NVNtexturePool*, const NVNmemoryPool*, ptrdiff_t, int)>("nvnTexturePoolInitialize");
    auto texture_pool_get_pool = entry<const NVNmemoryPool* (*)(const NVNtexturePool*)>("nvnTexturePoolGetMemoryPool");
    auto texture_pool_get_offset = entry<ptrdiff_t (*)(const NVNtexturePool*)>("nvnTexturePoolGetMemoryOffset");
    auto texture_pool_get_size = entry<int (*)(const NVNtexturePool*)>("nvnTexturePoolGetSize");
    auto register_texture = entry<void (*)(const NVNtexturePool*, int, const NVNtexture*, const NVNtextureView*)>(
        "nvnTexturePoolRegisterTexture");
    auto texture_pool_finalize = entry<void (*)(NVNtexturePool*)>("nvnTexturePoolFinalize");

    CHECK(texture_pool_init(&textures, &pool.pool, 64, 128) == 1);
    CHECK(texture_pool_get_pool(&textures) == &pool.pool);
    CHECK(texture_pool_get_offset(&textures) == 64);
    CHECK(texture_pool_get_size(&textures) == 128);
    register_texture(&textures, 3, &texture, nullptr);  // registration is accepted
    texture_pool_finalize(&textures);
    CHECK(texture_pool_get_size(&textures) == 0);

    auto sampler_pool_init =
        entry<NVNboolean (*)(NVNsamplerPool*, const NVNmemoryPool*, ptrdiff_t, int)>("nvnSamplerPoolInitialize");
    auto register_sampler = entry<void (*)(const NVNsamplerPool*, int, const NVNsampler*)>("nvnSamplerPoolRegisterSampler");
    CHECK(sampler_pool_init(&samplers, &pool.pool, 0, 96) == 1);
    register_sampler(&samplers, 5, &sampler);
    auto get_size = entry<int (*)(const NVNsamplerPool*)>("nvnSamplerPoolGetSize");
    CHECK(get_size(&samplers) == 96);
}

void test_buffers() {
    TestPool pool;
    NVNbufferBuilder builder{};
    NVNbuffer buffer{};

    auto set_device = entry<void (*)(NVNbufferBuilder*, NVNdevice*)>("nvnBufferBuilderSetDevice");
    auto set_defaults = entry<void (*)(NVNbufferBuilder*)>("nvnBufferBuilderSetDefaults");
    auto set_storage =
        entry<void (*)(NVNbufferBuilder*, NVNmemoryPool*, ptrdiff_t, std::size_t)>("nvnBufferBuilderSetStorage");
    auto builder_get_device = entry<NVNdevice* (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetDevice");
    auto builder_get_pool = entry<NVNmemoryPool* (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetMemoryPool");
    auto builder_get_offset = entry<ptrdiff_t (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetMemoryOffset");
    auto builder_get_size = entry<std::size_t (*)(const NVNbufferBuilder*)>("nvnBufferBuilderGetSize");
    auto initialize = entry<NVNboolean (*)(NVNbuffer*, const NVNbufferBuilder*)>("nvnBufferInitialize");
    auto finalize = entry<void (*)(NVNbuffer*)>("nvnBufferFinalize");
    auto map = entry<void* (*)(const NVNbuffer*)>("nvnBufferMap");
    auto get_address = entry<NVNbufferAddress (*)(const NVNbuffer*)>("nvnBufferGetAddress");
    auto get_offset = entry<ptrdiff_t (*)(const NVNbuffer*)>("nvnBufferGetMemoryOffset");
    auto get_size = entry<std::size_t (*)(const NVNbuffer*)>("nvnBufferGetSize");
    auto get_debug_id = entry<std::uint64_t (*)(const NVNbuffer*)>("nvnBufferGetDebugID");

    set_defaults(&builder);
    CHECK(builder_get_pool(&builder) == nullptr && builder_get_offset(&builder) == 0 && builder_get_size(&builder) == 0);
    set_device(&builder, nullptr);
    set_storage(&builder, &pool.pool, 128, 1024);
    CHECK(builder_get_device(&builder) == nullptr);
    CHECK(builder_get_pool(&builder) == &pool.pool && builder_get_offset(&builder) == 128 &&
          builder_get_size(&builder) == 1024);
    CHECK(initialize(&buffer, &builder) == 1);
    CHECK(map(&buffer) == pool.storage + 128);
    CHECK(get_address(&buffer) == reinterpret_cast<NVNbufferAddress>(pool.storage + 128));
    CHECK(get_offset(&buffer) == 128 && get_size(&buffer) == 1024);
    CHECK(get_debug_id(&buffer) != 0);

    // Cache range calls are accepted without effect.
    auto flush = entry<void (*)(const NVNbuffer*, ptrdiff_t, std::size_t)>("nvnBufferFlushMappedRange");
    auto invalidate = entry<void (*)(const NVNbuffer*, ptrdiff_t, std::size_t)>("nvnBufferInvalidateMappedRange");
    flush(&buffer, 0, 1024);
    invalidate(&buffer, 0, 1024);
    CHECK(map(&buffer) == pool.storage + 128);

    // A builder without an initialized pool fails initialization.
    NVNbufferBuilder empty_builder{};
    NVNbuffer empty_buffer{};
    set_defaults(&empty_builder);
    CHECK(initialize(&empty_buffer, &empty_builder) == 0);

    // Invalid extents must not expose writable addresses outside the pool.
    NVNbufferBuilder outside_builder{};
    NVNbuffer outside_buffer{};
    set_defaults(&outside_builder);
    set_storage(&outside_builder, &pool.pool, 0, 1 << 20);
    CHECK(initialize(&outside_buffer, &outside_builder) == 0);
    CHECK(map(&outside_buffer) == nullptr);
    set_storage(&outside_builder, &pool.pool, -1, 16);
    CHECK(initialize(&outside_buffer, &outside_builder) == 0);
    set_storage(&outside_builder, &pool.pool, sizeof pool.storage - 16, 32);
    CHECK(initialize(&outside_buffer, &outside_builder) == 0);

    finalize(&buffer);
    CHECK(map(&buffer) == nullptr && get_address(&buffer) == 0);
}

void test_texture_storage_sizes() {
    NVNtextureBuilder builder{};
    auto defaults = entry<void (*)(NVNtextureBuilder*)>("nvnTextureBuilderSetDefaults");
    auto set_target = entry<void (*)(NVNtextureBuilder*, NVNtextureTarget)>("nvnTextureBuilderSetTarget");
    auto set_size_2d = entry<void (*)(NVNtextureBuilder*, int, int)>("nvnTextureBuilderSetSize2D");
    auto set_levels = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetLevels");
    auto set_format = entry<void (*)(NVNtextureBuilder*, NVNformat)>("nvnTextureBuilderSetFormat");
    auto set_flags = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetFlags");
    auto set_stride = entry<void (*)(NVNtextureBuilder*, ptrdiff_t)>("nvnTextureBuilderSetStride");
    auto set_width = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetWidth");
    auto set_depth = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetDepth");
    auto set_packaged =
        entry<void (*)(NVNtextureBuilder*, const NVNpackagedTextureLayout*)>("nvnTextureBuilderSetPackagedTextureLayout");
    auto storage_size = entry<std::size_t (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetStorageSize");
    auto storage_alignment = entry<std::size_t (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetStorageAlignment");

    // Builder defaults: 2D, one by one by one, a single level, no format.
    defaults(&builder);
    auto get_target = entry<NVNtextureTarget (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetTarget");
    auto get_width = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetWidth");
    auto get_height = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetHeight");
    auto get_depth = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetDepth");
    auto get_levels = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetLevels");
    auto get_format = entry<NVNformat (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetFormat");
    CHECK(get_target(&builder) == NVN_TEXTURE_TARGET_2D);
    CHECK(get_width(&builder) == 1 && get_height(&builder) == 1 && get_depth(&builder) == 1);
    CHECK(get_levels(&builder) == 1 && get_format(&builder) == NVN_FORMAT_NONE);

    // Pitch layouts: linear stride aligned to 32 (128 for linear render targets).
    defaults(&builder);
    set_size_2d(&builder, 4, 4);
    set_format(&builder, NVN_FORMAT_RGBA8);
    set_flags(&builder, NVN_TEXTURE_FLAGS_LINEAR);
    CHECK(storage_size(&builder) == 128 && storage_alignment(&builder) == 32);
    set_stride(&builder, 48);
    CHECK(storage_size(&builder) == 192);
    defaults(&builder);
    set_size_2d(&builder, 2, 2);
    set_format(&builder, NVN_FORMAT_RGBA8);
    set_flags(&builder, NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET);
    CHECK(storage_size(&builder) == 256 && storage_alignment(&builder) == 128);

    // Buffer textures are addressed linearly; 12-byte formats align to 4.
    defaults(&builder);
    set_target(&builder, NVN_TEXTURE_TARGET_BUFFER);
    set_width(&builder, 100);
    set_format(&builder, NVN_FORMAT_R32F);
    CHECK(storage_size(&builder) == 400 && storage_alignment(&builder) == 4);
    set_format(&builder, NVN_FORMAT_RGB32F);
    set_width(&builder, 5);
    CHECK(storage_size(&builder) == 60 && storage_alignment(&builder) == 4);

    // Block-linear layouts round every level into GOBs of 512 bytes.
    defaults(&builder);
    set_size_2d(&builder, 3, 2);
    set_format(&builder, NVN_FORMAT_R8);
    CHECK(storage_size(&builder) == 512 && storage_alignment(&builder) == 512);
    defaults(&builder);
    set_size_2d(&builder, 128, 128);
    set_format(&builder, NVN_FORMAT_RGBA8);
    CHECK(storage_size(&builder) == 65536);
    defaults(&builder);
    set_size_2d(&builder, 4, 4);
    set_levels(&builder, 2);
    set_format(&builder, NVN_FORMAT_R8);
    CHECK(storage_size(&builder) == 1024);
    defaults(&builder);
    set_size_2d(&builder, 8, 8);
    set_format(&builder, NVN_FORMAT_RGB_DXT1);
    CHECK(storage_size(&builder) == 512);

    // A packaged layout fixes the block height: 8x40 R8 packs into 2560 bytes
    // with one-GOB blocks where the automatic choice spends 4096.
    defaults(&builder);
    set_size_2d(&builder, 8, 40);
    set_format(&builder, NVN_FORMAT_R8);
    CHECK(storage_size(&builder) == 4096);
    NVNpackagedTextureLayout layout{};
    layout.layout[0] = 0;
    set_packaged(&builder, &layout);
    CHECK(storage_size(&builder) == 2560);
    set_packaged(&builder, nullptr);
    CHECK(storage_size(&builder) == 4096);

    // Layered targets repeat the aligned layer size.
    defaults(&builder);
    set_size_2d(&builder, 2, 2);
    set_depth(&builder, 3);
    set_target(&builder, NVN_TEXTURE_TARGET_2D_ARRAY);
    set_format(&builder, NVN_FORMAT_RGBA8);
    CHECK(storage_size(&builder) == 1536);
    defaults(&builder);
    set_size_2d(&builder, 4, 4);
    set_target(&builder, NVN_TEXTURE_TARGET_CUBEMAP);
    set_format(&builder, NVN_FORMAT_R8);
    CHECK(storage_size(&builder) == 3072);

    // Invalid mip counts and packaged block heights fail before layout loops.
    set_levels(&builder, -1);
    CHECK(storage_size(&builder) == 0);
    set_levels(&builder, 33);
    CHECK(storage_size(&builder) == 0);
    set_levels(&builder, 1);
    NVNpackagedTextureLayout invalid_layout{};
    invalid_layout.layout[0] = 32;
    set_packaged(&builder, &invalid_layout);
    CHECK(storage_size(&builder) == 0);

    // Unknown formats reserve nothing.
    defaults(&builder);
    set_size_2d(&builder, 16, 16);
    set_format(&builder, NVN_FORMAT_NONE);
    CHECK(storage_size(&builder) == 0);
}

void test_textures() {
    TestPool pool;
    NVNtextureBuilder builder{};
    NVNtexture texture{}, twin{};

    auto defaults = entry<void (*)(NVNtextureBuilder*)>("nvnTextureBuilderSetDefaults");
    auto set_device = entry<void (*)(NVNtextureBuilder*, NVNdevice*)>("nvnTextureBuilderSetDevice");
    auto set_size_2d = entry<void (*)(NVNtextureBuilder*, int, int)>("nvnTextureBuilderSetSize2D");
    auto set_format = entry<void (*)(NVNtextureBuilder*, NVNformat)>("nvnTextureBuilderSetFormat");
    auto set_levels = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetLevels");
    auto set_swizzle = entry<void (*)(NVNtextureBuilder*, NVNtextureSwizzle, NVNtextureSwizzle, NVNtextureSwizzle,
                                      NVNtextureSwizzle)>("nvnTextureBuilderSetSwizzle");
    auto set_storage = entry<void (*)(NVNtextureBuilder*, NVNmemoryPool*, ptrdiff_t)>("nvnTextureBuilderSetStorage");
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>("nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");
    auto get_address = entry<NVNtextureAddress (*)(const NVNtexture*)>("nvnTextureGetTextureAddress");
    auto get_size = entry<int (*)(const NVNtexture*)>("nvnTextureGetStorageSize");
    auto get_width = entry<int (*)(const NVNtexture*)>("nvnTextureGetWidth");
    auto get_format = entry<NVNformat (*)(const NVNtexture*)>("nvnTextureGetFormat");
    auto get_swizzle = entry<void (*)(const NVNtexture*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*,
                                      NVNtextureSwizzle*)>("nvnTextureGetSwizzle");
    auto get_offset = entry<ptrdiff_t (*)(const NVNtexture*)>("nvnTextureGetMemoryOffset");
    auto get_debug_id = entry<std::uint64_t (*)(const NVNtexture*)>("nvnTextureGetDebugID");
    auto compare = entry<NVNboolean (*)(const NVNtexture*, const NVNtexture*)>("nvnTextureCompare");

    defaults(&builder);
    set_device(&builder, nullptr);
    set_size_2d(&builder, 64, 32);
    set_format(&builder, NVN_FORMAT_RGBA8);
    set_swizzle(&builder, NVN_TEXTURE_SWIZZLE_A, NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B);
    set_storage(&builder, &pool.pool, 256);
    CHECK(initialize(&texture, &builder) == 1);

    const TextureState* state = reinterpret_cast<const TextureState*>(&texture);
    CHECK(state->tag == object_tag('T', 'E', 'X', 0));
    CHECK(state->serial != 0 && state->size == 8192);
    CHECK(get_address(&texture) == reinterpret_cast<NVNtextureAddress>(pool.storage + 256));
    CHECK(get_size(&texture) == 8192 && get_width(&texture) == 64 && get_format(&texture) == NVN_FORMAT_RGBA8);
    CHECK(get_offset(&texture) == 256);
    CHECK(get_debug_id(&texture) == state->serial);
    NVNtextureSwizzle r, g, b, a;
    get_swizzle(&texture, &r, &g, &b, &a);
    CHECK(r == NVN_TEXTURE_SWIZZLE_A && g == NVN_TEXTURE_SWIZZLE_R && b == NVN_TEXTURE_SWIZZLE_G &&
          a == NVN_TEXTURE_SWIZZLE_B);
    // An invalid swizzle is refused, leaving the previous mapping.
    set_swizzle(&builder, NVN_TEXTURE_SWIZZLE_ZERO, NVN_TEXTURE_SWIZZLE_ONE, NVN_TEXTURE_SWIZZLE_R,
                static_cast<NVNtextureSwizzle>(99));
    auto builder_get_swizzle = entry<void (*)(const NVNtextureBuilder*, NVNtextureSwizzle*, NVNtextureSwizzle*,
                                              NVNtextureSwizzle*, NVNtextureSwizzle*)>("nvnTextureBuilderGetSwizzle");
    builder_get_swizzle(&builder, &r, &g, &b, &a);
    CHECK(a == NVN_TEXTURE_SWIZZLE_B);

    // Re-initialization over a live object issues a fresh serial.
    const std::uint64_t first_serial = state->serial;
    CHECK(initialize(&texture, &builder) == 1);
    CHECK(state->serial != first_serial && get_debug_id(&texture) == state->serial);

    // Identical construction compares equal; a different width does not.
    CHECK(initialize(&twin, &builder) == 1);
    CHECK(compare(&texture, &twin) == 1);
    set_size_2d(&builder, 32, 32);
    NVNtexture other{};
    CHECK(initialize(&other, &builder) == 1);
    CHECK(compare(&texture, &other) == 0);
    finalize(&other);
    finalize(&twin);
    set_storage(&builder, &pool.pool, -1);
    CHECK(initialize(&other, &builder) == 0);
    set_storage(&builder, &pool.pool, sizeof pool.storage - 16);
    CHECK(initialize(&other, &builder) == 0);

    // View offsets: level offsets follow the level layout; layers step by the
    // aligned layer size.
    NVNtexture levels_texture{};
    defaults(&builder);
    set_size_2d(&builder, 4, 4);
    set_levels(&builder, 2);
    set_format(&builder, NVN_FORMAT_R8);
    set_storage(&builder, &pool.pool, 0);
    CHECK(initialize(&levels_texture, &builder) == 1);
    auto get_view_offset = entry<ptrdiff_t (*)(const NVNtexture*, const NVNtextureView*)>("nvnTextureGetViewOffset");
    NVNtextureView view{};
    auto view_defaults = entry<void (*)(NVNtextureView*)>("nvnTextureViewSetDefaults");
    auto view_set_levels = entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLevels");
    auto view_set_layers = entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers");
    auto view_get_levels = entry<NVNboolean (*)(const NVNtextureView*, int*, int*)>("nvnTextureViewGetLevels");
    auto view_get_layers = entry<NVNboolean (*)(const NVNtextureView*, int*, int*)>("nvnTextureViewGetLayers");
    view_defaults(&view);
    int base = -1, count = -1;
    CHECK(view_get_levels(&view, &base, &count) == 0);  // nothing set yet
    CHECK(get_view_offset(&levels_texture, &view) == 0);
    view_set_levels(&view, 1, 1);
    CHECK(view_get_levels(&view, &base, &count) == 1 && base == 1 && count == 1);
    CHECK(get_view_offset(&levels_texture, &view) == 512);
    view_defaults(&view);
    view_set_layers(&view, 1, 1);
    CHECK(view_get_layers(&view, &base, &count) == 1 && base == 1 && count == 1);
    // A layer view of a texture without layers steps by its whole storage.
    CHECK(get_view_offset(&levels_texture, &view) == 1024);

    NVNtexture array_texture{};
    defaults(&builder);
    auto set_depth = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetDepth");
    auto set_target = entry<void (*)(NVNtextureBuilder*, NVNtextureTarget)>("nvnTextureBuilderSetTarget");
    set_size_2d(&builder, 2, 2);
    set_depth(&builder, 3);
    set_target(&builder, NVN_TEXTURE_TARGET_2D_ARRAY);
    set_format(&builder, NVN_FORMAT_RGBA8);
    set_storage(&builder, &pool.pool, 0);
    CHECK(initialize(&array_texture, &builder) == 1);
    view_defaults(&view);
    view_set_layers(&view, 2, 1);
    CHECK(get_view_offset(&array_texture, &view) == 1024);  // two aligned 512-byte layers
    CHECK(get_view_offset(&array_texture, nullptr) == 0);
    finalize(&array_texture);
    finalize(&levels_texture);

    finalize(&texture);
    CHECK(state->tag == 0 && get_debug_id(&texture) == 0);
}

void test_samplers() {
    NVNsamplerBuilder builder{};
    NVNsampler sampler{}, twin{};

    auto defaults = entry<void (*)(NVNsamplerBuilder*)>("nvnSamplerBuilderSetDefaults");
    auto set_lod_clamp = entry<void (*)(NVNsamplerBuilder*, float, float)>("nvnSamplerBuilderSetLodClamp");
    auto initialize = entry<NVNboolean (*)(NVNsampler*, const NVNsamplerBuilder*)>("nvnSamplerInitialize");
    auto finalize = entry<void (*)(NVNsampler*)>("nvnSamplerFinalize");
    auto get_min_mag = entry<void (*)(const NVNsampler*, NVNminFilter*, NVNmagFilter*)>("nvnSamplerGetMinMagFilter");
    auto get_wrap = entry<void (*)(const NVNsampler*, NVNwrapMode*, NVNwrapMode*, NVNwrapMode*)>("nvnSamplerGetWrapMode");
    auto get_lod_clamp = entry<void (*)(const NVNsampler*, float*, float*)>("nvnSamplerGetLodClamp");
    auto get_max_anisotropy = entry<float (*)(const NVNsampler*)>("nvnSamplerGetMaxAnisotropy");
    auto compare = entry<NVNboolean (*)(const NVNsampler*, const NVNsampler*)>("nvnSamplerCompare");

    defaults(&builder);
    CHECK(initialize(&sampler, &builder) == 1);
    NVNminFilter min;
    NVNmagFilter mag;
    NVNwrapMode s, t, r;
    float low, high;
    get_min_mag(&sampler, &min, &mag);
    get_wrap(&sampler, &s, &t, &r);
    get_lod_clamp(&sampler, &low, &high);
    CHECK(min == NVN_MIN_FILTER_LINEAR && mag == NVN_MAG_FILTER_LINEAR);
    CHECK(s == NVN_WRAP_MODE_REPEAT && t == NVN_WRAP_MODE_REPEAT && r == NVN_WRAP_MODE_REPEAT);
    CHECK(low == 0.0f && high == 1000.0f && get_max_anisotropy(&sampler) == 1.0f);

    CHECK(initialize(&twin, &builder) == 1);
    CHECK(compare(&sampler, &twin) == 1);
    set_lod_clamp(&builder, 2.0f, 8.0f);
    NVNsampler clamped{};
    CHECK(initialize(&clamped, &builder) == 1);
    get_lod_clamp(&clamped, &low, &high);
    CHECK(low == 2.0f && high == 8.0f);
    CHECK(compare(&sampler, &clamped) == 0);
    finalize(&clamped);
    finalize(&twin);
    finalize(&sampler);
    CHECK(reinterpret_cast<const SamplerState*>(&sampler)->tag == 0);
}

void test_programs() {
    NVNprogram program{};
    auto initialize = entry<NVNboolean (*)(NVNprogram*, NVNdevice*)>("nvnProgramInitialize");
    auto finalize = entry<void (*)(NVNprogram*)>("nvnProgramFinalize");
    auto set_shaders = entry<NVNboolean (*)(NVNprogram*, int, const NVNshaderData*)>("nvnProgramSetShaders");
    auto set_shaders_ext = entry<NVNboolean (*)(NVNprogram*, int, const NVNshaderDataExt*)>("nvnProgramSetShadersExt");

    CHECK(initialize(&program, nullptr) == 1);
    const ProgramState* state = reinterpret_cast<const ProgramState*>(&program);
    CHECK(state->tag == object_tag('P', 'R', 'O', 'G') && state->stage_count == 0);

    NVNshaderData stages[2]{{0x1000, nullptr}, {0x2000, nullptr}};
    CHECK(set_shaders(&program, 2, stages) == 1);
    CHECK(state->stage_count == 2 && state->stages[0].data == 0x1000 && state->stages[1].data == 0x2000);
    const std::uint64_t serial = state->serial;
    CHECK(serial != 0);

    NVNshaderDataExt ext{0x3000, nullptr, nullptr};
    CHECK(set_shaders_ext(&program, 1, &ext) == 1);
    CHECK(state->stage_count == 1 && state->stages[0].data == 0x3000 && state->serial != serial);

    CHECK(set_shaders(&program, 7, stages) == 0);  // at most six stages
    CHECK(set_shaders(&program, 1, nullptr) == 0);
    CHECK(state->stage_count == 1 && state->serial != 0);  // unchanged by the refusal

    finalize(&program);
    CHECK(state->tag == 0 && state->serial == 0);
}

void test_queues_and_syncs() {
    NVNqueueBuilder builder{};
    NVNqueue queue{};

    auto defaults = entry<void (*)(NVNqueueBuilder*)>("nvnQueueBuilderSetDefaults");
    auto set_command_memory = entry<void (*)(NVNqueueBuilder*, std::size_t)>("nvnQueueBuilderSetCommandMemorySize");
    auto get_memory_size = entry<std::size_t (*)(const NVNqueueBuilder*)>("nvnQueueBuilderGetQueueMemorySize");
    auto set_queue_memory = entry<void (*)(NVNqueueBuilder*, void*, std::size_t)>("nvnQueueBuilderSetQueueMemory");
    auto initialize = entry<NVNboolean (*)(NVNqueue*, const NVNqueueBuilder*)>("nvnQueueInitialize");
    auto finalize = entry<void (*)(NVNqueue*)>("nvnQueueFinalize");
    auto get_error = entry<NVNqueueGetErrorResult (*)(NVNqueue*, NVNqueueErrorInfo*)>("nvnQueueGetError");
    auto memory_used = entry<std::size_t (*)(NVNqueue*)>("nvnQueueGetTotalCommandMemoryUsed");
    auto submit = entry<void (*)(NVNqueue*, int, const NVNcommandHandle*)>("nvnQueueSubmitCommands");
    auto flush = entry<void (*)(NVNqueue*)>("nvnQueueFlush");
    auto finish = entry<void (*)(NVNqueue*)>("nvnQueueFinish");
    auto fence_sync = entry<void (*)(NVNqueue*, NVNsync*, NVNsyncCondition, int)>("nvnQueueFenceSync");
    auto wait_sync = entry<NVNboolean (*)(NVNqueue*, const NVNsync*)>("nvnQueueWaitSync");

    // Builder defaults mirror the device queue memory properties.
    defaults(&builder);
    CHECK(get_memory_size(&builder) == 344064);  // 64 KiB + 256 KiB + 16 KiB, page aligned
    set_command_memory(&builder, 1);
    CHECK(get_memory_size(&builder) == 282624);  // rounded up to whole pages
    std::byte queue_memory[512]{};
    set_queue_memory(&builder, queue_memory, sizeof queue_memory);
    CHECK(initialize(&queue, &builder) == 1);

    NVNqueueErrorInfo info{};
    CHECK(get_error(&queue, &info) == NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR);
    CHECK(memory_used(&queue) == 0);

    // Syncs start unfenced: waiting reports a timeout and the queue does not
    // wait; after a submission every fence is complete immediately.
    NVNsync sync{};
    auto sync_initialize = entry<NVNboolean (*)(NVNsync*, NVNdevice*)>("nvnSyncInitialize");
    auto sync_finalize = entry<void (*)(NVNsync*)>("nvnSyncFinalize");
    auto sync_wait = entry<NVNsyncWaitResult (*)(const NVNsync*, std::uint64_t)>("nvnSyncWait");
    CHECK(sync_initialize(&sync, nullptr) == 1);
    CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);
    CHECK(wait_sync(&queue, &sync) == 0);

    QueueState* queue_state = reinterpret_cast<QueueState*>(&queue);
    const NVNcommandHandle handle = 0;
    submit(&queue, 1, &handle);
    flush(&queue);
    finish(&queue);
    CHECK(queue_state->submits == 1);
    fence_sync(&queue, &sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED);
    CHECK(wait_sync(&queue, &sync) == 1);
    submit(&queue, 0, nullptr);
    CHECK(queue_state->submits == 2);

    sync_finalize(&sync);
    CHECK(reinterpret_cast<const SyncState*>(&sync)->tag == 0);
    finalize(&queue);
    CHECK(queue_state->tag == 0);
}

void test_command_buffers() {
    NVNcommandBuffer commands{};
    auto initialize = entry<NVNboolean (*)(NVNcommandBuffer*, NVNdevice*)>("nvnCommandBufferInitialize");
    auto finalize = entry<void (*)(NVNcommandBuffer*)>("nvnCommandBufferFinalize");
    auto is_recording = entry<NVNboolean (*)(const NVNcommandBuffer*)>("nvnCommandBufferIsRecording");
    auto begin = entry<void (*)(NVNcommandBuffer*)>("nvnCommandBufferBeginRecording");
    auto end = entry<NVNcommandHandle (*)(NVNcommandBuffer*)>("nvnCommandBufferEndRecording");
    auto set_callback =
        entry<void (*)(NVNcommandBuffer*, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC)>("nvnCommandBufferSetMemoryCallback");
    auto get_callback =
        entry<PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC (*)(const NVNcommandBuffer*)>("nvnCommandBufferGetMemoryCallback");
    auto set_callback_data = entry<void (*)(NVNcommandBuffer*, void*)>("nvnCommandBufferSetMemoryCallbackData");
    auto add_command_memory =
        entry<void (*)(NVNcommandBuffer*, const NVNmemoryPool*, ptrdiff_t, std::size_t)>("nvnCommandBufferAddCommandMemory");
    auto add_control_memory = entry<void (*)(NVNcommandBuffer*, void*, std::size_t)>("nvnCommandBufferAddControlMemory");
    auto memory_size = entry<std::size_t (*)(const NVNcommandBuffer*)>("nvnCommandBufferGetCommandMemorySize");
    auto draw_arrays =
        entry<void (*)(NVNcommandBuffer*, NVNdrawPrimitive, int, int)>("nvnCommandBufferDrawArrays");

    CHECK(initialize(&commands, nullptr) == 1);
    CHECK(is_recording(&commands) == 0);
    begin(&commands);
    CHECK(is_recording(&commands) == 1);

    // Recording commands are accepted and dropped; nothing is consumed.
    static bool callback_ran = false;
    const auto callback = [](NVNcommandBuffer*, NVNcommandBufferMemoryEvent, std::size_t, void*) { callback_ran = true; };
    set_callback(&commands, callback);
    set_callback_data(&commands, &commands);
    CHECK(get_callback(&commands) == callback);
    add_command_memory(&commands, nullptr, 0, 4096);
    add_control_memory(&commands, nullptr, 256);
    draw_arrays(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    CHECK(memory_size(&commands) == 4096);
    CHECK(entry<std::size_t (*)(const NVNcommandBuffer*)>("nvnCommandBufferGetCommandMemoryFree")(&commands) == 4096);
    CHECK(entry<std::size_t (*)(const NVNcommandBuffer*)>("nvnCommandBufferGetControlMemoryFree")(&commands) == 256);
    CHECK(is_recording(&commands) == 1);

    // Empty recordings still have distinct opaque identities.
    const NVNcommandHandle handle = end(&commands);
    CHECK(handle != 0 && is_recording(&commands) == 0);
    begin(&commands);
    CHECK(end(&commands) != handle);
    CHECK(!callback_ran);  // the callback never runs: memory is never exhausted

    // A fence recorded into the stream is complete immediately.
    NVNsync sync{};
    entry<NVNboolean (*)(NVNsync*, NVNdevice*)>("nvnSyncInitialize")(&sync, nullptr);
    auto record_fence =
        entry<void (*)(NVNcommandBuffer*, NVNsync*, NVNsyncCondition, int)>("nvnCommandBufferFenceSync");
    record_fence(&commands, &sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    auto sync_wait = entry<NVNsyncWaitResult (*)(const NVNsync*, std::uint64_t)>("nvnSyncWait");
    CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED);

    finalize(&commands);
    CHECK(reinterpret_cast<const CommandBufferState*>(&commands)->tag == 0);
}

void test_windows() {
    TestPool pool;
    NVNtexture textures[6]{};
    NVNtextureBuilder builder{};
    auto texture_defaults = entry<void (*)(NVNtextureBuilder*)>("nvnTextureBuilderSetDefaults");
    auto texture_set_storage = entry<void (*)(NVNtextureBuilder*, NVNmemoryPool*, ptrdiff_t)>("nvnTextureBuilderSetStorage");
    auto texture_initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>("nvnTextureInitialize");
    texture_defaults(&builder);
    texture_set_storage(&builder, &pool.pool, 0);
    entry<void (*)(NVNtextureBuilder*, NVNformat)>("nvnTextureBuilderSetFormat")(&builder, NVN_FORMAT_RGBA8);
    for (auto& texture : textures) CHECK(texture_initialize(&texture, &builder) == 1);

    NVNwindowBuilder builder_{};
    NVNwindow window{};
    auto defaults = entry<void (*)(NVNwindowBuilder*)>("nvnWindowBuilderSetDefaults");
    auto set_native = entry<void (*)(NVNwindowBuilder*, NVNnativeWindow)>("nvnWindowBuilderSetNativeWindow");
    auto set_textures = entry<void (*)(NVNwindowBuilder*, int, NVNtexture* const*)>("nvnWindowBuilderSetTextures");
    auto set_interval = entry<void (*)(NVNwindowBuilder*, int)>("nvnWindowBuilderSetPresentInterval");
    auto builder_get_interval = entry<int (*)(const NVNwindowBuilder*)>("nvnWindowBuilderGetPresentInterval");
    auto builder_get_native = entry<NVNnativeWindow (*)(const NVNwindowBuilder*)>("nvnWindowBuilderGetNativeWindow");
    auto initialize = entry<NVNboolean (*)(NVNwindow*, const NVNwindowBuilder*)>("nvnWindowInitialize");
    auto finalize = entry<void (*)(NVNwindow*)>("nvnWindowFinalize");
    auto get_native = entry<NVNnativeWindow (*)(const NVNwindow*)>("nvnWindowGetNativeWindow");
    auto get_interval = entry<int (*)(const NVNwindow*)>("nvnWindowGetPresentInterval");
    auto set_window_interval = entry<void (*)(NVNwindow*, int)>("nvnWindowSetPresentInterval");
    auto set_crop = entry<void (*)(NVNwindow*, int, int, int, int)>("nvnWindowSetCrop");
    auto get_crop = entry<void (*)(const NVNwindow*, NVNrectangle*)>("nvnWindowGetCrop");
    auto acquire = entry<NVNwindowAcquireTextureResult (*)(NVNwindow*, NVNsync*, int*)>("nvnWindowAcquireTexture");
    auto queue_acquire = entry<NVNqueueAcquireTextureResult (*)(NVNqueue*, NVNwindow*, int*)>("nvnQueueAcquireTexture");
    auto present = entry<void (*)(NVNqueue*, NVNwindow*, int)>("nvnQueuePresentTexture");

    defaults(&builder_);
    CHECK(builder_get_interval(&builder_) == 1);
    NVNtexture* pointers[6]{};
    for (int i = 0; i < 6; ++i) pointers[i] = &textures[i];
    int native_marker = 0;
    set_native(&builder_, &native_marker);
    CHECK(builder_get_native(&builder_) == &native_marker);
    set_textures(&builder_, 6, pointers);  // clamped to four
    set_interval(&builder_, 2);

    CHECK(initialize(&window, &builder_) == 1);
    WindowState* state = reinterpret_cast<WindowState*>(&window);
    CHECK(state->tag == object_tag('W', 'I', 'N', 'D'));
    CHECK(state->texture_count == 4 && get_interval(&window) == 2);
    CHECK(get_native(&window) == &native_marker);

    // Acquisition cycles through the swapchain and fences the sync.
    NVNsync sync{};
    entry<NVNboolean (*)(NVNsync*, NVNdevice*)>("nvnSyncInitialize")(&sync, nullptr);
    auto sync_wait = entry<NVNsyncWaitResult (*)(const NVNsync*, std::uint64_t)>("nvnSyncWait");
    int index = -1;
    for (const int expected : {0, 1, 2, 3, 0}) {
        CHECK(acquire(&window, &sync, &index) == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
        CHECK(index == expected);
        CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED);
    }
    CHECK(queue_acquire(nullptr, &window, &index) == NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_SUCCESS);
    CHECK(index == 1);

    // Presents are counted per window and ticketed per image.
    const std::uint64_t tickets = state->present_ticket[0];
    present(nullptr, &window, 0);
    present(nullptr, &window, 3);
    present(nullptr, &window, 4);  // outside the swapchain: counted, not ticketed
    CHECK(state->presents == 3 && state->present_ticket[0] > tickets && state->present_ticket[3] > tickets);

    // Crop and present interval round-trip.
    set_crop(&window, 8, 16, 320, 240);
    NVNrectangle crop{};
    get_crop(&window, &crop);
    CHECK(crop.x == 8 && crop.y == 16 && crop.width == 320 && crop.height == 240);
    set_window_interval(&window, 3);
    CHECK(get_interval(&window) == 3);

    finalize(&window);
    CHECK(state->tag == 0);
}

} // namespace

int main() {
    // Nothing resolves until the API version is selected; the null device
    // registers only afterwards, exactly as driver setup would.
    CHECK(armigo_nvn_bootstrap_loader("nvnDeviceGetInteger") == nullptr);
    select_api_version({55, 15});
    register_null_entries();

    test_boot_and_device();
    test_memory_pools();
    test_descriptor_pools();
    test_buffers();
    test_texture_storage_sizes();
    test_textures();
    test_samplers();
    test_programs();
    test_queues_and_syncs();
    test_command_buffers();
    test_windows();

    std::printf("NVN null device failures: %u\n", failures);
    return failures ? 1 : 0;
}
