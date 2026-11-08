// ============================================================================
// Armigo Tests - NVN Texture Resources
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "entry_table.h"
#include "resource_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace armigo::nvn {
void register_resource_entries();
}

namespace {

using namespace armigo::nvn;
using namespace armigo::nvn::api;
namespace graphics = armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

template <typename Function>
Function entry(const char* name) {
    return reinterpret_cast<Function>(armigo_nvn_bootstrap_loader(name));
}

template <typename State, typename Object>
State& object_state(Object& object) {
    return *reinterpret_cast<State*>(&object);
}

struct MockResources final : ResourceBackend {
    unsigned registered = 0;
    unsigned released = 0;
    unsigned reads = 0;
    unsigned writes = 0;
    std::uint64_t flush_offset = 0;
    std::uint64_t flush_size = 0;
    std::uint64_t invalidate_offset = 0;
    std::uint64_t invalidate_size = 0;

    bool register_pool(const MemoryPoolState&) override { return true; }
    void release_pool(const MemoryPoolState&) override {}
    void flush_pool(const MemoryPoolState&, std::uint64_t offset, std::uint64_t size) override {
        flush_offset = offset;
        flush_size = size;
    }
    // What the storage held when the driver announced its write.
    unsigned prepares = 0;
    std::byte prepared_first{};
    void prepare_pool_write(const MemoryPoolState& pool, std::uint64_t offset, std::uint64_t) override {
        ++prepares;
        prepared_first = reinterpret_cast<const std::byte*>(pool.memory)[offset];
    }
    void invalidate_pool(const MemoryPoolState&, std::uint64_t offset, std::uint64_t size) override {
        invalidate_offset = offset;
        invalidate_size = size;
    }
    void register_texture(const TextureState&) override { ++registered; }
    void release_texture(const TextureState&) override { ++released; }
    void register_texture_descriptor(std::int32_t, const TextureState&,
                                     const TextureViewState*) override {}
    void register_sampler_descriptor(std::int32_t, const SamplerParameters&) override {}
    bool read_texels(const TextureState& texture, const TexelTransfer& transfer,
                     std::byte* out) override {
        ++reads;
        return transfer_pool_texels(texture, transfer, out, false);
    }
    bool write_texels(const TextureState& texture, const TexelTransfer& transfer,
                      const std::byte* in) override {
        ++writes;
        return transfer_pool_texels(texture, transfer, const_cast<std::byte*>(in), true);
    }
};

struct TextureFixture {
    alignas(TextureBuilderState) NVNtextureBuilder builder{};
    alignas(TextureState) NVNtexture texture{};
    alignas(MemoryPoolState) NVNmemoryPool pool{};
    std::vector<std::byte> memory;

    explicit TextureFixture(std::size_t bytes = 1u << 20) : memory(bytes) {
        auto& pool_state = object_state<MemoryPoolState>(pool);
        pool_state.tag = object_tag('P', 'O', 'O', 'L');
        pool_state.memory = reinterpret_cast<std::uint8_t*>(memory.data());
        pool_state.size = memory.size();
    }
    // Texture identities are per pool and process-wide: retire this pool's, as
    // releasing a real pool does, so a later fixture at the same addresses
    // starts new ones.
    ~TextureFixture() { retire_pool_textures(&object_state<MemoryPoolState>(pool)); }
    TextureFixture(const TextureFixture&) = delete;
    TextureFixture& operator=(const TextureFixture&) = delete;
};

void test_resource_layouts() {
    TextureDescription rgba{};
    rgba.target = NVN_TEXTURE_TARGET_2D;
    rgba.format = NVN_FORMAT_RGBA8;
    rgba.width = 64;
    rgba.height = 32;
    rgba.depth = 1;
    rgba.levels = 1;
    const auto rgba_layout = resource_layout(rgba);
    CHECK(rgba_layout.has_value());
    CHECK(rgba_layout && rgba_layout->total_size == 8192);
    CHECK(rgba_layout && rgba_layout->mips[0].gob_block_height_log2 == 2);
    CHECK(resource_alignment(rgba) == 512);
    CHECK(resource_format(NVN_FORMAT_RGBA8) == graphics::TextureFormat::R8G8B8A8Unorm);

    TextureDescription bc = rgba;
    bc.format = NVN_FORMAT_RGBA_DXT1;
    bc.width = 17;
    bc.height = 9;
    bc.levels = 3;
    const auto bc_layout = resource_layout(bc);
    CHECK(bc_layout.has_value());
    CHECK(bc_layout && bc_layout->total_size == 1536);
    CHECK(bc_layout && bc_layout->mips[0].block_columns == 5);
    CHECK(bc_layout && bc_layout->mips[0].block_rows == 3);
    CHECK(resource_format(NVN_FORMAT_RGBA_DXT1) == graphics::TextureFormat::Bc1RgbaUnorm);

    TextureDescription array = rgba;
    array.target = NVN_TEXTURE_TARGET_2D_ARRAY;
    array.width = 16;
    array.height = 16;
    array.depth = 3;
    array.levels = 2;
    const auto array_layout = resource_layout(array);
    CHECK(array_layout.has_value());
    CHECK(array_layout && array_layout->layer_stride == 2048);
    CHECK(array_layout && array_layout->total_size == 6144);

    array.depth = 1;
    const auto single_layer_array = resource_layout(array);
    CHECK(single_layer_array && single_layer_array->layer_stride == 2048);
    CHECK(single_layer_array && single_layer_array->total_size == 2048);

    {
    TextureDescription volume = rgba;
    volume.target = NVN_TEXTURE_TARGET_3D;
    volume.format = NVN_FORMAT_R8;
    volume.width = volume.height = 8;
    volume.depth = 11;
    volume.levels = 4;
    const auto volume_layout = resource_layout(volume);
    CHECK(volume_layout && volume_layout->total_size == 13824);
    if (volume_layout) {
        CHECK(volume_layout->mips[2].offset == 12288);
        CHECK(volume_layout->mips[2].size == 1024);
        CHECK(volume_layout->mips[3].offset == 13312);
        std::vector<std::byte> storage(volume_layout->total_size, std::byte{0x6d});
        TextureState texture{};
        texture.description = volume;
        texture.size = storage.size();
        texture.memory = reinterpret_cast<std::uint8_t*>(storage.data());
        for (std::uint32_t level = 0; level != 4; ++level) {
            const auto& mip = volume_layout->mips[level];
            TexelTransfer transfer{};
            transfer.level = level;
            transfer.region = {0, 0, 0, static_cast<int>(mip.width),
                               static_cast<int>(mip.height), static_cast<int>(mip.depth)};
            std::vector<std::byte> input(mip.linear_bytes, std::byte{static_cast<unsigned char>(level + 1)});
            std::vector<std::byte> output(input.size());
            CHECK(transfer_pool_texels(texture, transfer, input.data(), true));
            CHECK(transfer_pool_texels(texture, transfer, output.data(), false));
            CHECK(input == output);
        }
    }

    }

    TextureDescription linear = rgba;
    linear.flags = NVN_TEXTURE_FLAGS_LINEAR;
    linear.width = 13;
    linear.height = 7;
    const auto linear_layout = resource_layout(linear);
    CHECK(linear_layout.has_value());
    CHECK(linear_layout && linear_layout->pitch_bytes == 64);
    CHECK(linear_layout && linear_layout->total_size == 448);
    CHECK(resource_alignment(linear) == 32);

    TextureDescription volume{};
    volume.target = NVN_TEXTURE_TARGET_3D;
    volume.format = NVN_FORMAT_R8;
    volume.width = 1;
    volume.height = 1;
    volume.depth = 5;
    volume.levels = 1;
    const auto volume_layout = resource_layout(volume);
    CHECK(volume_layout.has_value());
    CHECK(volume_layout && volume_layout->total_size == 4096);
    CHECK(volume_layout && volume_layout->mips[0].size == 4096);

    TextureDescription packaged = rgba;
    packaged.height = 8;
    packaged.packaged_block_height = 5;
    const auto packaged_layout = resource_layout(packaged);
    CHECK(packaged_layout.has_value());
    CHECK(packaged_layout && packaged_layout->gob_block_height_log2 == 4);
    CHECK(packaged_layout && packaged_layout->mips[0].gob_block_height_log2 == 0);

    TextureDescription unknown = rgba;
    unknown.format = NVN_FORMAT_NONE;
    CHECK(!resource_layout(unknown).has_value());
    CHECK(resource_alignment(unknown) == 0);
    // BGRA8 keeps its memory byte order B, G, R, A; X variants map to the same format.
    CHECK(resource_format(NVN_FORMAT_BGRA8) == armigo::graphics::TextureFormat::B8G8R8A8Unorm);
    CHECK(resource_format(NVN_FORMAT_BGRX8) == armigo::graphics::TextureFormat::B8G8R8A8Unorm);
    CHECK(resource_format(NVN_FORMAT_BGRA8_SRGB) == armigo::graphics::TextureFormat::B8G8R8A8Srgb);
    CHECK(resource_format(NVN_FORMAT_BGRX8_SRGB) == armigo::graphics::TextureFormat::B8G8R8A8Srgb);
    CHECK(!resource_format(NVN_FORMAT_BGR565).has_value());
    CHECK(!resource_format(NVN_FORMAT_R8_UI2F).has_value());
    CHECK(!resource_format(NVN_FORMAT_RGBA16_I2F).has_value());
    TextureDescription multisample = rgba;
    multisample.target = NVN_TEXTURE_TARGET_2D_MULTISAMPLE;
    multisample.samples = 4;
    const auto multisample_layout = resource_layout(multisample);
    CHECK(multisample_layout && multisample_layout->total_size == 32768);
    if (multisample_layout) {
        std::vector<std::byte> storage(multisample_layout->total_size);
        TextureState texture{};
        texture.description = multisample;
        texture.memory = reinterpret_cast<std::uint8_t*>(storage.data());
        texture.size = storage.size();
        TexelTransfer transfer{};
        transfer.region = {0, 0, 0, 1, 1, 1};
        std::array<std::byte, 4> output{};
        CHECK(!transfer_pool_texels(texture, transfer, output.data(), false));
    }
}

void test_builder_and_view_entries() {
    TextureFixture fixture;
    auto set_device = entry<void (*)(NVNtextureBuilder*, NVNdevice*)>("nvnTextureBuilderSetDevice");
    auto set_defaults = entry<void (*)(NVNtextureBuilder*)>("nvnTextureBuilderSetDefaults");
    auto set_size = entry<void (*)(NVNtextureBuilder*, int, int)>("nvnTextureBuilderSetSize2D");
    auto set_format = entry<void (*)(NVNtextureBuilder*, NVNformat)>("nvnTextureBuilderSetFormat");
    auto set_levels = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetLevels");
    auto set_flags = entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetFlags");
    auto set_storage = entry<void (*)(NVNtextureBuilder*, NVNmemoryPool*, ptrdiff_t)>(
        "nvnTextureBuilderSetStorage");
    auto set_swizzle = entry<void (*)(NVNtextureBuilder*, NVNtextureSwizzle, NVNtextureSwizzle,
                                      NVNtextureSwizzle, NVNtextureSwizzle)>(
        "nvnTextureBuilderSetSwizzle");
    auto get_swizzle = entry<void (*)(const NVNtextureBuilder*, NVNtextureSwizzle*,
                                      NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*)>(
        "nvnTextureBuilderGetSwizzle");
    auto set_data = entry<void (*)(NVNtextureBuilder*, const void*)>(
        "nvnTextureBuilderSetPackagedTextureData");
    auto get_data = entry<const void* (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetPackagedTextureData");
    auto set_layout = entry<void (*)(NVNtextureBuilder*, const NVNpackagedTextureLayout*)>(
        "nvnTextureBuilderSetPackagedTextureLayout");
    auto get_layout = entry<NVNboolean (*)(const NVNtextureBuilder*, NVNpackagedTextureLayout*)>(
        "nvnTextureBuilderGetPackagedTextureLayout");
    auto set_gl_name = entry<void (*)(NVNtextureBuilder*, std::uint32_t)>(
        "nvnTextureBuilderSetGLTextureName");
    auto get_gl_name = entry<std::uint32_t (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetGLTextureName");
    auto get_size = entry<std::size_t (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetStorageSize");
    auto get_alignment = entry<std::size_t (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetStorageAlignment");
    auto get_width = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetWidth");
    auto get_height = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetHeight");
    auto get_levels = entry<int (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetLevels");
    auto get_flags = entry<NVNtextureFlags (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetFlags");
    auto get_pool = entry<NVNmemoryPool* (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetMemoryPool");
    auto get_offset = entry<ptrdiff_t (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetMemoryOffset");
    auto get_device = entry<NVNdevice* (*)(const NVNtextureBuilder*)>(
        "nvnTextureBuilderGetDevice");

    alignas(DeviceState) NVNdevice device{};
    set_device(&fixture.builder, &device);
    set_defaults(&fixture.builder);
    set_size(&fixture.builder, 64, 32);
    set_format(&fixture.builder, NVN_FORMAT_RGBA8);
    set_levels(&fixture.builder, 1);
    set_flags(&fixture.builder, 0);
    set_storage(&fixture.builder, &fixture.pool, 512);
    set_swizzle(&fixture.builder, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_G,
                NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_ONE);
    NVNtextureSwizzle r{}, g{}, b{}, a{};
    get_swizzle(&fixture.builder, &r, &g, &b, &a);
    CHECK(r == NVN_TEXTURE_SWIZZLE_B && g == NVN_TEXTURE_SWIZZLE_G);
    CHECK(b == NVN_TEXTURE_SWIZZLE_R && a == NVN_TEXTURE_SWIZZLE_ONE);
    const std::array packaged_data{std::byte{1}, std::byte{2}};
    set_data(&fixture.builder, packaged_data.data());
    CHECK(get_data(&fixture.builder) == packaged_data.data());
    NVNpackagedTextureLayout packaged_layout{{3, 1, 2, 3, 4, 5, 6, 7}};
    set_layout(&fixture.builder, &packaged_layout);
    NVNpackagedTextureLayout returned_layout{};
    CHECK(get_layout(&fixture.builder, &returned_layout));
    CHECK(std::memcmp(&returned_layout, &packaged_layout, sizeof(packaged_layout)) == 0);
    set_gl_name(&fixture.builder, 0x76543210u);
    CHECK(get_gl_name(&fixture.builder) == 0x76543210u);
    CHECK(get_width(&fixture.builder) == 64 && get_height(&fixture.builder) == 32);
    CHECK(get_levels(&fixture.builder) == 1 && get_flags(&fixture.builder) == 0);
    CHECK(get_pool(&fixture.builder) == &fixture.pool && get_offset(&fixture.builder) == 512);
    CHECK(get_device(&fixture.builder) == &device);
    CHECK(get_size(&fixture.builder) == 8192);
    CHECK(get_alignment(&fixture.builder) == 512);

    alignas(TextureViewState) NVNtextureView view{};
    auto view_defaults = entry<void (*)(NVNtextureView*)>("nvnTextureViewSetDefaults");
    auto view_set_levels = entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLevels");
    auto view_set_layers = entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers");
    auto view_set_format = entry<void (*)(NVNtextureView*, NVNformat)>("nvnTextureViewSetFormat");
    auto view_get_levels = entry<NVNboolean (*)(const NVNtextureView*, int*, int*)>(
        "nvnTextureViewGetLevels");
    auto view_get_layers = entry<NVNboolean (*)(const NVNtextureView*, int*, int*)>(
        "nvnTextureViewGetLayers");
    auto view_get_format = entry<NVNboolean (*)(const NVNtextureView*, NVNformat*)>(
        "nvnTextureViewGetFormat");
    auto view_compare = entry<NVNboolean (*)(const NVNtextureView*, const NVNtextureView*)>(
        "nvnTextureViewCompare");
    view_defaults(&view);
    view_set_levels(&view, 257, 258);
    view_set_layers(&view, -2, 5);
    view_set_format(&view, NVN_FORMAT_RGBA8);
    int base = 0, level_count = 0, first = 0, layer_count = 0;
    NVNformat format = NVN_FORMAT_NONE;
    CHECK(view_get_levels(&view, &base, &level_count));
    CHECK(base == 1 && level_count == 2);
    CHECK(view_get_layers(&view, &first, &layer_count));
    CHECK(first == -2 && layer_count == 5);
    CHECK(view_get_format(&view, &format) && format == NVN_FORMAT_RGBA8);
    alignas(TextureViewState) NVNtextureView copy = view;
    CHECK(view_compare(&view, &copy));
    view_defaults(&copy);
    CHECK(!view_compare(&view, &copy));
}

void configure_builder(TextureFixture& fixture, NVNtextureTarget target, NVNformat format,
                       int width, int height, int depth, int levels, int flags = 0,
                       ptrdiff_t stride = 0) {
    entry<void (*)(NVNtextureBuilder*)>("nvnTextureBuilderSetDefaults")(&fixture.builder);
    entry<void (*)(NVNtextureBuilder*, NVNtextureTarget)>("nvnTextureBuilderSetTarget")(
        &fixture.builder, target);
    entry<void (*)(NVNtextureBuilder*, int, int, int)>("nvnTextureBuilderSetSize3D")(
        &fixture.builder, width, height, depth);
    entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetLevels")(&fixture.builder,
                                                                          levels);
    entry<void (*)(NVNtextureBuilder*, NVNformat)>("nvnTextureBuilderSetFormat")(&fixture.builder,
                                                                                  format);
    entry<void (*)(NVNtextureBuilder*, int)>("nvnTextureBuilderSetFlags")(&fixture.builder,
                                                                           flags);
    entry<void (*)(NVNtextureBuilder*, ptrdiff_t)>("nvnTextureBuilderSetStride")(&fixture.builder,
                                                                                  stride);
    entry<void (*)(NVNtextureBuilder*, NVNmemoryPool*, ptrdiff_t)>("nvnTextureBuilderSetStorage")(
        &fixture.builder, &fixture.pool, 256);
}

// Serial and snapshot aside, an initialized object's bytes do not depend on what the storage held before.
void test_object_bytes_independent_of_storage(MockResources&) {
    TextureFixture fixture;
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 32, 1, 1);
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>(
        "nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");
    NVNtexture zero_filled, one_filled;
    std::memset(&zero_filled, 0x00, sizeof zero_filled);
    std::memset(&one_filled, 0xff, sizeof one_filled);
    CHECK(initialize(&zero_filled, &fixture.builder));
    CHECK(initialize(&one_filled, &fixture.builder));
    CHECK(std::memcmp(&zero_filled, &one_filled, offsetof(TextureState, serial)) == 0);
    finalize(&zero_filled);
    finalize(&one_filled);
}

void test_initialization_and_snapshot(MockResources& resources) {
    TextureFixture fixture;
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 32, 1, 1);
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>(
        "nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");
    CHECK(initialize(&fixture.texture, &fixture.builder));
    const auto& texture = object_state<TextureState>(fixture.texture);
    CHECK(texture.tag == object_tag('T', 'E', 'X', '\0'));
    CHECK(texture.size == 8192 && texture.layer_size == 8192);
    CHECK(texture.memory == reinterpret_cast<std::uint8_t*>(fixture.memory.data()) + 256);
    CHECK(texture.serial != 0 && texture.snapshot != nullptr);
    const auto* snapshot = static_cast<const TextureState*>(texture.snapshot);
    const std::uint32_t saved_width = snapshot->description.width;
    object_state<TextureState>(fixture.texture).description.width = 7;
    CHECK(snapshot->description.width == saved_width);
    CHECK(resources.registered == 1);
    // Finalizing an object leaves its identity registered: copies and other
    // objects of the same identity may still be in use, and records keep the
    // snapshot.
    auto copied = fixture.texture;
    const auto released_before = resources.released;
    finalize(&fixture.texture);
    CHECK(resources.released == released_before);
    CHECK(object_state<TextureState>(fixture.texture).tag == 0);
    CHECK(snapshot->description.width == saved_width);
    finalize(&copied);
    CHECK(snapshot->description.width == saved_width);
    // An identical initialization is the same texture: same serial, same
    // snapshot, no new host registration, whichever object it lands in.
    const auto first_serial = snapshot->serial;
    CHECK(initialize(&fixture.texture, &fixture.builder));
    CHECK(object_state<TextureState>(fixture.texture).serial == first_serial);
    CHECK(object_state<TextureState>(fixture.texture).snapshot == snapshot);
    alignas(TextureState) NVNtexture second{};
    CHECK(initialize(&second, &fixture.builder));
    CHECK(object_state<TextureState>(second).snapshot == snapshot);
    CHECK(resources.registered == 1);
    // A different identity has its own serial and registration.
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 32, 32, 1, 1);
    CHECK(initialize(&second, &fixture.builder));
    const auto* other = static_cast<const TextureState*>(object_state<TextureState>(second).snapshot);
    CHECK(other != snapshot && other->serial != first_serial);
    CHECK(resources.registered == 2);
    // Releasing the pool retires its identities: the host releases both, the
    // snapshots stay valid for records, and the next initialization starts a
    // new identity.
    retire_pool_textures(&object_state<MemoryPoolState>(fixture.pool));
    CHECK(resources.released == released_before + 2);
    CHECK(snapshot->serial == first_serial && snapshot->description.width == saved_width);
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 32, 1, 1);
    CHECK(initialize(&fixture.texture, &fixture.builder));
    CHECK(object_state<TextureState>(fixture.texture).serial != first_serial);
    CHECK(object_state<TextureState>(fixture.texture).snapshot != snapshot);
    CHECK(resources.registered == 3);
    finalize(&fixture.texture);
    finalize(&second);

    TextureFixture invalid(1024);
    configure_builder(invalid, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 32, 1, 1);
    CHECK(!initialize(&invalid.texture, &invalid.builder));
    auto& invalid_builder = object_state<TextureBuilderState>(fixture.builder);
    const auto set_packaged = entry<void (*)(NVNtextureBuilder*, const void*)>("nvnTextureBuilderSetPackagedTextureData");
    const auto set_gl_name = entry<void (*)(NVNtextureBuilder*, std::uint32_t)>("nvnTextureBuilderSetGLTextureName");
    // Packaged texture data is copied into the texture's storage at
    // initialization and published as a flush of exactly that range.
    const auto storage_size =
        entry<std::size_t (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetStorageSize")(&fixture.builder);
    std::vector<std::byte> packaged(storage_size);
    for (std::size_t i = 0; i < packaged.size(); ++i) packaged[i] = static_cast<std::byte>(i * 7 + 3);
    set_packaged(&fixture.builder, packaged.data());
    CHECK(entry<const void* (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetPackagedTextureData")(&fixture.builder) ==
          packaged.data());
    resources.flush_offset = resources.flush_size = 0;
    const auto packaged_offset = static_cast<std::size_t>(object_state<TextureBuilderState>(fixture.builder).offset);
    fixture.memory[packaged_offset] = std::byte{0xaa};
    const auto prepares = resources.prepares;
    CHECK(storage_size != 0 && initialize(&fixture.texture, &fixture.builder));
    // The backend drains its pool readers while the old bytes are in place,
    // then the bytes are copied and published.
    CHECK(resources.prepares == prepares + 1 && resources.prepared_first == std::byte{0xaa});
    CHECK(std::memcmp(fixture.memory.data() + packaged_offset, packaged.data(), packaged.size()) == 0);
    CHECK(resources.flush_offset == packaged_offset && resources.flush_size == packaged.size());
    finalize(&fixture.texture);
    set_packaged(&fixture.builder, nullptr);
    set_gl_name(&fixture.builder, 1);
    CHECK(entry<std::uint32_t (*)(const NVNtextureBuilder*)>("nvnTextureBuilderGetGLTextureName")(&fixture.builder) == 1);
    CHECK(!initialize(&fixture.texture, &fixture.builder));
    set_gl_name(&fixture.builder, 0);
    invalid_builder.description.flags |= NVN_TEXTURE_FLAGS_SPARSE;
    CHECK(!initialize(&fixture.texture, &fixture.builder));
}

void test_block_linear_transfers(MockResources& resources) {
    TextureFixture fixture;
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 16, 16, 3, 2);
    object_state<MemoryPoolState>(fixture.pool).flags = NVN_MEMORY_POOL_FLAGS_CPU_CACHED;
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>(
        "nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");
    CHECK(initialize(&fixture.texture, &fixture.builder));

    alignas(TextureViewState) NVNtextureView view{};
    entry<void (*)(NVNtextureView*)>("nvnTextureViewSetDefaults")(&view);
    entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLevels")(&view, 1, 1);
    entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers")(&view, 1, 2);
    const NVNcopyRegion region{1, 2, 0, 5, 3, 2};
    constexpr std::size_t row_stride = 32;
    constexpr std::size_t image_stride = 112;
    std::array<std::byte, image_stride * 2> source{};
    for (std::size_t image = 0; image < 2; ++image)
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t byte = 0; byte < 20; ++byte)
                source[image * image_stride + row * row_stride + byte] =
                    std::byte((image * 71 + row * 23 + byte + 1) & 0xff);
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*,
                   ptrdiff_t, ptrdiff_t)>("nvnTextureWriteTexelsStrided")(
        &fixture.texture, &view, &region, source.data(), row_stride, image_stride);
    std::array<std::byte, image_stride * 2> result{};
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, void*, ptrdiff_t,
                   ptrdiff_t)>("nvnTextureReadTexelsStrided")(
        &fixture.texture, &view, &region, result.data(), row_stride, image_stride);
    CHECK(source == result);
    CHECK(resources.writes >= 1 && resources.reads >= 1);

    auto& texture = object_state<TextureState>(fixture.texture);
    const auto layout = resource_layout(texture.description);
    CHECK(layout.has_value());
    if (layout) {
        for (std::uint32_t layer = 1; layer <= 2; ++layer) {
            const auto linear = graphics::convert_level_to_linear(
                *layout, std::span<const std::byte>(reinterpret_cast<const std::byte*>(texture.memory),
                                                    texture.size),
                1, layer);
            CHECK(std::holds_alternative<graphics::TextureLevelData>(linear));
        }
    }

    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)>(
        "nvnTextureFlushTexels")(&fixture.texture, &view, &region);
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)>(
        "nvnTextureInvalidateTexels")(&fixture.texture, &view, &region);
    CHECK(resources.flush_offset == 256 + 2048);
    CHECK(resources.flush_size == 4096);
    CHECK(resources.invalidate_offset == resources.flush_offset);
    CHECK(resources.invalidate_size == resources.flush_size);
    // A zero count selects the remaining layers, including on one-layer arrays.
    entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers")(&view, 1, 0);
    resources.flush_size = 0;
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)>(
        "nvnTextureFlushTexels")(&fixture.texture, &view, nullptr);
    CHECK(resources.flush_offset == 256 + 2048 && resources.flush_size == 4096);
    entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers")(&view, 3, 0);
    resources.flush_size = 0;
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)>(
        "nvnTextureFlushTexels")(&fixture.texture, &view, nullptr);
    CHECK(resources.flush_size == 0);
    // Uncached CPU writes apply to every layer even with an invalid view.
    object_state<MemoryPoolState>(fixture.pool).flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED;
    entry<void (*)(NVNtextureView*, int, int)>("nvnTextureViewSetLayers")(&view, -1, -1);
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)>(
        "nvnTextureFlushTexels")(&fixture.texture, &view, nullptr);
    CHECK(resources.flush_offset == 256 && resources.flush_size == 6144);
    finalize(&fixture.texture);
}

void test_compressed_and_linear_transfers() {
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>(
        "nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");

    TextureFixture bc;
    configure_builder(bc, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 17, 9, 1, 1);
    CHECK(initialize(&bc.texture, &bc.builder));
    const NVNcopyRegion bc_region{4, 4, 0, 13, 5, 1};
    std::array<std::byte, 64> bc_source{};
    for (std::size_t i = 0; i < 48; ++i) bc_source[i] = std::byte(i + 3);
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*)>(
        "nvnTextureWriteTexels")(&bc.texture, nullptr, &bc_region, bc_source.data());
    std::array<std::byte, 64> bc_result{};
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, void*)>(
        "nvnTextureReadTexels")(&bc.texture, nullptr, &bc_region, bc_result.data());
    CHECK(std::memcmp(bc_source.data(), bc_result.data(), 48) == 0);
    finalize(&bc.texture);

    TextureFixture linear;
    configure_builder(linear, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 13, 7, 1, 1,
                      NVN_TEXTURE_FLAGS_LINEAR, 80);
    CHECK(initialize(&linear.texture, &linear.builder));
    const NVNcopyRegion linear_region{2, 1, 0, 4, 3, 1};
    constexpr std::size_t row_stride = 24;
    std::array<std::byte, row_stride * 3> linear_source{};
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t byte = 0; byte < 16; ++byte)
            linear_source[row * row_stride + byte] = std::byte(row * 31 + byte);
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*,
                   ptrdiff_t, ptrdiff_t)>("nvnTextureWriteTexelsStrided")(
        &linear.texture, nullptr, &linear_region, linear_source.data(), row_stride, 0);
    std::array<std::byte, row_stride * 3> linear_result{};
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, void*, ptrdiff_t,
                   ptrdiff_t)>("nvnTextureReadTexelsStrided")(
        &linear.texture, nullptr, &linear_region, linear_result.data(), row_stride, 0);
    CHECK(linear_source == linear_result);
    const auto& texture = object_state<TextureState>(linear.texture);
    CHECK(texture.memory[80 + 2 * 4] == std::to_integer<std::uint8_t>(linear_source[0]));
    finalize(&linear.texture);
}

void test_overlapping_array_transfers() {
    TextureDescription description{};
    description.target = NVN_TEXTURE_TARGET_2D_ARRAY;
    description.format = NVN_FORMAT_RGBA8;
    description.width = description.height = description.levels = 1;
    description.depth = 3;
    const auto layout = resource_layout(description);
    CHECK(layout && layout->layer_stride == 512);
    if (!layout) return;
    std::vector<std::byte> storage(layout->total_size);
    TextureState texture{};
    texture.description = description;
    texture.size = storage.size();
    texture.memory = reinterpret_cast<std::uint8_t*>(storage.data());
    std::array<std::byte, 12> pixels{};
    std::fill(pixels.begin(), pixels.begin() + 4, std::byte{0x11});
    std::fill(pixels.begin() + 4, pixels.begin() + 8, std::byte{0x22});
    std::fill(pixels.begin() + 8, pixels.end(), std::byte{0x33});
    TexelTransfer all{};
    all.region = {0, 0, 0, 1, 1, 3};
    CHECK(transfer_pool_texels(texture, all, pixels.data(), true));
    TexelTransfer shifted{};
    shifted.first_layer = 1;
    shifted.region = {0, 0, 0, 1, 1, 2};
    shifted.image_stride = 512;
    CHECK(transfer_pool_texels(texture, shifted, storage.data(), true));
    std::array<std::byte, 12> result{};
    CHECK(transfer_pool_texels(texture, all, result.data(), false));
    CHECK(result[4] == std::byte{0x11} && result[8] == std::byte{0x22});
    CHECK(transfer_pool_texels(texture, all, pixels.data(), true));
    shifted.first_layer = 0;
    CHECK(transfer_pool_texels(texture, shifted, storage.data() + 512, false));
    CHECK(storage[512] == std::byte{0x11} && storage[1024] == std::byte{0x22});
}

void test_transfer_validation(MockResources& resources) {
    TextureFixture fixture;
    configure_builder(fixture, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 16, 16, 1, 1);
    auto initialize = entry<NVNboolean (*)(NVNtexture*, const NVNtextureBuilder*)>(
        "nvnTextureInitialize");
    auto finalize = entry<void (*)(NVNtexture*)>("nvnTextureFinalize");
    CHECK(initialize(&fixture.texture, &fixture.builder));
    const unsigned writes_before = resources.writes;
    const NVNcopyRegion misaligned{1, 0, 0, 4, 4, 1};
    std::array<std::byte, 16> bytes{};
    entry<void (*)(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*)>(
        "nvnTextureWriteTexels")(&fixture.texture, nullptr, &misaligned, bytes.data());
    CHECK(resources.writes == writes_before + 1);
    CHECK(std::all_of(fixture.memory.begin(), fixture.memory.end(),
                      [](std::byte value) { return value == std::byte{}; }));
    finalize(&fixture.texture);
}

} // namespace

// The bytes a GPU deswizzle copies for one addressed level: each 16-byte
// segment of a linear row from its contiguous guest run, at the addressing
// formula with the segment's place in its GOB written out here.
std::vector<std::byte> segment_copy(const graphics::BlockLinearLevelAddressing& level,
                                    const std::byte* storage, std::size_t storage_size,
                                    bool& in_bounds) {
    std::vector<std::byte> linear(level.row_bytes * level.rows * level.slices);
    const std::size_t height_mask = (std::size_t{1} << level.gob_block_height_log2) - 1;
    const std::size_t depth_mask = (std::size_t{1} << level.gob_block_depth_log2) - 1;
    in_bounds = true;
    for (std::size_t z = 0; z < level.slices; ++z)
        for (std::size_t y = 0; y < level.rows; ++y)
            for (std::size_t x = 0; x < level.row_bytes; x += 16) {
                const std::size_t count = std::min<std::size_t>(16, level.row_bytes - x);
                const std::size_t gob_y = y / 8;
                const std::size_t from =
                    level.guest_base + (z >> level.gob_block_depth_log2) * level.slice_stride +
                    (gob_y >> level.gob_block_height_log2) * level.row_stride +
                    ((x / 64) << (9 + level.gob_block_height_log2 + level.gob_block_depth_log2)) +
                    (gob_y & height_mask) * 512 +
                    (((z & depth_mask) * 512) << level.gob_block_height_log2) +
                    ((x % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x % 32) / 16) * 32 +
                    (y % 2) * 16;
                if (from + count > storage_size) {
                    in_bounds = false;
                    return linear;
                }
                std::memcpy(linear.data() + (z * level.rows + y) * level.row_bytes + x,
                            storage + from, count);
            }
    return linear;
}

void test_pool_texture_addressing() {
    // Whole-level reads of every layer and level through the addressing equal
    // the texel transfer's CPU deswizzle, for each block height, mip chains,
    // arrays, cube maps, volumes, compressed blocks and sizes that do not fill
    // their GOBs.
    std::vector<TextureDescription> descriptions;
    const auto add = [&](std::int32_t target, std::int32_t format, std::uint32_t width,
                         std::uint32_t height, std::uint32_t depth, std::uint32_t levels,
                         std::int32_t packaged_block_height = 0) {
        TextureDescription description{};
        description.target = target;
        description.format = format;
        description.samples = 1;
        description.width = width;
        description.height = height;
        description.depth = depth;
        description.levels = levels;
        description.packaged_block_height = packaged_block_height;
        descriptions.push_back(description);
    };
    for (std::int32_t packaged = 1; packaged <= 6; ++packaged)
        add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 70, 300, 1, 1, packaged);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 37, 23, 1, 3);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 64, 1, 7);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_R8, 5, 3, 1, 1);
    add(NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 20, 12, 3, 2);
    add(NVN_TEXTURE_TARGET_CUBEMAP, NVN_FORMAT_RGBA16F, 16, 16, 1, 3);
    add(NVN_TEXTURE_TARGET_1D_ARRAY, NVN_FORMAT_R8, 70, 3, 1, 2);
    add(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_R8, 8, 8, 11, 4);
    add(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8, 16, 16, 9, 3);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 36, 20, 1, 3);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_BPTC_UNORM, 24, 24, 1, 2);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_6x5, 50, 23, 1, 2);
    add(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGB32F, 13, 9, 1, 2);
    for (const auto& description : descriptions) {
        const auto layout = resource_layout(description);
        CHECK(layout.has_value());
        if (!layout) continue;
        std::vector<std::byte> storage(layout->total_size);
        std::uint32_t random = 0x9e3779b9u ^ static_cast<std::uint32_t>(description.format);
        for (auto& byte : storage) {
            random = random * 1664525u + 1013904223u;
            byte = std::byte{static_cast<unsigned char>(random >> 24)};
        }
        TextureState texture{};
        texture.description = description;
        texture.size = storage.size();
        texture.memory = reinterpret_cast<std::uint8_t*>(storage.data());
        std::vector<graphics::BlockLinearLevelAddressing> levels;
        CHECK(pool_texture_addressing(texture, levels));
        CHECK(levels.size() == std::size_t{layout->layers} * layout->levels);
        if (levels.size() != std::size_t{layout->layers} * layout->levels) continue;
        const bool is_3d = description.target == NVN_TEXTURE_TARGET_3D;
        for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
            for (std::uint32_t level = 0; level < layout->levels; ++level) {
                const auto& mip = layout->mips[level];
                const auto& addressing = levels[std::size_t{layer} * layout->levels + level];
                TexelTransfer transfer{};
                transfer.level = level;
                transfer.first_layer = is_3d ? 0 : layer;
                transfer.region = {0, 0, 0, static_cast<int>(mip.width),
                                   static_cast<int>(mip.height),
                                   static_cast<int>(is_3d ? mip.depth : 1)};
                std::vector<std::byte> expected(mip.linear_bytes);
                CHECK(transfer_pool_texels(texture, transfer, expected.data(), false));
                bool in_bounds = false;
                const auto copied = segment_copy(addressing, storage.data(), storage.size(),
                                                 in_bounds);
                CHECK(in_bounds);
                CHECK(addressing.guest_base + addressing.guest_size <= storage.size());
                CHECK(copied == expected);
            }
        }
    }

    // Pitch storage, multisample storage and storage smaller than its layout
    // have no whole-level block-linear reads.
    TextureDescription blocks{};
    blocks.target = NVN_TEXTURE_TARGET_2D;
    blocks.format = NVN_FORMAT_RGBA8;
    blocks.samples = 1;
    blocks.width = blocks.height = 4;
    blocks.depth = blocks.levels = 1;
    TextureDescription pitch = blocks;
    pitch.flags = NVN_TEXTURE_FLAGS_LINEAR;
    TextureDescription multisample = blocks;
    multisample.target = NVN_TEXTURE_TARGET_2D_MULTISAMPLE;
    multisample.samples = 4;
    const std::array<std::pair<TextureDescription, std::size_t>, 3> refused{{
        {pitch, 0}, {multisample, 0}, {blocks, 1}}};
    for (const auto& [description, missing] : refused) {
        const auto layout = resource_layout(description);
        CHECK(layout.has_value());
        if (!layout) continue;
        std::vector<std::byte> storage(layout->total_size);
        TextureState texture{};
        texture.description = description;
        texture.size = storage.size() - missing;
        texture.memory = reinterpret_cast<std::uint8_t*>(storage.data());
        std::vector<graphics::BlockLinearLevelAddressing> levels;
        CHECK(!pool_texture_addressing(texture, levels));
        CHECK(levels.empty());
    }
}

int main() {
    select_api_version({55, 15});
    register_resource_entries();
    MockResources resources;
    set_backends({&resources, nullptr});

    test_resource_layouts();
    test_builder_and_view_entries();
    test_initialization_and_snapshot(resources);
    test_block_linear_transfers(resources);
    test_compressed_and_linear_transfers();
    test_transfer_validation(resources);
    test_overlapping_array_transfers();
    test_pool_texture_addressing();
    test_object_bytes_independent_of_storage(resources);

    set_backends({});
    std::printf("NVN texture resource failures: %u\n", failures);
    return failures ? 1 : 0;
}
