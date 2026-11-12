// ============================================================================
// Armigo Tests - NVN Samplers and Descriptors
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "backend.h"
#include "entry_table.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

namespace armigo::nvn { void register_resource_entries(); }
namespace {
using namespace armigo::nvn;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (false)
template <Entry E> auto entry() {
    return reinterpret_cast<typename EntrySignature<E>::type>(armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

struct Backend final : ResourceBackend {
    int textures = 0, samplers = 0, last_id = -1, released_texture_pools = 0;
    int cleared_textures = 0;
    bool descriptor_pools_allowed = true;
    std::uint64_t serial = 0;
    SamplerParameters sampler{};
    TextureViewState view{};
    bool register_pool(const MemoryPoolState&) override { return true; }
    void release_pool(const MemoryPoolState&) override {}
    void flush_pool(const MemoryPoolState&, std::uint64_t, std::uint64_t) override {}
    void invalidate_pool(const MemoryPoolState&, std::uint64_t, std::uint64_t) override {}
    void register_texture(const TextureState&) override {}
    void release_texture(const TextureState&) override {}
    bool register_texture_pool(const DescriptorPoolState&) override { return descriptor_pools_allowed; }
    bool register_sampler_pool(const DescriptorPoolState&) override { return descriptor_pools_allowed; }
    void release_texture_pool(const DescriptorPoolState&) override { ++released_texture_pools; }
    void register_texture_descriptor(std::int32_t id, const TextureState& texture, const TextureViewState* v) override {
        ++textures; last_id = id; serial = texture.serial; if (v) view = *v;
    }
    void clear_texture_descriptor(std::int32_t id) override { ++cleared_textures; last_id = id; }
    void register_sampler_descriptor(std::int32_t id, const SamplerParameters& p) override {
        ++samplers; last_id = id; sampler = p;
    }
    bool read_texels(const TextureState&, const TexelTransfer&, std::byte*) override { return false; }
    bool write_texels(const TextureState&, const TexelTransfer&, const std::byte*) override { return false; }
};
}

int main() {
    using namespace armigo::nvn;
    register_resource_entries();
    select_api_version({55, 15});
    Backend backend;
    set_backends({&backend, nullptr});

    alignas(8) NVNsamplerBuilder builder{};
    NVNdevice device{};
    entry<Entry::nvnSamplerBuilderSetDevice>()(&builder, &device);
    entry<Entry::nvnSamplerBuilderSetDefaults>()(&builder);
    CHECK(entry<Entry::nvnSamplerBuilderGetDevice>()(&builder) == &device);
    NVNminFilter min{}; NVNmagFilter mag{};
    entry<Entry::nvnSamplerBuilderGetMinMagFilter>()(&builder, &min, &mag);
    CHECK(min == NVN_MIN_FILTER_LINEAR && mag == NVN_MAG_FILTER_LINEAR);
    const std::array<std::uint32_t, 4> integer_border{0xffffffffu, 0x80000000u, 0x7fc12345u, 17};
    entry<Entry::nvnSamplerBuilderSetBorderColorui>()(&builder, integer_border.data());
    std::array<std::uint32_t, 4> recovered{};
    entry<Entry::nvnSamplerBuilderGetBorderColorui>()(&builder, recovered.data());
    CHECK(integer_border == recovered);
    entry<Entry::nvnSamplerBuilderSetReductionFilter>()(&builder, NVN_SAMPLER_REDUCTION_MAX);
    entry<Entry::nvnSamplerBuilderSetLodSnap>()(&builder, 0.25f);
    CHECK(entry<Entry::nvnSamplerBuilderGetLodSnap>()(&builder) == 0.25f);
    alignas(8) NVNsampler first{}, second{};
    CHECK(entry<Entry::nvnSamplerInitialize>()(&first, &builder));
    CHECK(entry<Entry::nvnSamplerInitialize>()(&second, &builder));
    CHECK(entry<Entry::nvnSamplerGetDebugID>()(&first) != entry<Entry::nvnSamplerGetDebugID>()(&second));
    // Equal state gives equal object bytes (a game's sampler cache may hash them); only the debug ID tells them apart.
    CHECK(std::memcmp(&first, &second, sizeof(SamplerState)) == 0);
    CHECK(entry<Entry::nvnSamplerCompare>()(&first, &second));
    CHECK(entry<Entry::nvnSamplerGetReductionFilter>()(&first) == NVN_SAMPLER_REDUCTION_MAX);
    entry<Entry::nvnSamplerGetBorderColorui>()(&first, recovered.data());
    CHECK(integer_border == recovered);
    entry<Entry::nvnSamplerBuilderSetMinMagFilter>()(&builder, static_cast<NVNminFilter>(256), NVN_MAG_FILTER_LINEAR);
    CHECK(!entry<Entry::nvnSamplerInitialize>()(&second, &builder));
    entry<Entry::nvnSamplerBuilderSetDefaults>()(&builder);
    entry<Entry::nvnSamplerBuilderSetMaxAnisotropy>()(&builder, std::numeric_limits<float>::quiet_NaN());
    CHECK(!entry<Entry::nvnSamplerInitialize>()(&second, &builder));
    entry<Entry::nvnSamplerBuilderSetDefaults>()(&builder);

    alignas(4096) std::array<std::byte, 16384> memory{};
    alignas(8) NVNmemoryPool pool{};
    auto& pool_state = *reinterpret_cast<MemoryPoolState*>(&pool);
    pool_state.tag = object_tag('P', 'O', 'O', 'L');
    pool_state.flags = NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
    pool_state.memory = reinterpret_cast<std::uint8_t*>(memory.data());
    pool_state.size = memory.size();
    alignas(8) NVNsamplerPool samplers{};
    alignas(8) NVNtexturePool textures{};
    CHECK(!entry<Entry::nvnSamplerPoolInitialize>()(&samplers, &pool, 1, 300));
    CHECK(!entry<Entry::nvnSamplerPoolInitialize>()(&samplers, &pool, 0, 4097));
    CHECK(!entry<Entry::nvnTexturePoolInitialize>()(&textures, &pool, 0, 513));
    backend.descriptor_pools_allowed = false;
    CHECK(!entry<Entry::nvnSamplerPoolInitialize>()(&samplers, &pool, 256, 300));
    CHECK(!entry<Entry::nvnTexturePoolInitialize>()(&textures, &pool, 0, 300));
    backend.descriptor_pools_allowed = true;
    CHECK(entry<Entry::nvnSamplerPoolInitialize>()(&samplers, &pool, 256, 300));
    // Initialization registers the driver samplers in slots 0 and 1:
    // clamp-to-edge builder defaults, anisotropy 1 and 2.
    CHECK(backend.samplers == 2 && backend.last_id == 1);
    CHECK(backend.sampler.wrap_s == NVN_WRAP_MODE_CLAMP_TO_EDGE && backend.sampler.wrap_r == NVN_WRAP_MODE_CLAMP_TO_EDGE &&
          backend.sampler.min_filter == NVN_MIN_FILTER_LINEAR && backend.sampler.mag_filter == NVN_MAG_FILTER_LINEAR &&
          backend.sampler.max_lod == 1000.0f && backend.sampler.max_anisotropy == 2.0f);
    backend.samplers = 0;
    CHECK(entry<Entry::nvnTexturePoolInitialize>()(&textures, &pool, 0, 300));
    CHECK(entry<Entry::nvnSamplerPoolGetMemoryPool>()(&samplers) == &pool);
    CHECK(entry<Entry::nvnSamplerPoolGetMemoryOffset>()(&samplers) == 256);
    CHECK(entry<Entry::nvnTexturePoolGetSize>()(&textures) == 300);
    entry<Entry::nvnSamplerPoolRegisterSampler>()(&samplers, 255, &first);
    entry<Entry::nvnSamplerPoolRegisterSampler>()(&samplers, 300, &first);
    CHECK(backend.samplers == 0);
    entry<Entry::nvnSamplerPoolRegisterSampler>()(&samplers, 256, &first);
    CHECK(backend.samplers == 1 && backend.last_id == 256);
    CHECK(backend.sampler.reduction_filter == NVN_SAMPLER_REDUCTION_MAX);
    entry<Entry::nvnSamplerPoolRegisterSamplerBuilder>()(&samplers, 299, &builder);
    CHECK(backend.samplers == 2 && backend.last_id == 299);
    alignas(8) NVNtexture texture{};
    auto& texture_state = *reinterpret_cast<TextureState*>(&texture);
    texture_state.tag = object_tag('T', 'E', 'X', 0);
    texture_state.serial = 123;
    alignas(8) NVNtextureView view{};
    auto& view_state = *reinterpret_cast<TextureViewState*>(&view);
    view_state.sets_levels = 1; view_state.base_level = 2; view_state.level_count = 1;
    entry<Entry::nvnTexturePoolRegisterTexture>()(&textures, 256, &texture, &view);
    CHECK(backend.textures == 1 && backend.serial == 123 && backend.view.base_level == 2);
    entry<Entry::nvnTexturePoolRegisterImage>()(&textures, 257, &texture, nullptr);
    CHECK(backend.textures == 2 && backend.last_id == 257);
    entry<Entry::nvnTexturePoolRegisterTexture>()(&textures, 256, nullptr, &view);
    CHECK(backend.cleared_textures == 1 && backend.last_id == 256);
    entry<Entry::nvnTexturePoolRegisterImage>()(&textures, 257, nullptr, nullptr);
    CHECK(backend.cleared_textures == 2 && backend.last_id == 257);
    entry<Entry::nvnTexturePoolRegisterTexture>()(&textures, 255, nullptr, nullptr);
    entry<Entry::nvnTexturePoolRegisterImage>()(&textures, 300, nullptr, nullptr);
    CHECK(backend.cleared_textures == 2 && backend.textures == 2);
    entry<Entry::nvnTexturePoolFinalize>()(&textures);
    CHECK(backend.released_texture_pools == 1);
    entry<Entry::nvnTexturePoolRegisterImage>()(&textures, 257, &texture, nullptr);
    CHECK(backend.textures == 2);
    entry<Entry::nvnSamplerFinalize>()(&first);
    entry<Entry::nvnSamplerPoolRegisterSampler>()(&samplers, 256, &first);
    CHECK(backend.samplers == 2);
    set_backends({});
    std::printf("NVN sampler and descriptor failures: %u\n", failures);
    return failures != 0;
}
