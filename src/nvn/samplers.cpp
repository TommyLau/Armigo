// ============================================================================
// Armigo NVN - Samplers and Descriptor Pools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "backend.h"
#include "entry_table.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace armigo::nvn {
void register_device_entries() noexcept;
void register_memory_entries() noexcept;
void register_texture_entries();

namespace {
constexpr auto sampler_tag = object_tag('S', 'M', 'P', 0);
constexpr std::int32_t reserved_descriptors = 256;
constexpr std::uint64_t descriptor_size = 32;

template <typename State, typename Object> State& state(Object* object) {
    return *reinterpret_cast<State*>(object);
}
template <typename State, typename Object> const State& state(const Object* object) {
    return *reinterpret_cast<const State*>(object);
}
void invalid(const char* operation) {
    armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: invalid or unsupported %s\n", operation);
}
std::uint8_t enum_value(int value, int maximum) {
    if (value < 0 || value > maximum) {
        invalid("sampler enum");
        return 255;
    }
    return static_cast<std::uint8_t>(value);
}
// Enumerations must name defined values; LOD range, bias, anisotropy and snap
// take any number, as the API accepts them (negative minimum LODs included).
// The Metal backend clamps them to what Metal supports.
bool valid(const SamplerParameters& p) {
    return p.min_filter <= NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR && p.mag_filter <= NVN_MAG_FILTER_LINEAR &&
           p.wrap_s <= NVN_WRAP_MODE_CLAMP_TO_EDGE && p.wrap_t <= NVN_WRAP_MODE_CLAMP_TO_EDGE &&
           p.wrap_r <= NVN_WRAP_MODE_CLAMP_TO_EDGE && p.compare_mode <= NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE &&
           p.compare_function >= NVN_COMPARE_FUNC_NEVER && p.compare_function <= NVN_COMPARE_FUNC_ALWAYS &&
           p.reduction_filter <= NVN_SAMPLER_REDUCTION_MAX && !std::isnan(p.min_lod) && !std::isnan(p.max_lod) &&
           !std::isnan(p.lod_bias) && !std::isnan(p.max_anisotropy) && !std::isnan(p.lod_snap);
}
void nvnSamplerBuilderSetDefaults(NVNsamplerBuilder* builder) {
    auto& b = state<SamplerBuilderState>(builder);
    const auto device = b.device;
    b = {};
    b.device = device;
    b.parameters.min_filter = NVN_MIN_FILTER_LINEAR;
    b.parameters.mag_filter = NVN_MAG_FILTER_LINEAR;
    b.parameters.wrap_s = b.parameters.wrap_t = b.parameters.wrap_r = NVN_WRAP_MODE_REPEAT;
    b.parameters.compare_function = NVN_COMPARE_FUNC_LESS;
    b.parameters.max_lod = 1000.0f;
    b.parameters.max_anisotropy = 1.0f;
}
NVNdevice* nvnSamplerBuilderGetDevice(const NVNsamplerBuilder* b) { return state<SamplerBuilderState>(b).device; }
void nvnSamplerBuilderSetDevice(NVNsamplerBuilder* b, NVNdevice* d) { state<SamplerBuilderState>(b).device = d; }
void nvnSamplerBuilderSetMinMagFilter(NVNsamplerBuilder* b, NVNminFilter min, NVNmagFilter mag) {
    auto& p = state<SamplerBuilderState>(b).parameters;
    p.min_filter = enum_value(min, 5);
    p.mag_filter = enum_value(mag, 1);
}
void nvnSamplerBuilderSetWrapMode(NVNsamplerBuilder* b, NVNwrapMode s, NVNwrapMode t, NVNwrapMode r) {
    auto& p = state<SamplerBuilderState>(b).parameters;
    p.wrap_s = enum_value(s, 7); p.wrap_t = enum_value(t, 7); p.wrap_r = enum_value(r, 7);
}
void nvnSamplerBuilderSetLodClamp(NVNsamplerBuilder* b, float low, float high) {
    auto& p = state<SamplerBuilderState>(b).parameters;
    p.min_lod = low; p.max_lod = high;
}
void nvnSamplerBuilderSetCompare(NVNsamplerBuilder* b, NVNcompareMode mode, NVNcompareFunc function) {
    auto& p = state<SamplerBuilderState>(b).parameters;
    p.compare_mode = enum_value(mode, 1); p.compare_function = enum_value(function, 8);
}
void nvnSamplerBuilderSetReductionFilter(NVNsamplerBuilder* b, NVNsamplerReduction reduction) {
    state<SamplerBuilderState>(b).parameters.reduction_filter = enum_value(reduction, 2);
}
void nvnSamplerBuilderSetLodBias(NVNsamplerBuilder* b, float value) { state<SamplerBuilderState>(b).parameters.lod_bias = value; }
void nvnSamplerBuilderSetMaxAnisotropy(NVNsamplerBuilder* b, float value) { state<SamplerBuilderState>(b).parameters.max_anisotropy = value; }
void nvnSamplerBuilderSetLodSnap(NVNsamplerBuilder* b, float value) { state<SamplerBuilderState>(b).parameters.lod_snap = value; }
void nvnSamplerBuilderSetBorderColor(NVNsamplerBuilder* b, const float* color) {
    if (!color) { invalid("sampler border color"); return; }
    std::memcpy(state<SamplerBuilderState>(b).parameters.border, color, 16);
}
void nvnSamplerBuilderSetBorderColori(NVNsamplerBuilder* b, const int* color) {
    if (!color) { invalid("sampler border color"); return; }
    std::memcpy(state<SamplerBuilderState>(b).parameters.border, color, 16);
}
void nvnSamplerBuilderSetBorderColorui(NVNsamplerBuilder* b, const std::uint32_t* color) {
    if (!color) { invalid("sampler border color"); return; }
    std::memcpy(state<SamplerBuilderState>(b).parameters.border, color, 16);
}
void nvnSamplerBuilderGetMinMagFilter(const NVNsamplerBuilder* object, NVNminFilter* min, NVNmagFilter* mag) {
    const auto& p = state<SamplerBuilderState>(object).parameters;
    if (min) *min = static_cast<NVNminFilter>(p.min_filter);
    if (mag) *mag = static_cast<NVNmagFilter>(p.mag_filter);
}
void nvnSamplerBuilderGetWrapMode(const NVNsamplerBuilder* object, NVNwrapMode* s, NVNwrapMode* t, NVNwrapMode* r) {
    const auto& p = state<SamplerBuilderState>(object).parameters;
    if (s) *s = static_cast<NVNwrapMode>(p.wrap_s);
    if (t) *t = static_cast<NVNwrapMode>(p.wrap_t);
    if (r) *r = static_cast<NVNwrapMode>(p.wrap_r);
}
void nvnSamplerBuilderGetLodClamp(const NVNsamplerBuilder* object, float* low, float* high) {
    const auto& p = state<SamplerBuilderState>(object).parameters;
    if (low) *low = p.min_lod;
    if (high) *high = p.max_lod;
}
void nvnSamplerBuilderGetCompare(const NVNsamplerBuilder* object, NVNcompareMode* mode, NVNcompareFunc* function) {
    const auto& p = state<SamplerBuilderState>(object).parameters;
    if (mode) *mode = static_cast<NVNcompareMode>(p.compare_mode);
    if (function) *function = static_cast<NVNcompareFunc>(p.compare_function);
}
void nvnSamplerBuilderGetBorderColor(const NVNsamplerBuilder* object, float* color) {
    if (color) std::memcpy(color, state<SamplerBuilderState>(object).parameters.border, 16);
}
void nvnSamplerBuilderGetBorderColori(const NVNsamplerBuilder* object, int* color) {
    if (color) std::memcpy(color, state<SamplerBuilderState>(object).parameters.border, 16);
}
void nvnSamplerBuilderGetBorderColorui(const NVNsamplerBuilder* object, std::uint32_t* color) {
    if (color) std::memcpy(color, state<SamplerBuilderState>(object).parameters.border, 16);
}
float nvnSamplerBuilderGetLodBias(const NVNsamplerBuilder* object) {
    return static_cast<float>(state<SamplerBuilderState>(object).parameters.lod_bias);
}
float nvnSamplerBuilderGetMaxAnisotropy(const NVNsamplerBuilder* object) {
    return static_cast<float>(state<SamplerBuilderState>(object).parameters.max_anisotropy);
}
NVNsamplerReduction nvnSamplerBuilderGetReductionFilter(const NVNsamplerBuilder* object) {
    return static_cast<NVNsamplerReduction>(state<SamplerBuilderState>(object).parameters.reduction_filter);
}
float nvnSamplerBuilderGetLodSnap(const NVNsamplerBuilder* object) {
    return static_cast<float>(state<SamplerBuilderState>(object).parameters.lod_snap);
}
void nvnSamplerGetMinMagFilter(const NVNsampler* object, NVNminFilter* min, NVNmagFilter* mag) {
    const auto& p = state<SamplerState>(object).parameters;
    if (min) *min = static_cast<NVNminFilter>(p.min_filter);
    if (mag) *mag = static_cast<NVNmagFilter>(p.mag_filter);
}
void nvnSamplerGetWrapMode(const NVNsampler* object, NVNwrapMode* s, NVNwrapMode* t, NVNwrapMode* r) {
    const auto& p = state<SamplerState>(object).parameters;
    if (s) *s = static_cast<NVNwrapMode>(p.wrap_s);
    if (t) *t = static_cast<NVNwrapMode>(p.wrap_t);
    if (r) *r = static_cast<NVNwrapMode>(p.wrap_r);
}
void nvnSamplerGetLodClamp(const NVNsampler* object, float* low, float* high) {
    const auto& p = state<SamplerState>(object).parameters;
    if (low) *low = p.min_lod;
    if (high) *high = p.max_lod;
}
void nvnSamplerGetCompare(const NVNsampler* object, NVNcompareMode* mode, NVNcompareFunc* function) {
    const auto& p = state<SamplerState>(object).parameters;
    if (mode) *mode = static_cast<NVNcompareMode>(p.compare_mode);
    if (function) *function = static_cast<NVNcompareFunc>(p.compare_function);
}
void nvnSamplerGetBorderColor(const NVNsampler* object, float* color) {
    if (color) std::memcpy(color, state<SamplerState>(object).parameters.border, 16);
}
void nvnSamplerGetBorderColori(const NVNsampler* object, int* color) {
    if (color) std::memcpy(color, state<SamplerState>(object).parameters.border, 16);
}
void nvnSamplerGetBorderColorui(const NVNsampler* object, std::uint32_t* color) {
    if (color) std::memcpy(color, state<SamplerState>(object).parameters.border, 16);
}
float nvnSamplerGetLodBias(const NVNsampler* object) {
    return static_cast<float>(state<SamplerState>(object).parameters.lod_bias);
}
float nvnSamplerGetMaxAnisotropy(const NVNsampler* object) {
    return static_cast<float>(state<SamplerState>(object).parameters.max_anisotropy);
}
NVNsamplerReduction nvnSamplerGetReductionFilter(const NVNsampler* object) {
    return static_cast<NVNsamplerReduction>(state<SamplerState>(object).parameters.reduction_filter);
}
NVNboolean nvnSamplerInitialize(NVNsampler* sampler, const NVNsamplerBuilder* builder) {
    if (!sampler || !builder || !valid(state<SamplerBuilderState>(builder).parameters)) {
        invalid("sampler initialization"); return 0;
    }
    const auto& b = state<SamplerBuilderState>(builder);
    state<SamplerState>(sampler) = {sampler_tag, b.parameters, b.device, 0};
    return 1;
}
void nvnSamplerFinalize(NVNsampler* sampler) { if (sampler) state<SamplerState>(sampler) = {}; }
void nvnSamplerSetDebugLabel(NVNsampler*, const char*) {}
std::uint64_t nvnSamplerGetDebugID(const NVNsampler* sampler) { return reinterpret_cast<std::uintptr_t>(sampler); }
NVNboolean nvnSamplerCompare(const NVNsampler* a, const NVNsampler* b) {
    if (!a || !b) return 0;
    const auto& x = state<SamplerState>(a);
    const auto& y = state<SamplerState>(b);
    const auto& p = x.parameters;
    const auto& q = y.parameters;
    return x.tag == sampler_tag && y.tag == sampler_tag && x.device == y.device &&
           p.min_filter == q.min_filter && p.mag_filter == q.mag_filter &&
           p.wrap_s == q.wrap_s && p.wrap_t == q.wrap_t && p.wrap_r == q.wrap_r &&
           p.compare_mode == q.compare_mode && p.compare_function == q.compare_function &&
           p.reduction_filter == q.reduction_filter && p.min_lod == q.min_lod && p.max_lod == q.max_lod &&
           p.lod_bias == q.lod_bias && p.max_anisotropy == q.max_anisotropy && p.lod_snap == q.lod_snap &&
           std::memcmp(p.border, q.border, sizeof(p.border)) == 0;
}

template <typename Pool>
NVNboolean initialize_pool(Pool* descriptor_pool, const NVNmemoryPool* memory_pool,
                           ptrdiff_t offset, int count, int maximum) {
    if (!descriptor_pool || !memory_pool || offset < 0 || (offset & 255) ||
        count < reserved_descriptors || count > maximum) {
        invalid("descriptor pool initialization"); return 0;
    }
    const auto& pool = state<MemoryPoolState>(memory_pool);
    const auto bytes = static_cast<std::uint64_t>(count) * descriptor_size;
    if (!pool.tag || !pool.memory || (pool.flags & (NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS |
        NVN_MEMORY_POOL_FLAGS_PHYSICAL)) || static_cast<std::uint64_t>(offset) > pool.size ||
        bytes > pool.size - static_cast<std::uint64_t>(offset)) {
        invalid("descriptor pool storage"); return 0;
    }
    const DescriptorPoolState candidate{&pool, offset, count};
    auto* backend = backends().resources;
    if (!backend || !(maximum == (1 << 12) ? backend->register_sampler_pool(candidate)
                                           : backend->register_texture_pool(candidate))) {
        invalid("descriptor pool backend registration"); return 0;
    }
    state<DescriptorPoolState>(descriptor_pool) = candidate;
    return 1;
}
bool valid_slot(const DescriptorPoolState& pool, int id, bool sampler = false) {
    if (!pool.pool || !pool.pool->tag || id < reserved_descriptors || id >= pool.count) {
        invalid("descriptor pool slot"); return false;
    }
    if (!backends().resources) { invalid("descriptor registration without resource backend"); return false; }
    if (!(sampler ? backends().resources->register_sampler_pool(pool)
                   : backends().resources->register_texture_pool(pool))) {
        invalid("descriptor pool selection"); return false;
    }
    return true;
}
NVNboolean nvnTexturePoolInitialize(NVNtexturePool* p, const NVNmemoryPool* m, ptrdiff_t o, int n) {
    return initialize_pool(p, m, o, n, 1 << 20);
}
// Sampler pool initialization writes two driver samplers into slots 0 and 1:
// the builder defaults with clamp-to-edge wrapping in all three dimensions,
// slot 0 with anisotropy 1 and slot 1 with anisotropy 2. Shaders sample
// through slot 0 when a texture handle carries no sampler.
NVNboolean nvnSamplerPoolInitialize(NVNsamplerPool* p, const NVNmemoryPool* m, ptrdiff_t o, int n) {
    if (!initialize_pool(p, m, o, n, 1 << 12)) return 0;
    SamplerParameters builtin{};
    builtin.min_filter = NVN_MIN_FILTER_LINEAR;
    builtin.mag_filter = NVN_MAG_FILTER_LINEAR;
    builtin.wrap_s = builtin.wrap_t = builtin.wrap_r = NVN_WRAP_MODE_CLAMP_TO_EDGE;
    builtin.compare_function = NVN_COMPARE_FUNC_LESS;
    builtin.max_lod = 1000.0f;
    builtin.max_anisotropy = 1.0f;
    backends().resources->register_sampler_descriptor(0, builtin);
    builtin.max_anisotropy = 2.0f;
    backends().resources->register_sampler_descriptor(1, builtin);
    return 1;
}
void nvnTexturePoolRegisterTexture(const NVNtexturePool* p, int id, const NVNtexture* t, const NVNtextureView* v) {
    if (!p || (t && !state<TextureState>(t).tag)) { invalid("texture descriptor"); return; }
    if (!valid_slot(state<DescriptorPoolState>(p), id)) return;
    if (!t) {
        backends().resources->clear_texture_descriptor(id);
        return;
    }
    backends().resources->register_texture_descriptor(id, state<TextureState>(t),
        v ? &state<TextureViewState>(v) : nullptr);
}
void nvnTexturePoolRegisterImage(const NVNtexturePool* p, int id, const NVNtexture* t, const NVNtextureView* v) {
    nvnTexturePoolRegisterTexture(p, id, t, v);
}
void nvnSamplerPoolRegisterSampler(const NVNsamplerPool* p, int id, const NVNsampler* s) {
    if (!p || !s || state<SamplerState>(s).tag != sampler_tag) { invalid("sampler descriptor"); return; }
    if (!valid_slot(state<DescriptorPoolState>(p), id, true)) return;
    backends().resources->register_sampler_descriptor(id, state<SamplerState>(s).parameters);
}
void nvnSamplerPoolRegisterSamplerBuilder(const NVNsamplerPool* p, int id, const NVNsamplerBuilder* b) {
    if (!p || !b || !valid(state<SamplerBuilderState>(b).parameters)) { invalid("sampler builder descriptor"); return; }
    if (!valid_slot(state<DescriptorPoolState>(p), id, true)) return;
    backends().resources->register_sampler_descriptor(id, state<SamplerBuilderState>(b).parameters);
}
void nvnTexturePoolFinalize(NVNtexturePool* pool) {
    if (!pool) return;
    auto& descriptor = state<DescriptorPoolState>(pool);
    if (descriptor.pool && backends().resources)
        backends().resources->release_texture_pool(descriptor);
    descriptor = {};
}
void nvnTexturePoolSetDebugLabel(NVNtexturePool*, const char*) {}
const NVNmemoryPool* nvnTexturePoolGetMemoryPool(const NVNtexturePool* pool) {
    return reinterpret_cast<const NVNmemoryPool*>(state<DescriptorPoolState>(pool).pool);
}
ptrdiff_t nvnTexturePoolGetMemoryOffset(const NVNtexturePool* pool) { return state<DescriptorPoolState>(pool).offset; }
int nvnTexturePoolGetSize(const NVNtexturePool* pool) { return state<DescriptorPoolState>(pool).count; }
void nvnSamplerPoolFinalize(NVNsamplerPool* pool) {
    if (!pool) return;
    auto& descriptor = state<DescriptorPoolState>(pool);
    if (descriptor.pool && backends().resources)
        backends().resources->release_sampler_pool(descriptor);
    descriptor = {};
}
void nvnSamplerPoolSetDebugLabel(NVNsamplerPool*, const char*) {}
const NVNmemoryPool* nvnSamplerPoolGetMemoryPool(const NVNsamplerPool* pool) {
    return reinterpret_cast<const NVNmemoryPool*>(state<DescriptorPoolState>(pool).pool);
}
ptrdiff_t nvnSamplerPoolGetMemoryOffset(const NVNsamplerPool* pool) { return state<DescriptorPoolState>(pool).offset; }
int nvnSamplerPoolGetSize(const NVNsamplerPool* pool) { return state<DescriptorPoolState>(pool).count; }
} // namespace

void register_resource_entries() {
    register_device_entries();
    register_memory_entries();
    register_texture_entries();
    install<Entry::nvnTexturePoolInitialize>(&nvnTexturePoolInitialize);
    install<Entry::nvnTexturePoolSetDebugLabel>(&nvnTexturePoolSetDebugLabel);
    install<Entry::nvnTexturePoolFinalize>(&nvnTexturePoolFinalize);
    install<Entry::nvnTexturePoolRegisterTexture>(&nvnTexturePoolRegisterTexture);
    install<Entry::nvnTexturePoolRegisterImage>(&nvnTexturePoolRegisterImage);
    install<Entry::nvnTexturePoolGetMemoryPool>(&nvnTexturePoolGetMemoryPool);
    install<Entry::nvnTexturePoolGetMemoryOffset>(&nvnTexturePoolGetMemoryOffset);
    install<Entry::nvnTexturePoolGetSize>(&nvnTexturePoolGetSize);
    install<Entry::nvnSamplerPoolInitialize>(&nvnSamplerPoolInitialize);
    install<Entry::nvnSamplerPoolSetDebugLabel>(&nvnSamplerPoolSetDebugLabel);
    install<Entry::nvnSamplerPoolFinalize>(&nvnSamplerPoolFinalize);
    install<Entry::nvnSamplerPoolRegisterSampler>(&nvnSamplerPoolRegisterSampler);
    install<Entry::nvnSamplerPoolRegisterSamplerBuilder>(&nvnSamplerPoolRegisterSamplerBuilder);
    install<Entry::nvnSamplerPoolGetMemoryPool>(&nvnSamplerPoolGetMemoryPool);
    install<Entry::nvnSamplerPoolGetMemoryOffset>(&nvnSamplerPoolGetMemoryOffset);
    install<Entry::nvnSamplerPoolGetSize>(&nvnSamplerPoolGetSize);
    install<Entry::nvnSamplerBuilderGetDevice>(&nvnSamplerBuilderGetDevice);
    install<Entry::nvnSamplerBuilderSetDevice>(&nvnSamplerBuilderSetDevice);
    install<Entry::nvnSamplerBuilderSetDefaults>(&nvnSamplerBuilderSetDefaults);
    install<Entry::nvnSamplerBuilderSetMinMagFilter>(&nvnSamplerBuilderSetMinMagFilter);
    install<Entry::nvnSamplerBuilderSetWrapMode>(&nvnSamplerBuilderSetWrapMode);
    install<Entry::nvnSamplerBuilderSetLodClamp>(&nvnSamplerBuilderSetLodClamp);
    install<Entry::nvnSamplerBuilderSetLodBias>(&nvnSamplerBuilderSetLodBias);
    install<Entry::nvnSamplerBuilderSetCompare>(&nvnSamplerBuilderSetCompare);
    install<Entry::nvnSamplerBuilderSetBorderColor>(&nvnSamplerBuilderSetBorderColor);
    install<Entry::nvnSamplerBuilderSetBorderColori>(&nvnSamplerBuilderSetBorderColori);
    install<Entry::nvnSamplerBuilderSetBorderColorui>(&nvnSamplerBuilderSetBorderColorui);
    install<Entry::nvnSamplerBuilderSetMaxAnisotropy>(&nvnSamplerBuilderSetMaxAnisotropy);
    install<Entry::nvnSamplerBuilderSetReductionFilter>(&nvnSamplerBuilderSetReductionFilter);
    install<Entry::nvnSamplerBuilderSetLodSnap>(&nvnSamplerBuilderSetLodSnap);
    install<Entry::nvnSamplerBuilderGetMinMagFilter>(&nvnSamplerBuilderGetMinMagFilter);
    install<Entry::nvnSamplerBuilderGetWrapMode>(&nvnSamplerBuilderGetWrapMode);
    install<Entry::nvnSamplerBuilderGetLodClamp>(&nvnSamplerBuilderGetLodClamp);
    install<Entry::nvnSamplerBuilderGetLodBias>(&nvnSamplerBuilderGetLodBias);
    install<Entry::nvnSamplerBuilderGetCompare>(&nvnSamplerBuilderGetCompare);
    install<Entry::nvnSamplerBuilderGetBorderColor>(&nvnSamplerBuilderGetBorderColor);
    install<Entry::nvnSamplerBuilderGetBorderColori>(&nvnSamplerBuilderGetBorderColori);
    install<Entry::nvnSamplerBuilderGetBorderColorui>(&nvnSamplerBuilderGetBorderColorui);
    install<Entry::nvnSamplerBuilderGetMaxAnisotropy>(&nvnSamplerBuilderGetMaxAnisotropy);
    install<Entry::nvnSamplerBuilderGetReductionFilter>(&nvnSamplerBuilderGetReductionFilter);
    install<Entry::nvnSamplerBuilderGetLodSnap>(&nvnSamplerBuilderGetLodSnap);
    install<Entry::nvnSamplerInitialize>(&nvnSamplerInitialize);
    install<Entry::nvnSamplerFinalize>(&nvnSamplerFinalize);
    install<Entry::nvnSamplerSetDebugLabel>(&nvnSamplerSetDebugLabel);
    install<Entry::nvnSamplerGetMinMagFilter>(&nvnSamplerGetMinMagFilter);
    install<Entry::nvnSamplerGetWrapMode>(&nvnSamplerGetWrapMode);
    install<Entry::nvnSamplerGetLodClamp>(&nvnSamplerGetLodClamp);
    install<Entry::nvnSamplerGetLodBias>(&nvnSamplerGetLodBias);
    install<Entry::nvnSamplerGetCompare>(&nvnSamplerGetCompare);
    install<Entry::nvnSamplerGetBorderColor>(&nvnSamplerGetBorderColor);
    install<Entry::nvnSamplerGetBorderColori>(&nvnSamplerGetBorderColori);
    install<Entry::nvnSamplerGetBorderColorui>(&nvnSamplerGetBorderColorui);
    install<Entry::nvnSamplerGetMaxAnisotropy>(&nvnSamplerGetMaxAnisotropy);
    install<Entry::nvnSamplerGetReductionFilter>(&nvnSamplerGetReductionFilter);
    install<Entry::nvnSamplerCompare>(&nvnSamplerCompare);
    install<Entry::nvnSamplerGetDebugID>(&nvnSamplerGetDebugID);
}

} // namespace armigo::nvn
