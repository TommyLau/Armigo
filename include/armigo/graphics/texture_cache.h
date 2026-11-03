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
#include <armigo/graphics/texture.h>
#include <optional>
#include <tuple>

namespace armigo::graphics {

// Existing TIC decode and texture_view_layout remain the layout authority.
// Keys use backing-relative storage, never descriptor bytes or virtual address.
// descriptor is a validated view-relative shape; address/raw words/swizzle and
// coordinate mode are not storage identity. A sub-chain retains layer_stride.
struct TextureStorageKey {
    BackingRange bytes;
    TextureDescriptor descriptor;
    bool operator==(const TextureStorageKey& other) const {
        const auto shape = [](const TextureDescriptor& d) {
            return std::tie(d.backing_extent, d.row_bytes, d.storage_height,
                d.layer_stride, d.width, d.height, d.depth, d.levels, d.layers,
                d.base_level, d.samples, d.block_height_log2, d.block_depth_log2,
                d.format, d.layout, d.type, d.srgb);
        };
        return bytes == other.bytes && shape(descriptor) == shape(other.descriptor);
    }
};
enum class TextureAspect : std::uint8_t { Color = 1, Depth = 2, Stencil = 4, DepthStencil = 6 };
struct TextureViewKey {
    TextureStorageKey storage;
    TextureAspect aspect = TextureAspect::Color;
    std::uint32_t first_level = 0, level_count = 1;
    std::uint32_t first_layer = 0, layer_count = 1;
    // View target for a same-dimensions arrayness view admitted over this
    // storage (1D<->1D array, 2D<->2D array, cube<->cube array); nullopt
    // views keep the storage descriptor's own target.
    std::optional<TextureType> view_target = std::nullopt;
    std::array<TextureSwizzle, 4> swizzle{
        TextureSwizzle::Red, TextureSwizzle::Green,
        TextureSwizzle::Blue, TextureSwizzle::Alpha};
    bool operator==(const TextureViewKey&) const = default;
};
// Texture searches admit depth/colour storage
// aliases for shader sampler lookups (ForSampler) and the 2D engine's depth
// aliases for copy searches (DepthAlias). Every other consumer, including
// render targets and writes, keeps exact storage compatibility.
enum class TextureAliasScope : std::uint8_t { None = 0, Sampler, TwodEngine };
struct TextureRequest {
    RetainedRange range;
    TextureViewKey key;
    ResourceAccess access = ResourceAccess::Read;
    TextureAliasScope alias = TextureAliasScope::None;
};
struct ResidentTexture {
    TextureViewKey key;
    ResourceVersion uploaded_version;
    RetainedResource storage;
    // Producer dependencies must be encoded/waited, not merely retained.
    std::vector<CompletionToken> dependencies;
    // Ordered GPU transfers the consuming packet encodes before its work, such
    // as copies that bring sub-resources rendered through separate textures
    // into this storage. Platform-defined; each is encoded once.
    std::vector<RetainedResource> transfers;
};

// L4: exact compatible aliases reuse residency; incompatible aliases require
// ordered conversion/resolve or Unsupported. Live descriptor/fact admission is
// mandatory even on hits. acquire validates range == key.storage.bytes, every
// mip/layer, aspect, format and device capability. Upload unchanged CPU content
// once; never upload over GPU-newer bytes. Partial resolves preserve other
// components/aspects/padding. Eviction must not drop the sole dirty copy.
class TextureCache {
public:
    virtual ~TextureCache() = default;
    virtual ResidencyDomain domain() const noexcept = 0;
    virtual RendererResult<ResidentTexture> acquire(const TextureRequest& request) = 0;
    // Registry version changes invalidate ALL overlaps, including other views.
    virtual RendererStatus invalidate(BackingRange bytes) = 0;
    // Explicit requests only (or mandatory synchronous boundary fallback).
    // Resolve updates the shared registry only after successful conversion and
    // checked guest writeback; failure retains dirty state and resident storage.
    virtual RendererStatus resolve(const TextureViewKey& key, ResolveReason reason) = 0;
    virtual RendererStatus trim(std::uint64_t budget_bytes) = 0;
};

} // namespace armigo::graphics
