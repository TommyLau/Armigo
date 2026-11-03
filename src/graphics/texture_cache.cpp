// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Compatible texture storage admits the defined depth and colour alias pairs.

#include "texture_residency.h"
#include <algorithm>
#include <cstdio>
#include <bit>
#include <limits>
#include <optional>

namespace armigo::graphics {
namespace {
RendererError invalid(const char* message) {
    return RendererError({CommandStatus::Unsupported, message});
}
bool srgb_format(TextureFormat format) {
    switch (format) {
        case TextureFormat::R8G8B8A8Srgb:
        case TextureFormat::B8G8R8A8Srgb:
        case TextureFormat::Bc1RgbaSrgb:
        case TextureFormat::Bc2Srgb:
        case TextureFormat::Bc3Srgb:
        case TextureFormat::Bc7Srgb:
        case TextureFormat::Etc2RgbSrgb:
        case TextureFormat::Etc2RgbPtaSrgb:
        case TextureFormat::Etc2RgbaSrgb:
        case TextureFormat::Astc4x4Srgb:
        case TextureFormat::Astc5x4Srgb:
        case TextureFormat::Astc5x5Srgb:
        case TextureFormat::Astc6x5Srgb:
        case TextureFormat::Astc6x6Srgb:
        case TextureFormat::Astc8x5Srgb:
        case TextureFormat::Astc8x6Srgb:
        case TextureFormat::Astc8x8Srgb:
        case TextureFormat::Astc10x5Srgb:
        case TextureFormat::Astc10x6Srgb:
        case TextureFormat::Astc10x8Srgb:
        case TextureFormat::Astc10x10Srgb:
        case TextureFormat::Astc12x10Srgb:
        case TextureFormat::Astc12x12Srgb: return true;
        default: return false;
    }
}
TextureFormat linear_format(TextureFormat format) {
    switch (format) {
        case TextureFormat::R8G8B8A8Srgb: return TextureFormat::R8G8B8A8Unorm;
        case TextureFormat::B8G8R8A8Srgb: return TextureFormat::B8G8R8A8Unorm;
        case TextureFormat::Bc1RgbaSrgb: return TextureFormat::Bc1RgbaUnorm;
        case TextureFormat::Bc2Srgb: return TextureFormat::Bc2Unorm;
        case TextureFormat::Bc3Srgb: return TextureFormat::Bc3Unorm;
        case TextureFormat::Bc7Srgb: return TextureFormat::Bc7Unorm;
        case TextureFormat::Etc2RgbSrgb: return TextureFormat::Etc2RgbUnorm;
        case TextureFormat::Etc2RgbPtaSrgb: return TextureFormat::Etc2RgbPtaUnorm;
        case TextureFormat::Etc2RgbaSrgb: return TextureFormat::Etc2RgbaUnorm;
        case TextureFormat::Astc4x4Srgb: return TextureFormat::Astc4x4Unorm;
        case TextureFormat::Astc5x4Srgb: return TextureFormat::Astc5x4Unorm;
        case TextureFormat::Astc5x5Srgb: return TextureFormat::Astc5x5Unorm;
        case TextureFormat::Astc6x5Srgb: return TextureFormat::Astc6x5Unorm;
        case TextureFormat::Astc6x6Srgb: return TextureFormat::Astc6x6Unorm;
        case TextureFormat::Astc8x5Srgb: return TextureFormat::Astc8x5Unorm;
        case TextureFormat::Astc8x6Srgb: return TextureFormat::Astc8x6Unorm;
        case TextureFormat::Astc8x8Srgb: return TextureFormat::Astc8x8Unorm;
        case TextureFormat::Astc10x5Srgb: return TextureFormat::Astc10x5Unorm;
        case TextureFormat::Astc10x6Srgb: return TextureFormat::Astc10x6Unorm;
        case TextureFormat::Astc10x8Srgb: return TextureFormat::Astc10x8Unorm;
        case TextureFormat::Astc10x10Srgb: return TextureFormat::Astc10x10Unorm;
        case TextureFormat::Astc12x10Srgb: return TextureFormat::Astc12x10Unorm;
        case TextureFormat::Astc12x12Srgb: return TextureFormat::Astc12x12Unorm;
        default: return format;
    }
}
} // namespace

// Depth and colour sampling aliases share these pairwise storage formats.
std::optional<TextureFormat> depth_alias_storage(TextureFormat colour) noexcept {
    switch (colour) {
        case TextureFormat::R16Unorm: return TextureFormat::D16Unorm;
        case TextureFormat::R32Float: return TextureFormat::D32Float;
        default: return std::nullopt;
    }
}

std::optional<TextureFormat> colour_alias_storage(TextureFormat depth) noexcept {
    switch (depth) {
        case TextureFormat::D16Unorm: return TextureFormat::R16Unorm;
        case TextureFormat::D32Float: return TextureFormat::R32Float;
        default: return std::nullopt;
    }
}

bool compatible_texture_storage(const TextureStorageKey& a, const TextureStorageKey& b,
                                TextureAliasScope scope) {
    // Every admitted pairing, aliases included, names the same guest bytes;
    // reject other storage before copying and normalizing the keys.
    if (!(a.bytes == b.bytes)) return false;
    auto left = a, right = b;
    left.descriptor.format = linear_format(left.descriptor.format);
    right.descriptor.format = linear_format(right.descriptor.format);
    left.descriptor.srgb = right.descriptor.srgb = false;
    if (left == right) return true;
    if (scope == TextureAliasScope::None) return false;
    // D32F and R32F (and D16Unorm/R16Unorm) share one
    // internal representation, and the colour format is how NVN describes a
    // depth texture's sampling view; forSampler or depthAlias resolves such a
    // lookup onto the existing depth storage. Only depth storage serving a
    // colour-coded request is admitted here: handing a colour-format host
    // texture to a depth consumer has no native representation. The alias
    // changes the pixel format alone, so every other key field must already
    // match: a request whose tiling, block height or extent differs names a
    // different guest layout, and serving it from the depth allocation would
    // convert a source span sized by one descriptor through another's layout.
    const auto matches_alias = [&left](const TextureStorageKey& request,
                                       TextureFormat depth) {
        auto depth_keyed = request;
        depth_keyed.descriptor.format = depth;
        return left == depth_keyed;
    };
    if (const auto depth = depth_alias_storage(right.descriptor.format);
        depth && matches_alias(right, *depth)) return true;
    if (scope != TextureAliasScope::TwodEngine) return false;
    // The 2D engine describes depth-stencil surfaces with equivalent colour
    // formats. Armigo's format table has no S8Uint or B8G8R8A8Unorm entries,
    // so only the D32FS8 pair
    // of that branch exists here.
    return right.descriptor.format == TextureFormat::R32G32Float &&
           matches_alias(right, TextureFormat::D32FloatS8Uint);
}

RendererResult<TextureLayoutDescription> validate_texture_view(const TextureViewKey& key) {
    const auto& d = key.storage.descriptor;
    const auto& bytes = key.storage.bytes;
    if (!bytes.backing.owner || !bytes.backing.allocation || !bytes.backing.generation ||
        !bytes.size || bytes.offset > std::numeric_limits<std::uint64_t>::max() - bytes.size ||
        bytes.size != d.backing_extent || bytes.size > max_texture_layout_size)
        return invalid("texture backing identity or extent is invalid");
    if (!d.width || !d.height || !d.depth || !d.levels || !d.layers || d.samples != 1 ||
        d.width > 16384 || d.height > 16384 || d.depth > 2048 || d.layers > 2048 ||
        d.block_height_log2 > 5 || d.block_depth_log2 > 5 || d.base_level >= 16 ||
        d.levels > static_cast<unsigned>(std::bit_width(std::max({d.width, d.height, d.depth}))))
        return invalid("texture dimensions, levels or samples are unsupported");
    if (d.layout != TextureLayout::Pitch && d.layout != TextureLayout::BlockLinear)
        return invalid("texture layout is invalid");
    bool shape = false;
    switch (d.type) {
        case TextureType::Texture1D:
            shape = d.height == 1 && d.depth == 1 && d.layers == 1; break;
        case TextureType::Texture1DArray:
            shape = d.height == 1 && d.depth == 1; break;
        case TextureType::Texture2D:
            shape = d.depth == 1 && d.layers == 1; break;
        case TextureType::Texture2DRect:
            shape = d.depth == 1 && d.layers == 1 && d.levels == 1; break;
        case TextureType::Texture2DArray: shape = d.depth == 1; break;
        case TextureType::Texture3D:
            shape = d.layers == 1 && d.width <= 2048 && d.height <= 2048; break;
        case TextureType::Cubemap:
            shape = d.depth == 1 && d.width == d.height && d.layers == 6; break;
        case TextureType::CubemapArray:
            shape = d.depth == 1 && d.width == d.height && d.layers % 6 == 0; break;
    }
    // Any block-linear target may carry the descriptor's GOB blocks in Z;
    // the layout walk sizes them.
    if (!shape) {
        char detail[160];
        std::snprintf(detail, sizeof detail,
                      "texture target geometry is inconsistent (type %u %ux%ux%u layers %u levels %u "
                      "block depth log2 %u)", unsigned(d.type), d.width, d.height, d.depth, d.layers,
                      d.levels, unsigned(d.block_depth_log2));
        return RendererError({CommandStatus::Unsupported, std::string(detail)});
    }
    if (!key.level_count || key.first_level >= d.levels ||
        key.level_count > d.levels - key.first_level || !key.layer_count ||
        key.first_layer >= d.layers || key.layer_count > d.layers - key.first_layer)
        return invalid("texture view levels or layers exceed storage");
    if ((d.type == TextureType::Cubemap || d.type == TextureType::CubemapArray) &&
        (key.first_layer % 6 || key.layer_count % 6))
        return invalid("cube texture views require complete six-face groups");
    const auto info = texture_format_info(d.format);
    if (!info.bytes_per_block) return invalid("texture format is invalid");
    if (d.srgb != srgb_format(d.format))
        return invalid("texture sRGB flag disagrees with its decoded format");
    if ((!info.depth && key.aspect != TextureAspect::Color) ||
        (info.depth && key.aspect != TextureAspect::Depth &&
         !(info.stencil && (key.aspect == TextureAspect::Stencil ||
                            key.aspect == TextureAspect::DepthStencil))))
        return invalid("texture aspect is absent from storage");
    for (auto swizzle : key.swizzle)
        if (static_cast<unsigned>(swizzle) > static_cast<unsigned>(TextureSwizzle::OneFloat))
            return invalid("texture view swizzle is invalid");
    for (auto swizzle : d.swizzle)
        if (static_cast<unsigned>(swizzle) > static_cast<unsigned>(TextureSwizzle::OneFloat))
            return invalid("texture descriptor swizzle is invalid");
    auto calculated = texture_view_layout(d);
    if (auto* error = std::get_if<TextureLayoutError>(&calculated))
        return invalid(error->message());
    auto layout = std::get<TextureLayoutDescription>(std::move(calculated));
    const auto& base = layout.mips.front();
    const auto row = (base.linear_row_bytes + 63) & ~std::size_t{63};
    const auto alignment = std::size_t{8} << base.gob_block_height_log2;
    const auto height = (base.block_rows + alignment - 1) & ~(alignment - 1);
    if (layout.total_size != bytes.size || d.layer_stride != layout.layer_stride ||
        (d.layout == TextureLayout::Pitch ?
            d.storage_height != base.block_rows :
            d.row_bytes != row || d.storage_height != height))
        return invalid("texture descriptor storage geometry is inconsistent");
    return layout;
}
} // namespace armigo::graphics
