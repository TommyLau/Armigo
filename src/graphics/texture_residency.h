// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture residency admits the defined pairwise depth and colour storage aliases.

#pragma once
#include <armigo/graphics/texture_cache.h>
#include <optional>

namespace armigo::graphics {
// Checked portable admission shared by residency and native view creation.
[[nodiscard]] RendererResult<TextureLayoutDescription> validate_texture_view(
    const TextureViewKey& key);

// D16Unorm/R16Unorm and D32Float/R32Float alias pairwise
// because the pairs share one storage representation. NVN describes a depth
// texture's sampling view with a colour-coded TIC while the address holds the
// depth target the game rendered, so a sampler lookup must resolve onto the
// existing depth storage.
[[nodiscard]] std::optional<TextureFormat> depth_alias_storage(
    TextureFormat colour) noexcept;
[[nodiscard]] std::optional<TextureFormat> colour_alias_storage(
    TextureFormat depth) noexcept;

// a is the existing storage and b is the request. The depth alias admits a
// colour-coded request onto depth storage only; the reverse would hand a
// colour-format host texture to a depth consumer. The alias changes the
// pixel format alone: tiling, block height and extent must already match,
// so a depth-aliased allocation keeps exactly its request descriptor's
// layout and its uploads read a source span sized by the same descriptor.
[[nodiscard]] bool compatible_texture_storage(
    const TextureStorageKey& a, const TextureStorageKey& b,
    TextureAliasScope scope = TextureAliasScope::None);
} // namespace armigo::graphics
