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

#include <armigo/graphics/engines.h>
#include <armigo/graphics/texture_cache.h>
#include <optional>

namespace armigo::graphics {

// Signed origin and nonnegative extent. Intersection must use checked wide
// arithmetic; an empty rectangle is valid no-coverage, never unsigned wrap.
struct RenderArea {
    std::int64_t x = 0, y = 0;
    std::uint64_t width = 0, height = 0;
    bool operator==(const RenderArea&) const = default;
};
enum class AttachmentLoad : std::uint8_t { Load, Clear };
enum class AttachmentStore : std::uint8_t { Store };

// Format is the shared decoded texture format, never an MTLPixelFormat. Decode
// render-target register encodings explicitly; do not reinterpret them as TICs.
struct AttachmentPlan {
    std::uint8_t physical_target = 0;
    TextureDescriptor texture;
    TextureAspect aspect = TextureAspect::Color;
    // Guest RGBX encodings may use four-component host storage while the
    // padding channel still reads as one for destination-alpha blending.
    bool has_alpha = true;
    std::uint32_t level = 0, layer = 0, layer_count = 1;
    AttachmentLoad load = AttachmentLoad::Load;
    AttachmentStore store = AttachmentStore::Store;
    bool operator==(const AttachmentPlan&) const = default;
};
struct RenderTargetPlan {
    // Indexed by logical shader output; holes remain holes. physical_target
    // records the RtControl permutation. A depth-only pass has no colors.
    std::array<std::optional<AttachmentPlan>, 8> colors;
    std::optional<AttachmentPlan> depth;
    std::optional<AttachmentPlan> stencil;
    RenderArea area;
    std::uint32_t samples = 1;
    bool operator==(const RenderTargetPlan&) const = default;
};
struct ClearPlan {
    // A selected null depth attachment completes without accessing storage.
    bool no_op = false;
    // Clear names a PHYSICAL target, independent of draw-time permutation.
    // target carries the colour attachment for colour clears (aspect Color)
    // and the depth/stencil attachment otherwise; a combined clear keeps the
    // depth/stencil descriptor in depth_stencil. target.layer is the absolute
    // first cleared slice (colour target base layer plus the CLEAR_SURFACE
    // argument's layer field) and target.layer_count the cleared layer count,
    // read from the target's own array-mode register exactly as the guest
    // SDK's clear macros read it (MacroHLE ColorLayerCountOffset 0x206 words,
    // ZetaLayerCountOffset 0x48c words). target.texture describes the full
    // array storage, so its layers bound the cleared range.
    AttachmentPlan target;
    // Depth/stencil attachment of a combined colour and depth/stencil clear;
    // empty unless both aspects are cleared in one call.
    std::optional<AttachmentPlan> depth_stencil;
    RenderArea region;
    // Set only for aspects with bound storage and a nonempty region; a part
    // without work is dropped instead of failing the whole clear.
    std::uint8_t color_mask = 0;
    // Raw register values preserve f32/u32/s32 clear representation.
    std::array<std::uint32_t, 4> color_bits{};
    bool clear_depth = false, clear_stencil = false;
    double depth_value = 0;
    std::uint32_t stencil_value = 0, stencil_write_mask = 0;
    // Only true for an admitted unconditional, full-region/full-aspect clear
    // at pass start. Otherwise encode an operation preserving untouched data.
    // fold_into_load folds into the enclosing render pass load action and
    // therefore requires single-layer storage; a full-range clear of an
    // arrayed attachment instead sets fold_layers_into_load, which folds
    // into a standalone layered pass load action covering every cleared
    // layer.
    bool fold_into_load = false;
    bool fold_layers_into_load = false;
};

// Resident pass identity excludes pipeline, viewport, blend constants and
// uniforms. It includes exact storage/subresource/aspect and render area.
// Different identities, transfer hazards and ordering boundaries end the pass.
struct RenderPassKey {
    std::array<std::optional<TextureViewKey>, 8> colors;
    std::optional<TextureViewKey> depth, stencil;
    RenderArea area;
    std::uint32_t samples = 1;
    bool operator==(const RenderPassKey&) const = default;
};

// L6: planners read effective Maxwell registers from RenderStop. Validate eight
// slots, sparse/permuted outputs, sample agreement, layers and format/aspects.
// Attachment presence is independent of test/write enables. Depth layer flags
// absent from DepthDescription must be read from the immutable register image.
// Conditional render, unknown clear bits and unimplemented formats stop; a
// clear's layer field selects a range within the target's own array layer
// count (the guest SDK macros clear the programmed array). A cleared range
// outside the storage, unknown high argument bits, differing layer ranges in
// a combined colour and depth/stencil clear, and multi-slice 3D targets stop;
// no silent full-target clear or guessed DontCare load/store is permitted.
// Pitch widths retain the encoded row stride; block-linear targets retain GOB
// height/depth geometry and programmed array spacing. One-slice 3D color
// targets remain Texture3D so unused interleaved Z planes are preserved.
// Multi-slice 3D attachments and pitch-linear depth targets remain unsupported.
[[nodiscard]] RendererResult<RenderTargetPlan> plan_render_targets(const RenderStop& stop);
[[nodiscard]] RendererResult<ClearPlan> plan_render_clear(const RenderStop& stop);

} // namespace armigo::graphics
