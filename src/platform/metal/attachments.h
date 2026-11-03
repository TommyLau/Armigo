// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Render attachment formats map guest encodings to Metal formats. Render-pass
// load and store actions are assembled with the clear-folding contract.

#pragma once

#import <Metal/Metal.h>

#include <armigo/graphics/render_target.h>

#include <array>
#include <variant>

namespace armigo::platform::metal {

// Native texture bound to one attachment slot. A nil texture names a hole and
// is admitted only where the plan also has a hole. level and slice select the
// mip level and array layer of the resident storage.
struct NativeAttachment {
    id<MTLTexture> texture = nil;
    std::uint32_t level = 0;
    std::uint32_t slice = 0;
};

struct NativeRenderTargets {
    std::array<NativeAttachment, 8> colors;
    NativeAttachment depth;
    NativeAttachment stencil;
};

// Attachment-less passes use the surface extent within the common Metal 4
// render-target limit. Attached passes keep their planned subresource extent.
[[nodiscard]] graphics::RenderArea render_pass_area(
    const graphics::RenderTargetPlan& plan) noexcept;

// Clear values applied to every attachment whose plan names
// AttachmentLoad::Clear. Color components are linear values (or raw integer
// values as doubles) as Metal's clear color expects.
struct RenderPassClearValues {
    std::array<std::array<double, 4>, 8> color{};
    double depth = 0.0;
    std::uint32_t stencil = 0;
};

// Maps a planned guest attachment format and aspect onto the Metal renderable
// pixel format. Formats Metal cannot render into (and aspect requests the
// format does not carry) return a structured unsupported result; sampling
// mappings never substitute for attachment mappings.
[[nodiscard]] std::variant<MTLPixelFormat, graphics::CommandResult>
metal_render_format(graphics::TextureFormat format,
                    graphics::TextureAspect aspect) noexcept;

// Converts raw register clear words into Metal's double clear-color
// components for the attachment format: normalized encodings unpack to
// linear floats, float encodings bit-convert, and integer encodings pass
// their signed or unsigned values through as exact doubles.
[[nodiscard]] std::variant<std::array<double, 4>, graphics::CommandResult>
metal_clear_color(const std::array<std::uint32_t, 4>& bits,
                  graphics::TextureFormat format) noexcept;

// Validates the plan against the bound native targets (presence, pixel
// format, sample count, extent and aspect pairing) and returns a retained
// Metal 4 render pass descriptor encoding the plan's load/store state. A
// cleared attachment requires clear values; the fold decision itself belongs
// to the ClearPlan and to the pass-identity caller. The descriptor references
// the bound textures; the caller retains them for the pass lifetime.
[[nodiscard]] graphics::RendererResult<MTL4RenderPassDescriptor *>
make_render_pass(const graphics::RenderTargetPlan& plan,
                 const NativeRenderTargets& targets,
                 const RenderPassClearValues* clear_values) noexcept;

}  // namespace armigo::platform::metal
