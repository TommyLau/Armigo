// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Guest render-target formats map to Metal attachment formats. Passes without
// attachments remain valid and use the bounded surface extent.

#include "attachments.h"

#include <armigo/graphics/texture_format.h>

#include <algorithm>
#include <string_view>

namespace armigo::platform::metal {
namespace {

using graphics::CommandResult;
using graphics::CommandStatus;
using graphics::TextureFormat;

}  // namespace

std::variant<MTLPixelFormat, graphics::CommandResult>
metal_render_format(TextureFormat format, graphics::TextureAspect aspect) noexcept {
    const auto info = graphics::texture_format_info(format);
    switch (aspect) {
        case graphics::TextureAspect::Color:
            if (info.depth || info.stencil)
                return CommandResult{CommandStatus::Unsupported,
                                     "color attachment maps no depth format"};
            break;
        case graphics::TextureAspect::Depth:
        case graphics::TextureAspect::DepthStencil:
            if (!info.depth)
                return CommandResult{CommandStatus::Unsupported,
                                     "depth attachment maps a format without depth"};
            break;
        case graphics::TextureAspect::Stencil:
            if (!info.stencil)
                return CommandResult{CommandStatus::Unsupported,
                                     "stencil attachment maps a format without stencil"};
            break;
    }
    switch (format) {
        case TextureFormat::R8Unorm: return MTLPixelFormatR8Unorm;
        case TextureFormat::R8Snorm: return MTLPixelFormatR8Snorm;
        case TextureFormat::R8Uint: return MTLPixelFormatR8Uint;
        case TextureFormat::R8Sint: return MTLPixelFormatR8Sint;
        case TextureFormat::R16Unorm: return MTLPixelFormatR16Unorm;
        case TextureFormat::R16Snorm: return MTLPixelFormatR16Snorm;
        case TextureFormat::R16Uint: return MTLPixelFormatR16Uint;
        case TextureFormat::R16Sint: return MTLPixelFormatR16Sint;
        case TextureFormat::R16Float: return MTLPixelFormatR16Float;
        case TextureFormat::R32Uint: return MTLPixelFormatR32Uint;
        case TextureFormat::R32Sint: return MTLPixelFormatR32Sint;
        case TextureFormat::R32Float: return MTLPixelFormatR32Float;
        case TextureFormat::R8G8Unorm: return MTLPixelFormatRG8Unorm;
        case TextureFormat::R8G8Snorm: return MTLPixelFormatRG8Snorm;
        case TextureFormat::R8G8Uint: return MTLPixelFormatRG8Uint;
        case TextureFormat::R8G8Sint: return MTLPixelFormatRG8Sint;
        case TextureFormat::R16G16Unorm: return MTLPixelFormatRG16Unorm;
        case TextureFormat::R16G16Snorm: return MTLPixelFormatRG16Snorm;
        case TextureFormat::R16G16Uint: return MTLPixelFormatRG16Uint;
        case TextureFormat::R16G16Sint: return MTLPixelFormatRG16Sint;
        case TextureFormat::R16G16Float: return MTLPixelFormatRG16Float;
        case TextureFormat::R32G32Uint: return MTLPixelFormatRG32Uint;
        case TextureFormat::R32G32Sint: return MTLPixelFormatRG32Sint;
        case TextureFormat::R32G32Float: return MTLPixelFormatRG32Float;
        case TextureFormat::R8G8B8A8Unorm: return MTLPixelFormatRGBA8Unorm;
        case TextureFormat::R8G8B8A8Snorm: return MTLPixelFormatRGBA8Snorm;
        case TextureFormat::R8G8B8A8Uint: return MTLPixelFormatRGBA8Uint;
        case TextureFormat::R8G8B8A8Sint: return MTLPixelFormatRGBA8Sint;
        case TextureFormat::R8G8B8A8Srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
        case TextureFormat::B8G8R8A8Unorm: return MTLPixelFormatBGRA8Unorm;
        case TextureFormat::B8G8R8A8Srgb: return MTLPixelFormatBGRA8Unorm_sRGB;
        case TextureFormat::R16G16B16A16Unorm: return MTLPixelFormatRGBA16Unorm;
        case TextureFormat::R16G16B16A16Snorm: return MTLPixelFormatRGBA16Snorm;
        case TextureFormat::R16G16B16A16Uint: return MTLPixelFormatRGBA16Uint;
        case TextureFormat::R16G16B16A16Sint: return MTLPixelFormatRGBA16Sint;
        case TextureFormat::R16G16B16A16Float: return MTLPixelFormatRGBA16Float;
        case TextureFormat::R32G32B32A32Uint: return MTLPixelFormatRGBA32Uint;
        case TextureFormat::R32G32B32A32Sint: return MTLPixelFormatRGBA32Sint;
        case TextureFormat::R32G32B32A32Float: return MTLPixelFormatRGBA32Float;
        case TextureFormat::R10G10B10A2Unorm: return MTLPixelFormatRGB10A2Unorm;
        case TextureFormat::R10G10B10A2Uint: return MTLPixelFormatRGB10A2Uint;
        case TextureFormat::R11G11B10Float: return MTLPixelFormatRG11B10Float;
        // The depth aspect of a packed format binds the packed Metal format;
        // the planner never enables an aspect the guest format does not name.
        case TextureFormat::D16Unorm: return MTLPixelFormatDepth16Unorm;
        case TextureFormat::D32Float: return MTLPixelFormatDepth32Float;
        case TextureFormat::D32FloatS8Uint:
            return MTLPixelFormatDepth32Float_Stencil8;
        // D24 guest storage is converted through separate native float-depth
        // and stencil planes by the persistent texture cache.
        case TextureFormat::D24UnormS8Uint:
        case TextureFormat::S8UintD24Unorm:
        case TextureFormat::X8UintD24Unorm:
            return MTLPixelFormatDepth32Float_Stencil8;
        default:
            return CommandResult{CommandStatus::Unsupported,
                                 "attachment format has no renderable Metal mapping"};
    }
}

std::variant<std::array<double, 4>, graphics::CommandResult>
metal_clear_color(const std::array<std::uint32_t, 4>& bits,
                  TextureFormat format) noexcept {
    std::array<double, 4> values{};
    const auto numeric = graphics::texture_numeric_class(format);
    if (numeric != graphics::TextureNumericClass::Float) {
        // Both signed and unsigned 32-bit integers convert to double exactly.
        for (std::size_t component = 0; component < values.size(); ++component)
            values[component] = numeric == graphics::TextureNumericClass::Sint
                ? static_cast<double>(std::bit_cast<std::int32_t>(bits[component]))
                : static_cast<double>(bits[component]);
        return values;
    }
    for (std::size_t component = 0; component < values.size(); ++component)
        values[component] = static_cast<double>(std::bit_cast<float>(bits[component]));
    return values;
}

namespace {

using PassResult = graphics::RendererResult<MTL4RenderPassDescriptor *>;

PassResult pass_failure(std::string_view detail) {
    return graphics::RendererError{CommandResult{CommandStatus::Unsupported, detail}};
}

bool extent_covers(const id<MTLTexture> texture, const graphics::RenderArea& area) {
    return std::uint64_t(texture.width) >= area.x + std::int64_t(area.width) &&
           std::uint64_t(texture.height) >= area.y + std::int64_t(area.height);
}

// Validates one bound attachment against its plan and configures the Metal
// attachment descriptor (texture, level, slice, load/store, clear value).
std::optional<CommandResult>
configure_color(MTLRenderPassColorAttachmentDescriptor* attachment,
                const graphics::AttachmentPlan& plan,
                const NativeAttachment& binding,
                const graphics::RenderArea& area,
                const RenderPassClearValues* clear_values,
                std::size_t slot) {
    if (!binding.texture)
        return CommandResult{CommandStatus::Unsupported,
                             "planned color attachment has no resident texture"};
    const auto mapped = metal_render_format(plan.texture.format,
                                            graphics::TextureAspect::Color);
    if (const auto* error = std::get_if<CommandResult>(&mapped)) return *error;
    if (binding.texture.pixelFormat != std::get<MTLPixelFormat>(mapped))
        return CommandResult{CommandStatus::Unsupported,
                             "color attachment pixel format contradicts the plan"};
    const bool single_plane = binding.texture.textureType == MTLTextureType3D &&
                              binding.texture.depth == 1 && binding.slice == 0;
    if (binding.texture.textureType != MTLTextureType2D &&
        binding.texture.textureType != MTLTextureType2DArray && !single_plane)
        return CommandResult{CommandStatus::Unsupported,
                             "color attachment type or 3D depth plane is unsupported"};
    if (binding.texture.sampleCount != plan.texture.samples)
        return CommandResult{CommandStatus::Unsupported,
                             "color attachment sample count contradicts the pass"};
    if (!extent_covers(binding.texture, area))
        return CommandResult{CommandStatus::Unsupported,
                             "color attachment is smaller than the render area"};
    if (binding.level >= binding.texture.mipmapLevelCount ||
        (binding.texture.textureType == MTLTextureType2DArray &&
         binding.slice >= binding.texture.arrayLength))
        return CommandResult{CommandStatus::Unsupported,
                             "color attachment level or slice is out of range"};
    attachment.texture = binding.texture;
    attachment.level = binding.level;
    attachment.slice = binding.slice;
    // A single-slice 3D allocation retains its guest Z tiling in L4. Metal
    // addresses that slice as a depth plane, independently of array slices.
    attachment.depthPlane = 0;
    attachment.storeAction = MTLStoreActionStore;
    if (plan.load == graphics::AttachmentLoad::Clear) {
        if (!clear_values)
            return CommandResult{CommandStatus::Unsupported,
                                 "cleared attachment has no clear values"};
        attachment.loadAction = MTLLoadActionClear;
        const auto& color = clear_values->color[slot];
        attachment.clearColor = MTLClearColorMake(color[0], color[1], color[2], color[3]);
    } else {
        attachment.loadAction = MTLLoadActionLoad;
    }
    return std::nullopt;
}

std::optional<CommandResult>
configure_depth_stencil(MTL4RenderPassDescriptor * pass,
                        const graphics::RenderTargetPlan& plan,
                        const graphics::AttachmentPlan& attachment_plan,
                        const NativeAttachment& binding,
                        const RenderPassClearValues* clear_values,
                        bool stencil) {
    if (!binding.texture)
        return CommandResult{CommandStatus::Unsupported,
                             "planned depth/stencil attachment has no resident texture"};
    const auto mapped = metal_render_format(
        attachment_plan.texture.format,
        stencil ? graphics::TextureAspect::Stencil : graphics::TextureAspect::Depth);
    if (const auto* error = std::get_if<CommandResult>(&mapped)) return *error;
    if (binding.texture.pixelFormat != std::get<MTLPixelFormat>(mapped))
        return CommandResult{CommandStatus::Unsupported,
                             "depth/stencil native format contradicts the attachment"};
    if (binding.texture.sampleCount != plan.samples)
        return CommandResult{CommandStatus::Unsupported,
                             "depth/stencil sample count contradicts the pass"};
    if (!extent_covers(binding.texture, plan.area))
        return CommandResult{CommandStatus::Unsupported,
                             "depth/stencil attachment is smaller than the render area"};
    if (stencil) {
        MTLRenderPassStencilAttachmentDescriptor* attachment = pass.stencilAttachment;
        attachment.texture = binding.texture;
        attachment.level = binding.level;
        attachment.slice = binding.slice;
        attachment.storeAction = MTLStoreActionStore;
        if (attachment_plan.load == graphics::AttachmentLoad::Clear) {
            if (!clear_values)
                return CommandResult{CommandStatus::Unsupported,
                                     "cleared depth/stencil attachment has no clear values"};
            attachment.loadAction = MTLLoadActionClear;
            attachment.clearStencil = clear_values->stencil;
        } else {
            attachment.loadAction = MTLLoadActionLoad;
        }
    } else {
        MTLRenderPassDepthAttachmentDescriptor* attachment = pass.depthAttachment;
        attachment.texture = binding.texture;
        attachment.level = binding.level;
        attachment.slice = binding.slice;
        attachment.storeAction = MTLStoreActionStore;
        if (attachment_plan.load == graphics::AttachmentLoad::Clear) {
            if (!clear_values)
                return CommandResult{CommandStatus::Unsupported,
                                     "cleared depth/stencil attachment has no clear values"};
            attachment.loadAction = MTLLoadActionClear;
            attachment.clearDepth = clear_values->depth;
        } else {
            attachment.loadAction = MTLLoadActionLoad;
        }
    }
    return std::nullopt;
}

}  // namespace

graphics::RenderArea render_pass_area(const graphics::RenderTargetPlan& plan) noexcept {
    auto area = plan.area;
    if (!plan.depth && !plan.stencil &&
        std::ranges::none_of(plan.colors, [](const auto& color) { return color.has_value(); })) {
        // 16K is supported across the admitted Apple Silicon Metal 4 profile.
        constexpr std::uint64_t max_extent = 16384;
        area.width = std::clamp(area.width, std::uint64_t{1}, max_extent);
        area.height = std::clamp(area.height, std::uint64_t{1}, max_extent);
    }
    return area;
}

graphics::RendererResult<MTL4RenderPassDescriptor *>
make_render_pass(const graphics::RenderTargetPlan& plan,
                 const NativeRenderTargets& targets,
                 const RenderPassClearValues* clear_values) noexcept {
    if (plan.samples != 1 && plan.samples != 2 && plan.samples != 4)
        return pass_failure("render pass sample count is not renderable");
    MTL4RenderPassDescriptor * pass = [MTL4RenderPassDescriptor new];
    if (!pass) return pass_failure("Metal render pass allocation failed");

    std::size_t bound_colors = 0;
    for (std::size_t slot = 0; slot < plan.colors.size(); ++slot) {
        const bool planned = plan.colors[slot].has_value();
        const bool bound = targets.colors[slot].texture != nil;
        if (planned != bound) {
            return pass_failure(planned
                ? "planned color attachment has no resident texture"
                : "bound color texture names no planned slot");
        }
        if (!planned) continue;
        ++bound_colors;
        if (const auto error = configure_color(pass.colorAttachments[slot],
                                               *plan.colors[slot],
                                               targets.colors[slot], plan.area,
                                               clear_values, slot))
            return pass_failure(error->detail);
    }
    if (plan.depth.has_value() != (targets.depth.texture != nil) ||
        plan.stencil.has_value() != (targets.stencil.texture != nil))
        return pass_failure("depth/stencil attachment presence differs from the plan");
    if (bound_colors == 0 && !plan.depth && !plan.stencil) {
        const auto area = render_pass_area(plan);
        pass.renderTargetWidth = area.width;
        pass.renderTargetHeight = area.height;
        pass.defaultRasterSampleCount = plan.samples;
    }

    if (plan.depth) {
        if (plan.stencil && (targets.stencil.texture != targets.depth.texture ||
                             targets.stencil.level != targets.depth.level ||
                             targets.stencil.slice != targets.depth.slice))
            return pass_failure("packed depth/stencil aspects must alias one subresource");
        if (const auto error = configure_depth_stencil(
                pass, plan, *plan.depth, targets.depth, clear_values, false))
            return pass_failure(error->detail);
    }
    if (plan.stencil) {
        if (const auto error = configure_depth_stencil(
                pass, plan, *plan.stencil, targets.stencil, clear_values, true))
            return pass_failure(error->detail);
    }
    return pass;
}

}  // namespace armigo::platform::metal
