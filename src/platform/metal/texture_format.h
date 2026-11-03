// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Guest format mapping records exact Metal matches, required channel swizzles,
// and formats without a direct Metal counterpart.

#pragma once

#import <Metal/Metal.h>

#include <armigo/graphics/texture.h>

#include <array>
#include <span>
#include <variant>
#include <vector>

namespace armigo::platform {

// The host sampling representation; packed depth uploads require conversion.
struct MetalSampledFormat {
    MTLPixelFormat pixel_format;
    MTLTextureSwizzleChannels swizzle;
};

// Packed depth/stencil formats map to their depth-only sampling aspect.
[[nodiscard]] std::variant<MetalSampledFormat, graphics::TextureError>
metal_pixel_format(graphics::TextureFormat format) noexcept;

// Reject stencil-aspect views when the shader requires depth comparison.
[[nodiscard]] std::variant<MetalSampledFormat, graphics::TextureError>
metal_sampled_format(const graphics::TextureDescriptor& texture) noexcept;

// A null sampled resource is a one-texel, shader-typed zero texture with no guest
// backing. Initialize every slice explicitly and complete the upload before
// returning. Multisample null resources remain unsupported.
[[nodiscard]] std::variant<id<MTLTexture>, graphics::CommandResult>
metal_null_texture(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                   const graphics::TextureDescriptor& texture,
                   graphics::TextureNumericClass scalar, bool depth);

// Extract packed depth into the Metal depth representation without exposing stencil.
[[nodiscard]] std::variant<std::vector<std::byte>, graphics::TextureError>
metal_depth_upload(graphics::TextureFormat format, std::span<const std::byte> bytes);

// Persistent storage retains both depth and stencil. Plane conversions preserve
// guest padding and untouched components; stencil is separate for Metal blits.
struct MetalTexturePlanes {
    std::vector<std::byte> color_or_depth;
    std::vector<std::byte> stencil;
};
[[nodiscard]] std::variant<MTLPixelFormat, graphics::TextureError>
metal_storage_format(graphics::TextureFormat format) noexcept;
[[nodiscard]] std::variant<MetalTexturePlanes, graphics::TextureError>
metal_texture_planes(graphics::TextureFormat format, std::span<const std::byte> bytes);
// Whether metal_texture_planes returns every whole-block input unchanged as
// the single plane: a format with no depth or stencil aspect, or depth alone
// in 16-bit unorm or 32-bit float, whose guest bytes the Metal texture stores
// as they are.
[[nodiscard]] bool metal_texture_planes_identity(graphics::TextureFormat format) noexcept;
[[nodiscard]] std::variant<std::vector<std::byte>, graphics::TextureError>
metal_merge_texture_planes(graphics::TextureFormat format,
    const MetalTexturePlanes& planes, std::span<const std::byte> original);

// The mapped host representation of one decoded TSC sampler. The custom-border
// flag selects shader-side configuration; it does not alter the native sampler
// descriptor and is intentionally excluded from host-state equality and cache
// deduplication.
struct MetalSamplerProperties {
    MTLSamplerAddressMode address_u = MTLSamplerAddressModeClampToEdge;
    MTLSamplerAddressMode address_v = MTLSamplerAddressModeClampToEdge;
    MTLSamplerAddressMode address_r = MTLSamplerAddressModeClampToEdge;
    MTLSamplerMinMagFilter mag_filter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMinMagFilter min_filter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMipFilter mip_filter = MTLSamplerMipFilterLinear;
    MTLSamplerReductionMode reduction_mode =
        MTLSamplerReductionModeWeightedAverage;
    MTLCompareFunction compare_function = MTLCompareFunctionNever;
    MTLSamplerBorderColor border_color = MTLSamplerBorderColorTransparentBlack;
    float lod_bias = 0.0f;
    float lod_min_clamp = 0.0f;
    float lod_max_clamp = 0.0f;
    float max_anisotropy = 1.0f;
    // Metal samplers always normalize; the decoded TSC float-coordinate
    // normalization bit selects whether the request is expressible at all.
    bool normalized_coordinates = true;
    bool custom_border = false;
};

bool operator==(const MetalSamplerProperties& left,
                const MetalSamplerProperties& right) noexcept;

// Maps a decoded TSC onto Metal sampler state. Address modes with no Metal
// equivalent use clamp fallbacks.
// Finite non-constant borders admit only the bounded shader-side profile;
// unestablished normalization, nonzero trilinear optimization and other custom
// border combinations return a structured unsupported error.
[[nodiscard]] std::variant<MetalSamplerProperties, graphics::TextureError>
metal_sampler_properties(const graphics::SamplerDescriptor& sampler,
                         bool srgb_texture = false) noexcept;

// Returns the effective shader-side custom border. The TSC conversion flag
// applies the packed-byte sRGB transfer only when the sampled texture is sRGB.
[[nodiscard]] std::array<float, 4> metal_custom_border_color(
    const graphics::SamplerDescriptor& sampler, bool srgb_texture) noexcept;

// Builds an MTLSamplerDescriptor for the mapped properties; the draw path
// creates the actual MTLSamplerState from it.
[[nodiscard]] MTLSamplerDescriptor* metal_sampler_descriptor(
    const MetalSamplerProperties& properties) noexcept;

}  // namespace armigo::platform
