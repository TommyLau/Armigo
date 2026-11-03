// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "texture_format.h"

#include <armigo/graphics/texture_format.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <utility>
#include <variant>

namespace {

using namespace armigo::graphics;
namespace platform = armigo::platform;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (false)

bool identity_swizzle(MTLTextureSwizzleChannels swizzle) {
    return swizzle.red == MTLTextureSwizzleRed &&
           swizzle.green == MTLTextureSwizzleGreen &&
           swizzle.blue == MTLTextureSwizzleBlue &&
           swizzle.alpha == MTLTextureSwizzleAlpha;
}

// Every mapped format must allocate a real MTLTexture with its swizzle; every
// unmapped format must report the structured unsupported error.
void test_metal_mapping(id<MTLDevice> device) {
    const auto first = TextureFormat::R8Unorm;
    const auto last = TextureFormat::Astc12x12Srgb;
    for (int value = static_cast<int>(first); value <= static_cast<int>(last);
         ++value) {
        const auto format = static_cast<TextureFormat>(value);
        const auto mapped = platform::metal_pixel_format(format);
        if (auto* error = std::get_if<armigo::graphics::TextureError>(&mapped)) {
            CHECK(error->code == TextureErrorCode::UnsupportedTextureFormat);
            CHECK(error->field == TextureField::Format);
            CHECK(error->message()[0] != '\0');
            continue;
        }
        const auto& sampled = std::get<platform::MetalSampledFormat>(mapped);
        CHECK(sampled.pixel_format != MTLPixelFormatInvalid);
        CHECK(identity_swizzle(sampled.swizzle));
        auto* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:sampled.pixel_format
                                         width:4 height:4 mipmapped:NO];
        descriptor.storageMode = texture_format_info(format).depth
            ? MTLStorageModePrivate : MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead;
        descriptor.swizzle = sampled.swizzle;
        id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
        CHECK(texture != nil);
    }
}

// The TIC swizzle composes with the format mapping into the Metal view
// swizzle: channels select the mapped Metal channel, zero and both one forms
// select Metal's constants, and the composed view still allocates.
void test_view_swizzle(id<MTLDevice> device) {
    TextureDescriptor texture;
    texture.format = TextureFormat::R8G8B8A8Unorm;
    texture.swizzle = {TextureSwizzle::Blue, TextureSwizzle::Green, TextureSwizzle::Red,
                       TextureSwizzle::OneFloat};
    auto mapped = platform::metal_sampled_format(texture);
    const auto* sampled = std::get_if<platform::MetalSampledFormat>(&mapped);
    CHECK(sampled && sampled->swizzle.red == MTLTextureSwizzleBlue &&
          sampled->swizzle.green == MTLTextureSwizzleGreen &&
          sampled->swizzle.blue == MTLTextureSwizzleRed &&
          sampled->swizzle.alpha == MTLTextureSwizzleOne);
    texture.format = TextureFormat::R32Uint;
    texture.swizzle = {TextureSwizzle::Red, TextureSwizzle::Zero, TextureSwizzle::Red,
                       TextureSwizzle::OneSInt};
    mapped = platform::metal_sampled_format(texture);
    sampled = std::get_if<platform::MetalSampledFormat>(&mapped);
    CHECK(sampled && sampled->swizzle.red == MTLTextureSwizzleRed &&
          sampled->swizzle.green == MTLTextureSwizzleZero &&
          sampled->swizzle.blue == MTLTextureSwizzleRed &&
          sampled->swizzle.alpha == MTLTextureSwizzleOne);
    if (sampled) {
        auto* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:sampled->pixel_format width:4 height:4 mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead;
        descriptor.swizzle = sampled->swizzle;
        CHECK([device newTextureWithDescriptor:descriptor] != nil);
    }
}

// Spot checks pinning the mapping identities that carry bit-layout meaning.
void test_mapping_identity(id<MTLDevice> device) {
    const std::array<std::pair<TextureFormat, MTLPixelFormat>, 18> exact{{
        {TextureFormat::D16Unorm, MTLPixelFormatDepth16Unorm},
        {TextureFormat::D32Float, MTLPixelFormatDepth32Float},
        {TextureFormat::D24UnormS8Uint, MTLPixelFormatDepth32Float},
        {TextureFormat::X8UintD24Unorm, MTLPixelFormatDepth32Float},
        {TextureFormat::S8UintD24Unorm, MTLPixelFormatDepth32Float},
        {TextureFormat::D32FloatS8Uint, MTLPixelFormatDepth32Float},
        {TextureFormat::R10G10B10A2Unorm, MTLPixelFormatRGB10A2Unorm},
        {TextureFormat::R10G10B10A2Uint, MTLPixelFormatRGB10A2Uint},
        {TextureFormat::R11G11B10Float, MTLPixelFormatRG11B10Float},
        {TextureFormat::R9G9B9E5Float, MTLPixelFormatRGB9E5Float},
        {TextureFormat::R5G6B5Unorm, MTLPixelFormatB5G6R5Unorm},
        {TextureFormat::R4G4B4A4Unorm, MTLPixelFormatABGR4Unorm},
        {TextureFormat::R5G5B5A1Unorm, MTLPixelFormatA1BGR5Unorm},
        {TextureFormat::R8G8B8A8Srgb, MTLPixelFormatRGBA8Unorm_sRGB},
        {TextureFormat::Bc1RgbaUnorm, MTLPixelFormatBC1_RGBA},
        {TextureFormat::Bc1RgbaSrgb, MTLPixelFormatBC1_RGBA_sRGB},
        {TextureFormat::Etc2RgbPtaUnorm, MTLPixelFormatETC2_RGB8A1},
        {TextureFormat::Astc8x8Srgb, MTLPixelFormatASTC_8x8_sRGB},
    }};
    for (const auto& [format, expected] : exact) {
        const auto mapped = platform::metal_pixel_format(format);
        CHECK(std::holds_alternative<platform::MetalSampledFormat>(mapped));
        if (const auto* sampled =
                std::get_if<platform::MetalSampledFormat>(&mapped)) {
            CHECK(sampled->pixel_format == expected);
        }
        static_cast<void>(device);
    }
    // Unsupported color packings retain structured errors.
    const std::array<TextureFormat, 4> unmapped{{
        TextureFormat::R32G32B32Float,
        TextureFormat::R32G32B32Uint,
        TextureFormat::R4G4Unorm,
        TextureFormat::A5B5G5R1Unorm,
    }};
    for (const TextureFormat format : unmapped) {
        const auto mapped = platform::metal_pixel_format(format);
        CHECK(std::holds_alternative<armigo::graphics::TextureError>(mapped));
    }
}

void test_depth_upload_and_aspects() {
    const std::array<std::uint32_t,4> depths{0,0x123456,0x800000,0xffffff};
    for (const auto format : {TextureFormat::D24UnormS8Uint,
                              TextureFormat::X8UintD24Unorm,
                              TextureFormat::S8UintD24Unorm}) {
        std::array<std::uint32_t,4> packed{};
        for (unsigned i=0;i<4;++i) packed[i]=(depths[i]<<8)|(0x51u+i);
        const auto converted=platform::metal_depth_upload(format,std::as_bytes(std::span(packed)));
        CHECK(std::holds_alternative<std::vector<std::byte>>(converted));
        if (const auto* bytes=std::get_if<std::vector<std::byte>>(&converted)) {
            CHECK(bytes->size()==sizeof(float)*depths.size());
            for (unsigned i=0;i<4;++i) {
                float actual=0;
                std::memcpy(&actual,bytes->data()+i*sizeof(float),sizeof(actual));
                CHECK(actual==static_cast<float>(depths[i])/16777215.0f);
            }
        }
        // A constant selector before R/G must not hide the selected aspect.
        TextureDescriptor texture;
        texture.format=format;
        const auto depth=format==TextureFormat::D24UnormS8Uint
            ?TextureSwizzle::Green:TextureSwizzle::Red;
        const auto stencil=format==TextureFormat::D24UnormS8Uint
            ?TextureSwizzle::Red:TextureSwizzle::Green;
        texture.swizzle={TextureSwizzle::Zero,depth,depth,TextureSwizzle::OneFloat};
        CHECK(std::holds_alternative<platform::MetalSampledFormat>(platform::metal_sampled_format(texture)));
        texture.swizzle[1]=stencil;
        CHECK(std::holds_alternative<TextureError>(platform::metal_sampled_format(texture)));
    }
    // D32F extraction preserves signed zero, infinity and a NaN payload.
    const std::array<std::uint32_t,6> packed32{0x80000000,0x5a,0x7f800000,0x6b,0x7fc00123,0x7c};
    const auto converted=platform::metal_depth_upload(TextureFormat::D32FloatS8Uint,
                                                     std::as_bytes(std::span(packed32)));
    const auto* bytes=std::get_if<std::vector<std::byte>>(&converted);
    CHECK(bytes && bytes->size()==12);
    if (bytes) for (unsigned i=0;i<3;++i) {
        std::uint32_t actual=0;
        std::memcpy(&actual,bytes->data()+i*4,4);
        CHECK(actual==packed32[i*2]);
    }
    const std::array<std::byte,3> incomplete{};
    CHECK(std::holds_alternative<TextureError>(platform::metal_depth_upload(
        TextureFormat::D24UnormS8Uint,incomplete)));
    for (const auto format : {TextureFormat::D16Unorm,TextureFormat::D32Float}) {
        TextureDescriptor texture;
        texture.format=format;
        texture.swizzle.fill(TextureSwizzle::Green);
        CHECK(std::holds_alternative<TextureError>(platform::metal_sampled_format(texture)));
    }
}

std::vector<std::byte> literal_bytes(std::initializer_list<unsigned char> values) {
    std::vector<std::byte> result;
    for (const auto value : values) result.push_back(std::byte{value});
    return result;
}

template <class T, std::size_t N>
std::vector<std::byte> array_bytes(const std::array<T, N>& values) {
    const auto bytes = std::as_bytes(std::span(values));
    return {bytes.begin(), bytes.end()};
}

void check_merged(TextureFormat format, const platform::MetalTexturePlanes& planes,
                  const std::vector<std::byte>& original,
                  const std::vector<std::byte>& expected) {
    const auto merged = platform::metal_merge_texture_planes(format, planes, original);
    const auto* bytes = std::get_if<std::vector<std::byte>>(&merged);
    CHECK(bytes != nullptr);
    if (bytes) CHECK(*bytes == expected);
}

void test_persistent_storage_formats(id<MTLDevice> device) {
    const std::array<std::pair<TextureFormat, MTLPixelFormat>, 7> formats{{
        {TextureFormat::D16Unorm, MTLPixelFormatDepth16Unorm},
        {TextureFormat::D32Float, MTLPixelFormatDepth32Float},
        {TextureFormat::D24UnormS8Uint, MTLPixelFormatDepth32Float_Stencil8},
        {TextureFormat::X8UintD24Unorm, MTLPixelFormatDepth32Float_Stencil8},
        {TextureFormat::S8UintD24Unorm, MTLPixelFormatDepth32Float_Stencil8},
        {TextureFormat::D32FloatS8Uint, MTLPixelFormatDepth32Float_Stencil8},
        {TextureFormat::R8G8B8A8Unorm, MTLPixelFormatRGBA8Unorm},
    }};
    for (const auto& [format, expected] : formats) {
        const auto mapped = platform::metal_storage_format(format);
        const auto* actual = std::get_if<MTLPixelFormat>(&mapped);
        CHECK(actual && *actual == expected);
        if (!actual) continue;
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:*actual
            width:4 height:4 mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        CHECK([device newTextureWithDescriptor:descriptor] != nil);
    }
    CHECK(std::holds_alternative<TextureError>(
        platform::metal_storage_format(TextureFormat::R32G32B32Float)));
}

void test_packed_depth_plane_preservation() {
    // Four independently chosen texels pin the low stencil/padding byte and
    // include nontrivial depth values that must survive an untouched readback.
    const auto original = literal_bytes({
        0x51, 0x00, 0x00, 0x00, 0x62, 0xff, 0xff, 0xff,
        0x73, 0x56, 0x34, 0x12, 0x84, 0x00, 0x00, 0x80});
    for (const auto format : {TextureFormat::D24UnormS8Uint,
                              TextureFormat::X8UintD24Unorm,
                              TextureFormat::S8UintD24Unorm}) {
        const auto converted = platform::metal_texture_planes(format, original);
        const auto* baseline = std::get_if<platform::MetalTexturePlanes>(&converted);
        CHECK(baseline != nullptr);
        if (!baseline) continue;
        CHECK(baseline->color_or_depth.size() == 16);
        if (baseline->color_or_depth.size() != 16) continue;
        std::array<float, 4> depths{};
        std::memcpy(depths.data(), baseline->color_or_depth.data(), sizeof(depths));
        CHECK(depths[0] == 0.0f);
        CHECK(depths[1] == 1.0f);
        const bool stencil = format != TextureFormat::X8UintD24Unorm;
        CHECK(baseline->stencil == (stencil ? literal_bytes({0x51, 0x62, 0x73, 0x84})
                                           : std::vector<std::byte>{}));
        check_merged(format, *baseline, original, original);

        auto changed = *baseline;
        changed.color_or_depth = array_bytes(std::array{1.0f, 0.0f, 0.5f, 0.25f});
        const auto expected_depth = literal_bytes({
            0x51, 0xff, 0xff, 0xff, 0x62, 0x00, 0x00, 0x00,
            0x73, 0x00, 0x00, 0x80, 0x84, 0x00, 0x00, 0x40});
        check_merged(format, changed, original, expected_depth);
        if (stencil) {
            changed = *baseline;
            changed.stencil = literal_bytes({0xf1, 0xe2, 0xd3, 0xc4});
            check_merged(format, changed, original, literal_bytes({
                0xf1, 0x00, 0x00, 0x00, 0xe2, 0xff, 0xff, 0xff,
                0xd3, 0x56, 0x34, 0x12, 0xc4, 0x00, 0x00, 0x80}));
        }
        for (const float invalid : {std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::quiet_NaN()}) {
            changed = *baseline;
            std::memcpy(changed.color_or_depth.data(), &invalid, sizeof(invalid));
            CHECK(std::holds_alternative<TextureError>(
                platform::metal_merge_texture_planes(format, changed, original)));
        }
    }
}

void test_float_depth_stencil_padding_preservation() {
    const auto format = TextureFormat::D32FloatS8Uint;
    // Signed zero, infinity and a NaN payload must retain their exact bits;
    // each stencil byte is followed by three unrelated guest padding bytes.
    const auto original = literal_bytes({
        0x00, 0x00, 0x00, 0x80, 0x51, 0xc3, 0xb2, 0xa1,
        0x00, 0x00, 0x80, 0x7f, 0x62, 0xf6, 0xe5, 0xd4,
        0x23, 0x01, 0xc0, 0x7f, 0x73, 0x59, 0x41, 0x31});
    const auto converted = platform::metal_texture_planes(format, original);
    const auto* baseline = std::get_if<platform::MetalTexturePlanes>(&converted);
    CHECK(baseline != nullptr);
    if (!baseline) return;
    CHECK(baseline->color_or_depth == literal_bytes({
        0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x7f, 0x23, 0x01, 0xc0, 0x7f}));
    CHECK(baseline->stencil == literal_bytes({0x51, 0x62, 0x73}));
    check_merged(format, *baseline, original, original);
    auto changed = *baseline;
    changed.color_or_depth = array_bytes(std::array{0.5f, 1.0f, 0.0f});
    check_merged(format, changed, original, literal_bytes({
        0x00, 0x00, 0x00, 0x3f, 0x51, 0xc3, 0xb2, 0xa1,
        0x00, 0x00, 0x80, 0x3f, 0x62, 0xf6, 0xe5, 0xd4,
        0x00, 0x00, 0x00, 0x00, 0x73, 0x59, 0x41, 0x31}));
    changed = *baseline;
    changed.stencil = literal_bytes({0xab, 0xcd, 0xef});
    check_merged(format, changed, original, literal_bytes({
        0x00, 0x00, 0x00, 0x80, 0xab, 0xc3, 0xb2, 0xa1,
        0x00, 0x00, 0x80, 0x7f, 0xcd, 0xf6, 0xe5, 0xd4,
        0x23, 0x01, 0xc0, 0x7f, 0xef, 0x59, 0x41, 0x31}));
}

void test_direct_planes_and_malformed_sizes() {
    for (const auto format : {TextureFormat::D16Unorm, TextureFormat::D32Float,
                              TextureFormat::R8G8B8A8Unorm}) {
        const auto original = literal_bytes({0x00, 0x00, 0x00, 0x80, 0x23, 0x01, 0xc0, 0x7f});
        const auto converted = platform::metal_texture_planes(format, original);
        const auto* baseline = std::get_if<platform::MetalTexturePlanes>(&converted);
        CHECK(baseline != nullptr);
        if (!baseline) continue;
        CHECK(baseline->color_or_depth == original);
        CHECK(baseline->stencil.empty());
        check_merged(format, *baseline, original, original);
        auto changed = *baseline;
        changed.color_or_depth = literal_bytes({0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe});
        check_merged(format, changed, original, changed.color_or_depth);
    }
    for (const auto format : {TextureFormat::D16Unorm, TextureFormat::D32Float,
                              TextureFormat::D24UnormS8Uint, TextureFormat::X8UintD24Unorm,
                              TextureFormat::S8UintD24Unorm, TextureFormat::D32FloatS8Uint,
                              TextureFormat::R8G8B8A8Unorm}) {
        const auto original = literal_bytes({0x51, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef});
        const auto converted = platform::metal_texture_planes(format, original);
        const auto* baseline = std::get_if<platform::MetalTexturePlanes>(&converted);
        CHECK(baseline != nullptr);
        if (!baseline) continue;
        auto truncated = original;
        truncated.pop_back();
        CHECK(std::holds_alternative<TextureError>(platform::metal_texture_planes(format, truncated)));
        CHECK(std::holds_alternative<TextureError>(
            platform::metal_merge_texture_planes(format, *baseline, truncated)));
        auto changed = *baseline;
        changed.color_or_depth.pop_back();
        CHECK(std::holds_alternative<TextureError>(
            platform::metal_merge_texture_planes(format, changed, original)));
        changed = *baseline;
        changed.color_or_depth.push_back(std::byte{0});
        CHECK(std::holds_alternative<TextureError>(
            platform::metal_merge_texture_planes(format, changed, original)));
        changed = *baseline;
        changed.stencil.push_back(std::byte{0});
        CHECK(std::holds_alternative<TextureError>(
            platform::metal_merge_texture_planes(format, changed, original)));
        if (!baseline->stencil.empty()) {
            changed = *baseline;
            changed.stencil.pop_back();
            CHECK(std::holds_alternative<TextureError>(
                platform::metal_merge_texture_planes(format, changed, original)));
        }
    }
}

// Sampler state: the full decoded-TSC-to-Metal mapping, including the
// documented fallbacks, the structured unsupported fields, and real sampler
// state allocation for representative descriptors.
void test_sampler_mapping(id<MTLDevice> device) {
    SamplerDescriptor sampler;

    // Every defined address mode maps; the OpenGL-clamp family follows the
    // documented address-mode fallbacks.
    const std::array<std::pair<SamplerWrap, MTLSamplerAddressMode>, 8> modes{{
        {SamplerWrap::Repeat, MTLSamplerAddressModeRepeat},
        {SamplerWrap::MirroredRepeat, MTLSamplerAddressModeMirrorRepeat},
        {SamplerWrap::ClampToEdge, MTLSamplerAddressModeClampToEdge},
        {SamplerWrap::ClampToBorder, MTLSamplerAddressModeClampToBorderColor},
        {SamplerWrap::Clamp, MTLSamplerAddressModeClampToEdge},
        {SamplerWrap::MirrorClampToEdge,
         MTLSamplerAddressModeMirrorClampToEdge},
        {SamplerWrap::MirrorClampToBorder,
         MTLSamplerAddressModeClampToBorderColor},
        {SamplerWrap::MirrorClamp, MTLSamplerAddressModeMirrorClampToEdge},
    }};
    for (const auto& [wrap, expected] : modes) {
        sampler.wrap = {wrap, wrap, wrap};
        const auto mapped = platform::metal_sampler_properties(sampler);
        CHECK(std::holds_alternative<platform::MetalSamplerProperties>(mapped));
        if (const auto* properties =
                std::get_if<platform::MetalSamplerProperties>(&mapped)) {
            CHECK(properties->address_u == expected);
            CHECK(properties->address_v == expected);
            CHECK(properties->address_r == expected);
        }
    }
    sampler.wrap = {SamplerWrap::Repeat, SamplerWrap::MirrorClampToEdge,
                    SamplerWrap::ClampToBorder};
    {
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties &&
              properties->address_u == MTLSamplerAddressModeRepeat &&
              properties->address_v ==
                  MTLSamplerAddressModeMirrorClampToEdge &&
              properties->address_r ==
                  MTLSamplerAddressModeClampToBorderColor);
    }
    sampler.wrap.fill(SamplerWrap::ClampToEdge);

    // Filters, mip filter and reduction map one to one.
    sampler.magnification = SamplerFilter::Nearest;
    sampler.minification = SamplerFilter::Nearest;
    sampler.mip_filter = SamplerMipFilter::None;
    sampler.reduction = SamplerReduction::Maximum;
    {
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties &&
              properties->mag_filter == MTLSamplerMinMagFilterNearest &&
              properties->min_filter == MTLSamplerMinMagFilterNearest &&
              properties->mip_filter == MTLSamplerMipFilterNotMipmapped &&
              properties->reduction_mode ==
                  MTLSamplerReductionModeMaximum);
    }
    sampler.magnification = SamplerFilter::Linear;
    sampler.minification = SamplerFilter::Linear;
    sampler.mip_filter = SamplerMipFilter::Linear;
    sampler.reduction = SamplerReduction::WeightedAverage;

    // Comparison: disabled keeps Metal's ordinary sampling default; enabled
    // uses the decoded ordered function.
    {
        sampler.depth_compare = true;
        sampler.compare_function = SamplerCompareFunction::GreaterOrEqual;
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties && properties->compare_function ==
                                MTLCompareFunctionGreaterEqual);
        sampler.depth_compare = false;
        sampler.compare_function = SamplerCompareFunction::Never;
    }

    // LOD bias, LOD clamps and anisotropy pass through to Metal 26 samplers.
    {
        sampler.lod_bias = -0.25f;
        sampler.min_lod = 1.5f;
        sampler.max_lod = 4.5f;
        sampler.max_anisotropy = 12.0f;
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties && properties->lod_bias == -0.25f &&
              properties->lod_min_clamp == 1.5f &&
              properties->lod_max_clamp == 4.5f &&
              properties->max_anisotropy == 12.0f);
        sampler.lod_bias = 0.0f;
        sampler.min_lod = 0.0f;
        sampler.max_lod = 0.0f;
        sampler.max_anisotropy = 1.0f;
    }

    // The three constant border colours remain native sampler state.
    const std::array<std::array<float, 4>, 3> constants{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f},
    }};
    const std::array<MTLSamplerBorderColor, 3> expected_colors{{
        MTLSamplerBorderColorTransparentBlack,
        MTLSamplerBorderColorOpaqueBlack,
        MTLSamplerBorderColorOpaqueWhite,
    }};
    for (unsigned constant = 0; constant < constants.size(); ++constant) {
        sampler.border_color = constants[constant];
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties &&
              properties->border_color == expected_colors[constant] &&
              !properties->custom_border);
    }

    // A finite custom border is admitted only for the bounded shader-side
    // profile. The native sampler clamps every axis to transparent black;
    // shader configuration does not affect native-state deduplication.
    {
        sampler.border_color = {0.25f, 0.5f, 0.75f, 1.0f};
        sampler.wrap.fill(SamplerWrap::ClampToBorder);
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* properties =
            std::get_if<platform::MetalSamplerProperties>(&mapped);
        CHECK(properties && properties->custom_border &&
              properties->address_u == MTLSamplerAddressModeClampToBorderColor &&
              properties->address_v == MTLSamplerAddressModeClampToBorderColor &&
              properties->address_r == MTLSamplerAddressModeClampToBorderColor &&
              properties->border_color ==
                  MTLSamplerBorderColorTransparentBlack);
        if (properties) {
            auto native_equal = *properties;
            native_equal.custom_border = false;
            CHECK(native_equal == *properties);
            CHECK(platform::metal_sampler_descriptor(*properties).borderColor ==
                  MTLSamplerBorderColorTransparentBlack);
        }

        // Every nearest/linear min-mag combination and decoded mip mode uses
        // exact native filter state beneath the shader-side border handling.
        for (const auto magnification : {SamplerFilter::Nearest,
                                         SamplerFilter::Linear}) {
            sampler.magnification = magnification;
            for (const auto minification : {SamplerFilter::Nearest,
                                            SamplerFilter::Linear}) {
                sampler.minification = minification;
                for (const auto mip : {SamplerMipFilter::None,
                                       SamplerMipFilter::Nearest,
                                       SamplerMipFilter::Linear}) {
                    sampler.mip_filter = mip;
                    CHECK(std::holds_alternative<platform::MetalSamplerProperties>(
                        platform::metal_sampler_properties(sampler)));
                }
            }
        }
        sampler.magnification = SamplerFilter::Linear;
        sampler.minification = SamplerFilter::Linear;
        sampler.mip_filter = SamplerMipFilter::Linear;

        // The TSC conversion flag applies only to RGB borders sampled through
        // an sRGB texture. Alpha and every non-converting combination stay raw.
        sampler.border_color = {0.9f, 0.8f, 0.7f, 0.25f};
        sampler.srgb_border_color = {10, 128, 255};
        sampler.srgb_border_conversion = false;
        CHECK(platform::metal_custom_border_color(sampler, true) ==
              sampler.border_color);
        sampler.srgb_border_conversion = true;
        CHECK(std::holds_alternative<platform::MetalSamplerProperties>(
            platform::metal_sampler_properties(sampler)));
        CHECK(platform::metal_custom_border_color(sampler, false) ==
              sampler.border_color);
        const auto converted =
            platform::metal_custom_border_color(sampler, true);
        CHECK(std::abs(converted[0] - 0.00303527f) < 0.000001f);
        CHECK(std::abs(converted[1] - 0.2158605f) < 0.000001f);
        CHECK(converted[2] == 1.0f);
        CHECK(converted[3] == 0.25f);

        // Texture context controls classification. Packed RGB is irrelevant
        // for a non-sRGB texture but can turn a native float constant into a
        // custom effective border when conversion is active.
        sampler.border_color = {1.0f, 1.0f, 1.0f, 1.0f};
        sampler.srgb_border_color = {64, 128, 192};
        const auto linear_fixed =
            platform::metal_sampler_properties(sampler, false);
        const auto* linear_properties =
            std::get_if<platform::MetalSamplerProperties>(&linear_fixed);
        CHECK(linear_properties && !linear_properties->custom_border &&
              linear_properties->border_color ==
                  MTLSamplerBorderColorOpaqueWhite);
        const auto srgb_fixed = platform::metal_sampler_properties(sampler, true);
        const auto* srgb_properties =
            std::get_if<platform::MetalSamplerProperties>(&srgb_fixed);
        CHECK(srgb_properties && srgb_properties->custom_border &&
              srgb_properties->border_color ==
                  MTLSamplerBorderColorTransparentBlack);

        sampler.border_color = {0.25f, 0.5f, 0.75f, 1.0f};
        sampler.srgb_border_color = {255, 255, 255};
        const auto packed_white =
            platform::metal_sampler_properties(sampler, true);
        const auto* packed_white_properties =
            std::get_if<platform::MetalSamplerProperties>(&packed_white);
        CHECK(packed_white_properties &&
              !packed_white_properties->custom_border &&
              packed_white_properties->border_color ==
                  MTLSamplerBorderColorOpaqueWhite);

        sampler.srgb_border_conversion = false;
        sampler.srgb_border_color = {};
        sampler.border_color = {0.25f, 0.5f, 0.75f, 1.0f};

        const auto rejected_custom = [&](TextureField field) {
            const auto rejected = platform::metal_sampler_properties(sampler);
            const auto* error = std::get_if<TextureError>(&rejected);
            CHECK(error &&
                  error->code == TextureErrorCode::UnsupportedSamplerState &&
                  error->field == field);
        };
        sampler.wrap[1] = SamplerWrap::MirrorClampToBorder;
        rejected_custom(TextureField::WrapMode);
        sampler.wrap[1] = SamplerWrap::ClampToEdge;
        rejected_custom(TextureField::WrapMode);
        sampler.wrap.fill(SamplerWrap::ClampToBorder);
        sampler.reduction = SamplerReduction::Minimum;
        rejected_custom(TextureField::ReductionFilter);
        sampler.reduction = SamplerReduction::WeightedAverage;
        sampler.depth_compare = true;
        sampler.compare_function = SamplerCompareFunction::Less;
        rejected_custom(TextureField::DepthCompare);
        sampler.depth_compare = false;
        sampler.compare_function = SamplerCompareFunction::Never;
        sampler.max_anisotropy = 2.0f;
        rejected_custom(TextureField::Anisotropy);
        sampler.max_anisotropy = 1.0f;
        sampler.float_coordinate_normalization = true;
        rejected_custom(TextureField::FloatCoordinateNormalization);
        sampler.float_coordinate_normalization = false;

        for (const float invalid : {
                 std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN()}) {
            for (unsigned component = 0; component < 4; ++component) {
                sampler.border_color = {0.25f, 0.5f, 0.75f, 1.0f};
                sampler.border_color[component] = invalid;
                rejected_custom(TextureField::BorderColor);
            }
        }
        sampler.border_color = {0.0f, 0.0f, 0.0f, 0.0f};
        sampler.wrap.fill(SamplerWrap::ClampToEdge);
    }

    // Fields that no host mapping consumes and Metal cannot express
    // return structured unsupported results.
    sampler.float_coordinate_normalization = true;
    {
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* error = std::get_if<TextureError>(&mapped);
        CHECK(error &&
              error->field == TextureField::FloatCoordinateNormalization);
        sampler.float_coordinate_normalization = false;
    }
    sampler.trilinear_optimization = 0.25f;
    {
        const auto mapped = platform::metal_sampler_properties(sampler);
        const auto* error = std::get_if<TextureError>(&mapped);
        CHECK(error && error->field == TextureField::TrilinearOptimization);
        sampler.trilinear_optimization = 0.0f;
    }

    // Representative mapped descriptors allocate real sampler states.
    for (const bool nearest : {false, true}) {
        sampler.magnification = nearest ? SamplerFilter::Nearest
                                        : SamplerFilter::Linear;
        sampler.minification = sampler.magnification;
        const auto mapped = platform::metal_sampler_properties(sampler);
        CHECK(std::holds_alternative<platform::MetalSamplerProperties>(mapped));
        if (const auto* properties =
                std::get_if<platform::MetalSamplerProperties>(&mapped)) {
            id<MTLSamplerState> state = [device
                newSamplerStateWithDescriptor:platform::metal_sampler_descriptor(
                                                  *properties)];
            CHECK(state != nil);
        }
    }
    sampler.magnification = SamplerFilter::Linear;
    sampler.minification = SamplerFilter::Linear;
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        CHECK(device != nil);
        if (device == nil) {
            std::fprintf(stderr, "No Metal device available\n");
            return 1;
        }
        test_metal_mapping(device);
        test_view_swizzle(device);
        test_mapping_identity(device);
        test_depth_upload_and_aspects();
        test_persistent_storage_formats(device);
        test_packed_depth_plane_preservation();
        test_float_depth_stencil_padding_preservation();
        test_direct_planes_and_malformed_sizes();
        test_sampler_mapping(device);
        std::printf("Metal texture format failures: %u\n", failures);
        return failures == 0 ? 0 : 1;
    }
}
