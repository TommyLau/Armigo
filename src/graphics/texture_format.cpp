// ============================================================================
// Armigo Graphics - GPU Texture Descriptors
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/texture_format.h>

#include <array>

namespace armigo::graphics {
namespace {

// TIC format index (word 0 bits 0-6) per storage layout, and the component
// type encodings shifted into word 0 bits 7-18.
constexpr std::uint32_t component_types(std::uint32_t r, std::uint32_t g,
                                        std::uint32_t b,
                                        std::uint32_t a) noexcept {
    return (r << 7) | (g << 10) | (b << 13) | (a << 16);
}

constexpr std::uint32_t all(std::uint32_t type) noexcept {
    return component_types(type, type, type, type);
}

constexpr std::uint32_t unorm_all = all(tic_component_unorm);
constexpr std::uint32_t snorm_all = all(tic_component_snorm);
constexpr std::uint32_t uint_all = all(tic_component_uint);
constexpr std::uint32_t sint_all = all(tic_component_sint);
constexpr std::uint32_t float_all = all(tic_component_float);
constexpr std::uint32_t unorm_srgb = unorm_all | tic_srgb_flag;

constexpr std::uint32_t unorm_uuu = component_types(
    tic_component_unorm, tic_component_uint, tic_component_uint,
    tic_component_uint);
constexpr std::uint32_t float_uuu = component_types(
    tic_component_float, tic_component_uint, tic_component_uint,
    tic_component_uint);
constexpr std::uint32_t uint_uuu = component_types(
    tic_component_uint, tic_component_unorm, tic_component_unorm,
    tic_component_unorm);
constexpr std::uint32_t float_uu = component_types(
    tic_component_float, tic_component_uint, tic_component_unorm,
    tic_component_unorm);

struct FormatEntry {
    std::uint32_t word;
    TextureFormat format;
};

// The guest texture format table: every admitted (format index,
// component types, sRGB) combination. Depth-stencil layouts map to the
// combined formats the guest table defines for them.
constexpr std::array<FormatEntry, 108> format_table{{
    {0x1d | unorm_all, TextureFormat::R8Unorm},
    {0x1d | snorm_all, TextureFormat::R8Snorm},
    {0x1d | uint_all, TextureFormat::R8Uint},
    {0x1d | sint_all, TextureFormat::R8Sint},
    {0x1b | float_all, TextureFormat::R16Float},
    {0x1b | unorm_all, TextureFormat::R16Unorm},
    {0x1b | snorm_all, TextureFormat::R16Snorm},
    {0x1b | uint_all, TextureFormat::R16Uint},
    {0x1b | sint_all, TextureFormat::R16Sint},
    {0x0f | float_all, TextureFormat::R32Float},
    {0x0f | uint_all, TextureFormat::R32Uint},
    {0x0f | sint_all, TextureFormat::R32Sint},
    {0x18 | unorm_all, TextureFormat::R8G8Unorm},
    {0x18 | snorm_all, TextureFormat::R8G8Snorm},
    {0x18 | uint_all, TextureFormat::R8G8Uint},
    {0x18 | sint_all, TextureFormat::R8G8Sint},
    {0x0c | float_all, TextureFormat::R16G16Float},
    {0x0c | unorm_all, TextureFormat::R16G16Unorm},
    {0x0c | snorm_all, TextureFormat::R16G16Snorm},
    {0x0c | uint_all, TextureFormat::R16G16Uint},
    {0x0c | sint_all, TextureFormat::R16G16Sint},
    {0x04 | float_all, TextureFormat::R32G32Float},
    {0x04 | uint_all, TextureFormat::R32G32Uint},
    {0x04 | sint_all, TextureFormat::R32G32Sint},
    {0x02 | float_all, TextureFormat::R32G32B32Float},
    {0x02 | uint_all, TextureFormat::R32G32B32Uint},
    {0x02 | sint_all, TextureFormat::R32G32B32Sint},
    {0x08 | unorm_all, TextureFormat::R8G8B8A8Unorm},
    {0x08 | snorm_all, TextureFormat::R8G8B8A8Snorm},
    {0x08 | uint_all, TextureFormat::R8G8B8A8Uint},
    {0x08 | sint_all, TextureFormat::R8G8B8A8Sint},
    {0x08 | unorm_srgb, TextureFormat::R8G8B8A8Srgb},
    {0x03 | float_all, TextureFormat::R16G16B16A16Float},
    {0x03 | unorm_all, TextureFormat::R16G16B16A16Unorm},
    {0x03 | snorm_all, TextureFormat::R16G16B16A16Snorm},
    {0x03 | uint_all, TextureFormat::R16G16B16A16Uint},
    {0x03 | sint_all, TextureFormat::R16G16B16A16Sint},
    {0x01 | float_all, TextureFormat::R32G32B32A32Float},
    {0x01 | uint_all, TextureFormat::R32G32B32A32Uint},
    {0x01 | sint_all, TextureFormat::R32G32B32A32Sint},
    {0x3a | unorm_all, TextureFormat::D16Unorm},
    {0x3a | unorm_uuu, TextureFormat::D16Unorm},
    {0x2f | float_uuu, TextureFormat::D32Float},
    {0x2f | float_all, TextureFormat::D32Float},
    {0x0e | uint_uuu, TextureFormat::D24UnormS8Uint},
    {0x29 | uint_uuu, TextureFormat::D24UnormS8Uint},
    {0x29 | component_types(tic_component_uint, tic_component_unorm,
                            tic_component_uint, tic_component_uint),
     TextureFormat::D24UnormS8Uint},
    {0x2a | unorm_uuu, TextureFormat::X8UintD24Unorm},
    {0x2b | unorm_uuu, TextureFormat::S8UintD24Unorm},
    {0x05 | float_uu, TextureFormat::D32FloatS8Uint},
    {0x30 | float_uu, TextureFormat::D32FloatS8Uint},
    {0x1e | unorm_all, TextureFormat::R4G4Unorm},
    {0x12 | unorm_all, TextureFormat::R4G4B4A4Unorm},
    {0x14 | unorm_all, TextureFormat::R5G5B5A1Unorm},
    {0x15 | unorm_all, TextureFormat::R5G6B5Unorm},
    {0x09 | unorm_all, TextureFormat::R10G10B10A2Unorm},
    {0x09 | uint_all, TextureFormat::R10G10B10A2Uint},
    {0x21 | float_all, TextureFormat::R11G11B10Float},
    {0x20 | float_all, TextureFormat::R9G9B9E5Float},
    {0x24 | unorm_all, TextureFormat::Bc1RgbaUnorm},
    {0x24 | unorm_srgb, TextureFormat::Bc1RgbaSrgb},
    {0x25 | unorm_all, TextureFormat::Bc2Unorm},
    {0x25 | unorm_srgb, TextureFormat::Bc2Srgb},
    {0x26 | unorm_all, TextureFormat::Bc3Unorm},
    {0x26 | unorm_srgb, TextureFormat::Bc3Srgb},
    {0x27 | unorm_all, TextureFormat::Bc4Unorm},
    {0x27 | snorm_all, TextureFormat::Bc4Snorm},
    {0x28 | unorm_all, TextureFormat::Bc5Unorm},
    {0x28 | snorm_all, TextureFormat::Bc5Snorm},
    {0x17 | unorm_all, TextureFormat::Bc7Unorm},
    {0x17 | unorm_srgb, TextureFormat::Bc7Srgb},
    {0x10 | float_all, TextureFormat::Bc6HSfloat},
    {0x11 | float_all, TextureFormat::Bc6HUfloat},
    {0x06 | unorm_all, TextureFormat::Etc2RgbUnorm},
    {0x06 | unorm_srgb, TextureFormat::Etc2RgbSrgb},
    {0x0a | unorm_all, TextureFormat::Etc2RgbPtaUnorm},
    {0x0a | unorm_srgb, TextureFormat::Etc2RgbPtaSrgb},
    {0x0b | unorm_all, TextureFormat::Etc2RgbaUnorm},
    {0x0b | unorm_srgb, TextureFormat::Etc2RgbaSrgb},
    {0x13 | unorm_all, TextureFormat::A5B5G5R1Unorm},
    {0x40 | unorm_all, TextureFormat::Astc4x4Unorm},
    {0x40 | unorm_srgb, TextureFormat::Astc4x4Srgb},
    {0x50 | unorm_all, TextureFormat::Astc5x4Unorm},
    {0x50 | unorm_srgb, TextureFormat::Astc5x4Srgb},
    {0x41 | unorm_all, TextureFormat::Astc5x5Unorm},
    {0x41 | unorm_srgb, TextureFormat::Astc5x5Srgb},
    {0x51 | unorm_all, TextureFormat::Astc6x5Unorm},
    {0x51 | unorm_srgb, TextureFormat::Astc6x5Srgb},
    {0x42 | unorm_all, TextureFormat::Astc6x6Unorm},
    {0x42 | unorm_srgb, TextureFormat::Astc6x6Srgb},
    {0x55 | unorm_all, TextureFormat::Astc8x5Unorm},
    {0x55 | unorm_srgb, TextureFormat::Astc8x5Srgb},
    {0x52 | unorm_all, TextureFormat::Astc8x6Unorm},
    {0x52 | unorm_srgb, TextureFormat::Astc8x6Srgb},
    {0x44 | unorm_all, TextureFormat::Astc8x8Unorm},
    {0x44 | unorm_srgb, TextureFormat::Astc8x8Srgb},
    {0x56 | unorm_all, TextureFormat::Astc10x5Unorm},
    {0x56 | unorm_srgb, TextureFormat::Astc10x5Srgb},
    {0x57 | unorm_all, TextureFormat::Astc10x6Unorm},
    {0x57 | unorm_srgb, TextureFormat::Astc10x6Srgb},
    {0x53 | unorm_all, TextureFormat::Astc10x8Unorm},
    {0x53 | unorm_srgb, TextureFormat::Astc10x8Srgb},
    {0x45 | unorm_all, TextureFormat::Astc10x10Unorm},
    {0x45 | unorm_srgb, TextureFormat::Astc10x10Srgb},
    {0x54 | unorm_all, TextureFormat::Astc12x10Unorm},
    {0x54 | unorm_srgb, TextureFormat::Astc12x10Srgb},
    {0x46 | unorm_all, TextureFormat::Astc12x12Unorm},
    {0x46 | unorm_srgb, TextureFormat::Astc12x12Srgb},
}};

}  // namespace

bool is_bc_family(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::Bc1RgbaUnorm:
        case TextureFormat::Bc1RgbaSrgb:
        case TextureFormat::Bc2Unorm:
        case TextureFormat::Bc2Srgb:
        case TextureFormat::Bc3Unorm:
        case TextureFormat::Bc3Srgb:
        case TextureFormat::Bc4Unorm:
        case TextureFormat::Bc4Snorm:
        case TextureFormat::Bc5Unorm:
        case TextureFormat::Bc5Snorm:
        case TextureFormat::Bc7Unorm:
        case TextureFormat::Bc7Srgb:
        case TextureFormat::Bc6HUfloat:
        case TextureFormat::Bc6HSfloat:
            return true;
        default:
            return false;
    }
}

bool lookup_texture_format(std::uint32_t encoded, bool srgb,
                           TextureFormat& out) noexcept {
    const std::uint32_t word =
        (encoded & (tic_srgb_flag - 1)) | (srgb ? tic_srgb_flag : 0u);
    for (const FormatEntry& entry : format_table) {
        if (entry.word == word) {
            out = entry.format;
            return true;
        }
    }
    return false;
}

TextureFormatInfo texture_format_info(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::R8Unorm:
        case TextureFormat::R8Snorm:
        case TextureFormat::R8Uint:
        case TextureFormat::R8Sint:
            return {1, 1, 1, 1, false, false};
        case TextureFormat::R4G4Unorm:
            return {1, 1, 1, 2, false, false};
        case TextureFormat::R16Unorm:
        case TextureFormat::R16Snorm:
        case TextureFormat::R16Uint:
        case TextureFormat::R16Sint:
        case TextureFormat::R16Float:
        case TextureFormat::D16Unorm:
            return {1, 1, 2, 1, format == TextureFormat::D16Unorm, false};
        case TextureFormat::R4G4B4A4Unorm:
        case TextureFormat::R5G5B5A1Unorm:
        case TextureFormat::A5B5G5R1Unorm:
            return {1, 1, 2, 4, false, false};
        case TextureFormat::R5G6B5Unorm:
            return {1, 1, 2, 3, false, false};
        case TextureFormat::R32Uint:
        case TextureFormat::R32Sint:
        case TextureFormat::R32Float:
        case TextureFormat::D32Float:
            return {1, 1, 4, 1, format == TextureFormat::D32Float, false};
        case TextureFormat::R8G8Unorm:
        case TextureFormat::R8G8Snorm:
        case TextureFormat::R8G8Uint:
        case TextureFormat::R8G8Sint:
            return {1, 1, 2, 2, false, false};
        case TextureFormat::R16G16Unorm:
        case TextureFormat::R16G16Snorm:
        case TextureFormat::R16G16Uint:
        case TextureFormat::R16G16Sint:
        case TextureFormat::R16G16Float:
            return {1, 1, 4, 2, false, false};
        case TextureFormat::R8G8B8A8Unorm:
        case TextureFormat::R8G8B8A8Snorm:
        case TextureFormat::R8G8B8A8Uint:
        case TextureFormat::R8G8B8A8Sint:
        case TextureFormat::R8G8B8A8Srgb:
        case TextureFormat::B8G8R8A8Unorm:
        case TextureFormat::B8G8R8A8Srgb:
        case TextureFormat::R10G10B10A2Unorm:
        case TextureFormat::R10G10B10A2Uint:
        case TextureFormat::R9G9B9E5Float:
            return {1, 1, 4, 4, false, false};
        case TextureFormat::R11G11B10Float:
            return {1, 1, 4, 3, false, false};
        case TextureFormat::D24UnormS8Uint:
        case TextureFormat::S8UintD24Unorm:
            return {1, 1, 4, 2, true, true};
        case TextureFormat::X8UintD24Unorm:
            return {1, 1, 4, 2, true, false};
        case TextureFormat::R32G32Uint:
        case TextureFormat::R32G32Sint:
        case TextureFormat::R32G32Float:
            return {1, 1, 8, 2, false, false};
        case TextureFormat::R16G16B16A16Unorm:
        case TextureFormat::R16G16B16A16Snorm:
        case TextureFormat::R16G16B16A16Uint:
        case TextureFormat::R16G16B16A16Sint:
        case TextureFormat::R16G16B16A16Float:
            return {1, 1, 8, 4, false, false};
        case TextureFormat::D32FloatS8Uint:
            return {1, 1, 8, 2, true, true};
        case TextureFormat::R32G32B32Uint:
        case TextureFormat::R32G32B32Sint:
        case TextureFormat::R32G32B32Float:
            return {1, 1, 12, 3, false, false};
        case TextureFormat::R32G32B32A32Uint:
        case TextureFormat::R32G32B32A32Sint:
        case TextureFormat::R32G32B32A32Float:
            return {1, 1, 16, 4, false, false};
        case TextureFormat::Bc1RgbaUnorm:
        case TextureFormat::Bc1RgbaSrgb:
            return {4, 4, 8, 4, false, false};
        case TextureFormat::Bc4Unorm:
        case TextureFormat::Bc4Snorm:
            return {4, 4, 8, 1, false, false};
        case TextureFormat::Bc2Unorm:
        case TextureFormat::Bc2Srgb:
        case TextureFormat::Bc3Unorm:
        case TextureFormat::Bc3Srgb:
        case TextureFormat::Bc7Unorm:
        case TextureFormat::Bc7Srgb:
        case TextureFormat::Bc6HUfloat:
        case TextureFormat::Bc6HSfloat:
            return {4, 4, 16, 4, false, false};
        case TextureFormat::Bc5Unorm:
        case TextureFormat::Bc5Snorm:
            return {4, 4, 16, 2, false, false};
        case TextureFormat::Etc2RgbUnorm:
        case TextureFormat::Etc2RgbSrgb:
            return {4, 4, 8, 3, false, false};
        case TextureFormat::Etc2RgbPtaUnorm:
        case TextureFormat::Etc2RgbPtaSrgb:
            return {4, 4, 8, 4, false, false};
        case TextureFormat::Etc2RgbaUnorm:
        case TextureFormat::Etc2RgbaSrgb:
            return {4, 4, 16, 4, false, false};
        case TextureFormat::Astc4x4Unorm:
        case TextureFormat::Astc4x4Srgb:
            return {4, 4, 16, 4, false, false};
        case TextureFormat::Astc5x4Unorm:
        case TextureFormat::Astc5x4Srgb:
            return {5, 4, 16, 4, false, false};
        case TextureFormat::Astc5x5Unorm:
        case TextureFormat::Astc5x5Srgb:
            return {5, 5, 16, 4, false, false};
        case TextureFormat::Astc6x5Unorm:
        case TextureFormat::Astc6x5Srgb:
            return {6, 5, 16, 4, false, false};
        case TextureFormat::Astc6x6Unorm:
        case TextureFormat::Astc6x6Srgb:
            return {6, 6, 16, 4, false, false};
        case TextureFormat::Astc8x5Unorm:
        case TextureFormat::Astc8x5Srgb:
            return {8, 5, 16, 4, false, false};
        case TextureFormat::Astc8x6Unorm:
        case TextureFormat::Astc8x6Srgb:
            return {8, 6, 16, 4, false, false};
        case TextureFormat::Astc8x8Unorm:
        case TextureFormat::Astc8x8Srgb:
            return {8, 8, 16, 4, false, false};
        case TextureFormat::Astc10x5Unorm:
        case TextureFormat::Astc10x5Srgb:
            return {10, 5, 16, 4, false, false};
        case TextureFormat::Astc10x6Unorm:
        case TextureFormat::Astc10x6Srgb:
            return {10, 6, 16, 4, false, false};
        case TextureFormat::Astc10x8Unorm:
        case TextureFormat::Astc10x8Srgb:
            return {10, 8, 16, 4, false, false};
        case TextureFormat::Astc10x10Unorm:
        case TextureFormat::Astc10x10Srgb:
            return {10, 10, 16, 4, false, false};
        case TextureFormat::Astc12x10Unorm:
        case TextureFormat::Astc12x10Srgb:
            return {12, 10, 16, 4, false, false};
        case TextureFormat::Astc12x12Unorm:
        case TextureFormat::Astc12x12Srgb:
            return {12, 12, 16, 4, false, false};
    }
    return {};
}

TextureNumericClass texture_numeric_class(TextureFormat format) noexcept {
    switch (format) {
    case TextureFormat::R8Uint:
    case TextureFormat::R16Uint:
    case TextureFormat::R32Uint:
    case TextureFormat::R8G8Uint:
    case TextureFormat::R16G16Uint:
    case TextureFormat::R32G32Uint:
    case TextureFormat::R32G32B32Uint:
    case TextureFormat::R8G8B8A8Uint:
    case TextureFormat::R16G16B16A16Uint:
    case TextureFormat::R32G32B32A32Uint:
    case TextureFormat::R10G10B10A2Uint:
        return TextureNumericClass::Uint;
    case TextureFormat::R8Sint:
    case TextureFormat::R16Sint:
    case TextureFormat::R32Sint:
    case TextureFormat::R8G8Sint:
    case TextureFormat::R16G16Sint:
    case TextureFormat::R32G32Sint:
    case TextureFormat::R32G32B32Sint:
    case TextureFormat::R8G8B8A8Sint:
    case TextureFormat::R16G16B16A16Sint:
    case TextureFormat::R32G32B32A32Sint:
        return TextureNumericClass::Sint;
    default: return TextureNumericClass::Float;
    }
}

}  // namespace armigo::graphics
