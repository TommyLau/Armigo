// ============================================================================
// Armigo Graphics - GPU Texture Descriptors
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

namespace armigo::graphics {

// One entry of the guest texture format table: the guest-side identity of
// an admitted texture encoding. Enum values carry no wire meaning; the TIC
// lookup in this header is the only producer.
enum class TextureFormat : std::uint8_t {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Float,
    R32Uint,
    R32Sint,
    R32Float,
    R8G8Unorm,
    R8G8Snorm,
    R8G8Uint,
    R8G8Sint,
    R16G16Unorm,
    R16G16Snorm,
    R16G16Uint,
    R16G16Sint,
    R16G16Float,
    R32G32Uint,
    R32G32Sint,
    R32G32Float,
    R32G32B32Uint,
    R32G32B32Sint,
    R32G32B32Float,
    R4G4Unorm,
    R4G4B4A4Unorm,
    R5G5B5A1Unorm,
    A5B5G5R1Unorm,
    R5G6B5Unorm,
    R10G10B10A2Unorm,
    R10G10B10A2Uint,
    R11G11B10Float,
    R9G9B9E5Float,
    R8G8B8A8Unorm,
    R8G8B8A8Snorm,
    R8G8B8A8Uint,
    R8G8B8A8Sint,
    R8G8B8A8Srgb,
    R16G16B16A16Unorm,
    R16G16B16A16Snorm,
    R16G16B16A16Uint,
    R16G16B16A16Sint,
    R16G16B16A16Float,
    R32G32B32A32Uint,
    R32G32B32A32Sint,
    R32G32B32A32Float,
    D16Unorm,
    D32Float,
    D24UnormS8Uint,
    X8UintD24Unorm,
    S8UintD24Unorm,
    D32FloatS8Uint,
    Bc1RgbaUnorm,
    Bc1RgbaSrgb,
    Bc2Unorm,
    Bc2Srgb,
    Bc3Unorm,
    Bc3Srgb,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc7Unorm,
    Bc7Srgb,
    Bc6HUfloat,
    Bc6HSfloat,
    Etc2RgbUnorm,
    Etc2RgbSrgb,
    Etc2RgbPtaUnorm,
    Etc2RgbPtaSrgb,
    Etc2RgbaUnorm,
    Etc2RgbaSrgb,
    Astc4x4Unorm,
    Astc4x4Srgb,
    Astc5x4Unorm,
    Astc5x4Srgb,
    Astc5x5Unorm,
    Astc5x5Srgb,
    Astc6x5Unorm,
    Astc6x5Srgb,
    Astc6x6Unorm,
    Astc6x6Srgb,
    Astc8x5Unorm,
    Astc8x5Srgb,
    Astc8x6Unorm,
    Astc8x6Srgb,
    Astc8x8Unorm,
    Astc8x8Srgb,
    Astc10x5Unorm,
    Astc10x5Srgb,
    Astc10x6Unorm,
    Astc10x6Srgb,
    Astc10x8Unorm,
    Astc10x8Srgb,
    Astc10x10Unorm,
    Astc10x10Srgb,
    Astc12x10Unorm,
    Astc12x10Srgb,
    Astc12x12Unorm,
    Astc12x12Srgb,
    // Bytes B, G, R, A in memory (NVN BGRA8 / BGRX8); appended so earlier
    // values stay stable.
    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
};

// Shader-visible color class: normalized and floating formats use float values.
// Integer clear words must retain their signedness and all 32 bits.
enum class TextureNumericClass : std::uint8_t { Float, Uint, Sint };
[[nodiscard]] TextureNumericClass texture_numeric_class(TextureFormat format) noexcept;

// Guest storage geometry and sampling shape of one format. Block extents are
// one for uncompressed formats and bytes_per_block counts one block.
struct TextureFormatInfo {
    std::uint8_t block_width = 1;
    std::uint8_t block_height = 1;
    std::uint8_t bytes_per_block = 0;
    std::uint8_t components = 0;
    bool depth = false;
    bool stencil = false;
};

// True when a texel spans a width x height block instead of one pixel.
[[nodiscard]] constexpr bool is_block_compressed(
    const TextureFormatInfo& info) noexcept {
    return (info.block_width | info.block_height) != 1;
}

// True for the BC1-7 family, whose Metal sampling requires the device BC
// capability rather than a per-format capability query.
[[nodiscard]] bool is_bc_family(TextureFormat format) noexcept;

// TIC component type encodings from word 0, bits 7 to 18 (three bits per
// component in RGBA order).
inline constexpr std::uint32_t tic_component_snorm = 1;
inline constexpr std::uint32_t tic_component_unorm = 2;
inline constexpr std::uint32_t tic_component_sint = 3;
inline constexpr std::uint32_t tic_component_uint = 4;
inline constexpr std::uint32_t tic_component_snorm_force_fp16 = 5;
inline constexpr std::uint32_t tic_component_unorm_force_fp16 = 6;
inline constexpr std::uint32_t tic_component_float = 7;
inline constexpr std::uint32_t tic_srgb_flag = 1u << 19;

// Resolves the low 19 bits of a TIC format word (format index plus component
// types) plus the word-4 sRGB flag to a table format. Returns false when the
// combination has no table entry; the caller chooses the structured error.
[[nodiscard]] bool lookup_texture_format(std::uint32_t encoded, bool srgb,
                                         TextureFormat& out) noexcept;

// Guest storage geometry of one table format. Every table entry has an info.
[[nodiscard]] TextureFormatInfo texture_format_info(
    TextureFormat format) noexcept;

}  // namespace armigo::graphics
