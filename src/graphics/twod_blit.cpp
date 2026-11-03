// ============================================================================
// Armigo Graphics - 2D Blit State Decode
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The 2D engine decodes the PixelsFromMemorySrcY0Int trigger, fixed-point
// source rectangle arithmetic, corner origin, linear-pitch wrapping, guest
// formats, the G80_2D register layout, surfaces, and blit configuration.

#include <armigo/graphics/twod_blit.h>

namespace armigo::graphics {
namespace {

// G80_2D register indices (byte offsets divided by four; the register file is
// the one engines.cpp captures for engine 0x902d). The SetDst*/SetSrc* surface
// blocks share one field layout.
struct SurfaceRegisters {
    std::size_t format;
    std::size_t memory_layout;
    std::size_t block_size;
    std::size_t depth;
    std::size_t layer;
    std::size_t pitch;
    std::size_t width;
    std::size_t height;
    std::size_t offset_upper;
    std::size_t offset_lower;
};
constexpr SurfaceRegisters twod_dst_registers{0x80, 0x81, 0x82, 0x83, 0x84,
                                              0x85, 0x86, 0x87, 0x88, 0x89};
constexpr SurfaceRegisters twod_src_registers{0x8c, 0x8d, 0x8e, 0x8f, 0x90,
                                              0x91, 0x92, 0x93, 0x94, 0x95};
constexpr std::size_t twod_clip_x0 = 0xa0;
constexpr std::size_t twod_clip_y0 = 0xa1;
constexpr std::size_t twod_clip_width = 0xa2;
constexpr std::size_t twod_clip_height = 0xa3;
constexpr std::size_t twod_clip_enable = 0xa4;
constexpr std::size_t twod_operation = 0xab;
constexpr std::size_t twod_direction = 0xb8;
constexpr std::size_t twod_sample_mode = 0x223;
constexpr std::size_t twod_dst_x0 = 0x22c;
constexpr std::size_t twod_dst_y0 = 0x22d;
constexpr std::size_t twod_dst_rect_width = 0x22e;
constexpr std::size_t twod_dst_rect_height = 0x22f;
constexpr std::size_t twod_du_dx_frac = 0x230;
constexpr std::size_t twod_du_dx_int = 0x231;
constexpr std::size_t twod_dv_dy_frac = 0x232;
constexpr std::size_t twod_dv_dy_int = 0x233;
constexpr std::size_t twod_src_x0_frac = 0x234;
constexpr std::size_t twod_src_x0_int = 0x235;
constexpr std::size_t twod_src_y0_frac = 0x236;
constexpr std::size_t twod_src_y0_int = 0x237;

TwodSurface decode_surface(const std::array<std::uint32_t, 0x1000>& registers,
                           const SurfaceRegisters& base) noexcept
{
    TwodSurface surface;
    surface.format = registers[base.format] & 0xff;
    surface.pitch = (registers[base.memory_layout] & 0x1) != 0;
    surface.block_height_log2 =
        static_cast<std::uint8_t>((registers[base.block_size] >> 4) & 0x7);
    surface.block_depth_log2 =
        static_cast<std::uint8_t>((registers[base.block_size] >> 8) & 0x7);
    surface.depth = registers[base.depth];
    surface.layer = registers[base.layer];
    surface.pitch_bytes = registers[base.pitch];
    surface.width = registers[base.width];
    surface.height = registers[base.height];
    // The upper offset has eight valid bits.
    surface.address = (std::uint64_t{registers[base.offset_upper] & 0xff} << 32) |
                      registers[base.offset_lower];
    return surface;
}

// Joins a 32.32 fixed-point pair. The fraction word is zero extended; the
// pair is read as an unsigned 64-bit value and reinterpreted signed.
constexpr std::int64_t fixed_point(std::uint32_t integer,
                                   std::uint32_t fraction) noexcept
{
    return static_cast<std::int64_t>(
        (std::uint64_t{integer} << 32) | std::uint64_t{fraction});
}

}  // namespace

bool decode_twod_format(std::uint32_t encoded, TextureFormat& out) noexcept
{
    // Guest color-format encodings shared with the render-target registers.
    // Channel reversed (BGR) encodings have no table entry and stay rejections.
    switch (encoded) {
        case 0xc0: out = TextureFormat::R32G32B32A32Float; return true;
        case 0xc1: out = TextureFormat::R32G32B32A32Sint; return true;
        case 0xc2: out = TextureFormat::R32G32B32A32Uint; return true;
        case 0xc3: out = TextureFormat::R32G32B32A32Float; return true;
        case 0xc4: out = TextureFormat::R32G32B32A32Sint; return true;
        case 0xc5: out = TextureFormat::R32G32B32A32Uint; return true;
        case 0xc6: out = TextureFormat::R16G16B16A16Unorm; return true;
        case 0xc7: out = TextureFormat::R16G16B16A16Snorm; return true;
        case 0xc8: out = TextureFormat::R16G16B16A16Sint; return true;
        case 0xc9: out = TextureFormat::R16G16B16A16Uint; return true;
        case 0xca: out = TextureFormat::R16G16B16A16Float; return true;
        case 0xcb: out = TextureFormat::R32G32Float; return true;
        case 0xcc: out = TextureFormat::R32G32Sint; return true;
        case 0xcd: out = TextureFormat::R32G32Uint; return true;
        case 0xce: out = TextureFormat::R16G16B16A16Float; return true;
        case 0xd1: out = TextureFormat::R10G10B10A2Unorm; return true;
        case 0xd2: out = TextureFormat::R10G10B10A2Uint; return true;
        case 0xd5: out = TextureFormat::R8G8B8A8Unorm; return true;
        case 0xd6: out = TextureFormat::R8G8B8A8Srgb; return true;
        case 0xd7: out = TextureFormat::R8G8B8A8Snorm; return true;
        case 0xd8: out = TextureFormat::R8G8B8A8Sint; return true;
        case 0xd9: out = TextureFormat::R8G8B8A8Uint; return true;
        case 0xda: out = TextureFormat::R16G16Unorm; return true;
        case 0xdb: out = TextureFormat::R16G16Snorm; return true;
        case 0xdc: out = TextureFormat::R16G16Sint; return true;
        case 0xdd: out = TextureFormat::R16G16Uint; return true;
        case 0xde: out = TextureFormat::R16G16Float; return true;
        case 0xe0: out = TextureFormat::R11G11B10Float; return true;
        case 0xe3: out = TextureFormat::R32Sint; return true;
        case 0xe4: out = TextureFormat::R32Uint; return true;
        case 0xe5: out = TextureFormat::R32Float; return true;
        case 0xe8: out = TextureFormat::R5G6B5Unorm; return true;
        case 0xea: out = TextureFormat::R8G8Unorm; return true;
        case 0xeb: out = TextureFormat::R8G8Snorm; return true;
        case 0xec: out = TextureFormat::R8G8Sint; return true;
        case 0xed: out = TextureFormat::R8G8Uint; return true;
        case 0xee: out = TextureFormat::R16Unorm; return true;
        case 0xef: out = TextureFormat::R16Snorm; return true;
        case 0xf0: out = TextureFormat::R16Sint; return true;
        case 0xf1: out = TextureFormat::R16Uint; return true;
        case 0xf2: out = TextureFormat::R16Float; return true;
        case 0xf3: out = TextureFormat::R8Unorm; return true;
        case 0xf4: out = TextureFormat::R8Snorm; return true;
        case 0xf5: out = TextureFormat::R8Sint; return true;
        case 0xf6: out = TextureFormat::R8Uint; return true;
        case 0xf9: out = TextureFormat::R8G8B8A8Unorm; return true;
        case 0xfa: out = TextureFormat::R8G8B8A8Srgb; return true;
        default: return false;
    }
}

TwodBlitResult decode_twod_blit(
    const std::array<std::uint32_t, 0x1000>& registers) noexcept
{
    const std::uint32_t operation = registers[twod_operation] & 0x7;
    if (operation > static_cast<std::uint32_t>(TwodOperation::BlendPremult)) {
        return CommandResult{CommandStatus::Unsupported,
                             "2D blit operation encoding is reserved"};
    }
    const std::uint32_t direction = registers[twod_direction];
    // Two two-bit fields: horizontal in bits 0-1, vertical in bits 4-5. Values
    // 0 (hardware decides) and 1 run in the default reading order, 2 reverses
    // it, and 3 plus any other set bit is a reserved encoding.
    if ((direction & ~0x13u) != 0 || (direction & 0x3) == 0x3 ||
        ((direction >> 4) & 0x3) == 0x3) {
        return CommandResult{CommandStatus::Unsupported,
                             "2D blit direction encoding is reserved"};
    }

    TwodBlit blit;
    blit.source = decode_surface(registers, twod_src_registers);
    blit.destination = decode_surface(registers, twod_dst_registers);
    blit.operation = static_cast<TwodOperation>(operation);
    blit.corner_origin = (registers[twod_sample_mode] & 0x1) != 0;
    blit.linear_filter = ((registers[twod_sample_mode] >> 4) & 0x1) != 0;
    blit.flip_x = (direction & 0x3) == 0x2;
    blit.flip_y = ((direction >> 4) & 0x3) == 0x2;
    blit.clip_x0 = registers[twod_clip_x0];
    blit.clip_y0 = registers[twod_clip_y0];
    blit.clip_width = registers[twod_clip_width];
    blit.clip_height = registers[twod_clip_height];
    blit.clip_enable = (registers[twod_clip_enable] & 0x1) != 0;
    blit.destination_x1 = static_cast<std::int32_t>(registers[twod_dst_x0]);
    blit.destination_y1 = static_cast<std::int32_t>(registers[twod_dst_y0]);
    blit.destination_width = registers[twod_dst_rect_width];
    blit.destination_height = registers[twod_dst_rect_height];

    // Fixed-point source origin and steps (TwodClass.PixelsFromMemorySrcY0Int).
    std::int64_t source_x =
        fixed_point(registers[twod_src_x0_int], registers[twod_src_x0_frac]);
    std::int64_t source_y =
        fixed_point(registers[twod_src_y0_int], registers[twod_src_y0_frac]);
    const std::int64_t du_dx =
        fixed_point(registers[twod_du_dx_int], registers[twod_du_dx_frac]);
    const std::int64_t dv_dy =
        fixed_point(registers[twod_dv_dy_int], registers[twod_dv_dy_frac]);
    if (blit.corner_origin) {
        // Corner origins arrive pre-centered by the guest by FactorXY / 2.0;
        // removing the whole part of half each step restores the true region
        // without rounding drift.
        source_x -= (du_dx >> 33) << 32;
        source_y -= (dv_dy >> 33) << 32;
    }
    blit.source_x1 = static_cast<std::int32_t>(source_x >> 32);
    blit.source_y1 = static_cast<std::int32_t>(source_y >> 32);
    // Round the step product up by adding 2^32-1 before the shift. Unsigned
    // arithmetic preserves the guest's unchecked wrap.
    const auto step_extent = [](std::int64_t step, std::uint32_t count) {
        const std::uint64_t product =
            static_cast<std::uint64_t>(step) * std::uint64_t{count};
        return static_cast<std::int32_t>((product + 0xffffffffu) >> 32);
    };
    blit.source_x2 = blit.source_x1 + step_extent(du_dx, blit.destination_width);
    blit.source_y2 = blit.source_y1 + step_extent(dv_dy, blit.destination_height);

    // Out-of-bounds linear copy: a copy extending past the encoded width of a
    // pitch-linear surface wraps to the next line, so skipping the first-line
    // gap by offsetting the source address keeps the copy inside the surface.
    TextureFormat source_format{};
    if (blit.source.pitch && blit.destination_width != 0 &&
        decode_twod_format(blit.source.format, source_format)) {
        const std::uint32_t bytes_per_pixel =
            texture_format_info(source_format).bytes_per_block;
        const std::uint32_t width_aligned =
            bytes_per_pixel != 0 ? blit.source.pitch_bytes / bytes_per_pixel : 0;
        if (std::uint32_t(blit.source_x2) == blit.source.width &&
            blit.source_x2 > 0 && std::uint32_t(blit.source_x2) > width_aligned &&
            blit.source_x1 > 0) {
            blit.source_address_offset =
                std::uint64_t{std::uint32_t(blit.source_x1)} * bytes_per_pixel;
            blit.source.address += blit.source_address_offset;
            blit.source.width -= std::uint32_t(blit.source_x1);
            blit.source_x2 -= blit.source_x1;
            blit.source_x1 = 0;
        }
    }
    return blit;
}

}  // namespace armigo::graphics
