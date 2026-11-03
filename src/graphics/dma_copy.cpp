// ============================================================================
// Armigo Graphics - DMA Copy Engine
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// DMA methods decode guest surfaces and block-linear offsets. Every transfer
// executes through guest memory on the CPU with each pixel addressed at its
// first byte and components stored contiguously.

#include <armigo/graphics/dma_copy.h>

#include <algorithm>
#include <bit>
#include <cstring>

namespace armigo::graphics {
namespace {

constexpr std::uint32_t multi_line_enable = 1u << 9;
constexpr std::uint32_t remap_enable = 1u << 10;
constexpr std::uint32_t src_pitch = 1u << 7;
constexpr std::uint32_t dst_pitch = 1u << 8;
constexpr std::int64_t gob_stride = 64;
constexpr std::int64_t gob_height = 8;
constexpr std::int64_t gob_size = gob_stride * gob_height;

// Remap destination component selectors.
enum class Remap : std::uint32_t { src_x, src_y, src_z, src_w, const_a, const_b, no_write };

std::uint64_t address(std::uint32_t upper, std::uint32_t lower) noexcept {
    return (std::uint64_t{upper & 0xffu} << 32) | lower;
}

// Byte offset of a pixel in one pitch or block-linear surface (layer 0).
struct Surface {
    bool linear = true;
    std::int64_t stride = 0;
    std::int64_t bpp = 1;
    std::int64_t origin_x = 0;
    std::int64_t origin_y = 0;
    std::int64_t width = 0;
    std::int64_t height = 0;
    // Block-linear addressing (BlockLinearLayout).
    std::int64_t rob_size = 0;
    std::int64_t height_mask = 0;
    std::uint32_t height_shift = 0;
    std::uint32_t x_shift = 0;

    std::int64_t offset(std::int64_t x, std::int64_t y) const noexcept {
        if (linear) return y * stride + x * bpp;
        const std::int64_t x_bytes = x * bpp;
        const std::int64_t gob_row = y / gob_height;
        std::int64_t value = (gob_row >> height_shift) * rob_size;
        value += (x_bytes / gob_stride) << x_shift;
        value += (gob_row & height_mask) * gob_size;
        value += ((x_bytes & 0x3f) >> 5) << 8;
        value += ((y & 0x07) >> 1) << 6;
        value += ((x_bytes & 0x1f) >> 4) << 5;
        value += (y & 0x01) << 4;
        value += x_bytes & 0x0f;
        return value;
    }
};

// Block size register then width, height, depth, layer and origin.
Surface surface(std::span<const std::uint32_t> r, std::uint32_t block, bool linear,
                std::uint32_t pitch, std::int64_t bpp) noexcept {
    Surface s;
    s.linear = linear;
    s.stride = static_cast<std::int32_t>(r[pitch]);
    s.bpp = bpp;
    if (linear) return s;
    s.width = r[block + 1];
    s.height = r[block + 2];
    s.origin_x = r[block + 5] & 0xffffu;
    s.origin_y = r[block + 5] >> 16;
    const std::uint32_t blocks_y_log2 = (r[block] >> 4) & 0xfu;
    const std::uint32_t blocks_z_log2 = (r[block] >> 8) & 0xfu;
    const std::int64_t blocks_y = std::int64_t{1} << blocks_y_log2;
    const std::int64_t blocks_z = std::int64_t{1} << blocks_z_log2;
    // The width is aligned to whole GOBs of pixels (OffsetCalculator).
    const std::int64_t width_aligned = (s.width + gob_stride / bpp - 1) / (gob_stride / bpp) * (gob_stride / bpp);
    const std::int64_t width_in_gobs = (width_aligned * bpp + gob_stride - 1) / gob_stride;
    s.rob_size = gob_size * blocks_y * blocks_z * width_in_gobs;
    s.height_mask = blocks_y - 1;
    s.height_shift = blocks_y_log2;
    s.x_shift = static_cast<std::uint32_t>(std::countr_zero(static_cast<std::uint64_t>(gob_size * blocks_y * blocks_z)));
    return s;
}

struct Geometry {
    Surface src;
    Surface dst;
    std::int64_t x_count = 0;
    std::int64_t y_count = 0;
    std::int64_t src_start = 0;
    std::int64_t dst_start = 0;
};

// Each offset component grows with x and with y separately, so the rectangle's
// lowest and highest bytes are at its corners (BlockLinearLayout.GetRectangleRange).
std::pair<std::int64_t, std::int64_t> range(const Surface& s, std::int64_t x_count,
                                            std::int64_t y_count) noexcept {
    const std::int64_t x0 = s.origin_x, y0 = s.origin_y;
    const std::int64_t x1 = x0 + x_count - 1, y1 = y0 + y_count - 1;
    const std::int64_t start = std::min(s.offset(x0, y0), s.offset(x0, y1));
    const std::int64_t end = std::max(s.offset(x1, y0), s.offset(x1, y1)) + s.bpp;
    return {start, end};
}

std::variant<Geometry, DmaCopyError> geometry(std::span<const std::uint32_t> r, std::uint32_t launch) noexcept {
    const bool remap = (launch & remap_enable) != 0;
    const std::int64_t component_size = ((r[dma_method::remap_components] >> 16) & 3u) + 1;
    const std::int64_t src_components = ((r[dma_method::remap_components] >> 20) & 3u) + 1;
    const std::int64_t dst_components = ((r[dma_method::remap_components] >> 24) & 3u) + 1;
    const std::int64_t src_bpp = remap ? src_components * component_size : 1;
    const std::int64_t dst_bpp = remap ? dst_components * component_size : 1;
    Geometry g;
    g.src = surface(r, dma_method::src_block_size, (launch & src_pitch) != 0, dma_method::pitch_in, src_bpp);
    g.dst = surface(r, dma_method::dst_block_size, (launch & dst_pitch) != 0, dma_method::pitch_out, dst_bpp);
    for (const auto* s : {&g.src, &g.dst}) {
        const auto block = s == &g.src ? dma_method::src_block_size : dma_method::dst_block_size;
        if (!s->linear && (r[block + 4] != 0 || (r[block] >> 8 & 0xfu) != 0))
            return DmaCopyError{"DMA copy of a layered or 3D block-linear surface is not implemented"};
    }
    g.x_count = r[dma_method::line_length_in];
    g.y_count = r[dma_method::line_count];
    return g;
}

template <std::size_t Size>
void copy_component(std::byte* dst, const std::byte* src) noexcept {
    std::memcpy(dst, src, Size);
}

} // namespace

std::variant<DmaCopyPlan, DmaCopyError> plan_dma_copy(std::span<const std::uint32_t> r,
                                                       std::uint32_t launch) noexcept {
    if (r.size() < dma_method::count) return DmaCopyError{"DMA copy register file is truncated"};
    DmaCopyPlan plan;
    plan.semaphore = (launch >> 3) & 3u;
    if (plan.semaphore == 3) return DmaCopyError{"DMA copy semaphore type is reserved"};
    const std::uint64_t size = r[dma_method::line_length_in];
    if (size == 0) return plan;
    const std::uint64_t source = address(r[dma_method::offset_in_upper], r[dma_method::offset_in_lower]);
    const std::uint64_t destination = address(r[dma_method::offset_out_upper], r[dma_method::offset_out_lower]);
    if (!(launch & multi_line_enable)) {
        const auto remap = r[dma_method::remap_components];
        if (launch & remap_enable) {
            // The constant fill: one four-byte component of constant A.
            if ((remap & 0xffffu) != 0x4444u || ((remap >> 16) & 3u) != 3 || ((remap >> 20) & 3u) != 0 ||
                ((remap >> 24) & 3u) != 0)
                return DmaCopyError{"DMA 1D copy with a component remap is not implemented"};
            plan.destination = destination;
            plan.destination_size = size * 4;
            plan.destination_covered = true;
            return plan;
        }
        if (!(launch & src_pitch) || !(launch & dst_pitch))
            return DmaCopyError{"DMA 1D copy of block-linear storage is not implemented"};
        plan.source = source;
        plan.source_size = size;
        plan.destination = destination;
        plan.destination_size = size;
        plan.destination_covered = true;
        return plan;
    }
    if (r[dma_method::line_count] == 0) return plan;
    auto planned = geometry(r, launch);
    if (const auto* error = std::get_if<DmaCopyError>(&planned)) return *error;
    const auto& g = std::get<Geometry>(planned);
    const auto [src_begin, src_end] = range(g.src, g.x_count, g.y_count);
    const auto [dst_begin, dst_end] = range(g.dst, g.x_count, g.y_count);
    plan.source = source + static_cast<std::uint64_t>(src_begin);
    plan.source_size = static_cast<std::uint64_t>(src_end - src_begin);
    plan.destination = destination + static_cast<std::uint64_t>(dst_begin);
    plan.destination_size = static_cast<std::uint64_t>(dst_end - dst_begin);
    // Only a tightly packed pitch destination is written in full.
    plan.destination_covered = g.dst.linear && (g.y_count == 1 || g.dst.stride == g.x_count * g.dst.bpp);
    return plan;
}

void execute_dma_copy(std::span<const std::uint32_t> r, std::uint32_t launch, const DmaCopyPlan& plan,
                      std::span<const std::byte> source, std::span<std::byte> destination) noexcept {
    if (plan.destination_size == 0) return;
    if (!(launch & multi_line_enable)) {
        if (launch & remap_enable) {
            const std::uint32_t value = r[dma_method::remap_const_a];
            for (std::size_t i = 0; i + 4 <= destination.size(); i += 4)
                std::memcpy(destination.data() + i, &value, 4);
            return;
        }
        std::memcpy(destination.data(), source.data(), std::min(source.size(), destination.size()));
        return;
    }
    auto planned = geometry(r, launch);
    if (std::holds_alternative<DmaCopyError>(planned)) return;
    auto& g = std::get<Geometry>(planned);
    const auto src_begin = range(g.src, g.x_count, g.y_count).first;
    const auto dst_begin = range(g.dst, g.x_count, g.y_count).first;
    const bool remap = (launch & remap_enable) != 0;
    const auto components = r[dma_method::remap_components];
    const std::int64_t component_size = ((components >> 16) & 3u) + 1;
    const std::int64_t dst_components = ((components >> 24) & 3u) + 1;
    bool identity = !remap;
    if (remap) {
        identity = true;
        for (std::int64_t i = 0; i < dst_components; ++i)
            identity = identity && ((components >> (4 * i)) & 7u) == static_cast<std::uint32_t>(i);
    }
    const auto in_bounds = [](std::span<const std::byte> span, std::int64_t offset, std::int64_t size) {
        return offset >= 0 && static_cast<std::uint64_t>(offset + size) <= span.size();
    };
    for (std::int64_t y = 0; y < g.y_count; ++y) {
        for (std::int64_t x = 0; x < g.x_count; ++x) {
            const std::int64_t so = g.src.offset(g.src.origin_x + x, g.src.origin_y + y) - src_begin;
            const std::int64_t d = g.dst.offset(g.dst.origin_x + x, g.dst.origin_y + y) - dst_begin;
            if (!in_bounds(destination, d, g.dst.bpp)) continue;
            if (identity) {
                if (in_bounds(source, so, g.dst.bpp))
                    std::memcpy(destination.data() + d, source.data() + so, static_cast<std::size_t>(g.dst.bpp));
                continue;
            }
            // CopyShuffle: each destination component takes a source
            // component or a constant, or keeps its bytes.
            for (std::int64_t i = 0; i < dst_components; ++i) {
                const auto select = static_cast<Remap>((components >> (4 * i)) & 7u);
                std::byte* out = destination.data() + d + i * component_size;
                if (select <= Remap::src_w) {
                    const std::int64_t from = so + static_cast<std::int64_t>(select) * component_size;
                    if (in_bounds(source, from, component_size))
                        std::memcpy(out, source.data() + from, static_cast<std::size_t>(component_size));
                } else if (select == Remap::const_a || select == Remap::const_b) {
                    const std::uint32_t value = r[select == Remap::const_a ? dma_method::remap_const_a
                                                                          : dma_method::remap_const_b];
                    std::memcpy(out, &value, static_cast<std::size_t>(component_size));
                }
            }
        }
    }
}

} // namespace armigo::graphics
