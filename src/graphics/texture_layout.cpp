// ============================================================================
// Armigo Graphics - GPU Texture Layout
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture layout uses a scalar, bounds-checked implementation: blocks move one
// at a time through validated byte offsets rather than unchecked fast paths.

#include <armigo/graphics/texture_layout.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <tuple>

namespace armigo::graphics {
namespace {

constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();

TextureLayoutError failure(TextureLayoutErrorCode code, std::uint64_t value = 0,
                           std::uint64_t expected = 0) noexcept {
    return {code, value, expected};
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t& result) noexcept {
    if (right > size_max - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) noexcept {
    if (left != 0 && right > size_max / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment,
              std::size_t& result) noexcept {
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    return checked_add(value, alignment - remainder, result);
}

std::size_t div_round_up(std::size_t value, std::size_t divisor) noexcept {
    return (value + divisor - 1) / divisor;
}

bool bytes_per_block_admitted(std::size_t bytes_per_block) noexcept {
    switch (bytes_per_block) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 12:
        case 16:
            return true;
        default:
            return false;
    }
}

// Per-level geometry shared by size calculation and conversion: the shrunk
// GOB block sizes, the block-unit extents, and the level's column alignment.
// Computes the per-level prologue shared by size calculation and conversion.
struct LevelGeometry {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t block_columns = 0;
    std::uint32_t block_rows = 0;
    std::uint32_t block_slices = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
    std::uint32_t column_alignment = 1;
};

LevelGeometry level_geometry(const TextureLayoutDescription& layout,
                             std::uint32_t level) noexcept {
    const std::size_t gob_block_height =
        std::size_t{1} << layout.gob_block_height_log2;
    const std::size_t gob_block_depth =
        std::size_t{1} << layout.gob_block_depth_log2;
    const std::size_t tile_columns =
        std::size_t{1} << layout.gob_block_width_log2;
    // Comparison extents stay in the encoded (unshrunk) sizes.
    const std::size_t gob_width_blocks =
        (texture_gob_width_bytes / layout.bytes_per_block) * tile_columns;
    const std::size_t gob_height_rows =
        gob_block_height * texture_gob_height_rows;

    LevelGeometry geometry;
    geometry.width = std::max<std::uint32_t>(1, layout.width >> level);
    geometry.height = std::max<std::uint32_t>(1, layout.height >> level);
    geometry.depth = std::max<std::uint32_t>(1, layout.depth >> level);
    geometry.block_columns = div_round_up(geometry.width, layout.block_width);
    geometry.block_rows = div_round_up(geometry.height, layout.block_height);
    geometry.block_slices = geometry.depth;

    std::size_t block_height = gob_block_height;
    std::size_t block_depth = gob_block_depth;
    while (geometry.block_rows <=
               (block_height >> 1) * texture_gob_height_rows &&
           block_height != 1) {
        block_height >>= 1;
    }
    // Carry the depth shrink across the walk: each level past the first may halve the
    // block depth once more, judged against that level's slice count.
    for (std::uint32_t past = 1; past <= level; ++past) {
        const std::uint32_t past_slices =
            std::max<std::uint32_t>(1, layout.depth >> past);
        if (past_slices <= (block_depth >> 1) && block_depth != 1) {
            block_depth >>= 1;
        }
    }
    geometry.gob_block_height_log2 =
        static_cast<std::uint32_t>(std::countr_zero(block_height));
    geometry.gob_block_depth_log2 =
        static_cast<std::uint32_t>(std::countr_zero(block_depth));

    // Use one GOB column unless the level is large along every axis.
    if (geometry.block_slices >= gob_block_depth &&
        geometry.block_columns > gob_width_blocks &&
        geometry.block_rows > gob_height_rows) {
        geometry.column_alignment = static_cast<std::uint32_t>(tile_columns);
    }
    return geometry;
}

// Byte addressing inside one mip level of block-linear storage: a level is a
// row of blocks-of-GOBs
// (rob) stacked into depth slices; within a GOB, 16-byte segments interleave
// with rows. All components add up in byte units.
struct GobLinearLayout {
    std::size_t rob_size = 0;
    std::size_t slice_size = 0;
    std::size_t block_height_mask = 0;
    std::size_t block_depth_mask = 0;
    std::uint32_t block_height_shift = 0;
    std::uint32_t block_depth_shift = 0;
    std::uint32_t x_shift = 0;

    std::size_t offset(std::size_t x_bytes, std::size_t y,
                       std::size_t z) const noexcept {
        const std::size_t gob_row = y / texture_gob_height_rows;
        std::size_t value = (z >> block_depth_shift) * slice_size;
        value += (gob_row >> block_height_shift) * rob_size;
        value += (x_bytes / texture_gob_width_bytes) << x_shift;
        value += (gob_row & block_height_mask) * texture_gob_size;
        value += ((z & block_depth_mask) * texture_gob_size)
                 << block_height_shift;
        value += ((x_bytes & 0x3f) >> 5) << 8;
        value += ((y & 0x07) >> 1) << 6;
        value += ((x_bytes & 0x1f) >> 4) << 5;
        value += (y & 0x01) << 4;
        value += x_bytes & 0x0f;
        return value;
    }
};

// Builds the addressing for one level from its aligned block-column count and
// the level's shrunk GOB block sizes. Returns false on overflow.
bool make_gob_linear_layout(std::size_t block_columns,
                            std::uint32_t column_alignment,
                            std::size_t block_rows,
                            std::uint32_t gob_block_height_log2,
                            std::uint32_t gob_block_depth_log2,
                            std::size_t bytes_per_block,
                            GobLinearLayout& layout) noexcept {
    const std::size_t block_height = std::size_t{1} << gob_block_height_log2;
    const std::size_t block_depth = std::size_t{1} << gob_block_depth_log2;

    std::size_t columns_aligned = 0;
    if (!align_up(block_columns, column_alignment, columns_aligned)) {
        return false;
    }
    std::size_t aligned_row_bytes = 0;
    if (!checked_multiply(columns_aligned, bytes_per_block,
                          aligned_row_bytes)) {
        return false;
    }
    const std::size_t width_in_gobs =
        div_round_up(aligned_row_bytes, texture_gob_width_bytes);

    std::size_t rob_size = 0;
    if (!checked_multiply(texture_gob_size, block_height, rob_size) ||
        !checked_multiply(rob_size, block_depth, rob_size) ||
        !checked_multiply(rob_size, width_in_gobs, rob_size)) {
        return false;
    }
    std::size_t slice_size = 0;
    if (!checked_multiply(div_round_up(
                              block_rows, block_height * texture_gob_height_rows),
                          rob_size, slice_size)) {
        return false;
    }

    layout.rob_size = rob_size;
    layout.slice_size = slice_size;
    layout.block_height_mask = block_height - 1;
    layout.block_depth_mask = block_depth - 1;
    layout.block_height_shift =
        static_cast<std::uint32_t>(std::countr_zero(block_height));
    layout.block_depth_shift =
        static_cast<std::uint32_t>(std::countr_zero(block_depth));
    layout.x_shift = 9 + gob_block_height_log2 + gob_block_depth_log2;
    return true;
}

// Rounds a layer's mip-chain size up to a whole block-of-GOBs so array layers
// start on the storage's natural boundary.
bool align_layer_size(std::size_t size, const TextureLayoutDescription& layout,
                      std::size_t& result) noexcept {
    const std::size_t tile_columns =
        std::size_t{1} << layout.gob_block_width_log2;
    std::size_t block_height = std::size_t{1} << layout.gob_block_height_log2;
    std::size_t block_depth = std::size_t{1} << layout.gob_block_depth_log2;
    if (tile_columns < 2) {
        const std::size_t rows =
            div_round_up(layout.height, layout.block_height);
        while (rows <= (block_height >> 1) * texture_gob_height_rows &&
               block_height != 1) {
            block_height >>= 1;
        }
        while (layout.depth <= (block_depth >> 1) && block_depth != 1) {
            block_depth >>= 1;
        }
        std::size_t block_of_gobs_size = 0;
        if (!checked_multiply(block_height, block_depth, block_of_gobs_size) ||
            !checked_multiply(block_of_gobs_size, texture_gob_size,
                              block_of_gobs_size)) {
            return false;
        }
        const std::size_t remainder = size % block_of_gobs_size;
        if (remainder != 0) {
            return checked_add(size, block_of_gobs_size - remainder, result);
        }
        result = size;
        return true;
    }
    std::size_t alignment = 0;
    if (!checked_multiply(tile_columns, texture_gob_size, alignment) ||
        !checked_multiply(alignment, block_height, alignment) ||
        !checked_multiply(alignment, block_depth, alignment)) {
        return false;
    }
    return align_up(size, alignment, result);
}

struct LevelPlacement {
    std::size_t guest_base = 0;
    std::uint32_t column_alignment = 1;
    const TextureLevelLayout* mip = nullptr;
};

bool level_placement(const TextureLayoutDescription& layout,
                     std::uint32_t level, std::uint32_t layer,
                     LevelPlacement& placement) noexcept {
    if (level >= layout.mips.size() || layer >= layout.layers) {
        return false;
    }
    std::size_t base = 0;
    if (!checked_multiply(std::size_t{layer}, layout.layer_stride, base)) {
        return false;
    }
    if (!checked_add(base, layout.mips[level].offset, base)) {
        return false;
    }
    placement.guest_base = base;
    placement.column_alignment = layout.storage == TextureStorageLayout::Pitch
                                     ? 1
                                     : level_geometry(layout, level).column_alignment;
    placement.mip = &layout.mips[level];
    return true;
}

// One block occupies contiguous guest bytes only while it stays inside a
// 16-byte GOB segment. Real block sizes (1, 2, 4, 8, 16 with aligned column
// starts) always do; a 12-byte texel can straddle, so it moves byte by byte
// through the address function.
void copy_block(std::byte* destination, const std::byte* guest_base,
                const GobLinearLayout& gob, std::size_t x_bytes,
                std::size_t y, std::size_t z, std::size_t bytes) noexcept {
    if ((x_bytes & 0x0f) + bytes <= texture_gob_width_bytes / 4) {
        std::memcpy(destination, guest_base + gob.offset(x_bytes, y, z), bytes);
        return;
    }
    for (std::size_t b = 0; b < bytes; ++b) {
        destination[b] = guest_base[gob.offset(x_bytes + b, y, z)];
    }
}

void copy_block_inverse(std::byte* guest_base, const std::byte* source,
                        const GobLinearLayout& gob, std::size_t x_bytes,
                        std::size_t y, std::size_t z,
                        std::size_t bytes) noexcept {
    if ((x_bytes & 0x0f) + bytes <= texture_gob_width_bytes / 4) {
        std::memcpy(guest_base + gob.offset(x_bytes, y, z), source, bytes);
        return;
    }
    for (std::size_t b = 0; b < bytes; ++b) {
        guest_base[gob.offset(x_bytes + b, y, z)] = source[b];
    }
}

// Copies one block-linear level of one layer from guest storage into tightly
// packed rows (depth planes outermost), checking every block against the
// level's storage size.
TextureLayoutResult<TextureLevelData> level_to_linear(
    const TextureLayoutDescription& layout, const LevelPlacement& placement,
    std::span<const std::byte> guest) {
    const TextureLevelLayout& mip = *placement.mip;
    const std::size_t bytes = layout.bytes_per_block;
    GobLinearLayout gob;
    if (!make_gob_linear_layout(mip.block_columns, placement.column_alignment,
                                mip.block_rows, mip.gob_block_height_log2,
                                mip.gob_block_depth_log2, bytes, gob)) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    TextureLevelData data;
    data.width = mip.width;
    data.height = mip.height;
    data.depth = mip.depth;
    data.row_bytes = mip.linear_row_bytes;
    data.bytes.resize(mip.linear_bytes);
    const std::size_t guest_end = placement.guest_base + mip.size;
    for (std::uint32_t z = 0; z < mip.block_slices; ++z) {
        for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
            std::byte* row =
                data.bytes.data() +
                (std::size_t{z} * mip.block_rows + y) * mip.linear_row_bytes;
            for (std::uint32_t x = 0; x < mip.block_columns; ++x) {
                const std::size_t offset =
                    gob.offset(std::size_t{x} * bytes, y, z);
                if (placement.guest_base + offset + bytes > guest_end) {
                    return failure(TextureLayoutErrorCode::ExtentOverflow);
                }
                copy_block(row + std::size_t{x} * bytes,
                           guest.data() + placement.guest_base, gob,
                           std::size_t{x} * bytes, y, z, bytes);
            }
        }
    }
    return data;
}

// Inverse of level_to_linear: writes the level's guest storage bytes.
TextureLayoutResult<TextureLevelData> level_to_block_linear(
    const TextureLayoutDescription& layout, const LevelPlacement& placement,
    std::span<const std::byte> linear) {
    const TextureLevelLayout& mip = *placement.mip;
    const std::size_t bytes = layout.bytes_per_block;
    if (linear.size() < mip.linear_bytes) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, linear.size(),
                       mip.linear_bytes);
    }
    GobLinearLayout gob;
    if (!make_gob_linear_layout(mip.block_columns, placement.column_alignment,
                                mip.block_rows, mip.gob_block_height_log2,
                                mip.gob_block_depth_log2, bytes, gob)) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    TextureLevelData data;
    data.width = mip.width;
    data.height = mip.height;
    data.depth = mip.depth;
    data.row_bytes = mip.size;
    data.bytes.resize(mip.size);
    for (std::uint32_t z = 0; z < mip.block_slices; ++z) {
        for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
            const std::byte* row =
                linear.data() +
                (std::size_t{z} * mip.block_rows + y) * mip.linear_row_bytes;
            for (std::uint32_t x = 0; x < mip.block_columns; ++x) {
                const std::size_t offset =
                    gob.offset(std::size_t{x} * bytes, y, z);
                if (offset + bytes > mip.size) {
                    return failure(TextureLayoutErrorCode::ExtentOverflow);
                }
                copy_block_inverse(data.bytes.data(),
                                   row + std::size_t{x} * bytes, gob,
                                   std::size_t{x} * bytes, y, z, bytes);
            }
        }
    }
    return data;
}

// Pitch storage copies whole block rows through the pitch stride.
TextureLayoutResult<TextureLevelData> pitch_level_to_linear(
    const TextureLayoutDescription& layout, const LevelPlacement& placement,
    std::span<const std::byte> guest) {
    const TextureLevelLayout& mip = *placement.mip;
    TextureLevelData data;
    data.width = mip.width;
    data.height = mip.height;
    data.depth = mip.depth;
    data.row_bytes = mip.linear_row_bytes;
    data.bytes.resize(mip.linear_bytes);
    for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
        std::memcpy(data.bytes.data() + std::size_t{y} * mip.linear_row_bytes,
                    guest.data() + placement.guest_base +
                        std::size_t{y} * layout.pitch_bytes,
                    mip.linear_row_bytes);
    }
    return data;
}

TextureLayoutResult<TextureLevelData> pitch_level_to_block_linear(
    const TextureLayoutDescription& layout, const LevelPlacement& placement,
    std::span<const std::byte> linear) {
    const TextureLevelLayout& mip = *placement.mip;
    if (linear.size() < mip.linear_bytes) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, linear.size(),
                       mip.linear_bytes);
    }
    TextureLevelData data;
    data.width = mip.width;
    data.height = mip.height;
    data.depth = mip.depth;
    data.row_bytes = mip.size;
    data.bytes.resize(mip.size);
    for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
        std::memcpy(data.bytes.data() + std::size_t{y} * layout.pitch_bytes,
                    linear.data() + std::size_t{y} * mip.linear_row_bytes,
                    mip.linear_row_bytes);
    }
    return data;
}

auto level_fields(const TextureLevelLayout& mip) noexcept {
    return std::tie(mip.width, mip.height, mip.depth, mip.block_columns,
                    mip.block_rows, mip.block_slices, mip.gob_block_height_log2,
                    mip.gob_block_depth_log2, mip.offset, mip.slice_size,
                    mip.size, mip.linear_row_bytes, mip.linear_bytes);
}

// Re-derive public layout geometry before addressing or allocating. Views may
// retain a larger parent-chain layer stride and omit final-layer padding.
bool require_layout(const TextureLayoutDescription& layout,
                    TextureLayoutError& error) {
    if (layout.mips.empty() || layout.mips.size() != layout.levels ||
        layout.levels > max_texture_layout_levels) {
        error = failure(TextureLayoutErrorCode::InvalidDimensions);
        return false;
    }
    const auto calculated = calculate_texture_size(layout);
    if (const auto* invalid = std::get_if<TextureLayoutError>(&calculated)) {
        error = *invalid;
        return false;
    }
    const auto& expected = std::get<TextureLayoutDescription>(calculated);
    for (std::size_t i = 0; i < layout.mips.size(); ++i) {
        if (level_fields(layout.mips[i]) != level_fields(expected.mips[i])) {
            error = failure(TextureLayoutErrorCode::InvalidDimensions, i);
            return false;
        }
    }
    if (layout.layer_stride < expected.layer_stride) {
        error = failure(TextureLayoutErrorCode::InvalidDimensions,
                        layout.layer_stride, expected.layer_stride);
        return false;
    }
    const TextureLevelLayout& last = layout.mips.back();
    std::size_t required = 0;
    if (!checked_multiply(layout.layers - 1, layout.layer_stride, required) ||
        !checked_add(required, last.offset, required) ||
        !checked_add(required, last.size, required)) {
        error = failure(TextureLayoutErrorCode::ExtentOverflow);
        return false;
    }
    if (layout.total_size > max_texture_storage_size ||
        layout.layer_stride > max_texture_storage_size) {
        error = failure(TextureLayoutErrorCode::ExtentLimit);
        return false;
    }
    if (layout.total_size < required) {
        error = failure(TextureLayoutErrorCode::InvalidDimensions,
                        layout.total_size, required);
        return false;
    }
    return true;
}

std::size_t summed_level_linear_bytes(
    const TextureLayoutDescription& layout, TextureLayoutError& error) noexcept {
    std::size_t layer_linear = 0;
    for (const TextureLevelLayout& mip : layout.mips) {
        if (!checked_add(layer_linear, mip.linear_bytes, layer_linear)) {
            error = failure(TextureLayoutErrorCode::ExtentOverflow);
            return 0;
        }
    }
    return layer_linear;
}

}  // namespace

const char* TextureLayoutError::message() const noexcept {
    switch (code) {
        case TextureLayoutErrorCode::InvalidDimensions:
            return "texture layout dimensions or levels are invalid";
        case TextureLayoutErrorCode::UnsupportedLayout:
            return "texture layout combination is outside the layout profile";
        case TextureLayoutErrorCode::UnsupportedBytesPerBlock:
            return "texture layout bytes per block is outside the admitted set";
        case TextureLayoutErrorCode::ExtentOverflow:
            return "texture layout size arithmetic overflows";
        case TextureLayoutErrorCode::ExtentLimit:
            return "texture layout total size exceeds the 256 MiB policy";
        case TextureLayoutErrorCode::SourceTooSmall:
            return "texture conversion source span does not cover the layout";
        case TextureLayoutErrorCode::DestinationTooSmall:
            return "texture conversion destination span does not cover the layout";
    }
    return "unknown texture layout error";
}

TextureLayoutResult<TextureLayoutDescription> calculate_texture_size(
    TextureLayoutDescription layout) {
    layout.mips.clear();
    if (layout.width == 0 || layout.height == 0 || layout.depth == 0) {
        return failure(TextureLayoutErrorCode::InvalidDimensions);
    }
    if (layout.block_width == 0 || layout.block_height == 0) {
        return failure(TextureLayoutErrorCode::InvalidDimensions,
                       layout.block_width == 0 ? layout.block_width
                                               : layout.block_height,
                       1);
    }
    if (layout.levels == 0 || layout.levels > max_texture_layout_levels) {
        return failure(TextureLayoutErrorCode::InvalidDimensions,
                       layout.levels, max_texture_layout_levels);
    }
    if (layout.layers == 0) {
        return failure(TextureLayoutErrorCode::InvalidDimensions, 0, 1);
    }
    if (!bytes_per_block_admitted(layout.bytes_per_block)) {
        return failure(TextureLayoutErrorCode::UnsupportedBytesPerBlock,
                       layout.bytes_per_block, 0);
    }
    if (layout.storage != TextureStorageLayout::Pitch &&
        layout.storage != TextureStorageLayout::BlockLinear) {
        return failure(TextureLayoutErrorCode::UnsupportedLayout);
    }

    if (layout.storage == TextureStorageLayout::Pitch) {
        // Pitch storage carries a single 2D level per layer on this hardware.
        if (layout.levels != 1 || layout.depth != 1) {
            return failure(TextureLayoutErrorCode::UnsupportedLayout,
                           layout.levels != 1 ? layout.levels : layout.depth,
                           1);
        }
        const std::size_t columns = div_round_up(layout.width, layout.block_width);
        const std::size_t rows = div_round_up(layout.height, layout.block_height);
        std::size_t row_bytes = 0;
        std::size_t level_size = 0;
        if (!checked_multiply(columns, layout.bytes_per_block, row_bytes)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        if (layout.pitch_bytes < row_bytes) {
            return failure(TextureLayoutErrorCode::UnsupportedLayout,
                           layout.pitch_bytes, row_bytes);
        }
        if (!checked_multiply(rows, layout.pitch_bytes, level_size)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        std::size_t total = 0;
        if (!checked_multiply(level_size, layout.layers, total)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        if (total > max_texture_storage_size) {
            return failure(TextureLayoutErrorCode::ExtentLimit, total,
                           max_texture_storage_size);
        }
        TextureLevelLayout mip;
        mip.width = layout.width;
        mip.height = layout.height;
        mip.depth = 1;
        mip.block_columns = static_cast<std::uint32_t>(columns);
        mip.block_rows = static_cast<std::uint32_t>(rows);
        mip.block_slices = 1;
        mip.offset = 0;
        mip.slice_size = level_size;
        mip.size = level_size;
        mip.linear_row_bytes = row_bytes;
        mip.linear_bytes = rows * row_bytes;
        layout.mips.push_back(mip);
        layout.layer_stride = level_size;
        layout.total_size = total;
        return layout;
    }

    if (layout.gob_block_height_log2 > 5 || layout.gob_block_depth_log2 > 5) {
        return failure(TextureLayoutErrorCode::InvalidDimensions,
                       layout.gob_block_height_log2 > 5
                           ? layout.gob_block_height_log2
                           : layout.gob_block_depth_log2,
                       5);
    }
    if (layout.gob_block_width_log2 > 7) {
        return failure(TextureLayoutErrorCode::InvalidDimensions,
                       layout.gob_block_width_log2, 7);
    }

    const bool is3d = layout.depth > 1 || layout.gob_block_depth_log2 > 0;
    std::size_t layer_size = 0;
    std::size_t layer_size_aligned = 0;

    for (std::uint32_t level = 0; level < layout.levels; ++level) {
        const LevelGeometry geometry = level_geometry(layout, level);
        const std::size_t block_height =
            std::size_t{1} << geometry.gob_block_height_log2;
        const std::size_t block_depth =
            std::size_t{1} << geometry.gob_block_depth_log2;

        std::size_t columns_aligned = 0;
        if (!align_up(std::size_t{geometry.block_columns},
                      geometry.column_alignment, columns_aligned)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        std::size_t aligned_row_bytes = 0;
        if (!checked_multiply(columns_aligned, layout.bytes_per_block,
                              aligned_row_bytes)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        const std::size_t width_in_gobs =
            div_round_up(aligned_row_bytes, texture_gob_width_bytes);

        std::size_t rob_size = 0;
        if (!checked_multiply(width_in_gobs, block_height, rob_size) ||
            !checked_multiply(rob_size, block_depth, rob_size) ||
            !checked_multiply(rob_size, texture_gob_size, rob_size)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        std::size_t slice_size = 0;
        if (!checked_multiply(
                div_round_up(div_round_up(geometry.block_rows,
                                          texture_gob_height_rows),
                             block_height),
                rob_size, slice_size)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        std::size_t level_size = 0;
        if (!checked_multiply(div_round_up(geometry.block_slices, block_depth),
                              slice_size, level_size)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }

        if (is3d && geometry.block_slices % block_depth != 0 &&
            level == layout.levels - 1) {
            // The last level only covers up to the end of its depth, not the
            // full aligned stack of depth blocks; keep the level size tight.
            std::size_t gob_size = 0;
            if (!checked_multiply(block_height, texture_gob_size, gob_size)) {
                return failure(TextureLayoutErrorCode::ExtentOverflow);
            }
            const std::size_t trim = gob_size * (block_depth -
                                                 geometry.block_slices %
                                                     block_depth);
            if (trim > level_size) {
                return failure(TextureLayoutErrorCode::ExtentOverflow);
            }
            level_size -= trim;
            if (slice_size > level_size) {
                slice_size = level_size;
            }
        }

        std::size_t linear_row_bytes = 0;
        std::size_t linear_bytes = 0;
        if (!checked_multiply(std::size_t{geometry.block_columns},
                              layout.bytes_per_block, linear_row_bytes) ||
            !checked_multiply(linear_row_bytes,
                              std::size_t{geometry.block_rows}, linear_bytes) ||
            !checked_multiply(linear_bytes,
                              std::size_t{geometry.block_slices},
                              linear_bytes)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }

        TextureLevelLayout mip;
        mip.width = geometry.width;
        mip.height = geometry.height;
        mip.depth = geometry.depth;
        mip.block_columns = geometry.block_columns;
        mip.block_rows = geometry.block_rows;
        mip.block_slices = geometry.block_slices;
        mip.gob_block_height_log2 = geometry.gob_block_height_log2;
        mip.gob_block_depth_log2 = geometry.gob_block_depth_log2;
        mip.offset = layer_size;
        mip.slice_size = slice_size;
        mip.size = level_size;
        mip.linear_row_bytes = linear_row_bytes;
        mip.linear_bytes = linear_bytes;
        layout.mips.push_back(mip);

        if (!checked_add(layer_size_aligned, level_size, layer_size_aligned) ||
            !checked_add(layer_size, level_size, layer_size)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
    }

    std::size_t total = 0;
    if (layout.layers > 1) {
        std::size_t aligned = 0;
        if (!align_layer_size(layer_size_aligned, layout, aligned)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        layer_size_aligned = aligned;
        if (!checked_multiply(aligned, layout.layers, total)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
    } else {
        total = layer_size;
    }
    if (total > max_texture_storage_size) {
        return failure(TextureLayoutErrorCode::ExtentLimit, total,
                       max_texture_storage_size);
    }
    layout.layer_stride = layer_size_aligned;
    layout.total_size = total;
    return layout;
}

TextureLevelResult convert_level_to_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> guest,
    std::uint32_t level, std::uint32_t layer) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    LevelPlacement placement;
    if (!level_placement(layout, level, layer, placement)) {
        return failure(TextureLayoutErrorCode::InvalidDimensions, level,
                       layout.mips.size());
    }
    std::size_t required_end = 0;
    if (!checked_add(placement.guest_base, placement.mip->size,
                     required_end)) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    if (guest.size() < required_end) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, guest.size(),
                       required_end);
    }
    if (layout.storage == TextureStorageLayout::Pitch) {
        return pitch_level_to_linear(layout, placement, guest);
    }
    return level_to_linear(layout, placement, guest);
}

TextureLayoutResult<BlockLinearLevelAddressing> block_linear_level_addressing(
    const TextureLayoutDescription& layout, std::uint32_t level,
    std::uint32_t layer) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    if (layout.storage != TextureStorageLayout::BlockLinear) {
        return failure(TextureLayoutErrorCode::UnsupportedLayout);
    }
    LevelPlacement placement;
    if (!level_placement(layout, level, layer, placement)) {
        return failure(TextureLayoutErrorCode::InvalidDimensions, level,
                       layout.mips.size());
    }
    const TextureLevelLayout& mip = *placement.mip;
    GobLinearLayout gob;
    if (!make_gob_linear_layout(mip.block_columns, placement.column_alignment,
                                mip.block_rows, mip.gob_block_height_log2,
                                mip.gob_block_depth_log2, layout.bytes_per_block,
                                gob)) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    // Each coordinate adds its own non-decreasing term to the address, so the
    // level's last byte has the highest address: inside the level's storage,
    // every byte the conversions copy is.
    if (mip.linear_row_bytes == 0 || mip.block_rows == 0 ||
        mip.block_slices == 0 ||
        gob.offset(mip.linear_row_bytes - 1, mip.block_rows - 1,
                   mip.block_slices - 1) >= mip.size) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    BlockLinearLevelAddressing addressing;
    addressing.guest_base = placement.guest_base;
    addressing.guest_size = mip.size;
    addressing.row_stride = gob.rob_size;
    addressing.slice_stride = gob.slice_size;
    addressing.row_bytes = mip.linear_row_bytes;
    addressing.rows = mip.block_rows;
    addressing.slices = mip.block_slices;
    addressing.gob_block_height_log2 = mip.gob_block_height_log2;
    addressing.gob_block_depth_log2 = mip.gob_block_depth_log2;
    return addressing;
}

TextureLevelResult convert_level_to_block_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> linear,
    std::uint32_t level, std::uint32_t layer) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    LevelPlacement placement;
    if (!level_placement(layout, level, layer, placement)) {
        return failure(TextureLayoutErrorCode::InvalidDimensions, level,
                       layout.mips.size());
    }
    if (layout.storage == TextureStorageLayout::Pitch) {
        return pitch_level_to_block_linear(layout, placement, linear);
    }
    return level_to_block_linear(layout, placement, linear);
}

TextureLayoutResult<std::monostate> write_level_from_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> linear,
    std::span<std::byte> guest, std::uint32_t level, std::uint32_t layer) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    LevelPlacement placement;
    if (!level_placement(layout, level, layer, placement)) {
        return failure(TextureLayoutErrorCode::InvalidDimensions);
    }
    const TextureLevelLayout& mip = *placement.mip;
    if (linear.size() < mip.linear_bytes) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, linear.size(),
                       mip.linear_bytes);
    }
    std::size_t required_end = 0;
    if (!checked_add(placement.guest_base, mip.size, required_end)) {
        return failure(TextureLayoutErrorCode::ExtentOverflow);
    }
    if (guest.size() < required_end) {
        return failure(TextureLayoutErrorCode::DestinationTooSmall,
                       guest.size(), required_end);
    }
    GobLinearLayout gob;
    if (layout.storage == TextureStorageLayout::BlockLinear) {
        if (!make_gob_linear_layout(
                mip.block_columns, placement.column_alignment, mip.block_rows,
                mip.gob_block_height_log2, mip.gob_block_depth_log2,
                layout.bytes_per_block, gob)) {
            return failure(TextureLayoutErrorCode::ExtentOverflow);
        }
        // X addressing increases within each row, including GOB boundaries.
        // Check every row's last byte before any store; this also covers
        // 12-byte blocks that straddle noncontiguous 16-byte GOB segments.
        for (std::uint32_t z = 0; z < mip.block_slices; ++z) {
            for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
                if (gob.offset(mip.linear_row_bytes - 1, y, z) >= mip.size) {
                    return failure(TextureLayoutErrorCode::ExtentOverflow);
                }
            }
        }
    }

    // An overlapping linear view must remain stable throughout the swizzle.
    linear = linear.first(mip.linear_bytes);
    const std::uintptr_t source =
        reinterpret_cast<std::uintptr_t>(linear.data());
    const std::uintptr_t destination =
        reinterpret_cast<std::uintptr_t>(guest.data() + placement.guest_base);
    const bool overlaps = source <= destination
                              ? destination - source < linear.size()
                              : source - destination < mip.size;
    std::vector<std::byte> retained;
    if (overlaps) {
        retained.assign(linear.begin(), linear.end());
        linear = retained;
    }
    std::byte* output = guest.data() + placement.guest_base;
    for (std::uint32_t z = 0; z < mip.block_slices; ++z) {
        for (std::uint32_t y = 0; y < mip.block_rows; ++y) {
            const std::byte* row =
                linear.data() +
                (std::size_t{z} * mip.block_rows + y) * mip.linear_row_bytes;
            if (layout.storage == TextureStorageLayout::Pitch) {
                std::memcpy(output + std::size_t{y} * layout.pitch_bytes, row,
                            mip.linear_row_bytes);
            } else {
                for (std::uint32_t x = 0; x < mip.block_columns; ++x) {
                    const std::size_t x_bytes =
                        std::size_t{x} * layout.bytes_per_block;
                    copy_block_inverse(output, row + x_bytes, gob, x_bytes, y,
                                       z, layout.bytes_per_block);
                }
            }
        }
    }
    return std::monostate{};
}

TextureLayoutResult<std::vector<std::byte>> convert_texture_to_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> guest) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    if (guest.size() < layout.total_size) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, guest.size(),
                       layout.total_size);
    }
    const std::size_t layer_linear = summed_level_linear_bytes(layout, error);
    if (layer_linear == 0) {
        return error;
    }
    std::size_t total_linear = 0;
    if (!checked_multiply(layer_linear, layout.layers, total_linear) ||
        total_linear > max_texture_layout_size) {
        return failure(TextureLayoutErrorCode::ExtentLimit, total_linear,
                       max_texture_layout_size);
    }
    std::vector<std::byte> output(total_linear);
    std::size_t cursor = 0;
    for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < layout.mips.size(); ++level) {
            const TextureLevelResult result =
                convert_level_to_linear(layout, guest, level, layer);
            if (const auto* level_error = std::get_if<TextureLayoutError>(&result)) {
                return *level_error;
            }
            const TextureLevelData& data = std::get<TextureLevelData>(result);
            std::memcpy(output.data() + cursor, data.bytes.data(),
                        data.bytes.size());
            cursor += data.bytes.size();
        }
    }
    return output;
}

TextureLayoutResult<std::vector<std::byte>> convert_texture_to_block_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> linear) {
    TextureLayoutError error{};
    if (!require_layout(layout, error)) {
        return error;
    }
    const std::size_t layer_linear = summed_level_linear_bytes(layout, error);
    if (layer_linear == 0) {
        return error;
    }
    std::size_t total_linear = 0;
    if (!checked_multiply(layer_linear, layout.layers, total_linear) ||
        linear.size() < total_linear) {
        return failure(TextureLayoutErrorCode::SourceTooSmall, linear.size(),
                       total_linear);
    }
    std::vector<std::byte> output(layout.total_size);
    std::size_t cursor = 0;
    for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < layout.mips.size(); ++level) {
            const TextureLevelResult result = convert_level_to_block_linear(
                layout, linear.subspan(cursor, layout.mips[level].linear_bytes),
                level, layer);
            if (const auto* level_error = std::get_if<TextureLayoutError>(&result)) {
                return *level_error;
            }
            const TextureLevelData& data = std::get<TextureLevelData>(result);
            std::size_t destination = 0;
            if (!checked_multiply(std::size_t{layer}, layout.layer_stride,
                                  destination) ||
                !checked_add(destination, layout.mips[level].offset,
                             destination)) {
                return failure(TextureLayoutErrorCode::ExtentOverflow);
            }
            std::memcpy(output.data() + destination, data.bytes.data(),
                        data.bytes.size());
            cursor += layout.mips[level].linear_bytes;
        }
    }
    return output;
}

}  // namespace armigo::graphics
