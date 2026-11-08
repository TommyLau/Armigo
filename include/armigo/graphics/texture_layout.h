// ============================================================================
// Armigo Graphics - GPU Texture Layout
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture storage calculations implement the guest block-linear layout and
// bounds-check every byte offset used by linear conversion.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace armigo::graphics {

// A GOB is the atomic 64-byte by 8-row tile of the Tegra block-linear layout.
inline constexpr std::size_t texture_gob_width_bytes = 64;
inline constexpr std::size_t texture_gob_height_rows = 8;
inline constexpr std::size_t texture_gob_size =
    texture_gob_width_bytes * texture_gob_height_rows;

// Total storage described by a layout is capped at the same policy limit the
// descriptor resolver applies, so no conversion can allocate beyond it.
inline constexpr std::size_t max_texture_layout_size = 256u * 1024u * 1024u;
// A layout's storage itself may be larger: the largest texture a targeted
// title allocates is a 12288 x 12288 DEPTH16 shadow atlas (288 MiB with its
// block-linear rows), which the GPU renders and samples in place. Conversions
// and CPU transfers stay within max_texture_layout_size.
inline constexpr std::size_t max_texture_storage_size = 1024u * 1024u * 1024u;
inline constexpr std::uint32_t max_texture_layout_levels = 16;

enum class TextureStorageLayout : std::uint8_t {
    BlockLinear,
    Pitch,
};

enum class TextureLayoutErrorCode : std::uint8_t {
    InvalidDimensions,
    UnsupportedLayout,
    UnsupportedBytesPerBlock,
    ExtentOverflow,
    ExtentLimit,
    SourceTooSmall,
    DestinationTooSmall,
};

struct TextureLayoutError {
    TextureLayoutErrorCode code = TextureLayoutErrorCode::InvalidDimensions;
    std::uint64_t value = 0;
    std::uint64_t expected = 0;

    [[nodiscard]] const char* message() const noexcept;
};

template <typename T>
using TextureLayoutResult = std::variant<T, TextureLayoutError>;

// Storage geometry of one mip level, in storage blocks: a block covers
// block_width by block_height texels and holds bytes_per_block bytes.
// width, height and depth are the texel extents of the level. The GOB block
// fields record the log2 GOB count per block-of-GOBs after per-level shrink,
// which may be smaller than the description's encoded fields. offset is
// relative to the start of one layer's mip chain; slice_size
// is the storage bytes of one full depth-slice row of blocks-of-GOBs, and size
// is the whole level. linear_row_bytes and linear_bytes describe the tightly
// packed image of the level (depth-major planes for 3D) produced and consumed
// by the conversion functions below.
struct TextureLevelLayout {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t block_columns = 0;
    std::uint32_t block_rows = 0;
    std::uint32_t block_slices = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
    std::size_t offset = 0;
    std::size_t slice_size = 0;
    std::size_t size = 0;
    std::size_t linear_row_bytes = 0;
    std::size_t linear_bytes = 0;
};

// Full layout description of one texture. The caller fills the geometry fields
// (extents, block footprint, bytes per block, encoded GOB block log2 fields,
// storage kind and pitch) and passes the value to calculate_texture_size,
// which validates it, fills mips and returns the layer stride and total size.
// For block-linear storage the GOB block fields follow the TIC encoding: each
// is log2 of GOBs per block-of-GOBs along that axis, so 0 means one GOB.
// gob_block_width_log2 is GOBs per tile across the X axis (1 outside sparse
// tiling). layer_stride is the "array of mip chains" distance between
// consecutive layers; for block-linear arrays each layer's chain is aligned to
// a block-of-GOBs boundary.
struct TextureLayoutDescription {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t levels = 0;
    std::uint32_t layers = 0;
    std::uint32_t block_width = 1;
    std::uint32_t block_height = 1;
    std::size_t bytes_per_block = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
    std::uint32_t gob_block_width_log2 = 0;
    TextureStorageLayout storage = TextureStorageLayout::BlockLinear;
    // Pitch storage only: stride in bytes between block rows of one layer.
    std::size_t pitch_bytes = 0;

    // Filled by calculate_texture_size.
    std::size_t layer_stride = 0;
    std::size_t total_size = 0;
    std::vector<TextureLevelLayout> mips;
};

// Validates the requested geometry and computes the storage layout: per-level
// texel extents and block dimensions (each level shrinks its GOB block height
// and, for 3D, block depth), per-level offsets
// within a layer, the layer stride and the total storage size. All arithmetic
// is overflow-checked; invalid input returns a structured error and never a
// partially filled description.
[[nodiscard]] TextureLayoutResult<TextureLayoutDescription>
calculate_texture_size(TextureLayoutDescription description);

// Tightly packed bytes of one converted mip level: block rows in row-major
// order, depth planes outermost for 3D. For a linear result row_bytes equals
// block columns times bytes per block and bytes holds depth times block rows
// of those rows; for a block-linear result row_bytes holds the level's whole
// guest storage size.
struct TextureLevelData {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::size_t row_bytes = 0;
    std::vector<std::byte> bytes;
};

using TextureLevelResult = TextureLayoutResult<TextureLevelData>;

// Converts one level of one layer between the guest storage described by the
// layout and the tightly packed image. The guest span must cover the requested
// layer up to the end of the requested level (layer * layer_stride plus the
// level offset and size); the linear span must cover the level's linear_bytes.
// Every block copy is bounds-checked against the spans; an uncovered region is
// a structured error, never a silent short read.
[[nodiscard]] TextureLevelResult convert_level_to_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> guest,
    std::uint32_t level, std::uint32_t layer);
[[nodiscard]] TextureLevelResult convert_level_to_block_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> linear,
    std::uint32_t level, std::uint32_t layer);

// Writes one tightly packed level into existing pitch or block-linear guest
// storage, preserving padding and every other level and layer. The guest span
// starts at the texture base and must cover the requested level's storage.
// Layout and span errors leave guest bytes unchanged. Overlapping source and
// destination spans are supported. Success returns std::monostate.
[[nodiscard]] TextureLayoutResult<std::monostate> write_level_from_linear(
    const TextureLayoutDescription& layout, std::span<const std::byte> linear,
    std::span<std::byte> guest, std::uint32_t level, std::uint32_t layer);

// The GOB addressing of one block-linear level of one layer, exactly as the
// conversions above walk it, for copies that address the guest bytes
// themselves. Byte x of block row y in depth slice z of the level lies at
//   guest_base + (z >> gob_block_depth_log2) * slice_stride
//              + ((y / 8) >> gob_block_height_log2) * row_stride
//              + ((x / 64) << (9 + gob_block_height_log2 + gob_block_depth_log2))
//              + ((y / 8) & (2^gob_block_height_log2 - 1)) * 512
//              + ((z & (2^gob_block_depth_log2 - 1)) * 512 << gob_block_height_log2)
//              + the byte's place inside its 64-byte by 8-row GOB,
// so the 16 bytes of one GOB segment (16-byte aligned x within one row) are
// contiguous. row_stride is one row of blocks-of-GOBs and slice_stride one
// depth block of them. row_bytes, rows and slices are the tightly packed
// extents of the level (block columns times bytes per block, block rows,
// depth slices); guest_size is the level's storage size, which every
// addressed byte lies within.
struct BlockLinearLevelAddressing {
    std::size_t guest_base = 0;
    std::size_t guest_size = 0;
    std::size_t row_stride = 0;
    std::size_t slice_stride = 0;
    std::size_t row_bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t slices = 0;
    std::uint32_t gob_block_height_log2 = 0;
    std::uint32_t gob_block_depth_log2 = 0;
};

// Validates the layout as the conversions do and returns the addressing of
// one level of one layer. Pitch storage is an UnsupportedLayout error.
[[nodiscard]] TextureLayoutResult<BlockLinearLevelAddressing>
block_linear_level_addressing(const TextureLayoutDescription& layout,
                              std::uint32_t level, std::uint32_t layer);

// Converts every level of every layer. Output is layer-major, then level,
// then depth plane, then block row; each level occupies exactly its
// linear_bytes, so a caller can slice the output with mips[i].linear_bytes.
// The guest span must cover total_size and the linear span the matching
// summed linear size.
[[nodiscard]] TextureLayoutResult<std::vector<std::byte>>
convert_texture_to_linear(const TextureLayoutDescription& layout,
                          std::span<const std::byte> guest);
[[nodiscard]] TextureLayoutResult<std::vector<std::byte>>
convert_texture_to_block_linear(const TextureLayoutDescription& layout,
                                std::span<const std::byte> linear);

}  // namespace armigo::graphics
