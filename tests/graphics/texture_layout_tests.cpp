// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic texture-layout tests use hand-computed sizes, offsets, and GOB byte
// addresses to verify independently authored descriptors and texel patterns.

#include <armigo/graphics/texture_layout.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

template <typename T>
bool is_error(const TextureLayoutResult<T>& result,
              TextureLayoutErrorCode code) {
    const auto* error = std::get_if<TextureLayoutError>(&result);
    return error != nullptr && error->code == code;
}

template <typename T>
bool is_data(const TextureLayoutResult<T>& result) {
    return std::get_if<T>(&result) != nullptr;
}

TextureLayoutDescription block_linear(std::uint32_t width, std::uint32_t height,
                                      std::uint32_t depth, std::uint32_t levels,
                                      std::uint32_t layers,
                                      std::size_t bytes_per_block,
                                      std::uint32_t block_width,
                                      std::uint32_t block_height,
                                      std::uint32_t gob_block_height_log2,
                                      std::uint32_t gob_block_depth_log2 = 0) {
    TextureLayoutDescription description;
    description.width = width;
    description.height = height;
    description.depth = depth;
    description.levels = levels;
    description.layers = layers;
    description.bytes_per_block = bytes_per_block;
    description.block_width = block_width;
    description.block_height = block_height;
    description.gob_block_height_log2 = gob_block_height_log2;
    description.gob_block_depth_log2 = gob_block_depth_log2;
    return description;
}

// The expected tightly packed image: layer-major, then level, then depth
// plane, then block row, each byte derived from its position.
std::vector<std::byte> build_linear(const TextureLayoutDescription& layout) {
    std::vector<std::byte> expected;
    for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < layout.mips.size(); ++level) {
            const std::size_t base = expected.size();
            expected.resize(base + layout.mips[level].linear_bytes);
            for (std::size_t i = 0; i < layout.mips[level].linear_bytes; ++i) {
                expected[base + i] = std::byte{
                    static_cast<unsigned char>((layer * 37 + level * 11 + i) &
                                               0xff)};
            }
        }
    }
    return expected;
}

void test_hand_computed_mip_sizes() {
    // 256x256 RGBA8, GOB block height 16, three levels. Hand-computed:
    // level 0: 16 GOBs wide, 2 GOB rows -> 16*16*512*2 = 262144 bytes.
    // level 1: 128x128 keeps block height 16 (128 > 64): 8*16*512 = 65536.
    // level 2: 64x64 shrinks block height to 8 (64 <= 64, then 64 > 32):
    //          4*8*512 = 16384.
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(block_linear(256, 256, 1, 3, 1, 4, 1, 1, 4));
    CHECK(is_data(result));
    if (!is_data(result)) {
        return;
    }
    const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
    CHECK(layout.mips.size() == 3);
    CHECK(layout.mips[0].offset == 0);
    CHECK(layout.mips[0].size == 262144);
    CHECK(layout.mips[0].slice_size == 262144);
    CHECK(layout.mips[0].block_columns == 256);
    CHECK(layout.mips[0].block_rows == 256);
    CHECK(layout.mips[0].gob_block_height_log2 == 4);
    CHECK(layout.mips[1].width == 128);
    CHECK(layout.mips[1].height == 128);
    CHECK(layout.mips[1].offset == 262144);
    CHECK(layout.mips[1].size == 65536);
    CHECK(layout.mips[1].gob_block_height_log2 == 4);
    CHECK(layout.mips[2].width == 64);
    CHECK(layout.mips[2].height == 64);
    CHECK(layout.mips[2].offset == 327680);
    CHECK(layout.mips[2].size == 16384);
    CHECK(layout.mips[2].gob_block_height_log2 == 3);
    CHECK(layout.layer_stride == 344064);
    CHECK(layout.total_size == 344064);
}

void test_array_layer_alignment() {
    // 256x40 RGBA8, GOB block height 16, four levels, six layers. Chain:
    // 65536 + 16384 + 4096 + 1024 = 87040. Layer stride rounds the chain up
    // to a multiple of 8*1*512 = 4096 (base-level block of GOBs): 90112.
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(block_linear(256, 40, 1, 4, 6, 4, 1, 1, 4));
    CHECK(is_data(result));
    if (!is_data(result)) {
        return;
    }
    const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
    CHECK(layout.mips.size() == 4);
    CHECK(layout.mips[0].size == 65536);
    CHECK(layout.mips[1].size == 16384);
    CHECK(layout.mips[1].gob_block_height_log2 == 2);
    CHECK(layout.mips[2].size == 4096);
    CHECK(layout.mips[2].gob_block_height_log2 == 1);
    CHECK(layout.mips[3].size == 1024);
    CHECK(layout.mips[3].gob_block_height_log2 == 0);
    CHECK(layout.layer_stride == 90112);
    CHECK(layout.total_size == 540672);

    // The whole texture converts; the per-layer padding between the mip chain
    // end and the aligned stride stays untouched (zero).
    std::size_t layer_linear = 0;
    for (const TextureLevelLayout& mip : layout.mips) {
        layer_linear += mip.linear_bytes;
    }
    CHECK(layer_linear == 54400);
    const std::vector<std::byte> linear = build_linear(layout);
    CHECK(linear.size() == 326400);
    const TextureLayoutResult<std::vector<std::byte>> tiled =
        convert_texture_to_block_linear(layout, linear);
    CHECK(is_data(tiled));
    if (!is_data(tiled)) {
        return;
    }
    const std::vector<std::byte>& guest = std::get<std::vector<std::byte>>(tiled);
    CHECK(guest.size() == 540672);
    CHECK(guest[87040] == std::byte{0});
    CHECK(guest[90112 + 87040] == std::byte{0});
    CHECK(guest[1] != std::byte{0});
    const TextureLayoutResult<std::vector<std::byte>> back =
        convert_texture_to_linear(layout, guest);
    CHECK(is_data(back));
    if (is_data(back)) {
        CHECK(std::get<std::vector<std::byte>>(back) == linear);
    }
}

void test_bcn_block_sizes() {
    // BC1: 32x32 texels of 4x4 blocks, 8 bytes per block, two levels. Each
    // level is one 64-byte GOB row of blocks and 8 or 4 block rows, both
    // within one GOB: 512 bytes each.
    const TextureLayoutResult<TextureLayoutDescription> bc1 =
        calculate_texture_size(block_linear(32, 32, 1, 2, 1, 8, 4, 4, 0));
    CHECK(is_data(bc1));
    if (is_data(bc1)) {
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(bc1);
        CHECK(layout.mips.size() == 2);
        CHECK(layout.mips[0].block_columns == 8);
        CHECK(layout.mips[0].block_rows == 8);
        CHECK(layout.mips[0].size == 512);
        CHECK(layout.mips[0].linear_row_bytes == 64);
        CHECK(layout.mips[0].linear_bytes == 512);
        CHECK(layout.mips[1].block_columns == 4);
        CHECK(layout.mips[1].block_rows == 4);
        CHECK(layout.mips[1].offset == 512);
        CHECK(layout.mips[1].size == 512);
        CHECK(layout.mips[1].linear_bytes == 128);
        CHECK(layout.total_size == 1024);
    }

    // ASTC footprints on synthetic images: every case fits one GOB.
    const std::array<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t,
                                std::uint32_t>, 5>
        astc_cases{{
            {20, 16, 5, 4},
            {12, 12, 6, 6},
            {16, 16, 8, 8},
            {10, 10, 10, 10},
            {12, 12, 12, 12},
        }};
    for (const auto& [width, height, block_w, block_h] : astc_cases) {
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(
                block_linear(width, height, 1, 1, 1, 16, block_w, block_h, 0));
        CHECK(is_data(result));
        if (is_data(result)) {
            const TextureLayoutDescription& layout =
                std::get<TextureLayoutDescription>(result);
            CHECK(layout.total_size == 512);
            CHECK(layout.mips[0].block_rows ==
                  (height + block_h - 1) / block_h);
        }
    }
}

void test_3d_sizes() {
    // 64x32x4, one byte per texel, GOB block height 4, depth 4, two levels:
    // level 0: 1 GOB wide, 1 block-of-GOBs row, 1 depth block: 4*4*512 = 8192.
    // level 1: 32x16x2 shrinks to block height 2 (16 <= 16) and depth 2
    //          (2 <= 2): 2*2*512 = 2048.
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(block_linear(64, 32, 4, 2, 1, 1, 1, 1, 2, 2));
    CHECK(is_data(result));
    if (!is_data(result)) {
        return;
    }
    const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
    CHECK(layout.mips.size() == 2);
    CHECK(layout.mips[0].block_slices == 4);
    CHECK(layout.mips[0].slice_size == 8192);
    CHECK(layout.mips[0].size == 8192);
    CHECK(layout.mips[0].linear_bytes == 64 * 32 * 4);
    CHECK(layout.mips[1].block_slices == 2);
    CHECK(layout.mips[1].offset == 8192);
    CHECK(layout.mips[1].size == 2048);
    CHECK(layout.mips[1].gob_block_height_log2 == 1);
    CHECK(layout.mips[1].gob_block_depth_log2 == 1);
    CHECK(layout.mips[1].linear_bytes == 32 * 16 * 2);
    CHECK(layout.total_size == 10240);

    // Depth 6 does not fill the last depth block: the final level only covers
    // its slices, trimming 2 GOBs of block height 4: 16384 - 2*2048 = 12288.
    const TextureLayoutResult<TextureLayoutDescription> trimmed =
        calculate_texture_size(block_linear(64, 32, 6, 1, 1, 1, 1, 1, 2, 2));
    CHECK(is_data(trimmed));
    if (is_data(trimmed)) {
        const TextureLayoutDescription& trim =
            std::get<TextureLayoutDescription>(trimmed);
        CHECK(trim.mips[0].slice_size == 8192);
        CHECK(trim.mips[0].size == 12288);
        CHECK(trim.total_size == 12288);
        CHECK(trim.mips[0].linear_bytes == 64 * 32 * 6);
    }
}

void test_pitch_layout() {
    TextureLayoutDescription description;
    description.width = 48;
    description.height = 16;
    description.depth = 1;
    description.levels = 1;
    description.layers = 2;
    description.bytes_per_block = 4;
    description.storage = TextureStorageLayout::Pitch;
    description.pitch_bytes = 256;
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(description);
    CHECK(is_data(result));
    if (!is_data(result)) {
        return;
    }
    const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
    CHECK(layout.mips.size() == 1);
    CHECK(layout.mips[0].block_columns == 48);
    CHECK(layout.mips[0].block_rows == 16);
    CHECK(layout.mips[0].size == 16 * 256);
    CHECK(layout.mips[0].linear_row_bytes == 192);
    CHECK(layout.mips[0].linear_bytes == 16 * 192);
    CHECK(layout.layer_stride == 4096);
    CHECK(layout.total_size == 8192);

    const std::vector<std::byte> linear = build_linear(layout);
    const TextureLayoutResult<std::vector<std::byte>> tiled =
        convert_texture_to_block_linear(layout, linear);
    CHECK(is_data(tiled));
    if (!is_data(tiled)) {
        return;
    }
    const std::vector<std::byte>& guest = std::get<std::vector<std::byte>>(tiled);
    CHECK(guest.size() == 8192);
    // Row 15 of layer 0: 192 payload bytes then 64 bytes of pitch padding.
    CHECK(guest[15 * 256 + 191] == linear[15 * 192 + 191]);
    CHECK(guest[15 * 256 + 192] == std::byte{0});
    CHECK(guest[4096 + 15 * 256 + 191] == linear[3072 + 15 * 192 + 191]);
    const TextureLayoutResult<std::vector<std::byte>> back =
        convert_texture_to_linear(layout, guest);
    CHECK(is_data(back));
    if (is_data(back)) {
        CHECK(std::get<std::vector<std::byte>>(back) == linear);
    }

    // A pitch smaller than one row cannot address a full row.
    description.pitch_bytes = 100;
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::UnsupportedLayout));
    description.levels = 2;
    description.pitch_bytes = 256;
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::UnsupportedLayout));
}

void test_gob_addresses() {
    // One hand-computed GOB address per bytes-per-block case. The GOB
    // interior offset of byte column x and row y is
    //   (x%64/32)*256 + (y%8/2)*64 + (x%32/16)*32 + (y%2)*16 + x%16.
    // Each texture below fits a single GOB, so the level offset is that
    // interior value. The linear source holds the byte's own position.
    const std::byte pattern_base{0};

    {
        // 1 byte per texel: texel (9, 5) -> 0 + 128 + 0 + 16 + 9 = 153.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(64, 8, 1, 1, 1, 1, 1, 1, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        CHECK(layout.total_size == 512);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[153] == std::byte{static_cast<unsigned char>(5 * 64 + 9)});
            (void)pattern_base;
        }
    }
    {
        // 2 bytes per texel: texel (3, 2) -> 0 + 64 + 0 + 0 + 6 = 70.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(32, 8, 1, 1, 1, 2, 1, 1, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[70] == std::byte{static_cast<unsigned char>((2 * 32 + 3) * 2)});
            CHECK(bytes[71] == std::byte{static_cast<unsigned char>((2 * 32 + 3) * 2 + 1)});
        }
    }
    {
        // 4 bytes per texel: texel (1, 3) -> 0 + 64 + 0 + 16 + 4 = 84.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(16, 8, 1, 1, 1, 4, 1, 1, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[84] == std::byte{static_cast<unsigned char>((3 * 16 + 1) * 4)});
        }
    }
    {
        // 8-byte 4x4 BCN blocks: block (5, 1) -> 256 + 0 + 0 + 16 + 8 = 280.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(32, 32, 1, 1, 1, 8, 4, 4, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        CHECK(layout.total_size == 512);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[280] == std::byte{static_cast<unsigned char>((1 * 8 + 5) * 8)});
        }
    }
    {
        // 16-byte 8x8 ASTC blocks: block (1, 0) -> 0 + 0 + 32 + 0 + 0 = 32.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(16, 16, 1, 1, 1, 16, 8, 8, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        CHECK(layout.total_size == 512);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[32] == std::byte{16});
        }
    }
    {
        // 12 bytes per texel: texel (1, 1) -> 0 + 0 + 0 + 16 + 12 = 28.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(5, 8, 1, 1, 1, 12, 1, 1, 0));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        CHECK(layout.total_size == 512);
        std::vector<std::byte> linear(512);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[28] == std::byte{static_cast<unsigned char>((1 * 5 + 1) * 12)});
        }
    }
    {
        // Multiple GOBs per row and a block height of 2: 128x32 RGBA8 is
        // 8 GOBs wide, 2 GOBs tall -> rob 8*2*512 = 8192, two rob rows.
        // texel (70, 9): 4*1024 + 512 + 64 + 32 + 16 + 8 = 4728;
        // texel (0, 20) crosses into the second rob row: 8192 + 128 = 8320.
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(128, 32, 1, 1, 1, 4, 1, 1, 1));
        CHECK(is_data(result));
        const TextureLayoutDescription& layout = std::get<TextureLayoutDescription>(result);
        CHECK(layout.total_size == 16384);
        std::vector<std::byte> linear(128 * 32 * 4);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = std::byte{static_cast<unsigned char>(i & 0xff)};
        }
        const TextureLevelResult tiled =
            convert_level_to_block_linear(layout, linear, 0, 0);
        CHECK(is_data(tiled));
        if (is_data(tiled)) {
            const std::vector<std::byte>& bytes =
                std::get<TextureLevelData>(tiled).bytes;
            CHECK(bytes[4728] ==
                  std::byte{static_cast<unsigned char>((9 * 128 + 70) * 4)});
            CHECK(bytes[4729] ==
                  std::byte{static_cast<unsigned char>(((9 * 128 + 70) * 4 + 1) & 0xff)});
            CHECK(bytes[8320] == std::byte{static_cast<unsigned char>(20 * 128 * 4)});
        }
    }
}

void test_round_trips() {
    // 96x40, GOB block height 2, three levels, two layers, for every admitted
    // bytes-per-block value (compressed cases use their block footprint).
    const std::array<std::tuple<std::size_t, std::uint32_t, std::uint32_t>, 6>
        cases{{
            std::tuple{std::size_t{1}, std::uint32_t{1}, std::uint32_t{1}},
            std::tuple{std::size_t{2}, std::uint32_t{1}, std::uint32_t{1}},
            std::tuple{std::size_t{4}, std::uint32_t{1}, std::uint32_t{1}},
            std::tuple{std::size_t{8}, std::uint32_t{4}, std::uint32_t{4}},
            std::tuple{std::size_t{12}, std::uint32_t{1}, std::uint32_t{1}},
            std::tuple{std::size_t{16}, std::uint32_t{8}, std::uint32_t{8}},
        }};
    for (const auto& [bytes_per_block, block_width, block_height] : cases) {
        const TextureLayoutResult<TextureLayoutDescription> result =
            calculate_texture_size(block_linear(
                96, 40, 1, 3, 2, bytes_per_block, block_width, block_height, 1));
        CHECK(is_data(result));
        if (!is_data(result)) {
            continue;
        }
        const TextureLayoutDescription& layout =
            std::get<TextureLayoutDescription>(result);
        std::size_t layer_linear = 0;
        for (const TextureLevelLayout& mip : layout.mips) {
            layer_linear += mip.linear_bytes;
        }
        const std::vector<std::byte> linear = build_linear(layout);
        CHECK(linear.size() == layer_linear * 2);

        const TextureLayoutResult<std::vector<std::byte>> tiled =
            convert_texture_to_block_linear(layout, linear);
        CHECK(is_data(tiled));
        if (!is_data(tiled)) {
            continue;
        }
        const std::vector<std::byte>& guest = std::get<std::vector<std::byte>>(tiled);
        CHECK(guest.size() == layout.total_size);

        const TextureLayoutResult<std::vector<std::byte>> back =
            convert_texture_to_linear(layout, guest);
        CHECK(is_data(back));
        if (is_data(back)) {
            CHECK(std::get<std::vector<std::byte>>(back) == linear);
        }

        // One level of one layer lands exactly at its described place.
        const TextureLevelResult single =
            convert_level_to_linear(layout, guest, 1, 1);
        CHECK(is_data(single));
        if (is_data(single)) {
            const TextureLevelData& data = std::get<TextureLevelData>(single);
            const std::size_t begin = layer_linear + layout.mips[0].linear_bytes;
            CHECK(data.bytes.size() == layout.mips[1].linear_bytes);
            CHECK(std::vector<std::byte>(linear.begin() + begin,
                                         linear.begin() + begin +
                                             data.bytes.size()) == data.bytes);
        }
        const std::size_t level_two_linear =
            layer_linear + layout.mips[0].linear_bytes +
            layout.mips[1].linear_bytes;
        const TextureLevelResult single_tiled =
            convert_level_to_block_linear(
                layout,
                std::span<const std::byte>(linear.data() + level_two_linear,
                                           layout.mips[2].linear_bytes),
                2, 1);
        CHECK(is_data(single_tiled));
        if (is_data(single_tiled)) {
            const TextureLevelData& data =
                std::get<TextureLevelData>(single_tiled);
            const std::size_t guest_begin = layout.layer_stride +
                                            layout.mips[2].offset;
            CHECK(data.bytes.size() == layout.mips[2].size);
            CHECK(std::vector<std::byte>(guest.begin() + guest_begin,
                                         guest.begin() + guest_begin +
                                             data.bytes.size()) == data.bytes);
        }
    }
}

void test_error_rejections() {
    TextureLayoutDescription description =
        block_linear(64, 32, 1, 2, 1, 4, 1, 1, 2);
    description.width = 0;
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    description = block_linear(64, 32, 1, 0, 1, 4, 1, 1, 2);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    description = block_linear(64, 32, 1, 17, 1, 4, 1, 1, 2);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    description = block_linear(64, 32, 1, 2, 0, 4, 1, 1, 2);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    description = block_linear(64, 32, 1, 2, 1, 3, 1, 1, 2);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::UnsupportedBytesPerBlock));
    description = block_linear(64, 32, 1, 2, 1, 4, 1, 1, 6);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    description = block_linear(64, 32, 1, 2, 1, 4, 0, 1, 2);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::InvalidDimensions));
    // 4 GiB-wide rows exceed the storage policy, not arithmetic.
    description = block_linear(0xfffffff0u, 1, 1, 1, 1, 16, 1, 1, 0);
    CHECK(is_error(calculate_texture_size(description),
                   TextureLayoutErrorCode::ExtentLimit));
    // A 12288 x 12288 16-bit layout (288 MiB) is above the conversion limit
    // and within the storage policy.
    description = block_linear(12288, 12288, 1, 1, 1, 2, 1, 1, 4);
    const auto atlas = calculate_texture_size(description);
    CHECK(std::holds_alternative<TextureLayoutDescription>(atlas));
    if (const auto* layout = std::get_if<TextureLayoutDescription>(&atlas))
        CHECK(layout->total_size > max_texture_layout_size && layout->total_size <= max_texture_storage_size);

    // Conversion bounds: the guest span must cover the requested region.
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(block_linear(256, 256, 1, 3, 1, 4, 1, 1, 4));
    CHECK(is_data(result));
    const TextureLayoutDescription& layout =
        std::get<TextureLayoutDescription>(result);
    const std::vector<std::byte> short_guest(layout.total_size - 1);
    CHECK(is_error(convert_texture_to_linear(layout, short_guest),
                   TextureLayoutErrorCode::SourceTooSmall));
    const std::vector<std::byte> full_guest(layout.total_size);
    const std::size_t level_two_end = layout.mips[2].offset + layout.mips[2].size;
    const std::vector<std::byte> short_level(level_two_end - 1);
    CHECK(is_error(convert_level_to_linear(layout, short_level, 2, 0),
                   TextureLayoutErrorCode::SourceTooSmall));
    CHECK(is_error(convert_level_to_linear(layout, full_guest, 3, 0),
                   TextureLayoutErrorCode::InvalidDimensions));
    CHECK(is_error(convert_level_to_linear(layout, full_guest, 0, 1),
                   TextureLayoutErrorCode::InvalidDimensions));
    const TextureLevelResult exact =
        convert_level_to_linear(layout, std::span<const std::byte>(
                                            full_guest.data(), level_two_end),
                                2, 0);
    CHECK(is_data(exact));

    // Conversions refuse a description that never went through calculation.
    TextureLayoutDescription raw =
        block_linear(64, 32, 1, 2, 1, 4, 1, 1, 2);
    CHECK(is_error(convert_level_to_linear(raw, full_guest, 0, 0),
                   TextureLayoutErrorCode::InvalidDimensions));
    CHECK(is_error(convert_texture_to_linear(raw, full_guest),
                   TextureLayoutErrorCode::InvalidDimensions));
}

void check_preserving_writes(TextureLayoutDescription layout) {
    std::vector<std::byte> guest(layout.total_size + 17);
    for (std::size_t i = 0; i < guest.size(); ++i) {
        guest[i] =
            std::byte{static_cast<unsigned char>((i * 37 + i / 251) & 255)};
    }
    for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < layout.levels; ++level) {
            const auto decoded =
                convert_level_to_linear(layout, guest, level, layer);
            CHECK(is_data(decoded));
            if (!is_data(decoded)) {
                return;
            }
            std::vector<std::byte> linear =
                std::get<TextureLevelData>(decoded).bytes;
            const auto before = guest;
            CHECK(is_data(
                write_level_from_linear(layout, linear, guest, level, layer)));
            // Restoring the original texels must preserve every guest byte,
            // including arbitrary row, GOB, mip-chain and array padding.
            CHECK(guest == before);
            for (std::byte& value : linear) {
                value ^= std::byte{0xff};
            }
            const std::size_t begin =
                layer * layout.layer_stride + layout.mips[level].offset;
            const std::size_t end = begin + layout.mips[level].size;
            CHECK(is_data(write_level_from_linear(
                layout, linear, std::span<std::byte>(guest).first(end), level,
                layer)));
            const auto changed =
                convert_level_to_linear(layout, guest, level, layer);
            CHECK(is_data(changed));
            if (is_data(changed)) {
                CHECK(std::get<TextureLevelData>(changed).bytes == linear);
            }
            std::size_t differences = 0;
            for (std::size_t i = 0; i < guest.size(); ++i) {
                differences += guest[i] != before[i];
                if (i < begin || i >= end) {
                    CHECK(guest[i] == before[i]);
                }
            }
            // Every payload byte changed by construction, leaving no room
            // for any padding byte to have changed unnoticed.
            CHECK(differences == linear.size());
        }
    }
}

void test_preserving_level_writes() {
    const std::array<TextureLayoutDescription, 5> cases{{
        block_linear(65, 35, 1, 4, 3, 4, 1, 1, 2),
        block_linear(17, 11, 1, 3, 2, 12, 1, 1, 1),
        block_linear(35, 27, 1, 4, 2, 16, 4, 4, 2),
        block_linear(67, 19, 6, 3, 1, 2, 1, 1, 2, 2),
        block_linear(67, 19, 6, 1, 1, 2, 1, 1, 2, 2),
    }};
    for (const auto& description : cases) {
        const auto result = calculate_texture_size(description);
        CHECK(is_data(result));
        if (is_data(result)) {
            check_preserving_writes(std::get<TextureLayoutDescription>(result));
        }
    }
    auto description = block_linear(19, 7, 1, 1, 3, 4, 1, 1, 0);
    description.storage = TextureStorageLayout::Pitch;
    description.pitch_bytes = 128;
    const auto pitch = calculate_texture_size(description);
    CHECK(is_data(pitch));
    if (is_data(pitch)) {
        check_preserving_writes(std::get<TextureLayoutDescription>(pitch));
    }

    // A sub-chain array view retains its parent chain's stride, and the last
    // layer's coverage ends at its final mip rather than trailing alignment.
    auto view_result =
        calculate_texture_size(block_linear(33, 19, 1, 3, 2, 4, 1, 1, 1));
    CHECK(is_data(view_result));
    if (is_data(view_result)) {
        auto view = std::get<TextureLayoutDescription>(view_result);
        view.layer_stride += 8192;
        view.total_size =
            view.layer_stride + view.mips.back().offset + view.mips.back().size;
        check_preserving_writes(view);
    }

    // Source bytes may alias the guest destination during an in-place swizzle.
    const auto alias_result =
        calculate_texture_size(block_linear(17, 9, 1, 1, 1, 4, 1, 1, 1));
    CHECK(is_data(alias_result));
    if (is_data(alias_result)) {
        const auto& layout = std::get<TextureLayoutDescription>(alias_result);
        std::vector<std::byte> guest(layout.total_size);
        for (std::size_t i = 0; i < guest.size(); ++i) {
            guest[i] = std::byte{static_cast<unsigned char>(i & 255)};
        }
        const std::vector<std::byte> original(
            guest.begin(), guest.begin() + layout.mips[0].linear_bytes);
        CHECK(is_data(write_level_from_linear(
            layout, std::span<const std::byte>(guest).first(original.size()),
            guest, 0, 0)));
        const auto decoded = convert_level_to_linear(layout, guest, 0, 0);
        CHECK(is_data(decoded));
        if (is_data(decoded)) {
            CHECK(std::get<TextureLevelData>(decoded).bytes == original);
        }
    }
}

void test_render_target_layout_round_trips() {
    // Keep 129 rows so every encoded GOB block height remains effective.
    // Smaller fixtures would shrink the larger block heights and leave their
    // addressing paths untested.
    for (std::uint32_t block_height_log2 = 0; block_height_log2 <= 5;
         ++block_height_log2) {
        auto description = block_linear(19, 129, 1, 1,
                                        block_height_log2 == 5 ? 2 : 1, 4,
                                        1, 1, block_height_log2);
        const auto calculated = calculate_texture_size(description);
        CHECK(is_data(calculated));
        if (!is_data(calculated)) {
            continue;
        }
        const auto& layout = std::get<TextureLayoutDescription>(calculated);
        CHECK(layout.mips[0].gob_block_height_log2 == block_height_log2);
        check_preserving_writes(layout);
    }

    // Render-target pitch is the exact guest row stride, including bytes
    // beyond the texel payload. Exercise two layers so the same stride also
    // determines the array-layer boundary.
    auto pitch_description = block_linear(19, 13, 1, 1, 2, 4, 1, 1, 0);
    pitch_description.storage = TextureStorageLayout::Pitch;
    pitch_description.pitch_bytes = 128;
    const auto calculated = calculate_texture_size(pitch_description);
    CHECK(is_data(calculated));
    if (is_data(calculated)) {
        const auto& layout = std::get<TextureLayoutDescription>(calculated);
        CHECK(layout.mips[0].linear_row_bytes == 76);
        CHECK(layout.layer_stride == 13 * 128);
        check_preserving_writes(layout);
    }

    // A depth-one 3D attachment still uses its encoded depth block in guest
    // storage. The last-level trim ends after slice zero without collapsing
    // the Y strides that block depth contributes.
    const auto depth_block = calculate_texture_size(
        block_linear(64, 16, 1, 1, 1, 1, 1, 1, 0, 4));
    CHECK(is_data(depth_block));
    if (is_data(depth_block)) {
        const auto& layout = std::get<TextureLayoutDescription>(depth_block);
        CHECK(layout.mips[0].gob_block_depth_log2 == 4);
        CHECK(layout.total_size == 8704);
        check_preserving_writes(layout);
        std::vector<std::byte> guest(layout.total_size, std::byte{0x5a});
        const std::vector<std::byte> linear(layout.mips[0].linear_bytes,
                                            std::byte{64});
        CHECK(is_data(write_level_from_linear(layout, linear, guest, 0, 0)));
        CHECK(std::all_of(guest.begin(), guest.begin() + 512,
                          [](std::byte value) { return value == std::byte{64}; }));
        CHECK(std::all_of(guest.begin() + 512, guest.begin() + 8192,
                          [](std::byte value) { return value == std::byte{0x5a}; }));
        CHECK(std::all_of(guest.begin() + 8192, guest.end(),
                          [](std::byte value) { return value == std::byte{64}; }));
        CHECK(guest[8192] == std::byte{64});
    }
}

void test_write_rejections_preserve_guest() {
    const auto result =
        calculate_texture_size(block_linear(33, 19, 1, 3, 2, 4, 1, 1, 2));
    CHECK(is_data(result));
    if (!is_data(result)) {
        return;
    }
    const auto& valid = std::get<TextureLayoutDescription>(result);
    const std::vector<std::byte> linear(valid.mips[0].linear_bytes,
                                        std::byte{0xe7});
    std::vector<std::byte> guest(valid.total_size, std::byte{0x5a});
    const auto original = guest;
    const auto reject = [&](const TextureLayoutDescription& layout) {
        CHECK(std::holds_alternative<TextureLayoutError>(
            write_level_from_linear(layout, linear, guest, 0, 0)));
        CHECK(guest == original);
        // Read and allocating conversion boundaries reject the same forged
        // metadata before it can control allocations, shifts or addressing.
        CHECK(std::holds_alternative<TextureLayoutError>(
            convert_level_to_linear(layout, guest, 0, 0)));
        CHECK(std::holds_alternative<TextureLayoutError>(
            convert_texture_to_block_linear(layout, linear)));
    };
    auto malformed = valid;
    malformed.bytes_per_block = 0;
    reject(malformed);
    malformed = valid;
    malformed.gob_block_height_log2 = 64;
    reject(malformed);
    malformed = valid;
    malformed.mips[0].gob_block_depth_log2 = 64;
    reject(malformed);
    malformed = valid;
    malformed.mips[0].linear_bytes = std::numeric_limits<std::size_t>::max();
    reject(malformed);
    malformed = valid;
    malformed.mips[0].offset = std::numeric_limits<std::size_t>::max();
    reject(malformed);
    malformed = valid;
    malformed.mips[0].block_rows = std::numeric_limits<std::uint32_t>::max();
    reject(malformed);
    malformed = valid;
    malformed.mips[0].size = 1;
    reject(malformed);
    malformed = valid;
    malformed.layer_stride = 1;
    reject(malformed);
    malformed = valid;
    malformed.layer_stride = std::numeric_limits<std::size_t>::max();
    reject(malformed);
    malformed = valid;
    malformed.total_size = 1;
    reject(malformed);
    malformed = valid;
    malformed.total_size = max_texture_storage_size + 1;
    reject(malformed);
    malformed = valid;
    malformed.storage = static_cast<TextureStorageLayout>(255);
    reject(malformed);
    malformed = valid;
    malformed.mips.pop_back();
    reject(malformed);

    CHECK(is_error(
        write_level_from_linear(
            valid, std::span<const std::byte>(linear).first(linear.size() - 1),
            guest, 0, 0),
        TextureLayoutErrorCode::SourceTooSmall));
    CHECK(is_error(
        write_level_from_linear(
            valid, linear,
            std::span<std::byte>(guest).first(valid.mips[0].size - 1), 0, 0),
        TextureLayoutErrorCode::DestinationTooSmall));
    CHECK(
        is_error(write_level_from_linear(valid, linear, guest, valid.levels, 0),
                 TextureLayoutErrorCode::InvalidDimensions));
    CHECK(
        is_error(write_level_from_linear(valid, linear, guest, 0, valid.layers),
                 TextureLayoutErrorCode::InvalidDimensions));
    CHECK(guest == original);
}

}  // namespace

// Copies one level as a GPU deswizzle does: every 16-byte segment of a linear
// row comes from one contiguous run of guest bytes at the published
// addressing formula, with the segment's place inside its GOB written out
// independently of the library.
std::vector<std::byte> segment_copy(const BlockLinearLevelAddressing& level,
                                    std::span<const std::byte> guest,
                                    bool& in_bounds) {
    std::vector<std::byte> linear(level.row_bytes * level.rows * level.slices);
    const std::size_t height_mask =
        (std::size_t{1} << level.gob_block_height_log2) - 1;
    const std::size_t depth_mask =
        (std::size_t{1} << level.gob_block_depth_log2) - 1;
    in_bounds = true;
    for (std::size_t z = 0; z < level.slices; ++z) {
        for (std::size_t y = 0; y < level.rows; ++y) {
            const std::size_t gob_y = y / 8;
            for (std::size_t x = 0; x < level.row_bytes; x += 16) {
                const std::size_t count = std::min<std::size_t>(16, level.row_bytes - x);
                const std::size_t from =
                    level.guest_base +
                    (z >> level.gob_block_depth_log2) * level.slice_stride +
                    (gob_y >> level.gob_block_height_log2) * level.row_stride +
                    ((x / 64) << (9 + level.gob_block_height_log2 +
                                  level.gob_block_depth_log2)) +
                    (gob_y & height_mask) * 512 +
                    (((z & depth_mask) * 512) << level.gob_block_height_log2) +
                    ((x % 64) / 32) * 256 + ((y % 8) / 2) * 64 +
                    ((x % 32) / 16) * 32 + (y % 2) * 16;
                if (from + count > level.guest_base + level.guest_size ||
                    from + count > guest.size()) {
                    in_bounds = false;
                    return linear;
                }
                for (std::size_t i = 0; i < count; ++i) {
                    linear[(z * level.rows + y) * level.row_bytes + x + i] =
                        guest[from + i];
                }
            }
        }
    }
    return linear;
}

void test_block_linear_level_addressing() {
    // Every level and layer of each layout: the segment copy through the
    // published addressing equals the conversion's own linear image.
    auto sparse = block_linear(301, 260, 1, 2, 1, 16, 1, 1, 4);
    sparse.gob_block_width_log2 = 1;
    const std::array descriptions{
        block_linear(37, 23, 1, 3, 2, 4, 1, 1, 2),
        block_linear(70, 45, 1, 1, 1, 4, 1, 1, 0),
        block_linear(70, 45, 1, 1, 1, 4, 1, 1, 3),
        block_linear(70, 300, 1, 2, 1, 4, 1, 1, 5),
        block_linear(64, 64, 1, 7, 1, 4, 1, 1, 3),
        block_linear(16, 16, 9, 3, 1, 4, 1, 1, 1, 2),
        block_linear(33, 17, 5, 2, 1, 1, 1, 1, 1, 3),
        block_linear(36, 20, 1, 3, 1, 8, 4, 4, 1),
        block_linear(24, 24, 1, 2, 3, 16, 4, 4, 3),
        block_linear(13, 9, 1, 2, 1, 12, 1, 1, 1),
        block_linear(5, 3, 1, 1, 6, 2, 1, 1, 0),
        sparse,
    };
    for (const auto& description : descriptions) {
        const auto calculated = calculate_texture_size(description);
        CHECK(is_data(calculated));
        if (!is_data(calculated)) {
            continue;
        }
        const auto& layout = std::get<TextureLayoutDescription>(calculated);
        std::vector<std::byte> guest(layout.total_size);
        std::uint32_t state = 0x2545f491u;
        for (auto& byte : guest) {
            state = state * 1664525u + 1013904223u;
            byte = std::byte{static_cast<unsigned char>(state >> 24)};
        }
        for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
            for (std::uint32_t level = 0; level < layout.levels; ++level) {
                const auto addressed =
                    block_linear_level_addressing(layout, level, layer);
                const auto converted =
                    convert_level_to_linear(layout, guest, level, layer);
                CHECK(is_data(addressed) && is_data(converted));
                if (!is_data(addressed) || !is_data(converted)) {
                    continue;
                }
                const auto& addressing =
                    std::get<BlockLinearLevelAddressing>(addressed);
                const auto& mip = layout.mips[level];
                CHECK(addressing.guest_base ==
                      layer * layout.layer_stride + mip.offset);
                CHECK(addressing.guest_size == mip.size);
                CHECK(addressing.row_bytes == mip.linear_row_bytes);
                CHECK(addressing.rows == mip.block_rows);
                CHECK(addressing.slices == mip.block_slices);
                CHECK(addressing.gob_block_height_log2 == mip.gob_block_height_log2);
                CHECK(addressing.gob_block_depth_log2 == mip.gob_block_depth_log2);
                bool in_bounds = false;
                const auto copied = segment_copy(addressing, guest, in_bounds);
                CHECK(in_bounds);
                CHECK(copied == std::get<TextureLevelData>(converted).bytes);
            }
        }
        CHECK(is_error(block_linear_level_addressing(layout, layout.levels, 0),
                       TextureLayoutErrorCode::InvalidDimensions));
        CHECK(is_error(block_linear_level_addressing(layout, 0, layout.layers),
                       TextureLayoutErrorCode::InvalidDimensions));
    }

    TextureLayoutDescription pitch = block_linear(16, 4, 1, 1, 1, 4, 1, 1, 0);
    pitch.storage = TextureStorageLayout::Pitch;
    pitch.pitch_bytes = 64;
    const auto pitch_layout = calculate_texture_size(pitch);
    CHECK(is_data(pitch_layout));
    if (is_data(pitch_layout)) {
        CHECK(is_error(block_linear_level_addressing(
                           std::get<TextureLayoutDescription>(pitch_layout), 0, 0),
                       TextureLayoutErrorCode::UnsupportedLayout));
    }
}

void test_ampere_gob_addresses() {
    auto description = block_linear(64, 8, 1, 1, 1, 1, 1, 1, 0);
    description.gob_order = BlockLinearGobOrder::Ampere;
    auto sized = calculate_texture_size(description);
    CHECK(is_data(sized));
    if (!is_data(sized)) return;
    const auto& layout = std::get<TextureLayoutDescription>(sized);
    std::vector<std::byte> guest(layout.total_size), expected(512);
    // Independent byte-address contract for all 32 pieces of one GOB.
    const std::array<std::size_t, 8> rows{0, 16, 32, 48, 128, 144, 160, 176};
    const std::array<std::size_t, 4> columns{0, 64, 256, 320};
    for (std::size_t y = 0; y < 8; ++y)
        for (std::size_t x = 0; x < 64; ++x) {
            const auto value = std::byte((x * 17 + y * 31) & 255);
            expected[y * 64 + x] = value;
            guest[rows[y] + columns[x / 16] + x % 16] = value;
        }
    auto decoded = convert_texture_to_linear(layout, guest);
    CHECK(is_data(decoded));
    if (is_data(decoded)) CHECK(std::get<std::vector<std::byte>>(decoded) == expected);
    auto encoded = convert_texture_to_block_linear(layout, expected);
    CHECK(is_data(encoded));
    if (is_data(encoded)) CHECK(std::get<std::vector<std::byte>>(encoded) == guest);
    auto addressing = block_linear_level_addressing(layout, 0, 0);
    CHECK(is_data(addressing));
    if (is_data(addressing))
        CHECK(std::get<BlockLinearLevelAddressing>(addressing).gob_order == BlockLinearGobOrder::Ampere);
}

int main() {
    test_ampere_gob_addresses();
    test_hand_computed_mip_sizes();
    test_array_layer_alignment();
    test_bcn_block_sizes();
    test_3d_sizes();
    test_pitch_layout();
    test_gob_addresses();
    test_round_trips();
    test_error_rejections();
    test_preserving_level_writes();
    test_render_target_layout_round_trips();
    test_write_rejections_preserve_guest();
    test_block_linear_level_addressing();
    std::printf("Texture layout failures: %u\n", failures);
    return failures == 0 ? 0 : 1;
}
