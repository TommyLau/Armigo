// ============================================================================
// Armigo Graphics Tests - DMA Copy Engine
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/dma_copy.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr std::uint32_t multi_line = 1u << 9;
constexpr std::uint32_t remap = 1u << 10;
constexpr std::uint32_t src_pitch = 1u << 7;
constexpr std::uint32_t dst_pitch = 1u << 8;

// Independent block-linear byte offset for one GOB-high block (gob blocks in
// Y = 2^block_log2), surface width in bytes rounded up to whole GOBs.
std::size_t block_linear(std::size_t x_bytes, std::size_t y, std::size_t width_bytes, unsigned block_log2) {
    const std::size_t gobs_wide = (width_bytes + 63) / 64;
    const std::size_t block_rows = 8u << block_log2;
    const std::size_t block_bytes = 512u << block_log2;
    const std::size_t block_row = y / block_rows;
    const std::size_t gob_in_block = (y % block_rows) / 8;
    const std::size_t gob_x = x_bytes / 64;
    const std::size_t within = ((x_bytes % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x_bytes % 32) / 16) * 32 +
                               (y % 2) * 16 + x_bytes % 16;
    return block_row * gobs_wide * block_bytes + gob_x * block_bytes + gob_in_block * 512 + within;
}

std::vector<std::uint32_t> registers() { return std::vector<std::uint32_t>(dma_method::count); }

void test_pitch_copy_keeps_padding() {
    auto r = registers();
    r[dma_method::offset_in_lower] = 0x1000;
    r[dma_method::offset_out_lower] = 0x8000;
    r[dma_method::pitch_in] = 8;
    r[dma_method::pitch_out] = 16;
    r[dma_method::line_length_in] = 6;
    r[dma_method::line_count] = 3;
    const auto launch = multi_line | src_pitch | dst_pitch | (1u << 3);
    const auto planned = plan_dma_copy(r, launch);
    CHECK(std::holds_alternative<DmaCopyPlan>(planned));
    const auto plan = std::get<DmaCopyPlan>(planned);
    CHECK(plan.source == 0x1000 && plan.source_size == 2 * 8 + 6);
    CHECK(plan.destination == 0x8000 && plan.destination_size == 2 * 16 + 6);
    CHECK(!plan.destination_covered && plan.semaphore == 1);
    std::vector<std::byte> source(plan.source_size), destination(plan.destination_size, std::byte{0xee});
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = std::byte(i);
    execute_dma_copy(r, launch, plan, source, destination);
    for (std::size_t y = 0; y < 3; ++y)
        for (std::size_t x = 0; x < 16 && y * 16 + x < destination.size(); ++x)
            CHECK(destination[y * 16 + x] == (x < 6 ? std::byte(y * 8 + x) : std::byte{0xee}));
}

void test_block_linear_to_pitch() {
    // A 32x16 RGBA8 (4-byte remapped pixels) block-linear source, block height
    // of two GOBs, read from origin (4, 3) into a tightly packed pitch image.
    constexpr std::size_t width = 32, height = 16, bpp = 4, count_x = 20, count_y = 9;
    auto r = registers();
    r[dma_method::offset_in_lower] = 0x10000;
    r[dma_method::offset_out_lower] = 0x40000;
    r[dma_method::pitch_out] = count_x * bpp;
    r[dma_method::line_length_in] = count_x;
    r[dma_method::line_count] = count_y;
    r[dma_method::remap_components] = 0x3210u | (3u << 20) | (3u << 24);
    r[dma_method::src_block_size] = 1u << 4;
    r[dma_method::src_block_size + 1] = width;
    r[dma_method::src_block_size + 2] = height;
    r[dma_method::src_block_size + 3] = 1;
    r[dma_method::src_block_size + 5] = 4u | (3u << 16);
    const auto launch = multi_line | remap | dst_pitch;
    const auto planned = plan_dma_copy(r, launch);
    CHECK(std::holds_alternative<DmaCopyPlan>(planned));
    const auto plan = std::get<DmaCopyPlan>(planned);
    CHECK(plan.destination == 0x40000 && plan.destination_size == count_x * count_y * bpp && plan.destination_covered);
    // Fill the whole guest surface, then hand the planned window to the copy.
    std::vector<std::byte> surface(block_linear(0, height, width * bpp, 1));
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x) {
            const std::uint32_t value = std::uint32_t(x | (y << 8) | 0xab0000u);
            std::memcpy(surface.data() + block_linear(x * bpp, y, width * bpp, 1), &value, 4);
        }
    CHECK(plan.source >= 0x10000 && plan.source - 0x10000 + plan.source_size <= surface.size());
    std::span<const std::byte> window(surface.data() + (plan.source - 0x10000), plan.source_size);
    std::vector<std::byte> destination(plan.destination_size);
    execute_dma_copy(r, launch, plan, window, destination);
    bool all = true;
    for (std::size_t y = 0; y < count_y; ++y)
        for (std::size_t x = 0; x < count_x; ++x) {
            std::uint32_t value = 0;
            std::memcpy(&value, destination.data() + (y * count_x + x) * bpp, 4);
            all = all && value == std::uint32_t((x + 4) | ((y + 3) << 8) | 0xab0000u);
        }
    CHECK(all);
}

void test_pitch_to_block_linear_and_remap() {
    // Destination X takes source Y and destination Y constant B (2-byte components).
    constexpr std::size_t width = 16, height = 8, count_x = 16, count_y = 8;
    auto r = registers();
    r[dma_method::offset_out_lower] = 0x20000;
    r[dma_method::pitch_in] = count_x * 4;
    r[dma_method::line_length_in] = count_x;
    r[dma_method::line_count] = count_y;
    r[dma_method::remap_const_b] = 0x7777;
    r[dma_method::remap_components] = 0x51u | (1u << 16) | (1u << 20) | (1u << 24);
    r[dma_method::dst_block_size + 1] = width;
    r[dma_method::dst_block_size + 2] = height;
    const auto launch = multi_line | remap | src_pitch;
    const auto plan = std::get<DmaCopyPlan>(plan_dma_copy(r, launch));
    CHECK(!plan.destination_covered);
    std::vector<std::byte> source(plan.source_size);
    for (std::size_t i = 0; i + 2 <= source.size(); i += 2) {
        const std::uint16_t value = std::uint16_t(i / 2);
        std::memcpy(source.data() + i, &value, 2);
    }
    std::vector<std::byte> destination(plan.destination_size, std::byte{0});
    execute_dma_copy(r, launch, plan, source, destination);
    bool all = true;
    for (std::size_t y = 0; y < count_y; ++y)
        for (std::size_t x = 0; x < count_x; ++x) {
            const auto at = block_linear(x * 4, y, width * 4, 0) - (plan.destination - 0x20000);
            std::array<std::uint16_t, 2> out{};
            std::memcpy(out.data(), destination.data() + at, 4);
            const std::uint16_t first = std::uint16_t((y * count_x + x) * 2);
            all = all && out[0] == first + 1 && out[1] == 0x7777;
        }
    CHECK(all);
}

void test_one_dimensional_and_refusals() {
    auto r = registers();
    r[dma_method::line_length_in] = 4;
    r[dma_method::offset_out_lower] = 0x100;
    r[dma_method::remap_const_a] = 0x11223344;
    r[dma_method::remap_components] = 0x4444u | (3u << 16);
    const auto fill = plan_dma_copy(r, remap);
    CHECK(std::holds_alternative<DmaCopyPlan>(fill));
    const auto plan = std::get<DmaCopyPlan>(fill);
    CHECK(plan.source_size == 0 && plan.destination_size == 16 && plan.destination_covered);
    std::vector<std::byte> destination(16);
    execute_dma_copy(r, remap, plan, {}, destination);
    std::uint32_t last = 0;
    std::memcpy(&last, destination.data() + 12, 4);
    CHECK(last == 0x11223344);
    CHECK(std::holds_alternative<DmaCopyError>(plan_dma_copy(r, remap | (1u << 3) | (1u << 4))));
    CHECK(std::holds_alternative<DmaCopyError>(plan_dma_copy(r, src_pitch)));
    r[dma_method::remap_components] = 0x0000u;
    CHECK(std::holds_alternative<DmaCopyError>(plan_dma_copy(r, remap | src_pitch | dst_pitch)));
    r[dma_method::line_count] = 1;
    r[dma_method::src_block_size + 4] = 1;
    CHECK(std::holds_alternative<DmaCopyError>(plan_dma_copy(r, multi_line | dst_pitch)));
    r[dma_method::line_length_in] = 0;
    CHECK(std::get<DmaCopyPlan>(plan_dma_copy(r, multi_line | dst_pitch)).destination_size == 0);
}

} // namespace

int main() {
    test_pitch_copy_keeps_padding();
    test_block_linear_to_pitch();
    test_pitch_to_block_linear_and_remap();
    test_one_dimensional_and_refusals();
    std::printf("DMA copy failures: %u\n", failures);
    return failures ? 1 : 0;
}
