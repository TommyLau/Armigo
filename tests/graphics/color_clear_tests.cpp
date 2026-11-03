// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <algorithm>
#include <armigo/graphics/color_clear.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::uint32_t pack_halves(std::uint32_t low, std::uint32_t high) noexcept
{
    return low | (high << 16);
}

RenderStop make_stop(std::uint32_t target_index,
                     std::uint32_t width,
                     std::uint32_t height,
                     std::uint32_t format,
                     std::uint32_t tile_mode,
                     std::uint32_t width_or_pitch)
{
    RenderStop stop{};
    stop.kind = RenderKind::Clear;
    stop.engine = engine_class::threed;
    stop.original = {0x674, 3, 0, 0x2000, true};
    stop.registers[0x674] = 0x3c | (target_index << 6);
    stop.registers[0xdf] = 1;
    stop.registers[0x3fd] = pack_halves(0, width);
    stop.registers[0x3fe] = pack_halves(0, height);
    stop.registers[0x556] = 1;
    stop.effective_render_enable_mode = 1;
    stop.registers[0x574] = 0;
    stop.registers[0x651] = 0;
    const std::array colors{0.125F, 0.375F, 0.625F, 0.875F};
    for (std::size_t index = 0; index < colors.size(); ++index) {
        stop.registers[0x360 + index] = std::bit_cast<std::uint32_t>(colors[index]);
    }

    TargetDescription& target = stop.color_targets[target_index];
    target.address = 0x18000;
    target.width = width_or_pitch;
    target.height = height;
    target.format = format;
    target.tile_mode = tile_mode;
    target.depth = 1;
    target.array_pitch = 0x800;
    target.base_layer = 0;
    stop.registers[0x200 + target_index * 0x10 + 6] = 1;
    return stop;
}

ColorClearPlan require_plan(const ColorClearResult& result)
{
    CHECK(std::holds_alternative<ColorClearPlan>(result));
    return std::holds_alternative<ColorClearPlan>(result) ? std::get<ColorClearPlan>(result)
                                                          : ColorClearPlan{};
}

std::size_t require_offset(const PixelByteOffsetResult& result)
{
    CHECK(std::holds_alternative<std::size_t>(result));
    return std::holds_alternative<std::size_t>(result) ? std::get<std::size_t>(result) : 0;
}

void check_rejected(const RenderStop& stop)
{
    const ColorClearResult result = plan_color_clear(stop);
    CHECK(std::holds_alternative<CommandResult>(result));
    if (std::holds_alternative<CommandResult>(result)) {
        CHECK(!std::get<CommandResult>(result).ok());
    }
}

void test_block_linear_plan_and_fixed_offsets()
{
    RenderStop stop = make_stop(2, 19, 139, 0xd1, 0x40, 19);
    const ColorClearPlan plan = require_plan(plan_color_clear(stop));
    CHECK(plan.target_address == 0x18000);
    CHECK(plan.target_index == 2);
    CHECK(plan.width == 19);
    CHECK(plan.height == 139);
    CHECK(plan.row_bytes == 128);
    CHECK(plan.storage_size == 32768);
    CHECK(plan.layout == ColorTargetLayout::BlockLinear);
    CHECK(plan.format == ColorTargetFormat::A2B10G10R10Unorm);
    CHECK(plan.pitch_bytes == 0);
    CHECK(plan.block_height_log2 == 4);
    const std::array expected_color{0.125F, 0.375F, 0.625F, 0.875F};
    CHECK(plan.color == expected_color);

    CHECK(require_offset(pixel_byte_offset(plan, 0, 0)) == 0);
    CHECK(require_offset(pixel_byte_offset(plan, 1, 0)) == 4);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 1)) == 16);
    CHECK(require_offset(pixel_byte_offset(plan, 4, 0)) == 32);
    CHECK(require_offset(pixel_byte_offset(plan, 8, 0)) == 256);
    CHECK(require_offset(pixel_byte_offset(plan, 16, 0)) == 8192);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 8)) == 512);
    CHECK(require_offset(pixel_byte_offset(plan, 18, 8)) == 8712);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 128)) == 16384);
    CHECK(require_offset(pixel_byte_offset(plan, 18, 138)) == 25160);

    std::vector<std::byte> storage(plan.storage_size, std::byte{0x7b});
    for (std::uint32_t y = 0; y < plan.height; ++y) {
        for (std::uint32_t x = 0; x < plan.width; ++x) {
            const std::size_t offset = require_offset(pixel_byte_offset(plan, x, y));
            std::fill_n(storage.begin() + static_cast<std::ptrdiff_t>(offset), 4, std::byte{0x21});
        }
    }
    CHECK(std::count(storage.begin(), storage.end(), std::byte{0x21}) == 19 * 139 * 4);
    CHECK(storage[8224] == std::byte{0x7b});
    CHECK(storage.back() == std::byte{0x7b});
}

void test_pitch_plan_and_padding()
{
    RenderStop stop = make_stop(0, 19, 9, 0xd5, 0x1000, 80);
    const ColorClearPlan plan = require_plan(plan_color_clear(stop));
    CHECK(plan.width == 19);
    CHECK(plan.height == 9);
    CHECK(plan.row_bytes == 80);
    CHECK(plan.storage_size == 720);
    CHECK(plan.layout == ColorTargetLayout::Pitch);
    CHECK(plan.format == ColorTargetFormat::R8G8B8A8Unorm);
    CHECK(plan.pitch_bytes == 80);
    CHECK(plan.block_height_log2 == 0);
    CHECK(require_offset(pixel_byte_offset(plan, 18, 8)) == 712);

    std::vector<std::byte> storage(plan.storage_size, std::byte{0x7b});
    for (std::uint32_t y = 0; y < plan.height; ++y) {
        for (std::uint32_t x = 0; x < plan.width; ++x) {
            const std::size_t offset = require_offset(pixel_byte_offset(plan, x, y));
            std::fill_n(storage.begin() + static_cast<std::ptrdiff_t>(offset), 4, std::byte{0x21});
        }
    }
    CHECK(std::count(storage.begin(), storage.end(), std::byte{0x21}) == 19 * 9 * 4);
    for (std::uint32_t y = 0; y < plan.height; ++y) {
        CHECK(storage[y * plan.row_bytes + 76] == std::byte{0x7b});
        CHECK(storage[y * plan.row_bytes + 79] == std::byte{0x7b});
    }
}

void test_effective_clear_and_full_scissor()
{
    RenderStop stop = make_stop(0, 17, 11, 0xd5, 0x00, 17);
    stop.original.argument = 0;
    stop.registers[0x43e] = 0x100;
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(0, 17);
    stop.registers[0x382] = pack_halves(0, 11);
    {
        const ColorClearPlan plan = require_plan(plan_color_clear(stop));
        CHECK(plan.region.x == 0 && plan.region.y == 0);
        CHECK(plan.region.width == 17 && plan.region.height == 11);
    }

    // A disabled clear scissor leaves stale rectangle registers harmless.
    stop.registers[0x380] = 0;
    stop.registers[0x381] = pack_halves(1, 2);
    {
        const ColorClearPlan plan = require_plan(plan_color_clear(stop));
        CHECK(plan.region.width == 17 && plan.region.height == 11);
    }

    // Clear control without the use-scissor bit ignores the scissor state.
    stop.registers[0x43e] = 0x1;
    stop.registers[0x380] = 1;
    {
        const ColorClearPlan plan = require_plan(plan_color_clear(stop));
        CHECK(plan.region.width == 17 && plan.region.height == 11);
    }
    stop.registers[0x43e] = 0x100;

    // A contained, nonempty scissor clears exactly its own extent.
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(1, 9);
    stop.registers[0x382] = pack_halves(2, 7);
    {
        const ColorClearPlan plan = require_plan(plan_color_clear(stop));
        CHECK(plan.region.x == 1 && plan.region.y == 2);
        CHECK(plan.region.width == 8 && plan.region.height == 5);
        CHECK(plan.width == 17 && plan.height == 11);
    }

    // Empty, inverted, overshooting, and unknown-state scissors are rejected.
    stop.registers[0x381] = pack_halves(9, 9);
    check_rejected(stop);
    stop.registers[0x381] = pack_halves(9, 4);
    check_rejected(stop);
    stop.registers[0x381] = pack_halves(16, 18);
    check_rejected(stop);
    stop.registers[0x381] = pack_halves(1, 9);
    stop.registers[0x382] = pack_halves(0, 12);
    check_rejected(stop);
    stop.registers[0x382] = pack_halves(2, 7);
    stop.registers[0x380] = 2;
    check_rejected(stop);

    stop = make_stop(0, 17, 11, 0xd5, 0x00, 17);
    stop.color_targets[0].address = 0;
    CHECK(std::holds_alternative<ColorClearPlan>(plan_color_clear(stop)));

    stop.registers[0x556] = 0;
    stop.effective_render_enable_mode = 0;
    stop.registers[0x651] = 1;
    CHECK(std::holds_alternative<ColorClearPlan>(plan_color_clear(stop)));
}

void test_srgb_format_and_scissor_region_geometry()
{
    // The sRGB format plans like its UNORM sibling; the region covers the
    // whole surface until clear control selects the scissor.
    RenderStop stop = make_stop(0, 21, 13, 0xd6, 0x1000, 84);
    ColorClearPlan plan = require_plan(plan_color_clear(stop));
    CHECK(plan.format == ColorTargetFormat::R8G8B8A8Srgb);
    CHECK(plan.width == 21 && plan.height == 13);
    CHECK(plan.row_bytes == 84 && plan.storage_size == 1092);
    CHECK(plan.region.x == 0 && plan.region.y == 0);
    CHECK(plan.region.width == 21 && plan.region.height == 13);

    stop.registers[0x43e] = 0x100;
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(3, 9);
    stop.registers[0x382] = pack_halves(2, 7);
    plan = require_plan(plan_color_clear(stop));
    CHECK(plan.region.x == 3 && plan.region.y == 2);
    CHECK(plan.region.width == 6 && plan.region.height == 5);
    // Storage geometry and pixel offsets keep describing the full surface.
    CHECK(plan.width == 21 && plan.height == 13);
    CHECK(plan.row_bytes == 84 && plan.storage_size == 1092);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 0)) == 0);
    CHECK(require_offset(pixel_byte_offset(plan, 3, 2)) == 2 * 84 + 3 * 4);
    CHECK(require_offset(pixel_byte_offset(plan, 20, 12)) == 12 * 84 + 20 * 4);

    // A block-linear region whose extent crosses GOB and block boundaries in
    // both axes still addresses texels through the unchanged full layout.
    RenderStop tiled = make_stop(1, 41, 139, 0xd6, 0x10, 41);
    tiled.registers[0x43e] = 0x100;
    tiled.registers[0x380] = 1;
    tiled.registers[0x381] = pack_halves(12, 25);
    tiled.registers[0x382] = pack_halves(20, 60);
    const ColorClearPlan tiled_plan = require_plan(plan_color_clear(tiled));
    CHECK(tiled_plan.format == ColorTargetFormat::R8G8B8A8Srgb);
    CHECK(tiled_plan.region.x == 12 && tiled_plan.region.y == 20);
    CHECK(tiled_plan.region.width == 13 && tiled_plan.region.height == 40);
    CHECK(tiled_plan.row_bytes == 192);
    CHECK(require_offset(pixel_byte_offset(tiled_plan, 12, 20)) == 3488);
    CHECK(require_offset(pixel_byte_offset(tiled_plan, 0, 0)) == 0);
}

void test_unsupported_clear_forms()
{
    const RenderStop valid = make_stop(0, 17, 11, 0xd5, 0x00, 17);

    RenderStop stop = valid;
    stop.kind = RenderKind::Draw;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x674] |= 1;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x674] &= ~(1U << 5);
    check_rejected(stop);
    stop = valid;
    stop.registers[0x674] |= 8U << 6;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x674] |= 1U << 10;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x674] |= 1U << 26;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x556] = 0;
    stop.effective_render_enable_mode = 0;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x556] = 2;
    stop.effective_render_enable_mode = 2;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x556] = 5;
    stop.effective_render_enable_mode = 5;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x651] = 2;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x651] = 3;
    check_rejected(stop);
    stop = valid;
    stop.registers[0xdf] = 0;
    check_rejected(stop);
    stop = valid;
    stop.registers[0xdf] = 2;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x43e] = 0x10;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x43e] = 0x1000;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x43e] = 2;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x574] = 1;
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].format = 0;
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].tile_mode = 0x1001;
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].tile_mode = 0x60;
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].depth = 2;
    stop.registers[0x206] = 2;
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].base_layer = 1;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x3fd] = pack_halves(1, 17);
    check_rejected(stop);
    stop = valid;
    stop.color_targets[0].address = std::numeric_limits<std::uint64_t>::max() - 32;
    check_rejected(stop);
    stop = valid;
    stop.registers[0x360] = 0x7f800000;
    check_rejected(stop);
    stop = make_stop(0, 65535, 65535, 0xd5, 0x50, 65535);
    check_rejected(stop);
}

void test_block_linear_padded_pitch_geometry()
{
    ColorClearPlan plan{};
    plan.width = 24;
    plan.height = 24;
    plan.layout = ColorTargetLayout::BlockLinear;
    plan.block_height_log2 = 1;
    plan.row_bytes = 192;
    plan.storage_size = 6144;
    CHECK(require_offset(pixel_byte_offset(plan, 16, 0)) == 1024);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 8)) == 512);
    CHECK(require_offset(pixel_byte_offset(plan, 0, 16)) == 3072);
    CHECK(require_offset(pixel_byte_offset(plan, 20, 20)) == 4256);

    auto malformed = plan;
    malformed.row_bytes = 64;
    malformed.storage_size = 2048;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
    malformed = plan;
    malformed.row_bytes = 160;
    malformed.storage_size = 5120;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
    malformed = plan;
    ++malformed.storage_size;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
}

void test_offset_bounds_and_malformed_plans()
{
    const ColorClearPlan plan = require_plan(plan_color_clear(make_stop(0, 19, 9, 0xd5, 0x40, 19)));
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(plan, 19, 0)));
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(plan, 0, 9)));

    ColorClearPlan malformed = plan;
    malformed.storage_size -= 1;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
    malformed = plan;
    malformed.row_bytes -= 1;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
    malformed = plan;
    malformed.block_height_log2 = 6;
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
    malformed = plan;
    malformed.layout = static_cast<ColorTargetLayout>(0xff);
    CHECK(std::holds_alternative<CommandResult>(pixel_byte_offset(malformed, 0, 0)));
}

}  // namespace

int main()
{
    test_block_linear_plan_and_fixed_offsets();
    test_pitch_plan_and_padding();
    test_effective_clear_and_full_scissor();
    test_srgb_format_and_scissor_region_geometry();
    test_unsupported_clear_forms();
    test_block_linear_padded_pitch_geometry();
    test_offset_bounds_and_malformed_plans();
    return failures == 0 ? 0 : 1;
}
