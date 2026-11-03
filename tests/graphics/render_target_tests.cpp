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
#include <armigo/graphics/render_target.h>
#include <armigo/graphics/resource_memory_registry.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <variant>
#include <utility>

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

constexpr std::uint32_t rt_control(std::uint32_t count,
                                   const std::array<std::uint32_t, 8>& permutation) noexcept
{
    std::uint32_t packed = count & 0xf;
    for (std::uint32_t index = 0; index < permutation.size(); ++index) {
        packed |= (permutation[index] & 7) << (4 + index * 3);
    }
    return packed;
}

// Populates the raw color-target registers and the frontend-decoded target
// description together, mirroring engines.cpp's snapshot behavior.
void set_target(RenderStop& stop,
                std::uint32_t physical,
                std::uint64_t address,
                std::uint32_t width_or_stride,
                std::uint32_t height,
                std::uint32_t format,
                std::uint32_t tile_mode,
                std::uint32_t layers,
                std::uint32_t base_layer = 0)
{
    const std::size_t base = 0x200 + physical * 0x10;
    stop.registers[base] = std::uint32_t(address >> 32);
    stop.registers[base + 1] = std::uint32_t(address);
    stop.registers[base + 2] = width_or_stride;
    stop.registers[base + 3] = height;
    stop.registers[base + 4] = format;
    stop.registers[base + 5] = tile_mode;
    stop.registers[base + 6] = layers;
    stop.registers[base + 7] = 0x1000;
    stop.registers[base + 8] = base_layer;
    TargetDescription& target = stop.color_targets[physical];
    target.address = address;
    target.width = width_or_stride;
    target.height = height;
    target.format = format;
    target.tile_mode = tile_mode;
    target.depth = layers;
    target.array_pitch = 0x1000;
    target.base_layer = base_layer;
}

void set_depth(RenderStop& stop,
               std::uint64_t address,
               std::uint32_t format,
               std::uint32_t tile_mode,
               std::uint32_t width,
               std::uint32_t height,
               std::uint32_t layers,
               bool layered,
               bool enabled = true)
{
    stop.registers[0x3f8] = std::uint32_t(address >> 32);
    stop.registers[0x3f9] = std::uint32_t(address);
    stop.registers[0x3fa] = format;
    stop.registers[0x3fb] = tile_mode;
    stop.registers[0x3fc] = 0x2000;
    stop.registers[0x48a] = width;
    stop.registers[0x48b] = height;
    stop.registers[0x48c] = (layered ? 0u : (1u << 16)) | layers;
    stop.registers[0x54e] = enabled ? 1 : 0;
    stop.depth.address = address;
    stop.depth.format = format;
    stop.depth.tile_mode = tile_mode;
    stop.depth.array_pitch = 0x2000;
    stop.depth.width = width;
    stop.depth.height = height;
    stop.depth.enabled = enabled;
}

RenderStop make_draw_stop()
{
    RenderStop stop{};
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    stop.registers[0x574] = 0;
    return stop;
}

RenderStop make_clear_stop(std::uint32_t argument, std::uint32_t clear_control = 0)
{
    RenderStop stop{};
    stop.kind = RenderKind::Clear;
    stop.engine = engine_class::threed;
    stop.registers[0x674] = argument;
    stop.registers[0xdf] = 1;
    stop.registers[0x574] = 0;
    stop.registers[0x651] = 0;
    stop.registers[0x43e] = clear_control;
    stop.registers[0x3fd] = pack_halves(0, 64);
    stop.registers[0x3fe] = pack_halves(0, 32);
    return stop;
}

RenderTargetPlan require_plan(const RendererResult<RenderTargetPlan>& result)
{
    CHECK(std::holds_alternative<RenderTargetPlan>(result));
    if (std::holds_alternative<RenderTargetPlan>(result)) {
        return std::get<RenderTargetPlan>(result);
    }
    return RenderTargetPlan{};
}

ClearPlan require_clear(const RendererResult<ClearPlan>& result)
{
    CHECK(std::holds_alternative<ClearPlan>(result));
    if (std::holds_alternative<ClearPlan>(result)) {
        return std::get<ClearPlan>(result);
    }
    return ClearPlan{};
}

void require_rejected(const RendererResult<RenderTargetPlan>& result)
{
    CHECK(std::holds_alternative<RendererError>(result));
    if (std::holds_alternative<RendererError>(result)) {
        CHECK(!std::get<RendererError>(result).command().ok());
    }
}

void require_clear_rejected(const RendererResult<ClearPlan>& result)
{
    CHECK(std::holds_alternative<RendererError>(result));
    if (std::holds_alternative<RendererError>(result)) {
        CHECK(!std::get<RendererError>(result).command().ok());
    }
}

void test_eight_distinct_outputs()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(8, {0, 1, 2, 3, 4, 5, 6, 7});
    const std::array<std::uint32_t, 8> formats{0xd5, 0xd6, 0xd1, 0xe0,
                                               0xca, 0xe5, 0xe4, 0xc0};
    for (std::uint32_t index = 0; index < 8; ++index) {
        set_target(stop, index, 0x100000ull * (index + 1), 32 + index, 48 + index,
                   formats[index], 0, 1);
    }
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    std::size_t bound = 0;
    std::uint32_t minimum_width = 0xffffffff;
    std::uint32_t minimum_height = 0xffffffff;
    for (std::uint32_t index = 0; index < 8; ++index) {
        CHECK(plan.colors[index].has_value());
        if (!plan.colors[index]) continue;
        ++bound;
        const AttachmentPlan& attachment = *plan.colors[index];
        CHECK(attachment.physical_target == index);
        CHECK(attachment.aspect == TextureAspect::Color);
        CHECK(attachment.texture.format ==
              (index == 1 ? TextureFormat::R8G8B8A8Srgb
                          : index == 0 ? TextureFormat::R8G8B8A8Unorm
                                       : attachment.texture.format));
        CHECK(attachment.texture.width == 32 + index);
        CHECK(attachment.texture.height == 48 + index);
        CHECK(attachment.texture.samples == 1);
        minimum_width = std::min(minimum_width, attachment.texture.width);
        minimum_height = std::min(minimum_height, attachment.texture.height);
    }
    CHECK(bound == 8);
    CHECK(plan.area == (RenderArea{0, 0, minimum_width, minimum_height}));
    CHECK(plan.samples == 1);
    CHECK(!plan.depth && !plan.stencil);
}

void test_sparse_and_permuted_outputs()
{
    RenderStop stop = make_draw_stop();
    // Logical outputs 0, 1, 2 bind physical targets 5, 0, 7; every other
    // logical slot stays a hole and physical targets 1-4, 6 stay unbound.
    stop.registers[0x487] = rt_control(3, {5, 0, 7, 0, 0, 0, 0, 0});
    set_target(stop, 5, 0x500000, 64, 32, 0xd5, 0, 1);
    set_target(stop, 0, 0x100000, 64, 40, 0xd1, 0, 1);
    set_target(stop, 7, 0x700000, 64, 32, 0xe5, 0, 1);
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    for (std::uint32_t logical = 3; logical < 8; ++logical) {
        CHECK(!plan.colors[logical].has_value());
    }
    CHECK(plan.colors[0].has_value() && plan.colors[1].has_value() &&
          plan.colors[2].has_value());
    if (plan.colors[0]) CHECK(plan.colors[0]->physical_target == 5);
    if (plan.colors[1]) CHECK(plan.colors[1]->physical_target == 0);
    if (plan.colors[2]) CHECK(plan.colors[2]->physical_target == 7);
    if (plan.colors[0]) CHECK(plan.colors[0]->texture.address == 0x500000);
    if (plan.colors[1]) CHECK(plan.colors[1]->texture.format ==
                              TextureFormat::R10G10B10A2Unorm);
    if (plan.colors[2]) CHECK(plan.colors[2]->texture.format == TextureFormat::R32Float);
    CHECK(plan.area == (RenderArea{0, 0, 64, 32}));
}

void test_disabled_slots_remain_holes()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(4, {0, 1, 2, 3, 0, 0, 0, 0});
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 1);
    // Physical targets 1 and 3 are disabled by a zero format and a zero
    // stride; logical slots 1 and 3 stay holes.
    set_target(stop, 1, 0x200000, 32, 16, 0, 0, 1);
    set_target(stop, 2, 0x300000, 32, 16, 0xd5, 0, 1);
    set_target(stop, 3, 0x400000, 0, 16, 0xd5, 0, 1);
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    CHECK(plan.colors[0].has_value());
    CHECK(!plan.colors[1].has_value());
    CHECK(plan.colors[2].has_value());
    CHECK(!plan.colors[3].has_value());
    CHECK(plan.area == (RenderArea{0, 0, 32, 16}));
}

void test_typed_formats_and_layouts()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(3, {0, 1, 2, 0, 0, 0, 0, 0});
    // Pitch layout: the width register is the byte stride.
    set_target(stop, 0, 0x100000, 256, 16, 0xd5, 0x1000, 1);
    // Block-linear 32x16 with block height 2 (log2 1): rows GOB-align to 128
    // bytes and storage rows align to 16.
    set_target(stop, 1, 0x200000, 32, 16, 0xe5, 0x10, 1);
    // Array target with a nonzero base layer.
    set_target(stop, 2, 0x300000, 16, 16, 0xca, 0, 4, 1);
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    CHECK(plan.colors[0]->texture.layout == TextureLayout::Pitch);
    CHECK(plan.colors[0]->texture.width == 64);
    CHECK(plan.colors[0]->texture.row_bytes == 256);
    CHECK(plan.colors[0]->texture.storage_height == 16);
    CHECK(plan.colors[0]->texture.backing_extent == 256 * 16);
    CHECK(plan.colors[0]->texture.type == TextureType::Texture2D);
    CHECK(plan.colors[1]->texture.layout == TextureLayout::BlockLinear);
    CHECK(plan.colors[1]->texture.format == TextureFormat::R32Float);
    CHECK(plan.colors[1]->texture.row_bytes == 128);
    CHECK(plan.colors[1]->texture.storage_height == 16);
    CHECK(plan.colors[1]->texture.backing_extent == 128 * 16);
    CHECK(plan.colors[2]->texture.layers == 4);
    CHECK(plan.colors[2]->texture.type == TextureType::Texture2DArray);
    CHECK(plan.colors[2]->layer == 1);
    CHECK(plan.colors[2]->layer_count == 1);
    CHECK(plan.colors[2]->level == 0);
}

void test_all_gob_heights_and_pitch_rows()
{
    for (std::uint32_t height_log2 = 0; height_log2 <= 5; ++height_log2) {
        RenderStop stop = make_draw_stop();
        stop.registers[0x487] = 1;
        set_target(stop, 0, 0x100000, 37, 257, 0xd5, height_log2 << 4, 1);
        const auto plan = require_plan(plan_render_targets(stop));
        if (!plan.colors[0]) continue;
        const auto& texture = plan.colors[0]->texture;
        const std::size_t block_rows = std::size_t{8} << height_log2;
        const std::size_t storage_rows = (257 + block_rows - 1) / block_rows * block_rows;
        CHECK(texture.width == 37 && texture.height == 257);
        CHECK(texture.row_bytes == 192);
        CHECK(texture.block_height_log2 == height_log2);
        CHECK(texture.storage_height == storage_rows);
        CHECK(texture.backing_extent == 192 * storage_rows);
        CHECK(std::holds_alternative<TextureLayoutDescription>(texture_view_layout(texture)));
    }

    RenderStop pitch = make_draw_stop();
    pitch.registers[0x487] = 1;
    set_target(pitch, 0, 0x200000, 176, 19, 0xd5, 0x1000, 1);
    const auto plan = require_plan(plan_render_targets(pitch));
    if (!plan.colors[0]) return;
    CHECK(plan.colors[0]->texture.width == 44);
    CHECK(plan.colors[0]->texture.row_bytes == 176);
    CHECK(plan.colors[0]->texture.storage_height == 19);
    CHECK(plan.colors[0]->texture.backing_extent == 176 * 19);
}

void test_one_slice_3d_storage()
{
    // Independently authored extents span two X GOBs and three Y GOBs.
    // The final unused Z planes are trimmed, but intervening planes remain.
    for (std::uint32_t depth_log2 = 0; depth_log2 <= 5; ++depth_log2) {
        RenderStop stop = make_draw_stop();
        stop.registers[0x487] = 1;
        set_target(stop, 0, 0x300000, 73, 17, 0xf3,
                   0x10000 | (depth_log2 << 8), 1);
        const auto plan = require_plan(plan_render_targets(stop));
        if (!plan.colors[0]) continue;
        const auto& texture = plan.colors[0]->texture;
        CHECK(texture.type == TextureType::Texture3D);
        CHECK(texture.depth == 1 && texture.layers == 1);
        CHECK(texture.block_depth_log2 == depth_log2);
        CHECK(texture.row_bytes == 128 && texture.storage_height == 24);
        CHECK(texture.backing_extent == (5 * (std::size_t{1} << depth_log2) + 1) * 512);
        CHECK(texture.raw_words[5] == (0x10000 | (depth_log2 << 8)));
        const auto layout = texture_view_layout(texture);
        CHECK(std::holds_alternative<TextureLayoutDescription>(layout));
        if (const auto* value = std::get_if<TextureLayoutDescription>(&layout)) {
            CHECK(value->total_size == texture.backing_extent);
            CHECK(value->mips[0].gob_block_depth_log2 == depth_log2);
        }
    }
    RenderStop clear = make_clear_stop(0xf << 2);
    set_target(clear, 0, 0x300000, 73, 17, 0xf3, 0x10300, 1);
    clear.registers[0x3fd] = pack_halves(0, 73);
    clear.registers[0x3fe] = pack_halves(0, 17);
    const auto plan = require_clear(plan_render_clear(clear));
    CHECK(plan.target.texture.type == TextureType::Texture3D);
    CHECK(plan.target.texture.block_depth_log2 == 3);
    CHECK(plan.target.texture.backing_extent == 41 * 512);
    CHECK(plan.fold_into_load);
}

void test_programmed_layer_stride()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = 1;
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0x10, 3, 2);
    auto plan = require_plan(plan_render_targets(stop));
    if (!plan.colors[0]) return;
    const auto& texture = plan.colors[0]->texture;
    CHECK(texture.layer_stride == 0x1000 * 4);
    CHECK(texture.backing_extent == 2 * 0x1000 * 4 + 2048);
    CHECK(plan.colors[0]->layer == 2);
    const auto layout = texture_view_layout(texture);
    CHECK(std::holds_alternative<TextureLayoutDescription>(layout));
    if (const auto* value = std::get_if<TextureLayoutDescription>(&layout)) {
        CHECK(value->layer_stride == texture.layer_stride);
        CHECK(value->total_size == texture.backing_extent);
    }
    for (const std::uint32_t encoded : {0u, 1u, 512u}) {
        stop.color_targets[0].array_pitch = stop.registers[0x207] = encoded;
        const auto fallback = require_plan(plan_render_targets(stop));
        if (fallback.colors[0]) {
            CHECK(fallback.colors[0]->texture.layer_stride == 2048);
            CHECK(fallback.colors[0]->texture.backing_extent == 6144);
        }
    }
    // Linear arrays derive the stride from their rows, ignoring LayerSize.
    set_target(stop, 0, 0x100000, 176, 19, 0xd5, 0x1000, 3, 1);
    plan = require_plan(plan_render_targets(stop));
    if (!plan.colors[0]) return;
    CHECK(plan.colors[0]->texture.layer_stride == 176 * 19);
    CHECK(plan.colors[0]->texture.backing_extent == 176 * 19 * 3);

    stop.registers[0x487] = 0;
    set_depth(stop, 0x500000, 0x0a, 0, 32, 16, 3, true);
    plan = require_plan(plan_render_targets(stop));
    CHECK(plan.depth.has_value());
    if (plan.depth) {
        CHECK(plan.depth->texture.layer_stride == 0x2000 * 4);
        CHECK(plan.depth->texture.backing_extent == 2 * 0x2000 * 4 + 2048);
    }
}

void test_layout_field_diagnostics()
{
    auto expect = [](const RenderStop& stop, std::uint32_t method,
                     std::string_view detail) {
        const auto result = plan_render_targets(stop);
        require_rejected(result);
        if (const auto* error = std::get_if<RendererError>(&result)) {
            const auto command = error->command();
            CHECK(command.method == method);
            CHECK(command.argument == stop.registers[method]);
            CHECK(std::string_view(command.detail).find(detail) != std::string_view::npos);
        }
    };
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = 1;
    for (const auto [mode, field] : std::array{
             std::pair{1u, "block width"}, std::pair{0x60u, "block height"},
             std::pair{0x10600u, "block depth"}, std::pair{0x2000u, "reserved"},
             std::pair{0x100u, "requires a 3D"},
             std::pair{0x11000u, "pitch-linear 3D"},
             std::pair{0x1010u, "nonzero GOB"}}) {
        set_target(stop, 0, 0x100000, 32, 16, 0xd5, mode, 1);
        expect(stop, 0x205, field);
    }
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0x10000, 2);
    expect(stop, 0x206, "3D depth");
    set_target(stop, 0, 0x100000, 129, 16, 0xd5, 0x1000, 1);
    expect(stop, 0x202, "byte pitch");
    set_target(stop, 0, 0x100000, 32, 0, 0xd5, 0, 1);
    expect(stop, 0x203, "height");
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 0x10001);
    stop.color_targets[0].depth = 1; // The frontend retains only the low half.
    expect(stop, 0x206, "array mode");
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 3);
    stop.color_targets[0].array_pitch = stop.registers[0x207] = 0xffffffffu;
    expect(stop, 0x207, "layer stride");
    stop.registers[0x487] = 0;
    set_depth(stop, 0x500000, 0x0a, 0x10000, 32, 16, 1, false);
    expect(stop, 0x3fb, "3D depth");
    set_depth(stop, 0x500000, 0x0a, 0, 32, 16, 0, true);
    expect(stop, 0x48c, "zero array");
    set_depth(stop, 0x500000, 0x0a, 0, 32, 16, 1, true);
    stop.registers[0x48c] |= 1u << 17;
    expect(stop, 0x48c, "array mode");
}

void test_color_target_alpha_semantics()
{
    for (const std::uint32_t format : {0xc3u, 0xc4u, 0xc5u, 0xceu, 0xf9u, 0xfau}) {
        RenderStop stop = make_draw_stop();
        stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
        set_target(stop, 0, 0x100000, 32, 16, format, 0, 1);
        const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
        CHECK(plan.colors[0].has_value());
        if (plan.colors[0]) {
            CHECK(!plan.colors[0]->has_alpha);
        }
    }

    RenderStop rgba16 = make_draw_stop();
    rgba16.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_target(rgba16, 0, 0x100000, 32, 16, 0xc6, 0, 1);
    const RenderTargetPlan plan = require_plan(plan_render_targets(rgba16));
    CHECK(plan.colors[0].has_value());
    if (plan.colors[0]) {
        CHECK(plan.colors[0]->has_alpha);
    }
}

void test_depth_only_and_depth_stencil_passes()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(0, {0, 0, 0, 0, 0, 0, 0, 0});
    set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 1, false);
    RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    CHECK(!plan.colors[0].has_value());
    CHECK(plan.depth.has_value());
    CHECK(!plan.stencil.has_value());
    CHECK(plan.depth->aspect == TextureAspect::Depth);
    CHECK(plan.depth->texture.format == TextureFormat::D32Float);
    CHECK(plan.depth->texture.address == 0x900000);
    CHECK(plan.area == (RenderArea{0, 0, 64, 32}));

    RenderStop packed = make_draw_stop();
    packed.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_target(packed, 0, 0x100000, 64, 64, 0xd5, 0, 1);
    set_depth(packed, 0x900000, 0x14, 0, 64, 32, 1, false);
    plan = require_plan(plan_render_targets(packed));
    CHECK(plan.depth.has_value() && plan.stencil.has_value());
    CHECK(plan.depth->aspect == TextureAspect::Depth);
    CHECK(plan.stencil->aspect == TextureAspect::Stencil);
    CHECK(plan.depth->texture.format == TextureFormat::D24UnormS8Uint);
    // The aspects share the physical storage and differ only by aspect.
    CHECK(plan.depth->texture.address == plan.stencil->texture.address);
    CHECK(plan.stencil->texture.format == TextureFormat::D24UnormS8Uint);
    CHECK(plan.area == (RenderArea{0, 0, 64, 32}));
}

void test_null_depth_and_attachment_less_passes()
{
    RenderStop colored = make_draw_stop();
    colored.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_target(colored, 0, 0x100000, 48, 24, 0xd5, 0, 1);
    set_depth(colored, 0, 0xffffffffu, 0xffffffffu, 0, 0, 0, true);
    colored.registers[0x48c] = 0xffffffffu;
    const RenderTargetPlan colored_plan = require_plan(plan_render_targets(colored));
    CHECK(colored_plan.colors[0].has_value());
    CHECK(!colored_plan.depth.has_value());
    CHECK(!colored_plan.stencil.has_value());
    CHECK(colored_plan.area == (RenderArea{0, 0, 48, 24}));

    RenderStop empty = make_draw_stop();
    empty.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_target(empty, 0, 0, 256, 256, 0xca, 0x40, 1);
    set_depth(empty, 0, 0xffffffffu, 0xffffffffu, 0, 0, 0, true);
    empty.registers[0x48c] = 0xffffffffu;
    empty.registers[0x3fd] = pack_halves(0, 640);
    empty.registers[0x3fe] = pack_halves(0, 360);
    const RenderTargetPlan empty_plan = require_plan(plan_render_targets(empty));
    for (const auto& color : empty_plan.colors) {
        CHECK(!color.has_value());
    }
    CHECK(!empty_plan.depth.has_value());
    CHECK(!empty_plan.stencil.has_value());
    CHECK(empty_plan.area == (RenderArea{0, 0, 640, 360}));
    CHECK(empty_plan.samples == 1);

    empty.registers[0x487] = 0;
    set_depth(empty, 0, 0x13, 0, 192, 192, 1, false);
    empty.registers[0x3fd] = pack_halves(0, 192);
    empty.registers[0x3fe] = pack_halves(0, 192);
    const auto depth_null_plan = require_plan(plan_render_targets(empty));
    CHECK(!depth_null_plan.depth && !depth_null_plan.stencil);
    CHECK(depth_null_plan.area == (RenderArea{0, 0, 192, 192}));

    RenderStop bound = colored;
    set_depth(bound, 0x900000, 0x14, 0, 48, 24, 1, false);
    const RenderTargetPlan bound_plan = require_plan(plan_render_targets(bound));
    CHECK(bound_plan.depth.has_value());
    CHECK(bound_plan.stencil.has_value());
    CHECK(bound_plan.depth->texture.address == 0x900000);

    // Planning retains nonzero bindings; the same registry lookup used by the
    // renderer must reject missing mappings instead of treating them as null.
    struct UnmappedBackend final : ResourceMemoryBackend {
        RendererStatus unexpected_access() {
            CHECK(false);
            return RendererError({CommandStatus::MemoryFault, "unexpected unmapped backing access"});
        }
        RendererStatus read_cpu(BackingRange, std::span<std::byte>) override {
            return unexpected_access();
        }
        RendererStatus write_cpu(BackingRange, std::span<const std::byte>) override {
            return unexpected_access();
        }
        RendererStatus wait_gpu(CompletionToken) override { return unexpected_access(); }
    };
    const auto memory_result = make_resource_memory({1, std::make_shared<UnmappedBackend>()});
    CHECK(std::holds_alternative<std::shared_ptr<ResourceMemoryRegistry>>(memory_result));
    if (const auto* memory = std::get_if<std::shared_ptr<ResourceMemoryRegistry>>(&memory_result)) {
        const auto retained = (*memory)->retain(bound_plan.depth->texture.address,
            bound_plan.depth->texture.backing_extent, ResourceAccess::ReadWrite);
        CHECK(std::holds_alternative<RendererError>(retained));
        if (const auto* error = std::get_if<RendererError>(&retained))
            CHECK(error->command().detail == "retained GPU address is unmapped");
    }

    bound.depth.format = bound.registers[0x3fa] = 0xffffffffu;
    const auto invalid = plan_render_targets(bound);
    require_rejected(invalid);
    if (const auto* error = std::get_if<RendererError>(&invalid)) {
        CHECK(error->command().method == 0x3fa);
        CHECK(std::string_view(error->command().detail).find("format") !=
              std::string_view::npos);
    }
}

void test_depth_layer_flags_from_register_image()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(0, {0, 0, 0, 0, 0, 0, 0, 0});
    set_depth(stop, 0x900000, 0x16, 0x20, 32, 32, 4, true);
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    CHECK(plan.depth.has_value() && plan.stencil.has_value());
    CHECK(plan.depth->texture.layers == 4);
    // Block height log2 2 aligns the 32-row surface to 32 storage rows.
    CHECK(plan.depth->texture.storage_height == 32);
    CHECK(plan.depth->texture.format == TextureFormat::S8UintD24Unorm);
    CHECK(plan.depth->texture.block_height_log2 == 2);
}

void test_x8d24_has_no_stencil_aspect()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(0, {0, 0, 0, 0, 0, 0, 0, 0});
    set_depth(stop, 0x900000, 0x15, 0, 32, 32, 1, false);
    const RenderTargetPlan plan = require_plan(plan_render_targets(stop));
    CHECK(plan.depth.has_value());
    CHECK(!plan.stencil.has_value());
    CHECK(plan.depth->texture.format == TextureFormat::X8UintD24Unorm);
}

void test_plan_rejections()
{
    RenderStop stop = make_draw_stop();
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 1);
    // Reserved MSAA mode encoding.
    stop.registers[0x574] = 1;
    require_rejected(plan_render_targets(stop));
    stop.registers[0x574] = 0;
    // Multisampled targets stay outside the v1 profile.
    stop.registers[0x574] = 2;
    require_rejected(plan_render_targets(stop));
    stop.registers[0x574] = 0;
    // Output count nibble above eight.
    stop.registers[0x487] = 9;
    require_rejected(plan_render_targets(stop));
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    // BGR encodings have no decoded table entry.
    set_target(stop, 0, 0x100000, 32, 16, 0xd0, 0, 1);
    require_rejected(plan_render_targets(stop));
    // Multiple 3D slices remain outside the attachment profile.
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0x10000, 2);
    require_rejected(plan_render_targets(stop));
    // Zero layers or a base layer beyond the count.
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 0);
    require_rejected(plan_render_targets(stop));
    set_target(stop, 0, 0x100000, 32, 16, 0xd5, 0, 2, 2);
    require_rejected(plan_render_targets(stop));
    // An attachment-less pass still needs a nonempty surface extent.
    set_target(stop, 0, 0, 0, 0, 0, 0, 1);
    require_rejected(plan_render_targets(stop));
    // Stencil-only depth format has no table entry.
    set_depth(stop, 0x900000, 0x17, 0, 32, 32, 1, false);
    require_rejected(plan_render_targets(stop));
    // Depth target without the enable bit.
    set_depth(stop, 0x900000, 0x0a, 0, 32, 32, 1, false, false);
    require_rejected(plan_render_targets(stop));
}

void test_full_and_scissored_color_clears()
{
    const std::array<float, 4> color{0.25f, 0.5f, 0.75f, 1.0f};
    auto make = [&](std::uint32_t argument, std::uint32_t clear_control) {
        RenderStop stop = make_clear_stop(argument, clear_control);
        set_target(stop, 2, 0x100000, 256, 32, 0xd5, 0x1000, 1);
        for (std::size_t index = 0; index < color.size(); ++index) {
            stop.registers[0x360 + index] = std::bit_cast<std::uint32_t>(color[index]);
        }
        return stop;
    };
    const RenderStop full = make(0xf << 2 | 2 << 6, 0);
    ClearPlan plan = require_clear(plan_render_clear(full));
    CHECK(plan.target.physical_target == 2);
    CHECK(plan.target.aspect == TextureAspect::Color);
    CHECK(plan.target.texture.format == TextureFormat::R8G8B8A8Unorm);
    CHECK(plan.target.texture.layout == TextureLayout::Pitch);
    CHECK(plan.target.texture.width == 64);
    CHECK(plan.color_mask == 0xf);
    CHECK(plan.region == (RenderArea{0, 0, 64, 32}));
    CHECK(plan.fold_into_load);
    CHECK(!plan.clear_depth && !plan.clear_stencil);
    for (std::size_t index = 0; index < 4; ++index) {
        CHECK(plan.color_bits[index] == std::bit_cast<std::uint32_t>(color[index]));
    }

    const RenderStop masked = make(0x5 << 2 | 2 << 6, 0);
    plan = require_clear(plan_render_clear(masked));
    CHECK(plan.color_mask == 0x5);
    CHECK(!plan.fold_into_load);

    const std::uint32_t control = 0x100;
    RenderStop scissored = make(0xf << 2 | 2 << 6, control);
    scissored.registers[0x380] = 1;
    scissored.registers[0x381] = pack_halves(3, 20);
    scissored.registers[0x382] = pack_halves(4, 30);
    plan = require_clear(plan_render_clear(scissored));
    CHECK(plan.region == (RenderArea{3, 4, 17, 26}));
    CHECK(!plan.fold_into_load);
}

void test_pitch_clear_preserves_padding()
{
    RenderStop stop = make_clear_stop(0xf << 2);
    set_target(stop, 0, 0x100000, 192, 32, 0xd5, 0x1000, 1);
    stop.registers[0x3fd] = pack_halves(0, 37);
    const ClearPlan plan = require_clear(plan_render_clear(stop));
    CHECK(plan.target.texture.width == 48);
    CHECK(plan.target.texture.row_bytes == 192);
    CHECK(plan.region == (RenderArea{0, 0, 37, 32}));
    CHECK(!plan.fold_into_load);

    stop.registers[0x3fd] = pack_halves(0, 48);
    CHECK(require_clear(plan_render_clear(stop)).fold_into_load);
    // A surface wider than the target is clipped to the host framebuffer.
    stop.registers[0x3fd] = pack_halves(0, 49);
    const ClearPlan clipped = require_clear(plan_render_clear(stop));
    CHECK(clipped.region == (RenderArea{0, 0, 48, 32}));
    CHECK(clipped.fold_into_load);
}

void test_clear_scissor_legacy_parity()
{
    const std::array rectangles{
        RenderArea{3, 4, 17, 26}, RenderArea{0, 0, 64, 32},
        RenderArea{8, 0, 0, 8}, RenderArea{9, 0, 0, 8},
        RenderArea{0, 0, 65, 32}};
    for (const auto control : {0u, 1u, 0x100u, 0x101u, 0x10u, 0x110u,
                               0x1000u, 0x1100u, 0x200u}) {
        for (const auto enable : {0u, 1u, 2u}) {
            for (std::size_t index = 0; index < rectangles.size(); ++index) {
                const auto& rectangle = rectangles[index];
                RenderStop stop = make_clear_stop(0xf << 2, control);
                set_target(stop, 0, 0x100000, 256, 32, 0xd5, 0x1000, 1);
                stop.registers[0x380] = enable;
                stop.registers[0x381] = pack_halves(rectangle.x,
                    index == 3 ? 8 : rectangle.x + rectangle.width);
                stop.registers[0x382] = pack_halves(rectangle.y,
                    rectangle.y + rectangle.height);
                const auto legacy = plan_color_clear(stop);
                const auto persistent = plan_render_clear(stop);
                const bool admitted = std::holds_alternative<ColorClearPlan>(legacy);
                // The persistent planner clips where the legacy one rejects;
                // whatever the legacy planner admits must agree exactly.
                if (admitted) CHECK(std::holds_alternative<ClearPlan>(persistent));
                if (!admitted || !std::holds_alternative<ClearPlan>(persistent)) continue;
                const auto& expected = std::get<ColorClearPlan>(legacy).region;
                const auto& actual = std::get<ClearPlan>(persistent);
                CHECK(actual.region == (RenderArea{expected.x, expected.y,
                                                   expected.width, expected.height}));
                if ((control & 0x100) == 0 || enable == 0) {
                    CHECK(actual.region == (RenderArea{0, 0, 64, 32}));
                    CHECK(actual.fold_into_load);
                }
            }
        }
    }
    // The shared region decoder must also ignore disabled depth/stencil scissors.
    RenderStop depth = make_clear_stop(3, 0x100);
    set_depth(depth, 0x900000, 0x14, 0, 64, 32, 1, false);
    const ClearPlan plan = require_clear(plan_render_clear(depth));
    CHECK(plan.region == (RenderArea{0, 0, 64, 32}));
    CHECK(plan.fold_into_load);
}

void test_depth_stencil_clears()
{
    auto make = [&](std::uint32_t argument, std::uint32_t clear_control,
                    std::uint32_t format) {
        RenderStop stop = make_clear_stop(argument, clear_control);
        set_depth(stop, 0x900000, format, 0, 64, 32, 1, false);
        stop.registers[0x364] = std::bit_cast<std::uint32_t>(0.5f);
        stop.registers[0x368] = 0x5a;
        return stop;
    };
    ClearPlan plan = require_clear(plan_render_clear(make(1, 0, 0x0a)));
    CHECK(plan.clear_depth && !plan.clear_stencil);
    CHECK(plan.target.aspect == TextureAspect::Depth);
    CHECK(plan.target.texture.format == TextureFormat::D32Float);
    CHECK(plan.depth_value == 0.5);
    CHECK(plan.region == (RenderArea{0, 0, 64, 32}));
    CHECK(plan.fold_into_load);

    // A packed target cleared for both aspects folds and writes all bits.
    plan = require_clear(plan_render_clear(make(3, 0, 0x14)));
    CHECK(plan.clear_depth && plan.clear_stencil);
    CHECK(plan.target.aspect == TextureAspect::DepthStencil);
    CHECK(plan.stencil_value == 0x5a);
    CHECK(plan.stencil_write_mask == 0xff);
    CHECK(plan.fold_into_load);

    // Clearing one aspect of a packed target preserves the other: a real
    // operation, never a folded load.
    plan = require_clear(plan_render_clear(make(1, 0, 0x14)));
    CHECK(plan.clear_depth && !plan.clear_stencil);
    CHECK(!plan.fold_into_load);

    // The stencil write mask honors ClearFlags bit 0 through the front mask.
    RenderStop masked = make(2, 0x1, 0x14);
    masked.registers[0x4e7] = 0x0f;
    plan = require_clear(plan_render_clear(masked));
    CHECK(plan.stencil_write_mask == 0x0f);
    CHECK(!plan.fold_into_load);

    // A scissored depth/stencil clear stays a real operation.
    RenderStop scissored = make(3, 0x100, 0x14);
    scissored.registers[0x380] = 1;
    scissored.registers[0x381] = pack_halves(8, 16);
    scissored.registers[0x382] = pack_halves(8, 16);
    plan = require_clear(plan_render_clear(scissored));
    CHECK(plan.region == (RenderArea{8, 8, 8, 8}));
    CHECK(!plan.fold_into_load);
}

void test_unbound_depth_clears()
{
    for (const auto aspect : {1u, 2u, 3u}) {
        auto stop = make_clear_stop(aspect);
        set_depth(stop, 0, 0x14, 0, 64, 32, 1, false);
        // An unrelated color slot can retain stale storage after unmapping.
        set_target(stop, 0, 0xa00000, 256, 32, 0xd5, 0x1000, 1);
        CHECK(require_clear(plan_render_clear(stop)).no_op);

        set_depth(stop, 0x900000, 0x14, 0, 64, 32, 1, false);
        const auto bound = require_clear(plan_render_clear(stop));
        CHECK(!bound.no_op);
        CHECK(bound.target.texture.address == 0x900000);
    }

    auto stop = make_clear_stop(1);
    set_depth(stop, 0, 0x14, 0, 64, 32, 1, false);
    // Attachment-specific state is irrelevant while no depth storage is bound.
    stop.depth.format = 0xffffffffu;
    stop.depth.tile_mode = 0xffffffffu;
    stop.registers[0x48a] = stop.registers[0x48b] = 0;
    stop.registers[0x48c] = 0xffffffffu;
    stop.registers[0x364] = 0x7fc00000u;
    CHECK(require_clear(plan_render_clear(stop)).no_op);
    stop.registers[0x674] = 1 | (0xfu << 2);
    require_clear_rejected(plan_render_clear(stop));
    stop.registers[0x674] = 1;
    stop.effective_render_enable_mode = 0;
    require_clear_rejected(plan_render_clear(stop));
    stop.effective_render_enable_mode = 1;
    stop.registers[0xdf] = 0;
    require_clear_rejected(plan_render_clear(stop));
}

void test_null_color_targets()
{
    RenderStop draw = make_draw_stop();
    draw.registers[0x487] = rt_control(2, {0, 1, 0, 0, 0, 0, 0, 0});
    set_target(draw, 0, 0, 32, 16, 0xd5, 0, 1);
    set_target(draw, 1, 0x200000, 32, 16, 0xd5, 0, 1);
    const RenderTargetPlan plan = require_plan(plan_render_targets(draw));
    CHECK(!plan.colors[0].has_value());
    CHECK(plan.colors[1].has_value());

    const std::uint32_t argument = 0xf << 2;
    RenderStop clear = make_clear_stop(argument);
    // The null binding keeps a previous target's format and geometry.
    set_target(clear, 0, 0, 256, 256, 0xca, 0x40, 1);
    CHECK(require_clear(plan_render_clear(clear)).no_op);
    // A combined argument with every binding absent is an empty clear too.
    clear.registers[0x674] = argument | 1;
    CHECK(require_clear(plan_render_clear(clear)).no_op);
    clear.registers[0x674] = argument;
    clear.effective_render_enable_mode = 0;
    require_clear_rejected(plan_render_clear(clear));
    clear.effective_render_enable_mode = 1;
    set_target(clear, 0, 0x100000, 256, 32, 0xd5, 0x1000, 1);
    const auto bound = require_clear(plan_render_clear(clear));
    CHECK(!bound.no_op);
    CHECK(bound.target.texture.address == 0x100000);
}

void test_clear_rejections()
{
    auto make = [&](std::uint32_t argument, std::uint32_t clear_control = 0) {
        RenderStop stop = make_clear_stop(argument, clear_control);
        set_target(stop, 2, 0x100000, 256, 32, 0xd5, 0x1000, 1);
        set_depth(stop, 0x900000, 0x14, 0, 64, 32, 1, false);
        return stop;
    };
    // No aspect selected.
    require_clear_rejected(plan_render_clear(make(0)));
    // A combined color and depth/stencil clear is one plan with both
    // attachments and a shared layer range.
    const ClearPlan combined =
        require_clear(plan_render_clear(make(3 | (0xf << 2) | (2 << 6))));
    CHECK(combined.color_mask == 0xf);
    CHECK(combined.clear_depth && combined.clear_stencil);
    CHECK(combined.target.aspect == TextureAspect::Color);
    CHECK(combined.depth_stencil.has_value());
    // Combined clears stay a real single-pass operation; only single-aspect
    // clears fold into a load action (the Metal 4 combined-pass fold drops
    // the depth aspect on the tested hardware).
    CHECK(!combined.fold_into_load && !combined.fold_layers_into_load);
    // A layer field beyond the bound target's array layer count is rejected.
    require_clear_rejected(plan_render_clear(make(0xf << 2 | 2 << 6 | 1 << 10)));
    // Out-of-range physical target.
    require_clear_rejected(plan_render_clear(make(0xf << 2 | 9 << 6)));
    // Conditional rendering and rasterizer discard.
    RenderStop conditional = make(0xf << 2 | 2 << 6);
    conditional.registers[0x651] = 2;
    require_clear_rejected(plan_render_clear(conditional));
    conditional.registers[0x651] = 0;
    conditional.effective_render_enable_mode = 0;
    require_clear_rejected(plan_render_clear(conditional));
    conditional.effective_render_enable_mode = 1;
    conditional.registers[0xdf] = 0;
    require_clear_rejected(plan_render_clear(conditional));
    conditional.registers[0xdf] = 1;
    // Unknown clear control bits and the clear-rectangle/viewport modes.
    require_clear_rejected(plan_render_clear(make(0xf << 2 | 2 << 6, 0x8)));
    require_clear_rejected(plan_render_clear(make(0xf << 2 | 2 << 6, 0x10)));
    require_clear_rejected(plan_render_clear(make(0xf << 2 | 2 << 6, 0x1000)));
    // Multisampled clears stay unsupported.
    RenderStop multisampled = make(0xf << 2 | 2 << 6);
    multisampled.registers[0x574] = 2;
    require_clear_rejected(plan_render_clear(multisampled));
    // Screen scissor with a nonzero origin or an empty extent.
    RenderStop offset = make(0xf << 2 | 2 << 6);
    offset.registers[0x3fd] = pack_halves(4, 64);
    require_clear_rejected(plan_render_clear(offset));
    // Scissor rectangles that are empty, inverted or overshoot the surface.
    const auto scissored = [&](std::uint32_t x1, std::uint32_t x2,
                               std::uint32_t y1, std::uint32_t y2) {
        // Target 2 is the bound color target; target 0 would be rejected as
        // disabled before the scissor is examined.
        RenderStop stop = make(0xf << 2 | 2 << 6, 0x100);
        stop.registers[0x380] = 1;
        stop.registers[0x381] = pack_halves(x1, x2);
        stop.registers[0x382] = pack_halves(y1, y2);
        return stop;
    };
    // Empty scissor intersections clear nothing; a scissor beyond the
    // surface is clipped to it.
    CHECK(require_clear(plan_render_clear(scissored(8, 8, 0, 8))).no_op);
    CHECK(require_clear(plan_render_clear(scissored(9, 8, 0, 8))).no_op);
    CHECK(require_clear(plan_render_clear(scissored(0, 65, 0, 8))).region ==
          (RenderArea{0, 0, 64, 8}));
    // A reserved scissor enable value with the scissor bit set.
    RenderStop unenabled = make(0xf << 2 | 2 << 6, 0x100);
    unenabled.registers[0x380] = 2;
    require_clear_rejected(plan_render_clear(unenabled));
    // A surface smaller than the target clears only the surface.
    RenderStop mismatched = make(0xf << 2 | 2 << 6);
    mismatched.registers[0x3fe] = pack_halves(0, 31);
    const ClearPlan partial = require_clear(plan_render_clear(mismatched));
    CHECK(partial.region == (RenderArea{0, 0, 64, 31}));
    CHECK(!partial.fold_into_load);
    // Non-finite clear color and depth values.
    RenderStop color = make(0xf << 2 | 2 << 6);
    color.registers[0x360] = 0x7f800000;
    require_clear_rejected(plan_render_clear(color));
    RenderStop depth = make(1);
    depth.registers[0x364] = 0x7fc00000;
    require_clear_rejected(plan_render_clear(depth));
    // Aspect requests the format does not carry.
    RenderStop aspect = make(2);
    set_depth(aspect, 0x900000, 0x0a, 0, 64, 32, 1, false);
    require_clear_rejected(plan_render_clear(aspect));
    // Maxwell SET_ZT_SELECT with TARGET_COUNT zero leaves no attachment for
    // depth, stencil, or combined clears, including stale nonzero bindings.
    for (const auto aspects : {1u, 2u, 3u}) {
        RenderStop disabled = make(aspects);
        disabled.registers[0x54e] = 0;
        disabled.depth.enabled = false;
        CHECK(require_clear(plan_render_clear(disabled)).no_op);
        disabled.depth.address = 0;
        CHECK(require_clear(plan_render_clear(disabled)).no_op);
    }
    // A layered depth target clears its whole programmed array as one real
    // operation (a depth-only clear of a packed format preserves stencil).
    RenderStop layered = make(1);
    layered.registers[0x48c] = 2;
    const ClearPlan layered_plan = require_clear(plan_render_clear(layered));
    CHECK(layered_plan.target.layer == 0);
    CHECK(layered_plan.target.layer_count == 2);
    CHECK(layered_plan.target.texture.layers == 2);
    CHECK(layered_plan.target.texture.type == TextureType::Texture2DArray);
    CHECK(!layered_plan.fold_into_load && !layered_plan.fold_layers_into_load);
    // A depth surface taller than the depth target is clipped to it.
    RenderStop depth_mismatch = make(1);
    depth_mismatch.registers[0x3fe] = pack_halves(0, 64);
    CHECK(require_clear(plan_render_clear(depth_mismatch)).region == (RenderArea{0, 0, 64, 32}));
}

void test_layered_and_combined_clears()
{
    // A full-array layered depth clear folds into a standalone layered pass.
    {
        RenderStop stop = make_clear_stop(1);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 4, true);
        stop.registers[0x364] = std::bit_cast<std::uint32_t>(1.0f);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.target.layer == 0 && plan.target.layer_count == 4);
        CHECK(plan.target.texture.layers == 4);
        CHECK(plan.target.texture.type == TextureType::Texture2DArray);
        CHECK(!plan.fold_into_load && plan.fold_layers_into_load);
        CHECK(plan.region == (RenderArea{0, 0, 64, 32}));
    }
    // A nonzero layer field names exactly one layer of the array; fields
    // beyond the programmed array are rejected, on arrayed and non-arrayed
    // targets alike.
    {
        RenderStop stop = make_clear_stop(1 | 2 << 10);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 4, true);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.target.layer == 2 && plan.target.layer_count == 1);
        CHECK(!plan.fold_into_load && !plan.fold_layers_into_load);
    }
    {
        RenderStop stop = make_clear_stop(1 | 3 << 10);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 4, true);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.target.layer == 3 && plan.target.layer_count == 1);
    }
    {
        RenderStop stop = make_clear_stop(1 | 4 << 10);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 4, true);
        require_clear_rejected(plan_render_clear(stop));
    }
    {
        RenderStop stop = make_clear_stop(1 | 1 << 10);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 1, false);
        require_clear_rejected(plan_render_clear(stop));
    }
    // A scissored layered clear stays a real operation on the full range.
    {
        RenderStop stop = make_clear_stop(1, 0x100);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 4, true);
        stop.registers[0x380] = 1;
        stop.registers[0x381] = pack_halves(2, 10);
        stop.registers[0x382] = pack_halves(4, 8);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.region == (RenderArea{2, 4, 8, 4}));
        CHECK(plan.target.layer_count == 4);
        CHECK(!plan.fold_into_load && !plan.fold_layers_into_load);
    }
    // A layered colour clear covers the colour target's programmed array.
    {
        RenderStop stop = make_clear_stop(0xf << 2);
        set_target(stop, 0, 0x100000, 256, 32, 0xd5, 0x1000, 2);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.target.layer_count == 2 && plan.target.texture.layers == 2);
        CHECK(plan.target.texture.type == TextureType::Texture2DArray);
        CHECK(plan.fold_layers_into_load && !plan.fold_into_load);
    }
    // A base layer offsets the cleared range and must stay inside storage.
    {
        RenderStop stop = make_clear_stop(0xf << 2);
        set_target(stop, 0, 0x100000, 256, 32, 0xd5, 0x1000, 2, 1);
        require_clear_rejected(plan_render_clear(stop));
    }
    // A multi-slice 3D colour target stays unsupported.
    {
        RenderStop stop = make_clear_stop(0xf << 2);
        set_target(stop, 0, 0x100000, 64, 32, 0xd5, 0x10000, 2);
        require_clear_rejected(plan_render_clear(stop));
    }
    // A combined clear of arrayed colour and depth targets shares the range
    // and folds into one layered pass load action.
    {
        RenderStop stop = make_clear_stop(3 | 0xf << 2 | 2 << 6);
        set_target(stop, 2, 0x100000, 256, 32, 0xd5, 0x1000, 2);
        set_depth(stop, 0x900000, 0x14, 0, 64, 32, 2, true);
        stop.registers[0x364] = std::bit_cast<std::uint32_t>(0.5f);
        stop.registers[0x368] = 0x5a;
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.color_mask == 0xf);
        CHECK(plan.clear_depth && plan.clear_stencil);
        CHECK(plan.target.layer_count == 2);
        CHECK(plan.depth_stencil.has_value() &&
              plan.depth_stencil->layer_count == 2);
        CHECK(!plan.fold_into_load && !plan.fold_layers_into_load);
    }
    // Differing colour and depth layer counts cannot share one plan.
    {
        RenderStop stop = make_clear_stop(1 | 0xf << 2 | 2 << 6);
        set_target(stop, 2, 0x100000, 256, 32, 0xd5, 0x1000, 2);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 1, false);
        require_clear_rejected(plan_render_clear(stop));
    }
    // A combined clear with a null colour binding clears only depth.
    {
        RenderStop stop = make_clear_stop(1 | 0xf << 2 | 2 << 6);
        set_target(stop, 2, 0, 256, 32, 0xd5, 0x1000, 1);
        set_depth(stop, 0x900000, 0x0a, 0, 64, 32, 1, false);
        stop.registers[0x364] = std::bit_cast<std::uint32_t>(0.5f);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.color_mask == 0 && plan.clear_depth && !plan.clear_stencil);
        CHECK(!plan.depth_stencil.has_value());
        CHECK(plan.target.aspect == TextureAspect::Depth);
        CHECK(plan.fold_into_load);
    }
    // A combined clear with unbound depth clears only colour.
    {
        RenderStop stop = make_clear_stop(1 | 0xf << 2 | 2 << 6);
        set_target(stop, 2, 0x100000, 256, 32, 0xd5, 0x1000, 1);
        const ClearPlan plan = require_clear(plan_render_clear(stop));
        CHECK(plan.color_mask == 0xf && !plan.clear_depth);
        CHECK(!plan.depth_stencil.has_value());
        CHECK(plan.fold_into_load);
    }
}

}  // namespace

int main() {
    test_eight_distinct_outputs();
    test_sparse_and_permuted_outputs();
    test_disabled_slots_remain_holes();
    test_typed_formats_and_layouts();
    test_all_gob_heights_and_pitch_rows();
    test_one_slice_3d_storage();
    test_programmed_layer_stride();
    test_layout_field_diagnostics();
    test_color_target_alpha_semantics();
    test_depth_only_and_depth_stencil_passes();
    test_null_depth_and_attachment_less_passes();
    test_depth_layer_flags_from_register_image();
    test_x8d24_has_no_stencil_aspect();
    test_plan_rejections();
    test_full_and_scissored_color_clears();
    test_pitch_clear_preserves_padding();
    test_clear_scissor_legacy_parity();
    test_depth_stencil_clears();
    test_unbound_depth_clears();
    test_null_color_targets();
    test_clear_rejections();
    test_layered_and_combined_clears();
    return failures == 0 ? 0 : 1;
}
