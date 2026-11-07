// ============================================================================
// Armigo Titles - 01007EF00011E000 Internal Resolution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/01007EF00011E000/resolution.h>
#include <armigo/loader/nso_module.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>

namespace armigo::titles::title_01007ef00011e000 {
using compat::ResolutionSites;

std::optional<float> gpu_budget_scale(GpuBudgetState& state, std::uint64_t gpu_work_ns,
                                      std::int64_t span_ticks, std::uint32_t target_rate) noexcept {
    if (!target_rate || target_rate > 240 || !std::isfinite(state.smoothed_ms) || state.smoothed_ms < 0)
        return {};
    if (gpu_work_ns) {
        state.missing_feedback_calls = 0;
        const double milliseconds = static_cast<double>(gpu_work_ns) / 1e6;
        state.smoothed_ms = state.smoothed_ms
            ? state.smoothed_ms + (milliseconds - state.smoothed_ms) / 8.0 : milliseconds;
    } else if (state.missing_feedback_calls < GpuBudgetState::missing_feedback_limit)
        ++state.missing_feedback_calls;
    const bool stale = state.missing_feedback_calls >= GpuBudgetState::missing_feedback_limit;
    const double budget = 0.66 * 1000.0 / target_rate;
    float load = 162.5f;
    // A stale low estimate must not keep raising resolution (nor a stale
    // high estimate keep lowering it). The neutral load holds the game's
    // current resolution even when its unchanged elapsed span varies.
    if (!stale && state.smoothed_ms > budget)
        load = static_cast<float>(162.5 * state.smoothed_ms / budget);
    else if (!stale && state.smoothed_ms < 0.85 * budget && ++state.qualifying_frames % 12 == 0)
        load = 100.0f;
    load = std::min(load, 999.0f);
    // The guest ignores a load of 1000 or more. Preserve its existing span
    // and change only the multiplier it uses for this controller's load.
    if (span_ticks <= 0 || !state.smoothed_ms) return 2.0f * target_rate;
    return static_cast<float>(static_cast<double>(load) * 192000.0 / static_cast<double>(span_ticks));
}

namespace {

// One MOVZ or MOVK: the 16-bit immediate, its halfword position, and whether
// it keeps the other bits of the register (MOVK).
struct Move {
    std::uint32_t word = 0;
    bool keep = false;
    unsigned shift = 0;
    std::uint16_t immediate = 0;
};

std::optional<Move> decode_move(std::uint32_t word) {
    if (((word >> 23) & 0x3f) != 0b100101) return {};
    const auto opc = (word >> 29) & 3;
    if (opc != 2 && opc != 3) return {};
    const auto hw = (word >> 21) & 3;
    if (!(word >> 31) && hw > 1) return {};
    return Move{word, opc == 3, hw * 16, static_cast<std::uint16_t>((word >> 5) & 0xffff)};
}

std::uint32_t with_immediate(const Move& move, std::uint16_t immediate) {
    return (move.word & ~(0xffffu << 5)) | (std::uint32_t(immediate) << 5);
}

std::uint32_t float_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

// The nearest float whose low 16 bits are zero, the form one MOVZ of the high
// halfword can load.
float high_half_float(float value) {
    const auto bits = float_bits(value);
    return std::bit_cast<float>((bits + 0x8000u) & 0xffff0000u);
}

// Bytes of one RGBA8 block-linear swapchain texture: 64-byte rows and the
// height in 128-row blocks.
std::uint64_t swapchain_bytes(std::uint64_t width, std::uint64_t height) {
    return ((width * 4 + 63) & ~std::uint64_t{63}) * ((height + 127) & ~std::uint64_t{127});
}

std::uint64_t round_up(std::uint64_t value, std::uint64_t granule) {
    return (value + granule - 1) & ~(granule - 1);
}

std::optional<std::uint32_t> environment_number(const char* name, std::string& error) {
    const char* text = std::getenv(name);
    if (!text || !*text) return {};
    std::uint32_t value = 0;
    const std::string_view view(text);
    const auto parsed = std::from_chars(view.data(), view.data() + view.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size()) {
        error = std::string(name) + " must be a decimal number.";
        return {};
    }
    return value;
}

} // namespace

std::variant<ResolutionSettings, std::string> resolution_settings(const ResolutionSites& sites) {
    ResolutionSettings settings{sites.maximum_width, sites.target_rate};
    std::string error;
    if (const auto width = environment_number("ARMIGO_COMPAT_RESOLUTION_WIDTH", error)) settings.maximum_width = *width;
    if (!error.empty()) return error;
    if (const auto rate = environment_number("ARMIGO_COMPAT_RESOLUTION_RATE", error)) settings.target_rate = *rate;
    if (!error.empty()) return error;
    if (settings.maximum_width && (settings.maximum_width < 1280 || settings.maximum_width > 3840))
        return std::string("Compatibility resolution width must be 0 or 1280..3840.");
    if (!settings.target_rate || settings.target_rate > 240)
        return std::string("Compatibility resolution rate must be 1..240.");
    const char* mode = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
    settings.gpu_budget = sites.gpu_budget_rate == 120 && settings.maximum_width &&
        settings.target_rate == 120 && mode && std::string_view(mode) == "120";
    if (const auto enabled = environment_number("ARMIGO_COMPAT_RESOLUTION_GPU_BUSY", error)) {
        if (*enabled > 1) return std::string("Compatibility GPU-work resolution feedback must be 0 or 1.");
        settings.gpu_budget = *enabled != 0;
    }
    if (!error.empty()) return error;
    if (settings.gpu_budget) {
        if (sites.gpu_budget_note.empty() || !settings.maximum_width || settings.target_rate != 120 ||
            !mode || std::string_view(mode) != "120")
            return std::string("GPU-work resolution feedback requires title support, an explicit width, and 120 frame/resolution rates.");
    }
    return settings;
}

std::variant<ResolutionPlan, std::string>
resolution_words(const ResolutionSites& sites, const ResolutionSettings& settings,
                 const loader::NsoModuleImage& image) {
    ResolutionPlan plan;
    if (!settings.maximum_width) {
        if (settings.gpu_budget) return std::string("GPU-work feedback requires an explicit resolution width.");
        return plan;
    }
    const auto& text = image.segment(loader::NsoSegmentKind::text);
    std::string error;
    const auto read = [&](std::uint64_t offset, bool keep, unsigned shift) -> std::optional<Move> {
        if (offset % 4 || offset < text.image_offset || offset - text.image_offset >= text.content_size ||
            text.content_size - (offset - text.image_offset) < 4 || offset > image.bytes.size() - 4) {
            error = "Compatibility resolution site is outside executable text.";
            return {};
        }
        std::uint32_t word = 0;
        std::memcpy(&word, image.bytes.data() + offset, 4);
        const auto move = decode_move(word);
        if (!move || move->keep != keep || move->shift != shift) {
            error = "Compatibility resolution site is not the expected MOVZ/MOVK.";
            return {};
        }
        return move;
    };
    const auto set = [&](std::uint64_t offset, const Move& move, std::uint16_t immediate) {
        plan.words.push_back({offset, move.word, with_immediate(move, immediate), false});
    };

    // Width rounded down to a multiple of 16 below 4096: the width as a float
    // then has a zero low halfword, so one MOVZ or MOVK of the high half loads it.
    const std::uint32_t width = settings.maximum_width & ~15u;
    const std::uint32_t height = width * 9 / 16;
    plan.width = width;
    plan.height = height;
    const auto width_float = float_bits(float(width));
    if (width_float & 0xffff) return std::string("Compatibility resolution width has no single-halfword float form.");

    if (sites.tv_size.empty()) return std::string("Compatibility resolution has no swapchain size site.");
    const auto tv_width = read(sites.tv_size[0][0], false, 0), tv_height = read(sites.tv_size[0][1], true, 32);
    const auto display_width = read(sites.screen_widths[0], false, 16), base_width = read(sites.screen_widths[1], true, 48);
    const auto size_width = read(sites.display_size[0], false, 0), size_height = read(sites.display_size[1], true, 32);
    const auto crop_width = read(sites.window_crop[0], false, 0), crop_height = read(sites.window_crop[1], false, 0);
    const auto floor_low = read(sites.floor[0], false, 0), floor_high = read(sites.floor[1], true, 16);
    const auto pool = read(sites.render_target_pool, false, 16), graphics = read(sites.graphics_heap, false, 16);
    const auto process = read(sites.process_heap, false, 16), root = read(sites.root_heap, false, 16);
    const auto scale = read(sites.load_scale, false, 16);
    if (!error.empty()) return error;
    // The console's docked swapchain size, as the game's own constants hold it.
    if (size_width->immediate != tv_width->immediate || crop_width->immediate != tv_width->immediate ||
        size_height->immediate != tv_height->immediate || crop_height->immediate != tv_height->immediate)
        return std::string("Compatibility resolution sites disagree on the console's output size.");

    // Every copy of the swapchain-size initializer: an unpatched copy that runs
    // first would keep the console's size.
    for (const auto& copy : sites.tv_size) {
        const auto copy_width = read(copy[0], false, 0), copy_height = read(copy[1], true, 32);
        if (!error.empty()) return error;
        if (copy_width->immediate != tv_width->immediate || copy_height->immediate != tv_height->immediate)
            return std::string("Compatibility resolution sites disagree on the console's output size.");
        set(copy[0], *copy_width, std::uint16_t(width));
        set(copy[1], *copy_height, std::uint16_t(height));
    }
    set(sites.screen_widths[0], *display_width, std::uint16_t(width_float >> 16));
    set(sites.screen_widths[1], *base_width, std::uint16_t(width_float >> 16));
    set(sites.display_size[0], *size_width, std::uint16_t(width));
    set(sites.display_size[1], *size_height, std::uint16_t(height));
    set(sites.window_crop[0], *crop_width, std::uint16_t(width));
    set(sites.window_crop[1], *crop_height, std::uint16_t(height));

    // Render targets scale with the pixel count; the GraphicsSystem heap grows
    // by the larger swapchain; the process and root heaps grow by both, so the
    // heaps the game sizes from the free space keep the room they have on the
    // console (heap granularity 2 MiB).
    const std::uint64_t pool_bytes = std::uint64_t(pool->immediate) << 16;
    const std::uint64_t graphics_bytes = std::uint64_t(graphics->immediate) << 16;
    const auto pixels = std::uint64_t(width) * height;
    plan.render_target_pool = std::max(pool_bytes,
        round_up(pool_bytes * pixels / (std::uint64_t(sites.pool_width) * sites.pool_height), 1u << 20));
    const auto console_swapchain = swapchain_bytes(tv_width->immediate, tv_height->immediate);
    const auto swapchain = swapchain_bytes(width, height);
    plan.graphics_heap = round_up(graphics_bytes +
        (swapchain > console_swapchain ? (swapchain - console_swapchain) * sites.swapchain_textures : 0), 1u << 20);
    const auto extra = round_up(plan.graphics_heap - graphics_bytes + plan.render_target_pool - pool_bytes, 2u << 20);
    plan.process_heap = (std::uint64_t(process->immediate) << 16) + extra;
    plan.root_heap = (std::uint64_t(root->immediate) << 16) + extra;
    for (const auto value : {plan.render_target_pool, plan.graphics_heap, plan.process_heap, plan.root_heap})
        if (value >> 32 || value & 0xffff) return std::string("Compatibility resolution heap does not fit one halfword.");
    set(sites.render_target_pool, *pool, std::uint16_t(plan.render_target_pool >> 16));
    set(sites.graphics_heap, *graphics, std::uint16_t(plan.graphics_heap >> 16));
    set(sites.process_heap, *process, std::uint16_t(plan.process_heap >> 16));
    set(sites.root_heap, *root, std::uint16_t(plan.root_heap >> 16));

    // The controller's floor keeps the console's lowest render width, never
    // above the game's own floor.
    const auto game_floor = std::bit_cast<float>(std::uint32_t(floor_low->immediate) |
                                                 std::uint32_t(floor_high->immediate) << 16);
    plan.floor = std::min(float(sites.minimum_render_width) / float(width), game_floor);
    const auto floor_bits = float_bits(plan.floor);
    set(sites.floor[0], *floor_low, std::uint16_t(floor_bits));
    set(sites.floor[1], *floor_high, std::uint16_t(floor_bits >> 16));

    // The controller computes a frame's load as span (s) * scale * 100 and
    // steps down above 175, up below 150. The scale puts 175 at 1.077 of a GPU
    // budget of 0.66 of the target frame, so the size holds while the span
    // stays between about 0.92 and 1.08 of the budget.
    const float budget = 0.66f / float(settings.target_rate);
    plan.load_scale = high_half_float(1.75f / (1.077f * budget));
    if (settings.gpu_budget) {
        const auto reg = scale->word & 31;
        if (sites.gpu_budget_note.empty() || settings.target_rate != 120 ||
            (scale->word & 0xffe00000u) != 0x52a00000u || reg == 18 || reg >= 30)
            return std::string("GPU-work resolution feedback needs a supported 32-bit load-scale constant.");
        compat::CallPatch hook{sites.load_scale, scale->word, sites.gpu_budget_note};
        hook.nvn_driver_only = true;
        hook.gpu_budget = compat::GpuBudgetInstruction{settings.target_rate};
        plan.gpu_budget_hook = std::move(hook);
    } else {
        set(sites.load_scale, *scale, std::uint16_t(float_bits(plan.load_scale) >> 16));
    }
    return plan;
}

} // namespace armigo::titles::title_01007ef00011e000
