// ============================================================================
// Armigo Titles - Internal Resolution Module
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Host-side measure with no console counterpart: the console has two fixed
// output modes, so the game keeps their sizes as instruction immediates. When
// a maximum width is set, these immediates are rewritten at load time so the
// swapchain, the docked display and render-base sizes, the heaps and render-
// target pool that hold them, and the dynamic-resolution floor and load scale
// follow that size. Every site is one MOVZ or MOVK whose 16-bit immediate is
// replaced; the original values are decoded from the instructions themselves.
// A title profile's "resolution" object supplies the sites, so any title with
// these sites selects the module by data. Without a maximum width nothing is
// changed.

#pragma once

#include <armigo/compat/patches.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::loader { struct NsoModuleImage; }

namespace armigo::titles::resolution {

struct ResolutionSettings {
    std::uint32_t maximum_width = 0;   // 0: the game's own sizes
    std::uint32_t target_rate = 60;    // Frame rate the load scale budgets for
    bool gpu_budget = false;
};

struct GpuBudgetState {
    static constexpr std::uint32_t missing_feedback_limit = 16;
    double smoothed_ms = 0;
    std::uint32_t missing_feedback_calls = 0;
    std::uint32_t qualifying_frames = 0;
};

// Host resolution feedback, separate from the game's GPU timestamp counters.
// gpu_work_ns is the sum of execution intervals of a completed frame; zero
// means unavailable. span_ticks is the controller's unchanged 19.2 MHz span.
// The returned float replaces only its load-scale constant. The caller owns
// and serializes the smoothing state. After bounded missing feedback, return
// the neutral load for the current span so the game holds its resolution.
std::optional<float> gpu_budget_scale(GpuBudgetState& state, std::uint64_t gpu_work_ns,
                                      std::int64_t span_ticks, std::uint32_t target_rate) noexcept;

// Width/rate settings override the description's defaults. GPU-work feedback
// requires a declared default or ARMIGO_COMPAT_RESOLUTION_GPU_BUSY=1, a title
// capability, an explicit width, and frame/resolution rates both set to 120.
// Returns an error for malformed or unsupported settings.
std::variant<ResolutionSettings, std::string>
resolution_settings(const compat::ResolutionSites& sites);

struct ResolutionPlan {
    std::uint32_t width = 0, height = 0;
    std::uint64_t render_target_pool = 0, graphics_heap = 0, process_heap = 0, root_heap = 0;
    float floor = 0, load_scale = 0;
    std::vector<compat::WordPatch> words;
    std::optional<compat::CallPatch> gpu_budget_hook;
};

// The words that apply the settings to the image, each with its original
// instruction as the expected preimage.
std::variant<ResolutionPlan, std::string>
resolution_words(const compat::ResolutionSites& sites, const ResolutionSettings& settings,
                 const loader::NsoModuleImage& image);

} // namespace armigo::titles::resolution
