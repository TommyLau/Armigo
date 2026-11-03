// ============================================================================
// Armigo Titles - 01007EF00011E000 Internal Resolution
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
// The private compatibility description supplies the sites. Without a maximum
// width nothing is changed.

#pragma once

#include <armigo/compat/patches.h>

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace armigo::loader { struct NsoModuleImage; }

namespace armigo::titles::title_01007ef00011e000 {

struct ResolutionSettings {
    std::uint32_t maximum_width = 0;   // 0: the game's own sizes
    std::uint32_t target_rate = 60;    // Frame rate the load scale budgets for
};

// Settings from ARMIGO_COMPAT_RESOLUTION_WIDTH and ARMIGO_COMPAT_RESOLUTION_RATE
// over the description's defaults. Returns an error for malformed values.
std::variant<ResolutionSettings, std::string>
resolution_settings(const compat::ResolutionSites& sites);

struct ResolutionPlan {
    std::uint32_t width = 0, height = 0;
    std::uint64_t render_target_pool = 0, graphics_heap = 0, process_heap = 0, root_heap = 0;
    float floor = 0, load_scale = 0;
    std::vector<compat::WordPatch> words;
};

// The words that apply the settings to the image, each with its original
// instruction as the expected preimage.
std::variant<ResolutionPlan, std::string>
resolution_words(const compat::ResolutionSites& sites, const ResolutionSettings& settings,
                 const loader::NsoModuleImage& image);

} // namespace armigo::titles::title_01007ef00011e000
