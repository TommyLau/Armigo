// ============================================================================
// Armigo Compatibility - Title Patch Planning
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/compat/patches.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::loader { struct NsoModuleImage; }

namespace armigo::compat {

// What a matched title description changes in its main module for one launch,
// before the NVN driver is chosen: the profile's words for the requested
// mode, and what each title module the profile selects contributes. Every
// launcher installs this plan; only stub emission, preparation coverage and
// backend capabilities stay with the launcher.
struct TitlePlanOptions {
    // The launch's title mode, as ARMIGO_COMPAT_FRAME_RATE spells it.
    std::string_view requested_rate;
    // Code compiled ahead of time keeps every launch-time site as a gate site,
    // so one preparation serves every launch setting.
    bool ahead_of_time = false;
};

struct TitlePlan {
    // Applied when the module loads.
    std::vector<WordPatch> words;
    // Applied by gate sites of code prepared ahead of time; empty otherwise.
    std::vector<WordPatch> launch_words;
    // Instruction hooks the title's modules add to its description's calls.
    std::vector<CallPatch> module_calls;
};

struct TitlePlanFailure {
    enum class Kind { invalid_input, artifact_mismatch, unsupported };
    Kind kind = Kind::invalid_input;
    // "compatibility-frame-rate" or "compatibility-resolution".
    std::string stage;
    std::string detail;
};

[[nodiscard]] std::variant<TitlePlan, TitlePlanFailure> plan_title_words(
    const PatchDescription& patch, const loader::NsoModuleImage& image, const TitlePlanOptions& options);

// Once the launch knows whether Armigo's NVN driver replaces the game's own:
// the title mode the words of `rate_patch` admit (zero without one or without
// a fixed pacing target), since words meant for the driver apply only with
// it; and the installed description's calls (module calls included), when
// there is one, narrowed to that mode and driver choice. A GPU-work feedback
// hook needs the driver.
[[nodiscard]] std::variant<std::uint32_t, TitlePlanFailure> plan_title_mode(
    const PatchDescription* rate_patch, PatchDescription* installed, std::string_view requested_rate,
    bool nvn_driver, bool gpu_feedback);

} // namespace armigo::compat
