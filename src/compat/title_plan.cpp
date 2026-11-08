// ============================================================================
// Armigo Compatibility - Title Patch Planning
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/title_plan.h>
#include <armigo/common/diagnostics.h>
#include <armigo/loader/nso_module.h>
#include <armigo/titles/resolution.h>

#include <algorithm>
#include <optional>

namespace armigo::compat {
namespace {

using Failure = TitlePlanFailure;

bool overlaps(const std::vector<WordPatch>& words, const WordPatch& word) {
    return std::any_of(words.begin(), words.end(), [&](const auto& other) { return other.offset == word.offset; });
}

// The internal-resolution module: the words for the launch's width, or for
// ahead-of-time code the gate sites of a fixed-width plan carrying them.
std::optional<Failure> plan_resolution(const ResolutionSites& sites, const loader::NsoModuleImage& image,
                                       const TitlePlanOptions& options, TitlePlan& plan) {
    namespace resolution = titles::resolution;
    const auto settings = resolution::resolution_settings(sites);
    if (const auto* failure = std::get_if<std::string>(&settings))
        return Failure{Failure::Kind::invalid_input, "compatibility-resolution", *failure};
    const auto planned = resolution::resolution_words(sites, std::get<resolution::ResolutionSettings>(settings), image);
    if (const auto* failure = std::get_if<std::string>(&planned))
        return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution", *failure};
    const auto& result = std::get<resolution::ResolutionPlan>(planned);
    if (result.gpu_budget_hook) plan.module_calls.push_back(*result.gpu_budget_hook);
    if (!result.words.empty())
        armigo::diag::write(armigo::diag::Level::info,  "compat_resolution width=%u height=%u render_target_pool=%llu "
                     "graphics_heap=%llu process_heap=%llu root_heap=%llu floor=%.4f load_scale=%.1f\n",
                     result.width, result.height,
                     static_cast<unsigned long long>(result.render_target_pool),
                     static_cast<unsigned long long>(result.graphics_heap),
                     static_cast<unsigned long long>(result.process_heap),
                     static_cast<unsigned long long>(result.root_heap),
                     result.floor, result.load_scale);
    // The launcher refuses feedback hooks in code prepared ahead of time.
    if (result.gpu_budget_hook && options.ahead_of_time) return std::nullopt;
    std::vector<WordPatch> planned_words = result.words;
    if (options.ahead_of_time) {
        // A plan at a fixed width names the sites, and this launch's plan (or
        // the game's own words) gives what they apply.
        auto fixed = std::get<resolution::ResolutionSettings>(settings);
        fixed.maximum_width = 1920;
        const auto fixed_sites = resolution::resolution_words(sites, fixed, image);
        if (const auto* failure = std::get_if<std::string>(&fixed_sites))
            return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution", *failure};
        for (auto site : std::get<resolution::ResolutionPlan>(fixed_sites).words) {
            site.replacement_instruction = site.expected_instruction;
            for (const auto& chosen : result.words)
                if (chosen.offset == site.offset) site.replacement_instruction = chosen.replacement_instruction;
            plan.launch_words.push_back(site);
        }
        // The planned word may change only the immediate the gate applies.
        if (std::any_of(plan.launch_words.begin(), plan.launch_words.end(), [](const auto& site) {
                return ((site.replacement_instruction ^ site.expected_instruction) & ~(0xffffu << 5)) != 0;
            }))
            return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution",
                           "A launch-time word changes more than its immediate."};
        if (!std::all_of(result.words.begin(), result.words.end(),
                         [&](const auto& chosen) { return overlaps(plan.launch_words, chosen); }))
            return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution",
                           "Compatibility resolution sites differ between widths."};
        planned_words.clear();
    }
    for (const auto& word : planned_words) {
        if (overlaps(plan.words, word))
            return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution",
                           "Compatibility resolution site overlaps an instruction word."};
        plan.words.push_back(word);
    }
    for (const auto& word : plan.launch_words)
        if (overlaps(plan.words, word))
            return Failure{Failure::Kind::artifact_mismatch, "compatibility-resolution",
                           "Compatibility resolution site overlaps an instruction word."};
    return std::nullopt;
}

} // namespace

std::variant<TitlePlan, TitlePlanFailure> plan_title_words(
    const PatchDescription& patch, const loader::NsoModuleImage& image, const TitlePlanOptions& options) {
    TitlePlan plan;
    auto selected = select_patch_words(patch, options.requested_rate);
    if (const auto* failure = std::get_if<std::string>(&selected))
        return Failure{Failure::Kind::invalid_input, "compatibility-frame-rate", *failure};
    plan.words = std::get<std::vector<WordPatch>>(std::move(selected));
    if (patch.resolution)
        if (auto failure = plan_resolution(*patch.resolution, image, options, plan)) return *std::move(failure);
    return plan;
}

} // namespace armigo::compat
