// ============================================================================
// Armigo Compatibility - Runtime Game Patches
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/titles/frame_rate.h>
#include <armigo/loader/nso_module.h>
#include <armigo/execution/gate.h>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::compat {
// Internal mode identity; profiles and the environment spell it "variable".
// It is never a numeric rate, default, display cadence or player preference.
inline constexpr std::uint32_t variable_frame_rate = UINT32_MAX;
inline constexpr char variable_frame_rate_warning[] =
    "Armigo: variable frame rate is experimental; movie timing and later menu repeats remain unresolved.";

struct TitleIdentity {
    std::uint64_t title_id = 0;
    std::array<std::byte, 32> main_build_id{};
    bool operator==(const TitleIdentity&) const = default;
};
struct CallPatch {
    std::uint64_t offset = 0;
    std::uint32_t expected_instruction = 0;
    // Plain English from the profile: what this call changes and why.
    // Required there, 1 to 200 characters. Empty for a call built in code.
    std::string note;
    std::uint32_t frame_rate = 0;
    bool nvn_driver_only = false;
};
// One instruction of the main module replaced at load time. The expected word
// must be present, unchanged by preparation, or the title is not patched.
// nvn_driver_only words apply only when Armigo's NVN driver replaces the
// game's own.
struct WordPatch {
    std::uint64_t offset = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    bool nvn_driver_only = false;
    // Plain English from the profile: what this word changes and why.
    // Required there, 1 to 200 characters. Empty for a word built in code.
    std::string note;
    // Zero applies at every rate; otherwise this word belongs to that title mode.
    // Distinct modes may replace one offset, but must share its original preimage.
    std::uint32_t frame_rate = 0;
};
struct FrameRateSettings {
    std::uint32_t default_rate = 30;
    // The guest display cadence can differ from the game's requested frame rate.
    std::uint32_t display_refresh_rate = 60;
    // Explicitly admitted modes, including the default.
    std::vector<std::uint32_t> supported_rates;
};
// Instruction sites of the internal-resolution patch
// (titles/01007EF00011E000/resolution.h). Each
// site is one MOVZ or MOVK; the pairs are the two halves of one value.
struct ResolutionSites {
    // Swapchain width, height << 32, once per place that initializes the size:
    // a compiler may inline the initializer of one guarded value into callers,
    // and whichever copy runs first sets it.
    std::vector<std::array<std::uint64_t, 2>> tv_size;
    std::array<std::uint64_t, 2> screen_widths{};  // display width, render-base width (f32 high halves)
    std::array<std::uint64_t, 2> display_size{};   // width, height << 32
    std::array<std::uint64_t, 2> window_crop{};    // width, height
    std::array<std::uint64_t, 2> floor{};          // dynamic-resolution floor (f32 low, high half)
    std::uint64_t render_target_pool = 0, graphics_heap = 0, process_heap = 0, root_heap = 0;
    std::uint64_t load_scale = 0;                  // dynamic-resolution load scale (f32 high half)
    std::uint32_t swapchain_textures = 0, pool_width = 0, pool_height = 0, minimum_render_width = 0;
    std::uint32_t maximum_width = 0, target_rate = 60;   // defaults; the environment overrides
};
// Where the packer finds this title's shader stages. Absent profile settings
// keep the historical choice: BNSH containers and a minus-one-to-one vertex
// depth. A gfd-cache description reads .gsc members of CPK archives plus the
// packages embedded in the main module.
enum class ShaderContainerSet { Bnsh, GfdCache };
enum class ShaderVertexDepth { NegativeOneToOne, ZeroToOne };
struct ShaderCacheFallback {
    std::string cache;
    std::vector<std::string> alternatives;
};
struct ShaderInputSettings {
    ShaderContainerSet containers = ShaderContainerSet::Bnsh;
    ShaderVertexDepth vertex_depth = ShaderVertexDepth::NegativeOneToOne;
    std::vector<std::string> caches;
    std::vector<ShaderCacheFallback> fallbacks;
    std::vector<std::uint64_t> ignored_needle_offsets;
};
struct PatchDescription {
    TitleIdentity identity;
    std::string module;
    std::uint64_t original_callee = 0;
    std::vector<CallPatch> calls;
    titles::FrameRateLayout layout;
    bool nvn_driver = false;
    // The title's control data requires a user account at startup, so the
    // launcher preselects the local user and hands it to the application as
    // its preselected-user launch parameter. Extracted input carries no
    // control data; the compatibility description supplies the setting.
    bool startup_user = false;
    // The title's pipeline list (the profile's "pipelines" file, resolved to
    // its path in the title folder); empty for none. Only the NVN driver
    // uses it, as its pipeline warm-up seed.
    std::string pso_manifest;
    // The title's frame-rate patch also asks for presentations paced by the
    // display link instead of the swap interval the guest requests: the patch
    // retimes the game to the display's refresh, so a present that waited for
    // the requested interval would cap it a whole refresh below the target.
    bool present_link = false;
    std::vector<WordPatch> words;
    std::optional<FrameRateSettings> frame_rate;
    std::optional<ResolutionSites> resolution;
    ShaderInputSettings shader_inputs;
    // Optional plain English on the description itself, 1 to 200 characters
    // when the profile includes it.
    std::string note;
};
struct InstalledPatch {
    PatchDescription description;
    std::uint64_t module_base = 0;
};
execution::GateStatus handle_patch(const InstalledPatch& patch, execution::GateCall& call,
                                    const horizon::GuestBuffers& memory) noexcept;
// Present policy of the patch installed for this process (false unless a
// title's compatibility description enables it). Read by the display service.
void set_present_link(bool enabled) noexcept;
bool present_link() noexcept;
using PatchDescriptions = std::vector<PatchDescription>;
using PatchReadResult = std::variant<PatchDescriptions, std::string>;
// An explicit file is fail-closed; absent configuration is handled by the runner.
PatchReadResult read_patch_descriptions(const std::filesystem::path& path);
// Empty or "0" selects the title's default. A title without a rate policy
// retains all of its ordinary words and ignores this title-only setting.
// A nondefault mode needs an effective rate-conditioned patch after driver filtering.
// When requested, admitted_rate is zero on failure, without a policy, or for
// variable mode (which has no fixed pacing target).
std::variant<std::vector<WordPatch>, std::string> select_patch_words(
    const PatchDescription& patch, std::string_view requested_rate = {}, bool nvn_driver_enabled = true,
    std::uint32_t* admitted_rate = nullptr);
// Uses the same admitted mode and actual driver choice as word selection.
std::variant<std::vector<CallPatch>, std::string> select_patch_calls(
    const PatchDescription& patch, std::string_view requested_rate = {}, bool nvn_driver_enabled = true);
const PatchDescription* match_patch(std::span<const PatchDescription> patches,
                                    const TitleIdentity& identity) noexcept;
// Validates executable placement, BL destination, preimages, data placement and
// instruction-word preimages.
// No byte is changed on success or failure.
std::string validate_patch(const PatchDescription& patch, const loader::NsoModuleImage& image);
} // namespace armigo::compat
