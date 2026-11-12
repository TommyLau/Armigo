// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "runner_shell.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::tools {

struct LanguageManifest {
    std::uint32_t supported_language_flag = 0;
    std::vector<std::string> languages;
};

std::variant<LanguageManifest, std::string> read_language_manifest(
    const std::filesystem::path& path);

// The languages a packed title admits, and a stored choice validated against
// them. "auto" follows the app's and host's preferred languages. A stale or
// unsupported stored choice falls back to Auto without becoming a launch error.
struct GameLanguageChoice {
    std::vector<std::string> supported_languages;
    std::string preference = "auto";
    bool fell_back = false;
};
[[nodiscard]] GameLanguageChoice game_language_choice(const LanguageManifest& manifest,
    std::optional<std::string> stored);

// Applies the title's supported-language mask and either an explicit supported
// game language or Auto. Auto maps each preferred host locale in order (the
// app's own language first) to a system language, and the firmware's priority
// order maps that to one of the title's languages. The region is that
// language's market, never China. Returns the system language set, or empty
// when the title admits none or an explicit choice is unsupported.
std::string apply_language_manifest(const LanguageManifest& manifest,
    std::span<const std::string_view> preferred_locales, std::string_view preference = "auto");

// The defaults key that keeps the game language: "game_language" in a packed
// app, which plays one title, and "game_language.<TitleID>" in the repository's
// app, which plays any. The new key is independent of the removed legacy
// "language" setting.
[[nodiscard]] std::string language_preference_key(std::optional<std::uint64_t> title_id);
std::optional<std::string> read_language_preference(const std::string& key);
void write_language_preference(const std::string& key, std::string_view language);

// The display settings both players keep under the same keys, on iOS in its
// Settings page and on macOS in the Settings window. They take effect when the
// game starts. macOS keeps its game-language choice separately per title.
struct PlayerChoices {
    bool handheld = false;             // console_mode: "docked" or "handheld"
    std::string resolution = "auto";   // auto, 720p, 900p, 1080p, 1440p, 2160p or native
    int frame_rate = 60;               // 0 (Auto), 30, 60 or macOS-only 120
    bool title_frame_rate = false;
    std::uint32_t default_frame_rate = 60;
    std::vector<std::uint32_t> supported_frame_rates = {30, 60};
    std::vector<std::uint32_t> experimental_frame_rates;
    std::uint32_t display_refresh_rate = 0; // zero follows the frame-rate choice
    bool vsync = true;                 // macOS; iOS always synchronises
    bool performance_overlay = false;  // Apple's Metal performance HUD
};

// Reads the display choices from the app's defaults, first removing the earlier
// layout's language, region and frame-rate-overlay keys. Its overlay toggle
// becomes the performance overlay only when that setting does not exist yet;
// current frame-rate and per-title game-language preferences stay untouched.
PlayerChoices read_player_choices();

// The frame-rate policy a title's settings declare for the running build: its
// default, guest display cadence and the rates it admits.
struct TitleFrameRatePolicy {
    std::uint64_t title_id = 0;
    std::uint32_t default_rate = 30;
    std::uint32_t display_refresh_rate = 60;
    std::vector<std::uint32_t> supported_rates;
    std::vector<std::uint32_t> experimental_rates;
};

// The policy of the title an ExeFS run starts, read the way the runtime finds
// it: the title id from main.npdm, the main module's build id from its NSO
// header, and the matching description in the title settings. Empty for other
// inputs, a title without settings or a build without a frame-rate policy.
[[nodiscard]] std::optional<TitleFrameRatePolicy> find_title_frame_rate_policy(
    const RunnerArguments& arguments);

// The frame-rate choices of a run started from the command line with a title
// policy. `stored` is the rate kept for the title in the app's defaults: 0
// (Auto), 30, 60 or 120. Without one the run keeps the title default, as
// before the preference existed.
[[nodiscard]] PlayerChoices title_frame_rate_choices(const TitleFrameRatePolicy& policy,
    std::optional<int> stored);

// The defaults key that keeps the frame rate: "frame_rate" in a packed app,
// which plays one title, and "frame_rate.<TitleID>" in the repository's app,
// which plays any.
[[nodiscard]] std::string frame_rate_preference_key(std::optional<std::uint64_t> title_id);
std::optional<int> read_frame_rate_preference(const std::string& key);
void write_frame_rate_preference(const std::string& key, int rate);

// macOS launch policy, resolved against the window's active display mode. An
// unknown display never admits 120. Auto remains zero for the title selector;
// unsupported stored choices fall back to the highest declared rate <= clock.
struct MacFrameRateSelection {
    std::vector<int> options;
    std::vector<std::uint32_t> experimental_rates;
    // The title default Auto selects; zero for a title without a policy.
    std::uint32_t default_rate = 0;
    int frame_rate = 0;
    double refresh_hz = 60.0;
    bool fell_back = false;
    std::optional<std::string> environment_override;
};
[[nodiscard]] MacFrameRateSelection select_mac_frame_rate(const PlayerChoices& choices,
    const platform::DisplayCapabilities& display);

// Production macOS launch callback. An explicit title-rate environment override
// wins over stored preferences; an explicit --refresh-hz wins over clock policy.
// The observer handles UI after the settings and their diagnostics are applied.
using MacPlayerLaunchObserver =
    std::function<void(const PlayerChoices&, const MacFrameRateSelection&)>;
[[nodiscard]] WindowedLaunchConfiguration mac_player_launch_configuration(
    PlayerChoices choices, std::uint32_t width, MacPlayerLaunchObserver observer = {});

// The same frame-rate policy for a run started from the command line with a
// title policy (title_frame_rate_choices): only the frame rate, its clock and
// guest display cadence follow the choice; window, resolution, operation mode
// and synchronization stay as the command line sets them. An Auto choice
// changes nothing unless the environment selects a title mode.
[[nodiscard]] WindowedLaunchConfiguration mac_title_frame_rate_configuration(
    PlayerChoices choices, MacPlayerLaunchObserver observer = {});

// The game's output width for a Resolution choice: a fixed size is its 16:9
// width, Native the widest 16:9 image the display's pixels hold, and Auto is
// Native on macOS and 1080p on iOS. Widths stay within 1280..3840 in multiples
// of 16. Returns 0 for an unknown choice.
std::uint32_t resolution_width(std::string_view choice, bool ios,
    std::uint32_t display_width, std::uint32_t display_height);

// Applies On or Off unless an explicit launch environment value wins. Metal
// reads MTL_HUD_ENABLED when the process first uses it, so the players call
// this before anything creates a Metal device. Restarts use the original launch
// environment, excluding any HUD value this function adds.
void apply_performance_overlay(const PlayerChoices& choices);

// Applies the choices to the launch of a packed game: the operation mode, the
// output width (through the title's resolution setting, for a title that has
// one) and a drawable no wider than it, V-Sync, and the frame-rate choices.
// Settings the environment already makes
// win.
void apply_player_choices(const PlayerChoices& choices, std::uint32_t width, RunnerArguments& arguments);
// Applies the frame-rate part alone: the frame rate the title's resolution
// controller budgets for, the title mode, and guest display cadence. Settings
// the environment already makes win.
void apply_frame_rate_choices(const PlayerChoices& choices);

// What a restart from the app changes in the command line and environment the
// game started with: the frame-rate overrides (--refresh-hz,
// ARMIGO_COMPAT_FRAME_RATE and ARMIGO_VI_REFRESH_HZ) give way to the choice
// just made, and --fullscreen follows whether the game's window is in full
// screen now. A packed app opened without arguments gains "run" before any
// argument it adds.
struct RestartCommand {
    std::vector<std::string> arguments;
    std::vector<std::string> environment;
};
[[nodiscard]] RestartCommand restart_command(std::span<const std::string> arguments,
    std::span<const std::string> environment, bool fullscreen);

// The host locales Auto follows, in order: the app's own language, then the
// user's preferred languages.
std::vector<std::string> preferred_host_locales();

} // namespace armigo::tools
