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

// Applies the title's supported-language mask and the system language the
// host's languages choose: each preferred host locale in order (the app's own
// language first) maps to a system language, and the firmware's priority
// order maps that to one of the title's languages. The region is that
// language's market, never China. Returns the system language set, or empty
// when the title admits none.
std::string apply_language_manifest(const LanguageManifest& manifest,
    std::span<const std::string_view> preferred_locales);

// The settings both players keep under the same keys, on iOS in its Settings
// page and on macOS in the Settings window. They take effect when the game
// starts. The game's language has no setting: it follows the app's language
// in the system's settings.
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

// Reads the choices from the app's defaults, first moving the settings of the
// earlier layout: its language and region choices and its frame rate (Auto
// by default then) are removed, and its frame-rate overlay toggle becomes the
// performance overlay.
PlayerChoices read_player_choices();

// macOS launch policy, resolved against the window's active display mode. An
// unknown display never admits 120. Auto remains zero for the title selector;
// unsupported stored choices fall back to the highest declared rate <= clock.
struct MacFrameRateSelection {
    std::vector<int> options;
    std::vector<std::uint32_t> experimental_rates;
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

// The game's output width for a Resolution choice: a fixed size is its 16:9
// width, Native the widest 16:9 image the display's pixels hold, and Auto is
// Native on macOS and 1080p on iOS. Widths stay within 1280..3840 in multiples
// of 16. Returns 0 for an unknown choice.
std::uint32_t resolution_width(std::string_view choice, bool ios,
    std::uint32_t display_width, std::uint32_t display_height);

// Turns on Apple's Metal performance HUD when the choices ask for it. Metal
// reads MTL_HUD_ENABLED when the process first uses it, so the players call
// this before anything creates a Metal device.
void apply_performance_overlay(const PlayerChoices& choices);

// Applies the choices to the launch of a packed game: the operation mode, the
// output width (through the title's resolution setting, for a title that has
// one) and a drawable no wider than it, V-Sync, and the frame rate the title's
// resolution controller budgets for, the title mode, and guest display cadence.
// Settings the environment already makes
// win.
void apply_player_choices(const PlayerChoices& choices, std::uint32_t width, RunnerArguments& arguments);

// The host locales Auto follows, in order: the app's own language, then the
// user's preferred languages.
std::vector<std::string> preferred_host_locales();

} // namespace armigo::tools
