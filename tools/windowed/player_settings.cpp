// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_settings.h"

#include <armigo/services/settings.h>
#include <armigo/common/diagnostics.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>

namespace armigo::tools {

std::variant<LanguageManifest, std::string> read_language_manifest(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return "Cannot read the packed language manifest.";
    std::string mask_line;
    std::getline(input, mask_line);
    if (!mask_line.empty() && mask_line.back() == '\r') mask_line.pop_back();
    LanguageManifest manifest;
    if (mask_line.size() != 10 || !mask_line.starts_with("0x")) {
        return "The packed language manifest has an invalid supported-language mask.";
    }
    const auto parsed = std::from_chars(mask_line.data() + 2, mask_line.data() + mask_line.size(),
                                       manifest.supported_language_flag, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != mask_line.data() + mask_line.size()) {
        return "The packed language manifest has an invalid supported-language mask.";
    }
    std::string language;
    while (std::getline(input, language)) {
        if (!language.empty() && language.back() == '\r') language.pop_back();
        if (!language.empty()) manifest.languages.push_back(language);
    }
    return manifest;
}

std::string apply_language_manifest(const LanguageManifest& manifest,
    std::span<const std::string_view> preferred_locales) {
    const auto language = services::set::automatic_language(preferred_locales, manifest.supported_language_flag);
    if (!language) return {};
    if (!services::set::configure_console_language(*language) ||
        !services::set::configure_console_region(services::set::market_region(*language))) return {};
    services::set::configure_supported_languages(manifest.supported_language_flag);
    return *language;
}

std::uint32_t resolution_width(std::string_view choice, bool ios,
    std::uint32_t display_width, std::uint32_t display_height) {
    std::uint32_t width = 0;
    if (choice == "auto") choice = ios ? "1080p" : "native";
    if (choice == "720p") width = 1280;
    else if (choice == "900p") width = 1600;
    else if (choice == "1080p") width = 1920;
    else if (choice == "1440p") width = 2560;
    else if (choice == "2160p") width = 3840;
    else if (choice == "native")
        width = static_cast<std::uint32_t>(std::min<std::uint64_t>(display_width, std::uint64_t(display_height) * 16 / 9));
    else return 0;
    return std::clamp<std::uint32_t>(width, 1280, 3840) & ~15u;
}

MacFrameRateSelection select_mac_frame_rate(const PlayerChoices& choices,
    const platform::DisplayCapabilities& display) {
    MacFrameRateSelection result;
    result.options.push_back(0);
    result.experimental_rates = choices.experimental_frame_rates;
    // Use the active mode, not a faster mode the display could switch to.
    // A mode with no fixed refresh is resolved by the platform to its maximum.
    const bool high_refresh = display.available() && std::isfinite(display.current.refresh_hz) &&
        display.current.refresh_hz >= 120.0;
    for (int rate : {30, 60, 120}) {
        if (rate == 120 && !high_refresh) continue;
        if (std::find(choices.supported_frame_rates.begin(), choices.supported_frame_rates.end(), rate) !=
            choices.supported_frame_rates.end()) result.options.push_back(rate);
    }
    result.frame_rate = choices.frame_rate;
    if (std::find(result.options.begin(), result.options.end(), choices.frame_rate) == result.options.end()) {
        result.frame_rate = result.options.back();
        result.fell_back = true;
    }
    // Existing 30/60/Auto launches retain exactly the runner's 60 Hz grid.
    // 120 is selected only after both the title and active display admit it.
    if (result.frame_rate == 120) result.refresh_hz = 120.0;
    return result;
}

WindowedLaunchConfiguration mac_player_launch_configuration(
    PlayerChoices choices, std::uint32_t width, MacPlayerLaunchObserver observer) {
    return [choices = std::move(choices), width, observer = std::move(observer)](
        RunnerArguments& launch, const platform::DisplayCapabilities& display) {
        const char* override = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
        auto requested = choices;
        if (override) {
            // Leave the environment verbatim for the title selector, including
            // invalid values it must reject and empty/zero values meaning Auto.
            const std::string_view value(override);
            requested.frame_rate = value == "120" ? 120 : value == "60" ? 60 : value == "30" ? 30 : 0;
        }
        auto selection = select_mac_frame_rate(requested, display);
        bool no_rate_fits_clock = false;
        if (!launch.refresh_hz_explicit && requested.frame_rate == 120 && selection.frame_rate == 120)
            launch.refresh_hz = 120.0;
        if (override) {
            selection.environment_override = override;
            selection.frame_rate = requested.frame_rate;
            selection.fell_back = false;
        } else if ((requested.frame_rate == 120 || selection.frame_rate == 120) && launch.refresh_hz < 120.0) {
            // UI preferences may not admit 120 words against a slower explicit
            // clock. Other existing 30/60/Auto behavior is left unchanged.
            selection.frame_rate = 0;
            for (int rate : selection.options)
                if (rate > 0 && rate <= launch.refresh_hz) selection.frame_rate = rate;
            if (selection.frame_rate == 0) {
                no_rate_fits_clock = true;
                selection.frame_rate = static_cast<int>(choices.default_frame_rate);
            }
            selection.fell_back = true;
        }
        selection.refresh_hz = launch.refresh_hz;
        auto effective = choices;
        effective.frame_rate = selection.frame_rate;
        if (effective.frame_rate == 120 && launch.refresh_hz == 120.0)
            effective.display_refresh_rate = 120;
        const auto mode = launch.operation_mode;
        apply_player_choices(effective, width, launch);
        launch.operation_mode = mode;
        if (no_rate_fits_clock)
            diag::warning("Armigo settings: stored %d fps cannot fit the %.3f Hz presentation/VI clock; "
                          "every declared rate exceeds it; using the title default of %d fps.\n",
                          choices.frame_rate, selection.refresh_hz, selection.frame_rate);
        else if (selection.fell_back)
            diag::warning("Armigo settings: stored %d fps is unavailable on this title/display/clock; using %d fps "
                          "with a %.0f Hz presentation/VI clock.\n",
                          choices.frame_rate, selection.frame_rate, selection.refresh_hz);
        diag::info("Armigo launch display: %.3f Hz; presentation/VI clock %.3f Hz is fixed for this run "
                   "(moving between displays does not change it).\n", display.current.refresh_hz, selection.refresh_hz);
        if (observer) observer(effective, selection);
    };
}

void apply_performance_overlay(const PlayerChoices& choices) {
    if (choices.performance_overlay) setenv("MTL_HUD_ENABLED", "1", 1);
}

void apply_player_choices(const PlayerChoices& choices, std::uint32_t width, RunnerArguments& arguments) {
    arguments.operation_mode = choices.handheld ? services::OperationMode::Handheld : services::OperationMode::Console;
    // A title mode selects game timing separately from the guest display cadence.
    const auto refresh_rate = choices.display_refresh_rate ? choices.display_refresh_rate : choices.frame_rate;
    if (refresh_rate && !std::getenv("ARMIGO_VI_REFRESH_HZ"))
        setenv("ARMIGO_VI_REFRESH_HZ", std::to_string(refresh_rate).c_str(), 1);
    if (width && !std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH"))
        setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", std::to_string(width).c_str(), 1);
    if (choices.frame_rate && !std::getenv("ARMIGO_COMPAT_RESOLUTION_RATE"))
        setenv("ARMIGO_COMPAT_RESOLUTION_RATE", std::to_string(choices.frame_rate).c_str(), 1);
    if (choices.title_frame_rate && !std::getenv("ARMIGO_COMPAT_FRAME_RATE"))
        setenv("ARMIGO_COMPAT_FRAME_RATE", std::to_string(choices.frame_rate).c_str(), 1);
    arguments.content_width = width;
    arguments.display_sync = choices.vsync;
}

} // namespace armigo::tools
