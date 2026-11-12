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

#include <armigo/compat/title_settings.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/services/settings.h>
#include <armigo/common/diagnostics.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

GameLanguageChoice game_language_choice(const LanguageManifest& manifest,
    std::optional<std::string> stored) {
    GameLanguageChoice choice;
    for (const auto& language : manifest.languages) {
        const auto selected = services::set::title_language(language, manifest.supported_language_flag);
        if (!selected || *selected != language ||
            std::find(choice.supported_languages.begin(), choice.supported_languages.end(), language) !=
                choice.supported_languages.end()) continue;
        choice.supported_languages.push_back(language);
    }
    if (!stored || *stored == "auto") return choice;
    if (std::find(choice.supported_languages.begin(), choice.supported_languages.end(), *stored) !=
        choice.supported_languages.end()) {
        choice.preference = *stored;
    } else {
        choice.fell_back = true;
    }
    return choice;
}

std::string apply_language_manifest(const LanguageManifest& manifest,
    std::span<const std::string_view> preferred_locales, std::string_view preference) {
    const auto choice = game_language_choice(manifest, std::string(preference));
    if (choice.fell_back) return {};
    const auto language = choice.preference == "auto"
        ? services::set::automatic_language(preferred_locales, manifest.supported_language_flag)
        : std::optional<std::string>(choice.preference);
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
    if (choices.title_frame_rate) result.default_rate = choices.default_frame_rate;
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

namespace {

// At most `limit` leading bytes of a file; empty when it cannot be read.
std::vector<std::byte> read_prefix(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream input(path, std::ios::binary);
    std::vector<std::byte> bytes(limit);
    if (!input) return {};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(limit));
    bytes.resize(static_cast<std::size_t>(input.gcount()));
    return bytes;
}

// Resolves the requested rate against the title, the launch display and an
// explicit clock, and sets the clock a 120 fps title mode needs. Returns the
// choices to apply.
PlayerChoices resolve_mac_frame_rate(const PlayerChoices& choices, RunnerArguments& launch,
    const platform::DisplayCapabilities& display, MacFrameRateSelection& selection) {
    const char* override = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
    auto requested = choices;
    if (override) {
        // Leave the environment verbatim for the title selector, including
        // invalid values it must reject and empty/zero values meaning Auto.
        const std::string_view value(override);
        requested.frame_rate = value == "120" ? 120 : value == "60" ? 60 : value == "30" ? 30 : 0;
    }
    selection = select_mac_frame_rate(requested, display);
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
    return effective;
}

} // namespace

WindowedLaunchConfiguration mac_player_launch_configuration(
    PlayerChoices choices, std::uint32_t width, MacPlayerLaunchObserver observer) {
    return [choices = std::move(choices), width, observer = std::move(observer)](
        RunnerArguments& launch, const platform::DisplayCapabilities& display) {
        MacFrameRateSelection selection;
        const auto effective = resolve_mac_frame_rate(choices, launch, display, selection);
        const auto mode = launch.operation_mode;
        apply_player_choices(effective, width, launch);
        launch.operation_mode = mode;
        if (observer) observer(effective, selection);
    };
}

WindowedLaunchConfiguration mac_title_frame_rate_configuration(
    PlayerChoices choices, MacPlayerLaunchObserver observer) {
    return [choices = std::move(choices), observer = std::move(observer)](
        RunnerArguments& launch, const platform::DisplayCapabilities& display) {
        MacFrameRateSelection selection;
        const auto effective = resolve_mac_frame_rate(choices, launch, display, selection);
        if (selection.environment_override) {
            // The environment chose the title mode; a 120 fps mode also needs
            // the guest cadence of the 120 Hz clock it was given above.
            if (effective.display_refresh_rate == 120 && !std::getenv("ARMIGO_VI_REFRESH_HZ"))
                setenv("ARMIGO_VI_REFRESH_HZ", "120", 1);
        } else if (effective.frame_rate) {
            // Auto keeps the run as the command line starts it: the runtime
            // selects the title default and the clocks keep their defaults.
            apply_frame_rate_choices(effective);
        }
        if (observer) observer(effective, selection);
    };
}

std::optional<TitleFrameRatePolicy> find_title_frame_rate_policy(const RunnerArguments& arguments) {
    if (arguments.exefs_directory.empty() || arguments.container_file || arguments.nup_file) return std::nullopt;
    // The NSO header holds the module's build id at 0x40; the image itself
    // need not be read.
    constexpr std::size_t build_id_offset = 0x40;
    const auto main = read_prefix(arguments.exefs_directory / "main", build_id_offset + 32);
    const auto npdm = read_prefix(arguments.exefs_directory / "main.npdm", 1u << 20);
    if (main.size() != build_id_offset + 32 || std::memcmp(main.data(), "NSO0", 4) != 0 || npdm.empty())
        return std::nullopt;
    const auto profile = runtime::derive_launch_profile(npdm,
        runtime::NpdmTrustPolicy::structurally_validated_extracted_input);
    const auto* admitted = std::get_if<runtime::LaunchProfile>(&profile);
    if (!admitted) return std::nullopt;
    compat::TitleIdentity identity{admitted->title_id, {}};
    std::copy_n(main.begin() + build_id_offset, identity.main_build_id.size(), identity.main_build_id.begin());
    const auto found = compat::find_title_settings(admitted->title_id);
    const auto* settings = std::get_if<std::optional<compat::TitleSettings>>(&found);
    if (!settings || !*settings) return std::nullopt;
    const auto* patch = compat::match_patch((*settings)->patches, identity);
    if (!patch || !patch->frame_rate) return std::nullopt;
    return TitleFrameRatePolicy{admitted->title_id, patch->frame_rate->default_rate,
        patch->frame_rate->display_refresh_rate, patch->frame_rate->supported_rates,
        patch->frame_rate->experimental_rates};
}

PlayerChoices title_frame_rate_choices(const TitleFrameRatePolicy& policy, std::optional<int> stored) {
    PlayerChoices choices;
    choices.title_frame_rate = true;
    choices.default_frame_rate = policy.default_rate;
    choices.display_refresh_rate = policy.display_refresh_rate;
    choices.supported_frame_rates = policy.supported_rates;
    choices.experimental_frame_rates = policy.experimental_rates;
    choices.frame_rate = stored && (*stored == 30 || *stored == 60 || *stored == 120) ? *stored : 0;
    return choices;
}

std::string language_preference_key(std::optional<std::uint64_t> title_id) {
    return title_id ? "game_language." + compat::title_id_text(*title_id) : std::string("game_language");
}

std::string frame_rate_preference_key(std::optional<std::uint64_t> title_id) {
    return title_id ? "frame_rate." + compat::title_id_text(*title_id) : std::string("frame_rate");
}

RestartCommand restart_command(std::span<const std::string> arguments,
    std::span<const std::string> environment, bool fullscreen) {
    RestartCommand command;
    std::size_t index = 0;
    if (index < arguments.size()) command.arguments.push_back(arguments[index++]);
    if (index < arguments.size() && arguments[index] == "run") command.arguments.push_back(arguments[index++]);
    // Every runner option takes one value except --help and --fullscreen, so
    // a value that reads like an option stays with its option.
    for (; index < arguments.size(); ++index) {
        const auto& option = arguments[index];
        if (option == "--fullscreen") continue;  // follows the window below
        if (option == "--help") {
            command.arguments.push_back(option);
            continue;
        }
        const bool kept = option != "--refresh-hz";
        if (kept) command.arguments.push_back(option);
        if (index + 1 < arguments.size()) {
            ++index;
            if (kept) command.arguments.push_back(arguments[index]);
        }
    }
    if (fullscreen) {
        if (command.arguments.size() == 1) command.arguments.emplace_back("run");
        command.arguments.emplace_back("--fullscreen");
    }
    for (const auto& entry : environment) {
        if (entry.starts_with("ARMIGO_COMPAT_FRAME_RATE=") || entry.starts_with("ARMIGO_VI_REFRESH_HZ=")) continue;
        command.environment.push_back(entry);
    }
    return command;
}

void apply_performance_overlay(const PlayerChoices& choices) {
    if (!std::getenv("MTL_HUD_ENABLED"))
        setenv("MTL_HUD_ENABLED", choices.performance_overlay ? "1" : "0", 1);
}

void apply_player_choices(const PlayerChoices& choices, std::uint32_t width, RunnerArguments& arguments) {
    arguments.operation_mode = choices.handheld ? services::OperationMode::Handheld : services::OperationMode::Console;
    apply_frame_rate_choices(choices);
    if (width && !std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH"))
        setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", std::to_string(width).c_str(), 1);
    arguments.content_width = width;
    arguments.display_sync = choices.vsync;
}

void apply_frame_rate_choices(const PlayerChoices& choices) {
    // A title mode selects game timing separately from the guest display cadence.
    const auto refresh_rate = choices.display_refresh_rate ? choices.display_refresh_rate : choices.frame_rate;
    if (refresh_rate && !std::getenv("ARMIGO_VI_REFRESH_HZ"))
        setenv("ARMIGO_VI_REFRESH_HZ", std::to_string(refresh_rate).c_str(), 1);
    if (choices.frame_rate && !std::getenv("ARMIGO_COMPAT_RESOLUTION_RATE"))
        setenv("ARMIGO_COMPAT_RESOLUTION_RATE", std::to_string(choices.frame_rate).c_str(), 1);
    if (choices.title_frame_rate && !std::getenv("ARMIGO_COMPAT_FRAME_RATE"))
        setenv("ARMIGO_COMPAT_FRAME_RATE", std::to_string(choices.frame_rate).c_str(), 1);
}

} // namespace armigo::tools
