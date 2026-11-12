// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_settings.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char* argv[]) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "-write_language" && index + 2 < argc) {
            armigo::tools::write_language_preference(argv[index + 1], argv[index + 2]);
            index += 2;
        } else if (std::string_view(argv[index]) == "-write_frame_rate" && index + 2 < argc) {
            armigo::tools::write_frame_rate_preference(argv[index + 1], std::atoi(argv[index + 2]));
            index += 2;
        }
    }
    auto choices = armigo::tools::read_player_choices();
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view(argv[index]) == "-expect_default" &&
            choices.default_frame_rate != std::strtoul(argv[index + 1], nullptr, 10)) return 1;
    }
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view(argv[index]) == "-expect_experimental" &&
            choices.experimental_frame_rates != std::vector<std::uint32_t>{static_cast<std::uint32_t>(
                std::strtoul(argv[index + 1], nullptr, 10))}) return 1;
    }
    for (int index = 1; index + 2 < argc; ++index) {
        if (std::string_view(argv[index]) == "-expect_language") {
            const auto stored = armigo::tools::read_language_preference(argv[index + 1]);
            if (stored.value_or("auto") != argv[index + 2]) return 1;
        }
    }
    armigo::platform::DisplayCapabilities display;
    display.current = {1920, 1080, 60, false};
    for (int index = 1; index + 1 < argc; ++index)
        if (std::string_view(argv[index]) == "-display_hz") display.current.refresh_hz = std::strtod(argv[index + 1], nullptr);
    const auto selection = armigo::tools::select_mac_frame_rate(choices, display);
    choices.frame_rate = selection.frame_rate;
    if (choices.frame_rate == 120) choices.display_refresh_rate = 120;
    armigo::tools::RunnerArguments arguments;
    arguments.refresh_hz = selection.refresh_hz;
    armigo::tools::apply_player_choices(choices, 1280, arguments);
    const char* selected = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
    const char* refresh = std::getenv("ARMIGO_VI_REFRESH_HZ");
    std::printf("%d %d %u %s %s\n", choices.frame_rate, choices.title_frame_rate,
                choices.display_refresh_rate, selected ? selected : "unset", refresh ? refresh : "unset");
    std::printf("clock=%.0f fallback=%d options=", arguments.refresh_hz, selection.fell_back);
    for (int rate : selection.options) std::printf("%d,", rate);
    std::printf("\n");

    const char* language_key = nullptr;
    armigo::tools::LanguageManifest manifest;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view(argv[index]) == "-language_key") {
            language_key = argv[index + 1];
        } else if (std::string_view(argv[index]) == "-languages") {
            std::string_view values(argv[index + 1]);
            while (!values.empty()) {
                const auto comma = values.find(',');
                manifest.languages.emplace_back(values.substr(0, comma));
                values = comma == std::string_view::npos ? std::string_view{} : values.substr(comma + 1);
            }
        }
    }
    if (language_key) {
        const auto choice = armigo::tools::game_language_choice(manifest,
            armigo::tools::read_language_preference(language_key));
        std::printf("language=%s fallback=%d supported=", choice.preference.c_str(), choice.fell_back);
        for (const auto& language : choice.supported_languages) std::printf("%s,", language.c_str());
        std::printf("\n");
    }
}
