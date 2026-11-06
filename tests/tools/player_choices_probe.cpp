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

int main(int argc, char* argv[]) {
    auto choices = armigo::tools::read_player_choices();
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view(argv[index]) == "-expect_default" &&
            choices.default_frame_rate != std::strtoul(argv[index + 1], nullptr, 10)) return 1;
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
}
