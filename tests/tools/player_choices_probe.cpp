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

int main() {
    const auto choices = armigo::tools::read_player_choices();
    armigo::tools::RunnerArguments arguments;
    armigo::tools::apply_player_choices(choices, 1280, arguments);
    const char* selected = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
    const char* refresh = std::getenv("ARMIGO_VI_REFRESH_HZ");
    std::printf("%d %d %u %s %s\n", choices.frame_rate, choices.title_frame_rate,
                choices.display_refresh_rate, selected ? selected : "unset", refresh ? refresh : "unset");
}
