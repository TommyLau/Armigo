// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_frame_rate_menu.h"

#include <cstdio>
#include <cstdlib>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)
}

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        struct Case {
            int stored;
            const char* override;
            bool high_only;
            double clock;
            const char* selected_title;
            bool enabled;
        };
        {
            armigo::tools::PlayerChoices choices;
            choices.title_frame_rate = true;
            choices.default_frame_rate = 60;
            choices.supported_frame_rates = {30, 60, 120};
            choices.experimental_frame_rates = {120};
            choices.frame_rate = 120;
            armigo::platform::DisplayCapabilities display;
            display.current = {1920, 1080, 120, false};
            const auto selection = armigo::tools::select_mac_frame_rate(choices, display);
            NSPopUpButton* button = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
            armigo::tools::populate_player_frame_rate_menu(button, selection, 120);
            CHECK([button.selectedItem.title isEqualToString:@"120 fps (Experimental)"]);
            CHECK(([button.selectedItem.representedObject isEqual:@[@"frame_rate", @120]]));
            CHECK([[button itemAtIndex:0].title isEqualToString:@"Auto"]);
            CHECK([[button itemAtIndex:2].title isEqualToString:@"60 fps"]);
        }
        for (const auto& test : {
            Case{120, "120", false, 60, "120 fps (ARMIGO_COMPAT_FRAME_RATE override)", false},
            Case{30, "120", false, 60, "120 fps (ARMIGO_COMPAT_FRAME_RATE override)", false},
            Case{120, "60", true, 60, "60 fps (ARMIGO_COMPAT_FRAME_RATE override)", false},
            Case{120, "30", false, 60, "30 fps", true},
            Case{120, nullptr, false, 60, "60 fps", true},
            Case{120, nullptr, false, 24, "30 fps", true},
        }) {
            for (const char* key : {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
                                   "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"}) unsetenv(key);
            if (test.override) setenv("ARMIGO_COMPAT_FRAME_RATE", test.override, 1);
            armigo::tools::PlayerChoices choices;
            choices.title_frame_rate = true;
            choices.default_frame_rate = 30;
            choices.display_refresh_rate = 60;
            choices.supported_frame_rates = test.high_only ? std::vector<std::uint32_t>{30, 120}
                                                          : std::vector<std::uint32_t>{30, 60, 120};
            choices.frame_rate = test.stored;
            armigo::tools::RunnerArguments launch;
            launch.refresh_hz = test.clock;
            launch.refresh_hz_explicit = true;
            armigo::platform::DisplayCapabilities display;
            display.current = {1920, 1080, 60, false};
            NSPopUpButton* button = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
            auto configure = armigo::tools::mac_player_launch_configuration(choices, 0,
                [&](const auto&, const auto& selection) {
                    armigo::tools::populate_player_frame_rate_menu(button, selection, test.stored);
                });
            configure(launch, display);
            [button.menu update];
            CHECK(button.indexOfSelectedItem >= 0);
            CHECK([button.selectedItem.title isEqualToString:@(test.selected_title)]);
            CHECK(button.selectedItem.enabled == test.enabled);
            if (!test.enabled) {
                CHECK(button.selectedItem == button.lastItem);
                CHECK(button.selectedItem.representedObject == nil);
                CHECK(!button.menu.autoenablesItems);
            }
        }
    }
    return failures ? 1 : 0;
}
