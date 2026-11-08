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

#include <armigo/compat/patches.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

// A selection as the launch resolves it, with the frame-rate environment set
// as given.
armigo::tools::MacFrameRateSelection launch_selection(const armigo::tools::PlayerChoices& choices,
    double display_hz, const char* override, double explicit_clock = 0) {
    for (const char* key : {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
                           "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"}) unsetenv(key);
    if (override) setenv("ARMIGO_COMPAT_FRAME_RATE", override, 1);
    armigo::tools::RunnerArguments launch;
    if (explicit_clock) {
        launch.refresh_hz = explicit_clock;
        launch.refresh_hz_explicit = true;
    }
    armigo::platform::DisplayCapabilities display;
    display.current = {1920, 1080, display_hz, false};
    armigo::tools::MacFrameRateSelection result;
    armigo::tools::mac_title_frame_rate_configuration(choices,
        [&](const auto&, const auto& selection) { result = selection; })(launch, display);
    for (const char* key : {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
                           "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"}) unsetenv(key);
    return result;
}

NSArray<NSString*>* titles(NSMenu* menu) {
    NSMutableArray<NSString*>* result = [NSMutableArray array];
    for (NSMenuItem* item in menu.itemArray) {
        if (item.separatorItem) [result addObject:@"-"];
        else [result addObject:item.state == NSControlStateValueOn ? [@"* " stringByAppendingString:item.title]
                                                                    : item.title];
    }
    return result;
}

} // namespace

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
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
            CHECK([[button itemAtIndex:0].title isEqualToString:@"Auto (60 fps)"]);
            CHECK([[button itemAtIndex:2].title isEqualToString:@"60 fps"]);
            // A title without a policy has a plain Auto.
            CHECK([armigo::tools::player_frame_rate_title(armigo::tools::MacFrameRateSelection{}, 0)
                isEqualToString:@"Auto"]);
        }

        NSString* key = [NSString stringWithFormat:@"frame_rate.TEST%d", getpid()];
        const std::string preference = key.UTF8String;
        armigo::tools::PlayerChoices p5r;
        p5r.title_frame_rate = true;
        p5r.default_frame_rate = 30;
        p5r.display_refresh_rate = 60;
        p5r.supported_frame_rates = {30, 60, 120, armigo::compat::variable_frame_rate};

        // The Settings control shows the stored choice; one the display does
        // not offer shows the rate the launch fell back to. The environment
        // no longer locks it: the status line shows what runs.
        struct Case {
            int stored;
            const char* override;
            double display_hz;
            double clock;
            const char* selected_title;
            const char* status;
        };
        for (const auto& test : {
            Case{120, "120", 60, 60, "60 fps", "Running at 120 fps (ARMIGO_COMPAT_FRAME_RATE); 60 fps after a restart"},
            Case{30, "120", 120, 0, "30 fps", "Running at 120 fps (ARMIGO_COMPAT_FRAME_RATE); 30 fps after a restart"},
            Case{120, "120", 120, 0, "120 fps", "Running at 120 fps (ARMIGO_COMPAT_FRAME_RATE)"},
            Case{0, "variable", 120, 0, "Auto (30 fps)",
                 "Running in variable mode (ARMIGO_COMPAT_FRAME_RATE); 30 fps after a restart"},
            Case{120, nullptr, 60, 0, "60 fps", "Running at 60 fps"},
            Case{120, nullptr, 120, 60, "120 fps", "Running at 60 fps; 120 fps after a restart"},
            Case{0, nullptr, 120, 0, "Auto (30 fps)", "Running at 30 fps"},
        }) {
            [NSUserDefaults.standardUserDefaults setInteger:test.stored forKey:key];
            auto choices = p5r;
            choices.frame_rate = test.stored;
            const auto selection = launch_selection(choices, test.display_hz, test.override, test.clock);
            ArmigoFrameRateMenu* menu = [[ArmigoFrameRateMenu alloc] initWithSelection:selection key:preference];
            NSPopUpButton* button = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
            armigo::tools::populate_player_frame_rate_menu(button, menu.selection, menu.storedRate);
            CHECK([button.selectedItem.title isEqualToString:@(test.selected_title)]);
            CHECK(button.selectedItem.enabled);
            CHECK(!button.menu.autoenablesItems);
            CHECK([menu.status isEqualToString:@(test.status)]);
        }

        // Choosing from the menu keeps the rate for the title and asks to
        // restart only when the running game does not use it.
        [NSUserDefaults.standardUserDefaults removeObjectForKey:key];
        auto choices = p5r;
        choices.frame_rate = 0;
        ArmigoFrameRateMenu* menu = [[ArmigoFrameRateMenu alloc]
            initWithSelection:launch_selection(choices, 120, nullptr) key:preference];
        __block int asked = 0;
        __block int restarted = 0;
        __block int changed = 0;
        __block BOOL answer = NO;
        __block NSString* question;
        menu.ask = ^(NSString* message, NSString* information, void (^reply)(BOOL)) {
            (void)information;
            ++asked;
            question = message;
            reply(answer);
        };
        menu.restart = ^{ ++restarted; };
        menu.changed = ^{ ++changed; };
        CHECK(menu.storedRate == 0 && !menu.restartPending);
        CHECK(([titles(menu.menu) isEqual:@[@"Running at 30 fps", @"-", @"* Auto (30 fps)", @"30 fps", @"60 fps",
                                           @"120 fps"]]));
        CHECK(!menu.menu.itemArray[0].enabled && menu.menu.itemArray[2].enabled);
        [menu chooseRate:30];
        CHECK(menu.storedRate == 30 && asked == 0 && changed == 1);
        CHECK(armigo::tools::read_frame_rate_preference(preference) == 30);
        [menu chooseRate:120];
        CHECK(asked == 1 && restarted == 0 && menu.restartPending);
        CHECK([question isEqualToString:@"Restart the game at 120 fps?"]);
        CHECK(armigo::tools::read_frame_rate_preference(preference) == 120);
        CHECK(([titles(menu.menu) isEqual:@[@"Running at 30 fps", @"120 fps after a restart", @"-",
                                           @"Auto (30 fps)", @"30 fps", @"60 fps", @"* 120 fps", @"-",
                                           @"Restart Game Now…"]]));
        answer = YES;
        NSMenuItem* now = menu.menu.itemArray.lastObject;
        [NSApp sendAction:now.action to:now.target from:now];
        CHECK(asked == 2 && restarted == 1);
        [menu chooseRate:60];
        CHECK(asked == 3 && restarted == 2);
        CHECK([question isEqualToString:@"Restart the game at 60 fps?"]);
        // Back to what runs: nothing to restart.
        [menu chooseRate:0];
        CHECK(asked == 3 && restarted == 2 && !menu.restartPending && changed == 4);
        CHECK(menu.menu.itemArray.count == 6);
        // The menu item the player adds to the menu bar.
        NSApp.mainMenu = [NSMenu new];
        [NSApp.mainMenu addItem:[NSMenuItem new]];
        [NSApp.mainMenu addItem:[NSMenuItem new]];
        [menu install];
        CHECK([NSApp.mainMenu.itemArray[1].title isEqualToString:@"Frame Rate"]);
        CHECK(NSApp.mainMenu.itemArray[1].submenu == menu.menu);
        [NSUserDefaults.standardUserDefaults removeObjectForKey:key];
    }
    return failures ? 1 : 0;
}
