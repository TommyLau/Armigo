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

#include "player_settings.h"
#import <AppKit/AppKit.h>

#include <string>

namespace armigo::tools {

// "Auto (30 fps)", "60 fps" or "120 fps (Experimental)" for one offered rate.
NSString* player_frame_rate_title(const MacFrameRateSelection& selection, int rate);

// Populate the Settings window's control and select the stored rate, or the
// rate the launch fell back to when the stored one is not offered.
void populate_player_frame_rate_menu(NSPopUpButton* button,
    const MacFrameRateSelection& selection, int stored_rate);

} // namespace armigo::tools

// The menu bar's Frame Rate menu, which the Settings window shares: the rate
// the game runs at, the rates the title offers on the launch display and the
// choice kept for the title. A title mode is chosen when the game starts, so a
// choice the running game does not use asks to restart the game; Later keeps
// it for the next start.
@interface ArmigoFrameRateMenu : NSObject <NSMenuDelegate>
- (instancetype)initWithSelection:(const armigo::tools::MacFrameRateSelection&)selection
                              key:(const std::string&)key;
@property (nonatomic, readonly) NSMenu* menu;
- (const armigo::tools::MacFrameRateSelection&)selection;
// The stored choice: 0 (Auto), 30, 60 or 120.
@property (nonatomic, readonly) int storedRate;
// "Running at 30 fps", and the rate the next start uses when it differs.
@property (nonatomic, readonly) NSString* status;
@property (nonatomic, readonly) BOOL restartPending;
// Restarts the game; installed by the player.
@property (nonatomic, copy) void (^restart)(void);
// Asks whether to restart now. The default shows an alert on the key window.
@property (nonatomic, copy) void (^ask)(NSString* message, NSString* information, void (^answer)(BOOL restart));
// Runs after every choice, so other controls can show it.
@property (nonatomic, copy) void (^changed)(void);
- (void)chooseRate:(int)rate;
// Adds the menu to the menu bar, after the app menu.
- (void)install;
@end
