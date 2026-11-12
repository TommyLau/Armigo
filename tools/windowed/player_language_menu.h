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

// The system's localized display name for a Horizon language code.
NSString* player_language_title(std::string_view language);

} // namespace armigo::tools

// The menu bar's Game Language menu, shared with the Settings window. Choices
// are launch-time console settings: a different next language asks to restart,
// while Later keeps the running game unchanged.
@interface ArmigoLanguageMenu : NSObject <NSMenuDelegate>
- (instancetype)initWithManifest:(const armigo::tools::LanguageManifest&)manifest
                             key:(const std::string&)key
                 runningLanguage:(const std::string&)runningLanguage;
@property (nonatomic, readonly) NSMenu* menu;
@property (nonatomic, readonly) NSString* status;
@property (nonatomic, readonly) BOOL restartPending;
@property (nonatomic, copy) void (^restart)(void);
@property (nonatomic, copy) void (^ask)(NSString* message, NSString* information, void (^answer)(BOOL restart));
@property (nonatomic, copy) void (^changed)(void);
- (void)chooseLanguage:(const std::string&)language;
- (void)populate:(NSPopUpButton*)button;
- (void)install;
@end
