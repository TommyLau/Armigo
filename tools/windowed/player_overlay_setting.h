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

#import <AppKit/AppKit.h>

// A launch-time setting shared by the Settings control and its restart prompt.
// Defaults and prompt callbacks can be isolated without starting a game.
@interface ArmigoOverlaySetting : NSObject
- (instancetype)initWithDefaults:(NSUserDefaults*)defaults
                            key:(NSString*)key
                 runningEnabled:(BOOL)enabled
            environmentOverride:(BOOL)environmentOverride;
@property (nonatomic, readonly) BOOL storedEnabled;
@property (nonatomic, readonly) BOOL restartPending;
@property (nonatomic, readonly) NSString* status;
@property (nonatomic, copy) void (^restart)(void);
@property (nonatomic, copy) void (^changed)(void);
@property (nonatomic, copy) void (^ask)(NSString* message, NSString* information, void (^answer)(BOOL restart));
- (void)populate:(NSPopUpButton*)button;
- (void)choose:(NSPopUpButton*)sender;
- (void)chooseEnabled:(BOOL)enabled;
@end
