// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_overlay_setting.h"

@implementation ArmigoOverlaySetting {
    NSUserDefaults* _defaults;
    NSString* _key;
    BOOL _runningEnabled;
    BOOL _environmentOverride;
}

- (instancetype)initWithDefaults:(NSUserDefaults*)defaults key:(NSString*)key
                 runningEnabled:(BOOL)enabled environmentOverride:(BOOL)environmentOverride {
    self = [super init];
    if (!self) return nil;
    _defaults = defaults;
    _key = [key copy];
    _runningEnabled = enabled;
    _environmentOverride = environmentOverride;
    _ask = ^(NSString* message, NSString* information, void (^answer)(BOOL)) {
        NSAlert* alert = [NSAlert new];
        alert.messageText = message;
        alert.informativeText = information;
        [alert addButtonWithTitle:@"Restart Now"];
        [alert addButtonWithTitle:@"Later"];
        void (^done)(NSModalResponse) = ^(NSModalResponse response) {
            dispatch_async(dispatch_get_main_queue(), ^{ answer(response == NSAlertFirstButtonReturn); });
        };
        NSWindow* window = NSApp.keyWindow ? NSApp.keyWindow : NSApp.mainWindow;
        if (window) [alert beginSheetModalForWindow:window completionHandler:done];
        else done([alert runModal]);
    };
    return self;
}

- (BOOL)storedEnabled { return [_defaults boolForKey:_key]; }
- (BOOL)restartPending { return !_environmentOverride && self.storedEnabled != _runningEnabled; }

- (NSString*)status {
    NSString* stored = self.storedEnabled ? @"On" : @"Off";
    if (_environmentOverride)
        return [NSString stringWithFormat:@"MTL_HUD_ENABLED overrides this setting; %@ stored", stored];
    NSString* running = _runningEnabled ? @"On" : @"Off";
    if (self.restartPending)
        return [NSString stringWithFormat:@"Running with overlay %@; %@ after a restart", running, stored];
    return [NSString stringWithFormat:@"Running with overlay %@", running];
}

- (void)populate:(NSPopUpButton*)button {
    [button removeAllItems];
    button.target = self;
    button.action = @selector(choose:);
    for (NSNumber* enabled in @[@NO, @YES]) {
        [button addItemWithTitle:enabled.boolValue ? @"On" : @"Off"];
        button.lastItem.representedObject = enabled;
        if (enabled.boolValue == self.storedEnabled) [button selectItem:button.lastItem];
    }
    if (self.restartPending) {
        [button.menu addItem:NSMenuItem.separatorItem];
        [button addItemWithTitle:@"Restart Game Now…"];
    }
}

- (void)choose:(NSPopUpButton*)sender {
    id value = sender.selectedItem.representedObject;
    if ([value isKindOfClass:NSNumber.class]) [self chooseEnabled:[value boolValue]];
    else {
        [self populate:sender];
        [self offerRestart];
    }
}

- (void)chooseEnabled:(BOOL)enabled {
    if (enabled == self.storedEnabled) return;
    [_defaults setBool:enabled forKey:_key];
    // exec replaces the process after Restart Now, so flush the saved choice first.
    [_defaults synchronize];
    if (self.changed) self.changed();
    [self offerRestart];
}

- (void)offerRestart {
    if (!self.restartPending || !self.ask) return;
    const BOOL requested = self.storedEnabled;
    __weak ArmigoOverlaySetting* weak = self;
    self.ask([NSString stringWithFormat:@"Restart the game with Performance Overlay %@?", requested ? @"On" : @"Off"],
        @"Metal reads this setting when the game starts. Armigo closes the game and starts it again; "
         @"progress since your last save is lost.\n\nChoose Later to keep playing and use the new choice "
         @"the next time the game starts.",
        ^(BOOL restart) {
            ArmigoOverlaySetting* setting = weak;
            if (restart && setting.restartPending && setting.storedEnabled == requested && setting.restart)
                setting.restart();
        });
}
@end
