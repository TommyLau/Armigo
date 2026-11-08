// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_frame_rate_menu.h"

#include <algorithm>

namespace armigo::tools {

NSString* player_frame_rate_title(const MacFrameRateSelection& selection, int rate) {
    if (!rate)
        return selection.default_rate ? [NSString stringWithFormat:@"Auto (%u fps)", selection.default_rate] : @"Auto";
    const bool experimental = std::find(selection.experimental_rates.begin(),
        selection.experimental_rates.end(), rate) != selection.experimental_rates.end();
    return [NSString stringWithFormat:experimental ? @"%d fps (Experimental)" : @"%d fps", rate];
}

void populate_player_frame_rate_menu(NSPopUpButton* button,
    const MacFrameRateSelection& selection, int stored_rate) {
    [button removeAllItems];
    button.menu.autoenablesItems = NO;
    for (int rate : selection.options) {
        [button addItemWithTitle:player_frame_rate_title(selection, rate)];
        button.lastItem.representedObject = @[@"frame_rate", @(rate)];
    }
    // A stored rate the launch display does not offer fell back to the
    // highest offered one.
    auto selected = std::find(selection.options.begin(), selection.options.end(), stored_rate);
    if (selected == selection.options.end() && !selection.options.empty()) selected = selection.options.end() - 1;
    if (selected != selection.options.end())
        [button selectItemAtIndex:std::distance(selection.options.begin(), selected)];
}

} // namespace armigo::tools

@implementation ArmigoFrameRateMenu {
    armigo::tools::MacFrameRateSelection _selection;
    std::string _key;
    NSMenu* _menu;
}

- (instancetype)initWithSelection:(const armigo::tools::MacFrameRateSelection&)selection
                              key:(const std::string&)key {
    self = [super init];
    if (!self) return nil;
    _selection = selection;
    _key = key;
    _menu = [[NSMenu alloc] initWithTitle:@"Frame Rate"];
    _menu.autoenablesItems = NO;
    _menu.delegate = self;
    _ask = ^(NSString* message, NSString* information, void (^answer)(BOOL)) {
        NSAlert* alert = [NSAlert new];
        alert.messageText = message;
        alert.informativeText = information;
        [alert addButtonWithTitle:@"Restart Now"];
        [alert addButtonWithTitle:@"Later"];
        // The sheet is still attached while its completion handler runs, and a
        // window with a sheet ignores a close, so the answer (which closes the
        // game for a restart) waits until the sheet has gone.
        void (^done)(NSModalResponse) = ^(NSModalResponse response) {
            const BOOL restart = response == NSAlertFirstButtonReturn;
            dispatch_async(dispatch_get_main_queue(), ^{
                answer(restart);
            });
        };
        // A sheet keeps the run loop in its ordinary mode, so the game's
        // display refreshes continue while it is open.
        NSWindow* window = NSApp.keyWindow ? NSApp.keyWindow : NSApp.mainWindow;
        if (window)
            [alert beginSheetModalForWindow:window completionHandler:done];
        else
            done([alert runModal]);
    };
    [self menuNeedsUpdate:_menu];
    return self;
}

- (NSMenu*)menu {
    return _menu;
}

- (const armigo::tools::MacFrameRateSelection&)selection {
    return _selection;
}

- (int)storedRate {
    const auto stored = armigo::tools::read_frame_rate_preference(_key);
    return stored && (*stored == 0 || *stored == 30 || *stored == 60 || *stored == 120) ? *stored : 0;
}

// The rate the launch display offers for a choice: a stored rate it does not
// offer falls back to the highest offered one, as the launch does.
- (int)offeredRate:(int)choice {
    const auto& options = _selection.options;
    if (std::find(options.begin(), options.end(), choice) == options.end() && !options.empty())
        return options.back();
    return choice;
}

// The title mode a choice selects: Auto is the title default.
- (int)modeOf:(int)choice {
    const int rate = [self offeredRate:choice];
    return rate ? rate : static_cast<int>(_selection.default_rate);
}

// The mode the game runs at, or -1 for an environment value that is not a
// rate (variable mode, or one the runtime rejects).
- (int)runningMode {
    if (_selection.environment_override) {
        const auto& value = *_selection.environment_override;
        if (value == "30" || value == "60" || value == "120") return std::stoi(value);
        if (value.empty() || value == "0") return static_cast<int>(_selection.default_rate);
        return -1;
    }
    return _selection.frame_rate ? _selection.frame_rate : static_cast<int>(_selection.default_rate);
}

- (NSString*)modeTitle:(int)mode {
    return mode ? [NSString stringWithFormat:@"%d fps", mode] : @"Auto";
}

- (BOOL)restartPending {
    return [self modeOf:self.storedRate] != [self runningMode];
}

- (NSString*)runningStatus {
    if (_selection.environment_override) {
        const int mode = [self runningMode];
        NSString* value = @(_selection.environment_override->c_str());
        return mode < 0 ? [NSString stringWithFormat:@"Running in %@ mode (ARMIGO_COMPAT_FRAME_RATE)", value]
                        : [NSString stringWithFormat:@"Running at %@ (ARMIGO_COMPAT_FRAME_RATE)",
                                                     [self modeTitle:mode]];
    }
    return [NSString stringWithFormat:@"Running at %@", [self modeTitle:[self runningMode]]];
}

- (NSString*)status {
    if (!self.restartPending) return [self runningStatus];
    return [NSString stringWithFormat:@"%@; %@ after a restart", [self runningStatus],
                                      [self modeTitle:[self modeOf:self.storedRate]]];
}

- (void)menuNeedsUpdate:(NSMenu*)menu {
    [menu removeAllItems];
    NSMenuItem* running = [menu addItemWithTitle:[self runningStatus] action:nil keyEquivalent:@""];
    running.enabled = NO;
    const int stored = self.storedRate;
    const BOOL pending = self.restartPending;
    if (pending) {
        NSMenuItem* next = [menu addItemWithTitle:[NSString stringWithFormat:@"%@ after a restart",
                                                       [self modeTitle:[self modeOf:stored]]]
                                           action:nil keyEquivalent:@""];
        next.enabled = NO;
    }
    [menu addItem:NSMenuItem.separatorItem];
    const int checked = [self offeredRate:stored];
    for (int rate : _selection.options) {
        NSMenuItem* item = [menu addItemWithTitle:armigo::tools::player_frame_rate_title(_selection, rate)
                                           action:@selector(choose:) keyEquivalent:@""];
        item.target = self;
        item.tag = rate;
        item.state = rate == checked ? NSControlStateValueOn : NSControlStateValueOff;
    }
    if (pending) {
        [menu addItem:NSMenuItem.separatorItem];
        NSMenuItem* restart = [menu addItemWithTitle:@"Restart Game Now…" action:@selector(restartNow:)
                                       keyEquivalent:@""];
        restart.target = self;
    }
}

- (void)choose:(NSMenuItem*)sender {
    [self chooseRate:static_cast<int>(sender.tag)];
}

- (void)chooseRate:(int)rate {
    armigo::tools::write_frame_rate_preference(_key, rate);
    [self menuNeedsUpdate:_menu];
    if (self.changed) self.changed();
    if (self.restartPending) [self offerRestart];
}

- (void)restartNow:(id)sender {
    (void)sender;
    [self offerRestart];
}

- (void)offerRestart {
    if (!self.ask) return;
    NSString* mode = [self modeTitle:[self modeOf:self.storedRate]];
    __weak ArmigoFrameRateMenu* weak = self;
    self.ask([NSString stringWithFormat:@"Restart the game at %@?", mode],
             @"The frame rate is chosen when the game starts. Armigo closes the game and starts it again at the "
             @"new rate; progress since your last save is lost.\n\nChoose Later to keep playing and use the new "
             @"rate the next time the game starts.",
             ^(BOOL restart) {
                 ArmigoFrameRateMenu* menu = weak;
                 if (restart && menu.restart) menu.restart();
             });
}

- (void)install {
    NSMenu* bar = NSApp.mainMenu;
    if (!bar) {
        // The player builds its menu bar when its window opens.
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2), dispatch_get_main_queue(), ^{
            [self install];
        });
        return;
    }
    NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:@"Frame Rate" action:nil keyEquivalent:@""];
    item.submenu = _menu;
    [bar insertItem:item atIndex:std::min<NSInteger>(1, bar.numberOfItems)];
}

@end
