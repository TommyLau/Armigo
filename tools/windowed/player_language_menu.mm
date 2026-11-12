// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_language_menu.h"

#include <armigo/services/settings.h>

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace armigo::tools {

NSString* player_language_title(std::string_view language) {
    NSString* identifier = [NSString stringWithUTF8String:std::string(language).c_str()];
    NSString* name = [NSLocale.currentLocale displayNameForKey:NSLocaleIdentifier value:identifier];
    return name.length ? name : identifier;
}

} // namespace armigo::tools

@implementation ArmigoLanguageMenu {
    armigo::tools::LanguageManifest _manifest;
    std::string _key;
    std::string _runningLanguage;
    NSMenu* _menu;
}

- (instancetype)initWithManifest:(const armigo::tools::LanguageManifest&)manifest
                             key:(const std::string&)key
                 runningLanguage:(const std::string&)runningLanguage {
    self = [super init];
    if (!self) return nil;
    _manifest = manifest;
    _key = key;
    _runningLanguage = runningLanguage;
    _menu = [[NSMenu alloc] initWithTitle:@"Game Language"];
    _menu.autoenablesItems = NO;
    _menu.delegate = self;
    _ask = ^(NSString* message, NSString* information, void (^answer)(BOOL)) {
        NSAlert* alert = [NSAlert new];
        alert.messageText = message;
        alert.informativeText = information;
        [alert addButtonWithTitle:@"Restart Now"];
        [alert addButtonWithTitle:@"Later"];
        void (^done)(NSModalResponse) = ^(NSModalResponse response) {
            const BOOL restart = response == NSAlertFirstButtonReturn;
            dispatch_async(dispatch_get_main_queue(), ^{
                answer(restart);
            });
        };
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

- (armigo::tools::GameLanguageChoice)choice {
    return armigo::tools::game_language_choice(_manifest, armigo::tools::read_language_preference(_key));
}

- (std::string)selectedLanguage:(const armigo::tools::GameLanguageChoice&)choice {
    if (choice.preference != "auto") return choice.preference;
    const auto locales = armigo::tools::preferred_host_locales();
    const std::vector<std::string_view> preferred(locales.begin(), locales.end());
    const auto selected = armigo::services::set::automatic_language(preferred, _manifest.supported_language_flag);
    return selected ? *selected : _runningLanguage;
}

- (NSString*)nextTitle:(const armigo::tools::GameLanguageChoice&)choice {
    NSString* language = armigo::tools::player_language_title([self selectedLanguage:choice]);
    return choice.preference == "auto" ? [NSString stringWithFormat:@"Auto (%@)", language] : language;
}

- (NSString*)autoTitle {
    auto choice = [self choice];
    choice.preference = "auto";
    return [self nextTitle:choice];
}

- (BOOL)environmentOverride {
    const char* override = std::getenv("ARMIGO_SYSTEM_LANGUAGE");
    return override && *override;
}

- (BOOL)restartPending {
    if ([self environmentOverride]) return NO;
    const auto choice = [self choice];
    return [self selectedLanguage:choice] != _runningLanguage;
}

- (NSString*)runningStatus {
    if ([self environmentOverride]) {
        return [NSString stringWithFormat:@"Running with %s (ARMIGO_SYSTEM_LANGUAGE)",
                                          std::getenv("ARMIGO_SYSTEM_LANGUAGE")];
    }
    return [NSString stringWithFormat:@"Running in %@",
                                      armigo::tools::player_language_title(_runningLanguage)];
}

- (NSString*)status {
    const auto choice = [self choice];
    if ([self environmentOverride])
        return [NSString stringWithFormat:@"%@; %@ stored", [self runningStatus], [self nextTitle:choice]];
    if (!self.restartPending) return [self runningStatus];
    return [NSString stringWithFormat:@"%@; %@ after a restart", [self runningStatus], [self nextTitle:choice]];
}

- (void)menuNeedsUpdate:(NSMenu*)menu {
    [menu removeAllItems];
    NSMenuItem* running = [menu addItemWithTitle:[self runningStatus] action:nil keyEquivalent:@""];
    running.enabled = NO;
    const auto choice = [self choice];
    const BOOL pending = self.restartPending;
    if (pending || [self environmentOverride]) {
        NSString* next = pending ? [NSString stringWithFormat:@"%@ after a restart", [self nextTitle:choice]]
                                 : [NSString stringWithFormat:@"%@ stored", [self nextTitle:choice]];
        NSMenuItem* item = [menu addItemWithTitle:next action:nil keyEquivalent:@""];
        item.enabled = NO;
    }
    [menu addItem:NSMenuItem.separatorItem];
    for (NSUInteger index = 0; index <= choice.supported_languages.size(); ++index) {
        const std::string language = index ? choice.supported_languages[index - 1] : "auto";
        NSString* title = index ? armigo::tools::player_language_title(language) : [self autoTitle];
        NSMenuItem* item = [menu addItemWithTitle:title action:@selector(choose:) keyEquivalent:@""];
        item.target = self;
        item.representedObject = [NSString stringWithUTF8String:language.c_str()];
        item.state = language == choice.preference ? NSControlStateValueOn : NSControlStateValueOff;
    }
    if (pending) {
        [menu addItem:NSMenuItem.separatorItem];
        NSMenuItem* restart = [menu addItemWithTitle:@"Restart Game Now…" action:@selector(restartNow:)
                                       keyEquivalent:@""];
        restart.target = self;
    }
}

- (void)choose:(NSMenuItem*)sender {
    [self chooseLanguage:((NSString*)sender.representedObject).UTF8String];
}

- (void)chooseLanguage:(const std::string&)language {
    const auto validated = armigo::tools::game_language_choice(_manifest, language);
    if (validated.fell_back) return;
    armigo::tools::write_language_preference(_key, validated.preference);
    [self menuNeedsUpdate:_menu];
    if (self.changed) self.changed();
    if (self.restartPending) [self offerRestart];
}

- (void)restartNow:(id)sender {
    (void)sender;
    [self offerRestart];
}

- (void)offerRestart {
    if (!self.ask || ![self restartPending]) return;
    NSString* language = [self nextTitle:[self choice]];
    __weak ArmigoLanguageMenu* weak = self;
    self.ask([NSString stringWithFormat:@"Restart the game with %@?", language],
             @"The game language is chosen when the game starts. Armigo closes the game and starts it again with "
             @"the new choice; progress since your last save is lost.\n\nChoose Later to keep playing and use the "
             @"new choice the next time the game starts.",
             ^(BOOL restart) {
                 ArmigoLanguageMenu* menu = weak;
                 if (restart && menu.restart) menu.restart();
             });
}

- (void)populate:(NSPopUpButton*)button {
    [button removeAllItems];
    button.menu.autoenablesItems = NO;
    const auto choice = [self choice];
    for (NSUInteger index = 0; index <= choice.supported_languages.size(); ++index) {
        const std::string language = index ? choice.supported_languages[index - 1] : "auto";
        NSString* title = index ? armigo::tools::player_language_title(language) : [self autoTitle];
        [button addItemWithTitle:title];
        button.lastItem.representedObject = [NSString stringWithUTF8String:language.c_str()];
        if (language == choice.preference) [button selectItem:button.lastItem];
    }
}

- (void)install {
    NSMenu* bar = NSApp.mainMenu;
    if (!bar) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2), dispatch_get_main_queue(), ^{
            [self install];
        });
        return;
    }
    NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:@"Game Language" action:nil keyEquivalent:@""];
    item.submenu = _menu;
    [bar insertItem:item atIndex:std::min<NSInteger>(2, bar.numberOfItems)];
}

@end
