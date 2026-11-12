// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_overlay_setting.h"

#include <cstdio>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)
void select(NSPopUpButton* button, NSInteger index) {
    [button selectItemAtIndex:index];
    CHECK([NSApp sendAction:button.action to:button.target from:button]);
}
}

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        for (NSNumber* initial in @[@NO, @YES]) {
            NSString* suite = [@"com.armigo.tests.overlay." stringByAppendingString:NSUUID.UUID.UUIDString];
            NSUserDefaults* defaults = [[NSUserDefaults alloc] initWithSuiteName:suite];
            [defaults setBool:initial.boolValue forKey:@"performance_overlay"];
            ArmigoOverlaySetting* setting = [[ArmigoOverlaySetting alloc] initWithDefaults:defaults
                key:@"performance_overlay" runningEnabled:initial.boolValue environmentOverride:NO];
            NSPopUpButton* button = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
            [setting populate:button];
            __block unsigned asked = 0, restarted = 0, changed = 0;
            __block void (^answer)(BOOL) = nil;
            __weak ArmigoOverlaySetting* weakSetting = setting;
            setting.restart = ^{ ++restarted; };
            setting.changed = ^{ ++changed; [weakSetting populate:button]; };
            setting.ask = ^(NSString* message, NSString* information, void (^reply)(BOOL)) {
                ++asked;
                CHECK([message containsString:@"Restart the game with Performance Overlay"]);
                CHECK([information containsString:@"progress since your last save is lost"]);
                CHECK([information containsString:@"Choose Later"]);
                NSUserDefaults* fresh = [[NSUserDefaults alloc] initWithSuiteName:suite];
                CHECK([fresh boolForKey:@"performance_overlay"] == weakSetting.storedEnabled);
                answer = [reply copy];
            };
            CHECK(!setting.restartPending && button.numberOfItems == 2);
            select(button, initial.boolValue ? 1 : 0);
            CHECK(asked == 0 && changed == 0 && restarted == 0);
            select(button, initial.boolValue ? 0 : 1);
            CHECK(asked == 1 && changed == 1 && restarted == 0 && setting.restartPending);
            CHECK(setting.storedEnabled != initial.boolValue);
            CHECK([setting.status containsString:initial.boolValue ? @"Running with overlay On; Off after a restart"
                                                                 : @"Running with overlay Off; On after a restart"]);
            answer(NO);
            CHECK(restarted == 0 && setting.restartPending);
            select(button, initial.boolValue ? 0 : 1);
            CHECK(asked == 1 && changed == 1);
            CHECK([button.lastItem.title isEqualToString:@"Restart Game Now…"]);
            select(button, button.numberOfItems - 1);
            CHECK(asked == 2);
            answer(YES);
            CHECK(restarted == 1);

            // A fresh launch consumes the saved choice without another prompt.
            NSUserDefaults* relaunchedDefaults = [[NSUserDefaults alloc] initWithSuiteName:suite];
            ArmigoOverlaySetting* relaunched = [[ArmigoOverlaySetting alloc] initWithDefaults:relaunchedDefaults
                key:@"performance_overlay" runningEnabled:[relaunchedDefaults boolForKey:@"performance_overlay"]
                environmentOverride:NO];
            CHECK(!relaunched.restartPending && relaunched.storedEnabled != initial.boolValue);
            CHECK([relaunched.status isEqualToString:initial.boolValue ? @"Running with overlay Off" : @"Running with overlay On"]);

            // Returning to the running value cancels pending state and a stale answer.
            void (^stale)(BOOL) = answer;
            select(button, initial.boolValue ? 1 : 0);
            CHECK(!setting.restartPending && asked == 2 && changed == 2);
            stale(YES);
            CHECK(restarted == 1);
            setting.ask = nil; setting.changed = nil; setting.restart = nil;
            [defaults removePersistentDomainForName:suite];
        }
        for (NSNumber* initial in @[@NO, @YES]) {
            NSString* suite = [@"com.armigo.tests.overlay." stringByAppendingString:NSUUID.UUID.UUIDString];
            NSUserDefaults* defaults = [[NSUserDefaults alloc] initWithSuiteName:suite];
            [defaults setBool:initial.boolValue forKey:@"performance_overlay"];
            ArmigoOverlaySetting* setting = [[ArmigoOverlaySetting alloc] initWithDefaults:defaults
                key:@"performance_overlay" runningEnabled:initial.boolValue environmentOverride:YES];
            __block unsigned asked = 0;
            setting.ask = ^(NSString*, NSString*, void (^)(BOOL)) { ++asked; };
            [setting chooseEnabled:!initial.boolValue];
            CHECK(asked == 0 && !setting.restartPending && setting.storedEnabled != initial.boolValue);
            CHECK([setting.status containsString:@"MTL_HUD_ENABLED overrides this setting"]);
            CHECK([setting.status containsString:initial.boolValue ? @"Off stored" : @"On stored"]);
            [defaults removePersistentDomainForName:suite];
        }
    }
    return failures ? 1 : 0;
}
