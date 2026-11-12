// ============================================================================
// Armigo Tools - Player Settings Storage
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_settings.h"

#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

namespace armigo::tools {

PlayerChoices read_player_choices() {
    @autoreleasepool {
        NSUserDefaults* defaults = NSUserDefaults.standardUserDefaults;
        // Remove only keys from the earlier layout. Its overlay toggle carries
        // over unless the current setting already exists; display, frame-rate
        // and per-title game-language choices are not migration casualties.
        if ([defaults objectForKey:@"language"] || [defaults objectForKey:@"region"] ||
            [defaults objectForKey:@"fps_overlay"]) {
            if ([defaults boolForKey:@"fps_overlay"] && ![defaults objectForKey:@"performance_overlay"])
                [defaults setBool:YES forKey:@"performance_overlay"];
            for (NSString* key in @[@"language", @"region", @"fps_overlay"])
                [defaults removeObjectForKey:key];
        }
        NSDictionary* rate_policy = NSBundle.mainBundle.infoDictionary[@"ArmigoTitleFrameRate"];
        const bool title_rate = [rate_policy isKindOfClass:[NSDictionary class]] &&
            ([rate_policy[@"Default"] isEqual:@30] || [rate_policy[@"Default"] isEqual:@60]) &&
            ([rate_policy[@"DisplayRefresh"] isEqual:@30] || [rate_policy[@"DisplayRefresh"] isEqual:@60]);
        const NSInteger default_rate = title_rate ? [rate_policy[@"Default"] integerValue] : 60;
        [defaults registerDefaults:@{@"console_mode": @"docked", @"resolution": @"auto", @"frame_rate": @(default_rate),
                                     @"vsync": @YES, @"performance_overlay": @NO}];
        PlayerChoices choices;
        choices.title_frame_rate = title_rate;
        choices.default_frame_rate = static_cast<std::uint32_t>(default_rate);
        if (title_rate) choices.display_refresh_rate = [rate_policy[@"DisplayRefresh"] unsignedIntValue];
        choices.handheld = [[defaults stringForKey:@"console_mode"] isEqualToString:@"handheld"];
        if (NSString* resolution = [defaults stringForKey:@"resolution"]) choices.resolution = resolution.UTF8String;
        const NSInteger rate = [defaults integerForKey:@"frame_rate"];
        choices.frame_rate = rate == 0 || rate == 30 ? static_cast<int>(rate) : 60;
#if !TARGET_OS_IPHONE
        if (rate == 120) choices.frame_rate = 120;
        if (title_rate) {
            id declared = rate_policy[@"SupportedRates"];
            if ([declared isKindOfClass:[NSArray class]] && [declared count]) {
                choices.supported_frame_rates.clear();
                for (id value in declared) {
                    if ([value isEqual:@30] || [value isEqual:@60] || [value isEqual:@120])
                        choices.supported_frame_rates.push_back([value unsignedIntValue]);
                }
                if (choices.supported_frame_rates.empty()) choices.supported_frame_rates = {30, 60};
            }
            id experimental = rate_policy[@"ExperimentalRates"];
            if ([experimental isKindOfClass:[NSArray class]]) {
                for (id value in experimental) {
                    if ([value isEqual:@30] || [value isEqual:@60] || [value isEqual:@120])
                        choices.experimental_frame_rates.push_back([value unsignedIntValue]);
                }
            }
        }
#endif
        choices.vsync = [defaults boolForKey:@"vsync"];
        choices.performance_overlay = [defaults boolForKey:@"performance_overlay"];
        return choices;
    }
}

std::optional<std::string> read_language_preference(const std::string& key) {
    @autoreleasepool {
        id value = [NSUserDefaults.standardUserDefaults objectForKey:@(key.c_str())];
        if (![value isKindOfClass:[NSString class]]) return std::nullopt;
        return [value UTF8String];
    }
}

void write_language_preference(const std::string& key, std::string_view language) {
    @autoreleasepool {
        NSString* value = [[NSString alloc] initWithBytes:language.data() length:language.size()
                                                 encoding:NSUTF8StringEncoding];
        if (!value) return;
        [NSUserDefaults.standardUserDefaults setObject:value forKey:@(key.c_str())];
        // A restart replaces the process right after the choice; keep it.
        [NSUserDefaults.standardUserDefaults synchronize];
    }
}

std::optional<int> read_frame_rate_preference(const std::string& key) {
    @autoreleasepool {
        id value = [NSUserDefaults.standardUserDefaults objectForKey:@(key.c_str())];
        if (![value isKindOfClass:[NSNumber class]]) return std::nullopt;
        return [value intValue];
    }
}

void write_frame_rate_preference(const std::string& key, int rate) {
    @autoreleasepool {
        [NSUserDefaults.standardUserDefaults setInteger:rate forKey:@(key.c_str())];
        // A restart replaces the process right after the choice; keep it.
        [NSUserDefaults.standardUserDefaults synchronize];
    }
}

std::vector<std::string> preferred_host_locales() {
    @autoreleasepool {
        // The app's own language (the per-app language the system's Language &
        // Region settings keep for it), then the user's preferred languages.
        std::vector<std::string> locales;
        for (NSString* locale in NSBundle.mainBundle.preferredLocalizations) locales.emplace_back(locale.UTF8String);
        for (NSString* locale in NSLocale.preferredLanguages) locales.emplace_back(locale.UTF8String);
        return locales;
    }
}

} // namespace armigo::tools
