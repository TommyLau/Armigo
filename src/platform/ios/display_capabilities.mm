// ============================================================================
// Armigo Platform - Host Display Capabilities (iOS)
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/display_capabilities.h>
#import <UIKit/UIKit.h>

#include <algorithm>
#include <cstdlib>

namespace armigo::platform {

DisplayCapabilities query_display_capabilities() noexcept {
    __block DisplayCapabilities result;
    const auto query = ^{
        @autoreleasepool {
            try {
                UIScreen* screen = nil;
                for (UIScene* scene in UIApplication.sharedApplication.connectedScenes) {
                    if (![scene isKindOfClass:UIWindowScene.class]) continue;
                    if (!screen || scene.activationState == UISceneActivationStateForegroundActive)
                        screen = ((UIWindowScene*)scene).screen;
                    if (scene.activationState == UISceneActivationStateForegroundActive) break;
                }
                if (!screen) return;
                const double refresh = screen.maximumFramesPerSecond;
                const auto mode = [refresh](CGSize size) {
                    return DisplayModeCapability{
                        static_cast<std::uint32_t>(std::max(size.width, size.height)),
                        static_cast<std::uint32_t>(std::min(size.width, size.height)), refresh, false};
                };
                result.current = mode(screen.currentMode.size);
                result.max_refresh_hz = refresh;
                result.modes.push_back(result.current);
                for (UIScreenMode* available in screen.availableModes) {
                    const auto candidate = mode(available.size);
                    if (candidate.width != result.current.width || candidate.height != result.current.height)
                        result.modes.push_back(candidate);
                }
                if (const char* override = std::getenv("ARMIGO_VI_REFRESH_HZ"); override && *override) {
                    const double rate = std::strtod(override, nullptr);
                    if (rate > 0 && rate < 10000) result.guest_override_hz = rate;
                }
            } catch (...) { result = {}; }
        }
    };
    if (NSThread.isMainThread) query();
    else dispatch_sync(dispatch_get_main_queue(), query);
    return result;
}

} // namespace armigo::platform
