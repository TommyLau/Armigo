// ============================================================================
// Armigo Platform - Host Display Capabilities (macOS)
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/display_capabilities.h>

#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>

#include <algorithm>
#include <cmath>

namespace armigo::platform {
namespace {

// Refresh rate of a CoreGraphics mode. Modes that describe no rate are the
// fixed-rate built-in panels; they report the panel's nominal rate through
// CGDisplayModeGetRefreshRate returning zero, so the display's reported frame
// rate from AppKit is used instead.
double mode_refresh_hz(CGDisplayModeRef mode, double fallback) {
    const double rate = CGDisplayModeGetRefreshRate(mode);
    if (rate > 0.0) return rate;
    return fallback > 0.0 ? fallback : 60.0;
}

std::uint32_t mode_width(CGDisplayModeRef mode) {
    return static_cast<std::uint32_t>(CGDisplayModeGetPixelWidth(mode));
}

std::uint32_t mode_height(CGDisplayModeRef mode) {
    return static_cast<std::uint32_t>(CGDisplayModeGetPixelHeight(mode));
}

} // namespace

DisplayCapabilities query_display_capabilities() noexcept {
    DisplayCapabilities result;
    @autoreleasepool {
        const CGDirectDisplayID display = CGMainDisplayID();
        CGDisplayModeRef active = CGDisplayCopyDisplayMode(display);
        // AppKit reports the highest rate the panel runs at, which is the rate a
        // fixed-rate mode without a described rate uses.
        double panel_rate = 0.0;
        if (NSScreen* screen = NSScreen.mainScreen)
            panel_rate = screen.maximumFramesPerSecond;
        if (active) {
            result.current.width = mode_width(active);
            result.current.height = mode_height(active);
            result.current.refresh_hz = mode_refresh_hz(active, panel_rate);
            result.current.variable_refresh = panel_rate > 0.0 &&
                result.current.refresh_hz + 0.01 < panel_rate;
            result.max_refresh_hz = std::max(result.current.refresh_hz, panel_rate);
            CGDisplayModeRelease(active);
        }
        // Modes the panel offers, so the service can describe them to an
        // application that asks for what the display supports.
        CFArrayRef modes = CGDisplayCopyAllDisplayModes(display, nullptr);
        if (modes) {
            const CFIndex count = CFArrayGetCount(modes);
            for (CFIndex index = 0; index < count; ++index) {
                auto* mode = static_cast<CGDisplayModeRef>(
                    const_cast<void*>(CFArrayGetValueAtIndex(modes, index)));
                if (!mode) continue;
                DisplayModeCapability entry;
                entry.width = mode_width(mode);
                entry.height = mode_height(mode);
                entry.refresh_hz = mode_refresh_hz(mode, panel_rate);
                entry.variable_refresh = panel_rate > 0.0 &&
                    entry.refresh_hz + 0.01 < panel_rate;
                result.max_refresh_hz = std::max(result.max_refresh_hz, entry.refresh_hz);
                result.modes.push_back(entry);
            }
            CFRelease(modes);
        }
        if (const char* override = std::getenv("ARMIGO_VI_REFRESH_HZ");
            override && *override) {
            const double rate = std::strtod(override, nullptr);
            if (rate > 0.0 && rate < 10000.0) {
                // Pins the guest's cadence for measurement; the panel's own
                // capability is left as the host reports it.
                result.guest_override_hz = rate;
            }
        }
    }
    return result;
}

} // namespace armigo::platform
