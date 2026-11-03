// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Internal to the Apple platform layer: the refreshes of the display a view or
// screen is shown on, delivered on a thread of their own.

#pragma once

#include <armigo/platform/application.h>

#include <chrono>
#include <memory>

#import <Foundation/Foundation.h>

namespace armigo::platform {

// Runs a display link until destroyed. Destruction stops it and waits for the
// thread, so no refresh is delivered afterwards.
class DisplayRefreshLink {
public:
    virtual ~DisplayRefreshLink() = default;
    virtual bool set_rate(double rate) noexcept {
        static_cast<void>(rate);
        return false;
    }
};

// A host time in seconds (display link and CACurrentMediaTime time base,
// which stops while the Mac sleeps) on the steady clock (which keeps counting
// through sleep), through the two clocks' offset now.
std::chrono::steady_clock::time_point steady_from_host_time(double seconds);

// Starts a display link for `source` (an NSView, which follows the screen the
// view is on, or an NSScreen) on its own thread and run loop, so refreshes do
// not wait for the application's event pump. Each refresh reports the vblank
// time on the steady clock and the display's current refresh period. Main
// thread. Returns null when the source cannot create a display link.
std::unique_ptr<DisplayRefreshLink> start_display_refresh(id source, DisplayRefreshCallback handler);

} // namespace armigo::platform
