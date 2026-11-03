// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// On-screen keyboard guide for the macOS game window. AppKit types stay in
// this platform file; portable headers do not see them.

#pragma once

#import <AppKit/AppKit.h>

namespace armigo::platform {

// Adds the guide as a sibling of the Metal view. It does not accept focus or
// clicks. Headless runs never call this: they do not create a window.
void install_key_guide(NSView* container, NSView* metal_view);

// Shows the guide if it is hidden, or hides it if it is showing. F1 is not a
// game key; the Metal view calls this from its own key handler.
void toggle_key_guide(NSView* metal_view);

}  // namespace armigo::platform
