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

#include "player_settings.h"
#import <AppKit/AppKit.h>

namespace armigo::tools {

// Populate the macOS control and select its stored/effective rate. An inherited
// override outside the ordinary choices remains visible as a disabled item.
void populate_player_frame_rate_menu(NSPopUpButton* button,
    const MacFrameRateSelection& selection, int stored_rate);

} // namespace armigo::tools
