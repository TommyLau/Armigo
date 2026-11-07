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

void populate_player_frame_rate_menu(NSPopUpButton* button,
    const MacFrameRateSelection& selection, int stored_rate) {
    [button removeAllItems];
    button.menu.autoenablesItems = NO;
    for (int rate : selection.options) {
        const bool experimental = std::find(selection.experimental_rates.begin(),
            selection.experimental_rates.end(), rate) != selection.experimental_rates.end();
        NSString* title = rate ? [NSString stringWithFormat:experimental ? @"%d fps (Experimental)" : @"%d fps", rate] : @"Auto";
        [button addItemWithTitle:title];
        button.lastItem.representedObject = @[@"frame_rate", @(rate)];
    }
    int selected_rate = stored_rate;
    if (selection.environment_override) {
        const auto& value = *selection.environment_override;
        const bool known = value.empty() || value == "0" || value == "30" || value == "60" || value == "120";
        if (!known || std::find(selection.options.begin(), selection.options.end(), selection.frame_rate) ==
                          selection.options.end()) {
            NSString* override = [NSString stringWithUTF8String:value.c_str()];
            [button addItemWithTitle:[NSString stringWithFormat:@"%@ fps (ARMIGO_COMPAT_FRAME_RATE override)", override]];
            button.lastItem.enabled = NO;
            [button selectItem:button.lastItem];
            return;
        }
        selected_rate = selection.frame_rate;
    }
    auto selected = std::find(selection.options.begin(), selection.options.end(), selected_rate);
    if (selected == selection.options.end())
        selected = std::find(selection.options.begin(), selection.options.end(), selection.frame_rate);
    if (selected != selection.options.end())
        [button selectItemAtIndex:std::distance(selection.options.begin(), selected)];
}

} // namespace armigo::tools
