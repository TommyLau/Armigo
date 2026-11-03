// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "icon.h"

#include <string>
#include <vector>

namespace armigo::pack {

std::variant<std::string, ControlIcon> select_control_icon(std::span<const ControlIcon> icons,
                                                           nacp::Language language) {
    std::vector<std::string_view> names;
    names.reserve(icons.size());
    for (const ControlIcon& icon : icons) {
        names.push_back(icon.name);
    }
    const nacp::IconChoice choice = nacp::choose_icon(names, language);
    if (!choice.ok()) {
        if (choice.detail.empty()) {
            return std::string(nacp::to_string(choice.error));
        }
        return choice.detail;
    }
    for (const ControlIcon& icon : icons) {
        if (icon.name == choice.file_name) {
            return icon;
        }
    }
    return "The chosen control icon " + std::string(choice.file_name) + " is missing.";
}

} // namespace armigo::pack
