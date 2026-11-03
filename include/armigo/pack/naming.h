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

#include <armigo/pack/nacp.h>

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

// Names taken from control data, plus the bundle id derived from the title id.
struct PackedNames {
    // CFBundleDisplayName: the control-data title, unchanged.
    std::string display_name;
    // CFBundleName and the app file name, from the control-data naming library.
    std::string file_name;
    // com.tommylau.armigo.<TitleID>, with the title id in sixteen uppercase hex digits.
    std::string bundle_identifier;
    // Language of display_name. Icon selection uses this.
    nacp::Language language = nacp::Language::american_english;
    // Horizon language codes admitted by SupportedLanguageFlag.
    std::vector<std::string> languages;
    // The control data's title in each of those languages that has one, as
    // (Horizon language code, title) pairs.
    std::vector<std::pair<std::string, std::string>> localized_titles;
    std::uint32_t supported_language_flag = 0;
    // Field problems the parser recorded. They do not reject the game.
    std::vector<std::string> warnings;
};

// Reads the packed app's names from control data. Display name, file name and
// icon language come from the control-data naming library. Returns the
// library's detail string when the control data or the title cannot be used.
[[nodiscard]] std::variant<std::string, PackedNames> names_from_control(
    std::uint64_t title_id, std::span<const std::byte> nacp);

} // namespace armigo::pack
