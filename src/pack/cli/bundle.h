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
#include <filesystem>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

struct BundleSpec {
    std::filesystem::path app_path;
    std::string display_name;
    std::string bundle_name;
    std::string bundle_identifier;
    std::string title_id;
    std::string build_id;
    std::uint32_t title_version = 0;
    std::string platform;
    std::string commit;
    std::filesystem::path player;
    std::filesystem::path title_folder;
    std::span<const std::byte> icon;
    std::vector<std::string> languages;
    // The title in each language that has one: (Horizon language code, title).
    std::vector<std::pair<std::string, std::string>> localized_titles;
    std::uint32_t supported_language_flag = 0;
    // Zero keeps the player's ordinary display-rate setting. These describe
    // a macOS title mode whose simulation rate is separate from display cadence.
    std::uint32_t default_frame_rate = 0;
    std::uint32_t display_refresh_rate = 0;
    std::vector<std::uint32_t> supported_frame_rates;
    std::vector<std::uint32_t> experimental_frame_rates;
};

[[nodiscard]] std::string info_plist(const BundleSpec& spec);
[[nodiscard]] std::string settings_bundle(const BundleSpec& spec);
// The InfoPlist.strings of each localization the app lists: its display name
// and bundle name in that language, the title's own name there when the
// control data has one and the app's display name otherwise. Pairs of
// (lproj directory name, file contents).
[[nodiscard]] std::vector<std::pair<std::string, std::string>> localized_names(const BundleSpec& spec);
[[nodiscard]] std::string language_manifest(const BundleSpec& spec);
[[nodiscard]] std::string pack_json(const BundleSpec& spec);

// Writes the unsigned bundle and copies the linked player to its platform path.
// iOS capability entitlements are written alongside the app; no signing occurs.
[[nodiscard]] std::variant<std::string, std::filesystem::path> assemble_application(
    const BundleSpec& spec);

} // namespace armigo::pack
