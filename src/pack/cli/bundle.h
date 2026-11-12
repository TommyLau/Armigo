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
    // The published app stem, independent of its temporary assembly path.
    std::string file_name;
    std::string executable_name;
    std::string bundle_identifier;
    std::string title_id;
    std::string build_id;
    std::uint32_t title_version = 0;
    std::string platform;
    std::string commit;
    bool development = false;
    bool development_metal_hud = false;
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

// The folder an app is assembled in, beside its final path: hidden and without
// a bundle extension, so LaunchServices and the Finder never record a partial
// app. `unique` keeps concurrent packs apart.
[[nodiscard]] std::filesystem::path assembly_path(const std::filesystem::path& app_path, std::uint64_t unique);
// Moves a complete assembly to its final path in one rename on the same volume.
// It fails, leaving both paths unchanged, when the final path already exists.
[[nodiscard]] std::string publish_application(const std::filesystem::path& assembled,
                                              const std::filesystem::path& app_path);
// Reads the raw CFBundleExecutable from a macOS app's Info.plist. Missing or
// invalid metadata is an error; no executable basename is assumed.
[[nodiscard]] std::variant<std::string, std::filesystem::path> packed_application_executable(
    const std::filesystem::path& app_path);
// Registers a published macOS app with LaunchServices, so the Finder shows the
// complete app instead of any state it recorded before.
[[nodiscard]] std::string register_application(const std::filesystem::path& app_path);

} // namespace armigo::pack
