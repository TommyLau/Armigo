// ============================================================================
// Armigo Compatibility - Title Settings
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/compat/patches.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>

namespace armigo::compat {

// A title id as Horizon prints it and as titles/ names its folders: sixteen
// uppercase hexadecimal digits.
[[nodiscard]] std::string title_id_text(std::uint64_t title_id);
// Parses that form back; nullopt for anything else.
[[nodiscard]] std::optional<std::uint64_t> parse_title_id_text(const std::string& text);

// A supported title's settings: its folder (profile.json and the pipeline
// lists the profile names) and the descriptions the profile holds, one per
// main-module build.
struct TitleSettings {
    std::filesystem::path folder;
    PatchDescriptions patches;
};

// Reads and checks one title folder: profile.json parses, every description
// names the folder's title, and every pipeline list it names exists in the
// folder. `title_id` is the title the folder is expected to describe.
[[nodiscard]] std::variant<std::string, TitleSettings> read_title_settings(
    const std::filesystem::path& folder, std::uint64_t title_id);

// The Armigo resources of a packed app: Armigo/ in the
// app's resources, or empty when this executable is not in a packed app.
[[nodiscard]] std::filesystem::path packed_resources();

// Finds the settings of `title_id`, in order: the packed app's own title
// folder (Armigo/title) when it describes this title; then <root>/<TitleID>/
// under the title root, which is ARMIGO_TITLES when set ("off" turns title
// settings off) and otherwise the titles/ folder installed next to the
// executable or in its app's resources. nullopt when no folder describes the
// title; a folder that does not read is an error.
using TitleSettingsResult = std::variant<std::string, std::optional<TitleSettings>>;
[[nodiscard]] TitleSettingsResult find_title_settings(std::uint64_t title_id);

// Where a title keeps its saves and its caches: <Documents>/Armigo/Saves/<TitleID>
// and <Caches>/Armigo/<TitleID>, in the user's (or, on iOS, the app
// container's) Documents and Caches directories.
[[nodiscard]] std::filesystem::path title_save_directory(std::uint64_t title_id);
[[nodiscard]] std::filesystem::path title_cache_directory(std::uint64_t title_id);

// Checks only the namespace roots. Never enumerates legacy mixed-content data.
// Existing legacy data requires an explicit import before a new save root is created.
[[nodiscard]] std::optional<std::string> prepare_save_directory(
    const std::filesystem::path& saves, const std::filesystem::path& legacy);

} // namespace armigo::compat
