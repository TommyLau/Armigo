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

#include <armigo/compat/title_settings.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>

namespace armigo::pack {

// A title folder and the description of the matched main-module build.
struct TitleMatch {
    std::filesystem::path folder;
    compat::TitleSettings settings;
    std::size_t index = 0;
    [[nodiscard]] const compat::PatchDescription& description() const { return settings.patches[index]; }
};

[[nodiscard]] std::string format_build_id(const std::array<std::byte, 32>& build_id);

[[nodiscard]] bool parse_build_id(std::string_view text, std::array<std::byte, 32>& build_id);

// Reads the 32-byte module build id at file offset 0x40 of an NSO.
[[nodiscard]] std::variant<std::string, std::array<std::byte, 32>> read_nso_build_id(
    const std::filesystem::path& nso_file);

// Reads titles/<TitleID>/ and matches `build_id` against its descriptions, as
// the player does. The title root is ARMIGO_TITLES when set ("off" turns
// settings off) and otherwise titles/ next to `executable`.
// Empty when a pack of `title_id` may target `platform` ("macos" or "ios"):
// some description of the title lists it, or no title settings describe the
// title (the match reports that later). Otherwise the refusal to print.
[[nodiscard]] std::string check_title_platform(std::uint64_t title_id, const std::string& platform,
                                               const std::filesystem::path& executable);

[[nodiscard]] std::variant<std::string, TitleMatch> match_title_build(
    std::uint64_t title_id, const std::array<std::byte, 32>& build_id,
    const std::filesystem::path& executable);

} // namespace armigo::pack
