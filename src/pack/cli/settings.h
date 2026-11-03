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

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>

namespace armigo::pack {

struct TitleMatch {
    std::filesystem::path folder;
};

[[nodiscard]] std::string format_build_id(const std::array<std::byte, 32>& build_id);

[[nodiscard]] bool parse_build_id(std::string_view text, std::array<std::byte, 32>& build_id);

// Reads the 32-byte module build id at file offset 0x40 of an NSO.
[[nodiscard]] std::variant<std::string, std::array<std::byte, 32>> read_nso_build_id(
    const std::filesystem::path& nso_file);

// Matches `build_id` against titles/<TitleID>/profile.json. The title root is
// ARMIGO_TITLES when set ("off" turns settings off) and otherwise titles/ next
// to `executable`.
[[nodiscard]] std::variant<std::string, TitleMatch> match_title_build(
    std::uint64_t title_id, const std::array<std::byte, 32>& build_id,
    const std::filesystem::path& executable);

} // namespace armigo::pack
