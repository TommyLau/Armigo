// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "settings.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string_view>
#include <system_error>
#include <vector>

namespace armigo::pack {

std::string format_build_id(const std::array<std::byte, 32>& build_id) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string text(64, '0');
    for (std::size_t index = 0; index < build_id.size(); ++index) {
        const auto value = std::to_integer<unsigned>(build_id[index]);
        text[index * 2] = kDigits[value >> 4];
        text[index * 2 + 1] = kDigits[value & 0x0f];
    }
    return text;
}

bool parse_build_id(std::string_view text, std::array<std::byte, 32>& build_id) {
    if (text.size() != 64) {
        return false;
    }
    for (std::size_t index = 0; index < build_id.size(); ++index) {
        unsigned value = 0;
        const auto* begin = text.data() + index * 2;
        const auto parsed = std::from_chars(begin, begin + 2, value, 16);
        if (parsed.ec != std::errc{} || parsed.ptr != begin + 2) {
            return false;
        }
        build_id[index] = static_cast<std::byte>(value);
    }
    return true;
}

std::variant<std::string, std::array<std::byte, 32>> read_nso_build_id(
    const std::filesystem::path& nso_file) {
    std::ifstream stream(nso_file, std::ios::binary);
    if (!stream) {
        return "Cannot read the main module " + nso_file.string() + ".";
    }
    std::array<char, 0x60> header{};
    if (!stream.read(header.data(), static_cast<std::streamsize>(header.size()))) {
        return "The main module " + nso_file.string() + " is shorter than an NSO header.";
    }
    if (std::string_view(header.data(), 4) != "NSO0") {
        return "The main module " + nso_file.string() + " is not an NSO.";
    }
    std::array<std::byte, 32> build_id{};
    for (std::size_t index = 0; index < build_id.size(); ++index) {
        build_id[index] =
            static_cast<std::byte>(static_cast<unsigned char>(header[0x40 + index]));
    }
    return build_id;
}

namespace {

// titles/<TitleID>/ under the title root, or why there is none.
std::variant<std::string, std::filesystem::path> title_folder(std::uint64_t title_id,
                                                             const std::filesystem::path& executable) {
    char identity_text[17];
    std::snprintf(identity_text, sizeof identity_text, "%016llX",
                  static_cast<unsigned long long>(title_id));
    const std::string identity(identity_text);
    std::filesystem::path root;
    if (const char* override_root = std::getenv("ARMIGO_TITLES"); override_root && *override_root) {
        if (std::string_view(override_root) == "off") {
            return std::string("Title settings are turned off.");
        }
        root = override_root;
    } else {
        root = executable.parent_path() / "titles";
    }
    const auto folder = root / identity;
    std::error_code error;
    if (!std::filesystem::is_directory(folder, error)) {
        return std::string("No title settings for " + identity + ".");
    }
    return folder;
}

} // namespace

std::string check_title_platform(std::uint64_t title_id, const std::string& platform,
                                 const std::filesystem::path& executable) {
    const auto found = title_folder(title_id, executable);
    if (std::holds_alternative<std::string>(found)) return {};
    const auto& folder = std::get<std::filesystem::path>(found);
    auto read = compat::read_title_settings(folder, title_id);
    if (std::holds_alternative<std::string>(read)) return {};
    std::string listed;
    for (const auto& patch : std::get<compat::TitleSettings>(read).patches) {
        if (std::find(patch.platforms.begin(), patch.platforms.end(), platform) != patch.platforms.end()) return {};
        for (const auto& name : patch.platforms)
            if (listed.find(name) == std::string::npos) listed += (listed.empty() ? "" : ", ") + name;
    }
    return "Title " + folder.filename().string() + " does not support packing for " + platform +
        "; its title settings (" + (folder / "profile.json").string() + ") list " + listed + ".";
}

std::variant<std::string, TitleMatch> match_title_build(
    std::uint64_t title_id, const std::array<std::byte, 32>& build_id,
    const std::filesystem::path& executable) {
    const auto found = title_folder(title_id, executable);
    if (const auto* failure = std::get_if<std::string>(&found)) return *failure;
    const auto& folder = std::get<std::filesystem::path>(found);
    auto read = compat::read_title_settings(folder, title_id);
    if (auto* failure = std::get_if<std::string>(&read)) {
        return std::move(*failure);
    }
    TitleMatch match{folder, std::get<compat::TitleSettings>(std::move(read))};
    if (const auto* description = compat::match_patch(match.settings.patches, {title_id, build_id})) {
        match.index = static_cast<std::size_t>(description - match.settings.patches.data());
        return match;
    }
    return "No title settings describe build id " + format_build_id(build_id) + ".";
}

} // namespace armigo::pack
