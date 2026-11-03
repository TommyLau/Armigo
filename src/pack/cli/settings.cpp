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

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string_view>
#include <system_error>
#include <vector>

namespace armigo::pack {
namespace {

std::string read_text(const std::filesystem::path& path, std::string& error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        error = "Cannot read " + path.string() + ".";
        return {};
    }
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > 1024ull * 1024ull) {
        error = path.string() + " is empty or larger than 1 MiB.";
        return {};
    }
    std::string text(static_cast<std::size_t>(size), '\0');
    stream.seekg(0);
    if (!stream.read(text.data(), static_cast<std::streamsize>(text.size()))) {
        error = "Cannot read " + path.string() + ".";
        return {};
    }
    return text;
}

std::vector<std::array<std::byte, 32>> profile_build_ids(std::string_view text) {
    std::vector<std::array<std::byte, 32>> ids;
    constexpr std::string_view kKey = "\"main_build_id\"";
    for (std::size_t at = 0; (at = text.find(kKey, at)) != std::string_view::npos;) {
        at += kKey.size();
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' ||
                                    text[at] == '\r')) {
            ++at;
        }
        if (at >= text.size() || text[at] != ':') {
            continue;
        }
        ++at;
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' ||
                                    text[at] == '\r')) {
            ++at;
        }
        if (at >= text.size() || text[at] != '"') {
            continue;
        }
        const auto value = text.substr(at + 1, 64);
        std::array<std::byte, 32> build_id{};
        if (value.size() == 64 && parse_build_id(value, build_id) &&
            at + 65 < text.size() && text[at + 65] == '"') {
            ids.push_back(build_id);
        }
    }
    return ids;
}

} // namespace

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

std::variant<std::string, TitleMatch> match_title_build(
    std::uint64_t title_id, const std::array<std::byte, 32>& build_id,
    const std::filesystem::path& executable) {
    char identity_text[17];
    std::snprintf(identity_text, sizeof identity_text, "%016llX",
                  static_cast<unsigned long long>(title_id));
    const std::string identity(identity_text);
    std::filesystem::path root;
    if (const char* override_root = std::getenv("ARMIGO_TITLES"); override_root && *override_root) {
        if (std::string_view(override_root) == "off") {
            return "Title settings are turned off.";
        }
        root = override_root;
    } else {
        root = executable.parent_path() / "titles";
    }
    const auto folder = root / identity;
    std::error_code error;
    if (!std::filesystem::is_directory(folder, error)) {
        return "No title settings for " + identity + ".";
    }
    const auto profile = folder / "profile.json";
    std::string read_error;
    const auto text = read_text(profile, read_error);
    if (!read_error.empty()) {
        return read_error;
    }
    const auto ids = profile_build_ids(text);
    for (const auto& candidate : ids) {
        if (candidate == build_id) {
            return TitleMatch{folder};
        }
    }
    return "No title settings describe build id " + format_build_id(build_id) + ".";
}

} // namespace armigo::pack
