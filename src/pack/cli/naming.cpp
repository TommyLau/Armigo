// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/pack/naming.h>

#include <array>
#include <cstdio>
#include <string>
#include <string_view>

namespace armigo::pack {
namespace {

std::string title_id_text(std::uint64_t title_id) {
    char text[17];
    std::snprintf(text, sizeof text, "%016llX", static_cast<unsigned long long>(title_id));
    return text;
}

std::string failure_detail(std::string detail, const char* fallback) {
    if (detail.empty()) {
        return fallback;
    }
    return detail;
}

constexpr std::array<std::string_view, 16> supported_language_codes{
    "en-US", "en-GB", "ja", "fr", "de", "es-419", "es", "it", "nl",
    "fr-CA", "pt", "ru", "ko", "zh-Hant", "zh-Hans", "pt-BR",
};

} // namespace

std::variant<std::string, PackedNames> names_from_control(std::uint64_t title_id,
                                                          std::span<const std::byte> nacp_bytes) {
    const auto* data = reinterpret_cast<const std::uint8_t*>(nacp_bytes.data());
    const nacp::NacpParseResult parsed =
        nacp::parse_nacp(std::span<const std::uint8_t>(data, nacp_bytes.size()));
    if (!parsed.ok()) {
        return failure_detail(parsed.detail, nacp::to_string(parsed.error));
    }
    std::size_t first = 0;
    while (first < supported_language_codes.size() &&
           !(parsed.nacp.supported_language_flag & (std::uint32_t{1} << first))) ++first;
    if (first == supported_language_codes.size()) return "The title declares no supported language.";
    const auto& title = parsed.nacp.entries[first];
    if (title.name.empty()) return "The title has no name in its first supported language.";
    const nacp::AppFileNameResult file = nacp::app_file_name(title.name);
    if (!file.ok() && file.error != nacp::AppNameError::empty_result) {
        return failure_detail(file.detail, nacp::to_string(file.error));
    }

    const auto id = title_id_text(title_id);
    PackedNames names;
    names.display_name = std::string(title.name);
    if (file.name.empty()) {
        names.file_name = id;
    } else {
        auto base = file.name;
        // Keep .<stem>.app.<uint64>.partial within a 255-byte path component.
        constexpr std::size_t max_bundle_stem_bytes = 221;
        constexpr std::size_t max_base = max_bundle_stem_bytes - 19;
        if (base.size() > max_base) {
            std::size_t end = max_base;
            while ((static_cast<unsigned char>(base[end]) & 0xc0) == 0x80) --end;
            base.resize(end);
        }
        names.file_name = base + " (" + id + ")";
    }
    std::string slug;
    for (const unsigned char character : file.name) {
        if ((character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
            (character >= '0' && character <= '9')) {
            slug += static_cast<char>(character);
        } else if (!slug.empty() && slug.back() != '-') {
            slug += '-';
        }
        if (slug.size() == 64) break;
    }
    if (!slug.empty() && slug.back() == '-') slug.pop_back();
    names.executable_name = slug.empty() ? id : slug + "-" + id;
    names.bundle_identifier = "com.tommylau.armigo." + id;
    names.language = static_cast<nacp::Language>(first);
    names.supported_language_flag = parsed.nacp.supported_language_flag;
    for (std::size_t index = 0; index < supported_language_codes.size(); ++index) {
        if (parsed.nacp.supported_language_flag & (std::uint32_t{1} << index)) {
            names.languages.emplace_back(supported_language_codes[index]);
            if (!parsed.nacp.entries[index].name.empty())
                names.localized_titles.emplace_back(supported_language_codes[index], parsed.nacp.entries[index].name);
        }
    }
    names.warnings = parsed.warnings;
    return names;
}

} // namespace armigo::pack
