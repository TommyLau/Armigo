// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_settings.h"

#include <armigo/services/settings.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>

namespace armigo::tools {

std::variant<LanguageManifest, std::string> read_language_manifest(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return "Cannot read the packed language manifest.";
    std::string mask_line;
    std::getline(input, mask_line);
    if (!mask_line.empty() && mask_line.back() == '\r') mask_line.pop_back();
    LanguageManifest manifest;
    if (mask_line.size() != 10 || !mask_line.starts_with("0x")) {
        return "The packed language manifest has an invalid supported-language mask.";
    }
    const auto parsed = std::from_chars(mask_line.data() + 2, mask_line.data() + mask_line.size(),
                                       manifest.supported_language_flag, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != mask_line.data() + mask_line.size()) {
        return "The packed language manifest has an invalid supported-language mask.";
    }
    std::string language;
    while (std::getline(input, language)) {
        if (!language.empty() && language.back() == '\r') language.pop_back();
        if (!language.empty()) manifest.languages.push_back(language);
    }
    return manifest;
}

std::string apply_language_manifest(const LanguageManifest& manifest,
    std::span<const std::string_view> preferred_locales) {
    const auto language = services::set::automatic_language(preferred_locales, manifest.supported_language_flag);
    if (!language) return {};
    if (!services::set::configure_console_language(*language) ||
        !services::set::configure_console_region(services::set::market_region(*language))) return {};
    services::set::configure_supported_languages(manifest.supported_language_flag);
    return *language;
}

std::uint32_t resolution_width(std::string_view choice, bool ios,
    std::uint32_t display_width, std::uint32_t display_height) {
    std::uint32_t width = 0;
    if (choice == "auto") choice = ios ? "1080p" : "native";
    if (choice == "720p") width = 1280;
    else if (choice == "900p") width = 1600;
    else if (choice == "1080p") width = 1920;
    else if (choice == "1440p") width = 2560;
    else if (choice == "2160p") width = 3840;
    else if (choice == "native")
        width = static_cast<std::uint32_t>(std::min<std::uint64_t>(display_width, std::uint64_t(display_height) * 16 / 9));
    else return 0;
    return std::clamp<std::uint32_t>(width, 1280, 3840) & ~15u;
}

void apply_performance_overlay(const PlayerChoices& choices) {
    if (choices.performance_overlay) setenv("MTL_HUD_ENABLED", "1", 1);
}

void apply_player_choices(const PlayerChoices& choices, std::uint32_t width, RunnerArguments& arguments) {
    arguments.operation_mode = choices.handheld ? services::OperationMode::Handheld : services::OperationMode::Console;
    if (width && !std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH"))
        setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", std::to_string(width).c_str(), 1);
    if (choices.frame_rate && !std::getenv("ARMIGO_COMPAT_RESOLUTION_RATE"))
        setenv("ARMIGO_COMPAT_RESOLUTION_RATE", std::to_string(choices.frame_rate).c_str(), 1);
    arguments.content_width = width;
    arguments.display_sync = choices.vsync;
}

} // namespace armigo::tools
