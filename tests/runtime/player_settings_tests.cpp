// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "player_settings.h"

#include <armigo/services/settings.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

void test_manifest_and_language_selection() {
    const auto directory = std::filesystem::temp_directory_path() / "armigo_player_language";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto manifest_path = directory / "languages.txt";
    {
        std::ofstream output(manifest_path);
        output << "0x0002fbff\nen-US\nja\nde\nfr\npt\nzh-Hans\n";
    }
    auto manifest = armigo::tools::read_language_manifest(manifest_path);
    CHECK(std::holds_alternative<armigo::tools::LanguageManifest>(manifest));
    if (const auto* parsed = std::get_if<armigo::tools::LanguageManifest>(&manifest)) {
        CHECK(parsed->supported_language_flag == 0x2fbff);
        CHECK(parsed->languages.size() == 6);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"de-DE"}}) == "de");
        CHECK(armigo::services::set::console_region_code() == 2);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"ja-JP"}}) == "ja");
        CHECK(armigo::services::set::console_region_code() == 0);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"zh-Hans-CN"}}) == "zh-Hans");
        CHECK(armigo::services::set::console_region_code() == 4);
        CHECK(armigo::services::set::desired_console_language() == 0x736e61482d687aull);
        // The app's own language comes first; a locale with no system
        // language falls to the next host language.
        const std::string_view unknown_then_french[] = {"xx-YY", "fr-FR"};
        CHECK(armigo::tools::apply_language_manifest(*parsed, unknown_then_french) == "fr");
        CHECK(armigo::services::set::console_region_code() == 2);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {}) == "en-US");
        CHECK(armigo::services::set::console_region_code() == 1);
    }
    {
        std::ofstream output(manifest_path);
        output << "invalid\nen-US\n";
    }
    CHECK(std::holds_alternative<std::string>(armigo::tools::read_language_manifest(manifest_path)));
    std::filesystem::remove_all(directory);
    armigo::services::set::configure_supported_languages(0xffff);
    CHECK(armigo::services::set::configure_console_language("en-US"));
    CHECK(armigo::services::set::configure_console_region(1));
}

} // namespace

void test_resolution_and_choices() {
    using armigo::tools::resolution_width;
    CHECK(resolution_width("auto", true, 2266, 1488) == 1920);
    CHECK(resolution_width("auto", false, 3456, 2234) == 3456);
    CHECK(resolution_width("native", true, 2266, 1488) == 2256);
    CHECK(resolution_width("native", false, 3440, 1440) == 2560);
    CHECK(resolution_width("native", false, 7680, 4320) == 3840);
    CHECK(resolution_width("native", false, 1024, 768) == 1280);
    CHECK(resolution_width("720p", false, 0, 0) == 1280);
    CHECK(resolution_width("900p", true, 0, 0) == 1600);
    CHECK(resolution_width("1440p", true, 0, 0) == 2560);
    CHECK(resolution_width("2160p", false, 0, 0) == 3840);
    CHECK(resolution_width("8k", false, 3456, 2234) == 0);

    unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
    armigo::tools::PlayerChoices choices;
    CHECK(!choices.handheld && choices.resolution == "auto" && choices.frame_rate == 60 && choices.vsync &&
          !choices.performance_overlay);
    choices.handheld = true;
    choices.frame_rate = 30;
    choices.vsync = false;
    choices.performance_overlay = true;
    armigo::tools::RunnerArguments arguments;
    armigo::tools::apply_player_choices(choices, 1920, arguments);
    CHECK(arguments.operation_mode == armigo::services::OperationMode::Handheld);
    CHECK(arguments.content_width == 1920 && !arguments.display_sync);
    unsetenv("MTL_HUD_ENABLED");
    armigo::tools::apply_performance_overlay(armigo::tools::PlayerChoices{});
    CHECK(!std::getenv("MTL_HUD_ENABLED"));
    armigo::tools::apply_performance_overlay(choices);
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "1");
    unsetenv("MTL_HUD_ENABLED");
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH")) == "1920");
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_RATE")) == "30");
    // A width the environment already chose wins.
    armigo::tools::apply_player_choices(choices, 2560, arguments);
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH")) == "1920");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
}

int main() {
    test_manifest_and_language_selection();
    test_resolution_and_choices();
    return failures == 0 ? 0 : 1;
}
