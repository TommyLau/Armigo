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
#include "npdm_fixture.h"

#include <armigo/services/settings.h>
#include <armigo/compat/patches.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>
#include <unistd.h>
#include <vector>

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

        // The file lists pt, but its supported-language bit is absent.
        const std::vector<std::string> expected_languages{"en-US", "ja", "de", "fr", "zh-Hans"};
        const auto automatic = armigo::tools::game_language_choice(*parsed, std::nullopt);
        CHECK(automatic.supported_languages == expected_languages);
        CHECK(automatic.preference == "auto" && !automatic.fell_back);
        const auto japanese = armigo::tools::game_language_choice(*parsed, std::string("ja"));
        CHECK(japanese.preference == "ja" && !japanese.fell_back);
        const auto stale = armigo::tools::game_language_choice(*parsed, std::string("es"));
        CHECK(stale.preference == "auto" && stale.fell_back);
        CHECK(armigo::tools::game_language_choice(*parsed, std::string("not-a-language")).fell_back);
        auto duplicated = *parsed;
        duplicated.supported_language_flag = 0x9;
        duplicated.languages = {"en-US", "not-a-language", "ja", "fr", "en-US"};
        CHECK(armigo::tools::game_language_choice(duplicated, std::nullopt).supported_languages ==
              std::vector<std::string>({"en-US", "fr"}));

        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"ja-JP"}}, "fr") == "fr");
        CHECK(armigo::services::set::console_region_code() == 2);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"de-DE"}}, "ja") == "ja");
        CHECK(armigo::services::set::console_region_code() == 0);
        CHECK(armigo::tools::apply_language_manifest(*parsed, {{"ja-JP"}}, "es").empty());
        CHECK(armigo::tools::apply_language_manifest(*parsed, {}) == "en-US");
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
    unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    unsetenv("ARMIGO_VI_REFRESH_HZ");
    armigo::tools::PlayerChoices choices;
    CHECK(!choices.handheld && choices.resolution == "auto" && choices.frame_rate == 60 && choices.vsync &&
          !choices.performance_overlay);
    choices.handheld = true;
    choices.frame_rate = 30;
    choices.vsync = false;
    choices.performance_overlay = true;
    armigo::tools::RunnerArguments arguments;
    armigo::tools::apply_player_choices(choices, 1920, arguments);
    CHECK(!arguments.input_script_file);
    CHECK(arguments.operation_mode == armigo::services::OperationMode::Handheld);
    CHECK(arguments.content_width == 1920 && !arguments.display_sync);
    unsetenv("MTL_HUD_ENABLED");
    armigo::tools::apply_performance_overlay(armigo::tools::PlayerChoices{});
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "0");
    unsetenv("MTL_HUD_ENABLED");  // A fresh process starts with the original launch environment.
    armigo::tools::apply_performance_overlay(choices);
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "1");
    // Explicit launch overrides survive either preference and restart.
    setenv("MTL_HUD_ENABLED", "0", 1);
    armigo::tools::apply_performance_overlay(choices);
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "0");
    setenv("MTL_HUD_ENABLED", "1", 1);
    armigo::tools::apply_performance_overlay(armigo::tools::PlayerChoices{});
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "1");
    unsetenv("MTL_HUD_ENABLED");
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH")) == "1920");
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_RATE")) == "30");
    CHECK(std::string_view(std::getenv("ARMIGO_VI_REFRESH_HZ")) == "30");
    CHECK(!std::getenv("ARMIGO_COMPAT_FRAME_RATE"));
    choices.title_frame_rate = true;
    choices.display_refresh_rate = 60;
    choices.frame_rate = 0;
    unsetenv("ARMIGO_VI_REFRESH_HZ");
    armigo::tools::apply_player_choices(choices, 1920, arguments);
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_FRAME_RATE")) == "0");
    CHECK(std::string_view(std::getenv("ARMIGO_VI_REFRESH_HZ")) == "60");
    choices.frame_rate = 60;
    armigo::tools::apply_player_choices(choices, 1920, arguments);
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_FRAME_RATE")) == "0");
    unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    armigo::tools::apply_player_choices(choices, 1920, arguments);
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_FRAME_RATE")) == "60");
    unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    unsetenv("ARMIGO_VI_REFRESH_HZ");
    // A width the environment already chose wins.
    armigo::tools::apply_player_choices(choices, 2560, arguments);
    CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_RESOLUTION_WIDTH")) == "1920");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
}

void test_mac_frame_rates() {
    using namespace armigo::tools;
    // All valid declared sets: each has a 30/60 default, optionally 120.
    struct Policy {
        std::vector<std::uint32_t> rates;
        std::vector<int> low_options;
        std::vector<int> high_options;
    };
    const Policy policies[] = {
        {{30}, {0, 30}, {0, 30}},
        {{60}, {0, 60}, {0, 60}},
        {{30, 60}, {0, 30, 60}, {0, 30, 60}},
        {{30, 120}, {0, 30}, {0, 30, 120}},
        {{60, 120}, {0, 60}, {0, 60, 120}},
        {{30, 60, 120}, {0, 30, 60}, {0, 30, 60, 120}},
        {{120, 30, 60}, {0, 30, 60}, {0, 30, 60, 120}},
    };
    for (const auto& policy : policies) {
        for (double hz : {0.0, 30.0, 59.94, 60.0, 90.0, 119.88, 120.0, 144.0, 240.0,
                          std::numeric_limits<double>::quiet_NaN()}) {
            armigo::platform::DisplayCapabilities display;
            display.current = {1920, 1080, hz, false};
            // A faster available mode must not admit 120 on an active 60 Hz mode.
            display.max_refresh_hz = 240;
            const auto& expected = hz >= 120 ? policy.high_options : policy.low_options;
            for (int requested : {0, 30, 60, 120}) {
                PlayerChoices choices;
                choices.title_frame_rate = true;
                choices.supported_frame_rates = policy.rates;
                choices.frame_rate = requested;
                const auto result = select_mac_frame_rate(choices, display);
                CHECK(result.options == expected);
                const bool admitted = std::find(expected.begin(), expected.end(), requested) != expected.end();
                CHECK(result.frame_rate == (admitted ? requested : expected.back()));
                CHECK(result.fell_back == !admitted);
                CHECK(result.refresh_hz == (result.frame_rate == 120 ? 120.0 : 60.0));
            }
        }
    }
    PlayerChoices legacy;
    armigo::platform::DisplayCapabilities fast;
    fast.current = {1920, 1080, 120, true};
    CHECK(select_mac_frame_rate(legacy, fast).options == std::vector<int>({0, 30, 60}));
    legacy.supported_frame_rates = {30, 60, 120};
    legacy.frame_rate = 120;
    CHECK(select_mac_frame_rate(legacy, {}).frame_rate == 60);
    legacy.supported_frame_rates.push_back(armigo::compat::variable_frame_rate);
    legacy.frame_rate = 0;
    const auto hidden = select_mac_frame_rate(legacy, fast);
    CHECK(hidden.options == std::vector<int>({0, 30, 60, 120}));
    CHECK(hidden.frame_rate == 0 && !hidden.fell_back);
    legacy.frame_rate = -1;
    const auto rejected = select_mac_frame_rate(legacy, fast);
    CHECK(rejected.frame_rate == 120 && rejected.fell_back);

}

namespace {

const char* const frame_rate_environment[] = {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
    "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"};

std::string environment_text(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "unset";
}

void write_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary).write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

} // namespace

void test_title_frame_rate_policy() {
    namespace fs = std::filesystem;
    using namespace armigo::tools;
    const auto root = fs::temp_directory_path() / ("armigo-player-policy-" + std::to_string(::getpid()));
    fs::remove_all(root);
    const auto exefs = root / "exefs";
    const auto npdm = npdm_fixture::profile_image_a().bytes;
    write_file(exefs / "main.npdm", npdm.data(), npdm.size());
    std::vector<unsigned char> header(0x100);
    std::memcpy(header.data(), "NSO0", 4);
    for (unsigned index = 0; index < 32; ++index) header[0x40 + index] = static_cast<unsigned char>(index + 1);
    write_file(exefs / "main", header.data(), header.size());
    const std::string profile = R"({"version":1,"patches":[{"title_id":"0x0200000000000A42",)"
        R"("main_build_id":"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",)"
        R"("frame_rate":{"default":30,"display_refresh_rate":60,"experimental_rates":[120]},)"
        R"("frame_rates":[30,60,120,"variable"]}]})";
    write_file(root / "titles" / "0200000000000A42" / "profile.json", profile.data(), profile.size());
    setenv("ARMIGO_TITLES", (root / "titles").c_str(), 1);

    RunnerArguments arguments;
    arguments.exefs_directory = exefs;
    const auto policy = find_title_frame_rate_policy(arguments);
    CHECK(policy.has_value());
    if (policy) {
        CHECK(policy->title_id == npdm_fixture::kProgramId);
        CHECK(policy->default_rate == 30 && policy->display_refresh_rate == 60);
        CHECK(policy->supported_rates ==
              std::vector<std::uint32_t>({30, 60, 120, armigo::compat::variable_frame_rate}));
        CHECK(policy->experimental_rates == std::vector<std::uint32_t>({120}));
        CHECK(frame_rate_preference_key(policy->title_id) == "frame_rate.0200000000000A42");
        CHECK(language_preference_key(policy->title_id) == "game_language.0200000000000A42");
        const auto automatic = title_frame_rate_choices(*policy, std::nullopt);
        CHECK(automatic.title_frame_rate && automatic.frame_rate == 0 && automatic.default_frame_rate == 30);
        CHECK(automatic.display_refresh_rate == 60 && automatic.experimental_frame_rates == policy->experimental_rates);
        CHECK(title_frame_rate_choices(*policy, 120).frame_rate == 120);
        CHECK(title_frame_rate_choices(*policy, 45).frame_rate == 0);
    }
    CHECK(frame_rate_preference_key(std::nullopt) == "frame_rate");
    CHECK(language_preference_key(std::nullopt) == "game_language");
    // Other inputs, a build the settings do not describe, and settings off.
    auto container = arguments;
    container.container_file = root / "game.nsp";
    CHECK(!find_title_frame_rate_policy(container));
    header[0x40] = 0xff;
    write_file(exefs / "main", header.data(), header.size());
    CHECK(!find_title_frame_rate_policy(arguments));
    header[0x40] = 1;
    write_file(exefs / "main", header.data(), header.size());
    CHECK(find_title_frame_rate_policy(arguments).has_value());
    setenv("ARMIGO_TITLES", "off", 1);
    CHECK(!find_title_frame_rate_policy(arguments));
    unsetenv("ARMIGO_TITLES");
    fs::remove_all(root);
}

void test_command_line_frame_rate() {
    using namespace armigo::tools;
    TitleFrameRatePolicy policy{0x0200000000000A42, 30, 60, {30, 60, 120}, {}};
    armigo::platform::DisplayCapabilities fast;
    fast.current = {1920, 1080, 120, false};
    armigo::platform::DisplayCapabilities slow;
    slow.current = {1920, 1080, 60, false};
    struct Case {
        std::optional<int> stored;
        const char* override;
        bool high_refresh;
        double explicit_clock;  // zero: the default clock
        int selected;
        double clock;
        const char* frame_rate;
        const char* vi;
        const char* resolution_rate;
    };
    for (const auto& test : {
        // Auto changes nothing: the runtime selects the title default.
        Case{std::nullopt, nullptr, true, 0, 0, 60, "unset", "unset", "unset"},
        Case{0, nullptr, true, 0, 0, 60, "unset", "unset", "unset"},
        Case{30, nullptr, true, 0, 30, 60, "30", "60", "30"},
        Case{60, nullptr, false, 0, 60, 60, "60", "60", "60"},
        Case{120, nullptr, true, 0, 120, 120, "120", "120", "120"},
        // A 60 Hz display falls back, and so does a slower explicit clock.
        Case{120, nullptr, false, 0, 60, 60, "60", "60", "60"},
        Case{120, nullptr, true, 60, 60, 60, "60", "60", "60"},
        Case{120, nullptr, true, 120, 120, 120, "120", "120", "120"},
        // The environment wins; only a 120 fps mode gains its clock.
        Case{60, "120", true, 0, 120, 120, "120", "120", "unset"},
        Case{120, "30", true, 0, 30, 60, "30", "unset", "unset"},
        Case{120, "variable", true, 0, 0, 60, "variable", "unset", "unset"},
        Case{30, "120", true, 60, 120, 60, "120", "unset", "unset"},
    }) {
        for (const char* name : frame_rate_environment) unsetenv(name);
        if (test.override) setenv("ARMIGO_COMPAT_FRAME_RATE", test.override, 1);
        RunnerArguments launch;
        launch.content_width = 1600;
        launch.display_sync = false;
        launch.operation_mode = armigo::services::OperationMode::Handheld;
        if (test.explicit_clock) {
            launch.refresh_hz = test.explicit_clock;
            launch.refresh_hz_explicit = true;
        }
        MacFrameRateSelection observed;
        int observed_rate = -1;
        auto configure = mac_title_frame_rate_configuration(title_frame_rate_choices(policy, test.stored),
            [&](const PlayerChoices& effective, const MacFrameRateSelection& selection) {
                observed = selection;
                observed_rate = effective.frame_rate;
            });
        configure(launch, test.high_refresh ? fast : slow);
        CHECK(observed_rate == test.selected);
        CHECK(observed.default_rate == 30);
        CHECK(observed.options == (test.high_refresh ? std::vector<int>({0, 30, 60, 120})
                                                     : std::vector<int>({0, 30, 60})));
        CHECK(launch.refresh_hz == test.clock && observed.refresh_hz == test.clock);
        CHECK(environment_text("ARMIGO_COMPAT_FRAME_RATE") == test.frame_rate);
        CHECK(environment_text("ARMIGO_VI_REFRESH_HZ") == test.vi);
        CHECK(environment_text("ARMIGO_COMPAT_RESOLUTION_RATE") == test.resolution_rate);
        // The command line keeps everything else.
        CHECK(environment_text("ARMIGO_COMPAT_RESOLUTION_WIDTH") == "unset");
        CHECK(launch.content_width == 1600 && !launch.display_sync);
        CHECK(launch.operation_mode == armigo::services::OperationMode::Handheld);
        CHECK(observed.environment_override.has_value() == (test.override != nullptr));
    }
    for (const char* name : frame_rate_environment) unsetenv(name);
}

void test_restart_command() {
    using armigo::tools::restart_command;
    const std::vector<std::string> environment{"HOME=/home", "ARMIGO_COMPAT_FRAME_RATE=120",
        "ARMIGO_VI_REFRESH_HZ=120", "ARMIGO_COMPAT_RESOLUTION_RATE=120", "ARMIGO_AUDIO=0", "MTL_HUD_ENABLED=1"};
    const std::vector<std::string> run{"Armigo", "run", "--refresh-hz", "120", "--exefs", "exefs", "--romfs",
        "romfs", "--width", "1600"};
    auto command = restart_command(run, environment, false);
    CHECK(command.arguments ==
          std::vector<std::string>({"Armigo", "run", "--exefs", "exefs", "--romfs", "romfs", "--width", "1600"}));
    CHECK(command.environment == std::vector<std::string>({"HOME=/home", "ARMIGO_COMPAT_RESOLUTION_RATE=120",
                                                            "ARMIGO_AUDIO=0", "MTL_HUD_ENABLED=1"}));
    const auto original_without_hud = restart_command(run, {environment.data(), environment.size() - 1}, false);
    CHECK(std::none_of(original_without_hud.environment.begin(), original_without_hud.environment.end(),
        [](const auto& entry) { return entry.starts_with("MTL_HUD_ENABLED="); }));
    // Armigo's injected On value is absent from the original launch snapshot.
    // Applying the saved Off choice in a fresh process must explicitly disable it.
    armigo::tools::PlayerChoices overlay_on;
    overlay_on.performance_overlay = true;
    unsetenv("MTL_HUD_ENABLED");
    armigo::tools::apply_performance_overlay(overlay_on);
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "1");
    CHECK(restart_command(run, {}, false).environment.empty());
    unsetenv("MTL_HUD_ENABLED");
    armigo::tools::apply_performance_overlay(armigo::tools::PlayerChoices{});
    CHECK(std::string_view(std::getenv("MTL_HUD_ENABLED")) == "0");
    unsetenv("MTL_HUD_ENABLED");
    for (const char* external : {"0", "1"}) {
        const std::vector<std::string> explicit_hud{std::string("MTL_HUD_ENABLED=") + external};
        CHECK(restart_command(run, explicit_hud, false).environment == explicit_hud);
    }
    command = restart_command(run, environment, true);
    CHECK(command.arguments.back() == "--fullscreen" && command.arguments.size() == 9);
    // --fullscreen follows the window: kept once in full screen, dropped when
    // the game left full screen.
    const std::vector<std::string> fullscreen{"Armigo", "run", "--fullscreen", "--saves", "saves"};
    CHECK(restart_command(fullscreen, {}, true).arguments ==
          std::vector<std::string>({"Armigo", "run", "--saves", "saves", "--fullscreen"}));
    CHECK(restart_command(fullscreen, {}, false).arguments ==
          std::vector<std::string>({"Armigo", "run", "--saves", "saves"}));
    // Values that read like options stay with their options.
    const std::vector<std::string> values{"Armigo", "run", "--report", "--refresh-hz", "--saves", "--fullscreen",
                                          "--refresh-hz"};
    CHECK(restart_command(values, {}, false).arguments ==
          std::vector<std::string>({"Armigo", "run", "--report", "--refresh-hz", "--saves", "--fullscreen"}));
    CHECK(restart_command(values, {}, true).arguments ==
          std::vector<std::string>({"Armigo", "run", "--report", "--refresh-hz", "--saves", "--fullscreen",
                                    "--fullscreen"}));
    // A packed app opened from the Finder.
    const std::vector<std::string> packed{"/Applications/Game (0100000000000001).app/Contents/MacOS/Game-0100000000000001"};
    CHECK(restart_command(packed, {}, false).arguments == packed);
    CHECK(restart_command(packed, {}, true).arguments ==
          std::vector<std::string>({packed[0], "run", "--fullscreen"}));
}

int main() {
    test_manifest_and_language_selection();
    test_resolution_and_choices();
    test_mac_frame_rates();
    test_title_frame_rate_policy();
    test_command_line_frame_rate();
    test_restart_command();
    return failures == 0 ? 0 : 1;
}
