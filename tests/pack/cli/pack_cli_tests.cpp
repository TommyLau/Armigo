// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "bundle.h"
#include "command.h"
#include "firmware.h"
#include "icon.h"
#include "ios.h"
#include "sample_jpeg.h"
#include "settings.h"
#include "space.h"

#include <armigo/pack/naming.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

namespace pack = armigo::pack;
namespace fs = std::filesystem;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

std::variant<pack::CommandFailure, pack::Command> parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
        argv.push_back(arg.data());
    }
    return pack::parse_command(static_cast<int>(argv.size()), argv.data());
}

const pack::Command* command_of(const std::variant<pack::CommandFailure, pack::Command>& parsed) {
    return std::get_if<pack::Command>(&parsed);
}

const pack::CommandFailure* failure_of(const std::variant<pack::CommandFailure, pack::Command>& parsed) {
    return std::get_if<pack::CommandFailure>(&parsed);
}

std::vector<std::byte> sample_jpeg() {
    std::vector<std::byte> bytes(std::size(kSampleJpeg));
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>(kSampleJpeg[index]);
    }
    return bytes;
}

void test_arguments() {
    const auto help = parse({"armigo", "--help"});
    CHECK(command_of(help) && command_of(help)->help);
    const auto missing = parse({"armigo", "game.xci"});
    CHECK(failure_of(missing) && failure_of(missing)->exit_code == 2);
    const auto unknown = parse({"armigo", "game.xci", "--keys", "keys.txt", "--platform", "windows"});
    CHECK(failure_of(unknown) && failure_of(unknown)->exit_code == 2);
    const auto ios = parse({"armigo", "game.xci", "--keys", "keys.txt", "--platform", "ios",
                            "--reuse", "Packed.app", "--ios-build", "ios-build", "--keep-app",
                            "--firmware", "Firmware.zip"});
    CHECK(command_of(ios) && command_of(ios)->platform == "ios");
    if (const pack::Command* command = command_of(ios)) {
        CHECK(command->reuse == "Packed.app");
        CHECK(command->ios_build == "ios-build");
        CHECK(command->keep_app);
        CHECK(command->firmware == "Firmware.zip");
    }
    const auto macos = parse({"armigo", "game.xci", "--update", "update.nsp", "--dlc=dlc.nsp", "--keys=keys.txt",
                              "--output", "out", "--firmware", "firmware"});
    CHECK(command_of(macos));
    if (const pack::Command* command = command_of(macos)) {
        CHECK(command->game == "game.xci");
        CHECK(command->updates.size() == 1 && command->updates[0] == "update.nsp");
        CHECK(command->dlc.size() == 1 && command->dlc[0] == "dlc.nsp");
        CHECK(command->keys == "keys.txt");
        CHECK(command->output == "out");
        CHECK(command->platform == "macos");
        CHECK(!command->keep_app);
    }
    {
        const auto no_ipa = parse({"armigo", "game.xci", "--keys", "keys.txt", "--firmware", "fw", "--platform", "ios",
                                   "--no-ipa"});
        const auto* command = std::get_if<pack::Command>(&no_ipa);
        CHECK(command && command->no_ipa && !command->keep_app);
        CHECK(std::holds_alternative<pack::CommandFailure>(
            parse({"armigo", "game.xci", "--keys", "keys.txt", "--firmware", "fw", "--no-ipa"})));
    }
    const auto wrong_platform = parse({"armigo", "game.xci", "--keys", "keys.txt", "--keep-app"});
    CHECK(failure_of(wrong_platform) && failure_of(wrong_platform)->exit_code == 2);
    const auto duplicate = parse({"armigo", "game.xci", "--keys", "a", "--keys", "b"});
    CHECK(failure_of(duplicate) && failure_of(duplicate)->exit_code == 2);
    const auto two_games = parse({"armigo", "a.xci", "b.xci", "--keys", "keys.txt"});
    CHECK(failure_of(two_games) && failure_of(two_games)->exit_code == 2);
}

void test_space() {
    CHECK(pack::space_is_enough(5, 5));
    CHECK(!pack::space_is_enough(4, 5));
    const auto overflow = pack::expected_pack_bytes(UINT64_MAX - 10);
    CHECK(std::holds_alternative<std::string>(overflow));
    const auto expected = pack::expected_pack_bytes(100);
    CHECK(std::holds_alternative<std::uint64_t>(expected));
    if (const auto* bytes = std::get_if<std::uint64_t>(&expected)) {
        CHECK(*bytes == 100 + pack::kPackMarginBytes);
    }
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_space";
    fs::remove_all(directory);
    fs::create_directories(directory);
    const fs::path file = directory / "content.bin";
    {
        std::ofstream stream(file, std::ios::binary);
        stream << std::string(100, 'x');
    }
    const auto once = pack::sum_content_bytes({{file, 0}, {file, 0}});
    CHECK(std::holds_alternative<std::uint64_t>(once));
    if (const auto* bytes = std::get_if<std::uint64_t>(&once)) {
        CHECK(*bytes == 100);
    }
    const auto known = pack::sum_content_bytes({{file, 40}, {file, 40}});
    CHECK(std::holds_alternative<std::uint64_t>(known) && std::get<std::uint64_t>(known) == 80);
    const pack::SpaceCheck volume = pack::check_free_space(directory, 1);
    CHECK(volume.error.empty());
    CHECK(volume.available > 0);
    CHECK(volume.enough);
    fs::remove_all(directory);
}

void test_names() {
    const std::string_view title = "Sample Chronicle: Open Country";
    std::vector<std::byte> nacp(armigo::pack::nacp::control_file_size, std::byte{0});
    std::memcpy(nacp.data(), title.data(), title.size());
    std::memcpy(nacp.data() + 0x600, "Japanese title", 14);
    std::memcpy(nacp.data() + 0x900, "French title", 12);
    nacp[0x302c] = std::byte{0x0d};
    const auto named = pack::names_from_control(0x01007EF00011E000ull, nacp);
    CHECK(std::holds_alternative<pack::PackedNames>(named));
    if (const auto* name = std::get_if<pack::PackedNames>(&named)) {
        CHECK(name->display_name == title);
        CHECK(name->file_name == "Sample Chronicle - Open Country");
        CHECK(name->bundle_identifier == "com.tommylau.armigo.01007EF00011E000");
        CHECK(name->language == armigo::pack::nacp::Language::american_english);
        CHECK(name->supported_language_flag == 0x0d);
        CHECK(name->languages.size() == 3);
        CHECK(name->languages[0] == "en-US");
        CHECK(name->languages[1] == "ja");
        CHECK(name->languages[2] == "fr");
    }
    const auto missing = pack::names_from_control(0x01007EF00011E000ull, {});
    CHECK(std::holds_alternative<std::string>(missing));
    if (const auto* error = std::get_if<std::string>(&missing)) {
        CHECK(error->find("control data") != std::string::npos);
    }
}

void test_build_id() {
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_build";
    fs::remove_all(directory);
    const fs::path titles = directory / "titles";
    const fs::path folder = titles / "0100000000000001";
    fs::create_directories(folder);
    const std::string present(64, 'a');
    const std::string absent(64, 'b');
    {
        std::ofstream profile(folder / "profile.json");
        profile << "{\"patches\":[{\"title_id\":\"0x0100000000000001\",\"main_build_id\":\"" << present
                << "\"}]}";
    }
    std::array<std::byte, 32> build_id{};
    CHECK(pack::parse_build_id(present, build_id));
    CHECK(pack::format_build_id(build_id) == present);
    setenv("ARMIGO_TITLES", titles.c_str(), 1);
    const auto matched = pack::match_title_build(0x0100000000000001ull, build_id, directory / "armigo");
    CHECK(std::holds_alternative<pack::TitleMatch>(matched));
    std::array<std::byte, 32> other{};
    CHECK(pack::parse_build_id(absent, other));
    const auto missing = pack::match_title_build(0x0100000000000001ull, other, directory / "armigo");
    CHECK(std::holds_alternative<std::string>(missing));
    if (const auto* text = std::get_if<std::string>(&missing)) {
        CHECK(text->find(absent) != std::string::npos);
        CHECK(text->find("No title settings describe build id") != std::string::npos);
    }
    setenv("ARMIGO_TITLES", "off", 1);
    const auto disabled = pack::match_title_build(0x0100000000000001ull, build_id, directory / "armigo");
    CHECK(std::holds_alternative<std::string>(disabled));
    unsetenv("ARMIGO_TITLES");
    const fs::path executable = directory / "bin" / "armigo";
    fs::create_directories(executable.parent_path());
    const auto no_folder = pack::match_title_build(0x0100000000000001ull, build_id, executable);
    CHECK(std::holds_alternative<std::string>(no_folder));

    std::array<char, 0x60> header{};
    header.fill(0);
    std::memcpy(header.data(), "NSO0", 4);
    for (std::size_t index = 0; index < 32; ++index) {
        header[0x40 + index] = static_cast<char>(index + 1);
    }
    const fs::path nso = directory / "main";
    {
        std::ofstream stream(nso, std::ios::binary);
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    const auto read = pack::read_nso_build_id(nso);
    CHECK((std::holds_alternative<std::array<std::byte, 32>>(read)));
    if (const auto* id = std::get_if<std::array<std::byte, 32>>(&read)) {
        CHECK(std::to_integer<unsigned>((*id)[0]) == 1);
        CHECK(std::to_integer<unsigned>((*id)[31]) == 32);
    }
    const auto short_file = directory / "short";
    {
        std::ofstream stream(short_file, std::ios::binary);
        stream << "NSO0";
    }
    CHECK(std::holds_alternative<std::string>(pack::read_nso_build_id(short_file)));
    fs::remove_all(directory);
}

void test_bundle_and_icon() {
    pack::BundleSpec spec;
    spec.display_name = "A&B<C>";
    spec.bundle_name = "A:B/C";
    spec.bundle_identifier = "com.tommylau.armigo.01007EF00011E000";
    spec.title_id = "01007EF00011E000";
    spec.build_id = std::string(64, 'c');
    spec.title_version = 42;
    spec.platform = "macos";
    spec.commit = "abc";
    const std::string plist = pack::info_plist(spec);
    CHECK(plist.find("<key>LSHasLocalizedDisplayName</key>\n    <true/>") != std::string::npos);
    CHECK(plist.find("com.tommylau.armigo.01007EF00011E000") != std::string::npos);
    CHECK(plist.find("A&amp;B&lt;C&gt;") != std::string::npos);
    CHECK(plist.find("<string>AppIcon</string>") != std::string::npos);
    CHECK(plist.find("LSSupportsGameMode") != std::string::npos);
    CHECK(plist.find("GCSupportsGameMode") != std::string::npos);
    CHECK(plist.find("<string>26.0</string>") != std::string::npos);
    CHECK(plist.find("NSApplication") != std::string::npos);
    CHECK(plist.find("<string>Armigo</string>") != std::string::npos);
    CHECK(plist.find("ArmigoTitleFrameRate") == std::string::npos);
    spec.default_frame_rate = 30;
    spec.display_refresh_rate = 60;
    spec.supported_frame_rates = {30, 60, 120};
    spec.experimental_frame_rates = {120};
    const auto title_rate_plist = pack::info_plist(spec);
    CHECK(title_rate_plist.find("<key>ArmigoTitleFrameRate</key><dict><key>Default</key><integer>30</integer>") != std::string::npos);
    CHECK(title_rate_plist.find("<key>DisplayRefresh</key><integer>60</integer>") != std::string::npos);
    CHECK(title_rate_plist.find("<key>SupportedRates</key><array><integer>30</integer><integer>60</integer>"
                               "<integer>120</integer></array>") != std::string::npos);
    CHECK(title_rate_plist.find("<key>ExperimentalRates</key><array><integer>120</integer></array>") != std::string::npos);
    spec.platform = "ios";
    CHECK(pack::info_plist(spec).find("SupportedRates") == std::string::npos);
    CHECK(pack::info_plist(spec).find("ArmigoTitleFrameRate") == std::string::npos);
    spec.platform = "macos";
    spec.default_frame_rate = spec.display_refresh_rate = 0;
    const std::string json = pack::pack_json(spec);
    CHECK(json.find("\"title_id\"") != std::string::npos);
    CHECK(json.find("\"build_id\"") != std::string::npos);
    CHECK(json.find("\"title_version\": 42") != std::string::npos);
    CHECK(json.find("\"platform\": \"macos\"") != std::string::npos);
    CHECK(json.find("\"commit\": \"abc\"") != std::string::npos);
    CHECK(json.find("\"key\"") == std::string::npos);
    CHECK(json.find("keys") == std::string::npos);

    const std::vector<std::byte> jpeg = sample_jpeg();
    CHECK(jpeg.size() >= 2 && jpeg[0] == std::byte{0xff} && jpeg[1] == std::byte{0xd8});
    const std::vector<std::byte> empty{std::byte{0x00}};
    const pack::ControlIcon icons[] = {
        {"readme.txt", jpeg},
        {"icon_Japanese.dat", jpeg},
        {"icon_AmericanEnglish.dat", empty},
        {"icon_BritishEnglish.dat", jpeg},
    };
    const auto chosen = pack::select_control_icon(icons, armigo::pack::nacp::Language::american_english);
    CHECK(std::holds_alternative<pack::ControlIcon>(chosen));
    if (const auto* icon = std::get_if<pack::ControlIcon>(&chosen)) {
        CHECK(icon->name == "icon_AmericanEnglish.dat");
    }
    const pack::ControlIcon fallback[] = {
        {"icon_Japanese.dat", jpeg},
        {"icon_BritishEnglish.dat", jpeg},
    };
    const auto british =
        pack::select_control_icon(fallback, armigo::pack::nacp::Language::american_english);
    CHECK(std::holds_alternative<pack::ControlIcon>(british));
    if (const auto* icon = std::get_if<pack::ControlIcon>(&british)) {
        CHECK(icon->name == "icon_BritishEnglish.dat");
    }
    const auto missing =
        pack::select_control_icon({}, armigo::pack::nacp::Language::american_english);
    CHECK(std::holds_alternative<std::string>(missing));
    const auto icns = pack::jpeg_to_icns(jpeg);
    if (const auto* error = std::get_if<std::string>(&icns)) {
        std::printf("icon conversion: %s\n", error->c_str());
    }
    CHECK(std::holds_alternative<std::vector<std::byte>>(icns));
    if (const auto* bytes = std::get_if<std::vector<std::byte>>(&icns)) {
        CHECK(bytes->size() >= 4);
        CHECK(std::memcmp(bytes->data(), "icns", 4) == 0);
    }
    const auto png = pack::jpeg_to_ios_png(jpeg);
    CHECK(std::holds_alternative<std::vector<std::byte>>(png));
    if (const auto* bytes = std::get_if<std::vector<std::byte>>(&png)) {
        CHECK(bytes->size() > 24);
        CHECK(std::memcmp(bytes->data(), "\211PNG\r\n\032\n", 8) == 0);
        CHECK(std::to_integer<unsigned>((*bytes)[18]) == 4);
        CHECK(std::to_integer<unsigned>((*bytes)[22]) == 4);
    }

    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_bundle";
    fs::remove_all(directory);
    const fs::path title = directory / "title";
    fs::create_directories(title);
    {
        std::ofstream profile(title / "profile.json");
        profile << "{\"version\":1}\n";
    }
    const fs::path player = directory / "player";
    const std::string player_bytes = "player-bytes";
    {
        std::ofstream stream(player, std::ios::binary);
        stream << player_bytes;
    }
    spec.app_path = directory / "Sample.app";
    spec.player = player;
    spec.title_folder = title;
    spec.icon = jpeg;
    const auto assembled = pack::assemble_application(spec);
    CHECK(std::holds_alternative<fs::path>(assembled));
    const fs::path contents = spec.app_path / "Contents";
    CHECK(fs::is_regular_file(contents / "Info.plist"));
    CHECK(fs::is_regular_file(contents / "Resources" / "AppIcon.icns"));
    CHECK(fs::is_regular_file(contents / "Resources" / "Armigo" / "pack.json"));
    CHECK(fs::is_regular_file(contents / "Resources" / "Armigo" / "title" / "profile.json"));
    CHECK(!fs::exists(contents / "_CodeSignature"));
    std::ifstream copied(contents / "MacOS" / "Armigo", std::ios::binary);
    std::string copied_bytes((std::istreambuf_iterator<char>(copied)), std::istreambuf_iterator<char>());
    CHECK(copied_bytes == player_bytes);

    spec.app_path = directory / "SampleIos.app";
    spec.platform = "ios";
    spec.icon = std::get<std::vector<std::byte>>(png);
    spec.languages = {"en-US", "ja", "fr", "zh-TW", "zh-CN", "zh-Hant", "zh-Hans"};
    spec.supported_language_flag = 0x2fbff;
    spec.localized_titles = {{"ja", "\xe3\x82\xb5\xe3\x83\xb3\xe3\x83\x97\xe3\x83\xab"}, {"zh-Hans", "Sample \"S\""}};
    const std::string settings = pack::settings_bundle(spec);
    for (const char* entry : {"<string>Display</string>", "<string>Console Mode</string>", "<string>Resolution</string>",
                              "<string>Auto (1080p)</string>", "<string>Native</string>", "<string>Frame Rate</string>",
                              "<string>Overlay</string>", "<string>Performance Overlay</string>",
                              "<key>Key</key><string>performance_overlay</string>",
                              "<key>Key</key><string>frame_rate</string><key>DefaultValue</key><integer>60</integer>",
                              "<key>Key</key><string>console_mode</string><key>DefaultValue</key><string>docked</string>",
                              "Game language follows this app's language in system Settings."})
        CHECK(settings.find(entry) != std::string::npos);
    for (const char* absent : {"Show Frame Rate", "fps_overlay", "<string>language</string>", "<string>region</string>",
                               "<string>4K</string>", "V-Sync", "<integer>120</integer>", "zh-Hans"})
        CHECK(settings.find(absent) == std::string::npos);
    const auto names = pack::localized_names(spec);
    CHECK(names.size() == 5);
    if (names.size() == 5) {
        CHECK(names[0].first == "en.lproj");
        CHECK(names[0].second == "\"CFBundleDisplayName\" = \"A&B<C>\";\n\"CFBundleName\" = \"A&B<C>\";\n");
        CHECK(names[1].first == "ja.lproj");
        CHECK(names[1].second.find("\"CFBundleDisplayName\" = \"\xe3\x82\xb5\xe3\x83\xb3\xe3\x83\x97\xe3\x83\xab\";") == 0);
        CHECK(names[3].first == "zh-Hant.lproj");
        CHECK(names[3].second.find("= \"A&B<C>\";") != std::string::npos);
        CHECK(names[4].first == "zh-Hans.lproj");
        CHECK(names[4].second.find("= \"Sample \\\"S\\\"\";") != std::string::npos);
    }
    const std::string ios_plist = pack::info_plist(spec);
    CHECK(ios_plist.find("MinimumOSVersion") != std::string::npos);
    CHECK(ios_plist.find("<integer>1</integer><integer>2</integer>") != std::string::npos);
    CHECK(ios_plist.find("UIInterfaceOrientationLandscapeLeft") != std::string::npos);
    CHECK(ios_plist.find("UIRequiresFullScreen") != std::string::npos);
    CHECK(ios_plist.find("LSApplicationCategoryType</key><string>public.app-category.games") != std::string::npos);
    CHECK(ios_plist.find("LSSupportsGameMode</key><true/>") != std::string::npos);
    CHECK(ios_plist.find("GCSupportsGameMode</key><true/>") != std::string::npos);
    CHECK(ios_plist.find("GCSupportedGameControllers") != std::string::npos);
    CHECK(ios_plist.find("CFBundleLocalizations</key><array><string>en</string><string>ja</string><string>fr</string><string>zh-Hant</string><string>zh-Hans</string></array>") != std::string::npos);
    CHECK(pack::language_manifest(spec).starts_with("0x0002fbff\nen-US\nja\nfr\n"));
    const auto ios_assembled = pack::assemble_application(spec);
    CHECK(std::holds_alternative<fs::path>(ios_assembled));
    CHECK(fs::is_regular_file(spec.app_path / "Info.plist"));
    CHECK(fs::is_regular_file(spec.app_path / "ArmigoPlayer"));
    CHECK(fs::is_regular_file(spec.app_path / "AppIcon.png"));
    CHECK(fs::is_regular_file(spec.app_path / "Armigo" / "pack.json"));
    CHECK(fs::is_regular_file(spec.app_path / "Armigo" / "languages.txt"));
    CHECK(fs::is_regular_file(spec.app_path / "Armigo" / "title" / "profile.json"));
    CHECK(fs::is_regular_file(spec.app_path / "Settings.bundle" / "Root.plist"));
    CHECK(fs::is_regular_file(spec.app_path / "zh-Hans.lproj" / "InfoPlist.strings"));
    CHECK(pack::run_pack_tool({"/usr/bin/plutil", "-lint",
        (spec.app_path / "zh-Hans.lproj" / "InfoPlist.strings").string()}).empty());
    CHECK(pack::run_pack_tool({"/usr/bin/plutil", "-lint",
        (spec.app_path / "Settings.bundle" / "Root.plist").string()}).empty());
    CHECK(fs::is_regular_file(directory / "entitlements.plist"));
    CHECK(!fs::exists(spec.app_path / "Contents"));
    CHECK(!fs::exists(spec.app_path / "_CodeSignature"));
    CHECK(pack::create_ios_ipa(spec.app_path).empty());
    const fs::path ipa = directory / "SampleIos.ipa";
    CHECK(fs::is_regular_file(ipa));
    CHECK(pack::run_pack_tool({"/usr/bin/unzip", "-tqq", ipa.string()}).empty());
    CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
        "import sys,zipfile;z=zipfile.ZipFile(sys.argv[1]);assert all(i.compress_type==0 for i in z.infolist())",
        ipa.string()}).empty());
    std::ifstream archive(ipa, std::ios::binary);
    const std::string archive_bytes((std::istreambuf_iterator<char>(archive)), std::istreambuf_iterator<char>());
    CHECK(archive_bytes.find("Payload/SampleIos.app/ArmigoPlayer") != std::string::npos);
    CHECK(!pack::create_ios_ipa(spec.app_path).empty());
    fs::remove_all(directory);
}

void test_ios_paths_and_selection() {
    const fs::path staging = fs::path("work") / "stages";
    const auto fresh = pack::shader_stage_paths(staging, "app/Armigo", {}, true);
    CHECK(fresh.enumeration == staging / "enumeration");
    CHECK(fresh.macos_package == staging / "macos-shaders");
    CHECK(fresh.enumeration != fresh.macos_package);
    const auto reused = pack::shader_stage_paths(staging, "app/Armigo", "Packed.app", true);
    CHECK(reused.macos_package == "Packed.app/Contents/Resources/Armigo/shaders");
    const auto macos = pack::shader_stage_paths(staging, "app/Armigo", {}, false);
    CHECK(macos.enumeration == staging && macos.macos_package == "app/Armigo/shaders");

    const pack::PackedIdentity identity{"0100000000000001", std::string(64, 'd'), "macos", 7};
    CHECK(pack::validate_reused_selection(identity, {}, identity.title_id, identity.build_id, 7, {}).empty());
    CHECK(!pack::validate_reused_selection(identity, {}, identity.title_id, identity.build_id, 8, {}).empty());
    CHECK(!pack::validate_reused_selection(identity, {}, identity.title_id, std::string(64, 'e'), 7, {}).empty());
    const std::vector<fs::path> actual{fs::path("addons") / "0100000000000003"};
    const std::vector<std::string> selected{"0100000000000003"};
    CHECK(pack::validate_reused_selection(identity, actual, identity.title_id, identity.build_id, 7, selected).empty());
    CHECK(!pack::validate_reused_selection(identity, actual, identity.title_id, identity.build_id, 7, {}).empty());
}

void test_reused_addon_directories() {
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_addons";
    fs::remove_all(directory);
    const fs::path game = directory / "game";
    fs::create_directories(game);

    const auto missing = pack::reused_addon_directories(game);
    CHECK(std::holds_alternative<std::vector<fs::path>>(missing));
    if (const auto* paths = std::get_if<std::vector<fs::path>>(&missing)) CHECK(paths->empty());

    {
        std::ofstream file(game / "addons");
        file << "not-a-directory";
    }
    CHECK(std::holds_alternative<std::string>(pack::reused_addon_directories(game)));
    fs::remove(game / "addons");

    const fs::path addons = game / "addons";
    fs::create_directories(addons / "0100000000000003");
    {
        std::ofstream note(addons / "note.txt");
        note << "ignored";
    }
    const auto present = pack::reused_addon_directories(game);
    CHECK(std::holds_alternative<std::vector<fs::path>>(present));
    if (const auto* paths = std::get_if<std::vector<fs::path>>(&present)) {
        CHECK(paths->size() == 1);
        CHECK(paths->front().filename() == "0100000000000003");
    }

    fs::remove_all(addons / "0100000000000003");
    fs::remove(addons / "note.txt");
    fs::permissions(addons, fs::perms::none);
    CHECK(std::holds_alternative<std::string>(pack::reused_addon_directories(game)));
    fs::permissions(addons, fs::perms::owner_all);
    fs::remove_all(addons);
    fs::create_symlink(directory / "missing-addon-target", addons);
    CHECK(std::holds_alternative<std::string>(pack::reused_addon_directories(game)));
    fs::remove(addons);
    const fs::path linked_tree = directory / "linked-addons";
    fs::create_directories(linked_tree / "0100000000000004");
    fs::create_directory_symlink(linked_tree, addons);
    const auto linked = pack::reused_addon_directories(game);
    CHECK(std::holds_alternative<std::vector<fs::path>>(linked));
    if (const auto* paths = std::get_if<std::vector<fs::path>>(&linked)) {
        CHECK(paths->size() == 1);
        CHECK(paths->front().filename() == "0100000000000004");
    }
    fs::remove_all(directory);
}

void test_ios_reuse() {
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_reuse";
    fs::remove_all(directory);
    const fs::path app = directory / "Packed.app";
    const fs::path resources = app / "Contents" / "Resources" / "Armigo";
    fs::create_directories(resources / "game" / "exefs");
    pack::BundleSpec spec;
    spec.title_id = "0100000000000001";
    spec.build_id = std::string(64, 'd');
    spec.platform = "macos";
    {
        std::ofstream manifest(resources / "pack.json");
        manifest << pack::pack_json(spec);
        std::ofstream addon(resources / "game" / "romfs.bin");
        addon << "synthetic-game";
        std::array<char, 0x60> header{};
        std::memcpy(header.data(), "NSO0", 4);
        std::ofstream main(resources / "game" / "exefs" / "main", std::ios::binary);
        main.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    const auto identity = pack::read_packed_identity(app);
    CHECK(std::holds_alternative<pack::PackedIdentity>(identity));
    const auto selected = pack::select_macos_reuse({}, directory, directory / "Output.app", spec.title_id);
    CHECK(std::holds_alternative<std::optional<fs::path>>(selected));
    if (const auto* path = std::get_if<std::optional<fs::path>>(&selected)) {
        CHECK(path->has_value() && **path == app);
    }
    const fs::path cloned = directory / "clone" / "game";
    const auto result = pack::clone_macos_game(app, cloned, spec.title_id);
    CHECK(std::holds_alternative<pack::PackedIdentity>(result));
    std::ifstream input(cloned / "romfs.bin");
    std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    CHECK(bytes == "synthetic-game");
    CHECK(fs::is_regular_file(cloned / "exefs" / "main"));
    const auto wrong = pack::clone_macos_game(app, directory / "wrong", "0100000000000002");
    CHECK(std::holds_alternative<std::string>(wrong));
    CHECK(!fs::exists(directory / "wrong"));
    fs::remove_all(directory);
}

void test_firmware_input() {
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_firmware";
    fs::remove_all(directory);
    fs::create_directories(directory / "folder");
    const auto folder = pack::prepare_firmware_input(directory / "folder", directory / "unused");
    CHECK(std::holds_alternative<fs::path>(folder) && std::get<fs::path>(folder) == directory / "folder");
    const fs::path archive = directory / "firmware.zip";
    CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
        "import sys,zipfile;z=zipfile.ZipFile(sys.argv[1],'w');"
        "z.writestr('first.nca',b'first');z.writestr('second.cnmt.nca',b'meta');z.close()", archive.string()}).empty());
    auto prepared = pack::prepare_firmware_input(archive, directory / "expanded");
    CHECK(std::holds_alternative<fs::path>(prepared));
    CHECK(fs::is_regular_file(directory / "expanded" / "first.nca"));
    CHECK(fs::is_regular_file(directory / "expanded" / "second.cnmt.nca"));
    CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
        "import sys,zipfile;z=zipfile.ZipFile(sys.argv[1],'w');z.writestr('../escape.nca',b'bad');z.close()",
        archive.string()}).empty());
    CHECK(std::holds_alternative<std::string>(pack::prepare_firmware_input(archive, directory / "rejected")));
    const pack::Inputs inputs;
    const armigo::loader::ContainerKeys keys;
    CHECK(std::holds_alternative<std::string>(pack::extract_firmware_resources(
        inputs, directory / "folder", keys, directory / "system")));
    std::array<char, 256> version_bytes{};
    version_bytes[8] = 'N';
    version_bytes[9] = 'X';
    std::memcpy(version_bytes.data() + 0x68, "20.5.0", 6);
    const fs::path version_file = directory / "version";
    std::ofstream(version_file, std::ios::binary).write(version_bytes.data(), version_bytes.size());
    const auto version = pack::read_firmware_version(version_file);
    CHECK(std::holds_alternative<pack::FirmwareResources>(version));
    CHECK(std::get<pack::FirmwareResources>(version).version == "20.5.0");
    fs::remove_all(directory);
}

void test_embedded_preparer_environment() {
    const char* inherited = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
    const bool had_inherited = inherited != nullptr;
    const std::string saved = inherited ? inherited : "";
    for (const char* rate : {"60", "120"}) {
        setenv("ARMIGO_COMPAT_FRAME_RATE", rate, 1);
        for (const char* phase : {"capture", "verify"}) {
            CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
                "import os,sys;assert os.environ['ARMIGO_COMPAT_FRAME_RATE']=='0',sys.argv[1]", phase}, true).empty());
            CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_FRAME_RATE")) == rate);
        }
        CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
            "import os,sys;assert os.environ['ARMIGO_COMPAT_FRAME_RATE']==sys.argv[1]", rate}).empty());
    }
    if (had_inherited) setenv("ARMIGO_COMPAT_FRAME_RATE", saved.c_str(), 1);
    else unsetenv("ARMIGO_COMPAT_FRAME_RATE");
}

void test_git_work_tree() {
    const fs::path directory = fs::temp_directory_path() / "armigo_pack_cli_git";
    fs::remove_all(directory);
    const fs::path repo = directory / "repo";
    fs::create_directories(repo / "nested");
    {
        std::ofstream git(repo / ".git");
        git << "gitdir: nowhere\n";
    }
    CHECK(pack::inside_git_work_tree(repo / "nested" / "app"));
    CHECK(!pack::inside_git_work_tree(directory / "outside"));
    fs::remove_all(directory);
}

} // namespace

int main() {
    test_arguments();
    test_space();
    test_names();
    test_build_id();
    test_bundle_and_icon();
    test_ios_paths_and_selection();
    test_reused_addon_directories();
    test_ios_reuse();
    test_firmware_input();
    test_embedded_preparer_environment();
    test_git_work_tree();
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("pack.cli checks passed\n");
    return 0;
}
