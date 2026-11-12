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
#include "package_tools.h"
#include "pack_flow.h"
#include "run.h"
#include "sample_jpeg.h"
#include "settings.h"
#include "space.h"

#include <armigo/pack/naming.h>

#include <algorithm>
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
    const auto development = parse({"armigo", "game.xci", "--keys", "keys.txt", "--platform", "ios",
                                    "--firmware", "Firmware.zip", "--development-metal-hud"});
    CHECK(command_of(development) && command_of(development)->development_metal_hud);
    CHECK(failure_of(parse({"armigo", "game.xci", "--keys", "keys.txt", "--firmware", "Firmware.zip",
                            "--development-metal-hud"})));
    const auto development_pack = parse({"armigo", "game.xci", "--keys", "keys.txt", "--firmware", "Firmware.zip",
                                         "--development"});
    CHECK(command_of(development_pack) && command_of(development_pack)->development);
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
    // "pack" is the default verb; a shaders-only pack needs no firmware but an explicit output.
    const auto verb = parse({"armigo", "pack", "game.xci", "--keys", "keys.txt", "--firmware", "fw"});
    CHECK(command_of(verb) && command_of(verb)->game == "game.xci" && !command_of(verb)->shaders_only);
    const auto shaders = parse({"armigo", "pack", "game.nup", "--keys", "keys.txt", "--shaders-only", "--output", "out"});
    CHECK(command_of(shaders) && command_of(shaders)->shaders_only && command_of(shaders)->output == "out");
    const auto shaders_default_output = parse({"armigo", "game.xci", "--keys", "keys.txt", "--shaders-only"});
    CHECK(failure_of(shaders_default_output) && failure_of(shaders_default_output)->exit_code == 2);
    const auto shaders_ios = parse({"armigo", "game.xci", "--keys", "keys.txt", "--shaders-only", "--output", "out",
                                    "--platform", "ios"});
    CHECK(failure_of(shaders_ios) && failure_of(shaders_ios)->exit_code == 2);
    const auto no_firmware = parse({"armigo", "game.xci", "--keys", "keys.txt", "--output", "out"});
    CHECK(failure_of(no_firmware) && failure_of(no_firmware)->exit_code == 2);
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
        CHECK(name->file_name == "Sample Chronicle - Open Country (01007EF00011E000)");
        CHECK(name->executable_name == "Sample-Chronicle-Open-Country-01007EF00011E000");
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

void test_executable_names() {
    const auto named = [](std::string_view title, std::uint64_t id = 0x0100000000000001ull) {
        std::vector<std::byte> nacp(pack::nacp::control_file_size);
        std::memcpy(nacp.data(), title.data(), title.size());
        nacp[0x302c] = std::byte{1};
        return std::get<pack::PackedNames>(pack::names_from_control(id, nacp));
    };
    const auto first = named("Sample Chronicle: Open Country");
    const auto second = named("Sample Chronicle: Open Country", 0x0100000000000002ull);
    CHECK(first.executable_name == "Sample-Chronicle-Open-Country-0100000000000001");
    CHECK(first.executable_name != second.executable_name);
    CHECK(first.file_name != second.file_name);
    CHECK(first.file_name == "Sample Chronicle - Open Country (0100000000000001)");
    CHECK(named("Sample Chronicle: Open Country").executable_name == first.executable_name);
    CHECK(named("\xe6\xb8\xac\xe8\xa9\xa6").executable_name == "0100000000000001");
    CHECK(named(" ... ").file_name == "0100000000000001");
    for (const auto& title : {std::string("../../A\\B: \"C\" *?"), std::string(500, 'x'),
                              std::string("\xe6\xb8\xac\xe8\xa9\xa6")}) {
        const auto result = named(title);
        CHECK(result.executable_name.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-") == std::string::npos);
        CHECK(result.executable_name.size() <= 81);
        CHECK(result.file_name.size() <= 221);
        CHECK(pack::assembly_path(result.file_name + ".app", UINT64_MAX).filename().string().size() <= 255);
        CHECK(result.file_name.find_first_of("/:") == std::string::npos);
    }
    std::string long_utf8;
    for (int i = 0; i < 100; ++i) long_utf8 += "\xe6\xb8\xac";
    const auto unicode = named(long_utf8);
    CHECK(unicode.file_name == long_utf8.substr(0, 201) + " (0100000000000001)");

    std::vector<std::byte> nacp(pack::nacp::control_file_size);
    std::memcpy(nacp.data(), "Unsupported English", 19);
    std::memcpy(nacp.data() + 0x600, "Supported Japanese", 18);
    nacp[0x302c] = std::byte{4};
    CHECK(std::get<pack::PackedNames>(pack::names_from_control(1, nacp)).display_name == "Supported Japanese");
    nacp[0x302c] = std::byte{0};
    CHECK(std::holds_alternative<std::string>(pack::names_from_control(1, nacp)));
    nacp[0x302c] = std::byte{2};
    CHECK(std::holds_alternative<std::string>(pack::names_from_control(1, nacp)));
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
        profile << "{\"version\":1,\"patches\":[{\"title_id\":\"0x0100000000000001\",\"main_build_id\":\""
                << present << "\"}]}";
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
    // A pack may target only the platforms a title's settings list: macOS
    // only by default, and an iOS pack is refused before any work.
    CHECK(pack::check_title_platform(0x0100000000000001ull, "macos", directory / "armigo").empty());
    const auto refused = pack::check_title_platform(0x0100000000000001ull, "ios", directory / "armigo");
    CHECK(refused.find("does not support packing for ios") != std::string::npos);
    CHECK(refused.find("list macos") != std::string::npos);
    {
        std::ofstream profile(folder / "profile.json");
        profile << "{\"version\":1,\"patches\":[{\"title_id\":\"0x0100000000000001\",\"main_build_id\":\""
                << present << "\",\"platforms\":[\"macos\",\"ios\"]}]}";
    }
    CHECK(pack::check_title_platform(0x0100000000000001ull, "ios", directory / "armigo").empty());
    CHECK(pack::check_title_platform(0x0100000000000002ull, "ios", directory / "armigo").empty());
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
    spec.file_name = "A&B<C> (01007EF00011E000)";
    spec.executable_name = "Sample-01007EF00011E000";
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
    CHECK(plist.find("<string>Sample-01007EF00011E000</string>") != std::string::npos);
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
    // The app is assembled under a hidden name that is not a bundle and
    // appears under its own name only when published whole.
    const fs::path app = directory / (spec.file_name + ".app");
    const fs::path assembly = pack::assembly_path(app, 42);
    CHECK(assembly.parent_path() == directory);
    CHECK(assembly.filename().string().starts_with("."));
    CHECK(assembly.extension() != ".app");
    CHECK(pack::assembly_path(app, 43) != assembly);
    spec.app_path = assembly;
    spec.player = player;
    spec.title_folder = title;
    spec.icon = jpeg;
    auto invalid_spec = spec;
    invalid_spec.executable_name = "../outside";
    CHECK(std::holds_alternative<std::string>(pack::assemble_application(invalid_spec)));
    const auto assembled = pack::assemble_application(spec);
    CHECK(std::holds_alternative<fs::path>(assembled));
    CHECK(!fs::exists(app));
    CHECK(pack::publish_application(assembly, app).empty());
    CHECK(!fs::exists(assembly));
    const fs::path contents = app / "Contents";
    CHECK(fs::is_regular_file(contents / "Info.plist"));
    CHECK(fs::is_regular_file(contents / "Resources" / "AppIcon.icns"));
    CHECK(fs::is_regular_file(contents / "Resources" / "Armigo" / "pack.json"));
    CHECK(fs::is_regular_file(contents / "Resources" / "Armigo" / "licenses" / "Opus.txt"));
    CHECK(fs::is_regular_file(contents / "Resources" / "Armigo" / "title" / "profile.json"));
    CHECK(!fs::exists(contents / "_CodeSignature"));
    std::ifstream copied(contents / "MacOS" / spec.executable_name, std::ios::binary);
    std::string copied_bytes((std::istreambuf_iterator<char>(copied)), std::istreambuf_iterator<char>());
    CHECK(copied_bytes == player_bytes);
    // Publishing never replaces an app that is already there.
    fs::create_directories(assembly / "Contents");
    const std::string refused = pack::publish_application(assembly, app);
    CHECK(refused.find("already exists") != std::string::npos);
    CHECK(fs::is_directory(assembly / "Contents"));
    CHECK(fs::is_regular_file(contents / "Info.plist"));
    fs::remove_all(assembly);

    const fs::path ios_app = directory / "SampleIos.app";
    const fs::path ios_assembly = pack::assembly_path(ios_app, 42);
    spec.app_path = ios_assembly;
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
    CHECK(ios_plist.find("<key>CFBundleExecutable</key><string>" + spec.executable_name + "</string>") != std::string::npos);
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
    CHECK(pack::info_plist(spec).find("ArmigoDevelopmentMetalHud") == std::string::npos);
    CHECK(pack::info_plist(spec).find("ArmigoDevelopmentPack") == std::string::npos);
    auto development_spec = spec;
    development_spec.development = true;
    CHECK(pack::info_plist(development_spec).find("ArmigoDevelopmentPack</key><true/>") != std::string::npos);
    CHECK(pack::settings_bundle(spec).find("development_metal_hud") == std::string::npos);
    spec.development_metal_hud = true;
    CHECK(pack::info_plist(spec).find("ArmigoDevelopmentMetalHud</key><true/>") != std::string::npos);
    CHECK(pack::info_plist(spec).find("ArmigoDevelopmentPack</key><true/>") != std::string::npos);
    CHECK(pack::settings_bundle(spec).find("Metal Performance HUD") != std::string::npos);
    const auto ios_assembled = pack::assemble_application(spec);
    CHECK(std::holds_alternative<fs::path>(ios_assembled));
    CHECK(fs::is_regular_file(spec.app_path / "Info.plist"));
    CHECK(fs::is_regular_file(spec.app_path / spec.executable_name));
    CHECK(fs::is_regular_file(spec.app_path / "AppIcon.png"));
    CHECK(fs::is_regular_file(spec.app_path / "Armigo" / "pack.json"));
    CHECK(fs::is_regular_file(spec.app_path / "Armigo" / "languages.txt"));
    CHECK(fs::file_size(spec.app_path / "Armigo" / "licenses" / "Opus.txt") > 1000);
    CHECK(fs::file_size(spec.app_path / "Armigo" / "licenses" / "Zstandard.txt") > 1000);
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
    // The IPA is made from the assembly and names the app by its final name.
    CHECK(pack::create_ios_ipa(ios_assembly, ios_app).empty());
    CHECK(!fs::exists(ios_app));
    const fs::path ipa = directory / "SampleIos.ipa";
    CHECK(fs::is_regular_file(ipa));
    CHECK(pack::run_pack_tool({"/usr/bin/unzip", "-tqq", ipa.string()}).empty());
    CHECK(pack::run_pack_tool({"/usr/bin/python3", "-c",
        "import sys,zipfile;z=zipfile.ZipFile(sys.argv[1]);assert all(i.compress_type==0 for i in z.infolist())",
        ipa.string()}).empty());
    std::ifstream archive(ipa, std::ios::binary);
    const std::string archive_bytes((std::istreambuf_iterator<char>(archive)), std::istreambuf_iterator<char>());
    CHECK(archive_bytes.find("Payload/SampleIos.app/Sample-01007EF00011E000") != std::string::npos);
    CHECK(archive_bytes.find(".partial") == std::string::npos);
    CHECK(!pack::create_ios_ipa(ios_assembly, ios_app).empty());
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
    std::memcpy(version_bytes.data() + 0x68, "23.0.1", 6);
    const fs::path version_file = directory / "version";
    std::ofstream(version_file, std::ios::binary).write(version_bytes.data(), version_bytes.size());
    const auto current = pack::read_firmware_version(version_file);
    CHECK(std::holds_alternative<pack::FirmwareResources>(current));
    CHECK(std::get<pack::FirmwareResources>(current).version == "23.0.1");
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

void write_text(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

void test_compare_packages() {
    const auto compare = parse({"armigo", "pack", "--compare", "old", "new"});
    CHECK(command_of(compare) && command_of(compare)->compare.size() == 2 && command_of(compare)->compare[1] == "new");
    CHECK(failure_of(parse({"armigo", "--compare", "old"})));
    CHECK(failure_of(parse({"armigo", "game.xci", "--keys", "k", "--compare", "old", "new"})));

    const fs::path root = fs::temp_directory_path() / "armigo_pack_compare";
    fs::remove_all(root);
    const fs::path a = root / "a", b = root / "b";
    const std::string profile_a = R"({\"metal_version\":\"1\",\"source_version\":\"aa\"})";
    const std::string profile_b = R"({\"metal_version\":\"1\",\"source_version\":\"bb\"})";
    write_text(a / "manifest.json", "{\"chain_id\":\"c1\",\"entries\":[1,2],\"profile_text\":\"" + profile_a + "\"}");
    write_text(b / "manifest.json", "{\"chain_id\":\"c2\",\"entries\":[1,2],\"profile_text\":\"" + profile_b + "\"}");
    write_text(a / "stages/s1/program.bin", "p1");
    write_text(b / "stages/s1/program.bin", "p1");
    write_text(a / "stages/s1/reflection.json", "r1");
    write_text(b / "stages/s1/reflection.json", "r2");
    write_text(a / "stages/s2/program.bin", "p2");
    write_text(b / "stages/s3/program.bin", "p3");
    const auto result = pack::compare_packages(a, b);
    CHECK(std::holds_alternative<pack::PackageComparison>(result));
    if (const auto* comparison = std::get_if<pack::PackageComparison>(&result)) {
        CHECK(!comparison->same());
        CHECK(comparison->common == 1 && comparison->only_old == 1 && comparison->only_new == 1);
        CHECK(comparison->identical.at("program.bin") == 1 && comparison->differing.at("reflection.json") == 1);
        const auto text = pack::format_comparison(*comparison);
        CHECK(text.find("chain_id: \"c1\" -> \"c2\"") != std::string::npos);
        CHECK(text.find("profile.source_version: \"aa\" -> \"bb\"") != std::string::npos);
        CHECK(text.find("profile.metal_version") == std::string::npos);
        CHECK(text.find("entries") == std::string::npos);
        CHECK(text.find("differs: s1/reflection.json") != std::string::npos);
    }
    const auto self = pack::compare_packages(a, a);
    CHECK(std::holds_alternative<pack::PackageComparison>(self) && std::get<pack::PackageComparison>(self).same());
    CHECK(std::holds_alternative<std::string>(pack::compare_packages(a, root / "missing")));
    fs::remove_all(root);
}

const std::string* environment_value(const pack::RunPlan& plan, std::string_view name) {
    for (const auto& entry : plan.environment) {
        if (entry.first == name) return &entry.second;
    }
    return nullptr;
}

std::vector<std::string>::const_iterator flag_at(const std::vector<std::string>& arguments, std::string_view flag) {
    return std::find(arguments.begin(), arguments.end(), std::string(flag));
}

void test_run_command() {
    CHECK(pack::command_usage().find("armigo run <App.app|game>") != std::string::npos);
    const auto launched = parse({"armigo", "run", "game.nup", "--frames", "30", "--rate", "60"});
    const auto* command = command_of(launched);
    CHECK(command && command->run && command->game == "game.nup");
    if (command) {
        CHECK(command->run_options.frames == 30);
        CHECK(command->run_options.rate == "60");
        CHECK(command->run_options.time_limit == "300");
        CHECK(!command->run_options.audio);
    }
    const auto audible = parse({"armigo", "run", "game.nup", "--audio"});
    CHECK(command_of(audible) && command_of(audible)->run_options.audio);
    const auto missing = parse({"armigo", "run"});
    CHECK(failure_of(missing) && failure_of(missing)->exit_code == 2);
    CHECK(failure_of(parse({"armigo", "run", "g.xci", "--output", "x"})));
    CHECK(failure_of(parse({"armigo", "run", "App.app", "--width", "10"})));
    CHECK(failure_of(parse({"armigo", "run", "g.xci", "--frames", "0"})));
    CHECK(failure_of(parse({"armigo", "run", "g.xci", "--update", "a.nsp", "--update", "b.nsp"})));

    const fs::path root = fs::temp_directory_path() / "armigo_pack_cli_run";
    fs::remove_all(root);
    const fs::path app = (root / "App.app").lexically_normal();
    const fs::path resources = app / "Contents" / "Resources" / "Armigo";
    const std::string build_id(64, 'a');
    write_text(resources / "pack.json",
               std::string("{\"title_id\":\"0100000000000001\",\"build_id\":\"") + build_id +
                   "\",\"platform\":\"macos\",\"title_version\":0}");
    fs::create_directories(resources / "game" / "exefs");
    fs::create_directories(resources / "game" / "romfs");
    fs::create_directories(resources / "game" / "addons" / "0100000000001001");
    fs::create_directories(resources / "shaders");
    fs::create_directories(resources / "system");
    fs::create_directories(resources / "pipelines");
    write_text(resources / "pipelines" / "archive.bin", "cache");
    write_text(resources / "title" / "profile.json", "{}\n");

    pack::Command run;
    run.run = true;
    run.game = app;
    const fs::path runner = root / "armigo-headless";
    const fs::path scratch = root / "scratch";
    const auto planned = pack::plan_run(run, runner, scratch);
    CHECK(std::holds_alternative<pack::RunPlan>(planned));
    if (const auto* plan = std::get_if<pack::RunPlan>(&planned)) {
        const auto exefs = flag_at(plan->arguments, "--exefs");
        const auto romfs = flag_at(plan->arguments, "--romfs");
        const auto shaders = flag_at(plan->arguments, "--shaders");
        const auto addon = flag_at(plan->arguments, "--addon");
        CHECK(exefs != plan->arguments.end() && romfs != plan->arguments.end());
        CHECK(shaders != plan->arguments.end() && addon != plan->arguments.end());
        CHECK(exefs < romfs && romfs < shaders && shaders < addon);
        CHECK(exefs != plan->arguments.end() && exefs + 1 != plan->arguments.end() &&
              *(exefs + 1) == (resources / "game" / "exefs").string());
        CHECK(romfs != plan->arguments.end() && romfs + 1 != plan->arguments.end() &&
              *(romfs + 1) == (resources / "game" / "romfs").string());
        CHECK(shaders != plan->arguments.end() && shaders + 1 != plan->arguments.end() &&
              *(shaders + 1) == (resources / "shaders").string());
        CHECK(addon != plan->arguments.end() && addon + 1 != plan->arguments.end() &&
              *(addon + 1) == (resources / "game" / "addons" / "0100000000001001").string());
        CHECK(std::count(plan->arguments.begin(), plan->arguments.end(), "--shaders") == 1);
        const std::string* audio = environment_value(*plan, "ARMIGO_AUDIO");
        const std::string* titles = environment_value(*plan, "ARMIGO_TITLES");
        CHECK(audio && *audio == "0");
        CHECK(titles && *titles == (scratch / "titles").string());
        CHECK(plan->clones.size() == 1);
        CHECK(plan->clones[0].first == resources / "pipelines");
        CHECK(plan->clones[0].second == scratch / "pso");
        std::error_code error;
        CHECK(plan->links.size() == 1);
        CHECK(plan->links[0].first == fs::absolute(resources / "title", error));
        CHECK(plan->links[0].second == scratch / "titles" / "0100000000000001");
    }

    // A packed .nup app keeps its archive as game/title.nup and has no RomFS tree.
    {
        const fs::path nup_app = (root / "Nup.app").lexically_normal();
        const fs::path nup_resources = nup_app / "Contents" / "Resources" / "Armigo";
        write_text(nup_resources / "pack.json",
                   std::string("{\"title_id\":\"0400000000000001\",\"build_id\":\"") + build_id +
                       "\",\"platform\":\"macos\",\"title_version\":0}");
        fs::create_directories(nup_resources / "game" / "exefs");
        write_text(nup_resources / "game" / "title.nup", "nup");
        fs::create_directories(nup_resources / "shaders");
        write_text(nup_resources / "title" / "profile.json", "{}\n");
        pack::Command nup_run;
        nup_run.run = true;
        nup_run.game = nup_app;
        const auto nup_planned = pack::plan_run(nup_run, runner, scratch);
        CHECK(std::holds_alternative<pack::RunPlan>(nup_planned));
        if (const auto* plan = std::get_if<pack::RunPlan>(&nup_planned)) {
            const auto archive = flag_at(plan->arguments, "--nup");
            CHECK(archive != plan->arguments.end() && archive + 1 != plan->arguments.end() &&
                  *(archive + 1) == (nup_resources / "game" / "title.nup").string());
            CHECK(flag_at(plan->arguments, "--exefs") == plan->arguments.end());
            CHECK(flag_at(plan->arguments, "--shaders") != plan->arguments.end());
        }
    }

    write_text(root / "game.xci", "xci");
    write_text(root / "keys.txt", "keys");
    write_text(root / "title.keys", "title");
    write_text(root / "update.nsp", "update");
    pack::Command container;
    container.run = true;
    container.game = root / "game.xci";
    CHECK(std::holds_alternative<std::string>(pack::plan_run(container, runner, scratch)));
    container.run_options.keys = root / "keys.txt";
    container.run_options.title_keys = root / "title.keys";
    container.run_options.update = root / "update.nsp";
    const auto container_plan = pack::plan_run(container, runner, scratch);
    CHECK(std::holds_alternative<pack::RunPlan>(container_plan));
    if (const auto* plan = std::get_if<pack::RunPlan>(&container_plan)) {
        const auto file = flag_at(plan->arguments, "--container");
        const auto keys = flag_at(plan->arguments, "--keys");
        const auto title_keys = flag_at(plan->arguments, "--title-keys");
        const auto update = flag_at(plan->arguments, "--update");
        const auto cache = flag_at(plan->arguments, "--cache");
        CHECK(file < keys && keys < title_keys && title_keys < update && update < cache);
        CHECK(file != plan->arguments.end() && file + 1 != plan->arguments.end() &&
              *(file + 1) == container.game.string());
        CHECK(cache != plan->arguments.end() && cache + 1 != plan->arguments.end() &&
              *(cache + 1) == (scratch / "extract").string());
    }

    pack::Command nup;
    nup.run = true;
    nup.game = root / "game.nup";
    nup.run_options.keys = root / "keys.txt";
    nup.run_options.update = root / "update.nsp";
    write_text(nup.game, "nup");
    const auto nup_plan = pack::plan_run(nup, runner, scratch);
    CHECK(std::holds_alternative<pack::RunPlan>(nup_plan));
    if (const auto* plan = std::get_if<pack::RunPlan>(&nup_plan)) {
        const auto file = flag_at(plan->arguments, "--nup");
        CHECK(file != plan->arguments.end() && file + 1 != plan->arguments.end() && *(file + 1) == nup.game.string());
        CHECK(flag_at(plan->arguments, "--keys") == plan->arguments.end());
        CHECK(flag_at(plan->arguments, "--update") == plan->arguments.end());
    }

    pack::Command folder;
    folder.run = true;
    folder.game = (root / "extracted").lexically_normal();
    fs::create_directories(folder.game / "exefs");
    fs::create_directories(folder.game / "romfs");
    const auto folder_plan = pack::plan_run(folder, runner, scratch);
    CHECK(std::holds_alternative<pack::RunPlan>(folder_plan));
    if (const auto* plan = std::get_if<pack::RunPlan>(&folder_plan)) {
        const auto exefs = flag_at(plan->arguments, "--exefs");
        const auto romfs = flag_at(plan->arguments, "--romfs");
        CHECK(exefs < romfs);
        CHECK(exefs != plan->arguments.end() && exefs + 1 != plan->arguments.end() &&
              *(exefs + 1) == (folder.game / "exefs").string());
        CHECK(romfs != plan->arguments.end() && romfs + 1 != plan->arguments.end() &&
              *(romfs + 1) == (folder.game / "romfs").string());
    }

    const auto summary = pack::summarize_report(
        R"({"outcome":"frame-target","statistics":{"completed_frames":600,"elapsed_ns":10000000000}})");
    CHECK(summary.find("600 frames") != std::string::npos);
    CHECK(summary.find("60.0 fps") != std::string::npos);
    CHECK(pack::summarize_report("not json").empty());
    CHECK(summary.find("teardown") == std::string::npos);
    CHECK(pack::summarize_report(R"({"outcome":"success","teardown_failure":{"detail":"x"},)"
                                 R"("statistics":{"completed_frames":60,"elapsed_ns":1000000000}})")
              .find("; teardown failed") != std::string::npos);
    CHECK(pack::summarize_report(
              R"({"outcome":"success","statistics":{"completed_frames":-1,"elapsed_ns":-1000000000}})")
              .empty());
    fs::remove_all(root);
}

void test_pack_flow() {
    const char* cache_env = std::getenv("ARMIGO_SHADER_COMPILE_CACHE");
    const char* home_env = std::getenv("HOME");
    const std::string saved_cache = cache_env ? cache_env : std::string();
    const std::string saved_home = home_env ? home_env : std::string();
    const bool had_cache = cache_env != nullptr;
    const bool had_home = home_env != nullptr;
    const auto restore_env = [&]() {
        if (had_cache) setenv("ARMIGO_SHADER_COMPILE_CACHE", saved_cache.c_str(), 1);
        else unsetenv("ARMIGO_SHADER_COMPILE_CACHE");
        if (had_home) setenv("HOME", saved_home.c_str(), 1);
        else unsetenv("HOME");
    };

    unsetenv("ARMIGO_SHADER_COMPILE_CACHE");
    setenv("HOME", "/pack-home", 1);
    const fs::path cache = pack::default_shader_compile_cache("0100000000000001");
    CHECK(cache.string() == "/pack-home/Library/Caches/Armigo/0100000000000001/compile-cache");
    setenv("ARMIGO_SHADER_COMPILE_CACHE", "", 1);
    CHECK(pack::default_shader_compile_cache("0100000000000001").empty());
    unsetenv("ARMIGO_SHADER_COMPILE_CACHE");
    unsetenv("HOME");
    CHECK(pack::default_shader_compile_cache("0100000000000001").empty());
    restore_env();

    const pack::InputCapabilities game = pack::game_input_capabilities();
    const pack::InputCapabilities nup = pack::nup_input_capabilities();
    CHECK(game.updates_and_addons && game.ios && game.embeds_firmware);
    CHECK(!game.native_shaders && game.pipeline_list_required);
    CHECK(!nup.updates_and_addons && !nup.ios && !nup.embeds_firmware);
    CHECK(nup.native_shaders && !nup.pipeline_list_required);

    const fs::path root = fs::temp_directory_path() / "armigo_pack_flow_manifests";
    fs::remove_all(root);
    const fs::path shaders = root / "shaders";
    const fs::path compute = shaders / "native" / "compute-pipelines.tsv";
    fs::create_directories(compute.parent_path());
    {
        std::ofstream stream(compute);
        stream << "C\tabc\n";
    }
    armigo::compat::PatchDescription description;
    description.pso_manifest = (root / "pipelines.tsv").string();
    const auto game_lists = pack::select_pack_manifests(description, shaders, game);
    CHECK(game_lists.size() == 1 && game_lists[0].string() == description.pso_manifest);
    CHECK(pack::select_pack_manifests(description, shaders, game) == game_lists);
    description.pso_manifest.clear();
    const auto game_empty = pack::select_pack_manifests(description, shaders, game);
    CHECK(game_empty.size() == 1 && game_empty[0].empty());

    description.pso_manifest = (root / "pipelines.tsv").string();
    const auto nup_lists = pack::select_pack_manifests(description, shaders, nup);
    CHECK(nup_lists.size() == 2);
    if (nup_lists.size() == 2) {
        CHECK(nup_lists[0].string() == description.pso_manifest);
        CHECK(nup_lists[1].string() == compute.string());
    }
    CHECK(pack::select_pack_manifests(description, shaders, nup) == nup_lists);
    description.pso_manifest.clear();
    const auto compute_only = pack::select_pack_manifests(description, shaders, nup);
    CHECK(compute_only.size() == 1 && compute_only[0].string() == compute.string());
    fs::remove(compute);
    CHECK(pack::select_pack_manifests(description, shaders, nup).empty());

    // A .nup pack builds its shaders inside the hidden assembly, which is
    // renamed when the app is published: the bundle carries the packaged
    // compute list, and the warm-up must read it from the published app.
    {
        const fs::path app = root / "0400000000000001.app";
        const fs::path assembly = pack::assembly_path(app, 7);
        const fs::path armigo = assembly / "Contents" / "Resources" / "Armigo";
        const fs::path packaged = armigo / "shaders" / "native" / "compute-pipelines.tsv";
        fs::create_directories(packaged.parent_path());
        { std::ofstream(packaged) << "C\tabc\n"; }
        fs::create_directories(root / "title");
        { std::ofstream(root / "title" / "profile.json") << "{\"version\":1}\n"; }
        { std::ofstream(root / "player", std::ios::binary) << "player"; }
        const auto assembled_lists = pack::select_pack_manifests(description, armigo / "shaders", nup);
        pack::BundleSpec bundle;
        bundle.app_path = assembly;
        bundle.display_name = bundle.file_name = bundle.executable_name = bundle.title_id = "0400000000000001";
        bundle.bundle_identifier = "com.tommylau.armigo.0400000000000001";
        bundle.build_id = std::string(64, 'a');
        bundle.platform = "macos";
        bundle.player = root / "player";
        bundle.title_folder = root / "title";
        static const std::array<std::byte, 4> icon{std::byte{0x69}, std::byte{0x63}, std::byte{0x6e}, std::byte{0x73}};
        bundle.icon = icon;
        bundle.languages = {"en-US"};
        bundle.supported_language_flag = 1;
        CHECK(std::holds_alternative<fs::path>(pack::assemble_application(bundle)));
        CHECK(pack::publish_application(assembly, app).empty());
        const fs::path published = app / "Contents" / "Resources" / "Armigo" / "shaders" / "native" /
                                   "compute-pipelines.tsv";
        CHECK(fs::is_regular_file(published));
        CHECK(assembled_lists.size() == 1 && !fs::exists(assembled_lists[0]));
        const auto warm = pack::warm_up_manifests(description, app, nup);
        CHECK(warm.size() == 1 && warm[0] == published);
        description.pso_manifest = (root / "pipelines.tsv").string();
        const auto warm_both = pack::warm_up_manifests(description, app, nup);
        CHECK(warm_both.size() == 2 && warm_both[0].string() == description.pso_manifest && warm_both[1] == published);
        CHECK(pack::warm_up_manifests(description, app, game) ==
              std::vector<fs::path>{fs::path(description.pso_manifest)});
    }
    fs::remove_all(root);

    armigo::compat::FrameRateSettings rates;
    rates.default_rate = 30;
    rates.display_refresh_rate = 120;
    rates.supported_rates = {30, 60};
    rates.experimental_rates = {120};
    armigo::compat::PatchDescription rated;
    rated.frame_rate = rates;
    pack::BundleSpec spec;
    spec.display_name = "keep";
    pack::CommonBundleFields fields;
    fields.title_id = "0100000000000001";
    fields.build_id = "aa";
    fields.commit = "abc";
    fields.player = "Armigo.linker";
    fields.title_folder = "titles/0100000000000001";
    fields.platform = "macos";
    fields.apply_frame_rates = false;
    pack::fill_common_bundle_fields(spec, fields, rated);
    CHECK(spec.display_name == "keep");
    CHECK(spec.title_id == "0100000000000001");
    CHECK(spec.build_id == "aa");
    CHECK(spec.commit == "abc");
    CHECK(spec.player == "Armigo.linker");
    CHECK(spec.title_folder == "titles/0100000000000001");
    CHECK(spec.platform == "macos");
    CHECK(spec.default_frame_rate == 0);
    CHECK(spec.display_refresh_rate == 0);
    CHECK(spec.supported_frame_rates.empty());
    CHECK(spec.experimental_frame_rates.empty());
    fields.apply_frame_rates = true;
    pack::fill_common_bundle_fields(spec, fields, rated);
    CHECK(spec.default_frame_rate == 30);
    CHECK(spec.display_refresh_rate == 120);
    CHECK(spec.supported_frame_rates.size() == 2 && spec.supported_frame_rates[1] == 60);
    CHECK(spec.experimental_frame_rates.size() == 1 && spec.experimental_frame_rates[0] == 120);
    pack::BundleSpec ios_spec;
    fields.platform = "ios";
    fields.apply_frame_rates = true;
    pack::fill_common_bundle_fields(ios_spec, fields, armigo::compat::PatchDescription{});
    CHECK(ios_spec.platform == "ios");
    CHECK(ios_spec.default_frame_rate == 0);
    CHECK(ios_spec.supported_frame_rates.empty());
}

int main() {
    test_arguments();
    test_space();
    test_names();
    test_executable_names();
    test_build_id();
    test_bundle_and_icon();
    test_ios_paths_and_selection();
    test_reused_addon_directories();
    test_ios_reuse();
    test_firmware_input();
    test_embedded_preparer_environment();
    test_git_work_tree();
    test_compare_packages();
    test_run_command();
    test_pack_flow();
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("pack.cli checks passed\n");
    return 0;
}
