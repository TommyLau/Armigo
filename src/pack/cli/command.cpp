// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "command.h"

#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

namespace armigo::pack {
namespace {

CommandFailure usage(std::string detail) {
    return CommandFailure{std::move(detail), 2};
}

bool pack_only_option(std::string_view argument, std::string_view& flag) {
    static constexpr std::string_view flags[] = {
        "--dlc", "--firmware", "--output", "--platform", "--reuse", "--ios-build", "--keep-app", "--no-ipa", "--development", "--development-metal-hud",
        "--shaders-only", "--restamp", "--compare",
    };
    for (const std::string_view name : flags) {
        if (argument == name || (argument.size() > name.size() && argument.starts_with(name) &&
                                 argument[name.size()] == '=')) {
            flag = name;
            return true;
        }
    }
    return false;
}

std::optional<std::uint64_t> positive_integer(std::string_view text) {
    if (text.empty() || text.front() == '+' || text.front() == '-') return std::nullopt;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) return std::nullopt;
    return value;
}

// A positive number of seconds: digits with one optional decimal point.
bool positive_seconds(std::string_view text) {
    if (text.empty()) return false;
    bool dot = false;
    bool nonzero = false;
    for (const char character : text) {
        if (character == '.') {
            if (dot) return false;
            dot = true;
        } else if (character >= '0' && character <= '9') {
            if (character != '0') nonzero = true;
        } else {
            return false;
        }
    }
    return nonzero;
}

std::variant<CommandFailure, Command> parse_run(int argc, char** argv) {
    Command command;
    command.run = true;
    bool saw_game = false;
    bool saw_keys = false;
    bool saw_title_keys = false;
    bool saw_update = false;
    bool saw_shaders = false;
    bool saw_frames = false;
    bool saw_time_limit = false;
    bool saw_rate = false;
    bool saw_width = false;
    bool saw_height = false;
    bool saw_saves = false;
    bool saw_report = false;
    bool saw_fonts = false;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto take_value = [&](std::string_view flag) -> std::variant<CommandFailure, std::string_view> {
            const auto equals = argument.find('=');
            if (equals != std::string_view::npos) {
                const auto value = argument.substr(equals + 1);
                if (value.empty()) return usage(std::string(flag) + " requires a value.");
                return value;
            }
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty()) {
                return usage(std::string(flag) + " requires a value.");
            }
            return std::string_view(argv[++index]);
        };
        if (argument == "--help" || argument == "-h") {
            command.help = true;
            return command;
        }
        std::string_view pack_flag;
        if (pack_only_option(argument, pack_flag)) {
            return usage(std::string(pack_flag) + " cannot be used with run.");
        }
        if (argument == "--keys" || argument.starts_with("--keys=")) {
            if (saw_keys) return usage("--keys was given more than once.");
            auto value = take_value("--keys");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.keys = std::get<std::string_view>(value);
            saw_keys = true;
            continue;
        }
        if (argument == "--title-keys" || argument.starts_with("--title-keys=")) {
            if (saw_title_keys) return usage("--title-keys was given more than once.");
            auto value = take_value("--title-keys");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.title_keys = std::get<std::string_view>(value);
            saw_title_keys = true;
            continue;
        }
        if (argument == "--update" || argument.starts_with("--update=")) {
            if (saw_update) return usage("--update was given more than once.");
            auto value = take_value("--update");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.update = std::get<std::string_view>(value);
            saw_update = true;
            continue;
        }
        if (argument == "--shaders" || argument.starts_with("--shaders=")) {
            if (saw_shaders) return usage("--shaders was given more than once.");
            auto value = take_value("--shaders");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.shaders = std::get<std::string_view>(value);
            saw_shaders = true;
            continue;
        }
        if (argument == "--frames" || argument.starts_with("--frames=")) {
            if (saw_frames) return usage("--frames was given more than once.");
            auto value = take_value("--frames");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            const auto frames = positive_integer(std::get<std::string_view>(value));
            if (!frames) return usage("--frames requires a positive integer.");
            command.run_options.frames = *frames;
            saw_frames = true;
            continue;
        }
        if (argument == "--time-limit" || argument.starts_with("--time-limit=")) {
            if (saw_time_limit) return usage("--time-limit was given more than once.");
            auto value = take_value("--time-limit");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            const auto text = std::get<std::string_view>(value);
            if (!positive_seconds(text)) return usage("--time-limit requires a positive number of seconds.");
            command.run_options.time_limit = std::string(text);
            saw_time_limit = true;
            continue;
        }
        if (argument == "--rate" || argument.starts_with("--rate=")) {
            if (saw_rate) return usage("--rate was given more than once.");
            auto value = take_value("--rate");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.rate = std::string(std::get<std::string_view>(value));
            saw_rate = true;
            continue;
        }
        if (argument == "--width" || argument.starts_with("--width=")) {
            if (saw_width) return usage("--width was given more than once.");
            auto value = take_value("--width");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            const auto width = positive_integer(std::get<std::string_view>(value));
            if (!width || *width > std::numeric_limits<std::uint32_t>::max()) {
                return usage("--width requires a positive integer.");
            }
            command.run_options.width = static_cast<std::uint32_t>(*width);
            saw_width = true;
            continue;
        }
        if (argument == "--height" || argument.starts_with("--height=")) {
            if (saw_height) return usage("--height was given more than once.");
            auto value = take_value("--height");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            const auto height = positive_integer(std::get<std::string_view>(value));
            if (!height || *height > std::numeric_limits<std::uint32_t>::max()) {
                return usage("--height requires a positive integer.");
            }
            command.run_options.height = static_cast<std::uint32_t>(*height);
            saw_height = true;
            continue;
        }
        if (argument == "--saves" || argument.starts_with("--saves=")) {
            if (saw_saves) return usage("--saves was given more than once.");
            auto value = take_value("--saves");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.saves = std::get<std::string_view>(value);
            saw_saves = true;
            continue;
        }
        if (argument == "--report" || argument.starts_with("--report=")) {
            if (saw_report) return usage("--report was given more than once.");
            auto value = take_value("--report");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.report = std::get<std::string_view>(value);
            saw_report = true;
            continue;
        }
        if (argument == "--fonts" || argument.starts_with("--fonts=")) {
            if (saw_fonts) return usage("--fonts was given more than once.");
            auto value = take_value("--fonts");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.run_options.fonts = std::get<std::string_view>(value);
            saw_fonts = true;
            continue;
        }
        if (argument == "--audio") {
            if (command.run_options.audio) return usage("--audio was given more than once.");
            command.run_options.audio = true;
            continue;
        }
        if (argument.starts_with("-")) return usage("Unknown option: " + std::string(argument));
        if (saw_game) return usage("Only one game path can be given.");
        if (argument.empty()) return usage("A game file or folder is required.");
        command.game = argument;
        saw_game = true;
    }
    if (!saw_game) return usage("A game file or folder is required.");
    if (saw_width != saw_height) return usage("--width and --height must be given together.");
    return command;
}

} // namespace

std::string command_usage() {
    return "Usage: armigo [pack] <game.xci|nsp|nup or folder> [--update FILE]... [--dlc FILE]...\n"
           "             --keys FILE --firmware PATH [--title-keys FILE] [--output DIR]\n"
           "             [--platform macos|ios] [--reuse APP] [--ios-build DIR] [--keep-app] [--no-ipa] [--development] [--development-metal-hud]\n"
           "       armigo [pack] <game> [--update FILE]... [--dlc FILE]... --keys FILE [--title-keys FILE]\n"
           "             --shaders-only --output DIR [--restamp OLD_PACKAGE]\n"
           "       armigo [pack] --compare OLD_PACKAGE NEW_PACKAGE\n"
           "       armigo run <App.app|game> [--keys FILE] [--frames N] [...]\n";
}

std::variant<CommandFailure, Command> parse_command(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "run") return parse_run(argc, argv);
    Command command;
    bool saw_game = false;
    bool saw_keys = false;
    bool saw_title_keys = false;
    bool saw_output = false;
    bool saw_platform = false;
    bool saw_reuse = false;
    bool saw_ios_build = false;
    bool saw_firmware = false;
    // "pack" is the default verb and may be spelled out.
    const int first = argc > 1 && std::string_view(argv[1]) == "pack" ? 2 : 1;
    for (int index = first; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto take_value = [&](std::string_view flag) -> std::variant<CommandFailure, std::string_view> {
            const auto equals = argument.find('=');
            if (equals != std::string_view::npos) {
                const auto value = argument.substr(equals + 1);
                if (value.empty()) {
                    return usage(std::string(flag) + " requires a value.");
                }
                return value;
            }
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty()) {
                return usage(std::string(flag) + " requires a value.");
            }
            return std::string_view(argv[++index]);
        };
        if (argument == "--help" || argument == "-h") {
            command.help = true;
            return command;
        }
        if (argument == "--update" || argument.starts_with("--update=")) {
            auto value = take_value("--update");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.updates.emplace_back(std::get<std::string_view>(value));
            continue;
        }
        if (argument == "--dlc" || argument.starts_with("--dlc=")) {
            auto value = take_value("--dlc");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.dlc.emplace_back(std::get<std::string_view>(value));
            continue;
        }
        if (argument == "--keys" || argument.starts_with("--keys=")) {
            if (saw_keys) {
                return usage("--keys was given more than once.");
            }
            auto value = take_value("--keys");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.keys = std::get<std::string_view>(value);
            saw_keys = true;
            continue;
        }
        if (argument == "--title-keys" || argument.starts_with("--title-keys=")) {
            if (saw_title_keys) {
                return usage("--title-keys was given more than once.");
            }
            auto value = take_value("--title-keys");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.title_keys = std::get<std::string_view>(value);
            saw_title_keys = true;
            continue;
        }
        if (argument == "--output" || argument.starts_with("--output=")) {
            if (saw_output) {
                return usage("--output was given more than once.");
            }
            auto value = take_value("--output");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.output = std::get<std::string_view>(value);
            saw_output = true;
            continue;
        }
        if (argument == "--reuse" || argument.starts_with("--reuse=")) {
            if (saw_reuse) {
                return usage("--reuse was given more than once.");
            }
            auto value = take_value("--reuse");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.reuse = std::get<std::string_view>(value);
            saw_reuse = true;
            continue;
        }
        if (argument == "--ios-build" || argument.starts_with("--ios-build=")) {
            if (saw_ios_build) {
                return usage("--ios-build was given more than once.");
            }
            auto value = take_value("--ios-build");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.ios_build = std::get<std::string_view>(value);
            saw_ios_build = true;
            continue;
        }
        if (argument == "--firmware" || argument.starts_with("--firmware=")) {
            if (saw_firmware) return usage("--firmware was given more than once.");
            auto value = take_value("--firmware");
            if (auto* failure = std::get_if<CommandFailure>(&value)) return *failure;
            command.firmware = std::get<std::string_view>(value);
            saw_firmware = true;
            continue;
        }
        if (argument == "--development") {
            if (command.development) return usage("--development was given more than once.");
            command.development = true;
            continue;
        }
        if (argument == "--development-metal-hud") {
            if (command.development_metal_hud) return usage("--development-metal-hud was given more than once.");
            command.development_metal_hud = true;
            continue;
        }
        if (argument == "--no-ipa") {
            if (command.no_ipa) {
                return usage("--no-ipa was given more than once.");
            }
            command.no_ipa = true;
            continue;
        }
        if (argument == "--compare") {
            if (!command.compare.empty()) {
                return usage("--compare was given more than once.");
            }
            if (index + 2 >= argc || std::string_view(argv[index + 1]).empty() ||
                std::string_view(argv[index + 2]).empty()) {
                return usage("--compare requires two shader packages.");
            }
            command.compare = {argv[index + 1], argv[index + 2]};
            index += 2;
            continue;
        }
        if (argument == "--restamp" || argument.starts_with("--restamp=")) {
            if (!command.restamp.empty()) {
                return usage("--restamp was given more than once.");
            }
            auto value = take_value("--restamp");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.restamp = std::get<std::string_view>(value);
            continue;
        }
        if (argument == "--shaders-only") {
            if (command.shaders_only) {
                return usage("--shaders-only was given more than once.");
            }
            command.shaders_only = true;
            continue;
        }
        if (argument == "--keep-app") {
            if (command.keep_app) {
                return usage("--keep-app was given more than once.");
            }
            command.keep_app = true;
            continue;
        }
        if (argument == "--platform" || argument.starts_with("--platform=")) {
            if (saw_platform) {
                return usage("--platform was given more than once.");
            }
            auto value = take_value("--platform");
            if (auto* failure = std::get_if<CommandFailure>(&value)) {
                return *failure;
            }
            command.platform = std::string(std::get<std::string_view>(value));
            if (command.platform != "macos" && command.platform != "ios") {
                return usage("Unknown platform '" + command.platform + "'. Expected macos or ios.");
            }
            saw_platform = true;
            continue;
        }
        if (argument.starts_with("-")) {
            return usage("Unknown option: " + std::string(argument));
        }
        if (saw_game) {
            return usage("Only one game path can be given.");
        }
        command.game = argument;
        saw_game = true;
    }
    if (!command.compare.empty()) {
        if (saw_game || saw_keys || saw_firmware || command.shaders_only || saw_output || !command.restamp.empty() || command.development || command.development_metal_hud) {
            return usage("--compare takes only the two shader packages.");
        }
        return command;
    }
    if (!saw_game) {
        return usage("A game file or folder is required.");
    }
    if (!saw_keys) {
        return usage("--keys is required.");
    }
    if (command.shaders_only) {
        if (command.development || command.development_metal_hud)
            return usage("Development options require an application pack.");
        if (!saw_output) {
            return usage("--shaders-only requires --output.");
        }
        if (saw_platform && command.platform != "macos") {
            return usage("--shaders-only builds the macOS shader package.");
        }
    } else if (!saw_firmware) {
        return usage("--firmware is required.");
    }
    if (command.platform != "ios" && (saw_reuse || saw_ios_build || command.keep_app || command.no_ipa || command.development_metal_hud)) {
        return usage("--reuse, --ios-build, --keep-app, --no-ipa and --development-metal-hud require --platform ios.");
    }
    return command;
}

} // namespace armigo::pack
