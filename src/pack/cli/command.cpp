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

#include <string_view>
#include <utility>

namespace armigo::pack {
namespace {

CommandFailure usage(std::string detail) {
    return CommandFailure{std::move(detail), 2};
}

} // namespace

std::string command_usage() {
    return "Usage: armigo <game.xci|nsp or folder> [--update FILE]... [--dlc FILE]...\n"
           "             --keys FILE --firmware PATH [--title-keys FILE] [--output DIR]\n"
           "             [--platform macos|ios] [--reuse APP] [--ios-build DIR] [--keep-app] [--no-ipa]\n";
}

std::variant<CommandFailure, Command> parse_command(int argc, char** argv) {
    Command command;
    bool saw_game = false;
    bool saw_keys = false;
    bool saw_title_keys = false;
    bool saw_output = false;
    bool saw_platform = false;
    bool saw_reuse = false;
    bool saw_ios_build = false;
    bool saw_firmware = false;
    for (int index = 1; index < argc; ++index) {
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
        if (argument == "--no-ipa") {
            if (command.no_ipa) {
                return usage("--no-ipa was given more than once.");
            }
            command.no_ipa = true;
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
    if (!saw_game) {
        return usage("A game file or folder is required.");
    }
    if (!saw_keys) {
        return usage("--keys is required.");
    }
    if (!saw_firmware) {
        return usage("--firmware is required.");
    }
    if (command.platform != "ios" && (saw_reuse || saw_ios_build || command.keep_app || command.no_ipa)) {
        return usage("--reuse, --ios-build, --keep-app and --no-ipa require --platform ios.");
    }
    return command;
}

} // namespace armigo::pack
