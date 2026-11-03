// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

struct Command {
    std::filesystem::path game;
    std::vector<std::filesystem::path> updates;
    std::vector<std::filesystem::path> dlc;
    std::filesystem::path keys;
    std::filesystem::path title_keys;
    std::filesystem::path output = ".";
    std::filesystem::path reuse;
    std::filesystem::path ios_build;
    std::filesystem::path firmware;
    std::string platform = "macos";
    bool keep_app = false;
    // iOS only: keep the unsigned app and its entitlements and write no IPA.
    bool no_ipa = false;
    bool help = false;
};

struct CommandFailure {
    std::string detail;
    int exit_code = 2;
};

[[nodiscard]] std::string command_usage();

// Exit 2 is a usage error. `--help` is a successful Command with help set.
[[nodiscard]] std::variant<CommandFailure, Command> parse_command(int argc, char** argv);

} // namespace armigo::pack
