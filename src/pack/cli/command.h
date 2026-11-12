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

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

// Options of `armigo run`. Paths are empty when the flag was omitted.
struct RunOptions {
    std::filesystem::path keys;
    std::filesystem::path title_keys;
    std::filesystem::path update;
    std::filesystem::path shaders;
    std::uint64_t frames = 600;
    std::string time_limit = "300";
    std::string rate;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::filesystem::path saves;
    std::filesystem::path report;
    // True when --audio is set. The launch is muted unless this is set.
    bool audio = false;
    std::filesystem::path fonts;
};

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
    // Explicitly enables development-only launch controls in a packed player.
    bool development = false;
    // Development iOS pack: enable a user-switchable Metal performance HUD.
    bool development_metal_hud = false;
    // Writes only the shader package (output/shaders) and the pipeline
    // archive (output/pipelines) into an explicit output folder, new or
    // empty; no app, no firmware.
    bool shaders_only = false;
    // --compare OLD NEW: report the stage-by-stage difference of two shader
    // packages; takes no game.
    std::vector<std::filesystem::path> compare;
    // --restamp OLD_PACKAGE: a stage whose Metal source matches one the
    // earlier package compiled with the same compiler takes its library.
    std::filesystem::path restamp;
    // `armigo run`: headless launch. Pack-only options are rejected.
    bool run = false;
    RunOptions run_options;
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
