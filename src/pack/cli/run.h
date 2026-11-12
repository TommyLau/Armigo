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

#include "command.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::pack {

// One headless launch. `clones` are directory copies applied before the process
// starts, as (source, destination). `links` are directory symlinks, as
// (target, link). `directories` are created first; a clone destination is not
// listed there.
struct RunPlan {
    std::vector<std::string> arguments;
    std::vector<std::pair<std::string, std::string>> environment;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> clones;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> links;
    std::vector<std::filesystem::path> directories;
};

// Reads the game layout and records the launch. It does not create files,
// copy pipelines, or start the runner.
[[nodiscard]] std::variant<std::string, RunPlan> plan_run(const Command& command,
                                                          const std::filesystem::path& runner,
                                                          const std::filesystem::path& scratch);

// "Run: <outcome>, <frames> frames in <seconds> s, <fps> fps", or empty when
// the text is not a usable run report.
[[nodiscard]] std::string summarize_report(std::string_view report_json);

// Runs the plan through armigo-headless beside `executable` and prints the outcome.
[[nodiscard]] int run_game(const Command& command, const std::filesystem::path& executable);

} // namespace armigo::pack
