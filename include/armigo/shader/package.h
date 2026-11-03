// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::package {

struct BuildOptions {
    std::vector<std::filesystem::path> stages;
    std::filesystem::path translate_tool;
    std::filesystem::path output;
    std::optional<std::filesystem::path> compile_cache;
    unsigned jobs = 0;
};

struct BuildReport {
    std::size_t input_stages = 0;
    std::size_t unique_stages = 0;
};

// Validates and compiles translated stages, then atomically publishes the
// package at `output`. The destination must be new or empty and outside a Git
// work tree. Apple tool commands are bounded to 60 seconds each.
[[nodiscard]] std::variant<std::string, BuildReport> build(const BuildOptions& options);

} // namespace armigo::shader::package
