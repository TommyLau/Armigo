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

#include <cstddef>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

struct ShaderReport {
    std::size_t enumerated = 0;
    std::size_t translated = 0;
    std::size_t unsupported = 0;
    std::size_t structural_errors = 0;
};

// Enumerates shaders from the extracted ExeFS, RomFS and add-on RomFS trees,
// translates them and publishes the Metal package at `package`. `staging` is
// removed when the call returns. `compile_cache` optionally retains compiled
// libraries; the environment override is handled by the package library.
// `translate_tool` is the armigo-shader-translate built with this tree.
[[nodiscard]] std::variant<std::string, ShaderReport> build_shader_package(
    const std::filesystem::path& exefs, const std::filesystem::path& romfs,
    const std::vector<std::filesystem::path>& addons, const std::filesystem::path& staging,
    const std::filesystem::path& package, const std::filesystem::path& compile_cache,
    const std::filesystem::path& translate_tool);

} // namespace armigo::pack
