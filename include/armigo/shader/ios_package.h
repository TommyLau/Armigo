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

namespace armigo::shader::package {

struct IosBuildOptions {
    std::filesystem::path macos_package;
    std::filesystem::path output;
    std::optional<std::filesystem::path> compile_cache;
    unsigned jobs = 0;
};

struct IosBuildReport {
    std::size_t converted_stages = 0;
};

// Regenerates each packaged stage with the deterministic native translator,
// then compiles it for iOS 26 and Metal 4. Stage and variant identities are
// preserved; the package platform and chain describe the iPhoneOS toolchain.
[[nodiscard]] std::variant<std::string, IosBuildReport> build_ios_package(const IosBuildOptions& options);

} // namespace armigo::shader::package
