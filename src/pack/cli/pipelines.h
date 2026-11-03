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

#include <armigo/platform/pso_archive_build.h>

#include <filesystem>
#include <string>
#include <variant>

namespace armigo::pack {

// Builds the title's pipeline archive into `cache_root` (R/Armigo/pipelines).
[[nodiscard]] std::variant<std::string, armigo::platform::PipelineArchiveResult> build_pack_pipelines(
    const std::filesystem::path& package, const std::filesystem::path& manifest,
    const std::filesystem::path& cache_root, const std::string& expected_chain_id);

// Warms the packed player's own Metal caches against the archive already in
// the app, so its first launch sees the same cache state as a later launch.
[[nodiscard]] std::string warm_pack_pipelines(const std::filesystem::path& app,
                                              const std::filesystem::path& manifest);

} // namespace armigo::pack
