// ============================================================================
// Armigo Platform - Apple Platform Integration
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
#include <vector>

namespace armigo::platform {

// Builds, before a title runs, every pipeline its warm-up manifests name into a
// Metal binary archive for this shader package, OS build and GPU, so the
// driver takes those pipelines from the archive instead of compiling them.
struct PipelineArchiveBuild {
    std::filesystem::path package;
    // The translation chain the package must carry (the one launches accept).
    std::string expected_chain_id;
    // Warm-up manifests (seed rows); their rows are built in order, once each.
    std::vector<std::filesystem::path> manifests;
    // The pipeline cache root the driver reads at launch (its archives live in
    // a subdirectory named for the package, OS build and GPU, which a normal
    // build replaces).
    std::filesystem::path cache_root;
    // Load the existing archive and run the same warm-up without capturing or
    // replacing it, filling this process's Metal library and function caches.
    bool warm_only = false;
    unsigned threads = 0;   // 0: one per performance core
};

struct PipelineArchiveResult {
    bool ok = false;
    std::string error;
    std::uint64_t rows = 0, built = 0, unresolved = 0, undecodable = 0, failed = 0;
    std::filesystem::path archive;   // the last archive file written
    std::uint64_t archive_bytes = 0, archive_files = 0;   // all files of the archive directory
    double seconds = 0;
};

[[nodiscard]] PipelineArchiveResult build_pipeline_archive(const PipelineArchiveBuild& build);

} // namespace armigo::platform
