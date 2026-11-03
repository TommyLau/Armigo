// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/loader/exefs_directory.h>
#include <armigo/runtime/launch_result.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace armigo::runtime {

// Lane B owns deterministic offline preparation and its versioned payload
// encoding. This envelope is host adaptation, not firmware semantics.
// A load does not invent imports, a gate patch covers one selected site, and
// reachability is an inventory rather than permission to run natively.
// Payloads contain module-relative preimages, patches, coverage and fixups,
// never absolute host addresses or SDK replacements. Runtime validates identity,
// bounds, overlaps and all preimages before publication; unknown schemas,
// stale artifacts and unproven execution coverage fail, never trigger a whole
// text rescan or treat literal data as instructions. No private role files.
struct ModuleLocation {
    std::size_t module_index = 0;
    std::uint64_t image_offset = 0;
};
struct PreparedModuleArtifact {
    std::string module_name;
    // SHA-256 of the complete NSO input, not its load address or file name.
    std::array<std::byte, 32> source_sha256{};
    std::vector<std::byte> payload;
};
struct PreparedModuleArtifacts {
    // Zero is uninitialized and rejected. Lane B defines supported encoding
    // versions; this C++ object is not serialized with memcpy.
    std::uint32_t format_version = 0;
    std::string preparation_identity;
    // Ordered manifest of admitted modules. Ignored subsdk orphans have no index.
    std::vector<PreparedModuleArtifact> modules;
    // Input admission diagnostics, reconstructed when reading an artifact.
    std::vector<std::string> ignored_modules;
};
enum class PreparationMode { automatic, original_process, synthetic_test };
struct PreparationOptions {
    // Expected identity comes from the installed build/tool, not the artifact.
    std::string required_preparation_identity;
    std::size_t max_artifact_bytes = 0;
    // Automatic recognizes the original slot roster even without NPDM.
    // Synthetic fixtures must opt in when they use original roster names.
    // During artifact reading, automatic accepts the stored execution mode.
    PreparationMode mode = PreparationMode::automatic;
};
[[nodiscard]] LaunchResult<PreparedModuleArtifacts> prepare_modules(
    const loader::ExeFsDirectoryInput& input, const PreparationOptions& options);
[[nodiscard]] LaunchResult<PreparedModuleArtifacts> read_prepared_modules(
    const std::filesystem::path& directory, const loader::ExeFsDirectoryInput& input,
    const PreparationOptions& options);
[[nodiscard]] LaunchStatus write_prepared_modules(
    const std::filesystem::path& directory, const PreparedModuleArtifacts& artifacts);

} // namespace armigo::runtime
