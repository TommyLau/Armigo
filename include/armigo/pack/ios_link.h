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

namespace armigo::pack {

struct IosLinkOptions {
    // An already configured ios-release build of this source revision, with
    // ARMIGO_HOST_SHADER_TRANSLATE pointing at the matching macOS tool.
    std::filesystem::path player_build_directory;
    // Output from armigo-prepare --embedded, captured AND verified with the
    // same title profile that will be installed in Armigo/title/profile.json.
    std::filesystem::path embedded_directory;
    // Private generated sources, outside the repository. Must remain present
    // until linking finishes; image files are referenced by assembly incbin.
    std::filesystem::path work_directory;
    // The title-specific executable path; independent of the generic build target name.
    std::filesystem::path output_executable;
};

struct IosLinkReport {
    std::size_t modules = 0;
    std::size_t executable_bytes = 0;
};

// Emits embedded.S and embedded.cpp into output_directory. One player configure
// consumes them through ARMIGO_IOS_EMBEDDED_SOURCES_DIR in its environment.
// Only page-aligned code pieces go in __TEXT; original offsets are recorded
// in the backing table.
// Data/key pages stay non-executable and are privately remapped by the loader.
[[nodiscard]] std::variant<std::string, IosLinkReport> generate_ios_embedded_sources(
    const std::filesystem::path& embedded_directory,
    const std::filesystem::path& output_directory);

// Generates the sources, configures the supplied build with their directory for
// that invocation, builds armigo-ios-player, checks its arm64/iOS 26 Mach-O
// identity and exactly one executable segment, then publishes output_executable.
// The directory is removed from the CMake cache, and the build is configured
// again with no generated sources, on success and on failure. No signing tool
// is invoked. The caller owns extraction, settings, preparation and bundle output.
[[nodiscard]] std::variant<std::string, IosLinkReport> link_ios_player(const IosLinkOptions& options);

} // namespace armigo::pack
