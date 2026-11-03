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

// Two gibibytes left for the shader package, the pipeline archive and the
// player after the game content is extracted.
inline constexpr std::uint64_t kPackMarginBytes = 2ull << 30;

struct ContentSize {
    std::filesystem::path file;
    // Program and data content-record bytes. Zero means the size is unknown
    // and the file size is used once per path.
    std::uint64_t content_bytes = 0;
};

struct SpaceCheck {
    std::uint64_t available = 0;
    std::uint64_t expected = 0;
    bool enough = false;
    std::string error;
};

[[nodiscard]] bool space_is_enough(std::uint64_t available, std::uint64_t expected) noexcept;

// Content bytes plus the packing margin. Empty when the sum overflows.
[[nodiscard]] std::variant<std::string, std::uint64_t> expected_pack_bytes(
    std::uint64_t content_bytes);

// Sums content-record sizes. A zero content size falls back to that file's
// size, counted once when several sets share the file.
[[nodiscard]] std::variant<std::string, std::uint64_t> sum_content_bytes(
    const std::vector<ContentSize>& items);

[[nodiscard]] bool inside_git_work_tree(const std::filesystem::path& path);

// Free space on the volume that holds `output` (or its nearest existing
// parent). `error` is set when the volume cannot be queried.
[[nodiscard]] SpaceCheck check_free_space(const std::filesystem::path& output,
                                          std::uint64_t expected);

} // namespace armigo::pack
