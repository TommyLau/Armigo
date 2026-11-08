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
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

// Stage-by-stage difference of two shader packages (manifest.json and
// stages/<variant>/...).
struct PackageComparison {
    // Manifest fields that differ, as "field: old -> new"; profile fields are
    // compared one by one, the entry lists by count.
    std::vector<std::string> fields;
    std::size_t old_stages = 0, new_stages = 0, common = 0, only_old = 0, only_new = 0;
    // Per file name in the common stages: how many are byte-identical and how
    // many differ; the first three differing paths per file name.
    std::map<std::string, std::size_t> identical, differing;
    std::vector<std::string> examples;
    [[nodiscard]] bool same() const;
};

[[nodiscard]] std::variant<std::string, PackageComparison> compare_packages(
    const std::filesystem::path& old_package, const std::filesystem::path& new_package);

// The comparison as text, one finding per line.
[[nodiscard]] std::string format_comparison(const PackageComparison& comparison);

} // namespace armigo::pack
