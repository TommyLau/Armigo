// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "space.h"

#include <sys/statvfs.h>

#include <system_error>
#include <unordered_set>

namespace armigo::pack {
namespace {

std::filesystem::path existing_ancestor(std::filesystem::path path) {
    std::error_code error;
    while (!path.empty() && !std::filesystem::exists(path, error)) {
        const auto parent = path.parent_path();
        if (parent == path) {
            return {};
        }
        path = parent;
    }
    return path;
}

} // namespace

bool space_is_enough(std::uint64_t available, std::uint64_t expected) noexcept {
    return available >= expected;
}

std::variant<std::string, std::uint64_t> expected_pack_bytes(std::uint64_t content_bytes) {
    if (content_bytes > UINT64_MAX - kPackMarginBytes) {
        return std::string("The expected pack size overflows.");
    }
    return content_bytes + kPackMarginBytes;
}

std::variant<std::string, std::uint64_t> sum_content_bytes(const std::vector<ContentSize>& items) {
    std::uint64_t sum = 0;
    std::unordered_set<std::string> fallback_files;
    for (const ContentSize& item : items) {
        std::uint64_t bytes = item.content_bytes;
        if (bytes == 0) {
            std::error_code error;
            const auto canonical = std::filesystem::weakly_canonical(item.file, error);
            const std::string key = error ? item.file.string() : canonical.string();
            if (!fallback_files.insert(key).second) {
                continue;
            }
            const auto size = std::filesystem::file_size(item.file, error);
            if (error) {
                return "Cannot read the size of " + item.file.string() + ".";
            }
            bytes = static_cast<std::uint64_t>(size);
        }
        if (sum > UINT64_MAX - bytes) {
            return std::string("The content size overflows.");
        }
        sum += bytes;
    }
    return sum;
}

bool inside_git_work_tree(const std::filesystem::path& path) {
    std::error_code error;
    auto probe = existing_ancestor(path);
    if (probe.empty()) {
        return false;
    }
    probe = std::filesystem::weakly_canonical(probe, error);
    if (error) {
        probe = existing_ancestor(path);
    }
    while (true) {
        if (std::filesystem::exists(probe / ".git", error)) {
            return true;
        }
        const auto parent = probe.parent_path();
        if (parent == probe) {
            return false;
        }
        probe = parent;
    }
}

SpaceCheck check_free_space(const std::filesystem::path& output, std::uint64_t expected) {
    SpaceCheck check;
    check.expected = expected;
    const auto probe = existing_ancestor(output);
    if (probe.empty()) {
        check.error = "The output path does not exist and has no existing parent.";
        return check;
    }
    struct statvfs volume {};
    if (statvfs(probe.c_str(), &volume) != 0 || volume.f_frsize == 0) {
        check.error = "Cannot read free space for " + probe.string() + ".";
        return check;
    }
    const auto block = static_cast<std::uint64_t>(volume.f_frsize);
    const auto blocks = static_cast<std::uint64_t>(volume.f_bavail);
    if (block != 0 && blocks > UINT64_MAX / block) {
        check.available = UINT64_MAX;
    } else {
        check.available = blocks * block;
    }
    check.enough = space_is_enough(check.available, expected);
    return check;
}

} // namespace armigo::pack
