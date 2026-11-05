// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Readers for shader stages stored in CRI CPK archives and GFS0/GSC0 caches.
// A cache record of type 0 or 1 holds one HHNEEDLE package of NVN program
// descriptors. The same package also occurs inside a main module image.
#pragma once

#include <armigo/shader_input/shader_input.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader_input {

struct GfdCacheError {
    std::string message;
};

struct OwnedShaderStage {
    std::string source;
    Stage stage = Stage::Vertex;
    std::vector<std::byte> control;
    std::vector<std::byte> code;
};

struct CpkMember {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t stored = 0;
    std::uint64_t extracted = 0;
};

struct CpkListing {
    std::vector<CpkMember> members;
};

using CpkListResult = std::variant<CpkListing, GfdCacheError>;
using GfdBytesResult = std::variant<std::vector<std::byte>, GfdCacheError>;
using GfdStagesResult = std::variant<std::vector<OwnedShaderStage>, GfdCacheError>;

// Resource policy, not a container-format limit.
inline constexpr std::size_t gfd_member_limit = 256ull * 1024ull * 1024ull;
inline constexpr std::size_t gfd_table_limit = 64ull * 1024ull * 1024ull;

// Lists every TOC member. The file is read only for its header and table of
// contents; member payloads stay on disk.
[[nodiscard]] CpkListResult list_cpk(const std::filesystem::path& path);

// Reads one member and, when the payload is CRILAYLA, decodes it. The decoded
// length must equal `member.extracted`.
[[nodiscard]] GfdBytesResult read_cpk_member(const std::filesystem::path& path,
                                             const CpkMember& member);

[[nodiscard]] GfdBytesResult decompress_crilayla(std::span<const std::byte> input,
                                                 std::size_t limit = gfd_member_limit);

// Parses one GFS0/GSC0 shader cache. A record whose type is not a vertex or
// fragment stage is skipped. An empty payload is recorded and not interpreted.
[[nodiscard]] GfdStagesResult parse_gsc(std::span<const std::byte> input,
                                        std::string_view label);

struct NeedleScan {
    std::vector<OwnedShaderStage> stages;
    // HHNEEDLE occurrences that are not packages, except those whose offset is
    // listed in the ignored set.
    std::vector<std::string> rejected;
    // Offsets of shader-control magics that no extracted stage accounts for.
    std::vector<std::uint64_t> uncovered_controls;
};

[[nodiscard]] NeedleScan scan_hhneedle(std::span<const std::byte> image, std::string_view label,
                                       std::span<const std::uint64_t> ignored_offsets);

// Required cache names that are neither present nor satisfied by an alternative.
[[nodiscard]] std::vector<std::string> missing_shader_caches(
    const std::vector<std::string>& present, const std::vector<std::string>& required,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& fallbacks);

}  // namespace armigo::shader_input
