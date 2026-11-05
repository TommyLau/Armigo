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
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::pack {

struct ShaderReport {
    std::size_t enumerated = 0;
    std::size_t translated = 0;
    std::size_t unsupported = 0;
    std::size_t structural_errors = 0;
};

// Which containers hold the title's stages, and the vertex depth recorded in
// the package. The defaults select BNSH enumeration and a minus-one-to-one
// vertex depth. gfd-cache reads .gsc members of CPK archives under the RomFS
// and add-on trees, then the packages embedded in the main module. `caches`
// names members that must be present; an empty list accepts every .gsc member.
// A fallback satisfies one required name when an alternative member is present.
// `ignored_needle_offsets` are main-module image offsets whose failed packages
// are not errors. A successful package at an ignored offset is still kept.
struct ShaderPackInput {
    enum class Containers { Bnsh, GfdCache };
    enum class VertexDepth { NegativeOneToOne, ZeroToOne };

    Containers containers = Containers::Bnsh;
    VertexDepth vertex_depth = VertexDepth::NegativeOneToOne;
    std::vector<std::string> caches;
    std::vector<std::pair<std::string, std::vector<std::string>>> fallbacks;
    std::vector<std::uint64_t> ignored_needle_offsets;
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
    const std::filesystem::path& translate_tool, const ShaderPackInput& input = {});

} // namespace armigo::pack
