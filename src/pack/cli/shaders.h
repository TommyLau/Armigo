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
// glslc reads the GLSLC compiler outputs compiled into the main module; every
// shader control in that image must belong to one of them.
// `hoist_implicit_samples` follows the title's nvn_policy and, when set, is
// applied to every translated stage and recorded in the package manifest.
// That record is outside the profile and does not change the chain id.
struct ShaderPackInput {
    enum class Containers { Bnsh, GfdCache, Glslc };
    enum class VertexDepth { NegativeOneToOne, ZeroToOne };

    Containers containers = Containers::Bnsh;
    VertexDepth vertex_depth = VertexDepth::NegativeOneToOne;
    bool hoist_implicit_samples = false;
    std::vector<std::string> caches;
    std::vector<std::pair<std::string, std::vector<std::string>>> fallbacks;
    std::vector<std::uint64_t> ignored_needle_offsets;
};

// The distinct vertex, fragment and compute stages the input's containers
// hold, before translation: each is written to `stages` (which must exist) as
// <control SHA-256>-<code SHA-256>/control.bin and code.bin. A hard failure
// is the error string; structural notes are counted, the first few kept.
struct ShaderEnumeration {
    std::size_t unique = 0;
    std::size_t unsupported = 0;
    std::size_t structural_errors = 0;
    std::vector<std::string> errors;
};
[[nodiscard]] std::variant<std::string, ShaderEnumeration> enumerate_shader_stages(
    const std::filesystem::path& exefs, const std::filesystem::path& romfs,
    const std::vector<std::filesystem::path>& addons, const std::filesystem::path& stages,
    const ShaderPackInput& input = {});

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
