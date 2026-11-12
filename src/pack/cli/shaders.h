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

#include <armigo/compat/patches.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::pack {

struct ShaderReport {
    // Package libraries, and those taken from a reused package.
    std::size_t libraries = 0;
    std::size_t reused = 0;
    std::size_t enumerated = 0;
    std::size_t translated = 0;
    std::size_t unsupported = 0;
    std::size_t structural_errors = 0;
};

// The title's shader input settings, as its profile states them (see
// compat::ShaderInputSettings), and the NVN policy translation follows. Each
// container set has one reader: bnsh scans every file of the ExeFS, RomFS and
// add-on trees for BNSH and BAHS containers, also inside Yaz0 and SARC;
// gfd-cache reads the .gsc members of CPK archives under the RomFS and add-on
// trees, then the packages embedded in the main module (`caches` must be
// present, a fallback stands in for one, and failed packages at
// `ignored_needle_offsets` are not errors); glslc reads the GLSLC outputs
// compiled into the main module, every shader control of which must belong to
// one. Native inputs belong to a .nup archive and are refused here.
// `hoist_implicit_samples` follows the title's nvn_policy and, when set, is
// applied to every translated stage and recorded in the package manifest.
// That record is outside the profile and does not change the chain id.
struct ShaderPackInput {
    compat::ShaderInputSettings settings;
    bool hoist_implicit_samples = false;
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
    const std::filesystem::path& translate_tool, const ShaderPackInput& input = {},
    const std::vector<std::filesystem::path>& reuse = {});

} // namespace armigo::pack
