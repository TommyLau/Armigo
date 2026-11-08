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

#include <armigo/shader_native/package.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

struct NativeShaderRequest {
    std::filesystem::path nup;
    std::filesystem::path output;
    std::filesystem::path translate_tool;
    std::filesystem::path compile_cache;
    std::string dictionary_member;
    // Empty selects every archive member. Each entry is a RomFS path or a
    // suffix of one, so a build can publish the programs one boot uses.
    std::vector<std::string> members;
    // Empty admits every program in the selected members. Each entry is a
    // 64-digit stage identity, so one large member can contribute only the
    // programs a boot prepares.
    std::vector<std::string> identities;
    unsigned jobs = 8;
    // Admit and count programs, and do not translate or compile them.
    bool enumerate_only = false;
    // Translate every admitted program, including task and mesh, and do not
    // compile Metal libraries or publish stage files.
    bool translate_only = false;
    // Write each stage's Metal source next to its library; a packed app keeps only the digest.
    bool keep_sources = true;
};

struct NativeShaderReport {
    std::uint64_t enumerated = 0;
    std::uint64_t unique = 0;
    std::uint64_t translated = 0;
    std::uint64_t compiled = 0;
    std::uint64_t duplicates = 0;
    std::map<std::string, std::uint64_t> by_stage;
    std::map<std::string, std::uint64_t> translated_by_stage;
    std::map<std::string, std::uint64_t> compiled_by_stage;
    std::map<std::string, std::uint64_t> refusals;
    // "<stage identity> <stage> <member>" for each admitted program, in admission order.
    std::vector<std::string> admitted;
};

// Streams a .nup archive, translates each native program the front end admits,
// and publishes an empty Metal package shell with a native shader section.
// Refusals are counted. A container, member or traversal failure fails the
// build. The output must be new or empty and outside a Git work tree.
[[nodiscard]] std::variant<std::string, NativeShaderReport> build_native_shader_package(
    const NativeShaderRequest& request);

} // namespace armigo::pack
