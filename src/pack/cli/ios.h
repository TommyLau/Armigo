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
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

struct PackedIdentity {
    std::string title_id;
    std::string build_id;
    std::string platform;
    std::uint32_t title_version = 0;
};

struct ShaderStagePaths {
    std::filesystem::path enumeration;
    std::filesystem::path macos_package;
};

[[nodiscard]] ShaderStagePaths shader_stage_paths(
    const std::filesystem::path& staging, const std::filesystem::path& resource_root,
    const std::filesystem::path& reused_app, bool ios);

[[nodiscard]] std::variant<std::string, PackedIdentity> read_packed_identity(
    const std::filesystem::path& macos_app);

[[nodiscard]] std::variant<std::string, std::optional<std::filesystem::path>> select_macos_reuse(
    const std::filesystem::path& requested, const std::filesystem::path& output_directory,
    const std::filesystem::path& destination_app, const std::string& title_id);

[[nodiscard]] std::variant<std::string, PackedIdentity> clone_macos_game(
    const std::filesystem::path& macos_app, const std::filesystem::path& destination_game,
    const std::string& expected_title_id);

// Directories under game/addons. A missing add-ons directory is zero add-ons.
// A dangling link, a non-directory, or an unreadable directory is an error.
[[nodiscard]] std::variant<std::string, std::vector<std::filesystem::path>> reused_addon_directories(
    const std::filesystem::path& game);

[[nodiscard]] std::string validate_reused_selection(
    const PackedIdentity& identity, const std::vector<std::filesystem::path>& reused_addons,
    const std::string& title_id, const std::string& build_id, std::uint32_t title_version,
    const std::vector<std::string>& selected_addons);

[[nodiscard]] std::variant<std::string, std::filesystem::path> prepare_ios_player_build(
    const std::filesystem::path& requested, const std::filesystem::path& packer_executable,
    const std::filesystem::path& host_shader_translator);

// Embedded capture and verification both use the title default, independent of
// the packer's inherited launch-mode override. The parent environment is unchanged.
[[nodiscard]] std::string run_pack_tool(const std::vector<std::string>& arguments,
                                      bool default_title_rate = false);
// Writes <app_path stem>.ipa beside app_path from the app assembled at
// `assembled`, which the IPA's Payload names by app_path's file name.
[[nodiscard]] std::string create_ios_ipa(const std::filesystem::path& assembled,
                                         const std::filesystem::path& app_path);

} // namespace armigo::pack
