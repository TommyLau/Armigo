// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/prepared_image.h>
#include <armigo/runtime/prepared_modules.h>
#include <armigo/runtime/preparation_format.h>
#include <armigo/linker/dynamic_module.h>
#include <armigo/loader/nso_module.h>

namespace armigo::runtime::preparation {

struct ModulePayload {
    bool original_process = false;
    std::uint32_t load_rank = 0;
    std::uint64_t text_offset = 0;
    std::uint64_t text_size = 0;
    std::vector<std::uint64_t> roots;
    execution::PreparedImage image;
};

struct DecodedInput {
    std::vector<loader::NsoModuleImage> images;
    std::vector<linker::DynamicModule> metadata;
    std::vector<std::size_t> order;
    bool original_process = false;
};

LaunchFailure failure(LaunchError code, std::string operation, std::string detail);
std::optional<std::vector<std::size_t>> original_module_order(
    const loader::ExeFsDirectoryInput& input);
// Returns a compact copy only when original roster admission ignores orphans.
std::optional<loader::ExeFsDirectoryInput> admitted_input(
    const loader::ExeFsDirectoryInput& input, PreparationMode mode);
LaunchResult<DecodedInput> decode_input(const loader::ExeFsDirectoryInput& input,
    PreparationMode mode = PreparationMode::automatic);
LaunchResult<std::vector<ModulePayload>> validate_artifacts(
    const loader::ExeFsDirectoryInput& input, const PreparedModuleArtifacts& artifacts,
    const PreparationOptions& options = {});
LaunchResult<ModulePayload> decode_payload(std::span<const std::byte> payload);
bool covers(const execution::PreparedImage& image, std::uint64_t offset) noexcept;

} // namespace armigo::runtime::preparation
