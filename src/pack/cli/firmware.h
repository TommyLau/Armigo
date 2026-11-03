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

#include "inputs.h"

#include <armigo/loader/container.h>

#include <filesystem>
#include <string>
#include <variant>

namespace armigo::pack {

struct FirmwareResources {
    std::string version;
};

std::variant<FirmwareResources, std::string> read_firmware_version(
    const std::filesystem::path& path);

// A folder is used directly. A zip is unpacked into work_directory after
// rejecting nested or non-NCA entry names; only encrypted NCAs enter staging.
std::variant<std::filesystem::path, std::string> prepare_firmware_input(
    const std::filesystem::path& input, const std::filesystem::path& work_directory);

// Extracts system version data for the report and the five shared-font data
// archives into output_directory/<SystemDataID>/ using the supplied keys.
std::variant<FirmwareResources, std::string> extract_firmware_resources(
    const Inputs& inputs, const std::filesystem::path& firmware_input,
    const loader::ContainerKeys& keys, const std::filesystem::path& output_directory);

} // namespace armigo::pack
