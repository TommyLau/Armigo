// ============================================================================
// Armigo Runtime - Native Process Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/runtime/launch_result.h>
#include <armigo/services/addon_content_catalog.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace armigo::runtime {

// The add-on content base id of an application. Firmware takes it from the
// application's control property; extracted input carries no control data, so
// the runtime uses the application id plus 0x1000, the base that add-on
// content data ids are allocated from.
constexpr std::uint64_t addon_content_base_id(std::uint64_t application_id) noexcept {
    return application_id + 0x1000;
}

// Builds the application's add-on content catalog from extracted add-on
// directories. Each directory is one item's RomFS tree, named by the item's
// data id as exactly sixteen hexadecimal digits, or a romfs subdirectory of
// that named directory. The id must lie in [base + 1, base + 0xfff] for the
// application's base id, and appear once.
// No directories gives an empty catalog with the base id. A directory that is
// misnamed, belongs to another application, repeats an id or cannot be opened
// as RomFS fails the launch with an inputs-stage failure naming it.
LaunchResult<std::shared_ptr<const services::AddOnContentCatalog>> open_addon_content(
    std::uint64_t application_id, std::span<const std::filesystem::path> directories);

}  // namespace armigo::runtime
