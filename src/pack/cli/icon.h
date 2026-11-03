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

#include <armigo/pack/nacp.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::pack {

struct ControlIcon {
    std::string_view name;
    std::span<const std::byte> bytes;
};

// The icon file the control-data naming library chooses for `language`.
// The returned views point at the matching entry of `icons`.
[[nodiscard]] std::variant<std::string, ControlIcon> select_control_icon(
    std::span<const ControlIcon> icons, nacp::Language language);

// Converts a JPEG to an icns image with ImageIO. No external tool is run.
[[nodiscard]] std::variant<std::string, std::vector<std::byte>> jpeg_to_icns(
    std::span<const std::byte> jpeg);

// Converts control JPEG data to an opaque 1024-pixel universal iOS PNG.
[[nodiscard]] std::variant<std::string, std::vector<std::byte>> jpeg_to_ios_png(
    std::span<const std::byte> jpeg);

} // namespace armigo::pack
