// ============================================================================
// Armigo Services - Operation Mode Profile
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>

namespace armigo::services {

enum class OperationMode : std::uint8_t {
    Handheld = 0,
    Console = 1,
};

struct OperationModeProfile {
    std::uint32_t performance_mode;
    std::uint32_t display_width;
    std::uint32_t display_height;
};

constexpr OperationModeProfile operation_mode_profile(OperationMode mode) noexcept {
    switch (mode) {
    case OperationMode::Handheld:
        return {0, 1280, 720};
    case OperationMode::Console:
        return {1, 1920, 1080};
    }
    return {1, 1920, 1080};
}

} // namespace armigo::services
