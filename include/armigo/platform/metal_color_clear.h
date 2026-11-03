// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/engines.h>
#include <cstdint>

namespace armigo::platform {

// Counts fully successful GPU clears whose guest writeback also completed.
// The caller serializes access when sharing statistics across driver clients.
struct MetalClearStatistics {
    std::uint64_t completed_clears = 0;
    std::uint64_t written_pixels = 0;
};

// Optional context points to caller-owned statistics, accessed synchronously.
// Configuration records are retained but unresolved records reject execution.
[[nodiscard]] graphics::CommandResult execute_metal_color_clear(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept;

}  // namespace armigo::platform
