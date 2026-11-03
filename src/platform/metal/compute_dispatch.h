// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Compute dispatch admits QMD constant buffers, decodes storage descriptors,
// and validates dispatch shape for Armigo's synchronous Metal 4 backend.

#pragma once

#include <armigo/graphics/engines.h>
#include <armigo/platform/metal_draw.h>
#include <cstdint>

namespace armigo::platform {

struct MetalComputeStatistics {
    std::uint64_t completed_dispatches = 0;
};

// Borrowed for the synchronous call, exactly like MetalDrawContext.
struct MetalComputeContext {
    // Resolves the compiled compute stage for the stop. A miss must be a
    // structured CommandResult naming the program and sizes.
    CompiledComputeLookup lookup = nullptr;
    void* lookup_context = nullptr;
    MetalComputeStatistics* statistics = nullptr;
    std::uint64_t completion_timeout_ns = 30'000'000'000ull;
};

// Validates a compute stop's launch shape without GPU work: grid/block
// dimensions, device threadgroup limits and reflected binding namespaces.
[[nodiscard]] graphics::CommandResult validate_metal_compute_state(
    const graphics::RenderStop& snapshot) noexcept;

// Synchronous GPU completion and guest writeback for one compute dispatch.
// Borrowed memory never escapes; the caller serializes guest access during
// the call. Success means every storage-buffer result is guest-visible.
[[nodiscard]] graphics::CommandResult execute_metal_compute(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept;

}  // namespace armigo::platform
