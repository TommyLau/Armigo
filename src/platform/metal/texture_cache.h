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

#include <armigo/graphics/texture_cache.h>

namespace armigo::platform::metal {

// L4 owns the Objective-C++ implementation. DeviceContext retains the Metal
// device and residency domain; its definition is native-only and shared through
// the integrator. Budgets are explicit; absent device/memory or zero budget is
// a structured error. The returned cache obeys the portable coherence contract,
// retains native objects through completion and never silently resolves reads
// after every draw. L1 supplies declarations only, not a successful stub.
class DeviceContext;
[[nodiscard]] graphics::RendererResult<std::shared_ptr<graphics::TextureCache>>
create_texture_cache(std::shared_ptr<DeviceContext> device,
                   std::shared_ptr<graphics::ResourceMemory> memory,
                   std::uint64_t budget_bytes);

// Narrow performance diagnostic used by the synthetic cache tests. A fast hit
// has repeated live descriptor, mapping, CPU-write and GPU-authority admission;
// ordinary compatible allocation reuse is not counted here.
[[nodiscard]] std::uint64_t texture_cache_fast_path_hits(
    const graphics::TextureCache& cache) noexcept;

} // namespace armigo::platform::metal
