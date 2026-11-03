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

#include <armigo/graphics/draw_plan.h>
#include <armigo/graphics/resource_memory.h>

namespace armigo::platform::metal {

// Refines an admitted L8 plan before binding preparation. Index snapshots are
// owned and read under L2 exclusion; failure leaves the input plan unchanged.
// Native base vertex compensates index rebasing without changing vertex IDs.
// Fixed native restart is used only for strips; other topologies retain L8's
// restart segments. No implicit shader variant or wrapping arithmetic is used.
[[nodiscard]] graphics::RendererStatus select_native_draw(
    graphics::DrawPlan& plan, const graphics::RenderStop& stop,
    graphics::ResourceMemory& memory);

} // namespace armigo::platform::metal
