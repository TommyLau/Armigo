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

#include "device_context.h"
#include "metal_renderer.h"

namespace armigo::platform::metal {

// The provider invokes lookup for every renderer request. lookup_context is
// borrowed and must remain alive until the returned provider is destroyed;
// the caller also serializes callback access with its existing package owner.
// compute_lookup shares the same context and serves compute dispatches.
// immutable_stages promises that returned stage objects never change, so
// their digests are memoized per stage; otherwise every request rehashes.
[[nodiscard]] graphics::RendererResult<std::shared_ptr<GraphicsStageProvider>>
create_engine_stage_provider(platform::CompiledGraphicsLookup lookup,
                             void* lookup_context,
                             platform::CompiledComputeLookup compute_lookup = nullptr,
                             bool immutable_stages = false);

} // namespace armigo::platform::metal
