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

#include <armigo/platform/metal_engine_renderer.h>
#include "metal_renderer.h"
#include <functional>

namespace armigo::platform::metal {

// L9 integration seam: inject a cache using the frozen PipelineStateCache and
// this exact device context. The default factory is the L9 native cache.
// Tests may inject a cache without changing the portable interface.
using EnginePipelineFactory = std::function<graphics::RendererResult<
    std::shared_ptr<PipelineStateCache>>(std::shared_ptr<DeviceContext>, std::uint64_t)>;
[[nodiscard]] graphics::RendererResult<graphics::RenderCallbacks>
create_engine_backend(MetalEngineOptions options, EnginePipelineFactory pipelines,
                      std::shared_ptr<SubmissionTestControl> test_control = {});

} // namespace armigo::platform::metal
