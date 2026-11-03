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

#import <Metal/Metal.h>
#include "pipeline_state.h"

namespace armigo::platform::metal {

// L10 constructs one retained native device/queue domain; L4/L5/L9 consume it.
// This header is Objective-C++ only and must never enter portable includes.
// Identity is stable and nonzero. No independent per-draw queue/device creation.
// Unsupported device features fail at creation, not via a substitute backend.
class DeviceContext {
public:
    virtual ~DeviceContext() = default;
    virtual id<MTLDevice> device() const noexcept = 0;
    virtual id<MTL4CommandQueue> queue() const noexcept = 0;
    virtual const PipelineContextKey& identity() const noexcept = 0;
};

// A nil preferred device selects the system default. The factory retains one
// Metal 4 queue for the complete context lifetime and assigns a stable nonzero
// identity used by every cache and pipeline key in the domain.
[[nodiscard]] graphics::RendererResult<std::shared_ptr<DeviceContext>>
create_device_context(id<MTLDevice> preferred_device = nil);

} // namespace armigo::platform::metal
