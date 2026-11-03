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
#include "native_resources.h"
#include "pipeline_state.h"

namespace armigo::platform::metal {

class PipelineCacheCharge;

// Typed native leases retain state through submission even after cache eviction.
struct NativePipelineLease final : MetalRenderPipelineResource {
    std::shared_ptr<PipelineCacheCharge> charge;
    id<MTLRenderPipelineState> state = nil;
    id<MTLLibrary> vertex_library = nil, fragment_library = nil;
    id<MTLRenderPipelineState> pipeline() const noexcept override { return state; }
};
struct NativeComputePipelineLease final : MetalComputePipelineResource {
    std::shared_ptr<PipelineCacheCharge> charge;
    id<MTLComputePipelineState> state = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipeline() const noexcept override { return state; }
};
struct NativeDepthStencilLease final : MetalDepthStencilResource {
    std::shared_ptr<PipelineCacheCharge> charge;
    id<MTLDepthStencilState> state = nil;
    id<MTLDepthStencilState> depth_stencil() const noexcept override { return state; }
};
struct NativeSamplerLease final : MetalSamplerResource {
    std::shared_ptr<PipelineCacheCharge> charge;
    id<MTLSamplerState> state = nil;
    id<MTLSamplerState> sampler() const noexcept override { return state; }
};

[[nodiscard]] graphics::RendererResult<std::shared_ptr<PipelineStateCache>>
create_pipeline_state_cache(std::shared_ptr<DeviceContext> device_context,
                            std::uint64_t budget_bytes);

} // namespace armigo::platform::metal
