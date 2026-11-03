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

#include <armigo/graphics/renderer_types.h>
#include <armigo/graphics/resource_memory_registry.h>

namespace armigo::platform::metal {

// Returns L5's native allocation behind an admitted buffer slice. The caller
// retains storage through completion; nil identifies a non-Metal lease.
[[nodiscard]] id<MTLBuffer> resident_native_buffer(
    const graphics::RetainedResource& storage);

// A range of guest memory bound through a zero-copy Metal buffer over the
// backing's host pages. Every memory pool is a newBufferWithBytesNoCopy view,
// and buffer bindings are expressed as pool plus offset. GPU writes land in
// guest memory, so as a write authority it is
// in place: resolving waits for the writer and copies nothing.
class GuestPoolSlice final : public graphics::ResourceMemoryStorage {
public:
    id<MTLBuffer> buffer = nil;
    std::uint64_t offset = 0;
    graphics::BackingRange bytes;
    const std::byte* host = nullptr;
    graphics::BackingRange coverage() const noexcept override { return bytes; }
    graphics::RendererStatus read_gpu(graphics::BackingRange range,
                                      std::span<std::byte> destination) const override;
    bool in_place() const noexcept override { return true; }
};

// PipelineStateCache returns these typed leases behind the portable retained
// resource fields. Submission rejects an untyped lease instead of guessing a
// native object or extending a portable interface with Apple framework types.
class MetalRenderPipelineResource : public graphics::ResourceLease {
public:
    virtual id<MTLRenderPipelineState> pipeline() const noexcept = 0;
};

class MetalComputePipelineResource : public graphics::ResourceLease {
public:
    virtual id<MTLComputePipelineState> pipeline() const noexcept = 0;
};

class MetalDepthStencilResource : public graphics::ResourceLease {
public:
    virtual id<MTLDepthStencilState> depth_stencil() const noexcept = 0;
};

class MetalSamplerResource : public graphics::ResourceLease {
public:
    virtual id<MTLSamplerState> sampler() const noexcept = 0;
};

} // namespace armigo::platform::metal
