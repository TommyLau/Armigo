// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture-cache blits select a direct copy or scaled blit after validating the
// accelerated surface-copy contract and rejecting unsupported operations.

#pragma once

#import <Metal/Metal.h>

#include <armigo/graphics/engines.h>
#include <armigo/graphics/memory.h>
#include <armigo/graphics/render_submission.h>
#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/graphics/texture_cache.h>
#include <armigo/graphics/twod_blit.h>

namespace armigo::platform::metal {

// Borrowed persistent-renderer services for one synchronous 2D blit. The
// device and queue stay owned by the caller's DeviceContext; the registry,
// texture cache and identity must belong to one domain.
struct MetalTwodResources {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    std::shared_ptr<graphics::ResourceMemoryRegistry> memory;
    graphics::TextureCache* textures = nullptr;
    // Strictly ordered unique identity this blit reserves in the shared
    // submission ordering; the registry versions GPU writes with it.
    graphics::SubmissionId submission{};
    std::uint64_t completion_timeout_ns = 0;
    // Deferred write-back leaves blit results GPU-authoritative; the owner's
    // RenderCallbacks::coherence publishes them. Otherwise this call ends
    // with a registry submission boundary and guest-visible memory.
    bool defer_gpu_writeback = false;
};

// Executes one stopped 2D blit synchronously. Both surfaces resolve through
// the texture cache; same-size same-format copies use a blit encoder, scaled
// or format-converting copies draw a textured quad with the requested filter.
// The destination becomes GPU-dirty in the registry like a draw. Unsupported
// operations, formats or geometries return an owned structured error naming
// them and perform no GPU work. Success means GPU execution and, unless the
// caller defers write-back, guest coherence completed.
[[nodiscard]] graphics::RendererStatus execute_metal_twod_blit(
    MetalTwodResources& resources, const graphics::RenderStop& stop,
    const graphics::GpuAddressSpace& memory) noexcept;

}  // namespace armigo::platform::metal
