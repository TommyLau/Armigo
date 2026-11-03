// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "device_context.h"

#include <atomic>
#include <limits>

namespace armigo::platform::metal {
namespace {

graphics::RendererError failure(const char* detail) {
    return graphics::RendererError({graphics::CommandStatus::Unsupported, detail});
}

class Context final : public DeviceContext {
public:
    id<MTLDevice> retained_device = nil;
    id<MTL4CommandQueue> retained_queue = nil;
    PipelineContextKey key;

    id<MTLDevice> device() const noexcept override { return retained_device; }
    id<MTL4CommandQueue> queue() const noexcept override { return retained_queue; }
    const PipelineContextKey& identity() const noexcept override { return key; }
};

} // namespace

graphics::RendererResult<std::shared_ptr<DeviceContext>>
create_device_context(id<MTLDevice> preferred_device) {
    @autoreleasepool {
        id<MTLDevice> device = preferred_device ? preferred_device
                                                : MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4])
            return failure("device context requires a Metal 4 device");
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (!queue) return failure("Metal 4 command queue creation failed");
        static std::atomic<std::uint64_t> next_identity{1};
        auto identity = next_identity.load(std::memory_order_relaxed);
        while (identity && identity != std::numeric_limits<std::uint64_t>::max() &&
               !next_identity.compare_exchange_weak(identity, identity + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
        if (!identity || identity == std::numeric_limits<std::uint64_t>::max())
            return failure("Metal device identity space is exhausted");
        try {
            auto context = std::make_shared<Context>();
            context->retained_device = device;
            context->retained_queue = queue;
            context->key = {identity, "metal-4-native", "apple-family-metal4"};
            return std::shared_ptr<DeviceContext>(std::move(context));
        } catch (const std::bad_alloc&) {
            return graphics::RendererError(
                {graphics::CommandStatus::MemoryFault, "device context allocation failed"});
        }
    }
}

} // namespace armigo::platform::metal
