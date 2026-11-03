// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Buffer coherence and lifetime policy are shared with the portable cache.

#import <Metal/Metal.h>

#include "buffer_cache.h"
#include "native_resources.h"
#include "device_context.h"
#include "../../graphics/buffer_residency.h"

#include <cstring>

namespace armigo::platform::metal {
namespace {
using namespace graphics;
struct Allocation final : BufferAllocation {
    id<MTLBuffer> buffer = nil;
    std::span<std::byte> contents() const noexcept override {
        return {static_cast<std::byte*>(buffer.contents), buffer.length};
    }
    std::uint64_t cost() const noexcept override { return buffer.allocatedSize; }
};
struct Allocator final : BufferAllocator {
    std::shared_ptr<DeviceContext> device;
    std::uint64_t device_identity() const noexcept override { return device->identity().device; }
    RendererResult<std::shared_ptr<BufferAllocation>> allocate(std::uint64_t size) override {
        if (size > device->device().maxBufferLength)
            return RendererError({CommandStatus::Unsupported, "buffer exceeds Metal device maximum length"});
        auto allocation = std::make_shared<Allocation>();
        allocation->buffer = [device->device() newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!allocation->buffer)
            return RendererError({CommandStatus::MemoryFault, "Metal buffer allocation failed"});
        return std::shared_ptr<BufferAllocation>(allocation);
    }
};
} // namespace

graphics::RendererResult<std::shared_ptr<graphics::BufferCache>> create_buffer_cache(
    std::shared_ptr<DeviceContext> device, std::shared_ptr<graphics::ResourceMemory> memory,
    std::uint64_t budget_bytes) {
    if (!device || !device->device() || !device->queue() || !device->identity().device)
        return graphics::RendererError({graphics::CommandStatus::Unsupported, "buffer cache requires a retained Metal device context"});
    auto allocator = std::make_shared<Allocator>();
    allocator->device = std::move(device);
    return graphics::make_resident_buffer_cache(std::move(allocator), std::move(memory), budget_bytes);
}
graphics::RendererStatus GuestPoolSlice::read_gpu(graphics::BackingRange range,
                                                  std::span<std::byte> destination) const {
    if (!(range.backing == bytes.backing) || range.offset < bytes.offset ||
        range.offset - bytes.offset > bytes.size || range.size > bytes.size - (range.offset - bytes.offset) ||
        destination.size() != range.size || !host)
        return graphics::RendererError({graphics::CommandStatus::MemoryFault,
                                        "guest pool readback range exceeds its slice"});
    std::memcpy(destination.data(), host + (range.offset - bytes.offset), destination.size());
    return graphics::renderer_success();
}

id<MTLBuffer> resident_native_buffer(const graphics::RetainedResource& storage) {
    if (const auto slice = std::dynamic_pointer_cast<const GuestPoolSlice>(storage)) return slice->buffer;
    const auto view = std::dynamic_pointer_cast<const graphics::BufferStorage>(storage);
    if (!view) return nil;
    const auto allocation = std::dynamic_pointer_cast<Allocation>(view->allocation());
    return allocation ? allocation->buffer : nil;
}
} // namespace armigo::platform::metal
