// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/metal_color_clear.h>
#include <armigo/graphics/color_clear.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <new>
#include <span>
#include <vector>

namespace armigo::platform {
namespace {

using namespace armigo::graphics;

CommandResult execute(const RenderStop& snapshot, const GpuAddressSpace& memory,
                      MetalClearStatistics* statistics) {
    const auto planned = plan_color_clear(snapshot);
    if (const auto* error = std::get_if<CommandResult>(&planned)) return *error;
    const auto& plan = std::get<ColorClearPlan>(planned);

    // Validate the complete guest extent before issuing work. This staging
    // memory is never used to restore padding over concurrent guest writes.
    std::vector<std::byte> preflight(plan.storage_size);
    if (!memory.read(plan.target_address, preflight))
        return {CommandStatus::MemoryFault, "Metal clear target backing is not readable"};

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory)
        return {CommandStatus::Unsupported, "Metal clear requires unified-memory Metal 4"};
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTLPixelFormat pixel_format = MTLPixelFormatInvalid;
    switch (plan.format) {
    case ColorTargetFormat::A2B10G10R10Unorm:
        pixel_format = MTLPixelFormatRGB10A2Unorm;
        break;
    case ColorTargetFormat::R8G8B8A8Unorm:
        pixel_format = MTLPixelFormatRGBA8Unorm;
        break;
    case ColorTargetFormat::R8G8B8A8Srgb:
        pixel_format = MTLPixelFormatRGBA8Unorm_sRGB;
        break;
    }
    // A partial region clears a scratch texture at the region size and the
    // exact requested pixel format, then scatters the GPU readback into guest
    // coordinates. MTLLoadActionClear ignores the scissor state, so encoding
    // the region as its own render target is what bounds the cleared area.
    const bool full_region = plan.region.x == 0 && plan.region.y == 0 &&
                             plan.region.width == plan.width &&
                             plan.region.height == plan.height;
    const std::uint32_t encode_width = full_region ? plan.width : plan.region.width;
    const std::uint32_t encode_height = full_region ? plan.height : plan.region.height;
    constexpr std::size_t scratch_alignment = 64;
    std::size_t encode_row_bytes = 0;
    if (full_region) {
        encode_row_bytes = plan.row_bytes;
    } else {
        const std::size_t logical = std::size_t{encode_width} * 4;
        encode_row_bytes = (logical + scratch_alignment - 1) / scratch_alignment * scratch_alignment;
    }
    const std::size_t readback_size = encode_row_bytes * encode_height;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                       width:encode_width height:encode_height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:descriptor];
    id<MTLBuffer> readback = [device newBufferWithLength:readback_size
                               options:MTLResourceStorageModeShared];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 2;
    NSError* allocation_error = nil;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor
                                   error:&allocation_error];
    if (!queue || !allocator || !commands || !target || !readback || !residency)
        return {CommandStatus::Unsupported, "Metal clear resource allocation failed"};
    [residency addAllocation:target];
    [residency addAllocation:readback];
    [residency commit];
    NSArray* retained = @[device, queue, allocator, commands, target, readback, residency];

    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    auto* pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(
        plan.color[0], plan.color[1], plan.color[2], plan.color[3]);
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) {
        [commands endCommandBuffer];
        return {CommandStatus::Unsupported, "Metal clear render encoder creation failed"};
    }
    [render endEncoding];
    id<MTL4ComputeCommandEncoder> transfer = [commands computeCommandEncoder];
    if (!transfer) {
        [commands endCommandBuffer];
        return {CommandStatus::Unsupported, "Metal clear readback encoder creation failed"};
    }
    [transfer barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
    [transfer copyFromTexture:target sourceSlice:0 sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(encode_width, encode_height, 1)
              toBuffer:readback destinationOffset:0 destinationBytesPerRow:encode_row_bytes
              destinationBytesPerImage:readback_size];
    [transfer endEncoding];
    [commands endCommandBuffer];

    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        // No guest-memory callback or borrowed C++ object escapes into this block.
        // Actual feedback retains every GPU allocation even after a timeout.
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(finished);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
        return {CommandStatus::Stopped, "Metal clear GPU completion timed out"};
    if (execution_error)
        return {CommandStatus::Stopped, "Metal clear GPU execution failed"};

    const auto pixels = std::span(static_cast<const std::byte*>(readback.contents), readback_size);
    if (full_region) {
        for (std::uint32_t y = 0; y < plan.height; ++y) {
            // Four RGBA words occupy one contiguous 16-byte segment of a tiled row.
            // Write only actual texels; row, GOB and block padding stay untouched.
            for (std::uint32_t x = 0; x < plan.width;) {
                const auto offset = pixel_byte_offset(plan, x, y);
                if (const auto* error = std::get_if<CommandResult>(&offset)) return *error;
                const std::uint32_t count = plan.layout == ColorTargetLayout::Pitch
                    ? plan.width - x : std::min(4u, plan.width - x);
                const auto source = pixels.subspan(y * encode_row_bytes + x * 4, count * 4);
                if (!memory.write(plan.target_address + std::get<std::size_t>(offset), source))
                    return {CommandStatus::MemoryFault, "Metal clear guest-memory writeback failed"};
                x += count;
            }
        }
    } else {
        // Scatter the region through the full-surface plan so guest texels
        // outside the region and all padding keep their prior bytes. Pitch rows
        // stay contiguous; block-linear texels are written individually because
        // a region may start and end mid-GOB.
        for (std::uint32_t row = 0; row < plan.region.height; ++row) {
            const auto source = pixels.subspan(row * encode_row_bytes, plan.region.width * 4);
            if (plan.layout == ColorTargetLayout::Pitch) {
                const auto offset = pixel_byte_offset(plan, plan.region.x, plan.region.y + row);
                if (const auto* error = std::get_if<CommandResult>(&offset)) return *error;
                if (!memory.write(plan.target_address + std::get<std::size_t>(offset), source))
                    return {CommandStatus::MemoryFault, "Metal clear guest-memory writeback failed"};
            } else {
                for (std::uint32_t column = 0; column < plan.region.width; ++column) {
                    const auto offset =
                        pixel_byte_offset(plan, plan.region.x + column, plan.region.y + row);
                    if (const auto* error = std::get_if<CommandResult>(&offset)) return *error;
                    const auto texel = source.subspan(column * 4, 4);
                    if (!memory.write(plan.target_address + std::get<std::size_t>(offset), texel))
                        return {CommandStatus::MemoryFault,
                                "Metal clear guest-memory writeback failed"};
                }
            }
        }
    }
    if (statistics) {
        ++statistics->completed_clears;
        statistics->written_pixels += std::uint64_t{plan.region.width} * plan.region.height;
    }
    return {};
}

}  // namespace

graphics::CommandResult execute_metal_color_clear(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept {
    @autoreleasepool {
        try {
            return execute(snapshot, memory, static_cast<MetalClearStatistics*>(context));
        } catch (const std::bad_alloc&) {
            return {graphics::CommandStatus::BudgetExceeded, "Metal clear staging allocation failed"};
        }
    }
}

}  // namespace armigo::platform
