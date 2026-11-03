// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/platform/headless_probe.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace armigo::platform {
namespace {

constexpr NSUInteger kWidth = 64;
constexpr NSUInteger kHeight = 32;
constexpr NSUInteger kBytesPerRow = kWidth * 4;
constexpr NSUInteger kBufferSize = kBytesPerRow * kHeight;
constexpr std::int64_t kTimeoutNanoseconds = 5 * NSEC_PER_SEC;

bool submit_and_wait(id<MTL4CommandQueue> queue,
                     id<MTL4CommandBuffer> command_buffer,
                     NSArray* retained_resources,
                     std::string& error) {
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        // Keep allocations alive even if the caller's bounded wait times out.
        static_cast<void>(retained_resources.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> commands[] = {command_buffer};
    [queue commit:commands count:1 options:options];
    const auto deadline = dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds);
    if (dispatch_semaphore_wait(completed, deadline) != 0) {
        error = "Metal 4 submission timed out.";
        return false;
    }
    if (execution_error != nil) {
        error = "Metal 4 submission failed with error code " +
                std::to_string(execution_error.code) + ".";
        return false;
    }
    return true;
}

bool run_checks(id<MTLDevice> device, MetalProbeResult& result) {
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    id<MTLBuffer> fill = [device newBufferWithLength:kBufferSize
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> readback = [device newBufferWithLength:kBufferSize
                                               options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* texture_descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:kWidth
                                                          height:kHeight
                                                       mipmapped:NO];
    texture_descriptor.storageMode = MTLStorageModePrivate;
    texture_descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:texture_descriptor];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 3;
    NSError* residency_error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&residency_error];
    if (!queue || !allocator || !commands || !fill || !readback || !target || !residency) {
        result.error = "Could not allocate Metal 4 self-test resources.";
        return false;
    }

    [residency addAllocation:fill];
    [residency addAllocation:readback];
    [residency addAllocation:target];
    [residency commit];
    NSArray* retained_resources = @[queue, allocator, commands, fill, readback, target, residency];
    constexpr std::array<std::uint8_t, 3> patterns = {0x5a, 0xa5, 0x3c};

    for (std::size_t iteration = 0; iteration < patterns.size(); ++iteration) {
        // The preceding iteration finished before either CPU writes or reuse.
        std::memset(fill.contents, 0, kBufferSize);
        std::memset(readback.contents, 0xcc, kBufferSize);
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];

        MTL4RenderPassDescriptor* render_pass = [MTL4RenderPassDescriptor new];
        auto* attachment = render_pass.colorAttachments[0];
        attachment.texture = target;
        attachment.loadAction = MTLLoadActionClear;
        attachment.storeAction = MTLStoreActionStore;
        attachment.clearColor = MTLClearColorMake(iteration == 0 ? 1.0 : 0.0,
                                                  iteration == 1 ? 1.0 : 0.0,
                                                  iteration == 2 ? 1.0 : 0.0, 1.0);
        id<MTL4RenderCommandEncoder> render =
            [commands renderCommandEncoderWithDescriptor:render_pass];
        if (!render) {
            [commands endCommandBuffer];
            result.error = "Could not create the Metal 4 render encoder.";
            return false;
        }
        [render endEncoding];

        id<MTL4ComputeCommandEncoder> transfer = [commands computeCommandEncoder];
        if (!transfer) {
            [commands endCommandBuffer];
            result.error = "Could not create the Metal 4 transfer encoder.";
            return false;
        }
        // The copy consumes the preceding render pass's stored color attachment.
        [transfer barrierAfterQueueStages:MTLStageAll
                              beforeStages:MTLStageBlit
                         visibilityOptions:MTL4VisibilityOptionDevice];
        [transfer fillBuffer:fill range:NSMakeRange(0, kBufferSize) value:patterns[iteration]];
        [transfer copyFromTexture:target
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                         toBuffer:readback
                destinationOffset:0
           destinationBytesPerRow:kBytesPerRow
         destinationBytesPerImage:kBufferSize];
        [transfer endEncoding];
        [commands endCommandBuffer];
        if (!submit_and_wait(queue, commands, retained_resources, result.error)) {
            return false;
        }

        const auto* bytes = static_cast<const std::uint8_t*>(fill.contents);
        for (NSUInteger index = 0; index < kBufferSize; ++index) {
            if (bytes[index] != patterns[iteration]) {
                result.error = "Metal 4 buffer fill verification failed.";
                return false;
            }
        }
        result.verified_buffer_bytes += kBufferSize;
        const auto* pixels = static_cast<const std::uint8_t*>(readback.contents);
        for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const std::uint8_t expected = channel == iteration || channel == 3 ? 255 : 0;
                if (pixels[pixel * 4 + channel] != expected) {
                    result.error = "Metal 4 offscreen color verification failed.";
                    return false;
                }
            }
        }
        result.verified_pixels += kWidth * kHeight;
        // Completion, not the iteration number, makes allocator reset safe.
        [allocator reset];
    }
    return true;
}

} // namespace

MetalProbeResult probe_metal(bool run_self_test) {
    @autoreleasepool {
        MetalProbeResult result;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            result.error = "No Metal device is available.";
            return result;
        }
        result.device_name = device.name.UTF8String;
        result.supported = [device supportsFamily:MTLGPUFamilyMetal4] && device.hasUnifiedMemory;
        if (!result.supported) {
            result.error = "Armigo requires a unified-memory Metal 4 device.";
            return result;
        }
        if (run_self_test) {
            result.self_test_ran = true;
            result.self_test_passed = run_checks(device, result);
        }
        return result;
    }
}

} // namespace armigo::platform
