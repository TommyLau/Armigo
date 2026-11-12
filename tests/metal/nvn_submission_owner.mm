// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_submission_owner.h"
#include "nvn_deswizzle.h"
#include "nvn_upload_batch.h"
#include "commit_feedback.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace armigo::platform::metal::nvn_driver;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); ++failures; } } while (false)

void retirement_prefix() {
    Metal4RetirementQueue retirement;
    auto order = std::make_shared<std::vector<unsigned>>();
    const auto first = retirement.reserve(nil, ^(id<MTL4CommitFeedback>) { order->push_back(1); });
    const auto second = retirement.reserve(nil, ^(id<MTL4CommitFeedback>) { order->push_back(2); });
    retirement.complete(second, nil);
    CHECK(!retirement.wait_gpu(second, std::chrono::milliseconds(1)));
    CHECK(retirement.retired() == 0 && order->empty());
    CHECK(!retirement.wait(second, std::chrono::milliseconds(1)));
    retirement.complete(first, nil);
    CHECK(retirement.retired() == second && *order == (std::vector<unsigned>{1, 2}));
}

void hardware_before_cpu_retirement(id<MTLDevice> device) {
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, [device newMTL4CommandQueue]);
    auto blocked = dispatch_semaphore_create(0);
    auto release = dispatch_semaphore_create(0);
    auto first_allocator = [device newCommandAllocator];
    auto second_allocator = [device newCommandAllocator];
    auto first = [device newCommandBuffer];
    auto second = [device newCommandBuffer];
    [first beginCommandBufferWithAllocator:first_allocator]; [first endCommandBuffer];
    [second beginCommandBufferWithAllocator:second_allocator]; [second endCommandBuffer];
    const auto a = owner->publish(first, ^(id<MTL4CommitFeedback>) {
        dispatch_semaphore_signal(blocked);
        dispatch_semaphore_wait(release, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
    });
    const auto b = owner->publish(second, nil);
    CHECK(a && b == a + 1);
    CHECK(dispatch_semaphore_wait(blocked, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
    CHECK(owner->wait_gpu(b, std::chrono::seconds(2)));
    CHECK(!owner->wait(b, std::chrono::milliseconds(1)));
    dispatch_semaphore_signal(release);
    CHECK(owner->wait(b, std::chrono::seconds(2)));
}

void canceled_publication_progress(id<MTLDevice> device) {
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, [device newMTL4CommandQueue]);
    const auto first = owner->reserve(), second = owner->reserve();
    CHECK(first && second == first + 1);
    CHECK(owner->cancel_reserved(second) && owner->published() == 0);
    CHECK(owner->cancel_reserved(first) && owner->published() == second);
    CHECK(owner->wait(second, std::chrono::seconds(2)));
    CHECK(!owner->cancel_reserved(first));
}

void ordered_gpu_work(id<MTLDevice> device) {
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, queue);
    CHECK(owner->valid());
    id<MTLBuffer> source = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> destination = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    NSError* error = nil;
    auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    CHECK(source && destination && residency && !error);
    if (!source || !destination || !residency) return;
    [residency addAllocation:source];
    [residency addAllocation:destination];
    [residency commit];
    std::memset(source.contents, 0, source.length);
    std::memset(destination.contents, 0, destination.length);
    id<MTLSharedEvent> gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    auto order = std::make_shared<std::vector<unsigned>>();
    __weak NSObject* weak_payload;
    std::uint64_t first = 0;
    @autoreleasepool {
        NSObject* payload = [NSObject new];
        weak_payload = payload;
        auto allocator = [device newCommandAllocator];
        auto commands = [device newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
        auto encoder = [commands computeCommandEncoder];
        [encoder fillBuffer:source range:NSMakeRange(0, 256) value:0x5a];
        [encoder barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll
                 visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding]; [commands endCommandBuffer];
        first = owner->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
            static_cast<void>(payload);
            static_cast<void>(allocator);
            CHECK(!feedback.error);
            order->push_back(1);
        });
    }
    auto allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    auto encoder = [commands computeCommandEncoder];
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
             visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:256];
    [encoder endEncoding]; [commands endCommandBuffer];
    const auto second = owner->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
        CHECK(!feedback.error);
        order->push_back(2);
    });
    CHECK(first && second == first + 1 && weak_payload != nil);
    CHECK(!owner->wait(second, std::chrono::milliseconds(1)));
    gate.signaledValue = 1;
    CHECK(owner->wait(second, std::chrono::seconds(5)));
    CHECK(weak_payload == nil && *order == (std::vector<unsigned>{1, 2}));
    const auto* bytes = static_cast<const unsigned char*>(destination.contents);
    for (unsigned index = 0; index != 256; ++index) CHECK(bytes[index] == 0x5a);
    CHECK(owner->publish(nil, nil) == 0 && owner->issued() == second);
}

void metal4_deswizzle_parameters(id<MTLDevice> device) {
    using namespace armigo;
    auto kernel = make_deswizzle_kernel(device);
    CHECK(kernel.pipeline != nil);
    if (!kernel.pipeline) return;
    auto queue = [device newMTL4CommandQueue];
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, queue);
    auto source = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    auto destination = [device newBufferWithLength:1536 options:MTLResourceStorageModeShared];
    auto parameters = [device newBufferWithLength:512 options:MTLResourceStorageModeShared];
    std::memset(source.contents, 0x3c, source.length);
    std::memset(destination.contents, 0xa5, destination.length);
    auto* input = static_cast<unsigned char*>(source.contents);
    for (unsigned y = 0; y != 8; ++y)
        for (unsigned x = 0; x != 64; ++x) {
            const auto gob = (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16;
            input[256 + gob] = static_cast<unsigned char>(x * 7 + y * 13);
        }
    NSError* error = nil;
    auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    auto* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 3;
    auto table = [device newArgumentTableWithDescriptor:descriptor error:&error];
    CHECK(residency && table && !error);
    if (!residency || !table) return;
    [residency addAllocation:source]; [residency addAllocation:destination];
    [residency addAllocation:parameters]; [residency commit];
    auto allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    auto encoder = [commands computeCommandEncoder];
    graphics::BlockLinearLevelAddressing level{};
    level.row_bytes = 64; level.rows = 8; level.slices = 1;
    level.row_stride = 512; level.slice_stride = 512; level.guest_size = 512;
    CHECK(encode_deswizzle_metal4(encoder, kernel.pipeline, table, parameters, 256,
        source, 256, destination, 128, 128, level));
    CHECK(!encode_deswizzle_metal4(encoder, kernel.pipeline, table, parameters, 512,
        source, 256, destination, 128, 128, level));
    [encoder endEncoding]; [commands endCommandBuffer];
    const auto ticket = owner->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
        CHECK(!feedback.error);
    });
    CHECK(owner->wait(ticket, std::chrono::seconds(5)));
    const auto* output = static_cast<const unsigned char*>(destination.contents);
    for (unsigned index = 0; index != destination.length; ++index) {
        unsigned char expected = 0xa5;
        if (index >= 128 && index < 1152) {
            const auto y = (index - 128) / 128, x = (index - 128) % 128;
            if (x < 64) expected = static_cast<unsigned char>(x * 7 + y * 13);
        }
        CHECK(output[index] == expected);
    }
}

void deferred_upload_publication(id<MTLDevice> device, bool failed) {
    auto queue = [device newMTL4CommandQueue];
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, queue,
        [failed](std::uint64_t ticket, id<MTL4CommitFeedback> feedback) -> id<MTL4CommitFeedback> {
            return failed && ticket == 1 ? failed_commit_feedback(feedback) : feedback;
        });
    auto kernel = make_deswizzle_kernel(device);
    CHECK(owner->valid() && kernel.pipeline);
    if (!owner->valid() || !kernel.pipeline) return;
    __weak id<MTLBuffer> weak_source, weak_staging;
    __weak id<MTLTexture> weak_texture;
    std::shared_ptr<Metal4UploadOutcome> outcome;
    std::vector<std::function<void()>> records;
    id<MTLBuffer> readback;
    id<MTLSharedEvent> gate;
    std::uint64_t consumed = 0;
    @autoreleasepool {
    auto allocator = [device newCommandAllocator];
    auto upload = std::make_shared<Metal4UploadBatch>(device, allocator);
    auto source = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    auto staging = [device newBufferWithLength:2048 options:MTLResourceStorageModePrivate];
    readback = [device newBufferWithLength:2048 options:MTLResourceStorageModeShared];
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:16 height:8 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.allowGPUOptimizedContents = NO;
    auto texture = [device newTextureWithDescriptor:descriptor];
    CHECK(source && staging && readback && texture && upload->valid());
    if (!source || !staging || !readback || !texture || !upload->valid()) return;
    auto* input = static_cast<unsigned char*>(source.contents);
    std::memset(input, 0x3c, source.length);
    std::memset(readback.contents, 0xa5, readback.length);
    for (unsigned y = 0; y != 8; ++y)
        for (unsigned x = 0; x != 64; ++x) {
            const auto gob = (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16;
            input[256 + gob] = static_cast<unsigned char>(x * 7 + y * 13);
        }
    armigo::graphics::BlockLinearLevelAddressing level{};
    level.row_bytes = 64; level.rows = 8; level.slices = 1;
    level.row_stride = level.slice_stride = level.guest_size = 512;
    CHECK(upload->deswizzle(kernel.pipeline, source, 256, staging, 0, 256, level));
    CHECK(upload->copy({staging, 0, 256, 2048, MTLSizeMake(16, 8, 1), texture, 0, 0, MTLBlitOptionNone, true}));
    weak_source = source; weak_staging = staging; weak_texture = texture;
    outcome = upload->outcome();
    upload->publish(owner, [&](std::function<void()> record) { records.push_back(std::move(record)); });
    CHECK(records.size() == 1 && owner->issued() == 1 && owner->published() == 0 && !outcome->done());
    auto consumer_allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    NSError* error = nil;
    auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    CHECK(residency && !error);
    [residency addAllocation:texture]; [residency addAllocation:readback]; [residency commit];
    [commands beginCommandBufferWithAllocator:consumer_allocator]; [commands useResidencySet:residency];
    auto encoder = [commands computeCommandEncoder];
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:MTLSizeMake(16, 8, 1) toBuffer:readback destinationOffset:0
        destinationBytesPerRow:256 destinationBytesPerImage:2048];
    [encoder endEncoding]; [commands endCommandBuffer];
    gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    consumed = owner->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(consumer_allocator); static_cast<void>(residency);
        CHECK(!feedback.error);
    });
    source = nil; staging = nil; texture = nil; upload.reset();
    }
    CHECK(weak_source && weak_staging && weak_texture);
    CHECK(consumed == 2 && owner->published() == 0 && !outcome->done());
    records.front()();
    records.clear();
    CHECK(owner->published() == consumed);
    gate.signaledValue = 1;
    CHECK(owner->wait(consumed, std::chrono::seconds(5)));
    CHECK(outcome->done() && outcome->failed() == failed);
    CHECK(!failed || outcome->error().code == 27);
    CHECK(!weak_source && !weak_staging);
    CHECK(!weak_texture);
    const auto* output = static_cast<const unsigned char*>(readback.contents);
    for (unsigned y = 0; y != 8; ++y)
        for (unsigned x = 0; x != 256; ++x)
            CHECK(output[y * 256 + x] == (x < 64 ? static_cast<unsigned char>(x * 7 + y * 13) : 0xa5));
}
} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        CHECK(device != nil);
        if (!device) return 1;
        retirement_prefix();
        hardware_before_cpu_retirement(device);
        canceled_publication_progress(device);
        ordered_gpu_work(device);
        metal4_deswizzle_parameters(device);
        for (const bool failed : {false, true}) deferred_upload_publication(device, failed);
    }
    return failures ? 1 : 0;
}
