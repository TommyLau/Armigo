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

#include "nvn_deswizzle.h"
#include "nvn_submission_owner.h"

#include <functional>
#include <unordered_set>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// A hardware outcome is independent of CPU retirement callbacks. A resource
// operation may wait on this while an earlier retirement needs its mutex.
class Metal4UploadOutcome {
public:
    void finish(bool failed, NSError* error) {
        {
            std::lock_guard lock(mutex_);
            failed_ = failed;
            error_ = error;
            done_ = true;
        }
        changed_.notify_all();
    }
    void wait() {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return done_; });
    }
    [[nodiscard]] bool done() const { std::lock_guard lock(mutex_); return done_; }
    [[nodiscard]] bool failed() const { std::lock_guard lock(mutex_); return failed_; }
    [[nodiscard]] NSError* error() const { std::lock_guard lock(mutex_); return error_; }
private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false, failed_ = false;
    NSError* error_ = nil;
};

// One separately encoded upload command buffer, published in the same stream
// as its replay consumer. The caller recycles its allocator only after outcome.
class Metal4UploadBatch : public std::enable_shared_from_this<Metal4UploadBatch> {
public:
    struct Copy {
        id<MTLBuffer> staging = nil;
        std::size_t offset = 0, row_bytes = 0, image_bytes = 0;
        MTLSize size{};
        id<MTLTexture> texture = nil;
        NSUInteger slice = 0, level = 0;
        MTLBlitOption options = MTLBlitOptionNone;
        bool first = false;
    };

    Metal4UploadBatch(id<MTLDevice> device, id<MTL4CommandAllocator> allocator)
        : device_(device), allocator_(allocator), outcome_(std::make_shared<Metal4UploadOutcome>()) {
        NSError* error = nil;
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = 64;
        residency_ = [device newResidencySetWithDescriptor:descriptor error:&error];
        commands_ = [device newCommandBuffer];
        if (valid()) {
            [commands_ beginCommandBufferWithAllocator:allocator_];
            [commands_ useResidencySet:residency_];
        }
    }

    [[nodiscard]] bool valid() const { return device_ && allocator_ && commands_ && residency_; }
    [[nodiscard]] std::shared_ptr<Metal4UploadOutcome> outcome() const { return outcome_; }

    bool deswizzle(id<MTLComputePipelineState> pipeline, id<MTLBuffer> source,
                    std::uint64_t source_offset, id<MTLBuffer> destination,
                    std::uint64_t destination_offset, std::uint64_t pitch,
                    const graphics::BlockLinearLevelAddressing& level) {
        if (!valid() || sealed_ || failed_) return false;
        if (!compute_) compute_ = begin_encoder();
        if (!table_) {
            auto* descriptor = [MTL4ArgumentTableDescriptor new];
            descriptor.maxBufferBindCount = 3;
            NSError* error = nil;
            table_ = [device_ newArgumentTableWithDescriptor:descriptor error:&error];
            keep(table_);
        }
        if (!parameters_ || parameter_offset_ + 256 > parameters_.length) {
            parameters_ = [device_ newBufferWithLength:64 * 1024 options:MTLResourceStorageModeShared];
            parameter_offset_ = 0;
            keep(parameters_);
        }
        keep(pipeline); keep(source); keep(destination);
        if (!compute_ || !encode_deswizzle_metal4(compute_, pipeline, table_, parameters_, parameter_offset_,
                source, source_offset, destination, destination_offset, pitch, level)) {
            failed_ = true;
            return false;
        }
        parameter_offset_ += 256;
        return true;
    }

    bool copy(const Copy& copy) {
        if (!valid() || sealed_ || !copy.staging || !copy.texture) {
            failed_ = true;
            return false;
        }
        keep(copy.staging); keep(copy.texture);
        copies_.push_back(copy);
        return true;
    }

    bool seal() {
        if (sealed_) return !failed_;
        sealed_ = true;
        end_encoder(compute_);
        id<MTL4ComputeCommandEncoder> copy_encoder = nil;
        std::unordered_set<const void*> destinations;
        for (const auto& copy : copies_) {
            const auto* destination = (__bridge const void*)copy.texture;
            if (copy.first && !destinations.insert(destination).second) {
                end_encoder(copy_encoder);
                destinations.clear();
                destinations.insert(destination);
            }
            if (!copy_encoder) copy_encoder = begin_encoder();
            if (!copy_encoder) { failed_ = true; break; }
            [copy_encoder copyFromBuffer:copy.staging sourceOffset:copy.offset
                sourceBytesPerRow:copy.row_bytes sourceBytesPerImage:copy.image_bytes sourceSize:copy.size
                toTexture:copy.texture destinationSlice:copy.slice destinationLevel:copy.level
                destinationOrigin:MTLOriginMake(0, 0, 0) options:copy.options];
        }
        end_encoder(copy_encoder);
        copies_.clear();
        if (commands_) [commands_ endCommandBuffer];
        [residency_ commit];
        return !failed_ && valid();
    }

    void publish(const std::shared_ptr<Metal4SubmissionOwner>& owner,
                 const UploadPublication& publication, Metal4Completion observed = nil) {
        const auto self = shared_from_this();
        if (!owner || !seal()) {
            release_gpu_references();
            outcome_->finish(true, nil);
            return;
        }
        const auto reserved = owner->reserve();
        auto submit = [self, owner, observed, reserved] {
            const auto outcome = self->outcome_;
            const auto published = owner->publish_reserved(reserved, self->commands_, nil, ^(id<MTL4CommitFeedback> feedback) {
                if (observed) observed(feedback);
                // Nothing retaining guest pool wrappers or staging remains in
                // this batch when a resource waiter observes its completion.
                self->release_gpu_references();
                outcome->finish(feedback.error != nil, feedback.error);
            });
            if (!published) {
                owner->cancel_reserved(reserved);
                self->release_gpu_references();
                outcome->finish(true, nil);
            }
        };
        if (publication) publication(std::move(submit));
        else submit();
    }

private:
    static constexpr MTL4VisibilityOptions visibility =
        MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias;

    id<MTL4ComputeCommandEncoder> begin_encoder() {
        auto encoder = [commands_ computeCommandEncoder];
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:visibility];
        return encoder;
    }
    static void end_encoder(__strong id<MTL4ComputeCommandEncoder>& encoder) {
        if (!encoder) return;
        [encoder barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:visibility];
        [encoder endEncoding];
        encoder = nil;
    }
    void keep(id object) {
        if (!object || !kept_.insert((__bridge const void*)object).second) return;
        [retained_ addObject:object];
        id allocation = object;
        if ([object conformsToProtocol:@protocol(MTLTexture)]) {
            id<MTLTexture> texture = object;
            while (texture.parentTexture) texture = texture.parentTexture;
            allocation = texture.buffer ? (id)texture.buffer : (id)texture;
            if (allocation != object) keep(allocation);
        }
        if ([allocation conformsToProtocol:@protocol(MTLAllocation)] &&
            resident_.insert((__bridge const void*)allocation).second)
            [residency_ addAllocation:allocation];
    }
    void release_gpu_references() {
        copies_.clear();
        compute_ = nil;
        parameters_ = nil;
        table_ = nil;
        residency_ = nil;
        commands_ = nil;
        retained_ = nil;
        allocator_ = nil;
    }

    id<MTLDevice> device_ = nil;
    id<MTL4CommandAllocator> allocator_ = nil;
    id<MTL4CommandBuffer> commands_ = nil;
    id<MTLResidencySet> residency_ = nil;
    id<MTL4ComputeCommandEncoder> compute_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    id<MTLBuffer> parameters_ = nil;
    NSUInteger parameter_offset_ = 0;
    NSMutableArray<id>* retained_ = [NSMutableArray new];
    std::unordered_set<const void*> kept_;
    std::unordered_set<const void*> resident_;
    std::vector<Copy> copies_;
    std::shared_ptr<Metal4UploadOutcome> outcome_;
    bool sealed_ = false, failed_ = false;
};

} // namespace armigo::platform::metal::nvn_driver
