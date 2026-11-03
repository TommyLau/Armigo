// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Presentation keeps layer resources resident across the queue commit, waits
// for drawable availability, signals GPU completion, and schedules presentation.
// Offscreen capture blits to readback storage and uses an aspect-preserving
// letterbox viewport. A headless native frame nobody reads back is copied on
// the GPU into a reused presenter-owned target instead.

#include <armigo/platform/presenter.h>
#include <TargetConditionals.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::platform {
namespace {

using graphics::Rgba8Image;

// Each color-target dimension is at most 16384.
constexpr std::uint32_t kMaxFrameExtent = 16384;
// Transform bits 0x0..0xF cover the known geometry/inversion values; anything
// outside that mask is not a value this runtime can account for.
constexpr std::uint32_t kKnownTransformMask = 0xFu;
// Completion waits poll the shared state at this cadence. Real completion is
// always reported by the commit feedback handler, never assumed by the poll.
constexpr auto kCompletionPoll = std::chrono::milliseconds(2);

// One committed presentation. The record owns every Metal resource the frame
// touches and stays presenter-owned until its completion is observed (or an
// explicit drain retirement), so a deadline expiry or cancellation never
// releases work the GPU may still execute. A raw pointer escapes into the
// commit feedback block; it stays valid because the presenter destroys records
// only after the handler reported completion under the record's mutex.
struct InflightRecord {
    std::uint64_t frame_number = 0;
    id<MTLResidencySet> residency = nil;
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commands = nil;
    // Headless path: uploaded source, offscreen consumer target, staging and
    // readback storage; for an unread native frame, the producer texture and
    // the reused target and residency set.
    id<MTLTexture> source = nil;
    id<MTLTexture> target = nil;
    id<MTLBuffer> staging = nil;
    id<MTLBuffer> readback_buffer = nil;
    Rgba8Image readback;
    // Windowed path: the drawable stays retained until GPU completion so a
    // timeout cannot release a frame the compositor may still schedule.
    id<CAMetalDrawable> drawable = nil;

    std::mutex mutex;
    std::condition_variable ready;
    bool completed = false;
    bool failed = false;
    std::string detail;
};

// Presented-notification counters live in a shared, record-free block so a
// presented handler that fires after a record retired cannot touch freed
// state. The block owns its captures, so late delivery is safe.
using PresentedCounter = std::atomic<std::uint64_t>;

std::atomic<std::uint64_t> g_test_feedback_delay_ns{0};

PresentationFailure failure(PresentationError code, std::string detail) {
    return PresentationFailure{code, std::move(detail)};
}

MTLTextureDescriptor* frame_texture_descriptor(std::uint32_t width, std::uint32_t height) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                          width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return descriptor;
}

Rgba8Image take_headless_readback(InflightRecord& record) {
    Rgba8Image image;
    image.width = static_cast<std::uint32_t>(record.source.width);
    image.height = static_cast<std::uint32_t>(record.source.height);
    const std::size_t bytes = std::size_t{image.width} * image.height * 4;
    image.pixels.resize(bytes);
    std::memcpy(image.pixels.data(), record.readback_buffer.contents, bytes);
    return image;
}

// Waits until the record completes, the deadline expires or the stop token
// fires. The caller keeps the record alive for the whole call.
enum class WaitOutcome { completed, deadline_expired, cancelled };
WaitOutcome wait_for_record(InflightRecord& record,
                            std::optional<std::chrono::steady_clock::time_point> deadline,
                            std::stop_token stop) {
    std::unique_lock lock(record.mutex);
    for (;;) {
        if (record.completed) return WaitOutcome::completed;
        if (stop.stop_requested()) return WaitOutcome::cancelled;
        const auto now = std::chrono::steady_clock::now();
        if (deadline && now >= *deadline) return WaitOutcome::deadline_expired;
        auto slice = kCompletionPoll;
        if (deadline) {
            const auto remaining = *deadline - now;
            if (remaining < slice) slice = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
        }
        record.ready.wait_for(lock, slice);
    }
}

class MetalPresenter final : public Presenter {
public:
    MetalPresenter(PresentationMode mode, id<MTLDevice> device, id<MTL4CommandQueue> queue)
        : mode_(mode), device_(device), queue_(queue),
          presented_(std::make_shared<PresentedCounter>(0)) {}

    ~MetalPresenter() override {
        // Destruction requires successful drainage: wait for real GPU
        // completion of every committed record instead of dropping resources.
        std::unique_lock lock(retire_mutex_);
        for (;;) {
            bool all_complete = true;
            for (const auto& record : inflight_) {
                std::lock_guard record_lock(record->mutex);
                if (!record->completed) {
                    all_complete = false;
                    break;
                }
            }
            if (all_complete) break;
            retire_signal_.wait_for(lock, kCompletionPoll);
        }
        inflight_.clear();
    }

    PresentationMode mode() const noexcept override { return mode_; }

    void attach_layer(CAMetalLayer* layer) { layer_ = layer; }

    // The runtime presents from a plain thread with no autorelease pool, and a
    // frame autoreleases framework objects (the window's drawable, encoders)
    // that hold its command allocator and surface state. Each call drains its
    // own; the in-flight record keeps what the GPU still needs.
    PresentResult present(const PresentationFrame& frame,
                          std::optional<std::chrono::steady_clock::time_point> deadline,
                          std::stop_token stop) override {
        @autoreleasepool {
            return present_frame(frame, deadline, stop);
        }
    }

    PresentResult present_frame(const PresentationFrame& frame,
                                std::optional<std::chrono::steady_clock::time_point> deadline,
                                std::stop_token stop) {
        if (stop.stop_requested())
            return failure(PresentationError::cancelled, "presentation was cancelled before submission");

        auto* pixels = frame.pixels.get();
        // A native frame's texture is consumed directly when it lives on this
        // device: a windowed presenter draws it when the crop is the whole
        // texture, and a headless one copies a frame nobody reads back.
        if (!pixels && frame.queued.native_texture &&
            (mode_ == PresentationMode::windowed || !frame.readback)) {
            id<MTLTexture> texture = (__bridge id<MTLTexture>)frame.queued.native_texture.get();
            if (texture.device != device_ || texture.textureType != MTLTextureType2D ||
                texture.width > kMaxFrameExtent || texture.height > kMaxFrameExtent)
                return failure(PresentationError::unsupported, "native frame texture cannot be used by this presenter");
            if (frame.queued.transform & ~kKnownTransformMask)
                return failure(PresentationError::unsupported, "frame carries unknown transform bits");
            if (mode_ == PresentationMode::headless)
                return present_headless_native(frame, texture, deadline, stop);
            return present_windowed(frame, nullptr, texture, deadline, stop);
        }
        if (!pixels || pixels->width == 0 || pixels->height == 0)
            return failure(PresentationError::invalid_frame, "presentation frame has no decoded pixels");
        const std::size_t expected = std::size_t{pixels->width} * pixels->height * 4;
        if (pixels->pixels.size() != expected)
            return failure(PresentationError::invalid_frame, "decoded pixel bytes do not match the frame extent");
        if (pixels->width > kMaxFrameExtent || pixels->height > kMaxFrameExtent)
            return failure(PresentationError::unsupported, "frame extent exceeds the 16384 color-target policy");
        if (frame.queued.transform & ~kKnownTransformMask)
            return failure(PresentationError::unsupported, "frame carries unknown transform bits");
        if (frame.queued.scaling_mode > 4)
            return failure(PresentationError::unsupported, "frame carries an unknown scaling mode");

        if (mode_ == PresentationMode::headless)
            return present_headless(frame, pixels, deadline, stop);
        return present_windowed(frame, pixels, nil, deadline, stop);
    }

    PresenterStatus drain(std::optional<std::chrono::steady_clock::time_point> deadline) override {
        std::unique_lock lock(retire_mutex_);
        for (;;) {
            bool all_complete = true;
            for (const auto& record : inflight_) {
                std::lock_guard record_lock(record->mutex);
                if (!record->completed) {
                    all_complete = false;
                    break;
                }
            }
            if (all_complete) break;
            if (deadline && std::chrono::steady_clock::now() >= *deadline)
                return failure(PresentationError::deadline_expired,
                               "inflight presentation work did not drain before the deadline");
            retire_signal_.wait_for(lock, kCompletionPoll);
        }

        // Drainage retires every completed record: successful work outright,
        // the first terminal GPU failure after reporting it (that work can
        // never complete, and retaining it forever would wedge the presenter
        // and its owner), while later failures stay inflight for the next
        // drain so no failure is ever swallowed.
        std::optional<PresentationFailure> reported;
        std::vector<std::unique_ptr<InflightRecord>> remaining;
        for (auto& record : inflight_) {
            std::lock_guard record_lock(record->mutex);
            if (record->failed && !reported)
                reported = failure(PresentationError::host_failure, record->detail);
            else if (record->failed)
                remaining.push_back(std::move(record));
        }
        inflight_ = std::move(remaining);
        retire_signal_.notify_all();
        if (reported) return PresenterStatus(*reported);
        return std::monostate{};
    }

    std::size_t inflight_count() const {
        std::lock_guard lock(retire_mutex_);
        return inflight_.size();
    }

    std::uint64_t presented_count() const noexcept override {
        return presented_->load(std::memory_order_relaxed);
    }

    // Test-only views of the unread headless path's reused resources. Present
    // is serialized, so callers read these between presents.
    id<MTLTexture> native_target() const { return native_target_; }
    std::uint64_t native_target_allocations() const noexcept { return native_target_allocations_; }
    std::uint64_t native_residency_allocations() const noexcept { return native_residency_allocations_; }

private:
    PresentResult present_headless(const PresentationFrame& frame, const Rgba8Image* pixels,
                                   std::optional<std::chrono::steady_clock::time_point> deadline,
                                   std::stop_token stop) {
        const std::uint32_t width = pixels->width;
        const std::uint32_t height = pixels->height;
        const std::size_t bytes = std::size_t{width} * height * 4;

        auto record = std::make_unique<InflightRecord>();
        record->frame_number = frame.queued.frame_number;
        record->source = [device_ newTextureWithDescriptor:frame_texture_descriptor(width, height)];
        record->target = [device_ newTextureWithDescriptor:frame_texture_descriptor(width, height)];
        record->staging = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        record->readback_buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        record->residency = make_residency_set(@[record->source, record->target, record->staging,
                                                record->readback_buffer], 4);
        if (!record->source || !record->target || !record->staging || !record->readback_buffer ||
            !record->residency)
            return failure(PresentationError::host_failure, "headless presentation resource allocation failed");
        std::memcpy(record->staging.contents, pixels->pixels.data(), bytes);
        [record->residency commit];

        record->allocator = [device_ newCommandAllocator];
        record->commands = [device_ newCommandBuffer];
        if (!record->allocator || !record->commands)
            return failure(PresentationError::host_failure, "headless presentation command allocation failed");
        [record->commands beginCommandBufferWithAllocator:record->allocator];
        [record->commands useResidencySet:record->residency];
        id<MTL4ComputeCommandEncoder> upload = [record->commands computeCommandEncoder];
        if (!upload)
            return failure(PresentationError::host_failure, "headless upload encoder creation failed");
        [upload copyFromBuffer:record->staging sourceOffset:0 sourceBytesPerRow:width * 4
            sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width, height, 1) toTexture:record->source
              destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
        [upload endEncoding];
        id<MTL4ComputeCommandEncoder> consumer = [record->commands computeCommandEncoder];
        if (!consumer)
            return failure(PresentationError::host_failure, "headless consumer encoder creation failed");
        [consumer barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [consumer copyFromTexture:record->source sourceSlice:0 sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                       toTexture:record->target destinationSlice:0 destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
        [consumer endEncoding];
        id<MTL4ComputeCommandEncoder> capture = [record->commands computeCommandEncoder];
        if (!capture)
            return failure(PresentationError::host_failure, "headless readback encoder creation failed");
        [capture barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                       visibilityOptions:MTL4VisibilityOptionDevice];
        [capture copyFromTexture:record->target sourceSlice:0 sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                       toBuffer:record->readback_buffer destinationOffset:0
          destinationBytesPerRow:width * 4 destinationBytesPerImage:bytes];
        [capture endEncoding];
        [record->commands endCommandBuffer];

        commit_and_track(std::move(record), frame, /*with_readback=*/true);
        return settle(deadline, stop);
    }

    // Consumes a native frame nobody reads back: the GPU copies its admitted
    // crop into a presenter-owned target. The target and its residency set
    // are reused while the extent, format and producer textures stay the
    // same, so a steady stream allocates no textures or residency sets; the
    // command allocator, buffer and commit options are made per frame as in
    // the other paths.
    PresentResult present_headless_native(const PresentationFrame& frame, id<MTLTexture> texture,
                                          std::optional<std::chrono::steady_clock::time_point> deadline,
                                          std::stop_token stop) {
        // The read-back path admits pixels only at the queued extent; hold the
        // unread path to the same rule so both accept the same frames.
        if (texture.width != frame.queued.width || texture.height != frame.queued.height)
            return failure(PresentationError::invalid_frame, "native frame texture does not match the queued extent");
        std::array<std::int32_t, 4> crop = frame.queued.crop;
        if (crop == std::array<std::int32_t, 4>{})
            crop = {0, 0, static_cast<std::int32_t>(texture.width), static_cast<std::int32_t>(texture.height)};
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] || crop[3] <= crop[1] ||
            static_cast<NSUInteger>(crop[2]) > texture.width || static_cast<NSUInteger>(crop[3]) > texture.height)
            return failure(PresentationError::invalid_frame, "native frame crop lies outside its texture");
        if (texture.sampleCount != 1)
            return failure(PresentationError::unsupported, "a multisampled native frame texture cannot be copied");
        const auto width = static_cast<NSUInteger>(crop[2] - crop[0]);
        const auto height = static_cast<NSUInteger>(crop[3] - crop[1]);

        // The target has the source's format, so the copy moves texels
        // unchanged; it is reallocated only when the extent or format changes.
        if (!native_target_ || native_target_.width != width || native_target_.height != height ||
            native_target_.pixelFormat != texture.pixelFormat) {
            auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
                                                                                  width:width height:height
                                                                              mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> target = [device_ newTextureWithDescriptor:descriptor];
            if (!target)
                return failure(PresentationError::host_failure, "headless presentation target allocation failed");
            native_target_ = target;
            native_residency_ = nil;
            ++native_target_allocations_;
        }

        // A producer cycles through a few textures. The residency set holds
        // the target and the recently presented ones; it is rebuilt when a
        // texture arrives or retires, never changed in place, because inflight
        // records may still use the previous set.
        const std::uint64_t present = ++native_presents_;
        bool known = false;
        bool rebuild = !native_residency_;
        for (auto& source : native_sources_) {
            if (source.texture == texture) {
                source.last_present = present;
                known = true;
            } else if (present - source.last_present > kNativeSourceIdlePresents) {
                rebuild = true;
            }
        }
        if (!known || rebuild) {
            std::vector<NativeSource> sources;
            NSMutableArray<id<MTLAllocation>>* allocations = [NSMutableArray arrayWithObject:native_target_];
            for (const auto& source : native_sources_) {
                if (present - source.last_present > kNativeSourceIdlePresents) continue;
                sources.push_back(source);
                [allocations addObject:source.texture];
            }
            if (!known) {
                sources.push_back(NativeSource{texture, present});
                [allocations addObject:texture];
            }
            id<MTLResidencySet> residency = make_residency_set(allocations, allocations.count);
            if (!residency)
                return failure(PresentationError::host_failure, "headless presentation residency allocation failed");
            [residency commit];
            native_sources_ = std::move(sources);
            native_residency_ = residency;
            ++native_residency_allocations_;
        }

        auto record = std::make_unique<InflightRecord>();
        record->frame_number = frame.queued.frame_number;
        // The record retains the producer texture and the reused target and
        // residency set until the copy completes, so a later reallocation
        // never releases storage this work still uses.
        record->source = texture;
        record->target = native_target_;
        record->residency = native_residency_;
        record->allocator = [device_ newCommandAllocator];
        record->commands = [device_ newCommandBuffer];
        if (!record->allocator || !record->commands)
            return failure(PresentationError::host_failure, "headless presentation command allocation failed");
        [record->commands beginCommandBufferWithAllocator:record->allocator];
        [record->commands useResidencySet:record->residency];
        id<MTL4ComputeCommandEncoder> consumer = [record->commands computeCommandEncoder];
        if (!consumer)
            return failure(PresentationError::host_failure, "headless consumer encoder creation failed");
        // Earlier work on this queue may still touch the reused target.
        [consumer barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [consumer copyFromTexture:texture sourceSlice:0 sourceLevel:0
                     sourceOrigin:MTLOriginMake(NSUInteger(crop[0]), NSUInteger(crop[1]), 0)
                       sourceSize:MTLSizeMake(width, height, 1)
                        toTexture:record->target destinationSlice:0 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
        [consumer endEncoding];
        [record->commands endCommandBuffer];

        commit_and_track(std::move(record), frame, /*with_readback=*/false);
        return settle(deadline, stop);
    }

    // Draws either decoded pixels (uploaded first) or a native texture.
    PresentResult present_windowed(const PresentationFrame& frame, const Rgba8Image* pixels,
                                   id<MTLTexture> native,
                                   std::optional<std::chrono::steady_clock::time_point> deadline,
                                   std::stop_token stop) {
        if (!layer_)
            return failure(PresentationError::host_failure, "windowed presenter has no attached CAMetalLayer");
        const CGSize drawable_size = layer_.drawableSize;
        if (drawable_size.width < 1.0 || drawable_size.height < 1.0)
            return dropped(frame);
        if (deadline && std::chrono::steady_clock::now() >= *deadline)
            return failure(PresentationError::deadline_expired, "deadline expired before a drawable was requested");

        // CAMetalLayer bounds its own nextDrawable wait; the deadline check
        // above keeps an already-expired deadline from requesting one at all.
        id<CAMetalDrawable> drawable = [layer_ nextDrawable];
        if (!drawable)
            return dropped(frame);
        [queue_ waitForDrawable:drawable];

        const std::uint32_t width = native ? std::uint32_t(native.width) : pixels->width;
        const std::uint32_t height = native ? std::uint32_t(native.height) : pixels->height;
        const std::size_t bytes = std::size_t{width} * height * 4;
        auto record = std::make_unique<InflightRecord>();
        record->frame_number = frame.queued.frame_number;
        record->drawable = drawable;
        if (native) {
            // The window shows the stored bytes, as the pixel path does: an
            // sRGB texture is read through its unorm view, not linearized.
            id<MTLTexture> storage = native;
            if (native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ||
                native.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB) {
                native = [native newTextureViewWithPixelFormat:native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB
                                                                   ? MTLPixelFormatRGBA8Unorm
                                                                   : MTLPixelFormatBGRA8Unorm];
                if (!native)
                    return failure(PresentationError::unsupported, "native sRGB frame has no unorm view");
            }
            // The record keeps the texture alive until the draw completes.
            record->source = native;
            record->residency = make_residency_set(native == storage ? @[native] : @[native, storage], 2);
            if (!record->residency)
                return failure(PresentationError::host_failure, "windowed presentation residency allocation failed");
        } else {
            record->source = [device_ newTextureWithDescriptor:frame_texture_descriptor(width, height)];
            record->staging = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            record->residency = make_residency_set(@[record->source, record->staging], 2);
            if (!record->source || !record->staging || !record->residency)
                return failure(PresentationError::host_failure, "windowed presentation resource allocation failed");
            std::memcpy(record->staging.contents, pixels->pixels.data(), bytes);
        }
        [record->residency commit];

        const id<MTLRenderPipelineState> pipeline = pipeline_for_format(layer_.pixelFormat);
        if (!pipeline)
            return failure(PresentationError::host_failure, "present blit pipeline compilation failed");
        const id<MTL4ArgumentTable> arguments = ensure_arguments();
        if (!arguments)
            return failure(PresentationError::host_failure, "present argument table allocation failed");

        record->allocator = [device_ newCommandAllocator];
        record->commands = [device_ newCommandBuffer];
        if (!record->allocator || !record->commands)
            return failure(PresentationError::host_failure, "windowed presentation command allocation failed");
        [record->commands beginCommandBufferWithAllocator:record->allocator];
        [record->commands useResidencySet:record->residency];
        [record->commands useResidencySet:layer_.residencySet];
        if (!native) {
            id<MTL4ComputeCommandEncoder> upload = [record->commands computeCommandEncoder];
            if (!upload)
                return failure(PresentationError::host_failure, "windowed upload encoder creation failed");
            [upload copyFromBuffer:record->staging sourceOffset:0 sourceBytesPerRow:width * 4
                sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width, height, 1) toTexture:record->source
                  destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            [upload endEncoding];
        }

        auto* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        id<MTL4RenderCommandEncoder> render = [record->commands renderCommandEncoderWithDescriptor:pass];
        if (!render)
            return failure(PresentationError::host_failure, "windowed render encoder creation failed");
        [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                      visibilityOptions:MTL4VisibilityOptionDevice];
        [render setViewport:letterbox_viewport(drawable_size, width, height)];
        [render setRenderPipelineState:pipeline];
        [arguments setTexture:record->source.gpuResourceID atIndex:0];
        [render setArgumentTable:arguments atStages:MTLRenderStageFragment];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [render endEncoding];
        [record->commands endCommandBuffer];

        commit_and_track(std::move(record), frame, /*with_readback=*/false);
        return settle(deadline, stop);
    }

    // A window with nothing to draw into (no drawable-sized surface, or
    // minimized or hidden long enough that the layer returns no drawable)
    // drops the frame, as a display drops a hidden layer's frame: the frame
    // still completes, so the guest's buffer queue keeps moving.
    static PresentResult dropped(const PresentationFrame& frame) {
        PresentedFrame result;
        result.frame_number = frame.queued.frame_number;
        result.completed_at = std::chrono::steady_clock::now();
        return PresentResult(std::move(result));
    }

    // Aspect-preserving letterbox: scale the frame to the layer size, center
    // it, and clear the bars. A near-identical aspect uses the full target.
    static MTLViewport letterbox_viewport(const CGSize& drawable_size, std::uint32_t width, std::uint32_t height) {
        const double dw = drawable_size.width;
        const double dh = drawable_size.height;
        const double ratio_error = (dw / dh) / (static_cast<double>(width) / height) - 1.0;
        if (std::fabs(ratio_error) < 0.01)
            return MTLViewport{0, 0, dw, dh, 0, 1};
        const double scale = std::min(dw / width, dh / height);
        const double view_width = std::round(width * scale);
        const double view_height = std::round(height * scale);
        return MTLViewport{std::floor((dw - view_width) / 2), std::floor((dh - view_height) / 2),
                           view_width, view_height, 0, 1};
    }

    id<MTLResidencySet> make_residency_set(NSArray<id<MTLAllocation>>* allocations, std::size_t capacity) {
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = capacity;
        NSError* error = nil;
        id<MTLResidencySet> residency = [device_ newResidencySetWithDescriptor:descriptor error:&error];
        if (!residency) return nil;
        for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
        return residency;
    }

    // Metal 4 render encoders bind textures through an argument table; the
    // slot is rewritten for every present before the encoder consumes it.
    id<MTL4ArgumentTable> ensure_arguments() {
        if (arguments_) return arguments_;
        auto* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxTextureBindCount = 1;
        descriptor.maxBufferBindCount = 0;
        NSError* error = nil;
        arguments_ = [device_ newArgumentTableWithDescriptor:descriptor error:&error];
        return arguments_;
    }

    id<MTLRenderPipelineState> pipeline_for_format(MTLPixelFormat format) {        std::lock_guard lock(pipeline_mutex_);
        for (const auto& entry : pipelines_)
            if (entry.first == format) return entry.second;
        NSString* source = @R"(
            #include <metal_stdlib>
            using namespace metal;

            struct PresentVertexOut {
                float4 position [[position]];
                float2 uv;
            };

            vertex PresentVertexOut armigo_present_vertex(uint vid [[vertex_id]]) {
                const float2 corners[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
                PresentVertexOut out;
                out.position = float4(corners[vid], 0.0, 1.0);
                out.uv = corners[vid] * float2(0.5, -0.5) + float2(0.5, 0.5);
                return out;
            }

            fragment float4 armigo_present_fragment(PresentVertexOut in [[stage_in]],
                                                    texture2d<float> source [[texture(0)]]) {
                constexpr sampler present_sampler(address::clamp_to_edge, filter::linear);
                return source.sample(present_sampler, in.uv);
            }
        )";
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
        if (!library) return nil;
        auto* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [library newFunctionWithName:@"armigo_present_vertex"];
        descriptor.fragmentFunction = [library newFunctionWithName:@"armigo_present_fragment"];
        descriptor.colorAttachments[0].pixelFormat = format;
        id<MTLRenderPipelineState> pipeline = [device_ newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!pipeline) return nil;
        pipelines_.emplace_back(format, pipeline);
        return pipeline;
    }

    // Commits the encoded work and keeps the record in the inflight list until
    // its completion is observed. The feedback handler touches only the record
    // (never borrowed C++ state); records are destroyed solely after that
    // report, so nothing released here can still be executing.
    void commit_and_track(std::unique_ptr<InflightRecord> record, const PresentationFrame& frame,
                          bool with_readback) {
        InflightRecord* context = record.get();
        context->frame_number = frame.queued.frame_number;
        auto* options = [MTL4CommitOptions new];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            const std::uint64_t delay = g_test_feedback_delay_ns.load(std::memory_order_relaxed);
            if (delay) std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
            std::lock_guard lock(context->mutex);
            if (feedback.error) {
                context->failed = true;
                context->detail = std::string("presentation GPU execution failed: ") +
                                  feedback.error.description.UTF8String;
            } else if (with_readback) {
                // GPU completion above makes the shared buffer stable; copy it
                // out on the handler thread so the waiter never reads early.
                context->readback = take_headless_readback(*context);
            }
            context->completed = true;
            context->ready.notify_all();
        }];
        if (record->drawable) {
            std::shared_ptr<std::atomic<bool>> observation = std::make_shared<std::atomic<bool>>(false);
            std::shared_ptr<PresentedCounter> presented = presented_;
            [record->drawable addPresentedHandler:^(id<MTLDrawable> drawable) {
                (void)drawable;
                observation->store(true, std::memory_order_relaxed);
                presented->fetch_add(1, std::memory_order_relaxed);
            }];
        }
        std::lock_guard lock(retire_mutex_);
        id<MTL4CommandBuffer> submitted[] = {record->commands};
        [queue_ commit:submitted count:1 options:options];
        // Metal 4 shows a drawable once the queue signals it and it is
        // presented; without present the layer never gets it back.
        if (record->drawable) {
            [queue_ signalDrawable:record->drawable];
            [record->drawable present];
        }
        inflight_.push_back(std::move(record));
    }

    PresentResult settle(std::optional<std::chrono::steady_clock::time_point> deadline, std::stop_token stop) {
        std::unique_ptr<InflightRecord>* tracked = nullptr;
        {
            std::lock_guard lock(retire_mutex_);
            tracked = &inflight_.back();
        }
        InflightRecord* context = tracked->get();
        switch (wait_for_record(*context, deadline, stop)) {
        case WaitOutcome::deadline_expired:
            return failure(PresentationError::deadline_expired,
                           "presentation deadline expired while GPU work was inflight; resources stay retained");
        case WaitOutcome::cancelled:
            return failure(PresentationError::cancelled,
                           "presentation cancelled while GPU work was inflight; resources stay retained");
        case WaitOutcome::completed:
            break;
        }
        std::string detail;
        Rgba8Image readback;
        std::uint64_t frame_number = 0;
        {
            std::lock_guard record_lock(context->mutex);
            if (context->failed) detail = context->detail;
            else {
                readback = std::move(context->readback);
                frame_number = context->frame_number;
            }
        }
        if (!detail.empty()) {
            retire(context);
            return failure(PresentationError::host_failure, detail);
        }
        PresentedFrame result;
        result.frame_number = frame_number;
        result.completed_at = std::chrono::steady_clock::now();
        if (!readback.pixels.empty()) result.readback = std::move(readback);
        retire(context);
        return PresentResult(std::move(result));
    }

    void retire(InflightRecord* context) {
        std::lock_guard lock(retire_mutex_);
        std::erase_if(inflight_, [context](const std::unique_ptr<InflightRecord>& record) {
            return record.get() == context;
        });
        retire_signal_.notify_all();
    }

    PresentationMode mode_;
    id<MTLDevice> device_ = nil;
    id<MTL4CommandQueue> queue_ = nil;
    // Windowed layer, retained for the presenter lifetime; nil in headless.
    CAMetalLayer* layer_ = nil;
    std::mutex pipeline_mutex_;
    std::vector<std::pair<MTLPixelFormat, id<MTLRenderPipelineState>>> pipelines_;
    id<MTL4ArgumentTable> arguments_ = nil;
    std::shared_ptr<PresentedCounter> presented_;
    mutable std::mutex retire_mutex_;
    std::condition_variable retire_signal_;
    std::vector<std::unique_ptr<InflightRecord>> inflight_;
    // Unread headless frames: the reused copy target, its residency set and
    // the producer textures that set keeps resident. Only the serialized
    // present path touches them; inflight records retain what they use.
    struct NativeSource {
        id<MTLTexture> texture = nil;
        std::uint64_t last_present = 0;
    };
    // A texture not presented for this many unread frames leaves the set, so
    // a producer's retired textures are neither kept alive nor resident.
    static constexpr std::uint64_t kNativeSourceIdlePresents = 8;
    id<MTLTexture> native_target_ = nil;
    id<MTLResidencySet> native_residency_ = nil;
    std::vector<NativeSource> native_sources_;
    std::uint64_t native_presents_ = 0;
    std::uint64_t native_target_allocations_ = 0;
    std::uint64_t native_residency_allocations_ = 0;
};

} // namespace

PresenterCreation create_presenter(const PresenterOptions& options) {
    @autoreleasepool {
        if (options.mode != PresentationMode::headless)
            return failure(PresentationError::unsupported,
                           "windowed presentation is created through create_application");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4])
            return failure(PresentationError::unsupported, "no Metal 4 device is available");
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (!queue)
            return failure(PresentationError::host_failure, "Metal 4 command queue creation failed");
        return PresenterCreation(std::make_shared<MetalPresenter>(PresentationMode::headless, device, queue));
    }
}

// Internal seam for the windows: the drawable for a layer this many pixels in
// size that shows a 16:9 game image content_width wide. It keeps the layer's
// shape and is no larger than the fitted image needs; the compositor scales
// it to the layer.
CGSize content_drawable_size(CGSize pixels, std::uint32_t content_width) {
    const CGFloat fitted = std::min(pixels.width, pixels.height * 16 / 9);
    if (!content_width || fitted <= content_width) return pixels;
    const CGFloat factor = content_width / fitted;
    return CGSizeMake(std::round(pixels.width * factor), std::round(pixels.height * factor));
}

// Internal seam for the AppKit window: attaches a caller-owned CAMetalLayer to
// a windowed presenter. Not part of the frozen portable contract; Apple types
// never appear in public headers.
std::shared_ptr<Presenter> create_layer_presenter(const PresenterOptions& options, CAMetalLayer* layer,
                                                  PresentationFailure* failure_out) {
    @autoreleasepool {
        if (options.mode != PresentationMode::windowed) {
            if (failure_out)
                *failure_out = failure(PresentationError::unsupported, "layer presenter requires mode=windowed");
            return nullptr;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            if (failure_out)
                *failure_out = failure(PresentationError::unsupported, "no Metal 4 device is available");
            return nullptr;
        }
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (!queue) {
            if (failure_out)
                *failure_out = failure(PresentationError::host_failure, "Metal 4 command queue creation failed");
            return nullptr;
        }
        // Layer-managed drawables stay resident for this queue's present work,
        // for Metal 4 CAMetalLayer presentation.
        layer.device = device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
#if !TARGET_OS_IPHONE
        layer.displaySyncEnabled = options.display_sync;
#endif
        [queue addResidencySet:layer.residencySet];
        auto presenter = std::make_shared<MetalPresenter>(PresentationMode::windowed, device, queue);
        presenter->attach_layer(layer);
        return presenter;
    }
}

void presenter_test_set_feedback_delay_ns(std::uint64_t nanoseconds) noexcept {
    g_test_feedback_delay_ns.store(nanoseconds, std::memory_order_relaxed);
}

std::size_t presenter_test_inflight_count(const Presenter& presenter) {
    const auto* impl = dynamic_cast<const MetalPresenter*>(&presenter);
    return impl ? impl->inflight_count() : 0;
}

std::uint64_t presenter_test_presented_count(const Presenter& presenter) {
    const auto* impl = dynamic_cast<const MetalPresenter*>(&presenter);
    return impl ? impl->presented_count() : 0;
}

id<MTLTexture> presenter_test_native_target(const Presenter& presenter) {
    const auto* impl = dynamic_cast<const MetalPresenter*>(&presenter);
    return impl ? impl->native_target() : nil;
}

std::uint64_t presenter_test_native_target_allocations(const Presenter& presenter) {
    const auto* impl = dynamic_cast<const MetalPresenter*>(&presenter);
    return impl ? impl->native_target_allocations() : 0;
}

std::uint64_t presenter_test_native_residency_allocations(const Presenter& presenter) {
    const auto* impl = dynamic_cast<const MetalPresenter*>(&presenter);
    return impl ? impl->native_residency_allocations() : 0;
}

} // namespace armigo::platform
