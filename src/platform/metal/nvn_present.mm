// ============================================================================
// Armigo Platform - NVN Texture Presentation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_present.h"
#include "nvn_resources.h"
#include "nvn_replay.h"
#include <armigo/common/diagnostics.h>
#include <cstring>
#include <future>

namespace armigo::platform::metal::nvn_driver {
namespace {
using graphics::ImageReadbackError;
using graphics::Rgba8ReadbackResult;

Rgba8ReadbackResult read_texture(id<MTLCommandQueue> queue,
                                 id<MTLTexture> texture,
                                 std::array<std::int32_t, 4> crop) {
  @autoreleasepool {
    const auto invalid = [](const char *message) -> Rgba8ReadbackResult {
      return ImageReadbackError{ImageReadbackError::Code::InvalidLayout,
                                message};
    };
    if (crop == std::array<std::int32_t, 4>{})
      crop = {0, 0, static_cast<std::int32_t>(texture.width),
              static_cast<std::int32_t>(texture.height)};
    if (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] ||
        crop[3] <= crop[1] ||
        static_cast<NSUInteger>(crop[2]) > texture.width ||
        static_cast<NSUInteger>(crop[3]) > texture.height)
      return ImageReadbackError{ImageReadbackError::Code::InvalidCrop,
                                "NVN texture crop is out of range"};
    const auto width = static_cast<std::uint32_t>(crop[2] - crop[0]);
    const auto height = static_cast<std::uint32_t>(crop[3] - crop[1]);
    const NSUInteger pitch = (NSUInteger(width) * 4 + 255) & ~NSUInteger(255);
    id<MTLBuffer> staging =
        [queue.device newBufferWithLength:pitch * height
                                  options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!staging || !command || !blit)
      return invalid("NVN texture readback allocation failed");
    [blit copyFromTexture:texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(crop[0], crop[1], 0)
                      sourceSize:MTLSizeMake(width, height, 1)
                        toBuffer:staging
               destinationOffset:0
          destinationBytesPerRow:pitch
        destinationBytesPerImage:pitch * height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return invalid("NVN texture readback GPU work failed");
    graphics::Rgba8Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(std::size_t(width) * height * 4);
    const auto *bytes = static_cast<const std::byte *>(staging.contents);
    for (std::uint32_t row = 0; row < height; ++row)
      std::memcpy(image.pixels.data() + std::size_t(row) * width * 4,
                  bytes + row * pitch, std::size_t(width) * 4);
    if (texture.pixelFormat == MTLPixelFormatBGRA8Unorm ||
        texture.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB) {
      for (std::size_t i = 0; i < image.pixels.size(); i += 4)
        std::swap(image.pixels[i], image.pixels[i + 2]);
    }
    return image;
  }
}
std::optional<services::vi::NativeTexture> export_allocation(
    id<MTLCommandQueue> queue, const PresentationTextureLookup& lookup,
    const armigo::nvn::TextureState& state) {
  @autoreleasepool {
    if (!queue || !lookup)
      return std::nullopt;
    id<MTLTexture> texture = lookup(state);
    if (!texture || texture.device != queue.device ||
        texture.textureType != MTLTextureType2D || texture.sampleCount != 1 ||
        texture.width != state.description.width ||
        texture.height != state.description.height || texture.width > 16384 ||
        texture.height > 16384 ||
        texture.storageMode == MTLStorageModeMemoryless)
      return std::nullopt;
    switch (texture.pixelFormat) {
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
      break;
    default:
      return std::nullopt;
    }
    services::vi::NativeTexture exported;
    exported.serial = state.serial;
    exported.width = state.description.width;
    exported.height = state.description.height;
    exported.read_pixels = [queue, texture](std::array<std::int32_t, 4> crop) {
      return read_texture(queue, texture, crop);
    };
    exported.native_texture = std::shared_ptr<const void>(
        (__bridge_retained const void *)texture,
        [](const void *object) { CFRelease(object); });
    return exported;
  }
}
Rgba8ReadbackResult read_ordered_texture(const Resources::NativeSnapshot& snapshot,
                                       const std::shared_ptr<Metal4NativeSubmission>& submission,
                                       std::array<std::int32_t, 4> crop) {
  @autoreleasepool {
    const auto invalid = [](const char* detail) -> Rgba8ReadbackResult {
      return ImageReadbackError{ImageReadbackError::Code::InvalidLayout, detail};
    };
    id<MTLTexture> texture = snapshot.texture;
    if (!texture || !submission) return invalid("NVN native readback has no submission context");
    if (crop == std::array<std::int32_t, 4>{})
      crop = {0, 0, static_cast<std::int32_t>(texture.width), static_cast<std::int32_t>(texture.height)};
    if (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] || crop[3] <= crop[1] ||
        static_cast<NSUInteger>(crop[2]) > texture.width || static_cast<NSUInteger>(crop[3]) > texture.height)
      return ImageReadbackError{ImageReadbackError::Code::InvalidCrop, "NVN texture crop is out of range"};
    id<MTLDevice> device = texture.device;
    const auto width = static_cast<std::uint32_t>(crop[2] - crop[0]);
    const auto height = static_cast<std::uint32_t>(crop[3] - crop[1]);
    const NSUInteger pitch = (NSUInteger(width) * 4 + 255) & ~NSUInteger(255);
    auto staging = [device newBufferWithLength:pitch * height options:MTLResourceStorageModeShared];
    auto allocator = [device newCommandAllocator];
    auto commands = [device newCommandBuffer];
    NSError* error = nil;
    auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!staging || !allocator || !commands || !residency) return invalid("NVN native readback allocation failed");
    id<MTLTexture> root = texture;
    while (root.parentTexture) root = root.parentTexture;
    [residency addAllocation:root.buffer ? (id<MTLAllocation>)root.buffer : (id<MTLAllocation>)root];
    [residency addAllocation:staging]; [residency commit];
    [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    auto copy = [commands computeCommandEncoder];
    if (!copy) return invalid("NVN native readback encoder failed");
    [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
               visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias];
    [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(crop[0], crop[1], 0)
        sourceSize:MTLSizeMake(width, height, 1) toBuffer:staging destinationOffset:0
        destinationBytesPerRow:pitch destinationBytesPerImage:pitch * height];
    [copy endEncoding]; [commands endCommandBuffer];
    auto failed = std::make_shared<std::atomic<bool>>(false);
    NSMutableArray* retained = [NSMutableArray arrayWithObjects:texture, staging, allocator, residency, nil];
    const auto ticket = submission->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
      static_cast<void>(retained.count);
      failed->store(feedback.error != nil, std::memory_order_release);
    });
    if (!ticket || !submission->wait(ticket, std::chrono::seconds(10)))
      return invalid("NVN native readback did not complete");
    // This explicit CPU reader also observes its completion/error callback;
    // it does not drain any producer published after its own frame position.
    if (failed->load(std::memory_order_acquire) || !submission->producer_error().empty())
      return invalid("NVN native readback producer or GPU work failed");
    graphics::Rgba8Image image;
    image.width = width; image.height = height;
    image.pixels.resize(std::size_t(width) * height * 4);
    const auto* bytes = static_cast<const std::byte*>(staging.contents);
    for (std::uint32_t row = 0; row != height; ++row)
      std::memcpy(image.pixels.data() + std::size_t(row) * width * 4, bytes + row * pitch, width * 4);
    if (texture.pixelFormat == MTLPixelFormatBGRA8Unorm || texture.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB)
      for (std::size_t offset = 0; offset < image.pixels.size(); offset += 4)
        std::swap(image.pixels[offset], image.pixels[offset + 2]);
    return image;
  }
}

std::optional<services::vi::NativeTexture> export_resource_allocation(
    Resources& resources, const armigo::nvn::TextureState& state, bool reserve,
    std::function<std::string()> producer_error = {}) {
  auto snapshot = resources.capture_native(state);
  id<MTLTexture> texture = snapshot.texture;
  const auto owner = resources.submission_owner();
  if (!texture || !owner || texture.device != owner->queue().device ||
      texture.textureType != MTLTextureType2D || texture.sampleCount != 1 ||
      texture.width != state.description.width || texture.height != state.description.height ||
      texture.width > 16384 || texture.height > 16384 || texture.storageMode == MTLStorageModeMemoryless) return {};
  switch (texture.pixelFormat) {
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB: break;
    default: return {};
  }
  if (!resources.publish_uploads()) return {};
  auto check_producer = [upload_error = snapshot.producer_error, producer_error = std::move(producer_error)] {
    if (upload_error) { auto error = upload_error(); if (!error.empty()) return error; }
    return producer_error ? producer_error() : std::string{};
  };
  auto submission = std::make_shared<Metal4NativeSubmission>(owner, reserve ? owner->reserve() : 0,
      std::move(check_producer), snapshot.allocation, snapshot.release_read);
  services::vi::NativeTexture exported;
  exported.serial = state.serial; exported.width = state.description.width; exported.height = state.description.height;
  exported.native_texture = snapshot.allocation;
  exported.native_submission = submission;
  exported.ordered_gpu_dependencies = reserve;
  if (reserve) exported.ordered_producer_ready = [submission] {
    return submission->producer_error().empty();
  };
  exported.cancel = [submission] { submission->cancel(); };
  exported.read_pixels = [snapshot, submission](std::array<std::int32_t, 4> crop) {
    return read_ordered_texture(snapshot, submission, crop);
  };
  return exported;
}

struct NativeCaptureRequest {
  std::mutex mutex;
  bool canceled = false, finished = false;
  std::shared_ptr<Metal4NativeSubmission> submission;
  std::promise<std::optional<services::vi::NativeTexture>> promise;
  bool is_canceled() { std::lock_guard lock(mutex); return canceled; }
  void cancel() {
    std::shared_ptr<Metal4NativeSubmission> native;
    { std::lock_guard lock(mutex); canceled = true; native = submission; }
    if (native) native->cancel();
  }
  void finish(std::optional<services::vi::NativeTexture> result) {
    std::unique_lock lock(mutex);
    if (finished) return;
    finished = true;
    if (result) submission = std::static_pointer_cast<Metal4NativeSubmission>(result->native_submission);
    if (canceled) {
      auto native = submission;
      lock.unlock();
      if (native) native->cancel();
      promise.set_value(std::nullopt);
    } else {
      lock.unlock();
      promise.set_value(std::move(result));
    }
  }
};

} // namespace

armigo::nvn::TextureExporter
make_texture_exporter(id<MTLDevice> device, PresentationTextureLookup lookup) {
  id<MTLCommandQueue> queue = [device newCommandQueue];
  return [queue, lookup = std::move(lookup)](const armigo::nvn::TextureState& state)
             -> std::optional<services::vi::NativeTexture> {
    auto exported = export_allocation(queue, lookup, state);
    if (exported)
      exported->resolve = [queue, lookup, state] {
        return export_allocation(queue, lookup, state);
      };
    return exported;
  };
}
armigo::nvn::TextureExporter make_texture_exporter(id<MTLDevice>, Resources& resources) {
  return [&resources](const armigo::nvn::TextureState& state) -> std::optional<services::vi::NativeTexture> {
    auto exported = export_resource_allocation(resources, state, false);
    if (exported) exported->resolve = [&resources, state] { return export_resource_allocation(resources, state, false); };
    return exported;
  };
}

armigo::nvn::TextureExporter make_texture_exporter(id<MTLDevice>, Resources& resources, MetalReplayBackend& replay) {
  return [&resources, &replay](const armigo::nvn::TextureState& state) -> std::optional<services::vi::NativeTexture> {
    // Registration is a template, not a permanent reader of a window buffer.
    auto validation = resources.capture_native(state);
    if (!validation.texture || validation.texture.textureType != MTLTextureType2D ||
        validation.texture.sampleCount != 1 || validation.texture.width != state.description.width ||
        validation.texture.height != state.description.height || validation.texture.width > 16384 ||
        validation.texture.height > 16384 || validation.texture.storageMode == MTLStorageModeMemoryless) return {};
    const auto format = validation.texture.pixelFormat;
    if (format != MTLPixelFormatRGBA8Unorm && format != MTLPixelFormatRGBA8Unorm_sRGB &&
        format != MTLPixelFormatBGRA8Unorm && format != MTLPixelFormatBGRA8Unorm_sRGB) return {};
    validation.release_read();
    services::vi::NativeTexture registered;
    registered.serial = state.serial; registered.width = state.description.width; registered.height = state.description.height;
    registered.read_pixels = [](std::array<std::int32_t, 4>) -> Rgba8ReadbackResult {
      return ImageReadbackError{ImageReadbackError::Code::InvalidLayout, "NVN frame has not resolved its ordered native snapshot"};
    };
    registered.capture = [&resources, &replay, state](std::uint64_t after_fence)
        -> std::optional<services::vi::NativeTexture> {
      auto request = std::make_shared<NativeCaptureRequest>();
      auto future = request->promise.get_future().share();
      auto producer_result = replay.retain_native_producer_result(after_fence);
      if (!replay.enqueue_frame_capture(after_fence, [&resources, &replay, state, request, producer_result = std::move(producer_result)](bool admitted) {
            if (!admitted || request->is_canceled()) { request->finish(std::nullopt); return; }
            auto captured = export_resource_allocation(resources, state, true,
                producer_result);
            if (captured) replay.track_native_submission(
                std::static_pointer_cast<Metal4NativeSubmission>(captured->native_submission));
            request->finish(std::move(captured));
          })) return {};
      services::vi::NativeTexture deferred;
      deferred.serial = state.serial; deferred.width = state.description.width; deferred.height = state.description.height;
      deferred.read_pixels = [](std::array<std::int32_t, 4>) -> Rgba8ReadbackResult {
        return ImageReadbackError{ImageReadbackError::Code::InvalidLayout, "NVN frame snapshot is unresolved"};
      };
      deferred.ordered_gpu_dependencies = true;
      deferred.resolve = [future] { return future.get(); };
      deferred.cancel = [request] { request->cancel(); };
      return deferred;
    };
    return registered;
  };
}
} // namespace armigo::platform::metal::nvn_driver
