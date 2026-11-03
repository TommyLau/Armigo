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
#include <cstring>

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
} // namespace

armigo::nvn::TextureExporter
make_texture_exporter(id<MTLDevice> device, PresentationTextureLookup lookup) {
  id<MTLCommandQueue> queue = [device newCommandQueue];
  return [queue,
          lookup = std::move(lookup)](const armigo::nvn::TextureState &state)
             -> std::optional<services::vi::NativeTexture> {
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
  };
}
} // namespace armigo::platform::metal::nvn_driver
