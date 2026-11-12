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

#import <Metal/Metal.h>

namespace armigo::platform::metal::nvn_driver {
// Image units access encoded texels without the transfer function used by
// sampling.
inline MTLPixelFormat raw_image_format(MTLPixelFormat format) {
  switch (format) {
  case MTLPixelFormatRGBA8Unorm_sRGB:
    return MTLPixelFormatRGBA8Unorm;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    return MTLPixelFormatBGRA8Unorm;
  case MTLPixelFormatBC1_RGBA_sRGB:
    return MTLPixelFormatBC1_RGBA;
  case MTLPixelFormatBC2_RGBA_sRGB:
    return MTLPixelFormatBC2_RGBA;
  case MTLPixelFormatBC3_RGBA_sRGB:
    return MTLPixelFormatBC3_RGBA;
  case MTLPixelFormatBC7_RGBAUnorm_sRGB:
    return MTLPixelFormatBC7_RGBAUnorm;
  default:
    return format;
  }
}
inline bool raw_image_format_twins(MTLPixelFormat a, MTLPixelFormat b) {
  return a != b && raw_image_format(a) == raw_image_format(b);
}
inline bool raw_image_view_required(id<MTLTexture> texture) {
  if (!texture)
    return false;
  const auto s = texture.swizzle;
  return raw_image_format(texture.pixelFormat) != texture.pixelFormat ||
         s.red != MTLTextureSwizzleRed || s.green != MTLTextureSwizzleGreen ||
         s.blue != MTLTextureSwizzleBlue || s.alpha != MTLTextureSwizzleAlpha;
}
// A buffer texture has no Metal texture-view operation. Required conversions
// remain unsupported rather than exposing sampled format or channel semantics.
inline id<MTLTexture> new_raw_image_view(id<MTLTexture> texture) {
  if (!raw_image_view_required(texture))
    return texture;
  if (texture.textureType == MTLTextureTypeTextureBuffer)
    return nil;
  id<MTLTexture> root = texture;
  NSUInteger level = 0, slice = 0;
  while (root.parentTexture) {
    level += root.parentRelativeLevel;
    slice += root.parentRelativeSlice;
    root = root.parentTexture;
  }
  const auto type = texture.textureType;
  const bool cube =
      type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
  const NSUInteger slices =
      type == MTLTextureType3D ? 1 : texture.arrayLength * (cube ? 6 : 1);
  // The root removes sampled channel mapping; image units use identity
  // channels.
  return [root
      newTextureViewWithPixelFormat:raw_image_format(texture.pixelFormat)
                        textureType:type
                             levels:NSMakeRange(level, texture.mipmapLevelCount)
                             slices:NSMakeRange(slice, slices)];
}
} // namespace armigo::platform::metal::nvn_driver
