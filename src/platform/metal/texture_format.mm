// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Guest format mapping records exact Metal matches, required channel swizzles,
// and formats without a direct counterpart. Packed border colors use the
// standard sRGB electro-optical transfer function.

#include "texture_format.h"

#include <armigo/graphics/texture_format.h>
#include <bit>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace armigo::platform {
namespace {

using graphics::TextureFormat;

MTLTextureSwizzleChannels identity_swizzle() noexcept {
    return MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed,
                                         MTLTextureSwizzleGreen,
                                         MTLTextureSwizzleBlue,
                                         MTLTextureSwizzleAlpha);
}

graphics::TextureError unsupported(TextureFormat format) noexcept {
    return {graphics::TextureErrorCode::UnsupportedTextureFormat,
            graphics::TextureField::Format, 0,
            static_cast<std::uint64_t>(format), 0,
            graphics::MemoryErrorCode::None};
}

}  // namespace

std::variant<id<MTLTexture>, graphics::CommandResult> metal_null_texture(
    id<MTLDevice> device, id<MTL4CommandQueue> queue,
    const graphics::TextureDescriptor& texture, graphics::TextureNumericClass scalar, bool depth) {
    using namespace graphics;
    if (!device || !queue || texture.samples != 1)
        return CommandResult{CommandStatus::Unsupported,
            "null texture requires a device, queue and single-sample descriptor"};
    MTLTextureType type;
    switch (texture.type) {
    case TextureType::Texture1D: type = MTLTextureType1D; break;
    case TextureType::Texture1DArray: type = MTLTextureType1DArray; break;
    case TextureType::Texture2D: case TextureType::Texture2DRect:
        type = MTLTextureType2D; break;
    case TextureType::Texture2DArray: type = MTLTextureType2DArray; break;
    case TextureType::Texture3D: type = MTLTextureType3D; break;
    case TextureType::Cubemap: type = MTLTextureTypeCube; break;
    case TextureType::CubemapArray: type = MTLTextureTypeCubeArray; break;
    default:
        return CommandResult{CommandStatus::Unsupported, "null texture type is unsupported"};
    }
    // The reflected shader declaration supplies the class. The null TIC has
    // no storage format or aspect that could determine its native resource.
    if (depth && scalar != TextureNumericClass::Float)
        return CommandResult{CommandStatus::Unsupported, "null depth texture requires a float scalar"};
    if (depth && type != MTLTextureType2D && type != MTLTextureType2DArray &&
        type != MTLTextureTypeCube && type != MTLTextureTypeCubeArray)
        return CommandResult{CommandStatus::Unsupported, "null depth texture type is unsupported"};
    MTLPixelFormat format = MTLPixelFormatRGBA8Unorm;
    if (depth) format = MTLPixelFormatDepth32Float;
    else if (scalar == TextureNumericClass::Uint)
        format = MTLPixelFormatRGBA32Uint;
    else if (scalar == TextureNumericClass::Sint)
        format = MTLPixelFormatRGBA32Sint;
    auto* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = type;
    descriptor.pixelFormat = format;
    descriptor.width = descriptor.height = descriptor.depth = 1;
    descriptor.arrayLength = 1;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> native = [device newTextureWithDescriptor:descriptor];
    id<MTLBuffer> zeros = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 2;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:nil];
    if (!native || !zeros || !allocator || !commands || !residency)
        return CommandResult{CommandStatus::BudgetExceeded, "null texture allocation failed"};
    std::memset(zeros.contents, 0, zeros.length);
    [residency addAllocation:native];
    [residency addAllocation:zeros];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) {
        [commands endCommandBuffer];
        return CommandResult{CommandStatus::Stopped, "null texture upload encoder allocation failed"};
    }
    const unsigned slices = type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray ? 6 : 1;
    for (unsigned slice = 0; slice != slices; ++slice)
        [copy copyFromBuffer:zeros sourceOffset:0 sourceBytesPerRow:256 sourceBytesPerImage:256
            sourceSize:MTLSizeMake(1, 1, 1) toTexture:native destinationSlice:slice
            destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [copy barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll
          visibilityOptions:MTL4VisibilityOptionDevice];
    [copy endEncoding];
    [commands endCommandBuffer];
    // Feedback owns every native object even if the bounded wait times out.
    NSArray* retained = @[device, queue, native, zeros, allocator, commands, residency];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block bool failed = false;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        failed = feedback.error != nil;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)))
        return CommandResult{CommandStatus::Stopped, "null texture upload timed out"};
    if (failed)
        return CommandResult{CommandStatus::Stopped, "null texture upload GPU execution failed"};
    return native;
}

std::variant<MetalSampledFormat, graphics::TextureError> metal_pixel_format(
    TextureFormat format) noexcept {
    const auto swizzle = identity_swizzle();
    switch (format) {
        case TextureFormat::D16Unorm:
            return MetalSampledFormat{MTLPixelFormatDepth16Unorm, swizzle};
        case TextureFormat::D32Float:
        case TextureFormat::D24UnormS8Uint:
        case TextureFormat::X8UintD24Unorm:
        case TextureFormat::S8UintD24Unorm:
        case TextureFormat::D32FloatS8Uint:
            return MetalSampledFormat{MTLPixelFormatDepth32Float, swizzle};
        case TextureFormat::R8Unorm:
            return MetalSampledFormat{MTLPixelFormatR8Unorm, swizzle};
        case TextureFormat::R8Snorm:
            return MetalSampledFormat{MTLPixelFormatR8Snorm, swizzle};
        case TextureFormat::R8Uint:
            return MetalSampledFormat{MTLPixelFormatR8Uint, swizzle};
        case TextureFormat::R8Sint:
            return MetalSampledFormat{MTLPixelFormatR8Sint, swizzle};
        case TextureFormat::R16Unorm:
            return MetalSampledFormat{MTLPixelFormatR16Unorm, swizzle};
        case TextureFormat::R16Snorm:
            return MetalSampledFormat{MTLPixelFormatR16Snorm, swizzle};
        case TextureFormat::R16Uint:
            return MetalSampledFormat{MTLPixelFormatR16Uint, swizzle};
        case TextureFormat::R16Sint:
            return MetalSampledFormat{MTLPixelFormatR16Sint, swizzle};
        case TextureFormat::R16Float:
            return MetalSampledFormat{MTLPixelFormatR16Float, swizzle};
        case TextureFormat::R32Uint:
            return MetalSampledFormat{MTLPixelFormatR32Uint, swizzle};
        case TextureFormat::R32Sint:
            return MetalSampledFormat{MTLPixelFormatR32Sint, swizzle};
        case TextureFormat::R32Float:
            return MetalSampledFormat{MTLPixelFormatR32Float, swizzle};
        case TextureFormat::R8G8Unorm:
            return MetalSampledFormat{MTLPixelFormatRG8Unorm, swizzle};
        case TextureFormat::R8G8Snorm:
            return MetalSampledFormat{MTLPixelFormatRG8Snorm, swizzle};
        case TextureFormat::R8G8Uint:
            return MetalSampledFormat{MTLPixelFormatRG8Uint, swizzle};
        case TextureFormat::R8G8Sint:
            return MetalSampledFormat{MTLPixelFormatRG8Sint, swizzle};
        case TextureFormat::R16G16Unorm:
            return MetalSampledFormat{MTLPixelFormatRG16Unorm, swizzle};
        case TextureFormat::R16G16Snorm:
            return MetalSampledFormat{MTLPixelFormatRG16Snorm, swizzle};
        case TextureFormat::R16G16Uint:
            return MetalSampledFormat{MTLPixelFormatRG16Uint, swizzle};
        case TextureFormat::R16G16Sint:
            return MetalSampledFormat{MTLPixelFormatRG16Sint, swizzle};
        case TextureFormat::R16G16Float:
            return MetalSampledFormat{MTLPixelFormatRG16Float, swizzle};
        case TextureFormat::R32G32Uint:
            return MetalSampledFormat{MTLPixelFormatRG32Uint, swizzle};
        case TextureFormat::R32G32Sint:
            return MetalSampledFormat{MTLPixelFormatRG32Sint, swizzle};
        case TextureFormat::R32G32Float:
            return MetalSampledFormat{MTLPixelFormatRG32Float, swizzle};
        case TextureFormat::R32G32B32Uint:
        case TextureFormat::R32G32B32Sint:
        case TextureFormat::R32G32B32Float:
            // Metal has no three-component pixel format; expanding on upload
            // would change the stored bytes.
            return unsupported(format);
        case TextureFormat::R4G4Unorm:
            // Metal has no 4-bit-per-channel two-component format.
            return unsupported(format);
        case TextureFormat::R4G4B4A4Unorm:
            // TIC A4B4G4R4 reads MSB-first exactly like Metal's packed ABGR4.
            return MetalSampledFormat{MTLPixelFormatABGR4Unorm, swizzle};
        case TextureFormat::R5G5B5A1Unorm:
            // TIC A1B5G5R5 reads MSB-first exactly like Metal's packed A1BGR5.
            return MetalSampledFormat{MTLPixelFormatA1BGR5Unorm, swizzle};
        case TextureFormat::A5B5G5R1Unorm:
            // TIC A5B5G5R1 has no Metal counterpart with the same bit
            // placement; the named 5551 formats put the one-bit channel
            // elsewhere.
            return unsupported(format);
        case TextureFormat::R5G6B5Unorm:
            return MetalSampledFormat{MTLPixelFormatB5G6R5Unorm, swizzle};
        case TextureFormat::R10G10B10A2Unorm:
            return MetalSampledFormat{MTLPixelFormatRGB10A2Unorm, swizzle};
        case TextureFormat::R10G10B10A2Uint:
            return MetalSampledFormat{MTLPixelFormatRGB10A2Uint, swizzle};
        case TextureFormat::R11G11B10Float:
            return MetalSampledFormat{MTLPixelFormatRG11B10Float, swizzle};
        case TextureFormat::R9G9B9E5Float:
            return MetalSampledFormat{MTLPixelFormatRGB9E5Float, swizzle};
        case TextureFormat::R8G8B8A8Unorm:
            return MetalSampledFormat{MTLPixelFormatRGBA8Unorm, swizzle};
        case TextureFormat::R8G8B8A8Snorm:
            return MetalSampledFormat{MTLPixelFormatRGBA8Snorm, swizzle};
        case TextureFormat::R8G8B8A8Uint:
            return MetalSampledFormat{MTLPixelFormatRGBA8Uint, swizzle};
        case TextureFormat::R8G8B8A8Sint:
            return MetalSampledFormat{MTLPixelFormatRGBA8Sint, swizzle};
        case TextureFormat::R8G8B8A8Srgb:
            return MetalSampledFormat{MTLPixelFormatRGBA8Unorm_sRGB, swizzle};
        case TextureFormat::B8G8R8A8Unorm:
            return MetalSampledFormat{MTLPixelFormatBGRA8Unorm, swizzle};
        case TextureFormat::B8G8R8A8Srgb:
            return MetalSampledFormat{MTLPixelFormatBGRA8Unorm_sRGB, swizzle};
        case TextureFormat::R16G16B16A16Unorm:
            return MetalSampledFormat{MTLPixelFormatRGBA16Unorm, swizzle};
        case TextureFormat::R16G16B16A16Snorm:
            return MetalSampledFormat{MTLPixelFormatRGBA16Snorm, swizzle};
        case TextureFormat::R16G16B16A16Uint:
            return MetalSampledFormat{MTLPixelFormatRGBA16Uint, swizzle};
        case TextureFormat::R16G16B16A16Sint:
            return MetalSampledFormat{MTLPixelFormatRGBA16Sint, swizzle};
        case TextureFormat::R16G16B16A16Float:
            return MetalSampledFormat{MTLPixelFormatRGBA16Float, swizzle};
        case TextureFormat::R32G32B32A32Uint:
            return MetalSampledFormat{MTLPixelFormatRGBA32Uint, swizzle};
        case TextureFormat::R32G32B32A32Sint:
            return MetalSampledFormat{MTLPixelFormatRGBA32Sint, swizzle};
        case TextureFormat::R32G32B32A32Float:
            return MetalSampledFormat{MTLPixelFormatRGBA32Float, swizzle};
        case TextureFormat::Bc1RgbaUnorm:
            return MetalSampledFormat{MTLPixelFormatBC1_RGBA, swizzle};
        case TextureFormat::Bc1RgbaSrgb:
            return MetalSampledFormat{MTLPixelFormatBC1_RGBA_sRGB, swizzle};
        case TextureFormat::Bc2Unorm:
            return MetalSampledFormat{MTLPixelFormatBC2_RGBA, swizzle};
        case TextureFormat::Bc2Srgb:
            return MetalSampledFormat{MTLPixelFormatBC2_RGBA_sRGB, swizzle};
        case TextureFormat::Bc3Unorm:
            return MetalSampledFormat{MTLPixelFormatBC3_RGBA, swizzle};
        case TextureFormat::Bc3Srgb:
            return MetalSampledFormat{MTLPixelFormatBC3_RGBA_sRGB, swizzle};
        case TextureFormat::Bc4Unorm:
            return MetalSampledFormat{MTLPixelFormatBC4_RUnorm, swizzle};
        case TextureFormat::Bc4Snorm:
            return MetalSampledFormat{MTLPixelFormatBC4_RSnorm, swizzle};
        case TextureFormat::Bc5Unorm:
            return MetalSampledFormat{MTLPixelFormatBC5_RGUnorm, swizzle};
        case TextureFormat::Bc5Snorm:
            return MetalSampledFormat{MTLPixelFormatBC5_RGSnorm, swizzle};
        case TextureFormat::Bc7Unorm:
            return MetalSampledFormat{MTLPixelFormatBC7_RGBAUnorm, swizzle};
        case TextureFormat::Bc7Srgb:
            return MetalSampledFormat{MTLPixelFormatBC7_RGBAUnorm_sRGB,
                                      swizzle};
        case TextureFormat::Bc6HUfloat:
            return MetalSampledFormat{MTLPixelFormatBC6H_RGBUfloat, swizzle};
        case TextureFormat::Bc6HSfloat:
            return MetalSampledFormat{MTLPixelFormatBC6H_RGBFloat, swizzle};
        case TextureFormat::Etc2RgbUnorm:
            return MetalSampledFormat{MTLPixelFormatETC2_RGB8, swizzle};
        case TextureFormat::Etc2RgbSrgb:
            return MetalSampledFormat{MTLPixelFormatETC2_RGB8_sRGB, swizzle};
        case TextureFormat::Etc2RgbPtaUnorm:
            return MetalSampledFormat{MTLPixelFormatETC2_RGB8A1, swizzle};
        case TextureFormat::Etc2RgbPtaSrgb:
            return MetalSampledFormat{MTLPixelFormatETC2_RGB8A1_sRGB, swizzle};
        case TextureFormat::Etc2RgbaUnorm:
            return MetalSampledFormat{MTLPixelFormatEAC_RGBA8, swizzle};
        case TextureFormat::Etc2RgbaSrgb:
            return MetalSampledFormat{MTLPixelFormatEAC_RGBA8_sRGB, swizzle};
        case TextureFormat::Astc4x4Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_4x4_LDR, swizzle};
        case TextureFormat::Astc4x4Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_4x4_sRGB, swizzle};
        case TextureFormat::Astc5x4Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_5x4_LDR, swizzle};
        case TextureFormat::Astc5x4Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_5x4_sRGB, swizzle};
        case TextureFormat::Astc5x5Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_5x5_LDR, swizzle};
        case TextureFormat::Astc5x5Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_5x5_sRGB, swizzle};
        case TextureFormat::Astc6x5Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_6x5_LDR, swizzle};
        case TextureFormat::Astc6x5Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_6x5_sRGB, swizzle};
        case TextureFormat::Astc6x6Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_6x6_LDR, swizzle};
        case TextureFormat::Astc6x6Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_6x6_sRGB, swizzle};
        case TextureFormat::Astc8x5Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_8x5_LDR, swizzle};
        case TextureFormat::Astc8x5Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_8x5_sRGB, swizzle};
        case TextureFormat::Astc8x6Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_8x6_LDR, swizzle};
        case TextureFormat::Astc8x6Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_8x6_sRGB, swizzle};
        case TextureFormat::Astc8x8Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_8x8_LDR, swizzle};
        case TextureFormat::Astc8x8Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_8x8_sRGB, swizzle};
        case TextureFormat::Astc10x5Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_10x5_LDR, swizzle};
        case TextureFormat::Astc10x5Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_10x5_sRGB, swizzle};
        case TextureFormat::Astc10x6Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_10x6_LDR, swizzle};
        case TextureFormat::Astc10x6Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_10x6_sRGB, swizzle};
        case TextureFormat::Astc10x8Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_10x8_LDR, swizzle};
        case TextureFormat::Astc10x8Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_10x8_sRGB, swizzle};
        case TextureFormat::Astc10x10Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_10x10_LDR, swizzle};
        case TextureFormat::Astc10x10Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_10x10_sRGB, swizzle};
        case TextureFormat::Astc12x10Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_12x10_LDR, swizzle};
        case TextureFormat::Astc12x10Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_12x10_sRGB, swizzle};
        case TextureFormat::Astc12x12Unorm:
            return MetalSampledFormat{MTLPixelFormatASTC_12x12_LDR, swizzle};
        case TextureFormat::Astc12x12Srgb:
            return MetalSampledFormat{MTLPixelFormatASTC_12x12_sRGB, swizzle};
    }
    return unsupported(format);
}

std::variant<MetalSampledFormat, graphics::TextureError> metal_sampled_format(
    const graphics::TextureDescriptor& texture) noexcept {
    if (graphics::texture_format_info(texture.format).depth) {
        // The first R/G selector chooses the aspect. D24S8 reverses the
        // logical channels: R selects stencil and G selects depth.
        auto selected = graphics::TextureSwizzle::Zero;
        for (const auto component : texture.swizzle) {
            if (component == graphics::TextureSwizzle::Red ||
                component == graphics::TextureSwizzle::Green) {
                selected = component;
                break;
            }
        }
        const bool stencil = texture.format == TextureFormat::D24UnormS8Uint
            ? selected == graphics::TextureSwizzle::Red
            : selected == graphics::TextureSwizzle::Green;
        if (stencil) return unsupported(texture.format);
        return metal_pixel_format(texture.format);
    }
    // The TIC swizzle selects guest channels; the format mapping says which
    // Metal channel holds each guest channel, so the view swizzle composes
    // them. Zero and both one forms map to Metal's constant selectors, which
    // yield the constant in the texture's own type.
    auto mapped = metal_pixel_format(texture.format);
    auto* sampled = std::get_if<MetalSampledFormat>(&mapped);
    if (!sampled) return mapped;
    const auto format = sampled->swizzle;
    const auto compose = [&](graphics::TextureSwizzle source) {
        switch (source) {
        case graphics::TextureSwizzle::Zero: return MTLTextureSwizzleZero;
        case graphics::TextureSwizzle::Red: return format.red;
        case graphics::TextureSwizzle::Green: return format.green;
        case graphics::TextureSwizzle::Blue: return format.blue;
        case graphics::TextureSwizzle::Alpha: return format.alpha;
        case graphics::TextureSwizzle::OneSInt:
        case graphics::TextureSwizzle::OneFloat: return MTLTextureSwizzleOne;
        }
        return MTLTextureSwizzleZero;
    };
    sampled->swizzle = MTLTextureSwizzleChannelsMake(
        compose(texture.swizzle[0]), compose(texture.swizzle[1]),
        compose(texture.swizzle[2]), compose(texture.swizzle[3]));
    return mapped;
}

std::variant<std::vector<std::byte>, graphics::TextureError> metal_depth_upload(
    TextureFormat format, std::span<const std::byte> bytes) {
    const auto info = graphics::texture_format_info(format);
    if (!info.depth || bytes.size() % info.bytes_per_block != 0)
        return unsupported(format);
    if (format == TextureFormat::D16Unorm || format == TextureFormat::D32Float)
        return std::vector<std::byte>(bytes.begin(), bytes.end());
    const auto count = bytes.size() / info.bytes_per_block;
    std::vector<std::byte> output(count * sizeof(float));
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t word = 0;
        for (unsigned b = 0; b < 4; ++b)
            word |= std::uint32_t(std::to_integer<unsigned char>(bytes[i * info.bytes_per_block + b])) << (b * 8);
        if (format != TextureFormat::D32FloatS8Uint) {
            // All packed D24 guest layouts store stencil/padding in the low byte.
            const float depth = static_cast<float>(word >> 8) / 16777215.0f;
            word = std::bit_cast<std::uint32_t>(depth);
        }
        std::memcpy(output.data() + i * sizeof(word), &word, sizeof(word));
    }
    return output;
}

// Persistent packed depth/stencil storage uses separate blit planes.
std::variant<MTLPixelFormat, graphics::TextureError> metal_storage_format(
    TextureFormat format) noexcept {
    if (graphics::texture_format_info(format).stencil ||
        format == TextureFormat::X8UintD24Unorm)
        return MTLPixelFormatDepth32Float_Stencil8;
    auto mapped = metal_pixel_format(format);
    if (auto* error = std::get_if<graphics::TextureError>(&mapped)) return *error;
    return std::get<MetalSampledFormat>(mapped).pixel_format;
}

std::variant<MetalTexturePlanes, graphics::TextureError> metal_texture_planes(
    TextureFormat format, std::span<const std::byte> bytes) {
    const auto info = graphics::texture_format_info(format);
    if (!info.bytes_per_block || bytes.size() % info.bytes_per_block)
        return unsupported(format);
    MetalTexturePlanes planes;
    if (info.depth) {
        auto depth = metal_depth_upload(format, bytes);
        if (auto* error = std::get_if<graphics::TextureError>(&depth)) return *error;
        planes.color_or_depth = std::get<std::vector<std::byte>>(std::move(depth));
    } else planes.color_or_depth.assign(bytes.begin(), bytes.end());
    if (info.stencil) {
        const auto count = bytes.size() / info.bytes_per_block;
        planes.stencil.resize(count);
        const auto offset = format == TextureFormat::D32FloatS8Uint ? 4u : 0u;
        for (std::size_t i = 0; i < count; ++i)
            planes.stencil[i] = bytes[i * info.bytes_per_block + offset];
    }
    return planes;
}

bool metal_texture_planes_identity(TextureFormat format) noexcept {
    const auto info = graphics::texture_format_info(format);
    if (info.bytes_per_block == 0 || info.stencil) return false;
    // Depth alone in 16-bit unorm or 32-bit float uploads its bytes unchanged
    // (metal_depth_upload); packed 24-bit depth is converted to float.
    return !info.depth || format == TextureFormat::D16Unorm || format == TextureFormat::D32Float;
}

std::variant<std::vector<std::byte>, graphics::TextureError> metal_merge_texture_planes(
    TextureFormat format, const MetalTexturePlanes& planes,
    std::span<const std::byte> original) {
    auto baseline_result = metal_texture_planes(format, original);
    auto* baseline = std::get_if<MetalTexturePlanes>(&baseline_result);
    if (!baseline || baseline->color_or_depth.size() != planes.color_or_depth.size() ||
        baseline->stencil.size() != planes.stencil.size()) return unsupported(format);
    const auto info = graphics::texture_format_info(format);
    if (!info.depth) return planes.color_or_depth;
    std::vector<std::byte> result(original.begin(), original.end());
    const auto count = original.size() / info.bytes_per_block;
    const auto depth_bytes = format == TextureFormat::D16Unorm ? 2u : 4u;
    for (std::size_t i = 0; i < count; ++i) {
        const auto* source = planes.color_or_depth.data() + i * depth_bytes;
        auto* destination = result.data() + i * info.bytes_per_block;
        // Preserve the original packed value exactly if the depth is unchanged.
        if (std::memcmp(source, baseline->color_or_depth.data() + i * depth_bytes,
                        depth_bytes) != 0) {
            if (format == TextureFormat::D24UnormS8Uint ||
                format == TextureFormat::S8UintD24Unorm ||
                format == TextureFormat::X8UintD24Unorm) {
                float depth;
                std::memcpy(&depth, source, sizeof(depth));
                if (!std::isfinite(depth)) return unsupported(format);
                const auto packed = static_cast<std::uint32_t>(std::llround(
                    std::clamp(static_cast<double>(depth), 0.0, 1.0) * 16777215.0));
                destination[1] = std::byte(packed & 255);
                destination[2] = std::byte((packed >> 8) & 255);
                destination[3] = std::byte((packed >> 16) & 255);
            } else std::memcpy(destination, source, depth_bytes);
        }
        if (info.stencil)
            destination[format == TextureFormat::D32FloatS8Uint ? 4 : 0] = planes.stencil[i];
    }
    return result;
}

// TSC samplers map field-by-field onto MTLSamplerDescriptor. OpenGL clamp and mirror-clamp-to-border have no Metal equivalent and use
// clamp fallbacks, while mirror-clamp uses Metal's native mirror-clamp-to-edge
// because that mode preserves the requested
// mirroring exactly. Border colours compare exactly against Metal's three
// constants. Other finite colours use transparent black native state only for
// the bounded shader-side custom-border profile. Metal 26 samplers carry LOD
// bias and min/max reduction natively, so both map directly.
namespace {

graphics::TextureError sampler_unsupported(
    graphics::TextureField field, std::uint32_t word, std::uint64_t value,
    std::uint64_t expected = 0) noexcept {
    return {graphics::TextureErrorCode::UnsupportedSamplerState, field, word,
            value, expected, graphics::MemoryErrorCode::None};
}

MTLSamplerAddressMode metal_address_mode(
    graphics::SamplerWrap wrap) noexcept {
    switch (wrap) {
        case graphics::SamplerWrap::Repeat:
            return MTLSamplerAddressModeRepeat;
        case graphics::SamplerWrap::MirroredRepeat:
            return MTLSamplerAddressModeMirrorRepeat;
        case graphics::SamplerWrap::ClampToEdge:
        case graphics::SamplerWrap::Clamp:
            return MTLSamplerAddressModeClampToEdge;
        case graphics::SamplerWrap::ClampToBorder:
        case graphics::SamplerWrap::MirrorClampToBorder:
            return MTLSamplerAddressModeClampToBorderColor;
        case graphics::SamplerWrap::MirrorClampToEdge:
        case graphics::SamplerWrap::MirrorClamp:
            return MTLSamplerAddressModeMirrorClampToEdge;
    }
    return MTLSamplerAddressModeClampToEdge;
}

// Matches a decoded border colour against the three constant colours Metal
// exposes, using exact component equality. Returns false for any other colour.
bool metal_border_color(const std::array<float, 4>& color,
                        MTLSamplerBorderColor& out) noexcept {
    const auto matches = [&](float r, float g, float b, float a) {
        return color[0] == r && color[1] == g && color[2] == b &&
               color[3] == a;
    };
    if (matches(0.0f, 0.0f, 0.0f, 0.0f)) {
        out = MTLSamplerBorderColorTransparentBlack;
        return true;
    }
    if (matches(0.0f, 0.0f, 0.0f, 1.0f)) {
        out = MTLSamplerBorderColorOpaqueBlack;
        return true;
    }
    if (matches(1.0f, 1.0f, 1.0f, 1.0f)) {
        out = MTLSamplerBorderColorOpaqueWhite;
        return true;
    }
    return false;
}

graphics::TextureError unsupported_border_color(
    const graphics::SamplerDescriptor& sampler) noexcept {
    const std::uint64_t low = std::bit_cast<std::uint32_t>(
        sampler.border_color[0]);
    const std::uint64_t high = std::bit_cast<std::uint32_t>(
        sampler.border_color[1]);
    return sampler_unsupported(
        graphics::TextureField::BorderColor, 4, low | (high << 32),
        std::bit_cast<std::uint32_t>(sampler.border_color[2]) |
            (std::uint64_t{std::bit_cast<std::uint32_t>(
                 sampler.border_color[3])}
             << 32));
}

bool finite_border_color(const std::array<float, 4>& color) noexcept {
    return std::all_of(color.begin(), color.end(),
                       [](float component) { return std::isfinite(component); });
}

float srgb_to_linear(std::uint8_t value) noexcept {
    const float encoded = static_cast<float>(value) / 255.0f;
    return encoded <= 0.04045f
               ? encoded / 12.92f
               : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

}  // namespace

bool operator==(const MetalSamplerProperties& left,
                const MetalSamplerProperties& right) noexcept {
    return left.address_u == right.address_u &&
           left.address_v == right.address_v &&
           left.address_r == right.address_r &&
           left.mag_filter == right.mag_filter &&
           left.min_filter == right.min_filter &&
           left.mip_filter == right.mip_filter &&
           left.reduction_mode == right.reduction_mode &&
           left.compare_function == right.compare_function &&
           left.border_color == right.border_color &&
           left.lod_bias == right.lod_bias &&
           left.lod_min_clamp == right.lod_min_clamp &&
           left.lod_max_clamp == right.lod_max_clamp &&
           left.max_anisotropy == right.max_anisotropy &&
           left.normalized_coordinates == right.normalized_coordinates;
}

std::variant<MetalSamplerProperties, graphics::TextureError>
metal_sampler_properties(const graphics::SamplerDescriptor& sampler,
                         bool srgb_texture) noexcept {
    MetalSamplerProperties result;
    result.address_u = metal_address_mode(sampler.wrap[0]);
    result.address_v = metal_address_mode(sampler.wrap[1]);
    result.address_r = metal_address_mode(sampler.wrap[2]);
    switch (sampler.magnification) {
        case graphics::SamplerFilter::Nearest:
            result.mag_filter = MTLSamplerMinMagFilterNearest;
            break;
        case graphics::SamplerFilter::Linear:
            result.mag_filter = MTLSamplerMinMagFilterLinear;
            break;
    }
    switch (sampler.minification) {
        case graphics::SamplerFilter::Nearest:
            result.min_filter = MTLSamplerMinMagFilterNearest;
            break;
        case graphics::SamplerFilter::Linear:
            result.min_filter = MTLSamplerMinMagFilterLinear;
            break;
    }
    switch (sampler.mip_filter) {
        case graphics::SamplerMipFilter::None:
            result.mip_filter = MTLSamplerMipFilterNotMipmapped;
            break;
        case graphics::SamplerMipFilter::Nearest:
            result.mip_filter = MTLSamplerMipFilterNearest;
            break;
        case graphics::SamplerMipFilter::Linear:
            result.mip_filter = MTLSamplerMipFilterLinear;
            break;
    }
    switch (sampler.reduction) {
        case graphics::SamplerReduction::WeightedAverage:
            result.reduction_mode = MTLSamplerReductionModeWeightedAverage;
            break;
        case graphics::SamplerReduction::Minimum:
            result.reduction_mode = MTLSamplerReductionModeMinimum;
            break;
        case graphics::SamplerReduction::Maximum:
            result.reduction_mode = MTLSamplerReductionModeMaximum;
            break;
    }
    if (sampler.depth_compare) {
        // The TSC comparison encoding is the ordered Metal sequence.
        result.compare_function = static_cast<MTLCompareFunction>(
            static_cast<std::uint32_t>(sampler.compare_function));
    }
    if (sampler.float_coordinate_normalization) {
        // This TSC word-1 bit is decoded but never consumed, so
        // its polarity is unestablished; Metal's only alternative state is
        // unnormalized coordinates, which would silently change every sample.
        return sampler_unsupported(
            graphics::TextureField::FloatCoordinateNormalization, 1,
            sampler.raw_words[1] & (1u << 25));
    }
    if (sampler.trilinear_optimization != 0.0f) {
        // The LodSnap field is decoded but no host backend consumes it and Metal has no equivalent control.
        return sampler_unsupported(
            graphics::TextureField::TrilinearOptimization, 1,
            (sampler.raw_words[1] >> 26) & 0x1fu, 0);
    }
    if (!finite_border_color(sampler.border_color)) {
        return unsupported_border_color(sampler);
    }
    const bool border_active = std::any_of(sampler.wrap.begin(), sampler.wrap.end(), [](auto wrap) {
        return wrap == graphics::SamplerWrap::ClampToBorder ||
               wrap == graphics::SamplerWrap::MirrorClampToBorder;
    });
    const auto effective_border = border_active
        ? metal_custom_border_color(sampler, srgb_texture) : sampler.border_color;
    if (!metal_border_color(effective_border, result.border_color)) {
        for (const auto wrap : sampler.wrap) {
            if (wrap != graphics::SamplerWrap::ClampToBorder) {
                return sampler_unsupported(
                    graphics::TextureField::WrapMode, 0,
                    static_cast<std::uint32_t>(wrap),
                    static_cast<std::uint32_t>(
                        graphics::SamplerWrap::ClampToBorder));
            }
        }
        if (sampler.reduction !=
            graphics::SamplerReduction::WeightedAverage) {
            return sampler_unsupported(
                graphics::TextureField::ReductionFilter, 1,
                static_cast<std::uint32_t>(sampler.reduction),
                static_cast<std::uint32_t>(
                    graphics::SamplerReduction::WeightedAverage));
        }
        if (sampler.depth_compare) {
            return sampler_unsupported(graphics::TextureField::DepthCompare,
                                       0, 1, 0);
        }
        if (sampler.max_anisotropy != 1.0f) {
            return sampler_unsupported(
                graphics::TextureField::Anisotropy, 0,
                std::bit_cast<std::uint32_t>(sampler.max_anisotropy),
                std::bit_cast<std::uint32_t>(1.0f));
        }
        result.border_color = MTLSamplerBorderColorTransparentBlack;
        result.custom_border = true;
    }
    result.lod_bias = sampler.lod_bias;
    result.lod_min_clamp = sampler.min_lod;
    result.lod_max_clamp = sampler.max_lod;
    result.max_anisotropy = sampler.max_anisotropy;
    result.normalized_coordinates = true;
    return result;
}

std::array<float, 4> metal_custom_border_color(
    const graphics::SamplerDescriptor& sampler, bool srgb_texture) noexcept {
    auto color = sampler.border_color;
    if (srgb_texture && sampler.srgb_border_conversion) {
        for (unsigned component = 0; component < 3; ++component) {
            color[component] =
                srgb_to_linear(sampler.srgb_border_color[component]);
        }
    }
    return color;
}

MTLSamplerDescriptor* metal_sampler_descriptor(
    const MetalSamplerProperties& properties) noexcept {
    auto* descriptor = [MTLSamplerDescriptor new];
    descriptor.sAddressMode = properties.address_u;
    descriptor.tAddressMode = properties.address_v;
    descriptor.rAddressMode = properties.address_r;
    descriptor.magFilter = properties.mag_filter;
    descriptor.minFilter = properties.min_filter;
    descriptor.mipFilter = properties.mip_filter;
    descriptor.reductionMode = properties.reduction_mode;
    descriptor.compareFunction = properties.compare_function;
    descriptor.borderColor = properties.border_color;
    descriptor.lodBias = properties.lod_bias;
    descriptor.lodMinClamp = properties.lod_min_clamp;
    descriptor.lodMaxClamp = properties.lod_max_clamp;
    descriptor.maxAnisotropy = static_cast<NSUInteger>(properties.max_anisotropy);
    descriptor.normalizedCoordinates = properties.normalized_coordinates;
    descriptor.supportArgumentBuffers = YES;
    return descriptor;
}

}  // namespace armigo::platform
