// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "icon.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

namespace armigo::pack {

std::variant<std::string, std::vector<std::byte>> jpeg_to_icns(std::span<const std::byte> jpeg) {
    if (jpeg.size() < 2 || jpeg[0] != std::byte{0xff} || jpeg[1] != std::byte{0xd8}) {
        return std::string("The control icon is not a JPEG.");
    }
    @autoreleasepool {
        NSData* input = [NSData dataWithBytes:jpeg.data() length:jpeg.size()];
        CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)input, nullptr);
        if (!source) {
            return std::string("The control icon JPEG could not be read.");
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
        CFRelease(source);
        if (!image) {
            return std::string("The control icon JPEG has no image.");
        }
        NSMutableData* output = [NSMutableData data];
        UTType* icns = [UTType typeWithIdentifier:@"com.apple.icns"];
        if (!icns) {
            CGImageRelease(image);
            return std::string("ImageIO has no icns type.");
        }
        CGImageDestinationRef destination = CGImageDestinationCreateWithData(
            (__bridge CFMutableDataRef)output, (__bridge CFStringRef)icns.identifier, 1, nullptr);
        if (!destination) {
            CGImageRelease(image);
            return std::string("An icns destination could not be created.");
        }
        CGImageDestinationAddImage(destination, image, nullptr);
        const bool wrote = CGImageDestinationFinalize(destination);
        CFRelease(destination);
        CGImageRelease(image);
        if (!wrote || output.length < 8) {
            return std::string("The control icon could not be written as icns.");
        }
        auto bytes = static_cast<const std::byte*>(output.bytes);
        return std::vector<std::byte>(bytes, bytes + output.length);
    }
}

std::variant<std::string, std::vector<std::byte>> jpeg_to_ios_png(std::span<const std::byte> jpeg) {
    if (jpeg.size() < 2 || jpeg[0] != std::byte{0xff} || jpeg[1] != std::byte{0xd8}) {
        return std::string("The control icon is not a JPEG.");
    }
    @autoreleasepool {
        NSData* input = [NSData dataWithBytes:jpeg.data() length:jpeg.size()];
        CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)input, nullptr);
        if (!source) {
            return std::string("The control icon JPEG could not be read.");
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
        CFRelease(source);
        if (!image) {
            return std::string("The control icon JPEG has no image.");
        }
        CGColorSpaceRef color = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef context = CGBitmapContextCreate(nullptr, 1024, 1024, 8, 4096, color,
                                                    kCGImageAlphaNoneSkipLast);
        CGColorSpaceRelease(color);
        if (!context) {
            CGImageRelease(image);
            return std::string("The icon bitmap could not be allocated.");
        }
        CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
        CGContextDrawImage(context, CGRectMake(0, 0, 1024, 1024), image);
        CGImageRelease(image);
        CGImageRef resized = CGBitmapContextCreateImage(context);
        CGContextRelease(context);
        if (!resized) {
            return std::string("The icon bitmap could not be rendered.");
        }
        NSMutableData* output = [NSMutableData data];
        CGImageDestinationRef destination = CGImageDestinationCreateWithData(
            (__bridge CFMutableDataRef)output, CFSTR("public.png"), 1, nullptr);
        if (!destination) {
            CGImageRelease(resized);
            return std::string("The PNG destination could not be created.");
        }
        CGImageDestinationAddImage(destination, resized, nullptr);
        const bool wrote = CGImageDestinationFinalize(destination);
        CFRelease(destination);
        CGImageRelease(resized);
        if (!wrote) {
            return std::string("The control icon could not be written as PNG.");
        }
        const auto* bytes = static_cast<const std::byte*>(output.bytes);
        return std::vector<std::byte>(bytes, bytes + output.length);
    }
}

} // namespace armigo::pack
