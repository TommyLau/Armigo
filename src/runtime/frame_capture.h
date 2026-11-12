// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/image_readback.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace armigo::runtime {

// A nearest-sample RGB capture, preserving the source's channel bytes. Zero
// retains the original extent; a positive maximum width never upscales it.
inline std::string encode_capture_ppm(const graphics::Rgba8Image& image, std::uint32_t maximum_width) {
    const auto source_pixels = std::uint64_t{image.width} * image.height;
    if (!image.width || !image.height || image.pixels.size() % 4 || image.pixels.size() / 4 != source_pixels)
        throw std::invalid_argument("Capture readback dimensions do not match its bytes.");
    const auto width = maximum_width ? std::min(image.width, maximum_width) : image.width;
    const auto height = std::max<std::uint32_t>(1, std::uint64_t{image.height} * width / image.width);
    std::string bytes = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    const auto offset = bytes.size();
    bytes.resize(offset + std::size_t{width} * height * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        const auto source_y = std::uint64_t{y} * image.height / height;
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto source_x = std::uint64_t{x} * image.width / width;
            const auto source = (source_y * image.width + source_x) * 4;
            const auto target = offset + (std::size_t{y} * width + x) * 3;
            for (unsigned channel = 0; channel < 3; ++channel)
                bytes[target + channel] = static_cast<char>(image.pixels[source + channel]);
        }
    }
    return bytes;
}

} // namespace armigo::runtime
