// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "frame_capture.h"
#include <cstdio>

int main() {
    armigo::graphics::Rgba8Image image;
    image.width = 4;
    image.height = 2;
    for (unsigned value = 1; value <= 32; ++value) image.pixels.push_back(std::byte(value));
    const auto reduced = armigo::runtime::encode_capture_ppm(image, 2);
    const std::string header = "P6\n2 1\n255\n";
    const std::string expected = header + std::string({1, 2, 3, 9, 10, 11});
    if (reduced != expected) return 1;
    const auto original = armigo::runtime::encode_capture_ppm(image, 0);
    if (original != armigo::runtime::encode_capture_ppm(image, 8) ||
        original.size() != std::string("P6\n4 2\n255\n").size() + 24) return 1;
    image.pixels.pop_back();
    bool rejected = false;
    try { static_cast<void>(armigo::runtime::encode_capture_ppm(image, 2)); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 1;
    std::puts("Capture encoding preserves authored RGB samples and bounds output size.");
}
