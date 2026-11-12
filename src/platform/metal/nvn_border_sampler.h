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
#include <array>
#include <cstdint>
#include <cstddef>

namespace armigo::platform::metal::nvn_driver {

// A primary sampler and exact color for an admitted coverage-v1 sampling pair.
// Custom colors use a transparent-black primary; the reflected shader combines
// it with a matching coverage image. Ordinary lookups never use this variant.
struct CoverageSampler {
    id<MTLSamplerState> sampler = nil;
    std::array<float, 4> color{};
    bool custom = false;
};

// Byte layout shared with the existing coverage-v1 MSL declaration.
struct alignas(16) CoverageBorderRecord {
    std::uint64_t coverage = 0, reserved = 0;
    std::array<float, 4> color{};
    std::array<std::uint32_t, 4> swizzle{};
    std::uint32_t enabled = 0, padding[3]{};
};
static_assert(sizeof(CoverageBorderRecord) == 64);
static_assert(offsetof(CoverageBorderRecord, color) == 16);
static_assert(offsetof(CoverageBorderRecord, swizzle) == 32);
static_assert(offsetof(CoverageBorderRecord, enabled) == 48);

} // namespace armigo::platform::metal::nvn_driver
