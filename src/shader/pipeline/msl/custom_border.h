// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <string_view>

namespace armigo::shader::pipeline::msl {
// Metal 3 direct-layout resource handles occupy eight bytes. Explicit padding
// keeps the shared CPU/MSL record at 64 bytes with float4 alignment.
inline constexpr std::string_view custom_border_msl = R"metal(
struct ArmigoCustomBorder {
    texture2d<float> coverage;
    ulong reserved;
    float4 color;
    uint4 swizzle;
    uint enabled;
    uint padding[3];
};
inline float4 armigo_custom_border(float4 black, float coverage,
                                  const device ArmigoCustomBorder& support) {
    const float4 color = black + (1.0f - coverage) * support.color;
    const float channels[6] = {color.x, color.y, color.z, color.w, 0.0f, 1.0f};
    return float4(channels[support.swizzle.x], channels[support.swizzle.y],
                  channels[support.swizzle.z], channels[support.swizzle.w]);
}
)metal";
} // namespace armigo::shader::pipeline::msl
