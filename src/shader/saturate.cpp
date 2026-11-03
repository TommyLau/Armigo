// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/saturate.h>

namespace armigo::shader {

// For nonnegative inputs the unsigned bit pattern orders exactly like the
// values, so one comparison against the encoding of 1.0 (0x3f800000) clamps.
std::uint32_t saturate_float32(std::uint32_t bits) noexcept {
    if ((bits & 0x80000000u) != 0) return 0;
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return (bits & 0x007fffffu) != 0 ? 0 : 0x3f800000u;
    }
    return bits > 0x3f800000u ? 0x3f800000u : bits;
}

std::string emit_saturate_msl() {
    return R"msl(inline uint armigo_saturate_float32(uint bits) {
    if ((bits & 0x80000000u) != 0u) return 0u;
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return (bits & 0x007fffffu) != 0u ? 0u : 0x3f800000u;
    return bits > 0x3f800000u ? 0x3f800000u : bits;
}
)msl";
}

}  // namespace armigo::shader
