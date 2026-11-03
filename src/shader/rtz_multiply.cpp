// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/rtz_multiply.h>

namespace armigo::shader {

std::uint32_t multiply_rtz_ftz(std::uint32_t lhs_bits,
                             std::uint32_t rhs_bits) noexcept {
    const auto sign = (lhs_bits ^ rhs_bits) & 0x80000000u;
    const auto lhs_exponent = (lhs_bits >> 23) & 255u;
    const auto rhs_exponent = (rhs_bits >> 23) & 255u;
    const auto lhs_fraction = lhs_bits & 0x007fffffu;
    const auto rhs_fraction = rhs_bits & 0x007fffffu;
    if ((lhs_exponent == 255 && lhs_fraction != 0) ||
        (rhs_exponent == 255 && rhs_fraction != 0)) {
        return 0x7fc00000u;
    }
    if (lhs_exponent == 255 || rhs_exponent == 255) {
        return lhs_exponent == 0 || rhs_exponent == 0
            ? 0x7fc00000u : sign | 0x7f800000u;
    }
    if (lhs_exponent == 0 || rhs_exponent == 0) {
        return sign;
    }

    // Each normal significand has 24 bits, so the exact product fits in 48.
    // Its leading bit is at 46 or 47; discarding lower bits is RTZ for either
    // sign. The normalized exponent is decided before applying output FTZ.
    const std::uint64_t product =
        std::uint64_t{lhs_fraction | 0x00800000u} *
        std::uint64_t{rhs_fraction | 0x00800000u};
    const auto carry = static_cast<std::uint32_t>(product >> 47);
    const int exponent = static_cast<int>(lhs_exponent) +
                         static_cast<int>(rhs_exponent) - 127 +
                         static_cast<int>(carry);
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 255) {
        return sign | 0x7f7fffffu;
    }
    const auto fraction = static_cast<std::uint32_t>(product >> (23 + carry)) &
                          0x007fffffu;
    return sign | (static_cast<std::uint32_t>(exponent) << 23) | fraction;
}

std::string emit_rtz_ftz_multiply_msl() {
    return R"msl(inline uint armigo_mul_rtz_ftz(uint lhs, uint rhs) {
    const uint sign = (lhs ^ rhs) & 0x80000000u;
    const uint le = (lhs >> 23) & 255u;
    const uint re = (rhs >> 23) & 255u;
    const uint lf = lhs & 0x007fffffu;
    const uint rf = rhs & 0x007fffffu;
    if ((le == 255u && lf != 0u) || (re == 255u && rf != 0u)) {
        return 0x7fc00000u;
    }
    if (le == 255u || re == 255u) {
        return le == 0u || re == 0u ? 0x7fc00000u : sign | 0x7f800000u;
    }
    if (le == 0u || re == 0u) {
        return sign;
    }
    const ulong product = ulong(lf | 0x00800000u) * ulong(rf | 0x00800000u);
    const uint carry = uint(product >> 47);
    const int exponent = int(le) + int(re) - 127 + int(carry);
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 255) {
        return sign | 0x7f7fffffu;
    }
    const uint fraction = uint(product >> (23u + carry)) & 0x007fffffu;
    return sign | (uint(exponent) << 23) | fraction;
}
)msl";
}

}  // namespace armigo::shader
