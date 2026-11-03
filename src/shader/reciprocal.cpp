// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/reciprocal.h>

namespace armigo::shader {
namespace {

std::uint32_t rounded_quotient(std::uint64_t numerator,
                               std::uint32_t denominator) noexcept {
    const auto quotient = numerator / denominator;
    const auto twice_remainder = (numerator % denominator) * 2;
    const bool increment = twice_remainder > denominator ||
        (twice_remainder == denominator && (quotient & 1) != 0);
    return static_cast<std::uint32_t>(quotient + (increment ? 1 : 0));
}

}  // namespace

std::uint32_t reciprocal_rne(std::uint32_t input_bits) noexcept {
    const auto sign = input_bits & 0x80000000u;
    const auto fraction = input_bits & 0x007fffffu;
    int exponent = static_cast<int>((input_bits >> 23) & 255u);
    if (exponent == 255) return fraction != 0 ? 0x7fc00000u : sign;
    if (exponent == 0 && fraction == 0) return sign | 0x7f800000u;

    // Normalize without flushing: x = significand * 2^(exponent - 150),
    // with 2^23 <= significand < 2^24 and -22 <= exponent <= 254.
    auto significand = fraction;
    if (exponent == 0) {
        exponent = 1;
        while (significand < 0x00800000u) {
            significand <<= 1;
            --exponent;
        }
    } else {
        significand |= 0x00800000u;
    }
    if (significand == 0x00800000u) {
        const int result_exponent = 254 - exponent;
        if (result_exponent >= 255) return sign | 0x7f800000u;
        // Only input exponent 254 gives a subnormal reciprocal power of two.
        if (result_exponent == 0) return sign | 0x00400000u;
        return sign | (static_cast<std::uint32_t>(result_exponent) << 23);
    }

    const int result_exponent = 253 - exponent;
    if (result_exponent >= 255) return sign | 0x7f800000u;
    if (result_exponent <= 0) {
        // Round directly in subnormal units (2^-149), avoiding an intermediate
        // normal significand rounding. The numerator shift is either 45 or 46.
        const auto numerator = std::uint64_t{1} << (299 - exponent);
        return sign | rounded_quotient(numerator, significand);
    }
    const auto rounded = rounded_quotient(std::uint64_t{1} << 47, significand);
    // For significand >= 2^23+1, the quotient is strictly below 2^24-1;
    // rounding cannot carry into another exponent. Powers of two were handled.
    return sign | (static_cast<std::uint32_t>(result_exponent) << 23) |
           (rounded & 0x007fffffu);
}

std::string emit_reciprocal_msl() {
    return R"msl(inline uint armigo_reciprocal_rne(uint input_bits) {
    const uint sign = input_bits & 0x80000000u;
    const uint fraction = input_bits & 0x007fffffu;
    int exponent = int((input_bits >> 23) & 255u);
    if (exponent == 255) return fraction != 0u ? 0x7fc00000u : sign;
    if (exponent == 0 && fraction == 0u) return sign | 0x7f800000u;
    uint significand = fraction;
    if (exponent == 0) {
        exponent = 1;
        while (significand < 0x00800000u) {
            significand <<= 1;
            --exponent;
        }
    } else {
        significand |= 0x00800000u;
    }
    if (significand == 0x00800000u) {
        const int result_exponent = 254 - exponent;
        if (result_exponent >= 255) return sign | 0x7f800000u;
        if (result_exponent == 0) return sign | 0x00400000u;
        return sign | (uint(result_exponent) << 23);
    }
    const int result_exponent = 253 - exponent;
    if (result_exponent >= 255) return sign | 0x7f800000u;
    const bool subnormal = result_exponent <= 0;
    const ulong numerator = ulong(1) << (subnormal ? 299 - exponent : 47);
    const ulong quotient = numerator / significand;
    const ulong twice_remainder = (numerator % significand) * 2;
    const bool increment = twice_remainder > significand ||
        (twice_remainder == significand && (quotient & 1) != 0);
    const uint rounded = uint(quotient + (increment ? 1 : 0));
    if (subnormal) return sign | rounded;
    return sign | (uint(result_exponent) << 23) | (rounded & 0x007fffffu);
}
)msl";
}

}  // namespace armigo::shader
