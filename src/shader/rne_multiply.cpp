// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/rne_multiply.h>

namespace armigo::shader {
namespace {

// value has at most 48 bits and shift is positive. Larger shifts are strictly
// below half a unit, so return zero without performing an oversized C++ shift.
std::uint32_t round_product(std::uint64_t value, unsigned shift) noexcept {
    if (shift > 48) return 0;
    const auto quotient = value >> shift;
    const auto remainder = value & ((std::uint64_t{1} << shift) - 1);
    const auto midpoint = std::uint64_t{1} << (shift - 1);
    return static_cast<std::uint32_t>(quotient +
        (remainder > midpoint || (remainder == midpoint && (quotient & 1) != 0) ? 1 : 0));
}

void normalize(std::uint32_t& significand, int& exponent) noexcept {
    if (exponent != 0) {
        significand |= 0x00800000u;
        return;
    }
    exponent = 1;
    while (significand < 0x00800000u) {
        significand <<= 1;
        --exponent;
    }
}

}  // namespace

std::uint32_t multiply_rne(std::uint32_t lhs_bits,
                           std::uint32_t rhs_bits) noexcept {
    const auto sign = (lhs_bits ^ rhs_bits) & 0x80000000u;
    int lhs_exponent = static_cast<int>((lhs_bits >> 23) & 255u);
    int rhs_exponent = static_cast<int>((rhs_bits >> 23) & 255u);
    auto lhs = lhs_bits & 0x007fffffu;
    auto rhs = rhs_bits & 0x007fffffu;
    if ((lhs_exponent == 255 && lhs != 0) || (rhs_exponent == 255 && rhs != 0)) {
        return 0x7fc00000u;
    }
    const bool lhs_zero = lhs_exponent == 0 && lhs == 0;
    const bool rhs_zero = rhs_exponent == 0 && rhs == 0;
    if (lhs_exponent == 255 || rhs_exponent == 255) {
        return lhs_zero || rhs_zero ? 0x7fc00000u : sign | 0x7f800000u;
    }
    if (lhs_zero || rhs_zero) return sign;
    normalize(lhs, lhs_exponent);
    normalize(rhs, rhs_exponent);

    // x*y = product * 2^(lhs_exponent + rhs_exponent - 300).
    // The exact product's leading bit is either 46 or 47.
    const std::uint64_t product = std::uint64_t{lhs} * rhs;
    const auto carry = static_cast<unsigned>(product >> 47);
    int exponent = lhs_exponent + rhs_exponent - 127 + static_cast<int>(carry);
    if (exponent >= 255) return sign | 0x7f800000u;
    if (exponent <= 0) {
        // Round directly in units of 2^-149. This can correctly carry the
        // fraction into the smallest normal encoding, without double rounding.
        const auto shift = static_cast<unsigned>(151 - lhs_exponent - rhs_exponent);
        return sign | round_product(product, shift);
    }
    auto rounded = round_product(product, 23 + carry);
    if (rounded == 0x01000000u) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent >= 255) return sign | 0x7f800000u;
    return sign | (static_cast<std::uint32_t>(exponent) << 23) | (rounded & 0x007fffffu);
}

std::string emit_rne_multiply_msl() {
    return R"msl(inline uint armigo_round_product_rne(ulong value, uint shift) {
    if (shift > 48u) return 0u;
    const ulong quotient = value >> shift;
    const ulong remainder = value & ((ulong(1) << shift) - 1);
    const ulong midpoint = ulong(1) << (shift - 1);
    return uint(quotient +
        (remainder > midpoint || (remainder == midpoint && (quotient & 1) != 0) ? 1 : 0));
}
inline uint armigo_multiply_rne(uint lhs_bits, uint rhs_bits) {
    const uint sign = (lhs_bits ^ rhs_bits) & 0x80000000u;
    int le = int((lhs_bits >> 23) & 255u);
    int re = int((rhs_bits >> 23) & 255u);
    uint lhs = lhs_bits & 0x007fffffu;
    uint rhs = rhs_bits & 0x007fffffu;
    if ((le == 255 && lhs != 0u) || (re == 255 && rhs != 0u)) return 0x7fc00000u;
    const bool lzero = le == 0 && lhs == 0u;
    const bool rzero = re == 0 && rhs == 0u;
    if (le == 255 || re == 255) return lzero || rzero ? 0x7fc00000u : sign | 0x7f800000u;
    if (lzero || rzero) return sign;
    if (le == 0) {
        le = 1;
        while (lhs < 0x00800000u) { lhs <<= 1; --le; }
    } else {
        lhs |= 0x00800000u;
    }
    if (re == 0) {
        re = 1;
        while (rhs < 0x00800000u) { rhs <<= 1; --re; }
    } else {
        rhs |= 0x00800000u;
    }
    const ulong product = ulong(lhs) * rhs;
    const uint carry = uint(product >> 47);
    int exponent = le + re - 127 + int(carry);
    if (exponent >= 255) return sign | 0x7f800000u;
    if (exponent <= 0) return sign | armigo_round_product_rne(product, uint(151 - le - re));
    uint rounded = armigo_round_product_rne(product, 23u + carry);
    if (rounded == 0x01000000u) { rounded >>= 1; ++exponent; }
    if (exponent >= 255) return sign | 0x7f800000u;
    return sign | (uint(exponent) << 23) | (rounded & 0x007fffffu);
}
)msl";
}

}  // namespace armigo::shader
