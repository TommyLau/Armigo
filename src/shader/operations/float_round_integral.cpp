// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Rounding of a raw binary32 word to an integral value: nearest-even (0),
// floor (1), ceil (2), or trunc (3). This is the F2F same-size rounding
// operation. Subnormal inputs are flushed first to the sign-preserving zero
// (the armigo_flush_ftz rule); infinities pass through; a NaN propagates
// with sign and payload because the same-size rounding path applies the
// roundToIntegral operation without the canonicalizing add its pass-through
// path uses.
#include "../graphics_operations.h"

#include <bit>
#include <cstdint>

namespace armigo::shader::graphics::detail {

bool admits_float_round_integral(std::uint32_t immediate) noexcept {
    return immediate < 4;
}

// Exact integer reference: the magnitude is rounded on the unsigned
// mantissa, which is symmetric for all four directions, then re-encoded.
// No host floating state or out-of-range conversion is consulted.
std::uint32_t evaluate_float_round_integral(const OperationWords& operands,
                                            std::uint32_t immediate) noexcept {
    const std::uint32_t word = operands[0];
    const std::uint32_t sign = word & 0x80000000u;
    const std::uint32_t exponent = (word >> 23) & 0xffu;
    // Flush-to-zero input: a subnormal becomes the sign-preserving zero.
    if (exponent == 0) return sign;
    // Infinities and NaNs are their own integral rounding result, and every
    // |x| >= 2^23 is already integral, so the word passes through.
    if (exponent >= 0xff) return word;
    if (exponent >= 150) return word;
    const std::uint32_t shift = 150 - exponent;
    const std::uint32_t mantissa = (word & 0x007fffffu) | 0x00800000u;
    // For shift > 24 the whole 24-bit mantissa is the fraction of a value
    // below 0.5, which is below every half point of the shifted scale; the
    // half comparison itself is only defined within that 24-bit range.
    const bool below_half_scale = shift > 24;
    const std::uint32_t fraction =
        below_half_scale ? mantissa : mantissa & ((1u << shift) - 1u);
    std::uint32_t magnitude = below_half_scale ? 0u : (mantissa >> shift);
    // IEEE roundToIntegral: floor rounds a negative value away from zero,
    // ceil a positive one, and nearest-even breaks exact halves to the even
    // magnitude.
    bool increment = false;
    switch (immediate) {
        case 0:
            if (!below_half_scale) {
                const std::uint32_t half = 1u << (shift - 1);
                increment = fraction > half ||
                            (fraction == half && (magnitude & 1u) != 0);
            }
            break;
        case 1: increment = sign != 0 && fraction != 0; break;
        case 2: increment = sign == 0 && fraction != 0; break;
        default: break;
    }
    if (increment) ++magnitude;
    // The rounded magnitude is at most 2^23, so it re-encodes exactly.
    if (magnitude == 0) return sign;
    const unsigned top = 31u - static_cast<unsigned>(std::countl_zero(magnitude));
    return sign | ((static_cast<std::uint32_t>(top) + 127u) << 23) |
           ((magnitude << (23 - top)) & 0x007fffffu);
}

std::string emit_float_round_integral(const OperationText& operands,
                                      std::uint32_t immediate) {
    return "armigo_round_integral(" + operands[0] + ", " +
           std::to_string(immediate) + "u)";
}

std::string_view helper_float_round_integral() noexcept {
    // Subnormal inputs flush to the signed zero before the operation and
    // infinities and NaNs return unchanged, so the Metal conversions only
    // ever see finite normals; floor, ceil, trunc and rint are exact on
    // every finite binary32 input (for |x| >= 2^23 the input is integral,
    // otherwise the result is an integer <= 2^23, exactly representable).
    return R"msl(inline uint armigo_round_integral(uint x, uint mode) {
    const uint exponent = (x >> 23u) & 0xffu;
    const uint sign = x & 0x80000000u;
    if (exponent == 0u) return sign;
    if (exponent >= 0xffu) return x;
    const float value = as_type<float>(x);
    float rounded = value;
    if (mode == 0u) rounded = rint(value);
    else if (mode == 1u) rounded = floor(value);
    else if (mode == 2u) rounded = ceil(value);
    else rounded = trunc(value);
    return as_type<uint>(rounded);
}
)msl";
}

}  // namespace armigo::shader::graphics::detail
