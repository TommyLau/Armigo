// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Integer-to-binary32 conversion (I2F). The immediate encodes the admitted
// conversion profiles: bit 0 signedness, bits 1..2 the source width code
// (1 => 16, 2 => 32), bits 3..4 the Maxwell rounding field (0 => nearest
// even, 3 => toward zero) and bit 5 the wraparound integer absolute value.
// The selector-0 extraction is implicit: the source word is the integer.
//
// Conversion uses pure integer arithmetic; no host floating state is
// consulted. The MSL helper assembles the IEEE bits directly for the same
// reason: Metal's integer-to-float conversion is not relied on for either
// rounding mode.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {
namespace {

// Exact binary32 bits of a non-negative integer under the rounding mode.
// The magnitude is normalized with integer steps, rounded, and the bits are
// assembled: every result is a finite normal float or zero.
std::uint32_t convert_magnitude(std::uint32_t magnitude, bool round_to_zero) noexcept {
    if (magnitude == 0) return 0;
    // Highest set bit position: magnitude in [2^p, 2^(p+1)).
    unsigned p = 0;
    std::uint32_t reduced = magnitude;
    if ((reduced & 0xffff0000u) != 0) { p += 16; reduced >>= 16; }
    if ((reduced & 0x0000ff00u) != 0) { p += 8; reduced >>= 8; }
    if ((reduced & 0x000000f0u) != 0) { p += 4; reduced >>= 4; }
    if ((reduced & 0x0000000cu) != 0) { p += 2; reduced >>= 2; }
    if ((reduced & 0x00000002u) != 0) { p += 1; }
    std::uint32_t mantissa = 0;
    if (p <= 23) {
        // p + 1 significant bits fit the 24-bit precision exactly.
        mantissa = magnitude << (23 - p);
    } else {
        const unsigned shift = p - 23;
        mantissa = magnitude >> shift;
        if (!round_to_zero) {
            const std::uint32_t round_bit = (magnitude >> (shift - 1)) & 1u;
            const std::uint32_t sticky = magnitude & ((1u << (shift - 1)) - 1u);
            if (round_bit != 0 && (sticky != 0 || (mantissa & 1u) != 0)) {
                ++mantissa;  // May carry into bit 24; the exponent absorbs it.
            }
        }
    }
    const unsigned exponent = 127 + p + (mantissa >> 24);
    return (exponent << 23) | (mantissa & 0x7fffffu);
}

}  // namespace

bool admits_integer_to_float(std::uint32_t immediate) noexcept {
    if ((immediate & ~0x3fu) != 0) return false;
    const bool is_signed = (immediate & 1u) != 0;
    const std::uint32_t width = (immediate >> 1) & 3u;
    const std::uint32_t rounding = (immediate >> 3) & 3u;
    const bool absolute = (immediate & 32u) != 0;
    // Admitted profiles: signed 32 nearest-even, unsigned 32 toward zero,
    // unsigned 32 nearest-even, signed 32 nearest-even with absolute, and
    // signed 16 nearest-even. The 8-bit width and the remaining combinations
    // are not decoded.
    if (width == 2) {
        if (rounding == 0) return !absolute || is_signed;
        if (rounding == 3) return !is_signed && !absolute;
        return false;
    }
    if (width == 1) return is_signed && rounding == 0 && !absolute;
    return false;
}

std::uint32_t evaluate_integer_to_float(const OperationWords& operands,
                                        std::uint32_t immediate) noexcept {
    const bool is_signed = (immediate & 1u) != 0;
    const bool width16 = ((immediate >> 1) & 3u) == 1;
    const std::uint32_t mask = width16 ? 0xffffu : 0xffffffffu;
    const bool round_to_zero = ((immediate >> 3) & 3u) == 3;
    const bool absolute = (immediate & 32u) != 0;
    std::uint32_t value = operands[0] & mask;
    if (is_signed) {
        const std::uint32_t sign_bit = width16 ? 0x8000u : 0x80000000u;
        if (absolute && (value & sign_bit) != 0) {
            // Wraparound absolute value inside the source width: the least
            // value stays itself under the guest conversion.
            value = (0u - value) & mask;
        }
        if ((value & sign_bit) != 0) {
            value |= width16 ? 0xffff0000u : 0u;
        }
    }
    const bool negative = is_signed && (value & 0x80000000u) != 0;
    const std::uint32_t magnitude = negative ? 0u - value : value;
    const std::uint32_t bits = convert_magnitude(magnitude, round_to_zero);
    return negative ? bits | 0x80000000u : bits;
}

std::string emit_integer_to_float(const OperationText& operands, std::uint32_t immediate) {
    const bool is_signed = (immediate & 1u) != 0;
    const char* width = ((immediate >> 1) & 3u) == 1 ? "16u" : "32u";
    const bool round_to_zero = ((immediate >> 3) & 3u) == 3;
    const bool absolute = (immediate & 32u) != 0;
    return "(armigo_i2f(" + operands[0] + ", " + (is_signed ? "true" : "false") + ", " +
           width + ", " + (round_to_zero ? "true" : "false") + ", " +
           (absolute ? "true" : "false") + "))";
}

std::string_view integer_to_float_msl_helper() noexcept {
    // The normalization, rounding, and bit assembly use only integer
    // operations, so every one of the 2^32 source words converts exactly.
    return R"msl(inline uint armigo_i2f_round(uint magnitude, bool round_to_zero) {
    if (magnitude == 0u) { return 0u; }
    uint p = 0u;
    uint reduced = magnitude;
    if ((reduced & 0xffff0000u) != 0u) { p += 16u; reduced >>= 16u; }
    if ((reduced & 0x0000ff00u) != 0u) { p += 8u; reduced >>= 8u; }
    if ((reduced & 0x000000f0u) != 0u) { p += 4u; reduced >>= 4u; }
    if ((reduced & 0x0000000cu) != 0u) { p += 2u; reduced >>= 2u; }
    if ((reduced & 0x00000002u) != 0u) { p += 1u; }
    uint mantissa;
    if (p <= 23u) {
        mantissa = magnitude << (23u - p);
    } else {
        const uint shift = p - 23u;
        mantissa = magnitude >> shift;
        if (!round_to_zero) {
            const uint round_bit = (magnitude >> (shift - 1u)) & 1u;
            const uint sticky = magnitude & ((1u << (shift - 1u)) - 1u);
            if (round_bit != 0u && (sticky != 0u || (mantissa & 1u) != 0u)) {
                mantissa += 1u;
            }
        }
    }
    const uint exponent = 127u + p + (mantissa >> 24u);
    return (exponent << 23u) | (mantissa & 0x7fffffu);
}
inline uint armigo_i2f(uint word, bool is_signed, uint width, bool round_to_zero,
                       bool absolute) {
    const uint mask = (width == 16u) ? 0xffffu : 0xffffffffu;
    uint value = word & mask;
    if (is_signed) {
        const uint sign_bit = (width == 16u) ? 0x8000u : 0x80000000u;
        if (absolute && (value & sign_bit) != 0u) {
            value = (0u - value) & mask;
        }
        if ((value & sign_bit) != 0u) {
            value |= (width == 16u) ? 0xffff0000u : 0x0u;
        }
    }
    const bool negative = is_signed && (value & 0x80000000u) != 0u;
    const uint magnitude = negative ? (0u - value) : value;
    const uint bits = armigo_i2f_round(magnitude, round_to_zero);
    return negative ? (bits | 0x80000000u) : bits;
}
)msl";
}

}  // namespace armigo::shader::graphics::detail
