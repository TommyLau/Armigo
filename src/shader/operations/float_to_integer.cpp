// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Float-to-integer conversion of one raw binary32 word (the F2I admitted
// profile: F32 source, flush-to-zero input, no absolute or negate modifier).
// The immediate encodes the destination signedness, width (16 or 32), and
// rounding mode: bit 0 is signedness, bit 1 selects a 32-bit destination,
// and bits 3..4 select the rounding mode. Conversion uses integer
// decomposition of the binary32 word: no host or MSL
// float-to-integer cast is consulted, because such a cast is undefined for
// out-of-range inputs.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {
namespace {

// The immediate layout shared by the CPU reference, the MSL helper, and the
// family lowerer. Only the five game profiles are admitted
// (admits_float_to_integer).
constexpr std::uint32_t immediate_signed = 1u << 0;
constexpr std::uint32_t immediate_width_32 = 1u << 1;
constexpr std::uint32_t immediate_rounding_shift = 3;

// A magnitude at or above 2^40 cannot fall inside any destination bound, so
// huge finite exponents and infinities saturate here before rounding.
constexpr std::uint64_t saturating_magnitude = std::uint64_t{1} << 40;

// Magnitude of the flushed input on the integer grid (bits below the grid
// are returned separately), with bounds saturation. A zero exponent is the
// flushed input itself and converts to zero before this point; a NaN
// exponent converts to zero before this point as well.
struct GridValue {
    std::uint64_t magnitude = saturating_magnitude;
    std::uint64_t below_grid = 0;
    unsigned shift = 0;
};

GridValue grid_value(std::uint32_t exponent, std::uint32_t mantissa) noexcept {
    GridValue value;
    if (exponent < 159u) {
        const std::uint64_t significand = mantissa | 0x00800000u;
        if (exponent >= 150u) {
            // The value is already an integer.
            value.magnitude = significand << (exponent - 150u);
        } else if (exponent < 126u) {
            // Every such normal is strictly below one half. Preserve a
            // nonzero remainder for directed rounding without a wide shift.
            value.magnitude = 0;
            value.shift = 25;
            value.below_grid = significand;
        } else {
            value.shift = 150u - exponent;
            value.magnitude = significand >> value.shift;
            value.below_grid = significand & ((std::uint64_t{1} << value.shift) - 1u);
        }
    }
    return value;
}

}  // namespace

bool admits_float_to_integer(std::uint32_t immediate) noexcept {
    switch (immediate) {
    // I16 unsigned, round to nearest even.
    case 0u:
    // I32 signed, floor / ceil / trunc.
    case immediate_signed | immediate_width_32 | (1u << immediate_rounding_shift):
    case immediate_signed | immediate_width_32 | (2u << immediate_rounding_shift):
    case immediate_signed | immediate_width_32 | (3u << immediate_rounding_shift):
    // I32 unsigned, trunc.
    case immediate_width_32 | (3u << immediate_rounding_shift):
        return true;
    default:
        return false;
    }
}

std::uint32_t evaluate_float_to_integer(const OperationWords& operands,
                                        std::uint32_t immediate) noexcept {
    const std::uint32_t word = operands[0];
    const std::uint32_t exponent = (word >> 23) & 0xffu;
    const std::uint32_t mantissa = word & 0x007fffffu;
    // FTZ flushes subnormal inputs to a sign-preserving zero, and every NaN
    // converts to zero.
    if (exponent == 0u || (exponent == 0xffu && mantissa != 0u)) {
        return 0;
    }
    const bool negative = (word & 0x80000000u) != 0u;
    const bool signed_destination = (immediate & immediate_signed) != 0u;
    const std::uint32_t destination_32 = immediate & immediate_width_32;
    const unsigned rounding =
        (immediate >> immediate_rounding_shift) & 3u;

    const GridValue value = grid_value(exponent, mantissa);
    std::uint64_t rounded = value.magnitude;
    switch (rounding) {
    case 0: {
        // Round to nearest, ties to even; the tie rule is symmetric in the
        // sign, so it applies to the magnitude for both signs.
        if (value.below_grid != 0u) {
            const std::uint64_t halfway =
                std::uint64_t{1} << (value.shift - 1u);
            if (value.below_grid > halfway ||
                (value.below_grid == halfway && (value.magnitude & 1u) != 0u)) {
                rounded = value.magnitude + 1u;
            }
        }
        break;
    }
    case 1:
        // Floor: negative values with a fraction round away from zero.
        if (negative && value.below_grid != 0u) rounded = value.magnitude + 1u;
        break;
    case 2:
        // Ceil: positive values with a fraction round toward +infinity.
        if (!negative && value.below_grid != 0u) rounded = value.magnitude + 1u;
        break;
    default:
        // Trunc: the magnitude is already the truncated value.
        break;
    }

    if (!signed_destination) {
        // Every rounding mode of a negative source is at most zero.
        if (negative) return 0;
        if (destination_32 != 0u) {
            return rounded > 0xffffffffu ? 0xffffffffu
                                         : static_cast<std::uint32_t>(rounded);
        }
        // 16-bit results are zero-extended into the register.
        return rounded > 65535u ? 65535u : static_cast<std::uint32_t>(rounded);
    }
    if (negative) {
        return rounded >= 0x80000000u
                   ? 0x80000000u
                   : static_cast<std::uint32_t>(0u - static_cast<std::uint32_t>(rounded));
    }
    // Values at or beyond 2^31 saturate to the positive bound. The largest
    // signed 32-bit integer rounds to 2^31 as f32, so positive saturation uses
    // the exact integer bound as an explicit Armigo policy.
    return rounded >= 0x80000000u ? 0x7fffffffu
                                  : static_cast<std::uint32_t>(rounded);
}

std::string emit_float_to_integer(const OperationText& operands,
                                  std::uint32_t immediate) {
    return "armigo_float_to_integer(" + operands[0] + ", 0x" +
           [&] {
               std::string text;
               for (int nibble = 7; nibble >= 0; --nibble) {
                   const unsigned value =
                       (immediate >> (nibble * 4)) & 0xfu;
                   text.push_back(value < 10u ? char('0' + value)
                                              : char('a' + value - 10));
               }
               return text;
           }()
           + "u)";
}

std::string_view helper_float_to_integer() noexcept {
    // Bounded 32-bit integer decomposition of the input: no MSL
    // float-to-integer cast is consulted, and every rejected input has an
    // explicit case.
    return R"msl(inline uint armigo_float_to_integer(uint word, uint profile) {
    const uint exponent = (word >> 23u) & 0xffu;
    const uint mantissa = word & 0x7fffffu;
    if (exponent == 0u || (exponent == 255u && mantissa != 0u)) {
        return 0u;  // Flushed subnormal or NaN converts to zero.
    }
    const bool negative = (word & 0x80000000u) != 0u;
    const bool is_signed = (profile & 1u) != 0u;
    const uint upper = is_signed ? 0x7fffffffu
                                : ((profile & 2u) != 0u ? 0xffffffffu : 65535u);
    if (!is_signed && negative) return 0u;
    // Classify infinity and out-of-range magnitudes before any shifting.
    if (exponent >= (is_signed ? 158u : 159u)) {
        return negative ? 0x80000000u : upper;
    }
    const uint significand = mantissa | 0x00800000u;
    uint magnitude = 0u;
    uint remainder = significand;
    uint shift = 25u;
    if (exponent >= 150u) {
        magnitude = significand << (exponent - 150u);
        remainder = 0u;
    } else if (exponent >= 126u) {
        shift = 150u - exponent;
        magnitude = significand >> shift;
        remainder = significand & ((1u << shift) - 1u);
    }
    const uint rounding = (profile >> 3u) & 3u;
    if (rounding == 0u && remainder != 0u) {
        const uint halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (magnitude & 1u) != 0u)) {
            ++magnitude;
        }
    } else if (rounding == 1u && negative && remainder != 0u) {
        ++magnitude;
    } else if (rounding == 2u && !negative && remainder != 0u) {
        ++magnitude;
    }
    if (negative) return 0u - magnitude;
    return min(magnitude, upper);
}
)msl";
}

}  // namespace armigo::shader::graphics::detail
