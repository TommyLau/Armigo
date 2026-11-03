// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Float compare to predicate: one IEEE binary32 comparison of two raw words
// per immediate code, the combining step of the Maxwell FSETP family. The
// code order follows the instruction encoding (F, LT, EQ, LE, GT, NE, GE,
// NUM, NAN, LTU, EQU, LEU, GTU, NEU, GEU, T); bit 4 of the immediate is the
// flush-to-zero flag. Flush-to-zero replaces zero and subnormal inputs with
// same-sign zero before comparing; NaNs and infinities pass through. Every
// result is computed from integer bit tests, independent of host or Metal
// floating-point flush modes, including when FTZ is disabled.
#include "../graphics_operations.h"

#include <string>
#include <string_view>

namespace armigo::shader::graphics::detail {
namespace {

constexpr std::uint32_t kCompareCodeMask = 0xfu;
constexpr std::uint32_t kFlushToZeroBit = 0x10u;

// Sign-preserving flush of zero and subnormal inputs to zero.
constexpr std::uint32_t flush_to_zero(std::uint32_t bits) noexcept {
    return (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
}

constexpr bool is_nan(std::uint32_t bits) noexcept {
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
}

}  // namespace

bool admits_float_compare_predicate(std::uint32_t immediate) noexcept {
    return (immediate & ~(kCompareCodeMask | kFlushToZeroBit)) == 0;
}

std::uint32_t evaluate_float_compare_predicate(
    const OperationWords& operands, std::uint32_t immediate) noexcept {
    const std::uint32_t code = immediate & kCompareCodeMask;
    std::uint32_t a = operands[0];
    std::uint32_t b = operands[1];
    if ((immediate & kFlushToZeroBit) != 0) {
        a = flush_to_zero(a);
        b = flush_to_zero(b);
    }
    // Unordered inputs: the ordered codes F, LT..GE and NUM are false; NAN,
    // the six U-suffixed variants and T are true.
    if (is_nan(a) || is_nan(b)) {
        return code >= 8u ? 1u : 0u;
    }
    // Binary32 magnitudes sort as unsigned words. Negative values reverse
    // that order; the two zero encodings denote the same value.
    const bool equal = a == b || ((a | b) & 0x7fffffffu) == 0;
    const bool less = !equal && (((a ^ b) & 0x80000000u) != 0
        ? (a & 0x80000000u) != 0
        : ((a & 0x80000000u) != 0 ? a > b : a < b));
    bool result = false;
    switch (code) {
        case 1: case 9: result = less; break;
        case 2: case 10: result = equal; break;
        case 3: case 11: result = less || equal; break;
        case 4: case 12: result = !less && !equal; break;
        case 5: case 13: result = !equal; break;
        case 6: case 14: result = !less; break;
        case 7: case 15: result = true; break;
        default: break;  // F and NAN are false on ordered inputs.
    }
    return result ? 1u : 0u;
}

std::string emit_float_compare_predicate(const OperationText& operands,
                                         std::uint32_t immediate) {
    const std::string code = std::to_string(immediate & kCompareCodeMask) + "u";
    const std::string flush =
        (immediate & kFlushToZeroBit) != 0 ? "true" : "false";
    return "armigo_float_compare(" + code + ", " + flush + ", " + operands[0] +
           ", " + operands[1] + ")";
}

std::string_view helper_float_compare_predicate() noexcept {
    // Integer comparisons preserve non-FTZ subnormals on every Metal device.
    // No floating-point conversion, arithmetic, or NaN folding is involved.
    return R"msl(inline uint armigo_fsetp_flush(uint x) {
    return (x & 0x7f800000u) == 0u ? x & 0x80000000u : x;
}
inline bool armigo_float_compare(uint code, bool flush_to_zero, uint raw_a, uint raw_b) {
    const uint a = flush_to_zero ? armigo_fsetp_flush(raw_a) : raw_a;
    const uint b = flush_to_zero ? armigo_fsetp_flush(raw_b) : raw_b;
    const bool unordered = ((a & 0x7f800000u) == 0x7f800000u && (a & 0x007fffffu) != 0u) ||
                           ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u);
    if (unordered) {
        return code >= 8u;
    }
    const bool equal = a == b || ((a | b) & 0x7fffffffu) == 0u;
    const bool less = !equal && (((a ^ b) & 0x80000000u) != 0u
        ? (a & 0x80000000u) != 0u
        : ((a & 0x80000000u) != 0u ? a > b : a < b));
    switch (code) {
        case 1u: case 9u: return less;
        case 2u: case 10u: return equal;
        case 3u: case 11u: return less || equal;
        case 4u: case 12u: return !less && !equal;
        case 5u: case 13u: return !equal;
        case 6u: case 14u: return !less;
        case 7u: case 15u: return true;
        default: return false;
    }
}
)msl";
}

}  // namespace armigo::shader::graphics::detail
