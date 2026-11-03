// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bit-exact binary32 minimum and maximum.
// Both kinds work on raw words with integer operations only, so neither the
// CPU reference nor the MSL helper depends on host min/max NaN or signed
// zero behaviour. The contract: inputs are flushed sign-preserving; exactly
// one NaN operand yields the other operand; two NaN operands yield the
// canonical quiet NaN; zeros of different signs order -0 below +0.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {
namespace {

constexpr std::uint32_t flush(std::uint32_t x) noexcept {
    return (x & 0x7f800000u) == 0u ? x & 0x80000000u : x;
}

constexpr bool is_nan(std::uint32_t x) noexcept {
    return (x & 0x7fffffffu) > 0x7f800000u;
}

// Maps a word onto an unsigned key that compares exactly as the value does,
// with -0 below +0 and no host floating state involved.
constexpr std::uint32_t order_key(std::uint32_t x) noexcept {
    return (x >> 31) != 0u ? ~x : x | 0x80000000u;
}

std::uint32_t combine(std::uint32_t a, std::uint32_t b, bool minimum) noexcept {
    const std::uint32_t left = flush(a);
    const std::uint32_t right = flush(b);
    const bool left_nan = is_nan(left);
    const bool right_nan = is_nan(right);
    if (left_nan || right_nan) {
        if (left_nan && right_nan) return 0x7fc00000u;
        return left_nan ? right : left;
    }
    return minimum == (order_key(left) <= order_key(right)) ? left : right;
}

}  // namespace

std::uint32_t evaluate_float_minimum(const OperationWords& operands,
                                     std::uint32_t) noexcept {
    return combine(operands[0], operands[1], true);
}

std::string emit_float_minimum(const OperationText& operands, std::uint32_t) {
    return "armigo_float_min(" + operands[0] + ", " + operands[1] + ")";
}

std::string_view helper_float_minimum() noexcept {
    return R"msl(inline uint armigo_float_min(uint a, uint b) {
    uint fa = (a & 0x7f800000u) == 0u ? a & 0x80000000u : a;
    uint fb = (b & 0x7f800000u) == 0u ? b & 0x80000000u : b;
    bool na = (fa & 0x7fffffffu) > 0x7f800000u;
    bool nb = (fb & 0x7fffffffu) > 0x7f800000u;
    if (na || nb) {
        return (na && nb) ? 0x7fc00000u : (na ? fb : fa);
    }
    uint ka = (fa >> 31) != 0u ? ~fa : fa | 0x80000000u;
    uint kb = (fb >> 31) != 0u ? ~fb : fb | 0x80000000u;
    return ka <= kb ? fa : fb;
}
)msl";
}

std::uint32_t evaluate_float_maximum(const OperationWords& operands,
                                     std::uint32_t) noexcept {
    return combine(operands[0], operands[1], false);
}

std::string emit_float_maximum(const OperationText& operands, std::uint32_t) {
    return "armigo_float_max(" + operands[0] + ", " + operands[1] + ")";
}

std::string_view helper_float_maximum() noexcept {
    return R"msl(inline uint armigo_float_max(uint a, uint b) {
    uint fa = (a & 0x7f800000u) == 0u ? a & 0x80000000u : a;
    uint fb = (b & 0x7f800000u) == 0u ? b & 0x80000000u : b;
    bool na = (fa & 0x7fffffffu) > 0x7f800000u;
    bool nb = (fb & 0x7fffffffu) > 0x7f800000u;
    if (na || nb) {
        return (na && nb) ? 0x7fc00000u : (na ? fb : fa);
    }
    uint ka = (fa >> 31) != 0u ? ~fa : fa | 0x80000000u;
    uint kb = (fb >> 31) != 0u ? ~fb : fb | 0x80000000u;
    return ka >= kb ? fa : fb;
}
)msl";
}

}  // namespace armigo::shader::graphics::detail
