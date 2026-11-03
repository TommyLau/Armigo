// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/rtz_multiply.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#ifndef ARMIGO_RTZ_ORACLE_ONLY
#include "test_support.h"
#endif

namespace armigo::shader::test {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits >= 48);

// Shared with the Metal test by compiling this source with
// ARMIGO_RTZ_ORACLE_ONLY. The oracle uses floating arithmetic, never the
// production helper's integer significand multiplication or normalization.
std::uint32_t rtz_multiply_oracle(std::uint32_t lhs, std::uint32_t rhs) {
    const auto nan = [](std::uint32_t value) {
        return (value & 0x7fffffffu) > 0x7f800000u;
    };
    if (nan(lhs) || nan(rhs)) return 0x7fc00000u;
    const auto input = [](std::uint32_t value) {
        if ((value & 0x7f800000u) == 0) value &= 0x80000000u;
        return static_cast<double>(std::bit_cast<float>(value));
    };
    // Two binary32 normal significands have at most 48 significant bits.
    // Their product and its exponent range are represented exactly in binary64.
    const double exact = input(lhs) * input(rhs);
    if (std::isnan(exact)) return 0x7fc00000u;
    if (std::isinf(exact)) {
        return std::signbit(exact) ? 0xff800000u : 0x7f800000u;
    }
    const auto sign = std::signbit(exact) ? 0x80000000u : 0u;
    const double magnitude = std::abs(exact);
    if (magnitude < std::numeric_limits<float>::min()) return sign;
    if (magnitude > std::numeric_limits<float>::max()) return sign | 0x7f7fffffu;
    float rounded = static_cast<float>(magnitude);
    if (static_cast<double>(rounded) > magnitude) {
        rounded = std::nextafter(rounded, 0.0f);
    }
    return sign | std::bit_cast<std::uint32_t>(rounded);
}

std::vector<std::pair<std::uint32_t, std::uint32_t>>
rtz_multiply_inputs(std::size_t random_count) {
    constexpr std::array<std::uint32_t, 26> edges = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x00800000u, 0x80800000u, 0x00800001u, 0x00ffffffu,
        0x3f000000u, 0xbf000000u, 0x3f7fffffu, 0x3f800000u,
        0xbf800000u, 0x3f800001u, 0x3f800800u, 0x3f801800u,
        0x40000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
        0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc12345u,
    };
    std::vector<std::pair<std::uint32_t, std::uint32_t>> inputs;
    for (const auto lhs : edges) {
        for (const auto rhs : edges) inputs.emplace_back(lhs, rhs);
    }
    // Cross the normal/subnormal boundary from both directions, including
    // products which RN would promote to min-normal but RTZ must flush.
    for (std::uint32_t step = 0; step < 4096; ++step) {
        const auto low = 0x00800000u + step;
        const auto below_one = 0x3f800000u - step;
        inputs.emplace_back(low, below_one);
        inputs.emplace_back(low | 0x80000000u, below_one);
        inputs.emplace_back(low, below_one | 0x80000000u);
        inputs.emplace_back(0x00ffffffu - step, 0x3f000000u + step);
    }
    // Fixed-seed full-bit-pattern sampling covers signs, exponent extremes,
    // arbitrary significands and nonfinite encodings without external data.
    std::uint32_t state = 0x67b94d21u;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (std::size_t index = 0; index < random_count; ++index) {
        const auto lhs = next();
        const auto rhs = next();
        inputs.emplace_back(lhs, rhs);
    }
    return inputs;
}

#ifndef ARMIGO_RTZ_ORACLE_ONLY

ARMIGO_TEST(rounding, multiply_authored_exact_results) {
    struct Case { std::uint32_t lhs, rhs, expected; };
    constexpr std::array cases = {
        Case{0x3f800000u, 0x40000000u, 0x40000000u},
        Case{0xbf800000u, 0x40000000u, 0xc0000000u},
        Case{0x80000000u, 0xbf800000u, 0u},
        Case{0u, 0xbf800000u, 0x80000000u},
        Case{1u, 0x7f7fffffu, 0u},
        Case{0x80000001u, 0x7f7fffffu, 0x80000000u},
        Case{0x00800000u, 0x3f800000u, 0x00800000u},
        Case{0x00800000u, 0x3f7fffffu, 0u},
        Case{0x80800000u, 0x3f7fffffu, 0x80000000u},
        Case{0x00800000u, 0x3f000000u, 0u},
        Case{0x7f7fffffu, 0x40000000u, 0x7f7fffffu},
        Case{0xff7fffffu, 0x40000000u, 0xff7fffffu},
        Case{0x7f800000u, 0xbf800000u, 0xff800000u},
        Case{0xff800000u, 0xbf800000u, 0x7f800000u},
        Case{0x7f800000u, 0u, 0x7fc00000u},
        Case{0x7f800000u, 1u, 0x7fc00000u},
        Case{0xff800001u, 0x3f800000u, 0x7fc00000u},
        Case{0x3f800800u, 0x3f801800u, 0x3f802001u},
        Case{0xbf800800u, 0x3f801800u, 0xbf802001u},
    };
    for (const auto& item : cases) {
        ARMIGO_CHECK_EQ(multiply_rtz_ftz(item.lhs, item.rhs), item.expected);
        ARMIGO_CHECK_EQ(multiply_rtz_ftz(item.rhs, item.lhs), item.expected);
        ARMIGO_CHECK_EQ(rtz_multiply_oracle(item.lhs, item.rhs), item.expected);
    }
}

ARMIGO_TEST(rounding, multiply_matches_independent_binary64_oracle) {
    ARMIGO_CHECK_EQ(std::fegetround(), FE_TONEAREST);
    const auto inputs = rtz_multiply_inputs(1000000);
    for (const auto& [lhs, rhs] : inputs) {
        const auto expected = rtz_multiply_oracle(lhs, rhs);
        const auto actual = multiply_rtz_ftz(lhs, rhs);
        if (actual != expected) {
            record_failure(__FILE__, __LINE__,
                "RTZ multiply differs from binary64 oracle for " +
                std::to_string(lhs) + " * " + std::to_string(rhs));
            return;
        }
    }
}

ARMIGO_TEST(rounding, multiply_is_independent_of_host_rounding_mode) {
    const int saved = std::fegetround();
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ARMIGO_CHECK_EQ(std::fesetround(mode), 0);
        ARMIGO_CHECK_EQ(multiply_rtz_ftz(0x3f800800u, 0x3f801800u), 0x3f802001u);
        ARMIGO_CHECK_EQ(multiply_rtz_ftz(0xbf800800u, 0x3f801800u), 0xbf802001u);
        ARMIGO_CHECK_EQ(multiply_rtz_ftz(0x00800000u, 0x3f7fffffu), 0u);
    }
    ARMIGO_CHECK_EQ(std::fesetround(saved), 0);
}

#endif

}  // namespace armigo::shader::test
