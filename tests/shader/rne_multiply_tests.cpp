// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/rne_multiply.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#ifndef ARMIGO_RNE_MULTIPLY_ORACLE_ONLY
#include "test_support.h"
#endif

namespace armigo::shader::test {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4 &&
              std::numeric_limits<float>::digits == 24);

std::uint32_t rne_multiply_oracle(std::uint32_t lhs, std::uint32_t rhs) {
    // Native IEEE binary32 multiplication is independent of the integer
    // significand algorithm. Volatile operands prevent constant folding.
    volatile float a = std::bit_cast<float>(lhs);
    volatile float b = std::bit_cast<float>(rhs);
    const float product = a * b;
    const auto bits = std::bit_cast<std::uint32_t>(product);
    return (bits & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : bits;
}

bool rne_multiply_oracle_ready() {
    return std::fegetround() == FE_TONEAREST &&
        rne_multiply_oracle(1u, 0x3f800000u) == 1u &&
        rne_multiply_oracle(0x00800000u, 0x3f000000u) == 0x00400000u &&
        rne_multiply_oracle(3u, 0x3f000000u) == 2u &&
        rne_multiply_oracle(0x3f800800u, 0x3f801800u) == 0x3f802002u;
}

namespace {
std::uint32_t power_bits(int power) {
    return power < -126 ? std::uint32_t{1} << (power + 149)
        : static_cast<std::uint32_t>(power + 127) << 23;
}
}

std::vector<std::pair<std::uint32_t, std::uint32_t>>
rne_multiply_inputs(std::size_t random_count) {
    constexpr std::array<std::uint32_t, 30> edges = {
        0u, 0x80000000u, 1u, 2u, 3u, 5u, 0x80000001u,
        0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
        0x00800001u, 0x00ffffffu, 0x33800000u, 0x34000000u,
        0x3f000000u, 0xbf000000u, 0x3f7fffffu, 0x3f800000u,
        0xbf800000u, 0x3f800001u, 0x3f800800u, 0x3f801800u,
        0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
        0x7fc00000u, 0x7f800001u, 0xffc12345u,
    };
    std::vector<std::pair<std::uint32_t, std::uint32_t>> inputs;
    for (const auto lhs : edges) {
        for (const auto rhs : edges) inputs.emplace_back(lhs, rhs);
    }
    for (int a = -149; a <= 127; ++a) {
        for (int b = -149; b <= 127; ++b) {
            inputs.emplace_back(power_bits(a), power_bits(b));
            inputs.emplace_back(power_bits(a) | 0x80000000u, power_bits(b));
        }
    }
    for (std::uint32_t step = 0; step < 4096; ++step) {
        inputs.emplace_back(0x00800000u + step, 0x3f800000u - step);
        inputs.emplace_back(0x80800000u + step, 0x3f800000u - step);
        inputs.emplace_back(0x007fffffu - step, 0x3f800001u + step);
        inputs.emplace_back(0x80000001u + step, 0x3f000000u);
        inputs.emplace_back(0x7f7fffffu - step, 0x3f800000u + step);
        inputs.emplace_back(0xff7fffffu - step, 0x3f800000u + step);
    }
    std::uint32_t state = 0x83b5a729u;
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

#ifndef ARMIGO_RNE_MULTIPLY_ORACLE_ONLY
namespace {
struct NearestScope {
    std::fenv_t saved{};
    NearestScope() {
        ARMIGO_CHECK_EQ(std::feholdexcept(&saved), 0);
        ARMIGO_CHECK_EQ(std::fesetround(FE_TONEAREST), 0);
    }
    ~NearestScope() { ARMIGO_CHECK_EQ(std::fesetenv(&saved), 0); }
};
}

ARMIGO_TEST(rne_multiply, authored_exact_results_and_ties) {
    const NearestScope nearest;
    ARMIGO_CHECK(rne_multiply_oracle_ready());
    struct Case { std::uint32_t lhs, rhs, expected; };
    constexpr std::array cases = {
        Case{0x3f800000u, 0x40000000u, 0x40000000u},
        Case{0xbf800000u, 0x40000000u, 0xc0000000u},
        Case{0u, 0xbf800000u, 0x80000000u},
        Case{0x80000000u, 0xbf800000u, 0u},
        Case{1u, 0x3f800000u, 1u},
        Case{0x80000001u, 0x3f800000u, 0x80000001u},
        Case{1u, 0x3f000000u, 0u},
        Case{0x80000001u, 0x3f000000u, 0x80000000u},
        Case{3u, 0x3f000000u, 2u},
        Case{5u, 0x3f000000u, 2u},
        Case{0x00800000u, 0x3f000000u, 0x00400000u},
        Case{0x00800000u, 0x34000000u, 1u},
        Case{0x00800000u, 0x33800000u, 0u},
        Case{0x007fffffu, 0x3f800001u, 0x00800000u},
        Case{0x3f800800u, 0x3f800800u, 0x3f801000u},
        Case{0x3f800800u, 0x3f801800u, 0x3f802002u},
        Case{0xbf800800u, 0x3f801800u, 0xbf802002u},
        Case{0x7f7fffffu, 0x3f800000u, 0x7f7fffffu},
        Case{0x7f7fffffu, 0x3f800001u, 0x7f800000u},
        Case{0xff7fffffu, 0x40000000u, 0xff800000u},
        Case{0x7f800000u, 0u, 0x7fc00000u},
        Case{0x7f800000u, 1u, 0x7f800000u},
        Case{0xff800000u, 1u, 0xff800000u},
        Case{0xffc12345u, 0u, 0x7fc00000u},
    };
    for (const auto& item : cases) {
        ARMIGO_CHECK_EQ(multiply_rne(item.lhs, item.rhs), item.expected);
        ARMIGO_CHECK_EQ(multiply_rne(item.rhs, item.lhs), item.expected);
        ARMIGO_CHECK_EQ(rne_multiply_oracle(item.lhs, item.rhs), item.expected);
    }
}

ARMIGO_TEST(rne_multiply, all_power_of_two_products) {
    for (int a = -149; a <= 127; ++a) {
        for (int b = -149; b <= 127; ++b) {
            const int result_power = a + b;
            const auto expected = result_power > 127 ? 0x7f800000u
                : result_power < -149 ? 0u : power_bits(result_power);
            for (const auto lhs_sign : {0u, 0x80000000u}) {
                for (const auto rhs_sign : {0u, 0x80000000u}) {
                    ARMIGO_CHECK_EQ(multiply_rne(power_bits(a) | lhs_sign, power_bits(b) | rhs_sign),
                                    expected | (lhs_sign ^ rhs_sign));
                }
            }
        }
    }
}

ARMIGO_TEST(rne_multiply, matches_independent_native_float_oracle) {
    const NearestScope nearest;
    if (!rne_multiply_oracle_ready()) {
        ARMIGO_CHECK(rne_multiply_oracle_ready());
        return;
    }
    for (const auto& [lhs, rhs] : rne_multiply_inputs(2000000)) {
        if (multiply_rne(lhs, rhs) != rne_multiply_oracle(lhs, rhs)) {
            record_failure(__FILE__, __LINE__, "RNE multiply differs from native oracle for " +
                std::to_string(lhs) + " * " + std::to_string(rhs));
            return;
        }
    }
}

ARMIGO_TEST(rne_multiply, independent_of_host_rounding_mode) {
    const NearestScope nearest;
    if (!rne_multiply_oracle_ready()) {
        ARMIGO_CHECK(rne_multiply_oracle_ready());
        return;
    }
    const auto inputs = rne_multiply_inputs(1024);
    std::vector<std::uint32_t> expected;
    for (const auto& [lhs, rhs] : inputs) expected.push_back(rne_multiply_oracle(lhs, rhs));
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ARMIGO_CHECK_EQ(std::fesetround(mode), 0);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (multiply_rne(inputs[index].first, inputs[index].second) != expected[index]) {
                record_failure(__FILE__, __LINE__, "RNE multiply changed with host rounding mode");
                return;
            }
        }
    }
}
#endif

}  // namespace armigo::shader::test
