// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/reciprocal.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <limits>
#include <vector>

#ifndef ARMIGO_RECIPROCAL_ORACLE_ONLY
#include "test_support.h"
#endif

namespace armigo::shader::test {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4 &&
              std::numeric_limits<float>::digits == 24);

// Shared with the GPU test. Volatile operands force native binary32 division;
// the independent oracle never uses integer significand quotient/remainder.
std::uint32_t reciprocal_oracle(std::uint32_t bits) {
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc00000u;
    volatile float numerator = 1.0f;
    volatile float denominator = std::bit_cast<float>(bits);
    const float result = numerator / denominator;
    return std::bit_cast<std::uint32_t>(result);
}

bool reciprocal_oracle_ready() {
    // Verify both directions of gradual underflow rather than assuming the
    // current host floating-point control register enables subnormal values.
    return std::fegetround() == FE_TONEAREST &&
        reciprocal_oracle(0x00400000u) == 0x7f000000u &&
        reciprocal_oracle(0x7f000000u) == 0x00400000u &&
        reciprocal_oracle(0x40400000u) == 0x3eaaaaabu;
}

std::vector<std::uint32_t> reciprocal_inputs(std::size_t random_count) {
    std::vector<std::uint32_t> inputs = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x00800000u, 0x80800000u, 0x00800001u, 0x3f7fffffu,
        0x3f800000u, 0xbf800000u, 0x3f800001u, 0x40400000u,
        0xc0400000u, 0x41200000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u,
        0xffc12345u,
    };
    // Include every positive/negative representable power of two.
    for (int power = -149; power <= 127; ++power) {
        const auto bits = power < -126 ? std::uint32_t{1} << (power + 149)
            : static_cast<std::uint32_t>(power + 127) << 23;
        inputs.push_back(bits);
        inputs.push_back(bits | 0x80000000u);
    }
    // Dense neighborhoods cover overflow, subnormal normalization, and the
    // reciprocal's normal/subnormal transition on both signs.
    constexpr std::array<std::uint32_t, 6> centers = {
        0x00200000u, 0x00400000u, 0x00800000u,
        0x3f800000u, 0x7e800000u, 0x7f000000u,
    };
    for (const auto center : centers) {
        for (int delta = -4096; delta <= 4096; ++delta) {
            const auto bits = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(center) + delta);
            inputs.push_back(bits);
            inputs.push_back(bits | 0x80000000u);
        }
    }
    std::uint32_t state = 0x49c21a83u;
    for (std::size_t index = 0; index < random_count; ++index) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        inputs.push_back(state);
    }
    return inputs;
}

#ifndef ARMIGO_RECIPROCAL_ORACLE_ONLY

namespace {

struct NearestScope {
    std::fenv_t saved{};
    NearestScope() {
        ARMIGO_CHECK_EQ(std::fegetenv(&saved), 0);
        ARMIGO_CHECK_EQ(std::fesetround(FE_TONEAREST), 0);
    }
    ~NearestScope() { ARMIGO_CHECK_EQ(std::fesetenv(&saved), 0); }
};

}  // namespace

ARMIGO_TEST(reciprocal, authored_exact_results) {
    const NearestScope nearest;
    ARMIGO_CHECK(reciprocal_oracle_ready());
    struct Case { std::uint32_t input, expected; };
    constexpr std::array cases = {
        Case{0u, 0x7f800000u}, Case{0x80000000u, 0xff800000u},
        Case{0x7f800000u, 0u}, Case{0xff800000u, 0x80000000u},
        Case{0x7f800001u, 0x7fc00000u}, Case{0xffc12345u, 0x7fc00000u},
        Case{0x3f800000u, 0x3f800000u}, Case{0xbf800000u, 0xbf800000u},
        Case{0x40000000u, 0x3f000000u}, Case{0x40400000u, 0x3eaaaaabu},
        Case{0xc0400000u, 0xbeaaaaabu}, Case{0x41200000u, 0x3dcccccdu},
        Case{0x00800000u, 0x7e800000u}, Case{0x00800001u, 0x7e7ffffeu},
        Case{0x007fffffu, 0x7e800001u}, Case{0x00200000u, 0x7f800000u},
        Case{0x00200001u, 0x7f7ffff8u}, Case{0x00400000u, 0x7f000000u},
        Case{0x7e7fffffu, 0x00800001u}, Case{0x7e800000u, 0x00800000u},
        Case{0x7f000000u, 0x00400000u}, Case{0x7f7fffffu, 0x00200000u},
        Case{0xff7fffffu, 0x80200000u}, Case{1u, 0x7f800000u},
    };
    for (const auto& item : cases) {
        ARMIGO_CHECK_EQ(reciprocal_rne(item.input), item.expected);
        ARMIGO_CHECK_EQ(reciprocal_oracle(item.input), item.expected);
    }
}

ARMIGO_TEST(reciprocal, every_binary32_power_of_two) {
    for (int power = -149; power <= 127; ++power) {
        const auto input = power < -126 ? std::uint32_t{1} << (power + 149)
            : static_cast<std::uint32_t>(power + 127) << 23;
        const int inverse_power = -power;
        const auto expected = inverse_power > 127 ? 0x7f800000u
            : inverse_power < -126 ? std::uint32_t{1} << (inverse_power + 149)
            : static_cast<std::uint32_t>(inverse_power + 127) << 23;
        ARMIGO_CHECK_EQ(reciprocal_rne(input), expected);
        ARMIGO_CHECK_EQ(reciprocal_rne(input | 0x80000000u), expected | 0x80000000u);
    }
}

ARMIGO_TEST(reciprocal, matches_native_division_with_gradual_underflow) {
    const NearestScope nearest;
    if (!reciprocal_oracle_ready()) {
        ARMIGO_CHECK(reciprocal_oracle_ready());
        return;
    }
    for (const auto input : reciprocal_inputs(1000000)) {
        if (reciprocal_rne(input) != reciprocal_oracle(input)) {
            record_failure(__FILE__, __LINE__,
                "Reciprocal differs from native binary32 division for " +
                std::to_string(input));
            return;
        }
    }
}

ARMIGO_TEST(reciprocal, independent_of_host_rounding_mode) {
    const NearestScope nearest;
    const auto inputs = reciprocal_inputs(1024);
    std::vector<std::uint32_t> expected;
    for (const auto input : inputs) expected.push_back(reciprocal_oracle(input));
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ARMIGO_CHECK_EQ(std::fesetround(mode), 0);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (reciprocal_rne(inputs[index]) != expected[index]) {
                record_failure(__FILE__, __LINE__,
                    "Reciprocal changed under host rounding mode " + std::to_string(mode));
                return;
            }
        }
    }
}

#endif

}  // namespace armigo::shader::test
