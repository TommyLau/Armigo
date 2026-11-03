// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/fma.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#ifndef ARMIGO_FMA_ORACLE_ONLY
#include "test_support.h"
#endif

namespace armigo::shader::test {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4 &&
              std::numeric_limits<float>::digits == 24);

std::uint32_t fma_oracle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    const auto flush = [](std::uint32_t value) {
        return (value & 0x7f800000u) == 0 ? value & 0x80000000u : value;
    };
    volatile float lhs = std::bit_cast<float>(flush(a));
    volatile float rhs = std::bit_cast<float>(flush(b));
    volatile float addend = std::bit_cast<float>(flush(c));
    const float value = std::fma(lhs, rhs, addend);
    const auto result = std::bit_cast<std::uint32_t>(value);
    return (result & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : flush(result);
}

bool fma_oracle_ready() {
    // The oracle must round before output FTZ. Native Metal fma has failed
    // this sentinel, which is why the emitted helper cannot use native fma.
    volatile float minimum = std::bit_cast<float>(0x00800000u);
    volatile float half_value = 0.5f;
    const float gradual = std::fma(minimum, half_value, 0.0f);
    return std::fegetround() == FE_TONEAREST &&
        std::bit_cast<std::uint32_t>(gradual) == 0x00400000u &&
        fma_oracle(0x00800000u, 0x3f7fffffu, 0u) == 0x00800000u &&
        fma_oracle(0x3f800800u, 0x3f800800u, 0xbf801000u) == 0x33800000u;
}

std::vector<std::array<std::uint32_t, 3>> fma_inputs(std::size_t random_count) {
    constexpr std::array<std::uint32_t, 24> edges = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x00800000u,
        0x80800000u, 0x00800001u, 0x3f000000u, 0xbf000000u,
        0x3f7fffffu, 0x3f800000u, 0xbf800000u, 0x3f800001u,
        0x3f800800u, 0xbf801000u, 0x40000000u, 0x7f7fffffu,
        0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u,
        0x7f800001u, 0xffc12345u,
    };
    std::vector<std::array<std::uint32_t, 3>> inputs;
    for (const auto a : edges) for (const auto b : edges) for (const auto c : edges) {
        inputs.push_back({a, b, c});
    }
    // Exercise the complete normal exponent grid, wide alignment gaps,
    // addition carries and subtraction borrows across accumulator limbs.
    for (unsigned ea = 1; ea <= 254; ++ea) {
        for (unsigned eb = 1; eb <= 254; ++eb) {
            const auto a = (ea << 23) | 0x007fffffu;
            const auto b = (eb << 23) | 0x00000001u;
            const auto c = (((ea + eb) % 254 + 1) << 23) | 0x00400000u;
            inputs.push_back({a, b, c});
            inputs.push_back({a, b, c | 0x80000000u});
        }
    }
    for (unsigned step = 0; step < 4096; ++step) {
        inputs.push_back({0x00800000u + step, 0x3f7fffffu - step, 0u});
        inputs.push_back({0x80800000u + step, 0x3f7fffffu - step, 0x80000000u});
        inputs.push_back({0x00800000u + step, 0x3f000000u, 0x80800000u});
        inputs.push_back({0x3f800000u + step, 0x3f800000u, 0xbf800000u + step});
        const auto a = 0x3f800000u + step * 17;
        const auto b = 0x3f800000u + step * 29;
        volatile float lhs = std::bit_cast<float>(a), rhs = std::bit_cast<float>(b);
        const float rounded_product = lhs * rhs;
        const auto cancellation = std::bit_cast<std::uint32_t>(rounded_product) ^ 0x80000000u;
        inputs.push_back({a, b, cancellation});
        inputs.push_back({a | 0x80000000u, b, cancellation ^ 0x80000000u});
    }
    std::uint32_t state = 0x95c2e37bu;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (std::size_t index = 0; index < random_count; ++index) {
        const auto a = next(), b = next(), c = next();
        inputs.push_back({a, b, c});
    }
    return inputs;
}

#ifndef ARMIGO_FMA_ORACLE_ONLY
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

ARMIGO_TEST(fma, authored_results_cancellation_ties_and_special_values) {
    const NearestScope nearest;
    ARMIGO_CHECK(fma_oracle_ready());
    struct Case { std::uint32_t a, b, c, expected; };
    constexpr std::array cases = {
        Case{0x00800000u, 0x3f7fffffu, 0u, 0x00800000u},
        Case{0x80800000u, 0x3f7fffffu, 0x80000000u, 0x80800000u},
        Case{0x00800000u, 0x3f000000u, 0u, 0u},
        Case{0x80800000u, 0x3f000000u, 0x80000000u, 0x80000000u},
        Case{0x3f800800u, 0x3f800800u, 0xbf801000u, 0x33800000u},
        Case{0xbf800800u, 0x3f800800u, 0x3f801000u, 0xb3800000u},
        Case{0x3f800000u, 0x3f800000u, 0x33800000u, 0x3f800000u},
        Case{0x3f800000u, 0x3f800001u, 0x33800000u, 0x3f800002u},
        Case{0x3f800000u, 0x3f800000u, 0xbf800000u, 0u},
        Case{0x7f7fffffu, 0x40000000u, 0xff7fffffu, 0x7f7fffffu},
        Case{0x7f7fffffu, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u},
        Case{0x80000000u, 0x40000000u, 0x80000000u, 0x80000000u},
        Case{0x80000000u, 0x40000000u, 0u, 0u},
        Case{0x007fffffu, 0x7f7fffffu, 0x3f800000u, 0x3f800000u},
        Case{0u, 0x3f800000u, 0x80000001u, 0u},
        Case{0x80000000u, 0x3f800000u, 0x80000001u, 0x80000000u},
        Case{0x7f800000u, 0u, 0x7f800000u, 0x7fc00000u},
        Case{0x7f800000u, 1u, 0u, 0x7fc00000u},
        Case{0x7f800000u, 0x3f800000u, 0xff800000u, 0x7fc00000u},
        Case{0xff800000u, 0x3f800000u, 0xff800000u, 0xff800000u},
        Case{0x3f800000u, 0x3f800000u, 0xff800000u, 0xff800000u},
        Case{0x7f800001u, 0u, 0u, 0x7fc00000u},
    };
    for (const auto& item : cases) {
        ARMIGO_CHECK_EQ(fma_rne_ftz(item.a, item.b, item.c), item.expected);
        ARMIGO_CHECK_EQ(fma_rne_ftz(item.b, item.a, item.c), item.expected);
        ARMIGO_CHECK_EQ(fma_oracle(item.a, item.b, item.c), item.expected);
    }
}

ARMIGO_TEST(fma, matches_independent_native_fused_oracle) {
    const NearestScope nearest;
    if (!fma_oracle_ready()) { ARMIGO_CHECK(fma_oracle_ready()); return; }
    for (const auto& triple : fma_inputs(1000000)) {
        if (fma_rne_ftz(triple[0], triple[1], triple[2]) != fma_oracle(triple[0], triple[1], triple[2])) {
            record_failure(__FILE__, __LINE__, "Integer FMA differs from native fused oracle: " +
                std::to_string(triple[0]) + ", " + std::to_string(triple[1]) + ", " + std::to_string(triple[2]));
            return;
        }
    }
}

ARMIGO_TEST(fma, independent_of_host_rounding_mode) {
    const NearestScope nearest;
    if (!fma_oracle_ready()) { ARMIGO_CHECK(fma_oracle_ready()); return; }
    const auto inputs = fma_inputs(1024);
    std::vector<std::uint32_t> expected;
    for (const auto& t : inputs) expected.push_back(fma_oracle(t[0], t[1], t[2]));
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ARMIGO_CHECK_EQ(std::fesetround(mode), 0);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& t = inputs[index];
            if (fma_rne_ftz(t[0], t[1], t[2]) != expected[index]) {
                record_failure(__FILE__, __LINE__, "Integer FMA changed with host rounding mode");
                return;
            }
        }
    }
}
#endif

}  // namespace armigo::shader::test
