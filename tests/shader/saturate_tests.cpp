// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/saturate.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef ARMIGO_SATURATE_ORACLE_ONLY
#include "test_support.h"
#endif

namespace armigo::shader::test {
namespace {

constexpr std::uint32_t kOne = 0x3f800000u;

// Shared sweep body: output bounds, sign, and idempotence. The generated
// sweeps claim invariant coverage, not independent semantic proof; the
// authored exact-bit fixtures carry the expected values. Returns a
// description of the first violating input, or an empty string.
std::string first_invariant_violation(const std::vector<std::uint32_t>& inputs) {
    for (const auto bits : inputs) {
        const auto once = saturate_float32(bits);
        if (saturate_float32(once) != once) {
            return "saturation of " + std::to_string(bits) + " is not idempotent";
        }
        if (once > kOne || (once & 0x80000000u) != 0) {
            return "saturation of " + std::to_string(bits) +
                   " escaped [0, 1]: " + std::to_string(once);
        }
    }
    return {};
}

}  // namespace

// Raw binary32 words covering the policy's decision boundaries. The Metal
// differential test links this fixture list and checks the generated helper
// against saturate_float32 on exactly these inputs.
std::vector<std::uint32_t> saturate_inputs(std::size_t random_count) {
    constexpr std::array<std::uint32_t, 32> edges = {
        0u, 0x80000000u, 1u, 2u, 3u, 0x007fffffu, 0x80000001u, 0x807fffffu,
        0x00800000u, 0x80800000u, 0x3f000000u, 0xbf000000u, 0x3f7ffffeu,
        0x3f7fffffu, 0xbf7fffffu, 0x3f800000u, 0xbf800000u, 0x3f800001u,
        0x3f800002u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
        0x7f800001u, 0x7f800002u, 0xff800001u, 0x7fc00000u, 0xffc00000u,
        0x7fc12345u, 0xffc12345u, 0x7fffffffu, 0xffffffffu,
    };
    std::vector<std::uint32_t> inputs(edges.begin(), edges.end());
    // Every exponent at selected fractions, both signs: 2^9 exponents sample.
    for (unsigned exponent = 0; exponent < 256; ++exponent) {
        for (const auto fraction : {0u, 1u, 0x400000u, 0x7fffffu}) {
            inputs.push_back((exponent << 23) | fraction);
            inputs.push_back(0x80000000u | (exponent << 23) | fraction);
        }
    }
    // Dense windows: both sides of one, the analogous window around negative
    // one, the subnormal range, and the NaN window on both sides of the
    // quiet-NaN anchor.
    for (std::uint32_t offset = 0; offset <= 0x2000; ++offset) {
        inputs.push_back(0x3f7ff000u + offset);
        inputs.push_back(0xbf7ff000u + offset);
    }
    for (std::uint32_t offset = 0; offset <= 0x1000; ++offset) {
        inputs.push_back(offset);
        inputs.push_back(0x007ff000u + offset);
        inputs.push_back(0x7f800000u + offset);
        inputs.push_back(0x7fbfff00u + offset);
    }
    for (std::size_t index = 0; index < random_count; ++index) {
        std::uint32_t state = 0x2545f491u + static_cast<std::uint32_t>(index);
        state ^= state << 7;
        state ^= state >> 11;
        state ^= state << 19;
        inputs.push_back(state);
    }
    return inputs;
}

#ifndef ARMIGO_SATURATE_ORACLE_ONLY

ARMIGO_TEST(saturate, authored_exact_bits) {
    struct Case { std::uint32_t input, expected; };
    constexpr std::array cases = {
        Case{0x00000000u, 0x00000000u},  // +0 stays +0
        Case{0x00000001u, 0x00000001u},  // smallest positive subnormal kept
        Case{0x00000002u, 0x00000002u},
        Case{0x007fffffu, 0x007fffffu},  // largest positive subnormal kept
        Case{0x00800000u, 0x00800000u},  // smallest positive normal kept
        Case{0x3f000000u, 0x3f000000u},  // 0.5 kept
        Case{0x3f7ffffeu, 0x3f7ffffeu},  // below one kept
        Case{0x3f7fffffu, 0x3f7fffffu},  // largest below one kept
        Case{0x3f800000u, 0x3f800000u},  // one kept
        Case{0x3f800001u, 0x3f800000u},  // smallest above one clamps
        Case{0x3f800002u, 0x3f800000u},
        Case{0x7f7fffffu, 0x3f800000u},  // largest finite clamps
        Case{0x7f800000u, 0x3f800000u},  // +infinity clamps
        Case{0x7f800001u, 0x00000000u},  // signaling NaN drops
        Case{0x7f800002u, 0x00000000u},
        Case{0x7fc00000u, 0x00000000u},  // quiet NaN drops
        Case{0x7fc12345u, 0x00000000u},  // quiet NaN with payload drops
        Case{0x7fffffffu, 0x00000000u},
        Case{0x80000000u, 0x00000000u},  // -0 maps to +0 by profile policy
        Case{0x80000001u, 0x00000000u},  // negative subnormal drops
        Case{0x807fffffu, 0x00000000u},
        Case{0x80800000u, 0x00000000u},  // negative smallest normal drops
        Case{0xbf7fffffu, 0x00000000u},  // just inside -1 drops
        Case{0xbf800000u, 0x00000000u},  // -1 drops
        Case{0xc0000000u, 0x00000000u},  // -2 drops
        Case{0xff7fffffu, 0x00000000u},  // negative largest finite drops
        Case{0xff800000u, 0x00000000u},  // -infinity drops
        Case{0xff800001u, 0x00000000u},  // negative signaling NaN drops
        Case{0xffc00000u, 0x00000000u},  // negative quiet NaN drops
        Case{0xffc12345u, 0x00000000u},
        Case{0xffffffffu, 0x00000000u},
    };
    for (const auto& item : cases) {
        ARMIGO_CHECK_EQ(saturate_float32(item.input), item.expected);
    }
}

ARMIGO_TEST(saturate, exponent_class_sweeps_hold_invariants) {
    // Full fraction sweeps for the exponent classes the policy branches on
    // and their neighbours: zero, the one-adjacent normals, and infinity/NaN.
    // They check the structural invariants only; the authored fixtures own
    // the exact expected values.
    for (const unsigned exponent : {0u, 126u, 127u, 128u, 254u, 255u}) {
        std::vector<std::uint32_t> inputs;
        inputs.reserve(2 * (1u << 23));
        for (std::uint32_t fraction = 0; fraction < 0x00800000u; ++fraction) {
            inputs.push_back(fraction | (exponent << 23));
            inputs.push_back(fraction | 0x80000000u | (exponent << 23));
        }
        const auto violation = first_invariant_violation(inputs);
        if (!violation.empty()) {
            record_failure(__FILE__, __LINE__,
                           "exponent " + std::to_string(exponent) + ": " + violation);
            return;
        }
    }
}

ARMIGO_TEST(saturate, dense_windows_stay_ordered_and_idempotent) {
    // Within the nonnegative non-NaN encodings the unsigned word order matches
    // the value order, so saturation must be monotone there.
    std::uint32_t previous = 0;
    for (std::uint32_t bits = 0x3f7f0000u; bits <= 0x3f810000u; ++bits) {
        const auto saturated = saturate_float32(bits);
        ARMIGO_CHECK((saturated & 0x80000000u) == 0);
        ARMIGO_CHECK(saturated <= kOne);
        if (bits > 0x3f7f0000u && saturated < previous) {
            record_failure(__FILE__, __LINE__,
                           "saturation decreased across " + std::to_string(bits));
            return;
        }
        ARMIGO_CHECK_EQ(saturate_float32(saturated), saturated);
        previous = saturated;
    }
    const auto violation = first_invariant_violation(saturate_inputs(4096));
    if (!violation.empty()) {
        record_failure(__FILE__, __LINE__, violation);
    }
}

ARMIGO_TEST(saturate, random_inputs_hold_invariants) {
    const auto violation = first_invariant_violation(saturate_inputs(1000000));
    if (!violation.empty()) {
        record_failure(__FILE__, __LINE__, violation);
    }
}

ARMIGO_TEST(saturate, emitted_msl_declares_integer_helper) {
    const auto emitted = emit_saturate_msl();
    ARMIGO_CHECK(emitted.find("inline uint armigo_saturate_float32(uint bits)") !=
                 std::string::npos);
    // The helper must be pure integer mapping: no host min/max dependence and
    // no floating-point type in the emitted code. The "float" hits inside the
    // helper's own name are preceded by '_' and followed by '32', never by a
    // space or an opening parenthesis.
    ARMIGO_CHECK(emitted.find("fmin") == std::string::npos);
    ARMIGO_CHECK(emitted.find("fmax") == std::string::npos);
    ARMIGO_CHECK(emitted.find(" float") == std::string::npos);
    ARMIGO_CHECK(emitted.find("(float") == std::string::npos);
    ARMIGO_CHECK(emitted.find("0x3f800000u") != std::string::npos);
}

#endif

}  // namespace armigo::shader::test
