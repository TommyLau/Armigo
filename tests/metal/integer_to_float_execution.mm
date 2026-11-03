// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU cases for IntegerToFloat: every admitted profile on the edge
// inputs (power-of-two boundaries, the minimum integer under the absolute
// modifier, and rounding carries). Integer conversion results are always
// zero or finite normal floats, so the raw words export directly.
#include "operation_harness.h"

namespace {

namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using gx::PureKind;

constexpr std::uint32_t kSigned32Rn = 0x05u;
constexpr std::uint32_t kUnsigned32Rz = 0x1cu;
constexpr std::uint32_t kUnsigned32Rn = 0x04u;
constexpr std::uint32_t kSigned32RnAbsolute = 0x25u;
constexpr std::uint32_t kSigned16Rn = 0x03u;

// One case per admitted profile: four runtime words convert in parallel.
harness::OperationCase profile(const char* name, std::uint32_t immediate,
                               std::array<std::uint32_t, 4> words) {
    harness::OperationCase item;
    item.name = name;
    for (unsigned lane = 0; lane < 4; ++lane) {
        item.nodes.push_back({8u + 8u * lane, gx::ConstantLoad{0, 4 * lane}});
        // Loads and conversions interleave: the load of lane L is node 2L and
        // its conversion node 2L + 1.
        item.nodes.push_back({48u + 8u * lane,
                              gx::Pure{PureKind::IntegerToFloat, {2u * lane, 0, 0}, immediate}});
        item.outputs[lane] = 2u * lane + 1u;
    }
    item.constants.assign(words.begin(), words.end());
    return item;
}

std::vector<harness::OperationCase> cases() {
    return {
        // Toward zero: the full word truncates; the maximum drops to the
        // largest float below 2^32.
        profile("i2f_u32_rtz", kUnsigned32Rz,
                {0xffffffffu, 0x80000001u, 0x00000001u, 0x7fffff83u}),
        // Nearest-even carries 0xffffffff to 2^32 and rounds the tie below
        // 2^24 to the even mantissa.
        profile("i2f_u32_rn", kUnsigned32Rn,
                {0xffffffffu, 0x80000001u, 0x00ffffffu, 0x01000001u}),
        // Signed: the minimum stays itself, the maximum rounds up to 2^31,
        // and small negatives convert exactly.
        profile("i2f_s32_rn", kSigned32Rn,
                {0x80000000u, 0x7fffffffu, 0xfffffffeu, 0x12345678u}),
        // Absolute: the minimum stays negative (wraparound absolute value);
        // everything else converts the magnitude.
        profile("i2f_s32_rn_abs", kSigned32RnAbsolute,
                {0x80000000u, 0x80000001u, 0xffffffffu, 0x12345678u}),
        // Signed 16-bit source: the low halfword, sign-extended.
        profile("i2f_s16_rn", kSigned16Rn,
                {0x00008000u, 0x00007fffu, 0x12345678u, 0x0000ffffu}),
    };
}

}  // namespace

int main() { return harness::run_cases(cases()); }
