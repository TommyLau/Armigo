// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU cases for the FloatRoundIntegral operation: the four rounding
// directions over halves, near-integers, zeros, subnormals and infinities,
// each compared bit for bit against the CPU reference. NaN payload checks
// stay on the CPU; here a rounded NaN is only observed through a later
// operation, because a float render target does not promise NaN payloads.
#include "operation_harness.h"

namespace {

namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using gx::PureKind;

gx::Node load(std::uint32_t pc, std::uint32_t word) {
    return {pc, gx::ConstantLoad{0, 4 * word}};
}
gx::Node pure(std::uint32_t pc, PureKind kind, std::array<std::uint32_t, 3> operands,
              std::uint32_t immediate = 0) {
    return {pc, gx::Pure{kind, operands, immediate}};
}

// One rounding direction over four finite inputs, exported directly.
harness::OperationCase direction(const char* name, std::uint32_t immediate,
                                 std::array<std::uint32_t, 4> inputs) {
    harness::OperationCase item;
    item.name = name;
    item.nodes = {load(8, 0), load(16, 1), load(24, 2), load(32, 3),
                  pure(40, PureKind::FloatRoundIntegral, {0}, immediate),
                  pure(48, PureKind::FloatRoundIntegral, {1}, immediate),
                  pure(56, PureKind::FloatRoundIntegral, {2}, immediate),
                  pure(64, PureKind::FloatRoundIntegral, {3}, immediate)};
    item.outputs = {4, 5, 6, 7};
    item.constants = {inputs[0], inputs[1], inputs[2], inputs[3]};
    return item;
}

}  // namespace

int main() {
    std::vector<harness::OperationCase> cases;
    // Halves and near-integers: floor toward -infinity, ceil toward
    // +infinity, trunc toward zero, nearest-even on exact ties.
    cases.push_back(direction("floor_finite", 1,
                              {0x3f000000u, 0xbf000000u, 0x3fc00000u, 0xbfc00000u}));
    cases.push_back(direction("ceil_finite", 2,
                              {0x3f000000u, 0xbf000000u, 0x3fffffffu, 0xc0200000u}));
    cases.push_back(direction("round_even_finite", 0,
                              {0x40200000u, 0x40600000u, 0xc0200000u, 0x4b000000u}));
    cases.push_back(direction("trunc_finite", 3,
                              {0x3fffffffu, 0xbfc00000u, 0x4affffffu, 0xbdcccccdu}));
    // Zeros keep their sign, subnormal inputs flush, infinities pass.
    cases.push_back(direction("specials", 1,
                              {0x80000000u, 0x00000001u, 0x80000001u, 0xff800000u}));
    cases.push_back(direction("infinity", 2,
                              {0x7f800000u, 0x4b000000u, 0x7f7fffffu, 0x3f800000u}));
    {
        // A rounded NaN keeps being a NaN: every direction's result is
        // nonzero, observed through a word-zero test and selection.
        harness::OperationCase item;
        item.name = "nan_stays_nonzero";
        item.nodes = {
            load(8, 0), load(16, 1), load(24, 2),                       // words 0..2
            pure(32, PureKind::FloatRoundIntegral, {0}, 1),             // 3 floor
            pure(40, PureKind::FloatRoundIntegral, {0}, 2),             // 4 ceil
            pure(48, PureKind::FloatRoundIntegral, {0}, 3),             // 5 trunc
            pure(56, PureKind::FloatRoundIntegral, {0}, 0),             // 6 rne
            pure(64, PureKind::WordIsZero, {3, 0, 0}),                  // 7
            pure(72, PureKind::WordIsZero, {4, 0, 0}),                  // 8
            pure(80, PureKind::WordIsZero, {5, 0, 0}),                  // 9
            pure(88, PureKind::WordIsZero, {6, 0, 0}),                  // 10
            pure(96, PureKind::Select, {7, 1, 2}),                      // 11
            pure(104, PureKind::Select, {8, 1, 2}),                     // 12
            pure(112, PureKind::Select, {9, 1, 2}),                     // 13
            pure(120, PureKind::Select, {10, 1, 2}),                    // 14
        };
        item.outputs = {11, 12, 13, 14};
        item.constants = {0x7fc12345u, 0x3f800000u, 0x40000000u};
        cases.push_back(item);
    }
    return harness::run_cases(cases);
}
