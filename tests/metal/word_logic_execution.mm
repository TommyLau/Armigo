// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU cases for WordLogic (every operation and inversion) and
// WordIsZero, observed through word selection. Inputs are runtime
// constant-buffer words; outputs are chosen to be ordinary finite floats.
#include "operation_harness.h"

namespace {

namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using gx::PureKind;

gx::Node load(std::uint32_t pc, std::uint32_t word) { return {pc, gx::ConstantLoad{0, 4 * word}}; }
gx::Node pure(std::uint32_t pc, PureKind kind, std::array<std::uint32_t, 3> operands,
              std::uint32_t immediate = 0) {
    return {pc, gx::Pure{kind, operands, immediate}};
}

std::vector<harness::OperationCase> cases() {
    std::vector<harness::OperationCase> result;
    // Words: 0 = A, 1 = B. Bit patterns are chosen so every result is a
    // normal float (exponent bits never all zero or all one).
    for (std::uint32_t control = 0; control < 16; control += 4) {
        harness::OperationCase item;
        item.name = control == 0 ? "logic_plain" : control == 4 ? "logic_invert_a"
                  : control == 8 ? "logic_invert_b" : "logic_invert_both";
        item.nodes = {load(8, 0), load(16, 1),
                      pure(24, PureKind::WordLogic, {0, 1, 0}, control | 0),
                      pure(32, PureKind::WordLogic, {0, 1, 0}, control | 1),
                      pure(40, PureKind::WordLogic, {0, 1, 0}, control | 2),
                      pure(48, PureKind::WordLogic, {0, 1, 0}, control | 3),
                      // Map each raw result into a normal float: clear the
                      // exponent's top bit and set its bottom bit.
                      {56, gx::Literal{0xbfffffffu}}, {64, gx::Literal{0x00800000u}}};
        // r & 0xbfffffff | 0x00800000 for lanes 0..3.
        for (unsigned lane = 0; lane < 4; ++lane) {
            const std::uint32_t base = static_cast<std::uint32_t>(item.nodes.size());
            item.nodes.push_back(pure(72 + 16 * lane, PureKind::WordLogic, {2 + lane, 6, 0}, 0));
            item.nodes.push_back(pure(80 + 16 * lane, PureKind::WordLogic, {base, 7, 0}, 1));
            item.outputs[lane] = base + 1;
        }
        item.constants = {0x3c96a5a5u, 0x4a17c33cu};
        result.push_back(item);
    }
    {
        harness::OperationCase item;
        item.name = "word_is_zero";
        item.nodes = {load(8, 0), load(16, 1), load(24, 2), load(32, 3),
                      pure(40, PureKind::WordIsZero, {0, 0, 0}),
                      pure(48, PureKind::WordIsZero, {1, 0, 0}),
                      pure(56, PureKind::Select, {4, 2, 3}),
                      pure(64, PureKind::Select, {5, 2, 3})};
        item.outputs = {6, 7, 2, 3};
        item.constants = {0u, 0x80000000u, 0x3f800000u, 0x40000000u};
        result.push_back(item);
    }
    return result;
}

}  // namespace

int main() { return harness::run_cases(cases()); }
