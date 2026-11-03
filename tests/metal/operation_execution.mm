// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU cases for the first registered pure operations: sign flip,
// absolute value, word and predicate selection, and the predicate
// combinators. Predicates come from runtime word comparisons so the GPU
// cannot fold them, and are observed through word selection.
#include "operation_harness.h"

namespace {

namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using gx::PureKind;

gx::Node load(std::uint32_t pc, std::uint32_t word) {
    return {pc, gx::ConstantLoad{0, 4 * word}};
}
gx::Node pure(std::uint32_t pc, PureKind kind, std::array<std::uint32_t, 3> operands) {
    return {pc, gx::Pure{kind, operands, 0}};
}

// Words 0..3 are values; words 4 and 5 are compared with word 6 to make two
// runtime predicates.
std::vector<harness::OperationCase> cases() {
    std::vector<harness::OperationCase> result;
    {
        harness::OperationCase item;
        item.name = "float_sign";
        item.nodes = {load(8, 0), load(16, 1), load(24, 2), load(32, 3),
                      pure(40, PureKind::FloatNegate, {0, 0, 0}),
                      pure(48, PureKind::FloatNegate, {1, 0, 0}),
                      pure(56, PureKind::FloatAbsolute, {2, 0, 0}),
                      pure(64, PureKind::FloatAbsolute, {3, 0, 0})};
        item.outputs = {4, 5, 6, 7};
        // 1.5, -0.0, -infinity, -pi. Subnormal and NaN payloads are CPU-tested:
        // a float render target need not preserve them.
        item.constants = {0x3fc00000u, 0x80000000u, 0xff800000u, 0xc0490fdbu};
        result.push_back(item);
    }
    for (const bool first : {false, true}) {
        for (const bool second : {false, true}) {
            harness::OperationCase item;
            item.name = first ? (second ? "predicates_tt" : "predicates_tf")
                              : (second ? "predicates_ft" : "predicates_ff");
            item.nodes = {
                load(8, 0), load(16, 1), load(24, 4), load(32, 5), load(40, 6),
                {48, gx::WordNotEqual{gx::ScalarId{2}, gx::ScalarId{4}}},   // 5: p first
                {56, gx::WordNotEqual{gx::ScalarId{3}, gx::ScalarId{4}}},   // 6: p second
                pure(64, PureKind::PredicateAnd, {5, 6, 0}),                // 7
                pure(72, PureKind::PredicateOr, {5, 6, 0}),                 // 8
                pure(80, PureKind::PredicateXor, {5, 6, 0}),                // 9
                pure(88, PureKind::PredicateSelect, {5, 6, 9}),             // 10
                pure(96, PureKind::Select, {7, 0, 1}),                      // 11
                pure(104, PureKind::Select, {8, 0, 1}),                     // 12
                pure(112, PureKind::Select, {9, 0, 1}),                     // 13
                pure(120, PureKind::Select, {10, 0, 1}),                    // 14
            };
            item.outputs = {11, 12, 13, 14};
            item.constants = {0x3f800000u, 0x40000000u, 0, 0,
                              first ? 1u : 0u, second ? 1u : 0u, 0u};
            result.push_back(item);
        }
    }
    return result;
}

}  // namespace

int main() {
    return harness::run_cases(cases());
}
