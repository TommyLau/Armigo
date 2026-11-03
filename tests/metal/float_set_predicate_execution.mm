// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU cases for FloatComparePredicate: all sixteen compare codes, with
// and without FTZ, over operand pairs that include NaN, signed zeros,
// subnormals and infinities. Each predicate is observed through a word
// selection between 1.0 and 2.0, so NaN inputs never reach the render target.
#include "operation_harness.h"

namespace {

namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using gx::PureKind;

std::vector<harness::OperationCase> cases() {
    static const std::array<std::pair<std::uint32_t, std::uint32_t>, 9> pairs{{
        {0x3f800000u, 0x40000000u},  // 1 < 2
        {0x40000000u, 0x3f800000u},  // 2 > 1
        {0x3f800000u, 0x3f800000u},  // equal
        {0x80000000u, 0x00000000u},  // -0 vs +0
        {0x00000001u, 0x00000000u},  // subnormal vs +0 (equal only under FTZ)
        {0x7fc00000u, 0x3f800000u},  // NaN vs 1
        {0x3f800000u, 0xffc12345u},  // 1 vs negative NaN
        {0xff800000u, 0x7f800000u},  // -inf vs +inf
        {0x807fffffu, 0x00800000u},  // negative subnormal vs smallest normal
    }};
    static std::vector<std::string> names;
    names.reserve(pairs.size() * 8);
    std::vector<harness::OperationCase> result;
    for (std::size_t p = 0; p < pairs.size(); ++p) {
        for (std::uint32_t ftz : {0u, 16u}) {
            for (std::uint32_t first = 0; first < 16; first += 4) {
                harness::OperationCase item;
                names.push_back("compare_pair" + std::to_string(p) + (ftz ? "_ftz_" : "_") +
                                std::to_string(first));
                item.name = names.back().c_str();
                item.nodes = {{8, gx::ConstantLoad{0, 0}}, {16, gx::ConstantLoad{0, 4}},
                              {24, gx::Literal{0x3f800000u}}, {32, gx::Literal{0x40000000u}}};
                for (std::uint32_t lane = 0; lane < 4; ++lane) {
                    const auto predicate = static_cast<std::uint32_t>(item.nodes.size());
                    item.nodes.push_back({40 + 16 * lane, gx::Pure{PureKind::FloatComparePredicate,
                                                                   {0, 1, 0}, (first + lane) | ftz}});
                    item.nodes.push_back({48 + 16 * lane, gx::Pure{PureKind::Select,
                                                                   {predicate, 2, 3}, 0}});
                    item.outputs[lane] = predicate + 1;
                }
                item.constants = {pairs[p].first, pairs[p].second};
                result.push_back(item);
            }
        }
    }
    return result;
}

}  // namespace

int main() { return harness::run_cases(cases()); }
