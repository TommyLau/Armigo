// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "operation_harness.h"

namespace {
namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;
using Kind = gx::PureKind;

struct Case { std::uint32_t a, b, minimum, maximum; };
constexpr std::array cases{
    Case{0x3f800000, 0x40000000, 0x3f800000, 0x40000000},
    Case{0xc0000000, 0xbf800000, 0xc0000000, 0xbf800000},
    Case{0xbf800000, 0x3f800000, 0xbf800000, 0x3f800000},
    Case{0, 0x80000000, 0x80000000, 0},
    Case{0x80000000, 0x80000000, 0x80000000, 0x80000000},
    Case{0, 0, 0, 0},
    Case{1, 0x807fffff, 0x80000000, 0},
    Case{0x007fffff, 0x00800000, 0, 0x00800000},
    Case{0x807fffff, 0x80800000, 0x80800000, 0x80000000},
    Case{0x7f800000, 0xff800000, 0xff800000, 0x7f800000},
    Case{0x7f7fffff, 0x7f800000, 0x7f7fffff, 0x7f800000},
    Case{0xff7fffff, 0xff800000, 0xff800000, 0xff7fffff},
    Case{0x3f800000, 0x3f800001, 0x3f800000, 0x3f800001},
    Case{0xbf800000, 0xbf800001, 0xbf800001, 0xbf800000},
    Case{0x7fc12345, 0x3f800000, 0x3f800000, 0x3f800000},
    Case{0xff812345, 0xff800000, 0xff800000, 0xff800000},
    Case{0x7f800001, 0x80000001, 0x80000000, 0x80000000},
    Case{0xffc12345, 0x7f812345, 0x7fc00000, 0x7fc00000},
    Case{0x7fc12345, 0x7fc12345, 0x7fc00000, 0x7fc00000},
};

std::vector<harness::OperationCase> gpu_cases() {
    std::vector<harness::OperationCase> result;
    for (const auto& edge : cases) {
        harness::OperationCase item;
        item.name = "float_min_max_exact_bits";
        item.constants = {edge.a, edge.b, edge.minimum, edge.maximum};
        for (unsigned i = 0; i < 4; ++i)
            item.nodes.push_back({8 * (i + 1), gx::ConstantLoad{0, 4 * i}});
        item.nodes.push_back({40, gx::Literal{0x3f800000}});
        item.nodes.push_back({48, gx::Literal{0x40000000}});
        for (unsigned lane = 0; lane < 4; ++lane) {
            const auto index = static_cast<std::uint32_t>(item.nodes.size());
            const bool maximum = lane % 2 != 0;
            const bool reversed = lane >= 2;
            item.nodes.push_back({8 * (index + 1), gx::Pure{
                maximum ? Kind::FloatMaximum : Kind::FloatMinimum,
                {reversed ? 1u : 0u, reversed ? 0u : 1u, 0}, 0}});
            // Compare the raw result before a float render target can
            // canonicalize a NaN or lose the sign of a zero.
            item.nodes.push_back({8 * (index + 2), gx::WordNotEqual{
                gx::ScalarId{index}, gx::ScalarId{maximum ? 3u : 2u}}});
            item.nodes.push_back({8 * (index + 3), gx::Pure{
                Kind::Select, {index + 1, 5, 4}, 0}});
            item.outputs[lane] = index + 2;
        }
        result.push_back(std::move(item));
    }
    return result;
}
}  // namespace

int main() {
    return harness::run_cases(gpu_cases());
}
