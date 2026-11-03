// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Compare every conversion bit through integer equality, then export finite
// floats. Raw integer results often encode NaNs or subnormal float values.
#include "operation_harness.h"

namespace {
namespace harness = armigo::shader::test::operation_harness;
namespace gx = armigo::shader::graphics;

std::vector<harness::OperationCase> cases() {
    constexpr std::uint32_t inputs[] = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x00800000u, 0x80800000u, 0x20000001u, 0xa0000001u,
        0x3effffffu, 0x3f000000u, 0x3f000001u, 0xbf000000u,
        0x3f800000u, 0xbf800000u, 0x3fc00000u, 0xbfc00000u,
        0x3fe00000u, 0xbfe00000u, 0x40200000u, 0x40600000u,
        0x477ffe80u, 0x477fff00u, 0x477fff80u, 0x47800000u,
        0x4effffffu, 0xceffffffu, 0x4f000000u, 0xcf000000u,
        0x4f000001u, 0xcf000001u, 0x4f7fffffu, 0x4f800000u,
        0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
        0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u,
        0x7fffffffu, 0xffffffffu,
    };
    std::vector<harness::OperationCase> result;
    for (const std::uint32_t profile : {0u, 11u, 19u, 26u, 27u}) {
        for (std::size_t first = 0; first < std::size(inputs); first += 4) {
            harness::OperationCase item;
            item.name = "float_to_integer";
            for (unsigned lane = 0; lane < 4; ++lane) {
                const auto input = inputs[first + lane];
                // Obtain the CPU's raw result before any render target can
                // canonicalize it. Each GPU comparison sees that word as a
                // runtime constant, independently of its F2I calculation.
                gx::GraphSpec spec;
                spec.stage = armigo::shader_input::Stage::Fragment;
                harness::put_header(spec, spec.stage);
                spec.sph[0x48] = std::byte{0xf};
                spec.nodes = {{8, gx::Literal{input}},
                              {16, gx::Pure{gx::PureKind::FloatToInteger,
                                            {0, 0, 0}, profile}}};
                for (std::uint16_t component = 0; component < 4; ++component) {
                    spec.exports.push_back({gx::OutputKind::Color, component,
                                            gx::ScalarId{1}});
                }
                spec.blocks = {{0, 2, gx::Return{}}};
                const auto graph = gx::make_graph(spec);
                const auto evaluated = gx::evaluate(std::get<gx::Program>(graph), {});
                const auto expected = std::get<gx::Evaluation>(evaluated).outputs[0].bits;
                const auto base = static_cast<std::uint32_t>(item.nodes.size());
                const auto append = [&](gx::Operation operation) {
                    item.nodes.push_back({8u * (static_cast<std::uint32_t>(item.nodes.size()) + 1),
                                          std::move(operation)});
                };
                append(gx::ConstantLoad{0, 8u * lane});
                append(gx::Pure{gx::PureKind::FloatToInteger, {base, 0, 0}, profile});
                append(gx::ConstantLoad{0, 8u * lane + 4u});
                append(gx::WordNotEqual{gx::ScalarId{base + 1}, gx::ScalarId{base + 2}});
                append(gx::Literal{0x40000000u});
                append(gx::Literal{0x3f800000u});
                append(gx::Pure{gx::PureKind::Select, {base + 3, base + 4, base + 5}, 0});
                item.outputs[lane] = base + 6;
                item.constants.push_back(input);
                item.constants.push_back(expected);
            }
            result.push_back(std::move(item));
        }
    }
    return result;
}
}  // namespace

int main() {
    return harness::run_cases(cases());
}
