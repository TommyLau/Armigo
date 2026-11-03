// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader/graphics_control_flow.h>
namespace armigo::shader::test {
inline graphics::GraphResult backend_graph_fixture() {
    graphics::GraphSpec spec;
    spec.stage = shader_input::Stage::Vertex;
    const auto header = shader_input::sph_stage_type_for(spec.stage) << 10;
    for (unsigned byte = 0; byte < 4; ++byte) spec.sph[byte] = std::byte(header >> (8 * byte));
    spec.sph[0x35] = std::byte{0xf0};
    spec.nodes.push_back({8, graphics::Literal{0x3f800000u}});
    spec.blocks.push_back({0, 1, graphics::Return{}});
    for (unsigned lane = 0; lane < 4; ++lane)
        spec.exports.push_back({graphics::OutputKind::Position,
            static_cast<std::uint16_t>(lane), graphics::ScalarId{0}});
    return graphics::make_graph(spec);
}
}  // namespace armigo::shader::test
