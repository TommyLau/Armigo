// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"
#include "graph_backend_fixture.h"
namespace armigo::shader::test {
ARMIGO_TEST(graphics_control_flow, cpu_executes_shared_backend_graph) {
    const auto result = backend_graph_fixture();
    const auto* program = std::get_if<graphics::Program>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    const auto output = graphics::evaluate(*program, {});
    const auto* evaluation = std::get_if<graphics::Evaluation>(&output);
    ARMIGO_CHECK(evaluation != nullptr);
    if (!evaluation) return;
    ARMIGO_CHECK_EQ(evaluation->outputs.size(), 4u);
    for (unsigned lane = 0; lane < 4 && lane < evaluation->outputs.size(); ++lane) {
        ARMIGO_CHECK(evaluation->outputs[lane].kind == graphics::OutputKind::Position);
        ARMIGO_CHECK_EQ(evaluation->outputs[lane].component,
                        static_cast<std::uint16_t>(lane));
        ARMIGO_CHECK_EQ(evaluation->outputs[lane].bits, 0x3f800000u);
    }
}
}  // namespace armigo::shader::test
