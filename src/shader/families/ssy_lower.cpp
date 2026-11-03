// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../graphics_families.h"

namespace armigo::shader::graphics::detail {

void lower_ssy(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* setup = std::get_if<maxwell::ReconvergenceSetup>(&instruction.payload);
    if (setup == nullptr || instruction.predicate.index != 7 ||
        instruction.predicate.inverted || setup->target_pc <= instruction.pc ||
        setup->target_pc % 8 != 0 || setup->target_pc % 32 == 0) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    // The decoder has propagated this token and replaced each SYNC with a
    // direct branch. Keeping the marker preserves instruction-slot traversal.
}

}  // namespace armigo::shader::graphics::detail
