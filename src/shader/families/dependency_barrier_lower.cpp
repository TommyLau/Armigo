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

#include <variant>

namespace armigo::shader::graphics::detail {

// The dependency-barrier lowering: the hardware form waits on scoreboard
// dependencies, which have no meaning in the in-order translation, so the
// barrier produces no nodes and no register writes.
void lower_dependency_barrier(const maxwell::Instruction& instruction,
                              FamilyLower& lower) {
    if (std::get_if<maxwell::DependencyBarrier>(&instruction.payload) ==
        nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
}

}  // namespace armigo::shader::graphics::detail
