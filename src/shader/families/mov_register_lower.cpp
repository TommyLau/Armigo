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

// Pilot family unit: the register-move lowering is behaviorally the former
// legacy case. It verifies the payload type and the admitted masks, then
// lowers to write(read) so RZ sources, RZ destinations, and destination and
// source overlap behave exactly as before.
void lower_mov_register(const maxwell::Instruction& instruction,
                        FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::MovRegister>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->write_mask != 15 && value->write_mask != 1) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    // A full-word register copy is an SSA alias: the source bits are
    // preserved exactly, with no arithmetic or NaN canonicalization.
    lower.write(value->destination, lower.read(value->source));
}

}  // namespace armigo::shader::graphics::detail
