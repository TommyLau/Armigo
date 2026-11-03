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

// The constant-buffer register-move lowering: verifies the payload type and
// the admitted masks, then lowers to write(constant) so a forged payload or
// an out-of-range binding and offset fail through the same checked constant
// path as the other constant forms.
void lower_mov_constant(const maxwell::Instruction& instruction,
                        FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::MovConstant>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->write_mask != 15 && value->write_mask != 1) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    // A full-word constant-buffer load is an SSA alias of the raw buffer
    // word: the stored bits are preserved exactly, with no arithmetic, no
    // NaN canonicalization, and no bit reinterpretation.
    lower.write(value->destination, lower.constant(value->source));
}

}  // namespace armigo::shader::graphics::detail
