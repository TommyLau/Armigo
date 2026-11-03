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

// The range-reduction lowering, shared by the register and constant-buffer
// forms: the admitted result is the source unchanged, independent of the
// mode. The paired multi-function op consumes the result as an ordinary
// float, so the copy preserves the source bits exactly, with no arithmetic
// and no NaN canonicalization. Absolute then negate apply as exact sign-bit
// operations in instruction order.
void lower_range_reduction(const maxwell::Instruction& instruction,
                           FamilyLower& lower) {
    const auto* value =
        std::get_if<maxwell::RangeReduction>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    const auto* register_source =
        std::get_if<maxwell::RegisterOperand>(&value->source);
    auto result = register_source != nullptr
        ? lower.read(*register_source)
        : lower.constant(std::get<maxwell::ConstantBufferOperand>(value->source));
    if (value->absolute) result = lower.operation(PureKind::FloatAbsolute, {result});
    if (value->negate) result = lower.operation(PureKind::FloatNegate, {result});
    lower.write(value->destination, result);
}

}  // namespace armigo::shader::graphics::detail
