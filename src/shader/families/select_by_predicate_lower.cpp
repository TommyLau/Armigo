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
// SEL writes A when (predicate XOR negate) holds and B otherwise. A negated
// predicate swaps the operands; selecting on the raw predicate with
// swapped candidates is the same raw-word choice. Both sources are read
// before the destination is written, so aliasing is safe.
void lower_select_by_predicate(const maxwell::Instruction& instruction,
                               FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::SelectByPredicate>(&instruction.payload);
    if (value == nullptr || value->predicate > 7) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    const auto a = lower.read(value->source_a);
    ScalarId b{0};
    if (const auto* reg = std::get_if<maxwell::RegisterOperand>(&value->source_b)) {
        b = lower.read(*reg);
    } else if (const auto* buffer =
                   std::get_if<maxwell::ConstantBufferOperand>(&value->source_b)) {
        b = lower.constant(*buffer);
    } else {
        b = lower.literal(std::get<std::uint32_t>(value->source_b));
    }
    const auto condition = lower.read_predicate(value->predicate);
    const auto chosen = value->negate_predicate
        ? lower.operation(PureKind::Select, {condition, b, a})
        : lower.operation(PureKind::Select, {condition, a, b});
    lower.write(value->destination, chosen);
}
}  // namespace armigo::shader::graphics::detail
