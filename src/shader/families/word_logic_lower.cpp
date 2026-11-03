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
// LOP: both sources are read before any write; the predicate result is
// written before the destination register, and is derived
// from the final word (F, T, result == 0, result != 0).
void lower_word_logic(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::WordLogic>(&instruction.payload);
    if (value == nullptr || value->predicate_destination > 7 ||
        static_cast<unsigned>(value->operation) > 3 ||
        static_cast<unsigned>(value->predicate_operation) > 3) {
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
    const std::uint32_t control = static_cast<std::uint32_t>(value->operation) |
                                  (value->invert_a ? 4u : 0u) |
                                  (value->invert_b ? 8u : 0u);
    const auto result = lower.operation(PureKind::WordLogic, {a, b}, control);
    using Op = maxwell::LogicPredicate;
    if (value->predicate_destination != 7) {
        PredicateId flag{0};
        switch (value->predicate_operation) {
            case Op::False: flag = lower.predicate_literal(false); break;
            case Op::True: flag = lower.predicate_literal(true); break;
            case Op::Zero: flag = lower.predicate_operation(PureKind::WordIsZero, {result}); break;
            case Op::NonZero:
                flag = lower.negate_predicate(
                    lower.predicate_operation(PureKind::WordIsZero, {result}));
                break;
        }
        lower.write_predicate(value->predicate_destination, flag);
    }
    lower.write(value->destination, result);
}
}  // namespace armigo::shader::graphics::detail
