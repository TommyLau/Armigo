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
void lower_float_min_max(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatMinMax>(&instruction.payload);
    if (!value) lower.fail(Error::Code::InvalidInstruction);
    if (value->flush != maxwell::FlushMode::FlushToZero)
        lower.fail(Error::Code::UnsupportedModifier);
    if (((value->negate_a || value->negate_b) &&
         (value->absolute_a || value->absolute_b)) ||
        (value->negate_a && value->negate_b))
        lower.fail(Error::Code::UnsupportedModifier);
    const bool source_matches =
        (instruction.family == maxwell::OpcodeFamily::FloatMinMaxRegister &&
         std::holds_alternative<maxwell::RegisterOperand>(value->source_b)) ||
        (instruction.family == maxwell::OpcodeFamily::FloatMinMaxConstant &&
         std::holds_alternative<maxwell::ConstantBufferOperand>(value->source_b)) ||
        (instruction.family == maxwell::OpcodeFamily::FloatMinMaxImmediate &&
         std::holds_alternative<maxwell::ShortFloatImmediate>(value->source_b));
    if (!source_matches) lower.fail(Error::Code::InvalidInstruction);
    // Read both sources before writing an aliased destination.
    auto left = lower.read(value->source_a);
    ScalarId b{0};
    if (const auto* source = std::get_if<maxwell::RegisterOperand>(&value->source_b)) {
        b = lower.read(*source);
    } else if (const auto* buffer =
                   std::get_if<maxwell::ConstantBufferOperand>(&value->source_b)) {
        b = lower.constant(*buffer);
    } else if (const auto* immediate =
                   std::get_if<maxwell::ShortFloatImmediate>(&value->source_b)) {
        b = lower.literal(immediate->raw_bits);
    } else {
        lower.fail(Error::Code::InvalidInstruction);
    }
    // Operand modifiers compose absolute first, then negate. Each is a
    // registered sign-bit node, so the min/max kind sees
    // the word the instruction's own operand modifier would produce.
    if (value->absolute_a) left = lower.operation(PureKind::FloatAbsolute, {left});
    if (value->negate_a) left = lower.operation(PureKind::FloatNegate, {left});
    if (value->absolute_b) b = lower.operation(PureKind::FloatAbsolute, {b});
    if (value->negate_b) b = lower.operation(PureKind::FloatNegate, {b});
    // The selecting predicate is admitted as PT only, so the negation bit
    // alone decides: neg_pred=0 selects the minimum, neg_pred=1 the maximum.
    const auto kind = value->negate_predicate ? PureKind::FloatMaximum
                                              : PureKind::FloatMinimum;
    lower.write(value->destination, lower.operation(kind, {left, b}));
}
}  // namespace armigo::shader::graphics::detail
