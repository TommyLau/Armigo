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

// One lowering for all three FSETP forms; the payload's source variant
// carries the second operand. The comparison runs through the registered
// FloatComparePredicate kind with the compare code and flush-to-zero flag in
// its immediate; the sign modifiers are separate pure sign-bit operations
// applied absolute-then-negate, upstream of the comparison's own flush. Both
// destination values are computed before either predicate register is
// written, destination A first, so an aliased source predicate or an
// A==B destination observes the instruction's own semantics and not a
// clobbered intermediate.
void lower_float_set_predicate(const maxwell::Instruction& instruction,
                               FamilyLower& lower) {
    const auto* value =
        std::get_if<maxwell::FloatSetPredicate>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->destination_a > 7 || value->destination_b > 7 ||
        value->boolean_predicate.index > 7) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (static_cast<unsigned>(value->comparison) > 15 ||
        static_cast<unsigned>(value->boolean_operation) > 2) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    using Family = maxwell::OpcodeFamily;
    const bool valid_source =
        (instruction.family == Family::FloatSetPredicateRegister &&
         std::holds_alternative<maxwell::RegisterOperand>(value->source_b)) ||
        (instruction.family == Family::FloatSetPredicateConstant &&
         std::holds_alternative<maxwell::ConstantBufferOperand>(value->source_b)) ||
        (instruction.family == Family::FloatSetPredicateImmediate &&
         std::holds_alternative<maxwell::ShortFloatImmediate>(value->source_b));
    if (!valid_source) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (const auto* immediate =
            std::get_if<maxwell::ShortFloatImmediate>(&value->source_b);
        immediate != nullptr && (immediate->raw_bits & 0xfffu) != 0) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    auto operand_a = lower.read(value->source_a);
    if (value->absolute_a) {
        operand_a = lower.operation(PureKind::FloatAbsolute, {operand_a});
    }
    if (value->negate_a) {
        operand_a = lower.operation(PureKind::FloatNegate, {operand_a});
    }
    auto operand_b = ScalarId{0};
    if (const auto* source =
            std::get_if<maxwell::RegisterOperand>(&value->source_b)) {
        operand_b = lower.read(*source);
    } else if (const auto* source =
                   std::get_if<maxwell::ConstantBufferOperand>(
                       &value->source_b)) {
        operand_b = lower.constant(*source);
    } else {
        const auto* immediate_source =
            std::get_if<maxwell::ShortFloatImmediate>(&value->source_b);
        if (immediate_source == nullptr) {
            lower.fail(Error::Code::InvalidInstruction);
        }
        operand_b = lower.literal(immediate_source->raw_bits);
    }
    if (value->absolute_b) {
        operand_b = lower.operation(PureKind::FloatAbsolute, {operand_b});
    }
    if (value->negate_b) {
        operand_b = lower.operation(PureKind::FloatNegate, {operand_b});
    }
    const std::uint32_t immediate =
        static_cast<std::uint32_t>(value->comparison) |
        (value->flush_to_zero ? 0x10u : 0u);
    const auto comparison = lower.predicate_operation(
        PureKind::FloatComparePredicate, {operand_a, operand_b}, immediate);
    auto gate = lower.read_predicate(value->boolean_predicate.index);
    if (value->boolean_predicate.inverted) {
        gate = lower.negate_predicate(gate);
    }
    const auto inverted_comparison = lower.negate_predicate(comparison);
    const auto combine = [&](PureKind kind) {
        return lower.predicate_operation(kind, {comparison, gate});
    };
    const auto combine_inverted = [&](PureKind kind) {
        return lower.predicate_operation(kind, {inverted_comparison, gate});
    };
    PredicateId result_a{0};
    PredicateId result_b{0};
    using Operation = maxwell::PredicateBooleanOperation;
    switch (value->boolean_operation) {
        case Operation::And:
            result_a = combine(PureKind::PredicateAnd);
            result_b = combine_inverted(PureKind::PredicateAnd);
            break;
        case Operation::Or:
            result_a = combine(PureKind::PredicateOr);
            result_b = combine_inverted(PureKind::PredicateOr);
            break;
        case Operation::Xor:
            result_a = combine(PureKind::PredicateXor);
            result_b = combine_inverted(PureKind::PredicateXor);
            break;
    }
    lower.write_predicate(value->destination_a, result_a);
    lower.write_predicate(value->destination_b, result_b);
}

}  // namespace armigo::shader::graphics::detail
