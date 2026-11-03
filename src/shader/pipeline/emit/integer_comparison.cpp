// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#include "pipeline/emit/integer_emit.h"

namespace armigo::shader::pipeline {
namespace {
using namespace integer_emit;

// IComp selectors preserved from the guest encoding: f=0, lt=1, eq=2, le=3,
// gt=4, ne=5, ge=6, t=7.
[[nodiscard]] Result<Operand> integer_comparison(EmitterContext& context, std::uint64_t selector,
                                                 Operand left, Operand right, bool is_signed, bool extended) {
    if (!extended) {
        switch (selector & 7) {
        case 0: return Operand{predicate_false};
        case 7: return Operand{predicate_true};
        default: break;
        }
        Compare relation{};
        switch (selector & 7) {
        case 1: relation = Compare::less; break;
        case 2: relation = Compare::equal; break;
        case 3: relation = Compare::less_equal; break;
        case 4: relation = Compare::greater; break;
        case 5: relation = Compare::not_equal; break;
        default: relation = Compare::greater_equal; break;
        }
        return compare(context, left, right, relation, is_signed);
    }

    // Long-word comparisons read the flags left by the preceding extended
    // operation. A partial subtract would be discarded here, so Armigo skips
    // the dead operations.
    switch (selector & 7) {
    case 0: return Operand{predicate_false};
    case 7: return Operand{predicate_true};
    case 2: { // Eq: xh == yh && xl == yl
        auto equal = compare(context, left, right, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto zero = read_cc(context, ConditionCode::zero);
        if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
        return logical(context, Opcode::logical_and, value(equal), value(zero));
    }
    case 1: { // Lt: xh < yh || (xh == yh && xl < yl)
        auto below = compare(context, left, right, Compare::less, is_signed);
        if (const auto* error = std::get_if<Diagnostic>(&below)) return *error;
        auto equal = compare(context, left, right, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto no_carry = context.unary(Opcode::logical_not, value(carry), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&no_carry)) return *error;
        auto previous = logical(context, Opcode::logical_and, value(equal), value(no_carry));
        if (const auto* error = std::get_if<Diagnostic>(&previous)) return *error;
        return logical(context, Opcode::logical_or, value(below), value(previous));
    }
    case 3: { // Le: xh < yh || (xh == yh && xl <= yl)
        auto below = compare(context, left, right, Compare::less, is_signed);
        if (const auto* error = std::get_if<Diagnostic>(&below)) return *error;
        auto equal = compare(context, left, right, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto zero = read_cc(context, ConditionCode::zero);
        if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto no_carry = context.unary(Opcode::logical_not, value(carry), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&no_carry)) return *error;
        auto within = logical(context, Opcode::logical_or, value(zero), value(no_carry));
        if (const auto* error = std::get_if<Diagnostic>(&within)) return *error;
        auto previous = logical(context, Opcode::logical_and, value(equal), value(within));
        if (const auto* error = std::get_if<Diagnostic>(&previous)) return *error;
        return logical(context, Opcode::logical_or, value(below), value(previous));
    }
    case 4: { // Gt: xh > yh || (xh == yh && xl > yl)
        auto above = compare(context, left, right, Compare::greater, is_signed);
        if (const auto* error = std::get_if<Diagnostic>(&above)) return *error;
        auto equal = compare(context, left, right, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto zero = read_cc(context, ConditionCode::zero);
        if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto not_zero = context.unary(Opcode::logical_not, value(zero), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&not_zero)) return *error;
        auto previous = logical(context, Opcode::logical_and, value(not_zero), value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&previous)) return *error;
        auto both = logical(context, Opcode::logical_and, value(equal), value(previous));
        if (const auto* error = std::get_if<Diagnostic>(&both)) return *error;
        return logical(context, Opcode::logical_or, value(above), value(both));
    }
    case 6: { // Ge: xh > yh || (xh == yh && xl >= yl)
        auto above = compare(context, left, right, Compare::greater, is_signed);
        if (const auto* error = std::get_if<Diagnostic>(&above)) return *error;
        auto equal = compare(context, left, right, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto previous = logical(context, Opcode::logical_and, value(equal), value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&previous)) return *error;
        return logical(context, Opcode::logical_or, value(above), value(previous));
    }
    default: { // Ne: xh != yh || xl != yl
        auto different = compare(context, left, right, Compare::not_equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&different)) return *error;
        auto zero = read_cc(context, ConditionCode::zero);
        if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
        auto not_zero = context.unary(Opcode::logical_not, value(zero), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&not_zero)) return *error;
        return logical(context, Opcode::logical_or, value(different), value(not_zero));
    }
    }
}

} // namespace

// ICMP: select between two words on the signed/unsigned comparison of source
// C against zero.
Status emit_icmp(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto is_signed = field_flag(context.instruction(), DecodedField::signedness);
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    auto right = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;

    auto comparison = integer_comparison(context, value(selector), value(right), word_zero, is_signed, false);
    if (const auto* error = std::get_if<Diagnostic>(&comparison)) return *error;
    auto result = context.select(value(comparison), value(left), value(middle), ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// ISET: comparison combined with a BoolOp against a guest predicate, written
// either as a word or as the floating 0.0/1.0 encoding.
Status emit_iset(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    auto write_flag = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_flag)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    const bool float_result = field_flag(context.instruction(), DecodedField::destination_type);
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;

    auto comparison = integer_comparison(context, value(selector), value(left), value(right), is_signed, value(extended));
    if (const auto* error = std::get_if<Diagnostic>(&comparison)) return *error;
    auto combined = pred_logical(context, value(logic), value(comparison), value(predicate));
    if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (float_result) {
        auto selected = context.select(value(combined), Operand{float_one}, Operand{word_zero}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&selected)) return *error;
        if (auto status = context.write(value(destination), value(selected)); failed(status)) return status;
        // Set ZF/NF with a floating comparison of the boolean encoding.
        auto zero = context.compare(value(selected), Operand{word_zero},
                                    ComparisonInfo{Compare::equal, false}, float32_compare);
        if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
        auto negative = context.compare(value(selected), Operand{word_zero},
                                        ComparisonInfo{Compare::less, false}, float32_compare);
        if (const auto* error = std::get_if<Diagnostic>(&negative)) return *error;
        if (value(write_flag) != 0) {
            if (auto status = write_cc(context, ConditionCode::zero, value(zero)); failed(status)) return status;
            return write_cc(context, ConditionCode::sign, value(negative));
        }
        return std::monostate{};
    }
    // The boolean result occupies a word container in the destination.
    auto word_result = predicate_mask(context, value(combined));
    if (const auto* error = std::get_if<Diagnostic>(&word_result)) return *error;
    if (auto status = context.write(value(destination), value(word_result)); failed(status)) return status;
    return set_zn_flags(context, value(word_result), value(write_flag), value(extended));
}

// ISETP: the comparison and its inversion, each BoolOp-combined with the
// guest predicate, land in the two destination predicates.
Status emit_isetp(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;

    auto comparison = integer_comparison(context, value(selector), value(left), value(right), is_signed, value(extended));
    if (const auto* error = std::get_if<Diagnostic>(&comparison)) return *error;
    auto inverted = context.unary(Opcode::logical_not, value(comparison), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto first = pred_logical(context, value(logic), value(comparison), value(predicate));
    if (const auto* error = std::get_if<Diagnostic>(&first)) return *error;
    auto second = pred_logical(context, value(logic), value(inverted), value(predicate));
    if (const auto* error = std::get_if<Diagnostic>(&second)) return *error;

    auto first_destination = role_register(context, RegisterRole::predicate_destination);
    if (const auto* error = std::get_if<Diagnostic>(&first_destination)) return *error;
    auto second_destination = role_register(context, RegisterRole::second_predicate_destination);
    if (const auto* error = std::get_if<Diagnostic>(&second_destination)) return *error;
    if (auto status = context.write(value(first_destination), value(first)); failed(status)) return status;
    return context.write(value(second_destination), value(second));
}

// IMNMX: signed or unsigned minimum/maximum chosen by the guest predicate.
Status emit_imnmx(EmitterContext& context) {
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    const NumericMode mode = is_signed ? word32_signed : word32_unsigned;
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;

    auto smallest = context.binary(Opcode::minimum, value(left), value(right), mode, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&smallest)) return *error;
    auto largest = context.binary(Opcode::maximum, value(left), value(right), mode, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&largest)) return *error;
    auto result = context.select(value(predicate), value(smallest), value(largest), ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (auto status = context.write(value(destination), value(result)); failed(status)) return status;
    // The extended (long word) flag is unimplemented here; the plain Z/N flags
    // are still set.
    return set_zn_flags(context, value(result), value(write_cc), false);
}
} // namespace armigo::shader::pipeline
