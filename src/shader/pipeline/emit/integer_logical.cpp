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

// Shared result path: optional predicate write, destination copy, Z/N flags.
[[nodiscard]] Status finish_word_result(EmitterContext& context, Operand result, std::uint64_t predicate_op,
                                        bool write_cc, bool extended) {
    if (auto status = emit_lop_pred_write(context, result, predicate_op); failed(status)) return status;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (auto status = context.write(value(destination), result); failed(status)) return status;
    return set_zn_flags(context, result, write_cc, extended);
}
} // namespace

// LOP: word logical operation with per-source inversion and an optional
// zero-test predicate write.
Status emit_lop(EmitterContext& context) {
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    auto predicate_op = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&predicate_op)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;

    auto inverted_left = invert_word(context, value(left), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_left)) return *error;
    auto inverted_right = invert_word(context, value(right), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_right)) return *error;

    Operand result;
    switch (value(logic) & 3) { // LogicOp: and=0, or=1, xor=2, pass_b=3.
    case 0: {
        auto combined = context.binary(Opcode::bit_and, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    case 1: {
        auto combined = context.binary(Opcode::bit_or, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    case 2: {
        auto combined = context.binary(Opcode::bit_xor, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    default: result = value(inverted_right); break;
    }

    return finish_word_result(context, result, value(predicate_op), value(write_cc), value(extended));
}

// LOP3: arbitrary three-input truth table over inverted-capable inputs.
Status emit_lop3(EmitterContext& context) {
    auto table = field(context.instruction(), DecodedField::permutation_mode);
    if (const auto* error = std::get_if<Diagnostic>(&table)) return *error;
    if (value(table) > 0xff) {
        return context.error(DiagnosticCode::unsupported_modifier, "LOP3 truth table is out of range");
    }
    auto predicate_op = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&predicate_op)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto third = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&third)) return *error;

    auto result = lop3_expression(context, value(left), value(right), value(third), static_cast<std::uint8_t>(value(table)));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    return finish_word_result(context, value(result), value(predicate_op), value(write_cc), value(extended));
}

// LOP32I: LOP against a 32-bit immediate; the predicate destination is fixed
// to PT, so only the word result and flags are produced.
Status emit_lop32i(EmitterContext& context) {
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;

    auto inverted_left = invert_word(context, value(left), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_left)) return *error;
    auto inverted_right = invert_word(context, value(right), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_right)) return *error;

    Operand result;
    switch (value(logic) & 3) {
    case 0: {
        auto combined = context.binary(Opcode::bit_and, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    case 1: {
        auto combined = context.binary(Opcode::bit_or, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    case 2: {
        auto combined = context.binary(Opcode::bit_xor, value(inverted_left), value(inverted_right), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        result = value(combined);
        break;
    }
    default: result = value(inverted_right); break;
    }

    // LOP32I fixes the predicate destination to PT; no predicate is written.
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (auto status = context.write(value(destination), result); failed(status)) return status;
    return set_zn_flags(context, result, value(write_cc), value(extended));
}
} // namespace armigo::shader::pipeline
