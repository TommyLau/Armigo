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

[[nodiscard]] Result<Operand> shift_right(EmitterContext& context, Operand value, Operand count, bool arithmetic) {
    return context.binary(Opcode::shift_right, value, count, arithmetic ? word32_signed : bits32, ValueType::word32);
}
[[nodiscard]] Result<Operand> shift_right_constant(EmitterContext& context, Operand value,
                                                   std::uint32_t count, bool arithmetic) {
    return shift_right(context, value, word_count(count), arithmetic);
}
} // namespace

// SHF: funnel shift over a low/high word pair. MaxShift selects the shift
// width and signedness of the long-word form; the wrap flag masks the count,
// otherwise counts past the width clamp the result to zero.
Status emit_shf(EmitterContext& context) {
    auto mode = field(context.instruction(), DecodedField::shift_mode);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    auto wrap = field(context.instruction(), DecodedField::shift_wrap);
    if (const auto* error = std::get_if<Diagnostic>(&wrap)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto count = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&count)) return *error;
    auto upper = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&upper)) return *error;

    const bool long_shift = (value(mode) & 2) != 0;        // U64 or S64.
    const bool signed_shift = (value(mode) & 3) == 3;      // S64.
    const bool right_shift = (value(mode) & 4) != 0;       // Direction: 0 = left.
    const std::uint32_t max_shift = long_shift ? 64 : 32;
    Operand width = value(count);
    if (value(wrap)) {
        auto masked = context.binary(Opcode::bit_and, value(count), word_count(max_shift - 1), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&masked)) return *error;
        width = value(masked);
    }

    auto funnel = context.ternary(right_shift ? Opcode::funnel_shift_right : Opcode::funnel_shift_left,
                                  value(left), value(upper), width, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&funnel)) return *error;
    Operand result = value(funnel);
    if (!right_shift) {
        if (long_shift) {
            // b >= 32 selects a << (b - 32).
            auto excess = subtract(context, width, word_count(32));
            if (const auto* error = std::get_if<Diagnostic>(&excess)) return *error;
            auto spill = context.binary(Opcode::shift_left, value(left), value(excess), bits32, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&spill)) return *error;
            auto beyond = compare(context, width, word_count(32), Compare::greater_equal, false);
            if (const auto* error = std::get_if<Diagnostic>(&beyond)) return *error;
            auto chosen = context.select(value(beyond), value(spill), result, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
            result = value(chosen);
        }
    } else {
        if (long_shift) {
            // b >= 32 selects c >> (b - 32), sign-filling for S64.
            auto excess = subtract(context, width, word_count(32));
            if (const auto* error = std::get_if<Diagnostic>(&excess)) return *error;
            auto spill = shift_right(context, value(upper), value(excess), signed_shift);
            if (const auto* error = std::get_if<Diagnostic>(&spill)) return *error;
            auto beyond = compare(context, width, word_count(32), Compare::greater_equal, false);
            if (const auto* error = std::get_if<Diagnostic>(&beyond)) return *error;
            auto chosen = context.select(value(beyond), value(spill), result, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
            result = value(chosen);
        }
    }

    if (!value(wrap)) {
        // Unwrapped counts past the width clamp the result to zero.
        auto within = compare(context, width, word_count(max_shift), Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&within)) return *error;
        auto chosen = context.select(value(within), result, Operand{word_zero}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
        result = value(chosen);
    }

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (auto status = context.write(value(destination), result); failed(status)) return status;
    // The extended (long word) flag is unimplemented here.
    return set_zn_flags(context, result, value(write_cc), false);
}

// SHL: left shift; the wrap flag masks the count, otherwise counts of 32 or
// more clamp to zero. The extended bit and condition-code write are
// unimplemented, so Armigo rejects them instead
// of emitting an unflagged result.
Status emit_shl(EmitterContext& context) {
    auto wrap = field(context.instruction(), DecodedField::shift_wrap);
    if (const auto* error = std::get_if<Diagnostic>(&wrap)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    if (value(write_cc) || value(extended)) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "SHL condition codes are not implemented");
    }
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto count = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&count)) return *error;

    Operand width = value(count);
    if (value(wrap)) {
        auto masked = context.binary(Opcode::bit_and, value(count), word_count(0x1f), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&masked)) return *error;
        width = value(masked);
    }
    auto result = context.binary(Opcode::shift_left, value(left), width, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    if (!value(wrap)) {
        auto within = compare(context, width, word_count(32), Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&within)) return *error;
        auto chosen = context.select(value(within), value(result), Operand{word_zero}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
        result = chosen;
    }

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// SHR: logical or arithmetic right shift with optional source bit-reversal.
// Unwrapped counts of 32 or more produce zero, or the sign fill when signed.
// The condition-code write is unimplemented, so Armigo rejects it instead of
// dropping the effect.
Status emit_shr(EmitterContext& context) {
    auto wrap = field(context.instruction(), DecodedField::shift_wrap);
    if (const auto* error = std::get_if<Diagnostic>(&wrap)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    if (value(write_cc)) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "SHR condition codes are not implemented");
    }
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    const bool reverse = field_flag(context.instruction(), DecodedField::bitfield_mode);
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto count = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&count)) return *error;

    Operand source_word = value(left);
    if (reverse) {
        auto reversed = context.unary(Opcode::bit_reverse, source_word, bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&reversed)) return *error;
        source_word = value(reversed);
    }
    Operand width = value(count);
    if (value(wrap)) {
        auto masked = context.binary(Opcode::bit_and, value(count), word_count(0x1f), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&masked)) return *error;
        width = value(masked);
    }
    auto result = shift_right(context, source_word, width, is_signed);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    if (!value(wrap)) {
        Operand clamped{word_zero};
        if (is_signed) {
            auto filled = shift_right_constant(context, source_word, 31, true);
            if (const auto* error = std::get_if<Diagnostic>(&filled)) return *error;
            clamped = value(filled);
        }
        auto within = compare(context, width, word_count(32), Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&within)) return *error;
        auto chosen = context.select(value(within), value(result), clamped, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
        result = chosen;
    }

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// BFE: signed or unsigned bit-field extraction; the position and size come
// from the low two bytes of source B, with optional source bit-reversal.
Status emit_bfe(EmitterContext& context) {
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    const bool reverse = field_flag(context.instruction(), DecodedField::bitfield_mode);
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;

    Operand source_word = value(left);
    if (reverse) {
        auto reversed = context.unary(Opcode::bit_reverse, source_word, bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&reversed)) return *error;
        source_word = value(reversed);
    }
    auto position = context.binary(Opcode::bit_and, value(right), word_count(0xff), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&position)) return *error;
    auto size = context.ternary(Opcode::bit_extract, value(right), word_count(8), word_count(8),
                                bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&size)) return *error;
    auto result = context.ternary(Opcode::bit_extract, source_word, value(position), value(size),
                                  is_signed ? word32_signed : bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// BFI: insert the low bits of source A into source C; position and size come
// from the low two bytes of source B.
Status emit_bfi(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto base = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&base)) return *error;

    auto position = context.binary(Opcode::bit_and, value(right), word_count(0xff), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&position)) return *error;
    auto size = context.ternary(Opcode::bit_extract, value(right), word_count(8), word_count(8),
                                bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&size)) return *error;
    const ValueType word = ValueType::word32;
    const Operand insert_inputs[] = {value(base), value(left), value(position), value(size)};
    auto inserted = context.emit(Opcode::bit_insert, bits32, insert_inputs, {&word, 1}, {}, Effects::none);
    if (const auto* error = std::get_if<Diagnostic>(&inserted)) return *error;

    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(inserted).front());
}

// FLO: find the most significant set bit, or the least significant one after
// a bit reversal; both support source inversion.
Status emit_flo(EmitterContext& context) {
    const bool is_signed = field_flag(context.instruction(), DecodedField::signedness);
    const bool from_lsb = field_flag(context.instruction(), DecodedField::bitfield_mode);
    auto source = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&source)) return *error;

    auto inverted = invert_word(context, value(source), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;

    if (from_lsb) {
        auto reversed = context.unary(Opcode::bit_reverse, value(inverted), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&reversed)) return *error;
        auto result = context.unary(Opcode::find_lsb, value(reversed), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
        auto destination = destination_register(context);
        if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
        return context.write(value(destination), value(result));
    }
    auto result = context.unary(Opcode::find_msb, value(inverted), is_signed ? word32_signed : bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// POPC: population count with optional source inversion.
Status emit_popc(EmitterContext& context) {
    auto source = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&source)) return *error;
    auto inverted = invert_word(context, value(source), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto result = context.unary(Opcode::bit_count, value(inverted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}
} // namespace armigo::shader::pipeline
