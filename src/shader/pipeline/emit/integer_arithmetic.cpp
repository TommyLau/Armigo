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

// IMAD maps AvgMode.NegA to source B and AvgMode.NegB to source C.
struct AvgModeSides {
    bool negate_a{}, negate_b{}, plus_one{};
};
[[nodiscard]] Result<AvgModeSides> avg_mode(EmitterContext& context) {
    auto mode = field(context.instruction(), DecodedField::function);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    switch (value(mode) & 3) {
    case 0: return AvgModeSides{};
    case 1: return AvgModeSides{false, true, false};
    case 2: return AvgModeSides{true, false, false};
    default: return context.error(DiagnosticCode::unsupported_modifier, "AvgMode plus-one is not supported");
    }
}
[[nodiscard]] bool saturated(EmitterContext& context) {
    return field_flag(context.instruction(), DecodedField::function, 2);
}
[[nodiscard]] Result<Register> output_destination(EmitterContext& context) {
    return destination_register(context);
}

[[nodiscard]] Status add_pair(EmitterContext& context, Operand left, Operand right) {
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto mode = avg_mode(context);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    if (saturated(context)) {
        return context.error(DiagnosticCode::unsupported_modifier, "Integer add saturation is not implemented");
    }
    auto negate_left = negate_word(context, left, value(mode).negate_a);
    if (const auto* error = std::get_if<Diagnostic>(&negate_left)) return *error;
    auto negate_right = negate_word(context, right, value(mode).negate_b);
    if (const auto* error = std::get_if<Diagnostic>(&negate_right)) return *error;

    auto result = add(context, value(negate_left), value(negate_right));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    if (value(extended)) {
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto carry_word = predicate_word(context, value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&carry_word)) return *error;
        auto carried = add(context, value(result), value(carry_word));
        if (const auto* error = std::get_if<Diagnostic>(&carried)) return *error;
        result = carried;
    }

    if (auto status = set_iadd_flags(context, value(result), value(negate_left), value(negate_right), value(write_cc), value(extended));
        failed(status)) return status;

    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

[[nodiscard]] Status iadd3_core(EmitterContext& context) {
    auto mode = field(context.instruction(), DecodedField::half_mode);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    // IADD3 condition codes and the extended flag are unimplemented. Reject
    // the combination instead of emitting
    // an unflagged result.
    if (value(write_cc) || value(extended)) {
        return context.error(DiagnosticCode::unsupported_modifier, "IADD3 condition codes are not implemented");
    }
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    auto right = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;

    // HalfSelect: b32 keeps the whole word; h0 masks and h1 shifts the high
    // half, zero-extending it.
    auto extend_half = [&](Operand word, unsigned part) -> Result<Operand> {
        switch ((value(mode) >> part) & 3) {
        case 0: return word;
        case 1: return context.binary(Opcode::bit_and, word, word_count(0xffff), bits32, ValueType::word32);
        default: return context.binary(Opcode::shift_right, word, word_count(16), bits32, ValueType::word32);
        }
    };
    auto part_a = extend_half(value(left), 0);
    if (const auto* error = std::get_if<Diagnostic>(&part_a)) return *error;
    auto part_b = extend_half(value(middle), 2);
    if (const auto* error = std::get_if<Diagnostic>(&part_b)) return *error;
    auto part_c = extend_half(value(right), 4);
    if (const auto* error = std::get_if<Diagnostic>(&part_c)) return *error;

    auto negate_a = negate_word(context, value(part_a), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&negate_a)) return *error;
    auto negate_b = negate_word(context, value(part_b), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&negate_b)) return *error;
    auto negate_c = negate_word(context, value(part_c), source_inverted(context, SourceRole::c));
    if (const auto* error = std::get_if<Diagnostic>(&negate_c)) return *error;

    auto result = add(context, value(negate_a), value(negate_b));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    switch ((value(mode) >> 6) & 3) { // Lrs post-shift.
    case 0: break;
    case 1: {
        auto carry = compare(context, value(result), value(negate_a), Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto carry_word = predicate_word(context, value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&carry_word)) return *error;
        auto carry_bit = shift_left(context, value(carry_word), 16);
        if (const auto* error = std::get_if<Diagnostic>(&carry_bit)) return *error;
        auto shifted = context.binary(Opcode::shift_right, value(result), word_count(16), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&shifted)) return *error;
        auto widened = add(context, value(shifted), value(carry_bit));
        if (const auto* error = std::get_if<Diagnostic>(&widened)) return *error;
        result = widened;
        break;
    }
    default: {
        auto shifted = shift_left(context, value(result), 16);
        if (const auto* error = std::get_if<Diagnostic>(&shifted)) return *error;
        result = shifted;
        break;
    }
    }

    auto combined = add(context, value(result), value(negate_c));
    if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;

    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(combined));
}

struct MultiplyOptions {
    AvgModeSides sides{};
    bool check_flags{};
};
[[nodiscard]] Status multiply_accumulate(EmitterContext& context, Operand left, Operand middle,
                                         Operand right, const MultiplyOptions& options) {
    const bool signed_a = field_flag(context.instruction(), DecodedField::signedness, 0);
    const bool signed_b = field_flag(context.instruction(), DecodedField::signedness, 1);
    const bool high = field_flag(context.instruction(), DecodedField::half_mode, 0);
    if (options.check_flags) {
        auto write_cc = field(context.instruction(), DecodedField::write_cc);
        if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
        auto extended = field(context.instruction(), DecodedField::extended);
        if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
        if (saturated(context)) {
            return context.error(DiagnosticCode::unsupported_modifier,
                                 "Integer multiply saturation is not implemented");
        }
        // IMAD condition codes and the extended flag are unimplemented; reject
        // instead of dropping them.
        if (value(write_cc) || value(extended)) {
            return context.error(DiagnosticCode::unsupported_modifier, "IMAD condition codes are not implemented");
        }
    }

    // The AvgMode slots negate source B and C.
    auto negate_middle = negate_word(context, middle, options.sides.negate_a);
    if (const auto* error = std::get_if<Diagnostic>(&negate_middle)) return *error;
    auto negate_right = negate_word(context, right, options.sides.negate_b);
    if (const auto* error = std::get_if<Diagnostic>(&negate_right)) return *error;

    Operand result;
    if (high) {
        if (signed_a && signed_b) {
            auto product = context.binary(Opcode::multiply_high, left, value(negate_middle),
                                          word32_signed, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&product)) return *error;
            result = value(product);
        } else {
            auto product = context.binary(Opcode::multiply_high, left, value(negate_middle),
                                          word32_unsigned, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&product)) return *error;
            result = value(product);
            if (signed_a != signed_b) {
                // Correct an unsigned high product only for a mixed-sign
                // multiply. Two unsigned operands need no sign contribution.
                const Operand factor = signed_a ? value(negate_middle) : left;
                const Operand sign_source = signed_a ? left : value(negate_middle);
                auto sign_fill = context.binary(Opcode::shift_right, sign_source, word_count(31),
                                                word32_signed, ValueType::word32);
                if (const auto* error = std::get_if<Diagnostic>(&sign_fill)) return *error;
                auto sign_product = context.binary(Opcode::multiply, factor, value(sign_fill),
                                                   word32_unsigned, ValueType::word32);
                if (const auto* error = std::get_if<Diagnostic>(&sign_product)) return *error;
                auto corrected = add(context, result, value(sign_product));
                if (const auto* error = std::get_if<Diagnostic>(&corrected)) return *error;
                result = value(corrected);
            }
        }
    } else {
        auto product = context.binary(Opcode::multiply, left, value(negate_middle),
                                      word32_unsigned, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&product)) return *error;
        result = value(product);
    }

    // A literal zero addend is dropped.
    const bool zero_addend = std::holds_alternative<LiteralBits>(right) &&
                             std::get<LiteralBits>(right).bits == 0;
    if (!zero_addend) {
        auto accumulated = add(context, result, value(negate_right));
        if (const auto* error = std::get_if<Diagnostic>(&accumulated)) return *error;
        result = value(accumulated);
    }

    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), result);
}

[[nodiscard]] Status iscadd_core(EmitterContext& context, Operand left, Operand right, bool use_avg_mode) {
    auto scale = field(context.instruction(), DecodedField::shift_mode);
    if (const auto* error = std::get_if<Diagnostic>(&scale)) return *error;
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    AvgModeSides sides;
    if (use_avg_mode) {
        auto mode = avg_mode(context);
        if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
        sides = value(mode);
    }
    auto shifted = shift_left(context, left, static_cast<std::uint32_t>(value(scale) & 31));
    if (const auto* error = std::get_if<Diagnostic>(&shifted)) return *error;
    auto negate_left = negate_word(context, value(shifted), sides.negate_a);
    if (const auto* error = std::get_if<Diagnostic>(&negate_left)) return *error;
    auto negate_right = negate_word(context, right, sides.negate_b);
    if (const auto* error = std::get_if<Diagnostic>(&negate_right)) return *error;

    auto result = add(context, value(negate_left), value(negate_right));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;

    if (auto status = set_iadd_flags(context, value(result), value(negate_left), value(negate_right), value(write_cc), false);
        failed(status)) return status;

    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

[[nodiscard]] Status lea_core(EmitterContext& context, Operand left, Operand right) {
    auto scale = field(context.instruction(), DecodedField::shift_mode);
    if (const auto* error = std::get_if<Diagnostic>(&scale)) return *error;
    auto shifted = shift_left(context, left, static_cast<std::uint32_t>(value(scale) & 31));
    if (const auto* error = std::get_if<Diagnostic>(&shifted)) return *error;
    auto negated = negate_word(context, value(shifted), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&negated)) return *error;
    auto result = add(context, value(negated), right);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

[[nodiscard]] Status lea_hi_core(EmitterContext& context, Operand left, Operand middle, Operand right) {
    auto scale = field(context.instruction(), DecodedField::shift_mode);
    if (const auto* error = std::get_if<Diagnostic>(&scale)) return *error;
    const auto shift = static_cast<std::uint32_t>(value(scale) & 31);
    auto low = shift_left(context, left, shift);
    if (const auto* error = std::get_if<Diagnostic>(&low)) return *error;
    Operand high{word_zero};
    if (shift != 0) {
        auto raised = context.binary(Opcode::shift_right, left, word_count(32 - shift), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&raised)) return *error;
        high = value(raised);
    }
    auto middle_shifted = shift_left(context, right, shift);
    if (const auto* error = std::get_if<Diagnostic>(&middle_shifted)) return *error;
    auto merged = context.binary(Opcode::bit_or, high, value(middle_shifted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&merged)) return *error;
    high = value(merged);

    if (source_inverted(context, SourceRole::a)) {
        if (auto status = negate_long(context, value(low), high); failed(status)) return status;
    }

    auto result = add(context, high, middle);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

[[nodiscard]] Status xmad_core(EmitterContext& context) {
    auto mode = field(context.instruction(), DecodedField::function);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    if ((value(mode) & 7) > 4) {
        return context.error(DiagnosticCode::unsupported_modifier, "XMAD mode is out of range");
    }
    auto write_cc = field(context.instruction(), DecodedField::write_cc);
    if (const auto* error = std::get_if<Diagnostic>(&write_cc)) return *error;
    auto extended = field(context.instruction(), DecodedField::extended);
    if (const auto* error = std::get_if<Diagnostic>(&extended)) return *error;
    auto half = field(context.instruction(), DecodedField::half_mode);
    if (const auto* error = std::get_if<Diagnostic>(&half)) return *error;
    const bool signed_a = field_flag(context.instruction(), DecodedField::signedness, 0);
    const bool signed_b = field_flag(context.instruction(), DecodedField::signedness, 1);
    const bool high_a = (value(half) & 1) != 0;
    const bool high_b = (value(half) & 2) != 0;
    const bool product_shift_left = (value(half) & 4) != 0;
    const bool merge = (value(half) & 8) != 0;

    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    auto right = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    const Operand unmodified_middle = value(middle);

    // Extend16To32: signed high shifts, signed low sign-extends a 16-bit
    // field, unsigned high shifts, and unsigned low masks.
    auto extend16 = [&](Operand value, bool high, bool is_signed) -> Result<Operand> {
        if (is_signed && high) {
            return context.binary(Opcode::shift_right, value, word_count(16), word32_signed, ValueType::word32);
        }
        if (is_signed) {
            return context.ternary(Opcode::bit_extract, value, word_count(0), word_count(16),
                                   word32_signed, ValueType::word32);
        }
        if (high) {
            return context.binary(Opcode::shift_right, value, word_count(16), bits32, ValueType::word32);
        }
        return context.binary(Opcode::bit_and, value, word_count(0xffff), bits32, ValueType::word32);
    };
    auto extended_left = extend16(value(left), high_a, signed_a);
    if (const auto* error = std::get_if<Diagnostic>(&extended_left)) return *error;
    auto extended_middle = extend16(value(middle), high_b, signed_b);
    if (const auto* error = std::get_if<Diagnostic>(&extended_middle)) return *error;

    auto product = context.binary(Opcode::multiply, value(extended_left), value(extended_middle),
                                  word32_unsigned, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&product)) return *error;

    Operand result = value(product);
    if (product_shift_left) {
        auto shifted = shift_left(context, result, 16);
        if (const auto* error = std::get_if<Diagnostic>(&shifted)) return *error;
        result = value(shifted);
    }
    const Operand product_for_flags = result;

    switch (value(mode) & 7) { // Ccop.
    case 0: break; // Cfull.
    case 1: {
        auto lowered = extend16(value(right), false, false);
        if (const auto* error = std::get_if<Diagnostic>(&lowered)) return *error;
        right = value(lowered);
        break;
    }
    case 2: {
        auto raised = extend16(value(right), true, false);
        if (const auto* error = std::get_if<Diagnostic>(&raised)) return *error;
        right = value(raised);
        break;
    }
    case 4: { // Cbcc.
        auto middle_raised = shift_left(context, unmodified_middle, 16);
        if (const auto* error = std::get_if<Diagnostic>(&middle_raised)) return *error;
        auto combined = add(context, value(right), value(middle_raised));
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        right = value(combined);
        break;
    }
    default: { // Csfu.
        auto left_fill = context.binary(Opcode::shift_right, value(extended_left), word_count(31),
                                        bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&left_fill)) return *error;
        auto middle_fill = context.binary(Opcode::shift_right, value(extended_middle), word_count(31),
                                          bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&middle_fill)) return *error;
        auto left_adjust = shift_left(context, value(left_fill), 16);
        if (const auto* error = std::get_if<Diagnostic>(&left_adjust)) return *error;
        auto middle_adjust = shift_left(context, value(middle_fill), 16);
        if (const auto* error = std::get_if<Diagnostic>(&middle_adjust)) return *error;
        auto adjust = add(context, value(left_adjust), value(middle_adjust));
        if (const auto* error = std::get_if<Diagnostic>(&adjust)) return *error;
        auto corrected = subtract(context, value(right), value(adjust));
        if (const auto* error = std::get_if<Diagnostic>(&corrected)) return *error;
        right = value(corrected);
        break;
    }
    }

    if (value(extended)) {
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto carry_word = predicate_word(context, value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&carry_word)) return *error;
        auto carried = add(context, result, value(carry_word));
        if (const auto* error = std::get_if<Diagnostic>(&carried)) return *error;
        result = value(carried);
    } else {
        auto accumulated = add(context, result, value(right));
        if (const auto* error = std::get_if<Diagnostic>(&accumulated)) return *error;
        result = value(accumulated);
    }

    if (auto status = set_iadd_flags(context, result, product_for_flags, value(right), value(write_cc), value(extended)); failed(status)) {
        return status;
    }

    if (merge) {
        auto lowered = context.binary(Opcode::bit_and, result, word_count(0xffff), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&lowered)) return *error;
        auto middle_raised = shift_left(context, unmodified_middle, 16);
        if (const auto* error = std::get_if<Diagnostic>(&middle_raised)) return *error;
        auto merged = context.binary(Opcode::bit_or, value(lowered), value(middle_raised), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&merged)) return *error;
        result = value(merged);
    }

    auto destination = output_destination(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), result);
}
} // namespace

Status emit_iadd(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    return add_pair(context, value(left), value(right));
}
Status emit_iadd3(EmitterContext& context) { return iadd3_core(context); }
Status emit_imad(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    auto right = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto mode = avg_mode(context);
    if (const auto* error = std::get_if<Diagnostic>(&mode)) return *error;
    return multiply_accumulate(context, value(left), value(middle), value(right), {value(mode), true});
}
Status emit_imul(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    // IMUL never negates and always drops the zero addend.
    return multiply_accumulate(context, value(left), value(middle), Operand{word_zero}, {});
}
Status emit_iscadd(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    return iscadd_core(context, value(left), value(right), true);
}
Status emit_iscadd32i(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    // ISCADD32I carries no negation modifier and always uses NoNeg.
    return iscadd_core(context, value(left), value(right), false);
}
Status emit_lea(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    return lea_core(context, value(left), value(right));
}
Status emit_lea_hi(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto middle = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&middle)) return *error;
    auto right = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    return lea_hi_core(context, value(left), value(middle), value(right));
}
Status emit_xmad(EmitterContext& context) { return xmad_core(context); }
} // namespace armigo::shader::pipeline
