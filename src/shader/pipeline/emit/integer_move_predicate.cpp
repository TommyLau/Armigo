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

// Ccc selectors preserved from the guest encoding.
// The reserved camera/capture selectors (24..29) are rejected instead of the
// silently accepting it as true.
[[nodiscard]] Result<Operand> guest_condition(EmitterContext& context, std::uint64_t selector) {
    switch (selector & 31) {
    case 0: return Operand{predicate_false};
    case 15: return Operand{predicate_true};
    case 24: case 25: case 26: case 27: case 28: case 29:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Reserved CSET condition has no defined semantics");
    default: break;
    }
    auto sign = read_cc(context, ConditionCode::sign);
    if (const auto* error = std::get_if<Diagnostic>(&sign)) return *error;
    auto zero = read_cc(context, ConditionCode::zero);
    if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
    auto carry = read_cc(context, ConditionCode::carry);
    if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
    auto overflow = read_cc(context, ConditionCode::overflow);
    if (const auto* error = std::get_if<Diagnostic>(&overflow)) return *error;
    auto not_sign = context.unary(Opcode::logical_not, value(sign), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&not_sign)) return *error;
    auto not_zero = context.unary(Opcode::logical_not, value(zero), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&not_zero)) return *error;
    auto not_carry = context.unary(Opcode::logical_not, value(carry), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&not_carry)) return *error;
    auto not_overflow = context.unary(Opcode::logical_not, value(overflow), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&not_overflow)) return *error;
    switch (selector & 31) {
    case 1: { // Lt.
        auto kept = logical(context, Opcode::logical_and, value(sign), value(not_zero));
        if (const auto* error = std::get_if<Diagnostic>(&kept)) return *error;
        return logical(context, Opcode::logical_xor, value(kept), value(overflow));
    }
    case 2: { // Eq.
        auto first = logical(context, Opcode::logical_and, value(not_sign), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&first)) return *error;
        return first;
    }
    case 3: { // Le.
        auto flagged = logical(context, Opcode::logical_or, value(zero), value(overflow));
        if (const auto* error = std::get_if<Diagnostic>(&flagged)) return *error;
        return logical(context, Opcode::logical_xor, value(sign), value(flagged));
    }
    case 4: { // Gt.
        auto diverged = logical(context, Opcode::logical_xor, value(sign), value(overflow));
        if (const auto* error = std::get_if<Diagnostic>(&diverged)) return *error;
        auto flagged = logical(context, Opcode::logical_or, value(diverged), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&flagged)) return *error;
        return context.unary(Opcode::logical_not, value(flagged), boolean_mode, ValueType::predicate);
    }
    case 5: return not_zero; // Ne.
    case 6: { // Ge.
        auto diverged = logical(context, Opcode::logical_xor, value(sign), value(overflow));
        if (const auto* error = std::get_if<Diagnostic>(&diverged)) return *error;
        return context.unary(Opcode::logical_not, value(diverged), boolean_mode, ValueType::predicate);
    }
    case 7: { // Num (not NaN).
        auto both = logical(context, Opcode::logical_and, value(sign), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&both)) return *error;
        return context.unary(Opcode::logical_not, value(both), boolean_mode, ValueType::predicate);
    }
    case 8: return logical(context, Opcode::logical_and, value(sign), value(zero)); // Nan.
    case 9: return logical(context, Opcode::logical_xor, value(sign), value(overflow)); // Ltu.
    case 10: return zero; // Equ.
    case 11: { // Leu.
        auto diverged = logical(context, Opcode::logical_xor, value(sign), value(overflow));
        if (const auto* error = std::get_if<Diagnostic>(&diverged)) return *error;
        return logical(context, Opcode::logical_or, value(diverged), value(zero));
    }
    case 12: { // Gtu.
        auto flagged = logical(context, Opcode::logical_or, value(overflow), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&flagged)) return *error;
        auto clear = context.unary(Opcode::logical_not, value(sign), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&clear)) return *error;
        return logical(context, Opcode::logical_xor, value(clear), value(flagged));
    }
    case 13: { // Neu.
        auto kept = logical(context, Opcode::logical_or, value(sign), value(not_zero));
        if (const auto* error = std::get_if<Diagnostic>(&kept)) return *error;
        return kept;
    }
    case 14: { // Geu.
        auto clear = context.unary(Opcode::logical_not, value(sign), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&clear)) return *error;
        auto flagged = logical(context, Opcode::logical_or, value(clear), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&flagged)) return *error;
        return logical(context, Opcode::logical_xor, value(flagged), value(overflow));
    }
    case 16: return not_overflow; // Off.
    case 17: return not_carry; // Lo.
    case 18: return not_sign; // Sff.
    case 19: { // Ls.
        auto clear = logical(context, Opcode::logical_or, value(zero), value(not_carry));
        if (const auto* error = std::get_if<Diagnostic>(&clear)) return *error;
        return clear;
    }
    case 20: return logical(context, Opcode::logical_and, value(carry), value(not_zero)); // Hi.
    case 21: return sign; // Sft.
    case 22: return carry; // Hs.
    case 23: return overflow; // Oft.
    case 30: return logical(context, Opcode::logical_or, value(sign), value(zero)); // Rle.
    case 31: { // Rgt.
        auto flagged = logical(context, Opcode::logical_or, value(sign), value(zero));
        if (const auto* error = std::get_if<Diagnostic>(&flagged)) return *error;
        return context.unary(Opcode::logical_not, value(flagged), boolean_mode, ValueType::predicate);
    }
    default: return Operand{predicate_true};
    }
}

// Writes a word result, or the floating 0.0/1.0 boolean encoding when the
// guest BVal modifier asks for it.
[[nodiscard]] Status write_boolean_word(EmitterContext& context, Operand result, bool float_result) {
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (float_result) {
        auto selected = context.select(result, Operand{float_one}, Operand{word_zero}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&selected)) return *error;
        return context.write(value(destination), value(selected));
    }
    // The boolean result occupies a word container in the destination.
    auto word_result = predicate_word(context, result);
    if (const auto* error = std::get_if<Diagnostic>(&word_result)) return *error;
    return context.write(value(destination), value(word_result));
}
} // namespace

// MOV: word copy from a register, literal, or constant-buffer source.
Status emit_mov(EmitterContext& context) {
    auto source = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&source)) return *error;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(source));
}

// SEL: word select on the (optionally inverted) guest predicate.
Status emit_sel(EmitterContext& context) {
    auto left = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&left)) return *error;
    auto right = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&right)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;
    auto result = context.select(value(predicate), value(left), value(right), ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}

// R2P: scatter the tested bits of a word into the predicate file or the four
// condition-code flags.
Status emit_r2p(EmitterContext& context) {
    auto bits = field(context.instruction(), DecodedField::bitfield_mode);
    if (const auto* error = std::get_if<Diagnostic>(&bits)) return *error;
    const auto byte_select = static_cast<unsigned>(value(bits) & 3);
    const bool flags = (value(bits) & 4) != 0;
    auto value_word = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&value_word)) return *error;
    auto mask = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&mask)) return *error;

    const unsigned count = flags ? 4 : 7;
    const unsigned shift = byte_select * 8;
    auto test = [&](Operand word, unsigned bit) -> Result<Operand> {
        auto picked = context.binary(Opcode::bit_and, word, word_count(1u << bit), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&picked)) return *error;
        return compare(context, value(picked), word_zero, Compare::not_equal, false);
    };
    for (unsigned bit = 0; bit < count; ++bit) {
        const Register flag = flags ? cc_register(static_cast<ConditionCode>(bit))
                                    : Register{RegisterFile::predicate, static_cast<std::uint16_t>(bit)};
        auto current = context.read(flag);
        if (const auto* error = std::get_if<Diagnostic>(&current)) return *error;
        auto take_mask = test(value(mask), bit);
        if (const auto* error = std::get_if<Diagnostic>(&take_mask)) return *error;
        auto take_value = test(value(value_word), bit + shift);
        if (const auto* error = std::get_if<Diagnostic>(&take_value)) return *error;
        auto result = context.select(value(take_mask), value(take_value), value(current), ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
        if (auto status = context.write(flag, value(result)); failed(status)) return status;
    }
    return std::monostate{};
}

// P2R: gather the predicate file or condition-code flags into a word under a
// byte mask, preserving unmasked destination bits.
Status emit_p2r(EmitterContext& context) {
    auto bits = field(context.instruction(), DecodedField::bitfield_mode);
    if (const auto* error = std::get_if<Diagnostic>(&bits)) return *error;
    const auto byte_select = static_cast<unsigned>(value(bits) & 3);
    const bool flags = (value(bits) & 4) != 0;
    auto value_word = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&value_word)) return *error;
    auto mask = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&mask)) return *error;
    // The destination is both read and written (read-modify-write).
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    auto current = context.read(value(destination));
    if (const auto* error = std::get_if<Diagnostic>(&current)) return *error;

    const unsigned count = flags ? 4 : 7;
    const unsigned shift = byte_select * 8;
    auto masked = context.binary(Opcode::bit_and, value(mask), word_count(0xff), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&masked)) return *error;
    Operand insert{word_zero};
    for (unsigned bit = 0; bit < count; ++bit) {
        const Register flag = flags ? cc_register(static_cast<ConditionCode>(bit))
                                    : Register{RegisterFile::predicate, static_cast<std::uint16_t>(bit)};
        auto condition = context.read(flag);
        if (const auto* error = std::get_if<Diagnostic>(&condition)) return *error;
        auto selected = context.select(value(condition), word_count(1u << (bit + shift)),
                                       Operand{word_zero}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&selected)) return *error;
        auto combined = context.binary(Opcode::bit_or, insert, value(selected), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        insert = value(combined);
    }
    auto mask_shifted = shift_left(context, value(masked), shift);
    if (const auto* error = std::get_if<Diagnostic>(&mask_shifted)) return *error;
    auto inverted = context.unary(Opcode::bit_not, value(mask_shifted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto kept = context.binary(Opcode::bit_and, value(current), value(inverted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&kept)) return *error;
    auto gathered = context.binary(Opcode::bit_and, insert, value(mask_shifted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&gathered)) return *error;
    auto result = context.binary(Opcode::bit_or, value(kept), value(gathered), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    return context.write(value(destination), value(result));
}

// CSET: condition-code expression combined with a BoolOp against a guest
// predicate, written as a word or as the floating boolean encoding.
Status emit_cset(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    const bool float_result = field_flag(context.instruction(), DecodedField::destination_type);
    auto condition = guest_condition(context, value(selector));
    if (const auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;
    auto combined = pred_logical(context, value(logic), value(condition), value(predicate));
    if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
    return write_boolean_word(context, value(combined), float_result);
}

// CSETP: the condition and its inversion, each BoolOp-combined with the guest
// predicate, land in the two destination predicates.
Status emit_csetp(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::condition);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&logic)) return *error;
    auto condition = guest_condition(context, value(selector));
    if (const auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    auto inverted = context.unary(Opcode::logical_not, value(condition), boolean_mode, ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto predicate = predicate_source(context);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;
    auto first = pred_logical(context, value(logic), value(condition), value(predicate));
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

// PSET: two guest predicates combined by BoolOpAB and then BoolOpC with a
// third, written as a word or as the floating boolean encoding.
Status emit_pset(EmitterContext& context) {
    auto first_logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&first_logic)) return *error;
    const bool float_result = field_flag(context.instruction(), DecodedField::destination_type);
    auto first = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&first)) return *error;
    auto second = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&second)) return *error;
    auto third = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&third)) return *error;
    auto inverted_first = invert_predicate(context, value(second), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_first)) return *error;
    auto inverted_second = invert_predicate(context, value(first), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_second)) return *error;
    auto inverted_third = invert_predicate(context, value(third), source_inverted(context, SourceRole::c));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_third)) return *error;
    auto combined = pred_logical(context, value(first_logic), value(inverted_first), value(inverted_second));
    if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
    auto result = pred_logical(context, (value(first_logic) >> 2) & 3, value(combined), value(inverted_third));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    return write_boolean_word(context, value(result), float_result);
}

// PSETP: the PSET combination plus its inversion in the two destinations.
Status emit_psetp(EmitterContext& context) {
    auto first_logic = field(context.instruction(), DecodedField::boolean_operation);
    if (const auto* error = std::get_if<Diagnostic>(&first_logic)) return *error;
    auto first = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&first)) return *error;
    auto second = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&second)) return *error;
    auto third = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&third)) return *error;
    auto inverted_first = invert_predicate(context, value(second), source_inverted(context, SourceRole::b));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_first)) return *error;
    auto inverted_second = invert_predicate(context, value(first), source_inverted(context, SourceRole::a));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_second)) return *error;
    auto inverted_third = invert_predicate(context, value(third), source_inverted(context, SourceRole::c));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_third)) return *error;
    auto combined = pred_logical(context, value(first_logic), value(inverted_first), value(inverted_second));
    if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
    auto result = pred_logical(context, (value(first_logic) >> 2) & 3, value(combined), value(inverted_third));
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto inverted_combined = context.unary(Opcode::logical_not, value(combined), boolean_mode,
                                            ValueType::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&inverted_combined)) return *error;
    auto inverted_result = pred_logical(context, (value(first_logic) >> 2) & 3,
                                        value(inverted_combined), value(inverted_third));
    if (const auto* error = std::get_if<Diagnostic>(&inverted_result)) return *error;
    auto first_destination = role_register(context, RegisterRole::predicate_destination);
    if (const auto* error = std::get_if<Diagnostic>(&first_destination)) return *error;
    auto second_destination = role_register(context, RegisterRole::second_predicate_destination);
    if (const auto* error = std::get_if<Diagnostic>(&second_destination)) return *error;
    if (auto status = context.write(value(first_destination), value(result)); failed(status)) return status;
    return context.write(value(second_destination), value(inverted_result));
}

// Compute identifiers remain explicit IO reads; unmodelled special registers fail.
Status emit_s2r(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::function);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    const auto sreg_value = value(selector);
    // Lane registers (SR_LANEID and the lane masks) exist in fragment stages
    // too, SR_YDIRECTION is a rasterization-state input for graphics stages,
    // and thread and workgroup identifiers stay compute-only. The viewport
    // W-scale factors (SR_WSCALEFACTOR_XY/Z) belong to graphics stages. They
    // read zero: in the observed vertex stages they only feed the predicates
    // of a VOTE.VTG cull vote, which is not translated.
    const bool lane_register = sreg_value == 0 || (sreg_value >= 56 && sreg_value <= 60);
    const bool w_scale = sreg_value == 30 || sreg_value == 31;
    const auto stage = context.decoded_program().stage;
    if (w_scale && stage == Stage::compute)
        return context.error(DiagnosticCode::unsupported_instruction,
                             "S2R viewport W-scale factors require a graphics stage");
    if (stage != Stage::compute && sreg_value != 18 && !w_scale && !(lane_register && stage == Stage::fragment))
        return context.error(DiagnosticCode::unimplemented,
                             "S2R compute identifiers require a compute stage");
    auto load = [&](IoSemantic semantic, unsigned component) -> Result<Operand> {
        IoInfo io;
        io.semantic = semantic;
        io.type = {ScalarType::u32, static_cast<std::uint8_t>(semantic == IoSemantic::subgroup_lane_id ? 1 : 3)};
        io.component_mask = static_cast<std::uint8_t>(1u << component);
        const ValueType type = ValueType::word32;
        auto result = context.load_io(io, {}, NumericMode{Domain::unsigned_integer, ScalarType::u32},
                                      std::span{&type, 1});
        if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
        return Operand{std::get<std::vector<ValueId>>(result).front()};
    };
    const auto sreg = value(selector);
    const bool known = sreg == 18 || sreg == 30 || sreg == 31 || sreg == 32 || (sreg >= 33 && sreg <= 35) ||
                       (sreg >= 37 && sreg <= 39) || sreg == 0 || (sreg >= 56 && sreg <= 60);
    if (!known) return context.error(DiagnosticCode::unsupported_instruction,
        "S2R special register is not implemented", "sreg " + std::to_string(sreg));
    // Every admitted selector assigns below; the placeholder never survives.
    Result<Operand> result = Operand{word_zero};
    if (sreg == 18) {
        // SR_YDIRECTION carries the float bit pattern of +1.0f, or -1.0f when
        // the guest YControl negates Y. The sign is draw state, so the stage
        // reads it through the y_direction interface input the MSL back end
        // specializes with the armigo_y_negate function constant; the raw
        // word lands in the destination exactly like any S2R result. Compute
        // has no rasterization Y direction.
        if (stage == Stage::compute)
            return context.error(DiagnosticCode::unsupported_instruction,
                "S2R SR_YDIRECTION requires a graphics stage");
        IoInfo io;
        io.semantic = IoSemantic::y_direction;
        io.type = {ScalarType::u32, 1};
        io.component_mask = 1;
        const ValueType type = ValueType::word32;
        auto loaded = context.load_io(io, {}, NumericMode{Domain::unsigned_integer, ScalarType::u32},
                                      std::span{&type, 1});
        if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
        result = Operand{std::get<std::vector<ValueId>>(loaded).front()};
    } else if (sreg == 30 || sreg == 31) {
        result = Operand{word_zero};  // see the W-scale note above
    } else if (sreg >= 33 && sreg <= 35) result = load(IoSemantic::thread_id, unsigned(sreg - 33));
    else if (sreg >= 37 && sreg <= 39) result = load(IoSemantic::workgroup_id, unsigned(sreg - 37));
    else if (sreg == 0 || (sreg >= 56 && sreg <= 60)) {
        auto lane = load(IoSemantic::subgroup_lane_id, 0);
        if (const auto* error = std::get_if<Diagnostic>(&lane)) return *error;
        if (sreg == 0) result = lane;
        else {
            auto bit = context.binary(Opcode::shift_left, Operand{LiteralBits{1, ValueType::word32}},
                                      value(lane), bits32, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&bit)) return *error;
            auto less = context.binary(Opcode::subtract, value(bit), Operand{LiteralBits{1, ValueType::word32}},
                                       word32_unsigned, ValueType::word32);
            if (const auto* error = std::get_if<Diagnostic>(&less)) return *error;
            if (sreg == 56) result = bit;
            else if (sreg == 57) result = less;
            else {
                auto le = context.binary(Opcode::bit_or, value(less), value(bit), bits32, ValueType::word32);
                if (const auto* error = std::get_if<Diagnostic>(&le)) return *error;
                if (sreg == 58) result = le;
                else result = context.unary(Opcode::bit_not, sreg == 59 ? value(le) : value(less), bits32, ValueType::word32);
            }
        }
    } else if (sreg == 32) {
        auto x = load(IoSemantic::thread_id, 0), y = load(IoSemantic::thread_id, 1), z = load(IoSemantic::thread_id, 2);
        if (const auto* error = std::get_if<Diagnostic>(&x)) return *error;
        if (const auto* error = std::get_if<Diagnostic>(&y)) return *error;
        if (const auto* error = std::get_if<Diagnostic>(&z)) return *error;
        auto ys = context.binary(Opcode::shift_left, value(y), Operand{LiteralBits{16, ValueType::word32}}, bits32, ValueType::word32);
        auto zs = context.binary(Opcode::shift_left, value(z), Operand{LiteralBits{26, ValueType::word32}}, bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&ys)) return *error;
        if (const auto* error = std::get_if<Diagnostic>(&zs)) return *error;
        auto yz = context.binary(Opcode::bit_or, value(ys), value(zs), bits32, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&yz)) return *error;
        result = context.binary(Opcode::bit_or, value(x), value(yz), bits32, ValueType::word32);
    }
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    auto destination = role_register(context, RegisterRole::destination);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    return context.write(value(destination), value(result));
}
} // namespace armigo::shader::pipeline
