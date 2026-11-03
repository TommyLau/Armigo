// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic instruction words only; independently authored for the Armigo
// ALU decode lane.

#include "pipeline/decoder.h"

#include <algorithm>
#include <iostream>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

// Mirrors the table-row encoding: index 0 is bit 63.
constexpr std::uint64_t opcode(std::string_view pattern) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        if (pattern[index] == '1') value |= std::uint64_t{1} << (63 - index);
    }
    return value;
}

constexpr std::uint64_t set(std::uint64_t word, unsigned first, unsigned width, std::uint64_t value) {
    const auto clear = ~(((std::uint64_t{1} << width) - 1) << first);
    return (word & clear) | (value << first);
}

const DecodeRule* find_rule(std::uint64_t word) {
    for (const auto& entry : alu_decode_rules()) {
        if ((word & entry.mask) == entry.value) return &entry;
    }
    return nullptr;
}

DecodedInstruction decodes(std::uint64_t word, std::string_view message) {
    const auto* entry = find_rule(word);
    check(entry != nullptr, message);
    if (entry == nullptr) return {};
    auto result = entry->decode(40, RawInstruction{word});
    check(std::holds_alternative<DecodedInstruction>(result), message);
    if (const auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << "  decode error: " << error->message << '\n';
        return {};
    }
    auto decoded = std::get<DecodedInstruction>(std::move(result));
    check(decoded.pc == 40, "Decoded instruction retains its stream PC");
    return decoded;
}

Diagnostic rejected(std::uint64_t word, std::string_view message,
                    DiagnosticCode code = DiagnosticCode::malformed_input) {
    const auto* entry = find_rule(word);
    check(entry != nullptr, message);
    if (entry == nullptr) return {};
    auto result = entry->decode(40, RawInstruction{word});
    check(std::holds_alternative<Diagnostic>(result), message);
    if (const auto* decoded = std::get_if<DecodedInstruction>(&result)) {
        std::cerr << "  unexpectedly decoded as " << instruction_name(decoded->name) << '\n';
        return {};
    }
    auto error = std::get<Diagnostic>(std::move(result));
    check(error.code == code, message);
    check(error.phase == Phase::decode, "Decode diagnostics keep the decode phase");
    check(error.pc == 40, "Decode diagnostics keep the instruction PC");
    return error;
}

const DecodedSource* source(const DecodedInstruction& decoded, SourceRole role) {
    for (const auto& entry : decoded.sources) {
        if (entry.role == role) return &entry;
    }
    return nullptr;
}

const RegisterAccess* access(const DecodedInstruction& decoded, RegisterRole role, std::uint16_t index) {
    for (const auto& entry : decoded.registers) {
        if (entry.role == role && entry.reg.index == index) return &entry;
    }
    return nullptr;
}

bool has_field(const DecodedInstruction& decoded, DecodedField key, std::uint64_t bits) {
    for (const auto& entry : decoded.fields) {
        if (entry.field == key && entry.bits == bits) return true;
    }
    return false;
}

void test_table() {
    const auto rules = alu_decode_rules();
    check(rules.size() > 100, "The ALU table admits its instruction families");
    for (std::size_t i = 0; i < rules.size(); ++i) {
        check(rules[i].decode != nullptr, "Every table row provides a decoder");
        for (std::size_t j = i + 1; j < rules.size(); ++j) {
            check(((rules[i].mask & rules[j].mask) & (rules[i].value ^ rules[j].value)) != 0,
                  "Mask/value rows never overlap");
        }
    }
}

void test_float_arithmetic() {
    // FADD R5, R6, R7 ; ftz, negate B, write CC.
    auto word = opcode("0101110001011");
    word = set(word, 0, 8, 5) | set(word, 8, 8, 6) | set(word, 20, 8, 7) | set(word, 16, 3, 2);
    word |= std::uint64_t{1} << 44 | std::uint64_t{1} << 45 | std::uint64_t{1} << 47;
    auto decoded = decodes(word, "FADD register form decodes");
    check(decoded.name == InstructionName::fadd && decoded.encoding == Encoding::registers, "FADD R name and route");
    check(decoded.predicate.index == 2 && !decoded.predicate.inverted && decoded.predicate.enabled, "FADD R predicate");
    check(access(decoded, RegisterRole::destination, 5) != nullptr &&
          access(decoded, RegisterRole::destination, 5)->access == Access::write, "FADD R destination");
    check(access(decoded, RegisterRole::source_a, 6) != nullptr, "FADD R source A");
    check(access(decoded, RegisterRole::source_b, 7) != nullptr, "FADD R source B");
    const auto* b = source(decoded, SourceRole::b);
    check(b != nullptr && std::holds_alternative<Register>(b->value) && b->modifiers.negate, "FADD R negate modifier");
    check(decoded.numeric && decoded.numeric->domain == Domain::floating && decoded.numeric->scalar == ScalarType::f32 &&
          decoded.numeric->rounding == Rounding::nearest_even && decoded.numeric->flush_inputs &&
          decoded.numeric->flush_output, "FADD R numeric mode keeps rounding and flush");
    check(access(decoded, RegisterRole::condition_code, static_cast<std::uint16_t>(ConditionCode::zero)) != nullptr &&
          access(decoded, RegisterRole::condition_code, static_cast<std::uint16_t>(ConditionCode::sign)) != nullptr,
          "WriteCC stores zero and sign flags");
    check(has_field(decoded, DecodedField::write_cc, 1), "WriteCC field is present");

    // FADD I: compact floating immediate with IEEE bits 31..12 in the encoding.
    word = opcode("0011100x01011");
    word = set(word, 20, 19, 0x12345) | std::uint64_t{1} << 56 | set(word, 39, 2, 3) | set(word, 8, 8, 1);
    decoded = decodes(word, "FADD immediate form decodes");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && std::holds_alternative<LiteralBits>(b->value) &&
          std::get<LiteralBits>(b->value).bits == 0x92345000, "FADD I expands its floating immediate");
    check(decoded.numeric && decoded.numeric->rounding == Rounding::toward_zero, "FADD I rounding selector");
    check(decoded.encoding == Encoding::immediate, "FADD I encoding");

    // FADD C: constant buffer slot and byte-normalized offset.
    word = opcode("0100110001011");
    word = set(word, 34, 5, 2) | set(word, 20, 14, 0x155) | set(word, 8, 8, 9);
    decoded = decodes(word, "FADD constant form decodes");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && std::holds_alternative<ConstantWord>(b->value), "FADD C source is a constant word");
    if (b != nullptr) {
        const auto& constant_word = std::get<ConstantWord>(b->value);
        check(constant_word.slot == 2 && constant_word.byte_offset == 0x554, "FADD C offset normalizes to bytes");
    }
    check(decoded.encoding == Encoding::constant_buffer, "FADD C encoding");

    // FADD32I carries independent AbsA and NegB modifiers.
    decoded = decodes(0x0863f80000070100, "FADD32I modifier word decodes");
    const auto* a = source(decoded, SourceRole::a);
    b = source(decoded, SourceRole::b);
    check(a != nullptr && a->modifiers.absolute, "FADD32I applies AbsA");
    check(b != nullptr && b->modifiers.negate, "FADD32I applies NegB");

    // FMUL R: result scale, flush selector, saturation.
    word = opcode("0101110001101");
    word = set(word, 41, 3, 5) | set(word, 44, 2, 2) | std::uint64_t{1} << 50;
    decoded = decodes(word, "FMUL register form decodes");
    check(decoded.name == InstructionName::fmul, "FMUL R name");
    check(has_field(decoded, DecodedField::result_scale, 5), "FMUL R preserves the result scale selector");
    check(has_field(decoded, DecodedField::function, 2), "FMUL R preserves the FMZ selector");
    check(decoded.numeric && decoded.numeric->flush_inputs && decoded.numeric->saturate, "FMUL R flush and saturate");

    // FFMA R: three sources, rounding at 51..52, FMZ flush, negate C.
    word = opcode("010110011");
    word = set(word, 8, 8, 1) | set(word, 20, 8, 2) | set(word, 39, 8, 3);
    word |= std::uint64_t{1} << 49 | set(word, 51, 2, 3) | set(word, 53, 2, 2);
    decoded = decodes(word, "FFMA register form decodes");
    check(decoded.name == InstructionName::ffma, "FFMA R name");
    check(access(decoded, RegisterRole::source_c, 3) != nullptr, "FFMA R third source");
    const auto* c = source(decoded, SourceRole::c);
    check(c != nullptr && c->modifiers.negate, "FFMA R negates C");
    check(decoded.numeric && decoded.numeric->rounding == Rounding::toward_zero && decoded.numeric->flush_inputs,
          "FFMA R rounding and FMZ flush");
    check(has_field(decoded, DecodedField::function, 2), "FFMA R preserves the FMZ selector");
    rejected(word | (set(word, 53, 2, 3) ^ set(word, 53, 2, 2)), "Reserved FFMA flush selector is malformed");

    // FFMA Rc multiplies by the register at 39..46 and adds the constant.
    word = set(opcode("010100011"), 34, 5, 1) | set(std::uint64_t{0}, 20, 14, 4) |
           set(std::uint64_t{0}, 39, 8, 9);
    decoded = decodes(word, "FFMA register-constant form decodes");
    check(decoded.name == InstructionName::ffma && decoded.encoding == Encoding::constant_buffer, "FFMA Rc route");
    b = source(decoded, SourceRole::b);
    c = source(decoded, SourceRole::c);
    check(b != nullptr && std::holds_alternative<Register>(b->value) &&
              std::get<Register>(b->value).index == 9,
          "FFMA Rc register is multiplier B");
    check(c != nullptr && std::holds_alternative<ConstantWord>(c->value),
          "FFMA Rc constant is addend C");

    // FFMA32I reads its addend from the old destination register.
    word = opcode("000011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
           set(std::uint64_t{0}, 20, 32, 0x3f800000);
    decoded = decodes(word, "FFMA32I destination addend decodes");
    c = source(decoded, SourceRole::c);
    check(c != nullptr && std::holds_alternative<Register>(c->value) &&
              std::get<Register>(c->value).index == 5,
          "FFMA32I addend reads the old destination");
}

void test_comparison_and_conversion() {
    // FSETP R3/P2, R1, R2, lt; FTZ at bit 47 is not a CC write.
    auto word = opcode("010110111011");
    word = set(word, 3, 3, 3) | set(word, 0, 3, 2) | set(word, 8, 8, 1) | set(word, 20, 8, 2);
    word |= std::uint64_t{1} << 47 | set(word, 48, 4, 1);
    auto decoded = decodes(word, "FSETP register form decodes");
    check(decoded.name == InstructionName::fsetp, "FSETP name");
    check(access(decoded, RegisterRole::predicate_destination, 3) != nullptr &&
          access(decoded, RegisterRole::second_predicate_destination, 2) != nullptr,
          "FSETP writes the paired predicates");
    check(has_field(decoded, DecodedField::condition, 1), "FSETP comparison selector");
    check(decoded.numeric && decoded.numeric->flush_inputs, "FSETP keeps its FTZ flush");
    check(!has_field(decoded, DecodedField::write_cc, 1), "FSETP bit 47 is not a CC write");
    const auto* a = source(decoded, SourceRole::a);
    const auto* b = source(decoded, SourceRole::b);
    check(a != nullptr && !a->modifiers.absolute && b != nullptr && !b->modifiers.negate,
          "FSETP ignores FSET modifier positions");
    decoded = decodes(word | (std::uint64_t{1} << 6) | (std::uint64_t{1} << 7),
                      "FSETP modifier bits decode");
    a = source(decoded, SourceRole::a);
    b = source(decoded, SourceRole::b);
    check(a != nullptr && a->modifiers.absolute && b != nullptr && b->modifiers.negate,
          "FSETP applies AbsA and NegB from bits 7 and 6");

    // ISETP: signed comparison and predicate source.
    word = opcode("010110110110");
    word = set(word, 48, 1, 1) | set(word, 49, 3, 2) | set(word, 39, 3, 5) | std::uint64_t{1} << 42;
    decoded = decodes(word, "ISETP register form decodes");
    check(decoded.name == InstructionName::isetp, "ISETP name");
    check(has_field(decoded, DecodedField::signedness, 1), "ISETP signedness selector");
    const auto* predicate = source(decoded, SourceRole::predicate);
    check(predicate != nullptr && predicate->modifiers.negate &&
          std::get<Register>(predicate->value).index == 5, "ISETP predicate source with inversion");

    // F2F: format selectors and pass-through rounding. The 3-bit integral
    // selector keeps its raw value in integral_round.
    word = opcode("0101110010101");
    word = set(word, 8, 2, 2) | set(word, 10, 2, 1) | std::uint64_t{1} << 42;
    decoded = decodes(word, "F2F register form decodes");
    check(decoded.name == InstructionName::f2f, "F2F name");
    check(has_field(decoded, DecodedField::source_type, 1) && has_field(decoded, DecodedField::destination_type, 2),
          "F2F keeps raw format selectors");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && b->modifiers.subword == Subword::half_low,
          "F2F selects the low half when Sh is clear");
    check(has_field(decoded, DecodedField::integral_round, 4), "F2F integral rounding selector");
    check(decoded.numeric && decoded.numeric->scalar == ScalarType::f32 &&
          decoded.numeric->rounding == Rounding::nearest_even, "F2F numeric mode follows the destination format");
    rejected(set(opcode("0101110010101"), 8, 2, 0), "Reserved F2F destination format is malformed");

    word = opcode("0101110010101");
    word = set(word, 8, 2, 2) | set(word, 10, 2, 1) | (std::uint64_t{1} << 41);
    decoded = decodes(word, "F2F high-half form decodes");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && b->modifiers.subword == Subword::half_high,
          "F2F preserves the high-half selector");

    word = opcode("0101110010101");
    word = set(word, 8, 2, 1) | set(word, 10, 2, 2) | set(word, 39, 2, 3);
    decoded = decodes(word, "F2F narrowing round-toward-zero form decodes");
    check(decoded.numeric && decoded.numeric->rounding == Rounding::toward_zero,
          "Cross-format F2F preserves conversion rounding");

    word = opcode("0101110010110");
    word = set(word, 8, 2, 2) | set(word, 10, 2, 1) | (std::uint64_t{1} << 41);
    decoded = decodes(word, "F2I high-half form decodes");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && b->modifiers.subword == Subword::half_high,
          "F2I preserves the high-half selector");

    // I2F source signedness is independent from the floating destination.
    word = opcode("0101110010111");
    word = set(word, 13, 1, 1) | set(word, 10, 2, 0) | set(word, 8, 2, 2);
    decoded = decodes(word, "I2F signed byte form decodes");
    check(has_field(decoded, DecodedField::source_type, 4) &&
              has_field(decoded, DecodedField::destination_type, 2),
          "I2F source signedness comes from bit 13");

    // I2I: byte lane select, integer formats and the saturation flag.
    word = opcode("0101110011100");
    word = set(word, 41, 2, 2) | set(word, 13, 1, 1) |
           set(word, 10, 2, 2) | set(word, 8, 2, 1) |
           std::uint64_t{1} << 50;
    decoded = decodes(word, "I2I register form decodes");
    check(decoded.name == InstructionName::i2i, "I2I name");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && b->modifiers.subword == Subword::byte2, "I2I byte lane select");
    check(has_field(decoded, DecodedField::source_type, 0b110) &&
          has_field(decoded, DecodedField::destination_type, 0b001),
          "I2I keeps independent source and destination signedness");
    check(has_field(decoded, DecodedField::integer_saturation, 1), "I2I saturation selector");
    check(!decoded.numeric, "I2I carries no floating interpretation");

    // ISET's Boolean-value format is independent from predicate register P2.
    word = opcode("010110110101");
    word = set(word, 39, 3, 2);
    decoded = decodes(word, "ISET integer Boolean result decodes");
    check(has_field(decoded, DecodedField::destination_type, 0),
          "ISET BVal comes from bit 44");
}

void test_integer_and_logical() {
    // IADD3 R: per-source halves, partial shift, carry chain, CC write. The
    // half selects and the partial-sum shift pack into half_mode.
    auto word = opcode("010111001100");
    word = set(word, 8, 8, 4) | set(word, 20, 8, 5) | set(word, 39, 8, 6);
    word = set(word, 35, 2, 1) | set(word, 33, 2, 2) | set(word, 37, 2, 2);
    word |= std::uint64_t{1} << 51 | std::uint64_t{1} << 48 | std::uint64_t{1} << 47;
    auto decoded = decodes(word, "IADD3 register form decodes");
    check(decoded.name == InstructionName::iadd3, "IADD3 name");
    check(has_field(decoded, DecodedField::half_mode, 1 | (2 << 2) | (2 << 6)),
          "IADD3 half selects and partial shift");
    const auto* a = source(decoded, SourceRole::a);
    check(a != nullptr && a->modifiers.negate, "IADD3 negates A");
    check(has_field(decoded, DecodedField::extended, 1), "IADD3 extended selector");
    check(access(decoded, RegisterRole::condition_code, static_cast<std::uint16_t>(ConditionCode::carry)) != nullptr,
          "IADD3 extended forms a carry chain");
    rejected(set(word, 35, 2, 3), "Reserved IADD3 half selector is malformed");

    // IADD publishes AvgMode and the saturation flag through `function`.
    word = opcode("0101110000010");
    word |= std::uint64_t{1} << 50 | set(word, 48, 2, 2);
    decoded = decodes(word, "IADD register form decodes");
    check(decoded.name == InstructionName::iadd, "IADD name");
    check(has_field(decoded, DecodedField::function, 2 | (1 << 2)), "IADD AvgMode and saturation selector");

    // ISCADD C keeps its shift amount.
    word = opcode("0100110000011");
    word = set(word, 39, 5, 9) | set(word, 34, 5, 3);
    decoded = decodes(word, "ISCADD constant form decodes");
    check(decoded.name == InstructionName::iscadd && has_field(decoded, DecodedField::shift_mode, 9),
          "ISCADD shift amount");
    const auto* iscadd_b = source(decoded, SourceRole::b);
    check(iscadd_b != nullptr && std::holds_alternative<ConstantWord>(iscadd_b->value), "ISCADD C constant source");

    // LOP3 R: truth table byte and predicate result.
    word = opcode("0101101111100");
    word = set(word, 28, 8, 0xCA) | set(word, 36, 2, 2) | std::uint64_t{1} << 38 | set(word, 48, 3, 5);
    decoded = decodes(word, "LOP3 register form decodes");
    check(decoded.name == InstructionName::lop3, "LOP3 name");
    check(has_field(decoded, DecodedField::permutation_mode, 0xCA), "LOP3 truth table byte");
    check(has_field(decoded, DecodedField::condition, 2), "LOP3 result predicate test");
    check(access(decoded, RegisterRole::predicate_destination, 5) != nullptr, "LOP3 predicate destination");
    check(has_field(decoded, DecodedField::extended, 1), "LOP3 extended selector");
    check(source(decoded, SourceRole::c) != nullptr, "LOP3 publishes its third source");

    // SHF L I: MaxShift and direction pack into shift_mode; M rides shift_wrap.
    word = opcode("0011011x11111");
    word = set(word, 20, 6, 7) | set(word, 37, 2, 2) | std::uint64_t{1} << 50;
    decoded = decodes(word, "SHF left immediate form decodes");
    check(decoded.name == InstructionName::shf, "SHF name");
    const auto* shf_b = source(decoded, SourceRole::b);
    check(shf_b != nullptr && std::get<LiteralBits>(shf_b->value).bits == 7, "SHF immediate shift amount");
    check(has_field(decoded, DecodedField::shift_mode, 2), "SHF MaxShift with left direction");
    check(has_field(decoded, DecodedField::shift_wrap, 1), "SHF wrap flag");
    word = opcode("0101110011111");
    word = set(word, 37, 2, 2);
    decoded = decodes(word, "SHF right register form decodes");
    check(has_field(decoded, DecodedField::shift_mode, 2 | (1 << 2)), "SHF right direction rides bit 2");

    // XMAD R: Ccop in function, HILO/PSL/Mrg in half_mode, signedness bits.
    word = opcode("0101101100");
    word = set(word, 53, 1, 1) | set(word, 48, 1, 1) | set(word, 49, 1, 1) | set(word, 50, 3, 2) |
           set(word, 36, 1, 1) | set(word, 37, 1, 1) | set(word, 35, 1, 1);
    decoded = decodes(word, "XMAD register form decodes");
    check(decoded.name == InstructionName::xmad, "XMAD name");
    check(has_field(decoded, DecodedField::function, 2), "XMAD Ccop selector");
    check(has_field(decoded, DecodedField::half_mode, 1 | (1 << 1) | (1 << 2) | (1 << 3)),
          "XMAD HiloA/HiloB/PSL/Mrg selectors");
    check(has_field(decoded, DecodedField::signedness, 1 | (1 << 1)), "XMAD per-source signedness");

    // Immediate XMAD takes its extended selector from bit 38.
    word = opcode("0011011x00") | (std::uint64_t{1} << 38);
    decoded = decodes(word, "XMAD immediate extended form decodes");
    check(has_field(decoded, DecodedField::extended, 1), "XMAD immediate X selector uses bit 38");

    // XMAD Rc multiplies by the register at 39..46 and accumulates the constant.
    word = opcode("010100010") | set(std::uint64_t{0}, 39, 8, 9) |
           set(std::uint64_t{0}, 34, 5, 2) | set(std::uint64_t{0}, 20, 14, 4);
    decoded = decodes(word, "XMAD register-constant form decodes");
    const auto* xmad_b = source(decoded, SourceRole::b);
    const auto* xmad_c = source(decoded, SourceRole::c);
    check(xmad_b != nullptr && std::holds_alternative<Register>(xmad_b->value) &&
              std::get<Register>(xmad_b->value).index == 9,
          "XMAD Rc register is multiplier B");
    check(xmad_c != nullptr && std::holds_alternative<ConstantWord>(xmad_c->value),
          "XMAD Rc constant is addend C");

    // IMAD Rc uses the same logical operand order, while IMAD32I accumulates Rd.
    word = opcode("010100100") | set(std::uint64_t{0}, 0, 8, 5) |
           set(std::uint64_t{0}, 39, 8, 9) | set(std::uint64_t{0}, 34, 5, 2) |
           set(std::uint64_t{0}, 20, 14, 4);
    decoded = decodes(word, "IMAD register-constant form decodes");
    const auto* imad_b = source(decoded, SourceRole::b);
    const auto* imad_c = source(decoded, SourceRole::c);
    check(imad_b != nullptr && std::holds_alternative<Register>(imad_b->value) &&
              std::get<Register>(imad_b->value).index == 9,
          "IMAD Rc register is multiplier B");
    check(imad_c != nullptr && std::holds_alternative<ConstantWord>(imad_c->value),
          "IMAD Rc constant is addend C");
    word = opcode("000100") | set(std::uint64_t{0}, 0, 8, 5) |
           set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 32, 2);
    decoded = decodes(word, "IMAD32I destination addend decodes");
    imad_c = source(decoded, SourceRole::c);
    check(imad_c != nullptr && std::holds_alternative<Register>(imad_c->value) &&
              std::get<Register>(imad_c->value).index == 5,
          "IMAD32I addend reads the old destination");

    // MUFU: multifunction selector and modifiers.
    word = opcode("0101000010000");
    word = set(word, 20, 4, 5) | std::uint64_t{1} << 46 | std::uint64_t{1} << 48 | std::uint64_t{1} << 50;
    decoded = decodes(word, "MUFU decodes");
    check(decoded.name == InstructionName::mufu && has_field(decoded, DecodedField::function, 5),
          "MUFU function selector");
    a = source(decoded, SourceRole::a);
    check(a != nullptr && a->modifiers.absolute && a->modifiers.negate, "MUFU absolute and negate modifiers");
    check(decoded.numeric && decoded.numeric->saturate, "MUFU saturation");

    // LEA and LEAHI publish their shift amounts; POPC and FLO read source B.
    word = opcode("0101101111010");
    word = set(word, 39, 5, 6) | std::uint64_t{1} << 41;
    decoded = decodes(word, "LEA register form decodes");
    check(decoded.name == InstructionName::lea && has_field(decoded, DecodedField::shift_mode, 6),
          "LEA shift amount");
    check(source(decoded, SourceRole::a) != nullptr && source(decoded, SourceRole::a)->modifiers.negate,
          "LEA negates A");
    word = opcode("0101101111011");
    word = set(word, 28, 5, 3) | set(word, 20, 8, 7) | set(word, 39, 8, 8);
    decoded = decodes(word, "LEAHI register form decodes");
    check(decoded.name == InstructionName::lea_hi && has_field(decoded, DecodedField::shift_mode, 3),
          "LEAHI shift amount");
    check(source(decoded, SourceRole::c) != nullptr, "LEAHI third source");
    word = opcode("0101110000001");
    decoded = decodes(word, "POPC register form decodes");
    check(decoded.name == InstructionName::popc && source(decoded, SourceRole::b) != nullptr,
          "POPC reads source B");
    word = opcode("0101110000000");
    word = set(word, 40, 1, 1) | set(word, 48, 1, 1);
    decoded = decodes(word, "BFE register form decodes");
    check(decoded.name == InstructionName::bfe && has_field(decoded, DecodedField::bitfield_mode, 1) &&
              has_field(decoded, DecodedField::signedness, 1),
          "BFE bit-reverse and signedness selectors");

    rejected(set(opcode("0101110000100"), 43, 2, 1), "IMNMX cross-lane selector is unsupported",
             DiagnosticCode::unsupported_modifier);
    rejected(set(opcode("0101110010001"), 39, 6, 1), "FCHK mode other than divide is unsupported",
             DiagnosticCode::unsupported_modifier);
}

void test_moves_predicates_and_half() {
    // MOV32I carries a raw 32-bit immediate.
    auto word = opcode("000000010000");
    word = set(word, 20, 32, 0xdeadbeef) | set(word, 0, 8, 9);
    auto decoded = decodes(word, "MOV32I decodes");
    check(decoded.name == InstructionName::mov32i, "MOV32I name");
    const auto* a = source(decoded, SourceRole::a);
    check(a != nullptr && std::get<LiteralBits>(a->value).bits == 0xdeadbeef, "MOV32I immediate");

    // SEL C selects between a register and a constant word by predicate.
    word = set(opcode("0100110010100"), 34, 5, 4);
    decoded = decodes(word, "SEL constant form decodes");
    const auto* sel_b = source(decoded, SourceRole::b);
    check(decoded.name == InstructionName::sel && sel_b != nullptr &&
          std::holds_alternative<ConstantWord>(sel_b->value), "SEL C source");
    check(source(decoded, SourceRole::predicate) != nullptr, "SEL predicate condition");

    // CSET combines a CC condition with a predicate through a Boolean op.
    word = opcode("0101000010011");
    word = set(word, 8, 5, 3) | set(word, 39, 3, 2) | set(word, 45, 2, 2);
    decoded = decodes(word, "CSET decodes");
    check(decoded.name == InstructionName::cset, "CSET name");
    check(has_field(decoded, DecodedField::condition, 3), "CSET CC condition select");
    check(has_field(decoded, DecodedField::boolean_operation, 2), "CSET Boolean operation");
    check(has_field(decoded, DecodedField::destination_type, 0),
          "CSET BVal is independent from the predicate index");

    word = opcode("0101000010001");
    word = set(word, 39, 3, 2);
    decoded = decodes(word, "PSET integer Boolean result decodes");
    check(has_field(decoded, DecodedField::destination_type, 0),
          "PSET BVal is independent from the predicate index");

    // PSETP writes paired predicates from three predicate reads.
    word = opcode("0101000010010");
    word = set(word, 29, 3, 1) | set(word, 12, 3, 2) | set(word, 39, 3, 3) | set(word, 3, 3, 4) | set(word, 0, 3, 5);
    decoded = decodes(word, "PSETP decodes");
    check(decoded.name == InstructionName::psetp, "PSETP name");
    check(access(decoded, RegisterRole::predicate_source, 1) != nullptr &&
          access(decoded, RegisterRole::predicate_source, 2) != nullptr &&
          access(decoded, RegisterRole::predicate_source, 3) != nullptr, "PSETP three predicate reads");
    check(access(decoded, RegisterRole::predicate_destination, 4) != nullptr &&
          access(decoded, RegisterRole::second_predicate_destination, 5) != nullptr, "PSETP paired destinations");

    // HADD2 R: half swizzles, packed output format, flush and saturation.
    word = opcode("0101110100010");
    word = set(word, 28, 2, 2) | set(word, 47, 2, 3) | set(word, 49, 2, 1);
    word |= std::uint64_t{1} << 39 | std::uint64_t{1} << 32;
    decoded = decodes(word, "HADD2 register form decodes");
    check(decoded.name == InstructionName::hadd2, "HADD2 name");
    const auto* b = source(decoded, SourceRole::b);
    check(b != nullptr && b->modifiers.subword == Subword::half_low, "HADD2 B half broadcast");
    a = source(decoded, SourceRole::a);
    check(a != nullptr && a->modifiers.subword == Subword::half_high, "HADD2 A half broadcast");
    check(has_field(decoded, DecodedField::half_mode, 1), "HADD2 output format selector");
    check(decoded.numeric && decoded.numeric->scalar == ScalarType::f16 && decoded.numeric->flush_inputs &&
          decoded.numeric->saturate, "HADD2 half numeric mode");

    // HADD2 I packs an f16x2 immediate; the high-half MSB lives at bit 56.
    word = opcode("0111101x0");
    word = set(word, 20, 10, 0x123) | set(word, 30, 9, 0x1AB) | std::uint64_t{1} << 56;
    decoded = decodes(word, "HADD2 immediate form decodes");
    b = source(decoded, SourceRole::b);
    check(b != nullptr && std::get<LiteralBits>(b->value).bits == (0x3ABu << 22 | 0x123u << 6) &&
              b->modifiers.scalar == ScalarType::f16,
          "HADD2 immediate expands both half significands into a packed F16 source");
    check(has_field(decoded, DecodedField::half_mode, 0), "HADD2 output selector rides half_mode");

    // HFMA2 Rc swaps the operand roles: the register is B, the constant is C.
    word = opcode("01100xxx1");
    word = set(word, 39, 8, 8) | set(word, 34, 5, 1);
    decoded = decodes(word, "HFMA2 register-constant form decodes");
    check(decoded.name == InstructionName::hfma2 && source(decoded, SourceRole::b) != nullptr &&
              std::holds_alternative<Register>(source(decoded, SourceRole::b)->value),
          "HFMA2 Rc register source rides B");
    check(source(decoded, SourceRole::c) != nullptr &&
              std::holds_alternative<ConstantWord>(source(decoded, SourceRole::c)->value),
          "HFMA2 Rc constant source rides C");

    // Packed-half multiply and FMA accept FTZ but reject FMZ because their
    // emitter has no zero-product selection semantics.
    struct FlushCase {
        std::uint64_t word;
        unsigned first;
        std::string_view accepted;
        std::string_view rejected_message;
    };
    for (const auto& test : {
             FlushCase{opcode("0101110100000"), 37, "HFMA2 R accepts FTZ", "HFMA2 R rejects FMZ"},
             FlushCase{opcode("01110xxx0"), 57, "HFMA2 I accepts FTZ", "HFMA2 I rejects FMZ"},
             FlushCase{opcode("01110xxx1"), 57, "HFMA2 C accepts FTZ", "HFMA2 C rejects FMZ"},
             FlushCase{opcode("01100xxx1"), 57, "HFMA2 Rc accepts FTZ", "HFMA2 Rc rejects FMZ"},
             FlushCase{opcode("0101110100001"), 39, "HMUL2 R accepts FTZ", "HMUL2 R rejects FMZ"},
             FlushCase{opcode("0111100x0"), 39, "HMUL2 I accepts FTZ", "HMUL2 I rejects FMZ"},
             FlushCase{opcode("0111100x1"), 39, "HMUL2 C accepts FTZ", "HMUL2 C rejects FMZ"},
             FlushCase{opcode("0010101"), 55, "HMUL232I accepts FTZ", "HMUL232I rejects FMZ"},
         }) {
        decoded = decodes(set(test.word, test.first, 2, 1), test.accepted);
        check(decoded.numeric && decoded.numeric->flush_inputs, test.accepted);
        rejected(set(test.word, test.first, 2, 2), test.rejected_message,
                 DiagnosticCode::unsupported_modifier);
    }
}

void test_shuffle() {
    const auto base = opcode("1110111100010");
    auto word = base | 4 | (std::uint64_t{5} << 8) | (std::uint64_t{6} << 20) |
                (std::uint64_t{7} << 39) | (std::uint64_t{2} << 30) | (std::uint64_t{1} << 48) |
                (std::uint64_t{3} << 16);
    auto decoded = decodes(word, "SHFL register form decodes");
    check(decoded.name == InstructionName::shfl && decoded.encoding == Encoding::registers,
          "SHFL RR name and route");
    check(decoded.predicate.index == 3 && !decoded.predicate.inverted, "SHFL keeps its guest predicate");
    check(access(decoded, RegisterRole::destination, 4) != nullptr &&
              access(decoded, RegisterRole::source_a, 5) != nullptr &&
              access(decoded, RegisterRole::source_b, 6) != nullptr &&
              access(decoded, RegisterRole::source_c, 7) != nullptr,
          "SHFL RR decodes four GPR roles");
    check(access(decoded, RegisterRole::predicate_destination, 1) != nullptr,
          "SHFL writes the in-range predicate destination");
    check(has_field(decoded, DecodedField::shuffle_mode, 2), "SHFL keeps the encoded mode selector");
    check(decoded.numeric && decoded.numeric->domain == Domain::bits &&
              decoded.numeric->scalar == ScalarType::u32,
          "SHFL is a raw word operation");

    // Immediate lane route (BFixShfl): a 5-bit literal at bit 20; the register
    // bits above it are ignored, and no mask route bit is consulted.
    word = base | 8 | (std::uint64_t{9} << 8) | (std::uint64_t{26} << 20) | (7ull << 16) |
           (std::uint64_t{4} << 39) | std::uint64_t{1} << 28 | (std::uint64_t{1} << 48);
    decoded = decodes(word, "SHFL immediate lane decodes");
    check(decoded.encoding == Encoding::immediate, "SHFL RI route");
    const auto* lane = source(decoded, SourceRole::b);
    check(lane != nullptr && std::holds_alternative<LiteralBits>(lane->value) &&
              std::get<LiteralBits>(lane->value).bits == 26,
          "SHFL lane immediate is a 5-bit literal");
    const auto* mask = source(decoded, SourceRole::c);
    check(mask != nullptr && std::holds_alternative<Register>(mask->value),
          "SHFL RI keeps the register mask");
    check(access(decoded, RegisterRole::source_c, 4) != nullptr, "SHFL RI mask register index");

    // Immediate mask route (CFixShfl): a 13-bit literal at bit 34.
    word = base | 8 | (std::uint64_t{9} << 8) | (std::uint64_t{12} << 20) | (7ull << 16) |
           (0x1f01ull << 34) | std::uint64_t{1} << 29 | (std::uint64_t{6} << 48);
    decoded = decodes(word, "SHFL immediate mask decodes");
    check(decoded.encoding == Encoding::immediate, "SHFL IR route");
    mask = source(decoded, SourceRole::c);
    check(mask != nullptr && std::holds_alternative<LiteralBits>(mask->value) &&
              std::get<LiteralBits>(mask->value).bits == 0x1f01,
          "SHFL mask immediate is a 13-bit literal");
    lane = source(decoded, SourceRole::b);
    check(lane != nullptr && std::holds_alternative<Register>(lane->value) &&
              access(decoded, RegisterRole::source_b, 12) != nullptr,
          "SHFL IR keeps the register lane");

    // Both immediates, butterfly mode, inverted guest predicate.
    word = base | 8 | (std::uint64_t{9} << 8) | (std::uint64_t{3} << 20) | (7ull << 16) |
           std::uint64_t{1} << 19 | (0x101full << 34) | std::uint64_t{1} << 29 |
           std::uint64_t{1} << 28 | (std::uint64_t{3} << 30) | (std::uint64_t{5} << 48);
    decoded = decodes(word, "SHFL both-immediate form decodes");
    check(decoded.encoding == Encoding::immediate, "SHFL II route");
    check(decoded.predicate.inverted, "SHFL decodes predicate inversion");
    check(has_field(decoded, DecodedField::shuffle_mode, 3), "SHFL butterfly mode");
    check(access(decoded, RegisterRole::predicate_destination, 5) != nullptr, "SHFL II predicate destination");
    lane = source(decoded, SourceRole::b);
    check(lane != nullptr && std::holds_alternative<LiteralBits>(lane->value) &&
              std::get<LiteralBits>(lane->value).bits == 3,
          "SHFL II lane immediate");
    mask = source(decoded, SourceRole::c);
    check(mask != nullptr && std::holds_alternative<LiteralBits>(mask->value) &&
              std::get<LiteralBits>(mask->value).bits == 0x101f,
          "SHFL II mask immediate");

    // The single hardware row fixes the 13-bit prefix; the neighbouring
    // 0xEF18..0xEF1F prefix and a foreign row find no rule.
    check(find_rule(base | std::uint64_t{1} << 51) == nullptr, "A non-SHFL 0xEF1 prefix finds no rule");
    check(find_rule(opcode("010110110111")) == nullptr, "DFMA is not admitted by this lane");
}

void test_unknown_encodings() {
    // Within-lane families without rules and foreign opcodes find no row.
    check(find_rule(0) == nullptr, "An all-zero word is not an instruction");
    check(find_rule(~std::uint64_t{0}) == nullptr, "A saturated word is not an instruction");
    // A row word with the low 50 bits masked away matches exactly one row.
    check(find_rule(opcode("0101110001011") | 0x5555'5555'5555ull) != nullptr,
          "Unused low bits do not affect matching");
}
} // namespace

int main() {
    test_table();
    test_float_arithmetic();
    test_comparison_and_conversion();
    test_integer_and_logical();
    test_moves_predicates_and_half();
    test_shuffle();
    test_unknown_encodings();
    if (failures) {
        std::cerr << failures << " decode assertion(s) failed\n";
        return 1;
    }
    std::cout << "Shader pipeline ALU decode passed\n";
    return 0;
}
