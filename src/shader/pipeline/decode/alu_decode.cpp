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

#include "pipeline/decoder.h"

#include <array>
#include <string_view>

namespace armigo::shader::pipeline {
namespace {
using Bits = RawInstruction;

constexpr std::uint64_t one(unsigned bit) { return std::uint64_t{1} << bit; }

// Decode table rows are 14 top-bit pattern strings; index 0 is bit 63.
constexpr DecodeRule rule(std::string_view pattern, DecodeInstruction decode) {
    std::uint64_t mask = 0, value = 0;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        const unsigned bit = 63 - static_cast<unsigned>(index);
        if (pattern[index] == '1') {
            mask |= one(bit);
            value |= one(bit);
        } else if (pattern[index] == '0') {
            mask |= one(bit);
        }
    }
    return DecodeRule{mask, value, decode};
}

// ---------------------------------------------------------------------------
// Shared population helpers. Selector values without a matching contract key
// stay in raw; the decoder comment names the bit so consumers either use it or
// fail explicitly. Reserved selector values are malformed input, never zero.
// ---------------------------------------------------------------------------

// Compile-time-selected positions use RawInstruction::bits; selector positions
// carried by parameters use this constexpr slice.
[[nodiscard]] constexpr std::uint64_t slice(std::uint64_t word, unsigned first, unsigned width) {
    return (word >> first) & ((std::uint64_t{1} << width) - 1);
}

[[nodiscard]] Diagnostic decode_error(Pc pc, InstructionName name, DiagnosticCode code, std::string what) {
    return diagnose(Phase::decode, code, {pc, name}, std::move(what));
}

[[nodiscard]] Result<DecodedInstruction> malformed(Pc pc, InstructionName name, std::string what) {
    return decode_error(pc, name, DiagnosticCode::malformed_input, std::move(what));
}

[[nodiscard]] Result<DecodedInstruction> unsupported(Pc pc, InstructionName name, std::string what) {
    return diagnose(Phase::decode, DiagnosticCode::unsupported_modifier, {pc, name}, std::move(what));
}

void add_field(DecodedInstruction& out, DecodedField key, std::uint64_t bits) {
    out.fields.push_back({key, bits});
}

void base(DecodedInstruction& out, Pc pc, Bits raw, InstructionName name, Encoding encoding) {
    out.pc = pc;
    out.raw = raw;
    out.name = name;
    out.encoding = encoding;
    out.predicate = {static_cast<std::uint8_t>(slice(raw.word, 16, 3)), slice(raw.word, 19, 1) != 0, true};
}

void gpr(DecodedInstruction& out, RegisterRole role, unsigned first, Access access) {
    out.registers.push_back({role, 0,
                             Register{RegisterFile::gpr, static_cast<std::uint16_t>(slice(out.raw.word, first, 8))},
                             access});
}

void destination(DecodedInstruction& out) { gpr(out, RegisterRole::destination, 0, Access::write); }

// F64 results occupy a consecutive register pair, low word first.
void second_destination(DecodedInstruction& out) {
    out.registers.push_back({RegisterRole::second_destination, 0,
                             Register{RegisterFile::gpr, static_cast<std::uint16_t>(slice(out.raw.word, 0, 8) + 1)},
                             Access::write});
}

void source_register(DecodedInstruction& out, RegisterRole role, SourceRole source, unsigned first) {
    gpr(out, role, first, Access::read);
    out.sources.push_back({source, Register{RegisterFile::gpr, static_cast<std::uint16_t>(slice(out.raw.word, first, 8))}, {}});
}

void source_a(DecodedInstruction& out) { source_register(out, RegisterRole::source_a, SourceRole::a, 8); }
void source_b(DecodedInstruction& out) { source_register(out, RegisterRole::source_b, SourceRole::b, 20); }
void source_c(DecodedInstruction& out) { source_register(out, RegisterRole::source_c, SourceRole::c, 39); }

// Standard signed 20-bit immediate: 19-bit magnitude at bit 20, sign at bit 56.
void immediate_20(DecodedInstruction& out, SourceRole role) {
    std::uint32_t value = static_cast<std::uint32_t>(slice(out.raw.word, 20, 19));
    if (slice(out.raw.word, 56, 1) != 0) value |= 0xfff80000u;
    out.sources.push_back({role, LiteralBits{value, ValueType::word32}, {}});
}

// Compact floating immediates store the high 20 bits of an IEEE-754 word.
// Bit 56 is immediate bit 19, not an integer sign-extension selector.
void float_immediate_20(DecodedInstruction& out, SourceRole role) {
    const auto value = static_cast<std::uint32_t>(slice(out.raw.word, 20, 19) |
                                                  (slice(out.raw.word, 56, 1) << 19));
    out.sources.push_back({role, LiteralBits{value << 12, ValueType::word32}, {}});
}

void immediate_32(DecodedInstruction& out, SourceRole role) {
    out.sources.push_back({role, LiteralBits{slice(out.raw.word, 20, 32), ValueType::word32}, {}});
}

// Byte offsets normalize once: the encoded word offset scales by four.
void constant(DecodedInstruction& out, SourceRole role) {
    out.sources.push_back({role,
                           ConstantWord{static_cast<std::uint8_t>(slice(out.raw.word, 34, 5)),
                                        static_cast<std::uint32_t>(slice(out.raw.word, 20, 14) * 4)},
                           {}});
}

// Route B either as a register, the standard immediate, or a constant word.
template <Encoding Route, RegisterRole Physical = RegisterRole::source_b>
void source_b_route(DecodedInstruction& out) {
    if constexpr (Route == Encoding::registers) source_b(out);
    else if constexpr (Route == Encoding::immediate) immediate_20(out, SourceRole::b);
    else constant(out, SourceRole::b);
}

template <Encoding Route>
void float_source_b_route(DecodedInstruction& out) {
    if constexpr (Route == Encoding::registers) source_b(out);
    else if constexpr (Route == Encoding::immediate) float_immediate_20(out, SourceRole::b);
    else constant(out, SourceRole::b);
}

// Predicate operand at 39..41 with inversion at 42.
void predicate_source(DecodedInstruction& out) {
    const auto index = static_cast<std::uint16_t>(slice(out.raw.word, 39, 3));
    const bool inverted = slice(out.raw.word, 42, 1) != 0;
    out.registers.push_back({RegisterRole::predicate_source, 0, Register{RegisterFile::predicate, index}, Access::read});
    out.sources.push_back({SourceRole::predicate, Register{RegisterFile::predicate, index},
                           SourceModifiers{false, inverted, Subword::full, ScalarType::u32}});
}

// Paired set-predicate results: Pd at 3..5, its complement at 0..2.
void predicate_destinations(DecodedInstruction& out) {
    out.registers.push_back({RegisterRole::predicate_destination, 0,
                             Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(out.raw.word, 3, 3))},
                             Access::write});
    out.registers.push_back({RegisterRole::second_predicate_destination, 0,
                             Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(out.raw.word, 0, 3))},
                             Access::write});
}

// WriteCC (bit 47; bit 52 on the 32-bit immediate forms) stores the zero and
// sign condition-code flags. The field is always
// published so consumers can require it; the physical flag writes appear only
// when the guest requests them.
void cc_write(DecodedInstruction& out, unsigned bit) {
    const bool set = slice(out.raw.word, bit, 1) != 0;
    add_field(out, DecodedField::write_cc, set ? 1 : 0);
    if (!set) return;
    for (const auto flag : {ConditionCode::zero, ConditionCode::sign}) {
        out.registers.push_back({RegisterRole::condition_code, 0,
                                 Register{RegisterFile::condition_code, static_cast<std::uint16_t>(flag)},
                                 Access::write});
    }
}

// Extended integer operations form a carry chain through the carry flag. The
// field is always published; the physical flag access appears only when set.
void carry_extended(DecodedInstruction& out, unsigned bit) {
    const bool set = slice(out.raw.word, bit, 1) != 0;
    add_field(out, DecodedField::extended, set ? 1 : 0);
    if (!set) return;
    out.registers.push_back({RegisterRole::condition_code, 0,
                             Register{RegisterFile::condition_code, static_cast<std::uint16_t>(ConditionCode::carry)},
                             Access::read_write});
}

// An extended-mode bit whose consumer semantics are not carry (recorded only).
void extended_field(DecodedInstruction& out, unsigned bit) {
    add_field(out, DecodedField::extended, slice(out.raw.word, bit, 1));
}

void negate(DecodedInstruction& out, SourceRole role, unsigned bit) {
    if (slice(out.raw.word, bit, 1) == 0) return;
    for (auto& source : out.sources) {
        if (source.role == role) source.modifiers.negate = true;
    }
}

void absolute(DecodedInstruction& out, SourceRole role, unsigned bit) {
    if (slice(out.raw.word, bit, 1) == 0) return;
    for (auto& source : out.sources) {
        if (source.role == role) source.modifiers.absolute = true;
    }
}

void signed_source(DecodedInstruction& out, SourceRole role) {
    for (auto& source : out.sources) {
        if (source.role == role) source.modifiers.scalar = ScalarType::s32;
    }
}

void signed_both(DecodedInstruction& out) {
    signed_source(out, SourceRole::a);
    signed_source(out, SourceRole::b);
    add_field(out, DecodedField::signedness, 1);
}

void subword(DecodedInstruction& out, SourceRole role, Subword selection) {
    for (auto& source : out.sources) {
        if (source.role == role) source.modifiers.subword = selection;
    }
}

void float_numeric(DecodedInstruction& out, Rounding rounding, bool flush, bool saturate,
                   ScalarType scalar = ScalarType::f32) {
    out.numeric = NumericMode{Domain::floating, scalar, rounding, flush, flush, saturate, false};
}

void integer_numeric(DecodedInstruction& out, Domain domain = Domain::unsigned_integer) {
    out.numeric = NumericMode{domain, ScalarType::u32};
}

void bits_numeric(DecodedInstruction& out) { out.numeric = NumericMode{Domain::bits, ScalarType::u32}; }

[[nodiscard]] constexpr Rounding map_round(std::uint64_t bits) {
    switch (bits) {
    case 1: return Rounding::toward_negative;
    case 2: return Rounding::toward_positive;
    case 3: return Rounding::toward_zero;
    default: return Rounding::nearest_even;
    }
}

// FMUL/FFMA/HFMA2 flush selectors: 0 keeps subnormals, 1 (FTZ) and 2 (FMZ)
// flush subnormal inputs and the rounded result; 3 is not defined. FMUL and
// FFMA retain the raw selector in `function`, allowing their emitter to
// distinguish FMZ's zero-product behavior from ordinary FTZ.
[[nodiscard]] Result<bool> map_flush(DecodedInstruction& out, unsigned first,
                                     bool supports_fmz = false) {
    const auto value = slice(out.raw.word, first, 2);
    if (value == 3) return decode_error(out.pc, out.name, DiagnosticCode::malformed_input, "Reserved flush modifier");
    if (value == 2 && !supports_fmz) {
        return decode_error(out.pc, out.name, DiagnosticCode::unsupported_modifier,
                            "FMZ is not implemented for packed-half arithmetic");
    }
    return value != 0;
}

// Half-operand swizzles (lane 7 convention): F16 keeps both packed halves and
// H0H0/H1H1 broadcast one half, both carrying scalar f16; F32 reads one f32
// value. Packed halves are recognisable by modifiers.scalar == f16.
void half_swizzle(DecodedInstruction& out, SourceRole role, unsigned first) {
    const auto selection = slice(out.raw.word, first, 2);
    for (auto& source : out.sources) {
        if (source.role != role) continue;
        switch (selection) {
        case 1: source.modifiers.scalar = ScalarType::f32; break;
        case 2:
            source.modifiers.scalar = ScalarType::f16;
            source.modifiers.subword = Subword::half_low;
            break;
        case 3:
            source.modifiers.scalar = ScalarType::f16;
            source.modifiers.subword = Subword::half_high;
            break;
        default: source.modifiers.scalar = ScalarType::f16; break;
        }
    }
}

// Byte-lane select: encoded 0..3 maps onto byte0..byte3.
void byte_select(DecodedInstruction& out, SourceRole role) {
    subword(out, role, static_cast<Subword>(static_cast<std::uint8_t>(Subword::byte0) + slice(out.raw.word, 41, 2)));
}

// ---------------------------------------------------------------------------
// Float arithmetic: FADD, FMUL, FFMA, FMNMX, FSWZADD, MUFU, RRO.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodedInstruction> finish(DecodedInstruction out) { return out; }

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_fadd_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::a, 48);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::a, 46);
    absolute(out, SourceRole::b, 49);
    float_numeric(out, map_round(slice(raw.word, 39, 2)), slice(raw.word, 44, 1) != 0, slice(raw.word, 50, 1) != 0);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_fadd_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::fadd32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    negate(out, SourceRole::a, 56);
    absolute(out, SourceRole::a, 54);
    negate(out, SourceRole::b, 53);
    absolute(out, SourceRole::b, 57);
    float_numeric(out, Rounding::nearest_even, slice(raw.word, 55, 1) != 0, false);
    cc_write(out, 52);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_fmul_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::a, 48);
    add_field(out, DecodedField::result_scale, slice(raw.word, 41, 3));
    auto flush = map_flush(out, 44, true);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    add_field(out, DecodedField::function, slice(raw.word, 44, 2));
    float_numeric(out, map_round(slice(raw.word, 39, 2)), std::get<bool>(flush), slice(raw.word, 50, 1) != 0);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_fmul_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::fmul32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    add_field(out, DecodedField::result_scale, 0); // No scale selector on 32I.
    auto flush = map_flush(out, 53, true);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    add_field(out, DecodedField::function, slice(raw.word, 53, 2));
    float_numeric(out, Rounding::nearest_even, std::get<bool>(flush), slice(raw.word, 55, 1) != 0);
    cc_write(out, 52);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, bool ReversedConstant = false>
[[nodiscard]] Result<DecodedInstruction> decode_ffma_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (ReversedConstant) {
        source_register(out, RegisterRole::source_b, SourceRole::b, 39);
        constant(out, SourceRole::c);
    } else {
        float_source_b_route<Route>(out);
        source_c(out);
    }
    negate(out, SourceRole::a, 48);
    negate(out, SourceRole::c, 49);
    auto flush = map_flush(out, 53, true);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    add_field(out, DecodedField::function, slice(raw.word, 53, 2));
    float_numeric(out, map_round(slice(raw.word, 51, 2)), std::get<bool>(flush), slice(raw.word, 50, 1) != 0);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_ffma_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::ffma32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    source_register(out, RegisterRole::source_c, SourceRole::c, 0);
    immediate_32(out, SourceRole::b);
    negate(out, SourceRole::a, 56);
    negate(out, SourceRole::c, 57);
    auto flush = map_flush(out, 53, true);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    add_field(out, DecodedField::function, slice(raw.word, 53, 2));
    float_numeric(out, Rounding::nearest_even, std::get<bool>(flush), slice(raw.word, 55, 1) != 0);
    cc_write(out, 52);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_fmnmx_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::a, 48);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::a, 46);
    absolute(out, SourceRole::b, 49);
    float_numeric(out, Rounding::nearest_even, slice(raw.word, 44, 1) != 0, false);
    predicate_source(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_fswzadd(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::fswzadd, Encoding::registers);
    destination(out);
    source_a(out);
    source_b(out);
    float_numeric(out, map_round(slice(raw.word, 39, 2)), slice(raw.word, 44, 1) != 0, false);
    // permutation_mode packs the PN component selector (bits 28..35) with the
    // NDV modifier bit 38.
    add_field(out, DecodedField::permutation_mode, slice(raw.word, 28, 8) | (slice(raw.word, 38, 1) << 8));
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_mufu(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::mufu, Encoding::registers);
    destination(out);
    source_a(out);
    add_field(out, DecodedField::function, slice(raw.word, 20, 4));
    absolute(out, SourceRole::a, 46);
    negate(out, SourceRole::a, 48);
    float_numeric(out, Rounding::nearest_even, false, slice(raw.word, 50, 1) != 0);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_rro_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::b, 49);
    add_field(out, DecodedField::function, slice(raw.word, 39, 1));
    float_numeric(out, Rounding::nearest_even, false, false);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Float comparison: FSET, FSETP, FCHK, FCMP. Bit 47 is both WriteCC and FTZ on
// FSETP; Armigo decodes it as the FTZ input flush there.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route, bool ToPredicate>
[[nodiscard]] Result<DecodedInstruction> decode_fset_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    if constexpr (ToPredicate) predicate_destinations(out);
    else destination(out);
    source_a(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::a, 43);
    absolute(out, SourceRole::b, 44);
    if constexpr (ToPredicate) {
        negate(out, SourceRole::b, 6);
        absolute(out, SourceRole::a, 7);
    } else {
        negate(out, SourceRole::b, 53);
        absolute(out, SourceRole::a, 54);
    }
    predicate_source(out);
    if constexpr (ToPredicate) {
        float_numeric(out, Rounding::nearest_even, slice(raw.word, 47, 1) != 0, false);
        add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
    } else {
        float_numeric(out, Rounding::nearest_even, slice(raw.word, 55, 1) != 0, false);
        add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
        add_field(out, DecodedField::half_mode, slice(raw.word, 52, 1)); // BVal.
        cc_write(out, 47);
    }
    add_field(out, DecodedField::condition, slice(raw.word, 48, 4));
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_fchk_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    out.registers.push_back({RegisterRole::predicate_destination, 0,
                             Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(raw.word, 3, 3))},
                             Access::write});
    source_a(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::a, 48);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::a, 46);
    absolute(out, SourceRole::b, 49);
    const auto mode = slice(raw.word, 39, 6);
    if (mode != 0) return unsupported(pc, Name, "FCHK mode other than divide");
    add_field(out, DecodedField::condition, mode);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, bool ReversedConstant>
[[nodiscard]] Result<DecodedInstruction> decode_fcmp_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (ReversedConstant) {
        source_register(out, RegisterRole::source_b, SourceRole::b, 39);
        constant(out, SourceRole::c);
    } else {
        float_source_b_route<Route>(out);
        source_c(out);
    }
    float_numeric(out, Rounding::nearest_even, slice(raw.word, 47, 1) != 0, false);
    add_field(out, DecodedField::condition, slice(raw.word, 48, 4));
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Conversions: F2F, F2I, I2F, I2I. Formats stay raw selectors in source_type /
// destination_type; the conversion callback maps them to ScalarType.
// ---------------------------------------------------------------------------

// Same-format F2F integral rounding is the guest IntegerRound: bit 42 joins
// bits 40..39. Values 0..3 pass through; 4 rounds to nearest even, 5 floors,
// 6 takes the ceiling and 7 truncates. Cross-format conversion instead uses
// bits 40..39 as RoundMode. The raw three-bit selector rides `integral_round`
// while NumericMode carries the applicable arithmetic/conversion rounding.
void f2f_round(DecodedInstruction& out) {
    const auto value = (slice(out.raw.word, 42, 1) << 2) | slice(out.raw.word, 39, 2);
    add_field(out, DecodedField::integral_round, value);
}

[[nodiscard]] constexpr Rounding map_integral_round(std::uint64_t bits) {
    return bits >= 4 ? map_round(bits & 3) : Rounding::nearest_even;
}

[[nodiscard]] Result<ScalarType> float_format(DecodedInstruction& out, std::uint64_t raw_format) {
    switch (raw_format) {
    case 1: return ScalarType::f16;
    case 2: return ScalarType::f32;
    case 3: return ScalarType::f64;
    default: return decode_error(out.pc, out.name, DiagnosticCode::malformed_input, "Reserved floating format selector");
    }
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_f2f_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::b, 49);
    add_field(out, DecodedField::source_type, slice(raw.word, 10, 2));
    add_field(out, DecodedField::destination_type, slice(raw.word, 8, 2));
    subword(out, SourceRole::b, slice(raw.word, 41, 1) != 0 ? Subword::half_high : Subword::half_low);
    if (slice(raw.word, 8, 2) == 3) second_destination(out);
    f2f_round(out);
    auto scalar = float_format(out, slice(raw.word, 8, 2));
    if (auto* error = std::get_if<Diagnostic>(&scalar)) return *error;
    const auto source_format = slice(raw.word, 10, 2);
    const auto destination_format = slice(raw.word, 8, 2);
    const auto rounding = source_format == destination_format
                              ? map_integral_round(slice(raw.word, 42, 1) << 2 | slice(raw.word, 39, 2))
                              : map_round(slice(raw.word, 39, 2));
    float_numeric(out, rounding,
                  slice(raw.word, 44, 1) != 0, slice(raw.word, 50, 1) != 0, std::get<ScalarType>(scalar));
    cc_write(out, 47);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_f2i_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    float_source_b_route<Route>(out);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::b, 49);
    add_field(out, DecodedField::source_type, slice(raw.word, 10, 2));
    add_field(out, DecodedField::destination_type, (slice(raw.word, 12, 1) << 2) | slice(raw.word, 8, 2));
    subword(out, SourceRole::b, slice(raw.word, 41, 1) != 0 ? Subword::half_high : Subword::half_low);
    auto scalar = float_format(out, slice(raw.word, 10, 2));
    if (auto* error = std::get_if<Diagnostic>(&scalar)) return *error;
    float_numeric(out, map_round(slice(raw.word, 39, 2)), slice(raw.word, 44, 1) != 0, false, std::get<ScalarType>(scalar));
    cc_write(out, 47);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_i2f_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_b_route<Route>(out);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::b, 49);
    byte_select(out, SourceRole::b);
    add_field(out, DecodedField::source_type, (slice(raw.word, 13, 1) << 2) | slice(raw.word, 10, 2));
    add_field(out, DecodedField::destination_type, slice(raw.word, 8, 2));
    if (slice(raw.word, 8, 2) == 3) second_destination(out);
    auto scalar = float_format(out, slice(raw.word, 8, 2));
    if (auto* error = std::get_if<Diagnostic>(&scalar)) return *error;
    float_numeric(out, map_round(slice(raw.word, 39, 2)), false, false, std::get<ScalarType>(scalar));
    cc_write(out, 47);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_i2i_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_b_route<Route>(out);
    negate(out, SourceRole::b, 45);
    absolute(out, SourceRole::b, 49);
    byte_select(out, SourceRole::b);
    add_field(out, DecodedField::source_type, (slice(raw.word, 13, 1) << 2) | slice(raw.word, 10, 2));
    add_field(out, DecodedField::destination_type, (slice(raw.word, 12, 1) << 2) | slice(raw.word, 8, 2));
    add_field(out, DecodedField::integer_saturation, slice(raw.word, 50, 1));
    cc_write(out, 47);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Integer arithmetic: IADD, IADD3, ISCADD, IMAD, IMUL, IMNMX, XMAD.
// AvgMode and the IADD/IMAD saturation flag pack into `function` (lane 6
// convention); IADD3 half selects and the partial-sum shift into `half_mode`.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_iadd_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    add_field(out, DecodedField::function,
              Name == InstructionName::iscadd
                  ? slice(raw.word, 48, 2)
                  : (slice(raw.word, 48, 2) | (slice(raw.word, 50, 1) << 2))); // AvgMode | sat.
    if constexpr (Name == InstructionName::iscadd) {
        add_field(out, DecodedField::shift_mode, slice(raw.word, 39, 5));
    } else {
        carry_extended(out, 43);
    }
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

template <InstructionName Name>
[[nodiscard]] Result<DecodedInstruction> decode_iadd_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    if constexpr (Name == InstructionName::iadd32i) {
        carry_extended(out, 53);
        add_field(out, DecodedField::function,
                  slice(raw.word, 55, 2) | (slice(raw.word, 54, 1) << 2)); // AvgMode | sat.
    } else {
        add_field(out, DecodedField::shift_mode, slice(raw.word, 53, 5));
    }
    integer_numeric(out);
    cc_write(out, 52);
    return finish(std::move(out));
}

// IADD3: per-source half selection, partial-sum shift, and negation all pack
// into `half_mode` (A/B/C selects at bits 1..0/3..2/5..4, Lrs at 7..6).
template <Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_iadd3_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::iadd3, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    source_c(out);
    if constexpr (Route == Encoding::registers) {
        if (slice(raw.word, 35, 2) == 3 || slice(raw.word, 33, 2) == 3 || slice(raw.word, 31, 2) == 3) {
            return malformed(pc, InstructionName::iadd3, "Reserved IADD3 half selector");
        }
        if (slice(raw.word, 37, 2) == 3) return malformed(pc, InstructionName::iadd3, "Reserved IADD3 shift selector");
        add_field(out, DecodedField::half_mode,
                  slice(raw.word, 35, 2) | (slice(raw.word, 33, 2) << 2) |
                      (slice(raw.word, 31, 2) << 4) | (slice(raw.word, 37, 2) << 6));
    } else {
        // The immediate and constant routes have no half selects or partial
        // shift; the emitter still requires the key, so publish the B32 default.
        add_field(out, DecodedField::half_mode, 0);
    }
    negate(out, SourceRole::a, 51);
    negate(out, SourceRole::b, 50);
    negate(out, SourceRole::c, 49);
    integer_numeric(out);
    carry_extended(out, 48);
    cc_write(out, 47);
    return finish(std::move(out));
}

// IMAD/IMAD32I: HILO packs into `half_mode` bit 0, per-source signedness into
// `signedness` (bit 0 A, bit 1 B), AvgMode and saturation into `function`.
template <InstructionName Name, Encoding Route, bool ReversedConstant = false>
[[nodiscard]] Result<DecodedInstruction> decode_imad_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (ReversedConstant) {
        source_register(out, RegisterRole::source_b, SourceRole::b, 39);
        constant(out, SourceRole::c);
    } else {
        source_b_route<Route>(out);
        source_c(out);
    }
    add_field(out, DecodedField::half_mode, slice(raw.word, 54, 1));
    add_field(out, DecodedField::signedness, slice(raw.word, 48, 1) | (slice(raw.word, 53, 1) << 1));
    add_field(out, DecodedField::function, slice(raw.word, 51, 2) | (slice(raw.word, 50, 1) << 2));
    carry_extended(out, 49);
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_imad_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::imad32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    source_register(out, RegisterRole::source_c, SourceRole::c, 0);
    add_field(out, DecodedField::half_mode, slice(raw.word, 53, 1));
    add_field(out, DecodedField::signedness, slice(raw.word, 54, 1) | (slice(raw.word, 57, 1) << 1));
    add_field(out, DecodedField::function, slice(raw.word, 55, 2));
    // IMAD32I carries no extended bit, but its emitter requires the key.
    add_field(out, DecodedField::extended, 0);
    integer_numeric(out);
    cc_write(out, 52);
    return finish(std::move(out));
}

// IMUL/IMUL32I: high-product half and signedness use the same keys as IMAD.
template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_imul_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    add_field(out, DecodedField::half_mode, slice(raw.word, 39, 1));
    add_field(out, DecodedField::signedness, slice(raw.word, 40, 1) | (slice(raw.word, 41, 1) << 1));
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_imul_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::imul32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    add_field(out, DecodedField::half_mode, slice(raw.word, 53, 1));
    add_field(out, DecodedField::signedness, slice(raw.word, 54, 1) | (slice(raw.word, 55, 1) << 1));
    integer_numeric(out);
    cc_write(out, 52);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_imnmx_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    if (slice(raw.word, 48, 1) != 0) {
        signed_both(out);
    }
    if (slice(raw.word, 43, 2) != 0) return unsupported(pc, Name, "Cross-lane selector is unsupported");
    integer_numeric(out);
    predicate_source(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

// XMAD: 16-bit multiply-add. The Ccop packs into `function` (bits 2..0) and
// HiloA/HiloB/PSL/Mrg into `half_mode` bits 3..0; signedness rides
// `signedness` (bit 0 A, bit 1 B). Ccop is 3 bits on the register and
// immediate routes and 2 bits on the constant-buffer routes.
template <InstructionName Name, Encoding Route, bool WithMrg>
[[nodiscard]] Result<DecodedInstruction> decode_xmad_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (Route == Encoding::immediate) {
        out.sources.push_back({SourceRole::b, LiteralBits{slice(raw.word, 20, 16), ValueType::word32}, {}});
    } else {
        source_b_route<Route>(out);
    }
    source_c(out);
    add_field(out, DecodedField::signedness, slice(raw.word, 48, 1) | (slice(raw.word, 49, 1) << 1));
    std::uint64_t half_mode = slice(raw.word, 53, 1); // HiloA.
    if constexpr (Route == Encoding::registers) {
        half_mode |= slice(raw.word, 35, 1) << 1; // HiloB.
        add_field(out, DecodedField::function, slice(raw.word, 50, 3));
        half_mode |= (slice(raw.word, 37, 1) << 3) | (slice(raw.word, 36, 1) << 2);
    } else if constexpr (Route == Encoding::immediate) {
        add_field(out, DecodedField::function, slice(raw.word, 50, 3));
        half_mode |= (slice(raw.word, 37, 1) << 3) | (slice(raw.word, 36, 1) << 2);
    } else {
        half_mode |= slice(raw.word, 52, 1) << 1; // HiloB.
        add_field(out, DecodedField::function, slice(raw.word, 50, 2));
        half_mode |= (slice(raw.word, 56, 1) << 3) | (slice(raw.word, 55, 1) << 2);
    }
    add_field(out, DecodedField::half_mode, half_mode);
    extended_field(out, Route == Encoding::constant_buffer ? 54 : 38);
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

// XMAD Rc: cbuf source at 20..33 with HILO-B at bit 52; no merge selectors.
[[nodiscard]] Result<DecodedInstruction> decode_xmad_rc(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::xmad, Encoding::constant_buffer);
    destination(out);
    source_a(out);
    source_register(out, RegisterRole::source_b, SourceRole::b, 39);
    constant(out, SourceRole::c);
    add_field(out, DecodedField::signedness, slice(raw.word, 48, 1) | (slice(raw.word, 49, 1) << 1));
    add_field(out, DecodedField::function, slice(raw.word, 50, 2));
    add_field(out, DecodedField::half_mode, slice(raw.word, 53, 1) | (slice(raw.word, 52, 1) << 1));
    extended_field(out, 54);
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Integer comparison: ISET, ISETP, ICMP.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route, bool ToPredicate>
[[nodiscard]] Result<DecodedInstruction> decode_iset_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    if constexpr (ToPredicate) predicate_destinations(out);
    else destination(out);
    source_a(out);
    source_b_route<Route>(out);
    if (slice(raw.word, 48, 1) != 0) signed_both(out);
    add_field(out, DecodedField::condition, slice(raw.word, 49, 3));
    predicate_source(out);
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
    if constexpr (!ToPredicate) {
        add_field(out, DecodedField::destination_type, slice(raw.word, 44, 1)); // BVal.
        cc_write(out, 47);
    }
    extended_field(out, 43);
    integer_numeric(out, Domain::bits);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, bool SourceCRegister>
[[nodiscard]] Result<DecodedInstruction> decode_icmp_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (SourceCRegister) source_b_route<Route>(out);
    else source_b(out);
    source_c(out);
    if (slice(raw.word, 48, 1) != 0) signed_both(out);
    add_field(out, DecodedField::condition, slice(raw.word, 49, 3));
    integer_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Logical: LOP, LOP3, LOP32I. The logic function packs into `boolean_operation`
// (LogicOp) and the result predicate test (PredicateOp) into `condition`; the
// LOP3 truth table rides `permutation_mode`.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_lop_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    if constexpr (Name == InstructionName::lop) {
        negate(out, SourceRole::a, 39);
        negate(out, SourceRole::b, 40);
        add_field(out, DecodedField::boolean_operation, slice(raw.word, 41, 2));
        add_field(out, DecodedField::condition, slice(raw.word, 44, 2));
        out.registers.push_back({RegisterRole::predicate_destination, 0,
                                 Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(raw.word, 48, 3))},
                                 Access::write});
        extended_field(out, 43);
        cc_write(out, 47);
    } else {
        source_c(out);
        if constexpr (Route == Encoding::registers) {
            add_field(out, DecodedField::permutation_mode, slice(raw.word, 28, 8));
            add_field(out, DecodedField::condition, slice(raw.word, 36, 2));
            extended_field(out, 38);
            out.registers.push_back({RegisterRole::predicate_destination, 0,
                                     Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(raw.word, 48, 3))},
                                     Access::write});
        } else {
            add_field(out, DecodedField::permutation_mode, slice(raw.word, 48, 8));
            // These routes have no predicate result or extended-Z input.
            // In the immediate route bit 56 belongs to the signed operand.
            add_field(out, DecodedField::condition, 0);
            add_field(out, DecodedField::extended, 0);
        }
        cc_write(out, 47);
    }
    bits_numeric(out);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_lop_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::lop32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    negate(out, SourceRole::a, 55);
    negate(out, SourceRole::b, 56);
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 53, 2));
    extended_field(out, 57);
    bits_numeric(out);
    cc_write(out, 52);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Shifts: SHF, SHL, SHR. SHF packs MaxShift (bits 1..0) and the table-row
// direction (bit 2) into `shift_mode` with the wrap flag in `shift_wrap`; the
// SHL extended bit and the SHR/SHL condition-code write stay in raw for the
// emit lane to accept or reject explicitly.
// ---------------------------------------------------------------------------

template <bool Left, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_shf_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::shf, Route);
    destination(out);
    source_a(out);
    if constexpr (Route == Encoding::registers) {
        source_b(out);
    } else {
        out.sources.push_back({SourceRole::b, LiteralBits{slice(raw.word, 20, 6), ValueType::word32}, {}});
    }
    source_c(out);
    add_field(out, DecodedField::shift_mode, slice(raw.word, 37, 2) | (Left ? 0 : (std::uint64_t{1} << 2)));
    add_field(out, DecodedField::shift_wrap, slice(raw.word, 50, 1));
    bits_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, bool SignedShift>
[[nodiscard]] Result<DecodedInstruction> decode_shift_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    if constexpr (SignedShift) {
        if (slice(raw.word, 48, 1) != 0) signed_both(out);
        if (slice(raw.word, 43, 2) != 0) return unsupported(pc, Name, "Cross-lane selector is unsupported");
        // SHR bit-reversal rides bitfield_mode bit 0.
        add_field(out, DecodedField::bitfield_mode, slice(raw.word, 40, 1));
    } else {
        extended_field(out, 43);
    }
    add_field(out, DecodedField::shift_wrap, slice(raw.word, 39, 1));
    bits_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// SHFL: one hardware row with two operand-route selectors. BFixShfl (bit 28)
// routes the lane operand to a 5-bit immediate at 20..24 and CFixShfl (bit 29)
// routes the clamp/segment mask to a 13-bit immediate at 34..46; the register
// forms read Rb at 20..27 and Rc at 39..46. The mode (Idx/Up/Down/Bfly) rides
// `shuffle_mode` and the in-range predicate destination is LPd at 48..50.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodedInstruction> decode_shfl(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::shfl, Encoding::registers);
    destination(out);
    source_a(out);
    const bool lane_immediate = slice(raw.word, 28, 1) != 0;
    const bool mask_immediate = slice(raw.word, 29, 1) != 0;
    if (lane_immediate) {
        out.sources.push_back({SourceRole::b, LiteralBits{slice(raw.word, 20, 5), ValueType::word32}, {}});
    } else {
        source_b(out);
    }
    if (mask_immediate) {
        out.sources.push_back({SourceRole::c, LiteralBits{slice(raw.word, 34, 13), ValueType::word32}, {}});
    } else {
        source_c(out);
    }
    add_field(out, DecodedField::shuffle_mode, slice(raw.word, 30, 2));
    out.registers.push_back({RegisterRole::predicate_destination, 0,
                             Register{RegisterFile::predicate, static_cast<std::uint16_t>(slice(raw.word, 48, 3))},
                             Access::write});
    if (lane_immediate || mask_immediate) out.encoding = Encoding::immediate;
    bits_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Bit-field helpers: BFE, BFI. Both share the bit-reverse/signedness keys;
// WriteCC (bit 47) has no emitter consumer and stays in raw.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_bfe_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    if (slice(raw.word, 48, 1) != 0) signed_both(out);
    add_field(out, DecodedField::bitfield_mode, slice(raw.word, 40, 1)); // Brev.
    bits_numeric(out);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, bool WithRegisterB>
[[nodiscard]] Result<DecodedInstruction> decode_bfi_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    if constexpr (WithRegisterB) {
        // Rc: the position/size register rides source b, the constant buffer
        // (the inserted-into base) rides source c.
        source_register(out, RegisterRole::source_b, SourceRole::b, 39);
        constant(out, SourceRole::c);
    } else {
        source_b_route<Route>(out);
        source_c(out);
    }
    bits_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// FLO and POPC read only source B; FLO's search-from-LSB selector rides
// bitfield_mode and POPC carries no other selector.
// ---------------------------------------------------------------------------

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_flo_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_b_route<Route>(out);
    if (slice(raw.word, 48, 1) != 0) signed_both(out);
    add_field(out, DecodedField::bitfield_mode, slice(raw.word, 37, 1)); // Sh.
    negate(out, SourceRole::b, 40);
    bits_numeric(out);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_popc_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_b_route<Route>(out);
    negate(out, SourceRole::b, 40);
    bits_numeric(out);
    return finish(std::move(out));
}

// S2R reads one special register; the selector rides `function` bits 7..0.
[[nodiscard]] Result<DecodedInstruction> decode_s2r(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::s2r, Encoding::registers);
    destination(out);
    add_field(out, DecodedField::function, slice(raw.word, 20, 8));
    bits_numeric(out);
    return finish(std::move(out));
}

// LEA computes (A << Imm5) + B; LEA_HI folds the shifted-out high bits of A
// into C first. The shift amount packs into `shift_mode`; the predicate
// destination (bits 48..50), WriteCC and X have no emitter consumer and stay
// in raw.
template <InstructionName Name, Encoding Route, unsigned ShiftFirst, unsigned NegateBit>
[[nodiscard]] Result<DecodedInstruction> decode_lea_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    add_field(out, DecodedField::shift_mode, slice(raw.word, ShiftFirst, 5));
    negate(out, SourceRole::a, NegateBit);
    integer_numeric(out);
    return finish(std::move(out));
}

template <InstructionName Name, Encoding Route, unsigned ShiftFirst, unsigned NegateBit>
[[nodiscard]] Result<DecodedInstruction> decode_lea_hi_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    source_c(out);
    add_field(out, DecodedField::shift_mode, slice(raw.word, ShiftFirst, 5));
    negate(out, SourceRole::a, NegateBit);
    integer_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Moves and selection: MOV, MOV32I, SEL, CS2R. The 64-bit move lane selector
// (QuadMask) has no contract key; MOV keeps it in raw at bits 39..42 and
// MOV32I at bits 12..15 for the emit lane.
// ---------------------------------------------------------------------------

template <Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_mov_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::mov, Route);
    destination(out);
    if constexpr (Route == Encoding::registers) {
        source_register(out, RegisterRole::source_a, SourceRole::a, 20);
    } else if constexpr (Route == Encoding::immediate) {
        immediate_20(out, SourceRole::a);
    } else {
        constant(out, SourceRole::a);
    }
    bits_numeric(out);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_mov_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::mov32i, Encoding::immediate32);
    destination(out);
    immediate_32(out, SourceRole::a);
    bits_numeric(out);
    return finish(std::move(out));
}

template <Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_sel_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::sel, Route);
    destination(out);
    source_a(out);
    source_b_route<Route>(out);
    predicate_source(out);
    bits_numeric(out);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_cs2r(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::cs2r, Encoding::registers);
    destination(out);
    // CS2R fills a destination pair from a special register; the selector has
    // no contract key, so source_type keeps its raw byte.
    add_field(out, DecodedField::source_type, slice(raw.word, 20, 8));
    bits_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Predicate arithmetic: CSET, CSETP, PSET, PSETP, P2R, R2P.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodedInstruction> decode_cset(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::cset, Encoding::registers);
    destination(out);
    add_field(out, DecodedField::condition, slice(raw.word, 8, 5)); // CC condition select.
    predicate_source(out);
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
    add_field(out, DecodedField::destination_type, slice(raw.word, 44, 1)); // BVal.
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_csetp(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::csetp, Encoding::registers);
    predicate_destinations(out);
    add_field(out, DecodedField::condition, slice(raw.word, 8, 5));
    predicate_source(out);
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
    cc_write(out, 47);
    return finish(std::move(out));
}

// PSET/PSETP combine three predicate operands. Each physical read joins the
// registers, and the logical inputs ride source roles a/b/c as Src2Pred
// (inversion bit 15), Src1Pred (bit 32) and SrcPred (bit 42).
void pset_predicate_inputs(DecodedInstruction& out) {
    struct Input { unsigned first; unsigned invert; SourceRole role; };
    for (const auto input : {Input{12u, 15u, SourceRole::a},
                             Input{29u, 32u, SourceRole::b},
                             Input{39u, 42u, SourceRole::c}}) {
        const auto index = static_cast<std::uint16_t>(slice(out.raw.word, input.first, 3));
        out.registers.push_back({RegisterRole::predicate_source, 0,
                                 Register{RegisterFile::predicate, index}, Access::read});
        out.sources.push_back({input.role, Register{RegisterFile::predicate, index},
                               SourceModifiers{false, slice(out.raw.word, input.invert, 1) != 0,
                                               Subword::full, ScalarType::u32}});
    }
}

[[nodiscard]] Result<DecodedInstruction> decode_pset(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::pset, Encoding::registers);
    destination(out);
    pset_predicate_inputs(out);
    // boolean_operation packs the AB and C selectors; BVal rides
    // destination_type bit 0.
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 24, 2) | (slice(raw.word, 45, 2) << 2));
    add_field(out, DecodedField::destination_type, slice(raw.word, 44, 1));
    integer_numeric(out);
    cc_write(out, 47);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_psetp(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::psetp, Encoding::registers);
    predicate_destinations(out);
    pset_predicate_inputs(out);
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 24, 2) | (slice(raw.word, 45, 2) << 2));
    return finish(std::move(out));
}

// P2R gathers predicates into a GPR byte; R2P scatters a source byte into the
// predicate file. bitfield_mode carries ByteSel (bits 1..0) and the CCPR
// flag-access bit (bit 2) for both.
template <InstructionName Name, Encoding Route, bool WritesGpr>
[[nodiscard]] Result<DecodedInstruction> decode_predicate_move_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    if constexpr (WritesGpr) destination(out);
    source_a(out);
    source_b_route<Route>(out);
    add_field(out, DecodedField::bitfield_mode, slice(raw.word, 41, 2) | (slice(raw.word, 40, 1) << 2));
    bits_numeric(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// Half-float arithmetic and comparison: HADD2, HFMA2, HMUL2, HSET2, HSETP2.
// Bit 47 is both the packed-immediate high-half MSB and an A-swizzle bit on the
// immediate routes; Armigo keeps both extractions.
// ---------------------------------------------------------------------------

void half_numeric(DecodedInstruction& out, bool flush, bool saturate = false) {
    float_numeric(out, Rounding::nearest_even, flush, saturate, ScalarType::f16);
}

// Packed f16x2 constant: low half at 20..29, high half at 30..38 plus the MSB
// at bit 56.
void half_immediate(DecodedInstruction& out) {
    const std::uint32_t low = static_cast<std::uint32_t>(slice(out.raw.word, 20, 10));
    const std::uint32_t high = static_cast<std::uint32_t>((slice(out.raw.word, 56, 1) << 9) | slice(out.raw.word, 30, 9));
    SourceModifiers packed;
    packed.scalar = ScalarType::f16;
    out.sources.push_back({SourceRole::b,
                          LiteralBits{(low << 6) | (high << 22), ValueType::word32}, packed});
}

// Half-operand output selector (guest OFmt) riding `half_mode`: 0 F16 pack,
// 1 F32 lane zero, 2 MrgH0, 3 MrgH1.
void half_output_selector(DecodedInstruction& out) {
    add_field(out, DecodedField::half_mode, slice(out.raw.word, 49, 2));
}

template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_hadd2_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    unsigned saturate_bit = 32;
    if constexpr (Route == Encoding::registers) {
        source_b(out);
        half_swizzle(out, SourceRole::b, 28);
        absolute(out, SourceRole::b, 30);
        negate(out, SourceRole::b, 31);
        half_output_selector(out);
    } else if constexpr (Route == Encoding::immediate) {
        half_immediate(out);
        half_output_selector(out);
        saturate_bit = 52;
    } else {
        constant(out, SourceRole::b);
        absolute(out, SourceRole::b, 54);
        negate(out, SourceRole::b, 56);
        half_output_selector(out);
        saturate_bit = 52;
    }
    half_swizzle(out, SourceRole::a, 47);
    absolute(out, SourceRole::a, 44);
    negate(out, SourceRole::a, 43);
    half_numeric(out, slice(raw.word, 39, 1) != 0, slice(raw.word, saturate_bit, 1) != 0);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_hadd2_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::hadd232i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    out.sources.back().modifiers.scalar = ScalarType::f16;
    half_swizzle(out, SourceRole::a, 53);
    negate(out, SourceRole::a, 56);
    half_numeric(out, slice(raw.word, 55, 1) != 0, slice(raw.word, 52, 1) != 0);
    // The 32-bit immediate form always packs both halves (OFmt F16).
    add_field(out, DecodedField::half_mode, 0);
    return finish(std::move(out));
}

// HFMA2 route layouts differ: the register form negates A (bit 31) and C
// (bit 30); the immediate form only negates C (bit 51); the constant form
// negates A (bit 56) and C; and the register-constant form swaps the roles —
// the register at 39..46 is the B operand (C swizzle, NegA) while the constant
// buffer is the C operand (NegC).
template <Encoding Route, bool RegisterConstant = false>
[[nodiscard]] Result<DecodedInstruction> decode_hfma2_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::hfma2, Route);
    destination(out);
    source_a(out);
    if constexpr (Route == Encoding::registers) {
        source_b(out);
        half_swizzle(out, SourceRole::b, 28);
        source_c(out);
        half_swizzle(out, SourceRole::c, 35);
        negate(out, SourceRole::a, 31);
        negate(out, SourceRole::c, 30);
        half_output_selector(out);
    } else if constexpr (Route == Encoding::immediate) {
        half_immediate(out);
        source_c(out);
        half_swizzle(out, SourceRole::c, 53);
        negate(out, SourceRole::c, 51);
        half_output_selector(out);
    } else if constexpr (RegisterConstant) {
        constant(out, SourceRole::c);
        negate(out, SourceRole::c, 51);
        source_register(out, RegisterRole::source_b, SourceRole::b, 39);
        half_swizzle(out, SourceRole::b, 53);
        negate(out, SourceRole::a, 56);
        half_output_selector(out);
    } else {
        source_register(out, RegisterRole::source_c, SourceRole::c, 39);
        half_swizzle(out, SourceRole::c, 53);
        negate(out, SourceRole::c, 51);
        constant(out, SourceRole::b);
        negate(out, SourceRole::a, 56);
        half_output_selector(out);
    }
    half_swizzle(out, SourceRole::a, 47);
    const unsigned flush_first = Route == Encoding::registers ? 37 : 57;
    const unsigned saturate_bit = Route == Encoding::registers ? 32 : 52;
    auto flush = map_flush(out, flush_first);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    half_numeric(out, std::get<bool>(flush), slice(raw.word, saturate_bit, 1) != 0);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_hfma2_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::hfma2_32i, Encoding::immediate32);
    destination(out);
    source_a(out);
    source_register(out, RegisterRole::source_c, SourceRole::c, 0);
    out.sources.back().modifiers.scalar = ScalarType::f16;
    immediate_32(out, SourceRole::b);
    out.sources.back().modifiers.scalar = ScalarType::f16;
    half_swizzle(out, SourceRole::a, 47);
    negate(out, SourceRole::c, 51);
    auto flush = map_flush(out, 57);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    half_numeric(out, std::get<bool>(flush));
    // The 32-bit immediate form always packs both halves (OFmt F16).
    add_field(out, DecodedField::half_mode, 0);
    return finish(std::move(out));
}

// HMUL2 route modifiers: the register form negates A at bit 31 (no B negate);
// the immediate and constant forms negate A at bit 43. Saturation sits at
// bit 32 on the register form and bit 52 on the immediate/constant forms.
template <InstructionName Name, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_hmul2_route(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    destination(out);
    source_a(out);
    unsigned saturate_bit = 52;
    if constexpr (Route == Encoding::registers) {
        source_b(out);
        half_swizzle(out, SourceRole::b, 28);
        absolute(out, SourceRole::b, 30);
        negate(out, SourceRole::a, 31);
        half_output_selector(out);
        saturate_bit = 32;
    } else if constexpr (Route == Encoding::immediate) {
        half_immediate(out);
        half_output_selector(out);
    } else {
        constant(out, SourceRole::b);
        absolute(out, SourceRole::b, 54);
        half_output_selector(out);
    }
    if constexpr (Route != Encoding::registers) {
        negate(out, SourceRole::a, 43);
    }
    half_swizzle(out, SourceRole::a, 47);
    absolute(out, SourceRole::a, 44);
    auto flush = map_flush(out, 39);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    half_numeric(out, std::get<bool>(flush), slice(raw.word, saturate_bit, 1) != 0);
    return finish(std::move(out));
}

[[nodiscard]] Result<DecodedInstruction> decode_hmul2_32i(Pc pc, Bits raw) {
    DecodedInstruction out;
    base(out, pc, raw, InstructionName::hmul232i, Encoding::immediate32);
    destination(out);
    source_a(out);
    immediate_32(out, SourceRole::b);
    out.sources.back().modifiers.scalar = ScalarType::f16;
    half_swizzle(out, SourceRole::a, 53);
    auto flush = map_flush(out, 55);
    if (auto* error = std::get_if<Diagnostic>(&flush)) return *error;
    half_numeric(out, std::get<bool>(flush), slice(raw.word, 52, 1) != 0);
    // The 32-bit immediate form always packs both halves (OFmt F16).
    add_field(out, DecodedField::half_mode, 0);
    return finish(std::move(out));
}

// HSET2 writes a boolean word; HSETP2 writes paired predicates and may AND the
// two per-half comparisons. `condition` carries the FComp selector,
// `boolean_operation` the Bop, and `half_mode` the BVal (HSET2) or HAnd
// (HSETP2) selector. FComp and the BVal/HAnd bit move between routes.
template <InstructionName Name, bool ToPredicate, Encoding Route>
[[nodiscard]] Result<DecodedInstruction> decode_hset_route(Pc pc, Bits raw) {
    constexpr bool reg = Route == Encoding::registers;
    DecodedInstruction out;
    base(out, pc, raw, Name, Route);
    if constexpr (ToPredicate) predicate_destinations(out);
    else destination(out);
    source_a(out);
    if constexpr (reg) {
        source_b(out);
        half_swizzle(out, SourceRole::b, 28);
        absolute(out, SourceRole::b, 30);
        negate(out, SourceRole::b, 31);
    } else if constexpr (Route == Encoding::immediate) {
        half_immediate(out);
    } else {
        constant(out, SourceRole::b);
        absolute(out, SourceRole::b, 54);
        negate(out, SourceRole::b, 56);
    }
    half_swizzle(out, SourceRole::a, 47);
    absolute(out, SourceRole::a, 44);
    negate(out, SourceRole::a, 43);
    add_field(out, DecodedField::condition, reg ? slice(raw.word, 35, 4) : slice(raw.word, 49, 4));
    add_field(out, DecodedField::boolean_operation, slice(raw.word, 45, 2));
    add_field(out, DecodedField::half_mode, reg ? slice(raw.word, 49, 1) : slice(raw.word, 53, 1));
    const unsigned flush_bit = ToPredicate ? 6u : (reg ? 50u : 54u);
    half_numeric(out, slice(raw.word, flush_bit, 1) != 0);
    predicate_source(out);
    return finish(std::move(out));
}

// ---------------------------------------------------------------------------
// The rule table is restricted to this lane's families.
// Double-precision (DFMA etc.) rows are absent, so unknown encodings stay a
// structured decode failure; the static_assert proves no two rows overlap.
// ---------------------------------------------------------------------------

constexpr DecodeRule alu_rules[]{
    rule("000010", &decode_fadd_32i),
    rule("000011", &decode_ffma_32i),
    rule("000100", &decode_imad_32i),
    rule("000101", &decode_iadd_32i<InstructionName::iscadd32i>),
    rule("0001110", &decode_iadd_32i<InstructionName::iadd32i>),
    rule("000110", &decode_lea_hi_route<InstructionName::lea_hi, Encoding::constant_buffer, 51, 56>),
    rule("00011110", &decode_fmul_32i),
    rule("00011111", &decode_imul_32i),
    rule("000001", &decode_lop_32i),
    rule("000000010000", &decode_mov_32i),
    rule("0000001x", &decode_lop_route<InstructionName::lop3, Encoding::constant_buffer>),
    rule("0010100", &decode_hfma2_32i),
    rule("0010101", &decode_hmul2_32i),
    rule("0010110", &decode_hadd2_32i),
    rule("0011000x", &decode_fset_route<InstructionName::fset, Encoding::immediate, false>),
    rule("0011001x1", &decode_ffma_route<InstructionName::ffma, Encoding::immediate>),
    rule("0011010x0", &decode_imad_route<InstructionName::imad, Encoding::immediate>),
    rule("0011011x00", &decode_xmad_route<InstructionName::xmad, Encoding::immediate, true>),
    rule("0011011x0100", &decode_icmp_route<InstructionName::icmp, Encoding::immediate, false>),
    rule("0011011x0101", &decode_iset_route<InstructionName::iset, Encoding::immediate, false>),
    rule("0011011x0110", &decode_iset_route<InstructionName::isetp, Encoding::immediate, true>),
    rule("0011011x1010", &decode_fcmp_route<InstructionName::fcmp, Encoding::immediate, false>),
    rule("0011011x1011", &decode_fset_route<InstructionName::fsetp, Encoding::immediate, true>),
    rule("0011011x11010", &decode_lea_route<InstructionName::lea, Encoding::immediate, 39, 41>),
    rule("0011011x11110", &decode_bfi_route<InstructionName::bfi, Encoding::immediate, false>),
    rule("0011011x11111", &decode_shf_route<true, Encoding::immediate>),
    rule("0011100x00000", &decode_bfe_route<InstructionName::bfe, Encoding::immediate>),
    rule("0011100x00001", &decode_popc_route<InstructionName::popc, Encoding::immediate>),
    rule("0011100x00010", &decode_iadd_route<InstructionName::iadd, Encoding::immediate>),
    rule("0011100x00011", &decode_iadd_route<InstructionName::iscadd, Encoding::immediate>),
    rule("0011100x00100", &decode_imnmx_route<InstructionName::imnmx, Encoding::immediate>),
    rule("0011100x00101", &decode_shift_route<InstructionName::shr, Encoding::immediate, true>),
    rule("0011100x00110", &decode_flo_route<InstructionName::flo, Encoding::immediate>),
    rule("0011100x00111", &decode_imul_route<InstructionName::imul, Encoding::immediate>),
    rule("0011100x01000", &decode_lop_route<InstructionName::lop, Encoding::immediate>),
    rule("0011100x01001", &decode_shift_route<InstructionName::shl, Encoding::immediate, false>),
    rule("0011100x01011", &decode_fadd_route<InstructionName::fadd, Encoding::immediate>),
    rule("0011100x01100", &decode_fmnmx_route<InstructionName::fmnmx, Encoding::immediate>),
    rule("0011100x01101", &decode_fmul_route<InstructionName::fmul, Encoding::immediate>),
    rule("0011100x10001", &decode_fchk_route<InstructionName::fchk, Encoding::immediate>),
    rule("0011100x10010", &decode_rro_route<InstructionName::rro, Encoding::immediate>),
    rule("0011100x10011", &decode_mov_route<Encoding::immediate>),
    rule("0011100x10100", &decode_sel_route<Encoding::immediate>),
    rule("0011100x10101", &decode_f2f_route<InstructionName::f2f, Encoding::immediate>),
    rule("0011100x10110", &decode_f2i_route<InstructionName::f2i, Encoding::immediate>),
    rule("0011100x10111", &decode_i2f_route<InstructionName::i2f, Encoding::immediate>),
    rule("0011100x1100", &decode_iadd3_route<Encoding::immediate>),
    rule("0011100x11100", &decode_i2i_route<InstructionName::i2i, Encoding::immediate>),
    rule("0011100x11101", &decode_predicate_move_route<InstructionName::p2r, Encoding::immediate, true>),
    rule("0011100x11110", &decode_predicate_move_route<InstructionName::r2p, Encoding::immediate, false>),
    rule("0011100x11111", &decode_shf_route<false, Encoding::immediate>),
    rule("001111", &decode_lop_route<InstructionName::lop3, Encoding::immediate>),
    rule("01001000", &decode_fset_route<InstructionName::fset, Encoding::constant_buffer, false>),
    rule("010010011", &decode_ffma_route<InstructionName::ffma, Encoding::constant_buffer>),
    rule("010010100", &decode_imad_route<InstructionName::imad, Encoding::constant_buffer>),
    rule("010010110100", &decode_icmp_route<InstructionName::icmp, Encoding::constant_buffer, false>),
    rule("010010110101", &decode_iset_route<InstructionName::iset, Encoding::constant_buffer, false>),
    rule("010010110110", &decode_iset_route<InstructionName::isetp, Encoding::constant_buffer, true>),
    rule("010010111010", &decode_fcmp_route<InstructionName::fcmp, Encoding::constant_buffer, false>),
    rule("010010111011", &decode_fset_route<InstructionName::fsetp, Encoding::constant_buffer, true>),
    rule("0100101111010", &decode_lea_route<InstructionName::lea, Encoding::constant_buffer, 39, 41>),
    rule("0100101111110", &decode_bfi_route<InstructionName::bfi, Encoding::constant_buffer, false>),
    rule("0100110000000", &decode_bfe_route<InstructionName::bfe, Encoding::constant_buffer>),
    rule("0100110000001", &decode_popc_route<InstructionName::popc, Encoding::constant_buffer>),
    rule("0100110000010", &decode_iadd_route<InstructionName::iadd, Encoding::constant_buffer>),
    rule("0100110000011", &decode_iadd_route<InstructionName::iscadd, Encoding::constant_buffer>),
    rule("0100110000100", &decode_imnmx_route<InstructionName::imnmx, Encoding::constant_buffer>),
    rule("0100110000101", &decode_shift_route<InstructionName::shr, Encoding::constant_buffer, true>),
    rule("0100110000110", &decode_flo_route<InstructionName::flo, Encoding::constant_buffer>),
    rule("0100110000111", &decode_imul_route<InstructionName::imul, Encoding::constant_buffer>),
    rule("0100110001000", &decode_lop_route<InstructionName::lop, Encoding::constant_buffer>),
    rule("0100110001001", &decode_shift_route<InstructionName::shl, Encoding::constant_buffer, false>),
    rule("0100110001011", &decode_fadd_route<InstructionName::fadd, Encoding::constant_buffer>),
    rule("0100110001100", &decode_fmnmx_route<InstructionName::fmnmx, Encoding::constant_buffer>),
    rule("0100110001101", &decode_fmul_route<InstructionName::fmul, Encoding::constant_buffer>),
    rule("0100110010001", &decode_fchk_route<InstructionName::fchk, Encoding::constant_buffer>),
    rule("0100110010010", &decode_rro_route<InstructionName::rro, Encoding::constant_buffer>),
    rule("0100110010011", &decode_mov_route<Encoding::constant_buffer>),
    rule("0100110010100", &decode_sel_route<Encoding::constant_buffer>),
    rule("0100110010101", &decode_f2f_route<InstructionName::f2f, Encoding::constant_buffer>),
    rule("0100110010110", &decode_f2i_route<InstructionName::f2i, Encoding::constant_buffer>),
    rule("0100110010111", &decode_i2f_route<InstructionName::i2f, Encoding::constant_buffer>),
    rule("010011001100", &decode_iadd3_route<Encoding::constant_buffer>),
    rule("0100110011100", &decode_i2i_route<InstructionName::i2i, Encoding::constant_buffer>),
    rule("0100110011101", &decode_predicate_move_route<InstructionName::p2r, Encoding::constant_buffer, true>),
    rule("0100110011110", &decode_predicate_move_route<InstructionName::r2p, Encoding::constant_buffer, false>),
    rule("0100111", &decode_xmad_route<InstructionName::xmad, Encoding::constant_buffer, true>),
    rule("0101000010000", &decode_mufu),
    rule("0101000010001", &decode_pset),
    rule("0101000010010", &decode_psetp),
    rule("0101000010011", &decode_cset),
    rule("0101000010100", &decode_csetp),
    rule("0101000011001", &decode_cs2r),
    rule("0101000011111", &decode_fswzadd),
    rule("010100010", &decode_xmad_rc),
    rule("010100011", &decode_ffma_route<InstructionName::ffma, Encoding::constant_buffer, true>),
    rule("010100100", &decode_imad_route<InstructionName::imad, Encoding::constant_buffer, true>),
    rule("010100110100", &decode_icmp_route<InstructionName::icmp, Encoding::constant_buffer, true>),
    rule("010100111010", &decode_fcmp_route<InstructionName::fcmp, Encoding::constant_buffer, true>),
    rule("0101001111110", &decode_bfi_route<InstructionName::bfi, Encoding::constant_buffer, true>),
    rule("01011000", &decode_fset_route<InstructionName::fset, Encoding::registers, false>),
    rule("010110011", &decode_ffma_route<InstructionName::ffma, Encoding::registers>),
    rule("010110100", &decode_imad_route<InstructionName::imad, Encoding::registers>),
    rule("0101101100", &decode_xmad_route<InstructionName::xmad, Encoding::registers, true>),
    rule("010110110100", &decode_icmp_route<InstructionName::icmp, Encoding::registers, false>),
    rule("010110110101", &decode_iset_route<InstructionName::iset, Encoding::registers, false>),
    rule("010110110110", &decode_iset_route<InstructionName::isetp, Encoding::registers, true>),
    rule("010110111010", &decode_fcmp_route<InstructionName::fcmp, Encoding::registers, false>),
    rule("010110111011", &decode_fset_route<InstructionName::fsetp, Encoding::registers, true>),
    rule("0101101111010", &decode_lea_route<InstructionName::lea, Encoding::registers, 39, 41>),
    rule("0101101111011", &decode_lea_hi_route<InstructionName::lea_hi, Encoding::registers, 28, 29>),
    rule("0101101111100", &decode_lop_route<InstructionName::lop3, Encoding::registers>),
    rule("0101101111110", &decode_bfi_route<InstructionName::bfi, Encoding::registers, false>),
    rule("0101101111111", &decode_shf_route<true, Encoding::registers>),
    rule("1110111100010", &decode_shfl),
    rule("0101110000000", &decode_bfe_route<InstructionName::bfe, Encoding::registers>),
    rule("0101110000001", &decode_popc_route<InstructionName::popc, Encoding::registers>),
    rule("0101110000010", &decode_iadd_route<InstructionName::iadd, Encoding::registers>),
    rule("0101110000011", &decode_iadd_route<InstructionName::iscadd, Encoding::registers>),
    rule("0101110000100", &decode_imnmx_route<InstructionName::imnmx, Encoding::registers>),
    rule("0101110000101", &decode_shift_route<InstructionName::shr, Encoding::registers, true>),
    rule("0101110000110", &decode_flo_route<InstructionName::flo, Encoding::registers>),
    rule("0101110000111", &decode_imul_route<InstructionName::imul, Encoding::registers>),
    rule("0101110001000", &decode_lop_route<InstructionName::lop, Encoding::registers>),
    rule("0101110001001", &decode_shift_route<InstructionName::shl, Encoding::registers, false>),
    rule("0101110001011", &decode_fadd_route<InstructionName::fadd, Encoding::registers>),
    rule("0101110001100", &decode_fmnmx_route<InstructionName::fmnmx, Encoding::registers>),
    rule("0101110001101", &decode_fmul_route<InstructionName::fmul, Encoding::registers>),
    rule("0101110010001", &decode_fchk_route<InstructionName::fchk, Encoding::registers>),
    rule("0101110010010", &decode_rro_route<InstructionName::rro, Encoding::registers>),
    rule("0101110010011", &decode_mov_route<Encoding::registers>),
    rule("0101110010100", &decode_sel_route<Encoding::registers>),
    rule("0101110010101", &decode_f2f_route<InstructionName::f2f, Encoding::registers>),
    rule("0101110010110", &decode_f2i_route<InstructionName::f2i, Encoding::registers>),
    rule("0101110010111", &decode_i2f_route<InstructionName::i2f, Encoding::registers>),
    rule("010111001100", &decode_iadd3_route<Encoding::registers>),
    rule("0101110011100", &decode_i2i_route<InstructionName::i2i, Encoding::registers>),
    rule("0101110011101", &decode_predicate_move_route<InstructionName::p2r, Encoding::registers, true>),
    rule("0101110011110", &decode_predicate_move_route<InstructionName::r2p, Encoding::registers, false>),
    rule("0101110011111", &decode_shf_route<false, Encoding::registers>),
    rule("0101110100000", &decode_hfma2_route<Encoding::registers>),
    rule("0101110100001", &decode_hmul2_route<InstructionName::hmul2, Encoding::registers>),
    rule("0101110100010", &decode_hadd2_route<InstructionName::hadd2, Encoding::registers>),
    rule("0101110100011", &decode_hset_route<InstructionName::hset2, false, Encoding::registers>),
    rule("0101110100100", &decode_hset_route<InstructionName::hsetp2, true, Encoding::registers>),
    rule("01100xxx1", &decode_hfma2_route<Encoding::constant_buffer, true>),
    rule("01110xxx0", &decode_hfma2_route<Encoding::immediate>),
    rule("01110xxx1", &decode_hfma2_route<Encoding::constant_buffer>),
    rule("0111100x0", &decode_hmul2_route<InstructionName::hmul2, Encoding::immediate>),
    rule("0111100x1", &decode_hmul2_route<InstructionName::hmul2, Encoding::constant_buffer>),
    rule("0111101x0", &decode_hadd2_route<InstructionName::hadd2, Encoding::immediate>),
    rule("0111101x1", &decode_hadd2_route<InstructionName::hadd2, Encoding::constant_buffer>),
    rule("0111110x0", &decode_hset_route<InstructionName::hset2, false, Encoding::immediate>),
    rule("0111110x1", &decode_hset_route<InstructionName::hset2, false, Encoding::constant_buffer>),
    rule("0111111x0", &decode_hset_route<InstructionName::hsetp2, true, Encoding::immediate>),
    rule("0111111x1", &decode_hset_route<InstructionName::hsetp2, true, Encoding::constant_buffer>),
    rule("1111000011001", &decode_s2r),
};

static_assert([] {
    for (std::size_t i = 0; i < std::size(alu_rules); ++i) {
        for (std::size_t j = i + 1; j < std::size(alu_rules); ++j) {
            const auto& left = alu_rules[i];
            const auto& right = alu_rules[j];
            if (((left.mask & right.mask) & (left.value ^ right.value)) == 0) return false;
        }
    }
    return true;
}(), "ALU decode rules must not overlap");
} // namespace

[[nodiscard]] std::span<const DecodeRule> alu_decode_rules() { return alu_rules; }
} // namespace armigo::shader::pipeline
