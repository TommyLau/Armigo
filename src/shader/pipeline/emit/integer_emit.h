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

#pragma once
#include "pipeline/emitter_context.h"

#include <array>

namespace armigo::shader::pipeline::integer_emit {

// Result<T> is a variant; this unwraps the success alternative after the
// caller has checked for a diagnostic.
template <class Variant>
[[nodiscard]] constexpr auto& value(Variant& result) noexcept {
    return std::get<std::variant_alternative_t<0, std::remove_reference_t<Variant>>>(result);
}
template <class Variant>
[[nodiscard]] constexpr const auto& value(const Variant& result) noexcept {
    return std::get<std::variant_alternative_t<0, std::remove_reference_t<Variant>>>(result);
}
[[nodiscard]] inline bool failed(const Status& status) noexcept {
    return std::holds_alternative<Diagnostic>(status);
}

// Shared helpers for the integer/move/predicate/condition-code emitter
// families (lane 6). Callbacks consume decoder-normalized operands through
// EmitterContext plus the shared DecodedField keys listed here; they never
// re-derive guest operand layouts from the raw word.
//
// DecodedField value conventions accepted by these families (one entry per
// field, published by the decode lanes for these instructions):
//   write_cc           bit 0: guest WriteCC flag.
//   extended           bit 0: guest X (extended/carry) flag.
//   signedness         bit 0: signed interpretation; IMAD/XMAD use bit 0 for
//                      source A and bit 1 for source B.
//   boolean_operation  bits 1..0: BoolOp for ISET/ISETP/CSET/CSETP/PSET/PSETP
//                      (and=0, or=1, xor=2, 3 passes the left input through,
//                      matching LOP PassB) and LogicOp for LOP/LOP32I
//                      (and=0, or=1, xor=2, pass_b=3). PSET/PSETP store their
//                      second selector in bits 3..2 (BoolOpC).
//   condition          bits 2..0: IComp for ICMP/ISET/ISETP (f=0, lt=1, eq=2,
//                      le=3, gt=4, ne=5, ge=6, t=7); bits 1..0 PredicateOp
//                      for LOP/LOP3 predicate writes (f=0, t=1, z=2, nz=3);
//                      bits 4..0: Ccc for CSET/CSETP.
//   permutation_mode   bits 7..0: LOP3 truth table; bit n selects the output
//                      for inputs a=(n>>2)&1, b=(n>>1)&1, c=n&1.
//   half_mode          IADD3: bits 1..0/3..2/5..4 are the A/B/C HalfSelect
//                      (b32=0, h0=1, h1=2) and bits 7..6 the Lrs mode (none=0,
//                      right=1, left=2). IMAD/IMAD32I: bit 0 is Hilo. XMAD:
//                      bit 0 HiloA, bit 1 HiloB, bit 2 PSL, bit 3 Mrg.
//   function           bits 1..0: AvgMode for IADD/IADD32I/IMAD/IMAD32I/
//                      ISCADD/ISCADD32I (no_neg=0, neg_b_slot=1, neg_a_slot=2,
//                      plus_one=3) and bit 2 the IADD/IMAD saturation flag.
//                      bits 2..0: XMAD Ccop (cfull=0, clo=1, chi=2, csfu=3,
//                      cbcc=4). bits 6..0: S2R special register index.
//   shift_mode         bits 4..0: LEA/LEA_HI immediate shift and ISCADD
//                      Imm5 scale; SHF: bits 1..0 MaxShift (u32=0, s32=1,
//                      u64=2, s64=3) and bit 2 direction (0 = left).
//   shift_wrap         bit 0: SHF/SHL/SHR wrap (guest M) flag.
//   bitfield_mode      bit 0: SHR/BFE bit-reverse or FLO search-from-LSB;
//                      R2P/P2R: bits 1..0 ByteSel and bit 2 CCPR flag access.
//   destination_type   bit 0: CSET/PSET BVal (destination receives the
//                      floating 0.0/1.0 boolean encoding).
//
// Source conventions: sources a/b/c carry the operand route (register,
// literal, or constant-buffer word). modifiers.negate is the family inversion
// flag for that operand: word inversion for LOP/LOP3/FLO/POPC, arithmetic
// negation for IADD3/LEA/LEA_HI, and logical inversion for predicate sources.
// ISET/ISETP/IMNMX/SEL/CSET/CSETP publish their single predicate input as
// source role `predicate`; PSET/PSETP publish Src1Pred/Src2Pred/SrcPred as
// roles a/b/c. RC encodings publish register C as source b and the constant
// buffer as source c; IMAD32I publishes its destination read as source c.
// IADD/ISCADD/IMAD publish AvgMode through `function`; the emitter maps the
// NegA/NegB slots, including IMAD negating its
// B and C slots from NegA/NegB.

inline constexpr NumericMode bits32{Domain::bits, ScalarType::u32};
inline constexpr NumericMode word32_unsigned{Domain::unsigned_integer, ScalarType::u32};
inline constexpr NumericMode word32_signed{Domain::signed_integer, ScalarType::s32};
inline constexpr NumericMode boolean_mode{Domain::boolean, ScalarType::boolean};
// Floating compares round nothing; the frozen contract still requires an
// explicit mode, so the comparison records the guest's nearest-even default.
inline constexpr NumericMode float32_compare{Domain::floating, ScalarType::f32, Rounding::nearest_even};

inline constexpr LiteralBits word_zero{std::uint32_t{0}, ValueType::word32};
inline constexpr LiteralBits word_all_ones{std::uint32_t{0xffffffff}, ValueType::word32};
inline constexpr LiteralBits predicate_true{std::uint64_t{1}, ValueType::predicate};
inline constexpr LiteralBits predicate_false{std::uint64_t{0}, ValueType::predicate};
inline constexpr LiteralBits float_one{std::uint32_t{0x3f800000}, ValueType::word32}; // 1.0f bits.
inline constexpr LiteralBits word_count(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }

inline constexpr Register cc_register(ConditionCode code) {
    return Register{RegisterFile::condition_code, static_cast<std::uint16_t>(code)};
}

[[nodiscard]] inline Result<std::uint64_t> field(const DecodedInstruction& instruction, DecodedField key) {
    return decoded_field(instruction, key);
}
[[nodiscard]] inline bool field_flag(const DecodedInstruction& instruction, DecodedField key, unsigned bit = 0) {
    auto value = decoded_field(instruction, key);
    if (const auto* bits = std::get_if<std::uint64_t>(&value)) return ((*bits >> bit) & 1) != 0;
    return false;
}

// Materializes a decoded source operand. Reads preserve bits; RZ/PT resolve to
// literals inside the context. The ConstantWord operand defers constant-buffer
// reads to resource planning.
[[nodiscard]] inline Result<Operand> source_value(EmitterContext& context, SourceRole role) {
    auto decoded = decoded_source(context.instruction(), role);
    if (const auto* error = std::get_if<Diagnostic>(&decoded)) return *error;
    const auto& source = std::get<DecodedSource>(decoded);
    if (const auto* reg = std::get_if<Register>(&source.value)) return context.read(*reg);
    if (const auto* literal = std::get_if<LiteralBits>(&source.value)) return Operand{*literal};
    if (const auto* word = std::get_if<ConstantWord>(&source.value)) return Operand{*word};
    return context.error(DiagnosticCode::malformed_input, "Decoded source has no operand value");
}
[[nodiscard]] inline bool source_inverted(EmitterContext& context, SourceRole role) {
    auto decoded = decoded_source(context.instruction(), role);
    if (const auto* source = std::get_if<DecodedSource>(&decoded)) return source->modifiers.negate;
    return false;
}

[[nodiscard]] inline Result<Register> destination_register(EmitterContext& context) {
    for (const auto& access : context.instruction().registers) {
        if (access.role == RegisterRole::destination && access.lane == 0 &&
            access.access == Access::write && access.reg.file == RegisterFile::gpr) return access.reg;
    }
    return context.error(DiagnosticCode::malformed_input, "Instruction has no decoded destination register");
}
[[nodiscard]] inline Result<Register> role_register(EmitterContext& context, RegisterRole role) {
    for (const auto& access : context.instruction().registers) {
        if (access.role == role && access.lane == 0) return access.reg;
    }
    return context.error(DiagnosticCode::malformed_input, "Instruction has no decoded register for a required role");
}
// Non-erroring variant for optional register roles.
[[nodiscard]] inline Result<Register> find_role_register(const EmitterContext& context, RegisterRole role) {
    for (const auto& access : context.instruction().registers) {
        if (access.role == role && access.lane == 0) return access.reg;
    }
    return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
                    {context.instruction().pc, context.instruction().name},
                    "Instruction has no decoded register for a required role");
}

[[nodiscard]] inline Result<Operand> read_cc(EmitterContext& context, ConditionCode code) {
    return context.read(cc_register(code));
}
[[nodiscard]] inline Status write_cc(EmitterContext& context, ConditionCode code, Operand value) {
    return context.write(cc_register(code), value);
}

// Reads the single guest predicate input; modifiers.negate inverts it.
[[nodiscard]] inline Result<Operand> predicate_source(EmitterContext& context) {
    auto decoded = decoded_source(context.instruction(), SourceRole::predicate);
    if (const auto* error = std::get_if<Diagnostic>(&decoded)) return *error;
    const auto& source = std::get<DecodedSource>(decoded);
    const auto* reg = std::get_if<Register>(&source.value);
    if (!reg || reg->file != RegisterFile::predicate) {
        return context.error(DiagnosticCode::malformed_input, "Predicate source is not a decoded predicate register");
    }
    auto flag = context.read(*reg);
    if (const auto* error = std::get_if<Diagnostic>(&flag)) return *error;
    if (source.modifiers.negate) {
        auto inverted = context.unary(Opcode::logical_not, value(flag), boolean_mode, ValueType::predicate);
        if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
        return inverted;
    }
    return flag;
}
[[nodiscard]] inline Result<Operand> invert_predicate(EmitterContext& context, Operand value, bool invert) {
    if (!invert) return value;
    return context.unary(Opcode::logical_not, value, boolean_mode, ValueType::predicate);
}

[[nodiscard]] inline Result<Operand> invert_word(EmitterContext& context, Operand value, bool invert) {
    if (!invert) return value;
    return context.unary(Opcode::bit_not, value, bits32, ValueType::word32);
}
[[nodiscard]] inline Result<Operand> negate_word(EmitterContext& context, Operand value, bool negate) {
    if (!negate) return value;
    return context.unary(Opcode::negate, value, word32_signed, ValueType::word32);
}

[[nodiscard]] inline Result<Operand> add(EmitterContext& context, Operand left, Operand right) {
    return context.binary(Opcode::add, left, right, word32_unsigned, ValueType::word32);
}
[[nodiscard]] inline Result<Operand> subtract(EmitterContext& context, Operand left, Operand right) {
    return context.binary(Opcode::subtract, left, right, word32_unsigned, ValueType::word32);
}
[[nodiscard]] inline Result<Operand> shift_left(EmitterContext& context, Operand value, std::uint32_t count) {
    return context.binary(Opcode::shift_left, value, word_count(count), bits32, ValueType::word32);
}
[[nodiscard]] inline Result<Operand> compare(EmitterContext& context, Operand left, Operand right,
                                             Compare relation, bool is_signed) {
    return context.compare(left, right, ComparisonInfo{relation, false}, is_signed ? word32_signed : word32_unsigned);
}
// Boolean 0/1 word from a predicate. Flag words are masked with
// one; the conversion already guarantees that shape.
[[nodiscard]] inline Result<Operand> predicate_word(EmitterContext& context, Operand value) {
    return context.convert(value, ConversionInfo{ScalarType::boolean, ScalarType::u32}, word32_unsigned, ValueType::word32);
}
// All-ones/zero word from a predicate, used by compare-and-set instructions.
[[nodiscard]] inline Result<Operand> predicate_mask(EmitterContext& context, Operand value) {
    return context.unary(Opcode::predicate_to_mask, value, NumericMode{}, ValueType::word32);
}
[[nodiscard]] inline Result<Operand> logical(EmitterContext& context, Opcode opcode, Operand left, Operand right) {
    return context.binary(opcode, left, right, boolean_mode, ValueType::predicate);
}

// BoolOp combination of a comparison or predicate result with a guest
// predicate; value 3 passes the input through, matching LOP PassB and the
// invalid BoolOp fallthrough.
[[nodiscard]] inline Result<Operand> pred_logical(EmitterContext& context, std::uint64_t operation,
                                                  Operand input, Operand predicate) {
    switch (operation & 3) {
    case 0: return logical(context, Opcode::logical_and, input, predicate);
    case 1: return logical(context, Opcode::logical_or, input, predicate);
    case 2: return logical(context, Opcode::logical_xor, input, predicate);
    default: return input;
    }
}

// ZF/NF writes; the extended form ANDs the zero test with the incoming flag so
// only a fully zero long word sets ZF.
[[nodiscard]] inline Status set_zn_flags(EmitterContext& context, Operand result, bool set_cc, bool extended) {
    if (!set_cc) return std::monostate{};
    auto zero = compare(context, result, word_zero, Compare::equal, false);
    if (const auto* error = std::get_if<Diagnostic>(&zero)) return *error;
    if (extended) {
        auto old_zero = read_cc(context, ConditionCode::zero);
        if (const auto* error = std::get_if<Diagnostic>(&old_zero)) return *error;
        auto combined = logical(context, Opcode::logical_and, value(zero), value(old_zero));
        if (const auto* error = std::get_if<Diagnostic>(&combined)) return *error;
        if (auto status = write_cc(context, ConditionCode::zero, value(combined)); failed(status)) return status;
    } else if (auto status = write_cc(context, ConditionCode::zero, value(zero)); failed(status)) {
        return status;
    }
    auto negative = compare(context, result, word_zero, Compare::less, true);
    if (const auto* error = std::get_if<Diagnostic>(&negative)) return *error;
    return write_cc(context, ConditionCode::sign, value(negative));
}

// C/V/Z/N from an integer addition.
[[nodiscard]] inline Status set_iadd_flags(EmitterContext& context, Operand result, Operand left,
                                           Operand right, bool set_cc, bool extended) {
    if (!set_cc) return std::monostate{};
    if (extended) {
        // C = (d == a && CIn) || d < a (unsigned).
        auto equal = compare(context, result, left, Compare::equal, false);
        if (const auto* error = std::get_if<Diagnostic>(&equal)) return *error;
        auto carry = read_cc(context, ConditionCode::carry);
        if (const auto* error = std::get_if<Diagnostic>(&carry)) return *error;
        auto carried = logical(context, Opcode::logical_and, value(equal), value(carry));
        if (const auto* error = std::get_if<Diagnostic>(&carried)) return *error;
        auto below = compare(context, result, left, Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&below)) return *error;
        auto flag = logical(context, Opcode::logical_or, value(carried), value(below));
        if (const auto* error = std::get_if<Diagnostic>(&flag)) return *error;
        if (auto status = write_cc(context, ConditionCode::carry, value(flag)); failed(status)) return status;
    } else {
        // C = d < a (unsigned).
        auto below = compare(context, result, left, Compare::less, false);
        if (const auto* error = std::get_if<Diagnostic>(&below)) return *error;
        if (auto status = write_cc(context, ConditionCode::carry, value(below)); failed(status)) return status;
    }
    // V = ((d ^ a) & ~(a ^ b)) < 0 (signed).
    auto left_xor = context.binary(Opcode::bit_xor, result, left, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&left_xor)) return *error;
    auto right_xor = context.binary(Opcode::bit_xor, left, right, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&right_xor)) return *error;
    auto inverted = context.unary(Opcode::bit_not, value(right_xor), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto mixed = context.binary(Opcode::bit_and, value(left_xor), value(inverted), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&mixed)) return *error;
    auto overflow = compare(context, value(mixed), word_zero, Compare::less, true);
    if (const auto* error = std::get_if<Diagnostic>(&overflow)) return *error;
    if (auto status = write_cc(context, ConditionCode::overflow, value(overflow)); failed(status)) return status;
    return set_zn_flags(context, result, set_cc, extended);
}

// AddWithCarry: returns the sum and reports the unsigned borrow as a
// predicate. The borrow is negated into an all-ones mask; the typed
// contract instead converts to 0/1 and callers subtract it from the high word.
[[nodiscard]] inline Result<Operand> add_with_borrow(EmitterContext& context, Operand left, Operand right,
                                                     Operand& borrow) {
    auto sum = add(context, left, right);
    if (const auto* error = std::get_if<Diagnostic>(&sum)) return *error;
    auto below = compare(context, value(sum), left, Compare::less, false);
    if (const auto* error = std::get_if<Diagnostic>(&below)) return *error;
    borrow = value(below);
    return sum;
}

// NegateLong: 64-bit two's-complement negation of a low/high word pair.
[[nodiscard]] inline Status negate_long(EmitterContext& context, Operand& low, Operand& high) {
    auto low_inverted = context.unary(Opcode::bit_not, low, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&low_inverted)) return *error;
    auto high_inverted = context.unary(Opcode::bit_not, high, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&high_inverted)) return *error;
    auto incremented = add(context, value(low_inverted), word_count(1));
    if (const auto* error = std::get_if<Diagnostic>(&incremented)) return *error;
    auto borrow = compare(context, value(incremented), value(low_inverted), Compare::less, false);
    if (const auto* error = std::get_if<Diagnostic>(&borrow)) return *error;
    auto borrow_word = predicate_word(context, value(borrow));
    if (const auto* error = std::get_if<Diagnostic>(&borrow_word)) return *error;
    auto adjusted = subtract(context, value(high_inverted), value(borrow_word));
    if (const auto* error = std::get_if<Diagnostic>(&adjusted)) return *error;
    low = value(incremented);
    high = value(adjusted);
    return std::monostate{};
}

namespace detail {
// Shannon expansion of the truth table over the three inputs. Every 8-bit
// table lowers with at most one bit_and, bit_not and bit_or per split, so the
// expansion is total and deterministic where a pattern search can
// fail to match.
[[nodiscard]] inline Result<Operand> expand_lut(EmitterContext& context, std::uint32_t table,
                                                const std::array<Operand, 3>& inputs, unsigned remaining) {
    const auto mask = (std::uint32_t{1} << (1u << remaining)) - 1;
    table &= mask;
    if (table == 0) return Operand{word_zero};
    if (table == mask) return Operand{word_all_ones};
    const Operand variable = inputs[3 - remaining];
    if (remaining == 1) {
        // Two-row table over the last input: identity or inversion.
        if (table == 1) return context.unary(Opcode::bit_not, variable, bits32, ValueType::word32);
        return variable;
    }
    std::uint32_t when_set = 0, when_clear = 0;
    const auto rows = std::uint32_t{1} << (remaining - 1);
    for (std::uint32_t row = 0; row < rows; ++row) {
        when_set |= ((table >> (2 * row + 1)) & 1) << row;
        when_clear |= ((table >> (2 * row)) & 1) << row;
    }
    auto selected = expand_lut(context, when_set, inputs, remaining - 1);
    if (const auto* error = std::get_if<Diagnostic>(&selected)) return *error;
    auto cleared = expand_lut(context, when_clear, inputs, remaining - 1);
    if (const auto* error = std::get_if<Diagnostic>(&cleared)) return *error;
    auto set_bits = context.binary(Opcode::bit_and, variable, value(selected), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&set_bits)) return *error;
    auto inverted = context.unary(Opcode::bit_not, variable, bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
    auto cleared_bits = context.binary(Opcode::bit_and, value(inverted), value(cleared), bits32, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&cleared_bits)) return *error;
    return context.binary(Opcode::bit_or, value(set_bits), value(cleared_bits), bits32, ValueType::word32);
}
} // namespace detail

// LOP3 truth-table expansion. Builds the guest function directly by Shannon
// expansion instead of matching expression patterns, so every 8-bit
// table lowers with bounded, deterministic work.
[[nodiscard]] inline Result<Operand> lop3_expression(EmitterContext& context, Operand a, Operand b, Operand c,
                                                     std::uint8_t table) {
    // Shannon expansion consumes the first input as truth-table bit 0. The
    // guest index is (A << 2) | (B << 1) | C, so expand C, B, then A.
    return detail::expand_lut(context, table, {std::move(c), std::move(b), std::move(a)}, 3);
}

// Predicate write value shared by LOP/LOP3.
[[nodiscard]] inline Result<Operand> predicate_result(EmitterContext& context, Operand result,
                                                      std::uint64_t predicate_op) {
    switch (predicate_op & 3) {
    case 0: return Operand{predicate_false};
    case 1: return Operand{predicate_true};
    case 2: return compare(context, result, word_zero, Compare::equal, false);
    default: return compare(context, result, word_zero, Compare::not_equal, false);
    }
}
// Absence of a decoded predicate destination (LOP32I, immediate/cbuf LOP3)
// leaves the flag untouched because PT is a fixed destination.
[[nodiscard]] inline Status emit_lop_pred_write(EmitterContext& context, Operand result, std::uint64_t predicate_op) {
    auto destination = find_role_register(context, RegisterRole::predicate_destination);
    if (std::holds_alternative<Diagnostic>(destination)) return std::monostate{};
    if (value(destination).file != RegisterFile::predicate) {
        return context.error(DiagnosticCode::malformed_input, "LOP predicate destination is not a predicate register");
    }
    auto flag_value = predicate_result(context, result, predicate_op);
    if (const auto* error = std::get_if<Diagnostic>(&flag_value)) return *error;
    return context.write(value(destination), value(flag_value));
}
} // namespace armigo::shader::pipeline::integer_emit
