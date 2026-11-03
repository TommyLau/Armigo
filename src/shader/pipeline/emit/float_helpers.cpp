// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline::float_emit {

Result<NumericMode> float_mode(const EmitterContext& context) {
    const auto& instruction = context.instruction();
    if (!instruction.numeric) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Instruction has no decoded numeric mode");
    }
    if (instruction.numeric->domain != Domain::floating) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Floating instruction decoded without a floating numeric mode");
    }
    if (instruction.numeric->rounding == Rounding::not_applicable) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Floating instruction decoded without an explicit rounding mode");
    }
    return *instruction.numeric;
}

NumericMode without_saturation(const NumericMode& mode) {
    auto result = mode;
    result.saturate = false;
    return result;
}

NumericMode comparison_mode(const NumericMode& mode) {
    auto result = without_saturation(mode);
    result.flush_output = false;
    return result;
}

NumericMode plain_float_mode(const NumericMode& mode) {
    auto result = comparison_mode(mode);
    result.flush_inputs = false;
    return result;
}

Result<Operand> source_operand(EmitterContext& context, const DecodedInstruction& instruction,
                               SourceRole role) {
    auto decoded = decoded_source(instruction, role);
    if (decoded.index() == 1) return std::move(std::get<1>(decoded));
    const auto& source = std::get<0>(decoded);
    return std::visit(
        [&context](const auto& value) -> Result<Operand> {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, Register>) {
                return context.read(value);
            } else {
                return Operand{value};
            }
        },
        source.value);
}

Result<Operand> apply_source_modifiers(EmitterContext& context, Operand value,
                                       const SourceModifiers& modifiers, const NumericMode& mode) {
    // Sign bits change before the consuming operation flushes its inputs.
    // These operations neither round nor own an arithmetic boundary.
    // The frozen validator requires an explicit rounding tag even on these
    // exact bit operations; nearest-even does not perform any rounding here.
    const NumericMode sign_mode{mode.domain, mode.scalar,
        mode.domain == Domain::floating ? Rounding::nearest_even : Rounding::not_applicable};
    if (modifiers.absolute) {
        auto absolute = context.unary(Opcode::absolute, value, sign_mode,
                                      storage_type(mode.scalar));
        if (absolute.index() == 1) return std::move(std::get<1>(absolute));
        value = std::get<0>(absolute);
    }
    if (modifiers.negate) {
        auto negated = context.unary(Opcode::negate, value, sign_mode,
                                     storage_type(mode.scalar));
        if (negated.index() == 1) return std::move(std::get<1>(negated));
        value = std::get<0>(negated);
    }
    return value;
}

Result<Operand> float64_source(EmitterContext& context, const DecodedSource& source) {
    return std::visit(
        [&context](const auto& value) -> Result<Operand> {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, Register>) {
                auto low = context.read(value);
                if (low.index() == 1) return std::get<1>(low);
                auto high = context.read(Register{RegisterFile::gpr,
                                                  static_cast<std::uint16_t>(value.index + 1)});
                if (high.index() == 1) return std::get<1>(high);
                return context.binary(Opcode::pack_word64, std::get<0>(low), std::get<0>(high),
                                      {Domain::bits, ScalarType::u32}, ValueType::word64);
            } else if constexpr (std::is_same_v<Value, ConstantWord>) {
                const auto high =
                    ConstantWord{value.slot, value.byte_offset + std::uint32_t{sizeof(std::uint32_t)}};
                return context.binary(Opcode::pack_word64, Operand{value}, Operand{high},
                                      {Domain::bits, ScalarType::u32}, ValueType::word64);
            } else {
                // Immediate doubles arrive as the low word; the high word is zero.
                return context.binary(Opcode::pack_word64, Operand{value},
                                      word(float_zero_bits),
                                      {Domain::bits, ScalarType::u32}, ValueType::word64);
            }
        },
        source.value);
}

Result<Operand> modified_source(EmitterContext& context, const DecodedInstruction& instruction,
                                SourceRole role, const NumericMode& mode) {
    auto decoded = decoded_source(instruction, role);
    if (decoded.index() == 1) return std::move(std::get<1>(decoded));
    const auto& source = std::get<0>(decoded);
    auto value = std::visit(
        [&context](const auto& inner) -> Result<Operand> {
            using Value = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<Value, Register>) {
                return context.read(inner);
            } else {
                return Operand{inner};
            }
        },
        source.value);
    if (value.index() == 1) return std::move(std::get<1>(value));
    return apply_source_modifiers(context, std::get<0>(value), source.modifiers, mode);
}

Result<SourceModifiers> source_modifiers(const DecodedInstruction& instruction, SourceRole role) {
    auto decoded = decoded_source(instruction, role);
    if (decoded.index() == 1) return std::move(std::get<1>(decoded));
    return std::get<0>(decoded).modifiers;
}

Result<bool> write_cc_enabled(EmitterContext& context) {
    auto field = decoded_field(context.instruction(), DecodedField::write_cc);
    if (field.index() == 1) return std::move(std::get<1>(field));
    return std::get<0>(field) != 0;
}

Result<Operand> predicate_operand(EmitterContext& context, const DecodedInstruction& instruction) {
    auto decoded = decoded_source(instruction, SourceRole::predicate);
    if (decoded.index() == 1) return std::move(std::get<1>(decoded));
    const auto& source = std::get<0>(decoded);
    if (!std::holds_alternative<Register>(source.value)) {
        return context.error(DiagnosticCode::malformed_input,
                             "Predicate source did not decode to a predicate register");
    }
    auto read = context.read(std::get<Register>(source.value));
    if (read.index() == 1) return std::move(std::get<1>(read));
    Operand value = std::get<0>(read);
    if (source.modifiers.negate) {
        auto inverted = context.unary(Opcode::logical_not, value,
                                      {Domain::boolean, ScalarType::boolean}, ValueType::predicate);
        if (inverted.index() == 1) return std::move(std::get<1>(inverted));
        value = std::get<0>(inverted);
    }
    return value;
}

Result<Register> destination_register(const DecodedInstruction& instruction) {
    for (const auto& access : instruction.registers) {
        if (access.role == RegisterRole::destination) return access.reg;
    }
    return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
                    {instruction.pc, instruction.name}, "Instruction has no destination register");
}

Result<Register> second_destination_register(const DecodedInstruction& instruction) {
    for (const auto& access : instruction.registers) {
        if (access.role == RegisterRole::second_destination) return access.reg;
    }
    return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
                    {instruction.pc, instruction.name},
                    "Instruction has no second destination register");
}

Status write_destination(EmitterContext& context, const DecodedInstruction& instruction,
                         Operand value) {
    auto destination = destination_register(instruction);
    if (destination.index() == 1) return std::move(std::get<1>(destination));
    return context.write(std::get<0>(destination), value);
}

Status write_float_flags(EmitterContext& context, Operand value, const NumericMode& mode,
                         bool write_cc) {
    if (!write_cc) return std::monostate{};
    const auto flags_mode = comparison_mode(mode);
    const ValueType storage = storage_type(mode.scalar);
    auto equal = context.compare(value, word(float_zero_bits, storage),
                                 ComparisonInfo{Compare::equal, false}, flags_mode);
    if (equal.index() == 1) return std::move(std::get<1>(equal));
    auto written = context.write(Register{RegisterFile::condition_code,
                                          static_cast<std::uint16_t>(ConditionCode::zero)},
                                 std::get<0>(equal));
    if (written.index() == 1) return written;
    auto less = context.compare(value, word(float_zero_bits, storage),
                                ComparisonInfo{Compare::less, false}, flags_mode);
    if (less.index() == 1) return std::move(std::get<1>(less));
    return context.write(Register{RegisterFile::condition_code,
                                  static_cast<std::uint16_t>(ConditionCode::sign)},
                         std::get<0>(less));
}

Status write_integer_flags(EmitterContext& context, Operand value, bool write_cc) {
    if (!write_cc) return std::monostate{};
    auto equal = context.compare(value, word(0), ComparisonInfo{Compare::equal, false},
                                 {Domain::unsigned_integer, ScalarType::u32});
    if (equal.index() == 1) return std::move(std::get<1>(equal));
    auto written = context.write(Register{RegisterFile::condition_code,
                                          static_cast<std::uint16_t>(ConditionCode::zero)},
                                 std::get<0>(equal));
    if (written.index() == 1) return written;
    auto less = context.compare(value, word(0), ComparisonInfo{Compare::less, false},
                                {Domain::signed_integer, ScalarType::s32});
    if (less.index() == 1) return std::move(std::get<1>(less));
    return context.write(Register{RegisterFile::condition_code,
                                  static_cast<std::uint16_t>(ConditionCode::sign)},
                         std::get<0>(less));
}

Result<Operand> float_comparison(EmitterContext& context, std::uint64_t selector, Operand left,
                                 Operand right, const NumericMode& mode) {
    constexpr std::uint64_t nan_flag = 8;
    constexpr std::uint64_t relation_mask = ~nan_flag;
    const auto flags_mode = comparison_mode(mode);
    switch (selector) {
    case 0: // F: never true.
        return word(0, ValueType::predicate);
    case 15: // T: always true.
        return word(1, ValueType::predicate);
    case 7: // NUM: true when both inputs are numbers.
    case 8: { // NAN: true when either input is NaN.
        auto left_nan = context.unary(Opcode::is_nan, left, flags_mode, ValueType::predicate);
        if (left_nan.index() == 1) return std::move(std::get<1>(left_nan));
        auto right_nan = context.unary(Opcode::is_nan, right, flags_mode, ValueType::predicate);
        if (right_nan.index() == 1) return std::move(std::get<1>(right_nan));
        auto either = context.binary(Opcode::logical_or, std::get<0>(left_nan),
                                     std::get<0>(right_nan),
                                     {Domain::boolean, ScalarType::boolean}, ValueType::predicate);
        if (either.index() == 1) return std::move(std::get<1>(either));
        if (selector == 8) return std::get<0>(either);
        return context.unary(Opcode::logical_not, std::get<0>(either),
                             {Domain::boolean, ScalarType::boolean}, ValueType::predicate);
    }
    default: {
        const std::uint64_t relation = selector & relation_mask;
        const bool unordered = (selector & nan_flag) != 0;
        Compare compare = Compare::never;
        switch (relation) {
        case 1: compare = Compare::less; break;
        case 2: compare = Compare::equal; break;
        case 3: compare = Compare::less_equal; break;
        case 4: compare = Compare::greater; break;
        case 5: compare = Compare::not_equal; break;
        case 6: compare = Compare::greater_equal; break;
        default:
            return context.error(DiagnosticCode::unsupported_modifier,
                                 "Unknown floating comparison selector", std::to_string(selector));
        }
        return context.compare(left, right, ComparisonInfo{compare, unordered}, flags_mode);
    }
    }
}

Result<Operand> predicate_combine(EmitterContext& context, std::uint64_t selector, Operand input,
                                  Operand predicate) {
    switch (selector) {
    case 0: // AND.
    case 1: // OR.
    case 2: { // XOR.
        const Opcode opcode = selector == 0   ? Opcode::logical_and
                              : selector == 1 ? Opcode::logical_or
                                              : Opcode::logical_xor;
        return context.binary(opcode, input, predicate, {Domain::boolean, ScalarType::boolean},
                              ValueType::predicate);
    }
    case 3: // Pass-through combination.
        return input;
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown predicate combination selector", std::to_string(selector));
    }
}

Result<std::array<Operand, 2>> half_source_pair(EmitterContext& context, const DecodedSource& source,
                                                const NumericMode& mode) {
    const bool packed = source.modifiers.scalar == ScalarType::f16;
    if (!packed && source.modifiers.subword != Subword::full) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Half source selected a subword without half precision");
    }
    auto resolved = std::visit(
        [&context](const auto& value) -> Result<Operand> {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, Register>) {
                return context.read(value);
            } else {
                return Operand{value};
            }
        },
        source.value);
    if (resolved.index() == 1) return std::move(std::get<1>(resolved));
    Operand value = std::get<0>(resolved);
    if (!packed) {
        // F32 swizzle: the full word feeds both lanes.
        return std::array<Operand, 2>{value, value};
    }
    const NumericMode unpack_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even,
                                  mode.flush_inputs, false, false, mode.precise};
    // Half unpacking yields both lanes, so it uses the multi-result emit path.
    const std::array<Operand, 1> unpack_input{value};
    const std::array<ValueType, 2> unpack_outputs{ValueType::word32, ValueType::word32};
    auto unpacked =
        context.emit(Opcode::unpack_half2, unpack_mode, unpack_input, unpack_outputs,
                     ConversionInfo{ScalarType::f16, ScalarType::f32}, Effects::none);
    if (unpacked.index() == 1) return std::move(std::get<1>(unpacked));
    const auto& lanes = std::get<std::vector<ValueId>>(unpacked);
    const Operand low{lanes[0]};
    const Operand high{lanes[1]};
    switch (source.modifiers.subword) {
    case Subword::full:
        return std::array<Operand, 2>{low, high};
    case Subword::half_low:
        return std::array<Operand, 2>{low, low};
    case Subword::half_high:
        return std::array<Operand, 2>{high, high};
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Half source selected a byte position");
    }
}

Result<std::array<Operand, 2>> modified_half_source_pair(EmitterContext& context,
                                                         const DecodedInstruction& instruction,
                                                         SourceRole role, const NumericMode& mode) {
    auto decoded = decoded_source(instruction, role);
    if (decoded.index() == 1) return std::get<1>(std::move(decoded));
    auto lanes = half_source_pair(context, std::get<0>(std::move(decoded)), mode);
    if (lanes.index() == 1) return lanes;
    auto pair = std::get<std::array<Operand, 2>>(std::move(lanes));
    const auto& modifiers = std::get<0>(decoded).modifiers;
    auto lane_mode = mode;
    lane_mode.scalar = ScalarType::f32;
    for (auto& lane : pair) {
        auto modified = apply_source_modifiers(context, lane, modifiers, lane_mode);
        if (modified.index() == 1) return std::get<1>(std::move(modified));
        lane = std::get<Operand>(std::move(modified));
    }
    return pair;
}

Result<Operand> pack_half_result(EmitterContext& context, std::uint64_t selector, Operand low,
                                 Operand high, const NumericMode& mode, Register destination) {
    const NumericMode pack_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even, false,
                                mode.flush_output, false, mode.precise};
    const auto pack = [&](Operand first, Operand second) -> Result<Operand> {
        const std::array<Operand, 2> inputs{first, second};
        const std::array<ValueType, 1> outputs{ValueType::word32};
        auto packed = context.emit(Opcode::pack_half2, pack_mode, inputs, outputs,
                                   ConversionInfo{ScalarType::f32, ScalarType::f16}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&packed)) return *error;
        return Operand{std::get<std::vector<ValueId>>(packed).front()};
    };
    switch (selector) {
    case 0: // F16: pack both lanes.
        return pack(low, high);
    case 1: // F32: only lane zero survives.
        return low;
    case 2: { // MrgH0: keep the destination high half.
        auto current = context.read(destination);
        if (current.index() == 1) return std::move(std::get<1>(current));
        auto packed = pack(low, word(float_zero_bits));
        if (const auto* error = std::get_if<Diagnostic>(&packed)) return *error;
        // Only the computed lane passes through the conversion/FTZ boundary.
        // Preserve the untouched half, including subnormals and NaN payloads.
        auto kept = context.binary(Opcode::bit_and, std::get<Operand>(current), word(0xffff0000u),
                                   {Domain::bits, ScalarType::u32}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&kept)) return *error;
        return context.binary(Opcode::bit_or, std::get<Operand>(packed), std::get<Operand>(kept),
                              {Domain::bits, ScalarType::u32}, ValueType::word32);
    }
    case 3: { // MrgH1: keep the destination low half.
        auto current = context.read(destination);
        if (current.index() == 1) return std::move(std::get<1>(current));
        auto packed = pack(word(float_zero_bits), high);
        if (const auto* error = std::get_if<Diagnostic>(&packed)) return *error;
        auto kept = context.binary(Opcode::bit_and, std::get<Operand>(current), word(0x0000ffffu),
                                   {Domain::bits, ScalarType::u32}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&kept)) return *error;
        return context.binary(Opcode::bit_or, std::get<Operand>(packed), std::get<Operand>(kept),
                              {Domain::bits, ScalarType::u32}, ValueType::word32);
    }
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown half output selector", std::to_string(selector));
    }
}

Result<Operand> extract_subword(EmitterContext& context, Operand value, std::uint32_t bit_start,
                                std::uint32_t bit_count, bool is_signed) {
    const NumericMode mode{is_signed ? Domain::signed_integer : Domain::unsigned_integer,
                           is_signed ? ScalarType::s32 : ScalarType::u32};
    return context.ternary(Opcode::bit_extract, value, word(bit_start), word(bit_count), mode,
                           ValueType::word32);
}

std::uint32_t subword_bit_start(Subword position) {
    switch (position) {
    case Subword::byte1: return 8;
    case Subword::byte2: return 16;
    case Subword::byte3: return 24;
    case Subword::half_high: return 16;
    default: return 0;
    }
}
} // namespace armigo::shader::pipeline::float_emit
