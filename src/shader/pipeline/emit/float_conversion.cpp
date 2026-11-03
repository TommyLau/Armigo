// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Format selectors in `source_type`/`destination_type` carry the guest field
// values: F2F/F2I floating formats 1=F16 2=F32 3=F64; F2I integer formats
// 1=U16 2=U32 3=U64 5=S16 6=S32 7=S64; I2F sources 0=U8 1=U16 2=U32 3=U64
// 4=S8 5=S16 6=S32 7=S64; I2I 0=U8 1=U16 2=U32 4=S8 5=S16 6=S32. Byte and
// half positions come from the source subword modifiers. Unsupported 64-bit
// F2I and I2F/I2I sources, along with invalid I2I encodings, are rejected with
// diagnostics. FP64 keeps
// word64 storage end to end; no FP32 substitution exists on any path.
//
// Contract notes: F2F carries its integral-rounding selector in
// `integral_round` (guest IntegerRound: 0..3 pass, 4 round, 5 floor, 6 ceil,
// 7 trunc) and I2I saturation in `integer_saturation`; neither rides NumericMode
// (saturation there is the floating [0,1] operation). A same-type selector of
// 4..7 becomes a dedicated integral-rounding operation carrying the directed
// rounding, followed by the guest saturation as its own boundary conversion.
// Rounding-free floating steps (the sign modifiers, the saturation boundary,
// a same-type conversion's binary16 pack) carry nearest-even as their explicit
// rounding tag; a cross-format conversion keeps its decoded rounding on the
// narrowing step, where a directed mode without a proven generator still
// fails explicitly.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
namespace {
constexpr NumericMode bits_mode{Domain::bits, ScalarType::u32};

Result<ScalarType> float_format_scalar(EmitterContext& context, std::uint64_t selector) {
    switch (selector) {
    case 1: return ScalarType::f16;
    case 2: return ScalarType::f32;
    case 3: return ScalarType::f64;
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown floating format selector", std::to_string(selector));
    }
}

NumericMode float_mode_for(const NumericMode& decoded, ScalarType scalar) {
    return {Domain::floating, scalar, decoded.rounding, decoded.flush_inputs,
            decoded.flush_output, decoded.saturate, decoded.precise};
}

// F16 values are carried as exact f32 conversions, so conversions targeting or
// coming from packed halves use f32 as their semantic scalar.
constexpr ScalarType conversion_scalar(ScalarType scalar) {
    return scalar == ScalarType::f16 ? ScalarType::f32 : scalar;
}

// Writes a floating value in the destination format: word32 for F32, packed
// half for F16, and an explicit word64 unpack for F64 (low word first).
Status write_float_destination(EmitterContext& context, const DecodedInstruction& instruction,
                               Operand value, ScalarType destination,
                               const NumericMode& boundary_mode) {
    if (destination == ScalarType::f32) {
        return float_emit::write_destination(context, instruction, value);
    }
    if (destination == ScalarType::f16) {
        auto pack_mode = float_mode_for(boundary_mode, ScalarType::f32);
        pack_mode.saturate = false;
        const std::array<Operand, 2> inputs{
            value, float_emit::word(float_emit::float_zero_bits)};
        const std::array<ValueType, 1> outputs{ValueType::word32};
        auto packed = context.emit(Opcode::pack_half2, pack_mode, inputs, outputs,
                                   ConversionInfo{ScalarType::f32, ScalarType::f16},
                                   Effects::none);
        if (packed.index() == 1) return std::move(std::get<1>(packed));
        return float_emit::write_destination(
            context, instruction, Operand{std::get<std::vector<ValueId>>(packed).front()});
    }
    const std::array<Operand, 1> unpack_input{value};
    const std::array<ValueType, 2> unpack_outputs{ValueType::word32, ValueType::word32};
    auto words = context.emit(Opcode::unpack_word64, bits_mode, unpack_input, unpack_outputs,
                              std::monostate{}, Effects::none);
    if (words.index() == 1) return std::move(std::get<1>(words));
    const auto& lanes = std::get<std::vector<ValueId>>(words);
    auto destination_register = float_emit::destination_register(instruction);
    if (destination_register.index() == 1) return std::move(std::get<1>(destination_register));
    auto second_register = float_emit::second_destination_register(instruction);
    if (second_register.index() == 1) return std::move(std::get<1>(second_register));
    Status written =
        context.write(std::get<Register>(destination_register), Operand{lanes[0]});
    if (written.index() == 1) return written;
    return context.write(std::get<Register>(second_register), Operand{lanes[1]});
}

// Resolves the B source according to the floating source format. F16 sources
// unpack both halves and pick the lane named by the subword modifier.
Result<Operand> conversion_source(EmitterContext& context, const DecodedInstruction& instruction,
                                  ScalarType source) {
    auto decoded = decoded_source(instruction, SourceRole::b);
    if (decoded.index() == 1) return std::move(std::get<1>(decoded));
    const auto& origin = std::get<0>(decoded);
    if (source == ScalarType::f64) return float_emit::float64_source(context, origin);
    auto value = std::visit(
        [&context](const auto& inner) -> Result<Operand> {
            using Value = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<Value, Register>) {
                return context.read(inner);
            } else {
                return Operand{inner};
            }
        },
        origin.value);
    if (value.index() == 1) return value;
    if (source != ScalarType::f16) return value;
    const NumericMode unpack_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even,
                                  false, false, false, false};
    const std::array<Operand, 1> unpack_input{std::get<Operand>(value)};
    const std::array<ValueType, 2> unpack_outputs{ValueType::word32, ValueType::word32};
    auto lanes = context.emit(Opcode::unpack_half2, unpack_mode, unpack_input, unpack_outputs,
                              ConversionInfo{ScalarType::f16, ScalarType::f32}, Effects::none);
    if (lanes.index() == 1) return std::move(std::get<1>(lanes));
    const auto& words = std::get<std::vector<ValueId>>(lanes);
    return Operand{words[origin.modifiers.subword == Subword::half_high ? 1 : 0]};
}

// F2F/F2I round-to-integral selection keyed off the decoded NumericMode. The
// dedicated operation owns the guest rounding alone: saturation is a later
// boundary step after rounding, so it never rides
// this operation.
Result<std::optional<Operand>> round_to_integral(EmitterContext& context, Operand value,
                                                 const NumericMode& mode) {
    Opcode opcode = Opcode::copy;
    switch (mode.rounding) {
    case Rounding::nearest_even: opcode = Opcode::round; break;
    case Rounding::toward_negative: opcode = Opcode::floor; break;
    case Rounding::toward_positive: opcode = Opcode::ceiling; break;
    case Rounding::toward_zero: opcode = Opcode::truncate; break;
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown conversion rounding mode");
    }
    auto rounded = context.unary(opcode, value, float_emit::without_saturation(mode),
                                 storage_type(mode.scalar));
    if (rounded.index() == 1) return std::move(std::get<1>(rounded));
    return std::optional<Operand>{std::get<0>(std::move(rounded))};
}

// Applies guest saturation in f32 space (F16 destinations saturate before
// packing) through a value-preserving same-width conversion. Saturation
// follows the rounding, ordering FPRound before
// FPSaturate, so the boundary conversion itself performs no rounding: the
// dedicated integral-rounding operation has already been emitted. The frozen
// validator requires an explicit rounding tag on every floating operation,
// so the rounding-free boundary carries nearest-even like the other exact
// sign and flush steps.
Result<Operand> apply_saturation(EmitterContext& context, Operand value, const NumericMode& mode) {
    const ScalarType carrier = conversion_scalar(mode.scalar);
    auto boundary = float_mode_for(mode, carrier);
    boundary.rounding = Rounding::nearest_even;
    auto saturated = context.convert(value, ConversionInfo{carrier, carrier},
                                     boundary, storage_type(carrier));
    if (saturated.index() == 1) return std::move(std::get<1>(saturated));
    return std::get<0>(saturated);
}
} // namespace

// F2F: floating re-interpretation with optional integral rounding, precision
// conversion and saturation; rounding applies only to
// same-type conversions and saturation follows the rounding).
Status emit_float_to_float(EmitterContext& context) {
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(decoded_mode, float_emit::float_mode(context));
    ARMIGO_FLOAT_TRY(source_selector, decoded_field(instruction, DecodedField::source_type));
    ARMIGO_FLOAT_TRY(destination_selector,
                     decoded_field(instruction, DecodedField::destination_type));
    ARMIGO_FLOAT_TRY(integral_round, decoded_field(instruction, DecodedField::integral_round));
    ARMIGO_FLOAT_TRY(source_scalar, float_format_scalar(context, source_selector));
    ARMIGO_FLOAT_TRY(destination_scalar, float_format_scalar(context, destination_selector));
    ARMIGO_FLOAT_TRY(source, conversion_source(context, instruction, source_scalar));
    ARMIGO_FLOAT_TRY(modifiers, float_emit::source_modifiers(instruction, SourceRole::b));
    const NumericMode source_mode =
        float_mode_for(decoded_mode, conversion_scalar(source_scalar));
    ARMIGO_FLOAT_TRY(value, float_emit::apply_source_modifiers(context, source, modifiers,
                                                               source_mode));
    if ((source_scalar == ScalarType::f16 && destination_scalar == ScalarType::f64) ||
        (source_scalar == ScalarType::f64 && destination_scalar == ScalarType::f16)) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "F16 and F64 conversion formats cannot be combined");
    }
    Operand result = value;
    // The numeric mode the destination write sees. A same-type conversion with
    // an integral selector has already spent the guest rounding on the
    // dedicated rounding operation, so an F16 destination pack becomes its own
    // nearest-even boundary instead of inheriting the spent direction.
    NumericMode destination_boundary = decoded_mode;
    if (source_scalar == destination_scalar) {
        // IntegerRound 0..3 pass the value through; 4..7 round to an integer.
        if (integral_round >= 4) {
            ARMIGO_FLOAT_TRY(rounded, round_to_integral(context, result, source_mode));
            result = *rounded;
            destination_boundary.rounding = Rounding::nearest_even;
        }
        if (decoded_mode.saturate) {
            ARMIGO_FLOAT_TRY(saturated, apply_saturation(context, result, source_mode));
            result = saturated;
        }
        if (integral_round < 4 &&
            (decoded_mode.flush_inputs || decoded_mode.flush_output) &&
            source_scalar == ScalarType::f32) {
            auto boundary_mode = float_mode_for(decoded_mode, source_scalar);
            // A same-type conversion rounds nothing; nearest-even keeps the
            // floating NumericMode valid while the boundary applies FTZ.
            boundary_mode.rounding = Rounding::nearest_even;
            boundary_mode.saturate = false;
            auto converted = context.convert(result,
                                             ConversionInfo{source_scalar, destination_scalar},
                                             boundary_mode, storage_type(destination_scalar));
            if (converted.index() == 1) return std::move(std::get<1>(converted));
            result = std::get<0>(converted);
        }
    } else if (conversion_scalar(source_scalar) == conversion_scalar(destination_scalar)) {
        // F16 and F32 share the f32 carrier; only saturation applies. The
        // destination pack below performs the actual narrowing.
        if (decoded_mode.saturate) {
            ARMIGO_FLOAT_TRY(saturated, apply_saturation(context, result, source_mode));
            result = saturated;
        }
    } else {
        // Cross-carrier conversions carry the guest rounding, flush and
        // saturation at the source precision boundary.
        const NumericMode convert_mode = float_mode_for(decoded_mode, source_mode.scalar);
        auto converted = context.convert(
            result, ConversionInfo{source_mode.scalar, destination_scalar},
            convert_mode, storage_type(destination_scalar));
        if (converted.index() == 1) return std::move(std::get<1>(converted));
        result = std::get<0>(converted);
    }
    return write_float_destination(context, instruction, result, destination_scalar,
                                   destination_boundary);
}

// F2I: float to integer with explicit integral rounding, an unsigned lower
// clamp, and small-destination range clamping.
Status emit_float_to_integer(EmitterContext& context) {
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(decoded_mode, float_emit::float_mode(context));
    ARMIGO_FLOAT_TRY(source_selector, decoded_field(instruction, DecodedField::source_type));
    ARMIGO_FLOAT_TRY(destination_selector,
                     decoded_field(instruction, DecodedField::destination_type));
    if (destination_selector == 3 || destination_selector == 7) {
        return context.error(DiagnosticCode::unsupported_capability,
                             "64-bit F2I destinations are not implemented");
    }
    ScalarType destination_scalar = ScalarType::u32;
    switch (destination_selector) {
    case 1: destination_scalar = ScalarType::u16; break;
    case 2: destination_scalar = ScalarType::u32; break;
    case 5: destination_scalar = ScalarType::s16; break;
    case 6: destination_scalar = ScalarType::s32; break;
    default:
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown F2I destination format",
                             std::to_string(destination_selector));
    }
    ARMIGO_FLOAT_TRY(source_scalar, float_format_scalar(context, source_selector));
    ARMIGO_FLOAT_TRY(source, conversion_source(context, instruction, source_scalar));
    ARMIGO_FLOAT_TRY(modifiers, float_emit::source_modifiers(instruction, SourceRole::b));
    const NumericMode source_mode = float_mode_for(decoded_mode, conversion_scalar(source_scalar));
    ARMIGO_FLOAT_TRY(value, float_emit::apply_source_modifiers(context, source, modifiers,
                                                               source_mode));
    ARMIGO_FLOAT_TRY(rounded, round_to_integral(context, value, source_mode));
    Operand integral = rounded ? *rounded : value;
    const bool signed_destination = destination_scalar == ScalarType::s16 ||
                                    destination_scalar == ScalarType::s32;
    if (!signed_destination) {
        // Negative floats have an undefined unsigned cast, so clamp first.
        const ScalarType carrier = conversion_scalar(source_scalar);
        auto clamped = context.binary(Opcode::maximum, integral,
                                      float_emit::word(float_emit::float_zero_bits,
                                                       storage_type(carrier)),
                                      float_emit::without_saturation(source_mode), storage_type(carrier));
        if (clamped.index() == 1) return std::move(std::get<1>(clamped));
        integral = std::get<0>(clamped);
    }
    const ScalarType integer_carrier = signed_destination ? ScalarType::s32 : ScalarType::u32;
    // The explicit guest rounding already applied; the conversion itself
    // truncates toward zero.
    const NumericMode convert_mode{Domain::floating, source_mode.scalar, Rounding::toward_zero,
                                   source_mode.flush_inputs, source_mode.flush_output, false,
                                   source_mode.precise};
    auto converted = context.convert(integral,
                                     ConversionInfo{conversion_scalar(source_scalar),
                                                    integer_carrier},
                                     convert_mode, storage_type(integer_carrier));
    if (converted.index() == 1) return std::move(std::get<1>(converted));
    Operand word = std::get<0>(converted);
    if (destination_scalar == ScalarType::u16 || destination_scalar == ScalarType::s16) {
        const std::uint32_t minimum = signed_destination ? 0xFFFF8000U : 0U;
        const std::uint32_t maximum = signed_destination ? 0x7FFFU : 0xFFFFU;
        const NumericMode clamp_mode{signed_destination ? Domain::signed_integer
                                                        : Domain::unsigned_integer,
                                     integer_carrier};
        auto clamped = context.ternary(Opcode::clamp, word, float_emit::word(minimum),
                                       float_emit::word(maximum), clamp_mode, ValueType::word32);
        if (clamped.index() == 1) return std::move(std::get<1>(clamped));
        word = std::get<0>(clamped);
    }
    return float_emit::write_destination(context, instruction, word);
}

// I2F: integer to float with byte extraction for narrow sources.
Status emit_integer_to_float(EmitterContext& context) {
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(decoded_mode, float_emit::float_mode(context));
    ARMIGO_FLOAT_TRY(source_selector, decoded_field(instruction, DecodedField::source_type));
    ARMIGO_FLOAT_TRY(destination_selector,
                     decoded_field(instruction, DecodedField::destination_type));
    if (source_selector == 3 || source_selector == 7) {
        return context.error(DiagnosticCode::unsupported_capability,
                             "64-bit I2F sources are not implemented");
    }
    const bool is_signed = source_selector >= 4;
    const bool is_small = source_selector <= 1 || source_selector == 4 || source_selector == 5;
    ARMIGO_FLOAT_TRY(destination_scalar, float_format_scalar(context, destination_selector));
    ARMIGO_FLOAT_TRY(source, float_emit::source_operand(context, instruction, SourceRole::b));
    ARMIGO_FLOAT_TRY(modifiers, float_emit::source_modifiers(instruction, SourceRole::b));
    const NumericMode integer_mode{is_signed ? Domain::signed_integer : Domain::unsigned_integer,
                                   is_signed ? ScalarType::s32 : ScalarType::u32};
    ARMIGO_FLOAT_TRY(value, float_emit::apply_source_modifiers(context, source, modifiers,
                                                               integer_mode));
    Operand widened = value;
    if (is_small) {
        const std::uint32_t size = (source_selector == 1 || source_selector == 5) ? 16 : 8;
        ARMIGO_FLOAT_TRY(extracted, float_emit::extract_subword(
                                        context, value,
                                        float_emit::subword_bit_start(modifiers.subword), size,
                                        is_signed));
        widened = extracted;
    }
    const ScalarType float_carrier = conversion_scalar(destination_scalar);
    const NumericMode convert_mode{Domain::floating, float_carrier, decoded_mode.rounding,
                                   false, decoded_mode.flush_output, false, decoded_mode.precise};
    auto converted = context.convert(widened, ConversionInfo{integer_mode.scalar, float_carrier},
                                     convert_mode, storage_type(float_carrier));
    if (converted.index() == 1) return std::move(std::get<1>(converted));
    return write_float_destination(context, instruction, std::get<0>(converted),
                                   destination_scalar, decoded_mode);
}

// I2I: integer re-interpretation with byte extraction, modifiers, optional
// saturation clamping and condition-code flags.
Status emit_integer_to_integer(EmitterContext& context) {
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(source_selector, decoded_field(instruction, DecodedField::source_type));
    ARMIGO_FLOAT_TRY(destination_selector,
                     decoded_field(instruction, DecodedField::destination_type));
    const auto selector_valid = [](std::uint64_t value) {
        switch (value) {
        case 0: case 1: case 2: case 4: case 5: case 6: return true;
        default: return false;
        }
    };
    if (!selector_valid(source_selector) || !selector_valid(destination_selector)) {
        return context.error(DiagnosticCode::unsupported_modifier, "Invalid I2I format selectors");
    }
    const bool source_signed = source_selector >= 4;
    const bool destination_signed = destination_selector >= 4;
    const bool source_small =
        source_selector == 0 || source_selector == 1 || source_selector == 4 ||
        source_selector == 5;
    ARMIGO_FLOAT_TRY(source, float_emit::source_operand(context, instruction, SourceRole::b));
    ARMIGO_FLOAT_TRY(modifiers, float_emit::source_modifiers(instruction, SourceRole::b));
    const NumericMode source_mode{source_signed ? Domain::signed_integer
                                                : Domain::unsigned_integer,
                                  source_signed ? ScalarType::s32 : ScalarType::u32};
    Operand value = source;
    if (source_small) {
        const std::uint32_t size = (source_selector == 1 || source_selector == 5) ? 16 : 8;
        ARMIGO_FLOAT_TRY(extracted, float_emit::extract_subword(
                                        context, value,
                                        float_emit::subword_bit_start(modifiers.subword), size,
                                        source_signed));
        value = extracted;
    }
    ARMIGO_FLOAT_TRY(modified, float_emit::apply_source_modifiers(context, value, modifiers,
                                                                  source_mode));
    value = modified;
    ARMIGO_FLOAT_TRY(saturate, decoded_field(instruction, DecodedField::integer_saturation));
    if (saturate != 0) {
        std::uint32_t minimum = 0;
        std::uint32_t maximum = 0;
        switch (destination_selector) {
        case 0: maximum = 0xFFU; break;
        case 1: maximum = 0xFFFFU; break;
        case 2: maximum = 0xFFFFFFFFU; break;
        case 4: minimum = 0xFFFFFF80U; maximum = 0x7FU; break;
        case 5: minimum = 0xFFFF8000U; maximum = 0x7FFFU; break;
        case 6: minimum = 0x80000000U; maximum = 0x7FFFFFFFU; break;
        default: break;
        }
        const NumericMode clamp_mode{destination_signed ? Domain::signed_integer
                                                        : Domain::unsigned_integer,
                                     destination_signed ? ScalarType::s32 : ScalarType::u32};
        auto clamped = context.ternary(Opcode::clamp, value, float_emit::word(minimum),
                                       float_emit::word(maximum), clamp_mode, ValueType::word32);
        if (clamped.index() == 1) return std::move(std::get<1>(clamped));
        value = std::get<0>(clamped);
    }
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    const Status written = float_emit::write_destination(context, instruction, value);
    if (written.index() == 1) return written;
    return float_emit::write_integer_flags(context, value, cc);
}
} // namespace armigo::shader::pipeline
