// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The Armigo contract keeps predicates separate from integer words, so
// boolean combinations use logical operations and the half-boolean packing
// converts predicates to words explicitly. Behavior and modifier preservation
// are unchanged. Family selector values in `half_mode`: FSET/HSET2 0/1 select
// integer or floating boolean results, HSETP2 0/1 selects the lane-AND mode.
// HSET2/HSETP2 compare the unpacked half lanes using FP32 comparison:
// every binary16 value is exact in binary32, so the f32 comparison is exact.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
namespace {
constexpr NumericMode boolean_mode{Domain::boolean, ScalarType::boolean};

Result<Operand> prepare_comparison(EmitterContext& context, const DecodedInstruction& instruction,
                                   const NumericMode& mode) {
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::condition));
    return float_emit::float_comparison(context, selector, a, b, mode);
}

// FSET: predicate input combined with the guest boolean operation, written as
// float (1.0/0.0) or integer word according to the boolean-result selector.
Status emit_fset_body(EmitterContext& context, const DecodedInstruction& instruction,
                      const NumericMode& mode, std::uint64_t boolean_operation,
                      Operand comparison, Operand predicate, bool boolean_result, bool write_cc) {
    ARMIGO_FLOAT_TRY(combined, float_emit::predicate_combine(context, boolean_operation, comparison,
                                                             predicate));
    const ValueType storage = storage_type(mode.scalar);
    if (boolean_result) {
        auto as_float = context.select(combined, float_emit::word(float_emit::float_one_bits),
                                       float_emit::word(float_emit::float_zero_bits), storage);
        if (as_float.index() == 1) return std::move(std::get<1>(as_float));
        const Status written =
            float_emit::write_destination(context, instruction, std::get<0>(as_float));
        if (written.index() == 1) return written;
        return float_emit::write_float_flags(context, std::get<0>(as_float), mode, write_cc);
    }
    // Integer comparison results use the guest's all-ones mask convention.
    auto as_word = context.unary(Opcode::predicate_to_mask, combined, {}, ValueType::word32);
    if (as_word.index() == 1) return std::move(std::get<1>(as_word));
    const Status written =
        float_emit::write_destination(context, instruction, std::get<0>(as_word));
    if (written.index() == 1) return written;
    return float_emit::write_integer_flags(context, std::get<0>(as_word), write_cc);
}

Result<std::pair<Register, Register>> predicate_destinations(
    EmitterContext& context, const DecodedInstruction& instruction) {
    std::optional<Register> primary, secondary;
    for (const auto& access : instruction.registers) {
        if (access.role == RegisterRole::predicate_destination) primary = access.reg;
        if (access.role == RegisterRole::second_predicate_destination) secondary = access.reg;
    }
    if (!primary || !secondary) {
        return context.error(DiagnosticCode::malformed_input,
                             "Comparison predicate instruction is missing a destination predicate");
    }
    return std::pair{*primary, *secondary};
}

// Packed-half comparisons run on the unpacked binary32 lanes: the decoded
// numeric mode names the guest half-precision storage, and scalar binary16
// arithmetic has no MSL generator. Widening only the scalar is exact (every
// binary16 value is representable in binary32) and keeps the decoded input
// flush, precision and rounding for the consuming comparison, which carries
// neither rounding nor output flush of its own.
NumericMode half_comparison_mode(const NumericMode& mode) {
    auto lane = mode;
    lane.scalar = ScalarType::f32;
    return lane;
}
} // namespace

// FSET: floating comparison into a word destination.
Status emit_float_set(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(comparison, prepare_comparison(context, instruction, mode));
    ARMIGO_FLOAT_TRY(predicate, float_emit::predicate_operand(context, instruction));
    ARMIGO_FLOAT_TRY(boolean_operation, decoded_field(instruction, DecodedField::boolean_operation));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::half_mode));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    return emit_fset_body(context, instruction, mode, boolean_operation, comparison, predicate,
                          selector != 0, cc);
}

// FSETP: floating comparison into two predicate destinations, the second one
// carrying the inverted first result combined under the same boolean operation.
Status emit_float_set_predicate(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(comparison, prepare_comparison(context, instruction, mode));
    ARMIGO_FLOAT_TRY(predicate, float_emit::predicate_operand(context, instruction));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::boolean_operation));
    auto inverted = context.unary(Opcode::logical_not, comparison, boolean_mode,
                                  ValueType::predicate);
    if (inverted.index() == 1) return std::move(std::get<1>(inverted));
    ARMIGO_FLOAT_TRY(first, float_emit::predicate_combine(context, selector, comparison, predicate));
    ARMIGO_FLOAT_TRY(second,
                     float_emit::predicate_combine(context, selector, std::get<0>(inverted),
                                                   predicate));
    ARMIGO_FLOAT_TRY(destinations, predicate_destinations(context, instruction));
    Status written = context.write(destinations.first, std::get<0>(first));
    if (written.index() == 1) return written;
    return context.write(destinations.second, std::get<0>(second));
}

// FCMP: floating conditional select between two values on a zero comparison.
Status emit_float_compare_select(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(c, float_emit::modified_source(context, instruction, SourceRole::c, mode));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::condition));
    ARMIGO_FLOAT_TRY(
        comparison,
        float_emit::float_comparison(context, selector, c,
                                     float_emit::word(float_emit::float_zero_bits,
                                                      storage_type(mode.scalar)),
                                     mode));
    auto result = context.select(comparison, a, b, storage_type(mode.scalar));
    if (result.index() == 1) return std::move(std::get<1>(result));
    return float_emit::write_destination(context, instruction, std::get<0>(result));
}

// HSET2: per-lane half comparisons packed as words or floating booleans.
Status emit_half_set(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const NumericMode lane_mode = half_comparison_mode(mode);
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_half_source_pair(context, instruction, SourceRole::a,
                                                              mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_half_source_pair(context, instruction, SourceRole::b,
                                                              mode));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::condition));
    ARMIGO_FLOAT_TRY(boolean_operation, decoded_field(instruction, DecodedField::boolean_operation));
    ARMIGO_FLOAT_TRY(boolean_result, decoded_field(instruction, DecodedField::half_mode));
    ARMIGO_FLOAT_TRY(predicate, float_emit::predicate_operand(context, instruction));
    ARMIGO_FLOAT_TRY(comparison_low, float_emit::float_comparison(context, selector, a[0], b[0],
                                                                  lane_mode));
    ARMIGO_FLOAT_TRY(comparison_high, float_emit::float_comparison(context, selector, a[1], b[1],
                                                                   lane_mode));
    ARMIGO_FLOAT_TRY(combined_low,
                     float_emit::predicate_combine(context, boolean_operation, comparison_low,
                                                   predicate));
    ARMIGO_FLOAT_TRY(combined_high,
                     float_emit::predicate_combine(context, boolean_operation, comparison_high,
                                                   predicate));
    if (boolean_result != 0) {
        auto low_float =
            context.select(combined_low, float_emit::word(float_emit::float_one_bits),
                           float_emit::word(float_emit::float_zero_bits), ValueType::word32);
        if (low_float.index() == 1) return std::move(std::get<1>(low_float));
        auto high_float =
            context.select(combined_high, float_emit::word(float_emit::float_one_bits),
                           float_emit::word(float_emit::float_zero_bits), ValueType::word32);
        if (high_float.index() == 1) return std::move(std::get<1>(high_float));
        auto packed = context.binary(Opcode::pack_half2, std::get<0>(low_float),
                                     std::get<0>(high_float),
                                     {Domain::floating, ScalarType::f32, Rounding::nearest_even},
                                     ValueType::word32);
        if (packed.index() == 1) return std::move(std::get<1>(packed));
        return float_emit::write_destination(context, instruction, std::get<0>(packed));
    }
    // Integer booleans use an all-ones mask in each packed half.
    auto low_word = context.select(combined_low, float_emit::word(0xFFFFU),
                                   float_emit::word(0), ValueType::word32);
    if (low_word.index() == 1) return std::move(std::get<1>(low_word));
    auto high_word = context.select(combined_high, float_emit::word(0xFFFFU),
                                    float_emit::word(0), ValueType::word32);
    if (high_word.index() == 1) return std::move(std::get<1>(high_word));
    auto masked_low = context.binary(Opcode::bit_and, std::get<0>(low_word),
                                     float_emit::word(0xFFFFU), {Domain::bits, ScalarType::u32},
                                     ValueType::word32);
    if (masked_low.index() == 1) return std::move(std::get<1>(masked_low));
    auto shifted_high = context.binary(Opcode::shift_left, std::get<0>(high_word),
                                       float_emit::word(16), {Domain::bits, ScalarType::u32},
                                       ValueType::word32);
    if (shifted_high.index() == 1) return std::move(std::get<1>(shifted_high));
    auto packed = context.binary(Opcode::bit_or, std::get<0>(masked_low),
                                 std::get<0>(shifted_high), {Domain::bits, ScalarType::u32},
                                 ValueType::word32);
    if (packed.index() == 1) return std::move(std::get<1>(packed));
    return float_emit::write_destination(context, instruction, std::get<0>(packed));
}

// HSETP2: two half comparisons; the lane-AND mode combines them before the
// second destination receives the inverted combined result.
Status emit_half_set_predicate(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const NumericMode lane_mode = half_comparison_mode(mode);
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_half_source_pair(context, instruction, SourceRole::a,
                                                              mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_half_source_pair(context, instruction, SourceRole::b,
                                                              mode));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::condition));
    ARMIGO_FLOAT_TRY(boolean_operation, decoded_field(instruction, DecodedField::boolean_operation));
    ARMIGO_FLOAT_TRY(lane_and, decoded_field(instruction, DecodedField::half_mode));
    ARMIGO_FLOAT_TRY(predicate, float_emit::predicate_operand(context, instruction));
    ARMIGO_FLOAT_TRY(comparison_low, float_emit::float_comparison(context, selector, a[0], b[0],
                                                                  lane_mode));
    ARMIGO_FLOAT_TRY(comparison_high, float_emit::float_comparison(context, selector, a[1], b[1],
                                                                   lane_mode));
    // Constant-true/false selectors produce literals, so the lane operands
    // move by value; only the lane-AND results need unwrapping below.
    Operand first = comparison_low;
    Operand second = comparison_high;
    if (lane_and != 0) {
        auto combined = context.binary(Opcode::logical_and, first, second, boolean_mode,
                                       ValueType::predicate);
        if (combined.index() == 1) return std::move(std::get<1>(combined));
        first = std::get<0>(combined);
        auto inverted = context.unary(Opcode::logical_not, first, boolean_mode,
                                      ValueType::predicate);
        if (inverted.index() == 1) return std::move(std::get<1>(inverted));
        second = std::get<0>(inverted);
    }
    ARMIGO_FLOAT_TRY(combined_low, float_emit::predicate_combine(context, boolean_operation, first,
                                                                 predicate));
    ARMIGO_FLOAT_TRY(combined_high, float_emit::predicate_combine(context, boolean_operation, second,
                                                                  predicate));
    ARMIGO_FLOAT_TRY(destinations, predicate_destinations(context, instruction));
    Status written = context.write(destinations.first, combined_low);
    if (written.index() == 1) return written;
    return context.write(destinations.second, combined_high);
}
} // namespace armigo::shader::pipeline
