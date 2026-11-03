// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The `function` selector carries the guest MUFU operation: 0=cos 1=sin 2=ex2
// 3=lg2 4=rcp 5=rsq 6=rcp64h 7=rsq64h 8=sqrt. The 64h helpers keep word64
// storage and FP64 semantics end to end (no FP32 substitution). Saturation is
// applied to the FP64 operation after unpacking the high word. Unknown
// selectors are diagnostics, not log-and-continue passes. RRO remains a
// modifier-preserving move followed by
// a matching MUFU.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
namespace {
// 64h helpers operate on a double assembled from a zero low word and the f32
// input as the high word; the result is the high word of the FP64 operation.
Result<Operand> reciprocal_high_word(EmitterContext& context, Operand source,
                                     const NumericMode& mode, bool square_root) {
    auto packed = context.binary(Opcode::pack_word64, float_emit::word(float_emit::float_zero_bits),
                                 source, {Domain::bits, ScalarType::u32}, ValueType::word64);
    if (packed.index() == 1) return std::move(std::get<1>(packed));
    const NumericMode double_mode{Domain::floating, ScalarType::f64, Rounding::nearest_even,
                                  mode.flush_inputs, mode.flush_output, mode.saturate,
                                  mode.precise};
    auto computed = context.unary(square_root ? Opcode::reciprocal_sqrt : Opcode::reciprocal,
                                  std::get<0>(packed), double_mode, ValueType::word64);
    if (computed.index() == 1) return std::move(std::get<1>(computed));
    const std::array<Operand, 1> unpack_input{std::get<0>(computed)};
    const std::array<ValueType, 2> unpack_outputs{ValueType::word32, ValueType::word32};
    auto words = context.emit(Opcode::unpack_word64, {Domain::bits, ScalarType::u32},
                              unpack_input, unpack_outputs, std::monostate{}, Effects::none);
    if (words.index() == 1) return std::move(std::get<1>(words));
    return Operand{std::get<std::vector<ValueId>>(words)[1]};
}
} // namespace

// MUFU: multifunction unit operation selected by the function field.
Status emit_multifunction(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(source, float_emit::modified_source(context, instruction, SourceRole::a,
                                                         mode));
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::function));
    Opcode opcode = Opcode::copy;
    bool high_word = false;
    switch (selector) {
    case 0: opcode = Opcode::cosine; break;
    case 1: opcode = Opcode::sine; break;
    case 2: opcode = Opcode::exp2; break;
    case 3: opcode = Opcode::log2; break;
    case 4: opcode = Opcode::reciprocal; break;
    case 5: opcode = Opcode::reciprocal_sqrt; break;
    case 8: opcode = Opcode::square_root; break;
    case 6: high_word = true; break;
    case 7: high_word = true; break;
    default:
        return context.error(DiagnosticCode::unsupported_modifier, "Unknown MUFU function",
                             std::to_string(selector));
    }
    Operand result;
    if (high_word) {
        ARMIGO_FLOAT_TRY(computed, reciprocal_high_word(context, source, mode, selector == 7));
        result = computed;
    } else {
        auto computed = context.unary(opcode, source, mode, storage_type(mode.scalar));
        if (computed.index() == 1) return std::move(std::get<1>(computed));
        result = std::get<0>(computed);
    }
    return float_emit::write_destination(context, instruction, result);
}

// RRO: range-reduction operator. It is always followed by a matching MUFU, so
// it translates to a modifier-preserving move.
Status emit_range_reduction(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(source, float_emit::modified_source(context, instruction, SourceRole::b,
                                                         mode));
    return float_emit::write_destination(context, instruction, source);
}
} // namespace armigo::shader::pipeline
