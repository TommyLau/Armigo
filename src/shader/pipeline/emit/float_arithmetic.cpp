// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Rounding, flush, saturation and precision come from the decoded NumericMode
// and are stamped onto the emitted operations unchanged. Unknown FMUL scale
// selectors fail with a diagnostic.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
namespace {
constexpr NumericMode boolean_mode{Domain::boolean, ScalarType::boolean};

// Multiply-family decoders retain the raw flush selector in `function`:
// 0 preserves subnormals, 1 is FTZ, and 2 is FMZ with the zero-product rule.
Result<bool> is_fmz(EmitterContext& context) {
    auto selector = decoded_field(context.instruction(), DecodedField::function);
    if (selector.index() == 1) return std::move(std::get<1>(selector));
    if (std::get<0>(selector) > 2) {
        return context.error(DiagnosticCode::unsupported_modifier,
                             "Unknown floating multiply flush selector",
                             std::to_string(std::get<0>(selector)));
    }
    return std::get<0>(selector) == 2;
}

Result<Operand> either_zero(EmitterContext& context, Operand a, Operand b,
                            const NumericMode& mode) {
    auto compare_mode = float_emit::comparison_mode(mode);
    // Equality does not round; use the accepted explicit comparison mode while
    // retaining FMZ's input flushing at the zero test.
    compare_mode.rounding = Rounding::nearest_even;
    const auto zero = float_emit::word(float_emit::float_zero_bits, storage_type(mode.scalar));
    auto a_zero = context.compare(a, zero, ComparisonInfo{Compare::equal, false}, compare_mode);
    if (a_zero.index() == 1) return std::move(std::get<1>(a_zero));
    auto b_zero = context.compare(b, zero, ComparisonInfo{Compare::equal, false}, compare_mode);
    if (b_zero.index() == 1) return std::move(std::get<1>(b_zero));
    return context.binary(Opcode::logical_or, std::get<0>(a_zero), std::get<0>(b_zero),
                          boolean_mode, ValueType::predicate);
}

Result<Operand> fmz_boundary(EmitterContext& context, Operand value,
                             const NumericMode& mode) {
    auto boundary_mode = float_emit::without_saturation(mode);
    boundary_mode.rounding = Rounding::nearest_even;
    return context.convert(value, ConversionInfo{mode.scalar, mode.scalar}, boundary_mode,
                           storage_type(mode.scalar));
}
} // namespace

// FADD/FADD32I: absolute/negate both inputs, add, saturate on the result.
Status emit_float_add(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    auto result = context.binary(Opcode::add, a, b, mode, storage_type(mode.scalar));
    if (result.index() == 1) return std::move(std::get<1>(result));
    const Status written = float_emit::write_destination(context, instruction, std::get<0>(result));
    if (written.index() == 1) return written;
    return float_emit::write_float_flags(context, std::get<0>(result), mode, cc);
}

// FMUL/FMUL32I: optional power-of-two scale on input A, then multiply.
Status emit_float_multiply(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(scale, decoded_field(instruction, DecodedField::result_scale));
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    ARMIGO_FLOAT_TRY(fmz, is_fmz(context));
    // The scale multiplies input A before the operation and never saturates.
    // Scale selectors order the divide selectors
    // count up while the multiply selectors run M8, M4, M2.
    if (scale != 0) {
        std::uint32_t scale_bits = 0;
        switch (scale) {
        case 1: scale_bits = float_emit::float_half_bits; break;
        case 2: scale_bits = float_emit::float_quarter_bits; break;
        case 3: scale_bits = float_emit::float_eighth_bits; break;
        case 4: scale_bits = float_emit::float_eight_bits; break;
        case 5: scale_bits = float_emit::float_four_bits; break;
        case 6: scale_bits = float_emit::float_two_bits; break;
        default:
            return context.error(DiagnosticCode::unsupported_modifier,
                                 "Unknown floating multiply scale", std::to_string(scale));
        }
        auto scaled = context.binary(Opcode::multiply, a, float_emit::word(scale_bits),
                                     float_emit::without_saturation(mode), storage_type(mode.scalar));
        if (scaled.index() == 1) return std::move(std::get<1>(scaled));
        a = std::get<0>(scaled);
    }
    auto result = context.binary(Opcode::multiply, a, b, mode, storage_type(mode.scalar));
    if (result.index() == 1) return std::move(std::get<1>(result));
    Operand value = std::get<0>(result);
    if (fmz && !mode.saturate) {
        ARMIGO_FLOAT_TRY(zero_product, either_zero(context, a, b, mode));
        auto selected = context.select(zero_product,
                                       float_emit::word(float_emit::float_zero_bits,
                                                        storage_type(mode.scalar)),
                                       value, storage_type(mode.scalar));
        if (selected.index() == 1) return std::move(std::get<1>(selected));
        value = std::get<0>(selected);
    }
    const Status written = float_emit::write_destination(context, instruction, value);
    if (written.index() == 1) return written;
    return float_emit::write_float_flags(context, value, mode, cc);
}

// FFMA/FFMA32I: negation on the decoded sources, then a single fused operation.
Status emit_float_fma(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(c, float_emit::modified_source(context, instruction, SourceRole::c, mode));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    ARMIGO_FLOAT_TRY(fmz, is_fmz(context));
    auto result = context.ternary(Opcode::fused_multiply_add, a, b, c, mode,
                                  storage_type(mode.scalar));
    if (result.index() == 1) return std::move(std::get<1>(result));
    Operand value = std::get<0>(result);
    if (fmz && !mode.saturate) {
        ARMIGO_FLOAT_TRY(zero_product, either_zero(context, a, b, mode));
        ARMIGO_FLOAT_TRY(flushed_c, fmz_boundary(context, c, mode));
        auto selected = context.select(zero_product, flushed_c, value,
                                       storage_type(mode.scalar));
        if (selected.index() == 1) return std::move(std::get<1>(selected));
        value = std::get<0>(selected);
    }
    const Status written = float_emit::write_destination(context, instruction, value);
    if (written.index() == 1) return written;
    return float_emit::write_float_flags(context, value, mode, cc);
}
} // namespace armigo::shader::pipeline
