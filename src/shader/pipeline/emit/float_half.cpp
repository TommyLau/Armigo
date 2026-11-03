// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Packed halves are decomposed into scalar f32 operations with explicit
// pack/unpack as the pipeline contract requires; no hidden lanes live in
// NumericMode. The `half_mode` selector carries the guest output format:
// 0=F16 1=F32 2=MrgH0 3=MrgH1.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
namespace {
// Runs one lane operation per half and packs according to the output selector.
Status emit_half_operation(EmitterContext& context, Opcode opcode, const NumericMode& mode) {
    // Unpacked lanes are computed in binary32, then explicitly packed into
    // the selected result. Half precision is a boundary format, not hidden IR lanes.
    auto lane_mode = mode;
    lane_mode.scalar = ScalarType::f32;
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_half_source_pair(context, instruction, SourceRole::a,
                                                              mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_half_source_pair(context, instruction, SourceRole::b,
                                                              mode));
    std::optional<std::array<Operand, 2>> c;
    if (opcode == Opcode::fused_multiply_add) {
        ARMIGO_FLOAT_TRY(loaded, float_emit::modified_half_source_pair(
                                     context, instruction, SourceRole::c, mode));
        c = std::move(loaded);
    }
    std::array<Operand, 2> results{};
    for (std::size_t lane = 0; lane < results.size(); ++lane) {
        if (c) {
            auto fused = context.ternary(opcode, a[lane], b[lane], (*c)[lane], lane_mode,
                                         ValueType::word32);
            if (fused.index() == 1) return std::move(std::get<1>(fused));
            results[lane] = std::get<0>(fused);
        } else {
            auto computed = context.binary(opcode, a[lane], b[lane], lane_mode, ValueType::word32);
            if (computed.index() == 1) return std::move(std::get<1>(computed));
            results[lane] = std::get<0>(computed);
        }
    }
    ARMIGO_FLOAT_TRY(selector, decoded_field(instruction, DecodedField::half_mode));
    ARMIGO_FLOAT_TRY(destination, float_emit::destination_register(instruction));
    ARMIGO_FLOAT_TRY(packed, float_emit::pack_half_result(context, selector, results[0],
                                                          results[1], mode,
                                                          destination));
    return float_emit::write_destination(context, instruction, packed);
}
} // namespace

// HADD2/HADD232I.
Status emit_half_add(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    return emit_half_operation(context, Opcode::add, mode);
}

// HMUL2/HMUL232I.
Status emit_half_multiply(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    return emit_half_operation(context, Opcode::multiply, mode);
}

// HFMA2/HFMA232I.
Status emit_half_fma(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    return emit_half_operation(context, Opcode::fused_multiply_add, mode);
}
} // namespace armigo::shader::pipeline
