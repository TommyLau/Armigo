// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"

namespace armigo::shader::pipeline {
// FMNMX: the minimum/maximum choice is predicate-controlled, so both results
// are computed and selected; guest NaN and flush behavior lives in the
// minimum/maximum operations' NumericMode.
Status emit_float_min_max(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(a, float_emit::modified_source(context, instruction, SourceRole::a, mode));
    ARMIGO_FLOAT_TRY(b, float_emit::modified_source(context, instruction, SourceRole::b, mode));
    ARMIGO_FLOAT_TRY(predicate, float_emit::predicate_operand(context, instruction));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));
    const ValueType storage = storage_type(mode.scalar);
    auto minimum = context.binary(Opcode::minimum, a, b, float_emit::without_saturation(mode), storage);
    if (minimum.index() == 1) return std::move(std::get<1>(minimum));
    auto maximum = context.binary(Opcode::maximum, a, b, float_emit::without_saturation(mode), storage);
    if (maximum.index() == 1) return std::move(std::get<1>(maximum));
    auto result = context.select(predicate, std::get<0>(minimum), std::get<0>(maximum), storage);
    if (result.index() == 1) return std::move(std::get<1>(result));
    const Status written = float_emit::write_destination(context, instruction, std::get<0>(result));
    if (written.index() == 1) return written;
    return float_emit::write_float_flags(context, std::get<0>(result), mode, cc);
}
} // namespace armigo::shader::pipeline
