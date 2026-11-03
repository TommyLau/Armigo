// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structured diagnostics reported by shader validation, emission, and
// reference evaluation.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace armigo::shader {

enum class DiagnosticCode : std::uint16_t {
    // Program declarations.
    invalid_entry_point,
    no_outputs,
    too_many_parameters,
    too_many_bindings,
    too_many_instructions,
    unknown_type,

    // Instruction encoding.
    unknown_opcode,
    unexpected_field,

    // Value references.
    operand_missing,
    operand_out_of_range,
    operand_not_dominating,
    operand_has_no_value,
    operand_type_mismatch,

    // Instruction semantics.
    result_type_mismatch,
    output_type_mismatch,
    nonfinite_constant,
    parameter_out_of_range,
    input_out_of_range,
    output_out_of_range,
    output_stored_twice,
    output_never_stored,

    // Reference evaluation arguments.
    parameter_count_mismatch,
    input_count_mismatch,
    input_too_short,
    invalid_subnormal_mode,
};

const char* to_string(DiagnosticCode code) noexcept;

struct Diagnostic {
    // Sentinel for diagnostics that concern declarations or arguments rather
    // than a single instruction.
    static constexpr std::uint32_t kNoInstruction =
        std::numeric_limits<std::uint32_t>::max();

    DiagnosticCode code;
    std::uint32_t instruction = kNoInstruction;
    std::string message;
};

// Formats a diagnostic as a single deterministic English line.
std::string format(const Diagnostic& diagnostic);

}  // namespace armigo::shader
