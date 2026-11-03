// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/diagnostics.h>

namespace armigo::shader {

const char* to_string(DiagnosticCode code) noexcept {
    switch (code) {
    case DiagnosticCode::invalid_entry_point: return "invalid_entry_point";
    case DiagnosticCode::no_outputs: return "no_outputs";
    case DiagnosticCode::too_many_parameters: return "too_many_parameters";
    case DiagnosticCode::too_many_bindings: return "too_many_bindings";
    case DiagnosticCode::too_many_instructions: return "too_many_instructions";
    case DiagnosticCode::unknown_type: return "unknown_type";
    case DiagnosticCode::unknown_opcode: return "unknown_opcode";
    case DiagnosticCode::unexpected_field: return "unexpected_field";
    case DiagnosticCode::operand_missing: return "operand_missing";
    case DiagnosticCode::operand_out_of_range: return "operand_out_of_range";
    case DiagnosticCode::operand_not_dominating: return "operand_not_dominating";
    case DiagnosticCode::operand_has_no_value: return "operand_has_no_value";
    case DiagnosticCode::operand_type_mismatch: return "operand_type_mismatch";
    case DiagnosticCode::result_type_mismatch: return "result_type_mismatch";
    case DiagnosticCode::output_type_mismatch: return "output_type_mismatch";
    case DiagnosticCode::nonfinite_constant: return "nonfinite_constant";
    case DiagnosticCode::parameter_out_of_range: return "parameter_out_of_range";
    case DiagnosticCode::input_out_of_range: return "input_out_of_range";
    case DiagnosticCode::output_out_of_range: return "output_out_of_range";
    case DiagnosticCode::output_stored_twice: return "output_stored_twice";
    case DiagnosticCode::output_never_stored: return "output_never_stored";
    case DiagnosticCode::parameter_count_mismatch:
        return "parameter_count_mismatch";
    case DiagnosticCode::input_count_mismatch: return "input_count_mismatch";
    case DiagnosticCode::input_too_short: return "input_too_short";
    case DiagnosticCode::invalid_subnormal_mode: return "invalid_subnormal_mode";
    }
    return "<unknown diagnostic>";
}

std::string format(const Diagnostic& diagnostic) {
    std::string text = to_string(diagnostic.code);
    if (diagnostic.instruction != Diagnostic::kNoInstruction) {
        text += " at instruction ";
        text += std::to_string(diagnostic.instruction);
    }
    text += ": ";
    text += diagnostic.message;
    return text;
}

}  // namespace armigo::shader
