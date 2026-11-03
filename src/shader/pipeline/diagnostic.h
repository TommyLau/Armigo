// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "fact_types.h"
#include "instruction_name.h"

#include <optional>
#include <string>
#include <variant>
#include <utility>

namespace armigo::shader::pipeline {
enum class Phase : std::uint8_t { decode, emit_ir, construct_ssa, legalize, structure, plan_resources, emit_msl };
enum class DiagnosticCode : std::uint8_t {
    malformed_input, unsupported_instruction, unsupported_modifier, missing_fact,
    invalid_fact, conflicting_fact, invalid_ir, undefined_read, unsupported_capability,
    unresolved_resource, limit_exceeded, duplicate_dispatch, unimplemented
};
struct Diagnostic {
    Phase phase{Phase::decode};
    DiagnosticCode code{DiagnosticCode::malformed_input};
    std::optional<Pc> pc;
    std::optional<InstructionName> instruction;
    std::string modifier;
    std::optional<FactKey> missing_fact;
    std::string message;
};
struct SourceLocation {
    std::optional<Pc> pc;
    std::optional<InstructionName> instruction;
};
template <class T> using Result = std::variant<T, Diagnostic>;
using Status = Result<std::monostate>;
[[nodiscard]] inline Diagnostic diagnose(Phase phase, DiagnosticCode code,
                                         SourceLocation source, std::string message) {
    return {phase, code, source.pc, source.instruction, {}, {}, std::move(message)};
}
static_assert(std::is_same_v<Result<int>, std::variant<int, Diagnostic>>);
} // namespace armigo::shader::pipeline
