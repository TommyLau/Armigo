// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "dispatch.h"

namespace armigo::shader::pipeline {
Status dispatch_instruction(EmitterContext& context) {
    const auto index = static_cast<std::size_t>(context.instruction().name);
    if (index >= instruction_emitters.size() || instruction_emitters[index] == nullptr) {
        return context.error(DiagnosticCode::unsupported_instruction, "No instruction emitter is registered");
    }
    return instruction_emitters[index](context);
}
Status dispatch_msl(MslContext& context, OperationId id) {
    if (id.value >= context.program().ir.operations.size()) {
        return diagnose(Phase::emit_msl, DiagnosticCode::invalid_ir, {}, "Invalid operation ID for MSL dispatch");
    }
    const auto index = static_cast<std::size_t>(context.operation(id).opcode);
    if (index >= msl_generators.size() || msl_generators[index] == nullptr) {
        return context.error(id, DiagnosticCode::unsupported_instruction, "No MSL opcode generator is registered");
    }
    return msl_generators[index](context, id);
}
} // namespace armigo::shader::pipeline
