// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Pure static metadata admission for one decoded forward instruction.
#pragma once

#include <armigo/shader/graphics.h>
#include <armigo/shader/maxwell.h>

#include <optional>

namespace armigo::shader::graphics {

struct Builder;

namespace detail {

// Validates the syntactic and profile metadata of one decoded instruction
// against the admission contracts of the existing Builder and its registered
// family lowerers, without executing any of them. The context must already
// be initialized through begin_graph; the validator queries only the
// program's stage, SPH header, and declared-interface rules and never
// mutates the builder, appends nodes, or reads register or stored values.
// Operand definedness is deliberately not checked: a guard-false
// instruction is admitted on metadata even though its registers are
// undefined, because definedness belongs to the actually executed lowering
// path. Unknown and non-implemented families are rejected explicitly, so a
// false-guard instruction can never be skipped as an unvalidated no-op.
// Returns the same Error code, pc, and detail the Builder would report,
// or nullopt when the instruction is admissible.
std::optional<Error> validate_forward_instruction_metadata(
    const maxwell::Instruction& instruction, const Builder& context);

}  // namespace detail
}  // namespace armigo::shader::graphics
