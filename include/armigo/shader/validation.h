// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural and type validation of shader IR programs.
#pragma once

#include <armigo/shader/diagnostics.h>
#include <armigo/shader/ir.h>

#include <vector>

namespace armigo::shader {

// Validates `program` and returns every problem found, ordered by declaration
// checks first and then by instruction index. An empty result means the
// program is valid. Operands that are themselves invalid do not produce
// additional type diagnostics.
std::vector<Diagnostic> validate(const Program& program);

// Returns true when `name` is accepted as a kernel entry point name: an ASCII
// identifier of at most kMaxEntryPointLength characters that is not reserved
// by C++, the Metal Shading Language, or the generated code.
bool is_valid_entry_point(const std::string& name);

}  // namespace armigo::shader
