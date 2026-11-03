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
#include "pipeline/ir.h"

#include <cstdint>
#include <optional>

namespace armigo::shader::pipeline {
// The provisional SSA and saved register program have identical block IDs.
// Tracks predicate identities, Boolean constants and selected phi inputs.
// Returns no program when no guard can be proven redundant.
[[nodiscard]] Result<std::optional<Program>> specialize_guard_paths(
    const Program& registers, const Program& provisional, std::uint64_t& steps);
// Proves missing phi inputs cannot reach any executed instruction, then
// replaces only those inputs with existing dominating, type-matching values.
// All CFG edges and operations remain intact, including collective execution.
[[nodiscard]] Result<std::optional<Program>> resolve_guarded_phi_inputs(
    const Program& provisional, std::uint64_t& steps);
} // namespace armigo::shader::pipeline
