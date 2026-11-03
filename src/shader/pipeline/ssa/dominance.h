// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// Depth-first postorder ranking and immediate dominators per function using
// the Cooper–Harvey–Kennedy scheme from "A Simple, Fast Dominance Algorithm",
// plus dominance frontiers for phi placement. Armigo stores the results in
// owned vectors indexed by program BlockId instead of managed node links.

#pragma once
#include "pipeline/ir.h"

#include <vector>

namespace armigo::shader::pipeline {
struct Dominators {
    std::vector<BlockId> postorder;              // Reached blocks in depth-first postorder.
    std::vector<std::uint32_t> postorder_number; // Per block; invalid_index until reached.
    std::vector<BlockId> immediate_dominator;    // Per block; the entry dominates itself.
};

// Ranks the blocks of one function and computes immediate dominators. Fails
// with invalid_ir when the function owns unreachable blocks and with
// limit_exceeded when the shared analysis step budget runs out. steps is
// advanced so callers can charge one budget across all SSA work.
[[nodiscard]] Result<Dominators> find_dominators(const Program& program, const Function& function,
                                                 Limits limits, std::uint64_t& steps);

// Dominance frontiers per program block for one function. Only blocks with at
// least two predecessors receive frontier entries.
[[nodiscard]] Result<std::vector<std::vector<BlockId>>> find_dominance_frontiers(
    const Program& program, const Function& function, const Dominators& dominators,
    Limits limits, std::uint64_t& steps);
} // namespace armigo::shader::pipeline
