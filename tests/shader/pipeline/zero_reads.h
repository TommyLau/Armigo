// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Shared check for SSA zero definitions of undefined reads.

#pragma once

#include "pipeline/ir.h"

#include <cstddef>
#include <variant>

namespace armigo::shader::pipeline::test {

// Count of explicit zero definitions that SSA construction placed at the
// function entry for reads no definition reaches (owner decision, 2026-10-24).
// A read that a guarded-path repair wrongly justified would have none.
inline std::size_t entry_zero_reads(const Program& program, ValueType type) {
    if (program.functions.size() != 1) return 0;
    std::size_t count = 0;
    for (const auto id : program.blocks[program.functions.front().entry.value].operations) {
        const auto& operation = program.operations[id.value];
        if (operation.opcode != Opcode::copy || operation.inputs.size() != 1) continue;
        const auto* literal = std::get_if<LiteralBits>(&operation.inputs[0]);
        if (literal && literal->bits == 0 && literal->type == type) ++count;
    }
    return count;
}

} // namespace armigo::shader::pipeline::test
