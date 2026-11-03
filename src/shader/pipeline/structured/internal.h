// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Checked arena allocation is Armigo code.

#pragma once
#include "pipeline/phases.h"

#include <algorithm>
#include <string>

namespace armigo::shader::pipeline::structured_detail {
[[noreturn]] inline void fail(std::string message, SourceLocation source = {},
                             DiagnosticCode code = DiagnosticCode::invalid_ir) {
    throw diagnose(Phase::structure, code, source, std::move(message));
}
class Budget {
public:
    explicit Budget(std::uint64_t limit) : remaining_(limit) {}
    void step(std::uint64_t count = 1) {
        if (count > remaining_) fail("Structurization analysis limit exceeded", {}, DiagnosticCode::limit_exceeded);
        remaining_ -= count;
    }
private:
    std::uint64_t remaining_;
};
inline void capacity(std::size_t size, std::uint32_t limit, const char* name) {
    if (size >= limit || size >= invalid_index)
        fail(std::string("Structurization ") + name + " limit exceeded", {}, DiagnosticCode::limit_exceeded);
}
inline ValueId value(Program& p, ValueType type, ValueKind kind = ValueKind::temporary) {
    capacity(p.values.size(), p.limits.values, "value");
    ValueId id{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({type, kind});
    return id;
}
inline void operation(Program& p, BlockId block, Operation op) {
    capacity(p.operations.size(), p.limits.operations, "operation");
    if (op.inputs.size() + op.outputs.size() > p.limits.operands_per_operation)
        fail("Structurization operand limit exceeded", op.source, DiagnosticCode::limit_exceeded);
    OperationId id{static_cast<std::uint32_t>(p.operations.size())};
    p.operations.push_back(std::move(op));
    p.blocks[block.value].operations.push_back(id);
}
inline void copy(Program& p, BlockId block, ValueId destination, Operand source, SourceLocation location = {}) {
    operation(p, block, {location, Opcode::copy, Effects::none, {}, {source}, {destination}, {}});
}
inline BlockId block(Program& p, FunctionId function, SourceLocation source = {}) {
    capacity(p.blocks.size(), p.limits.blocks, "block");
    BlockId id{static_cast<std::uint32_t>(p.blocks.size())};
    p.blocks.push_back({function, source, {}, {}, {}, Unreachable{}});
    p.functions[function.value].blocks.push_back(id);
    return id;
}
inline void rebuild_predecessors(Program& p, Budget& budget) {
    for (auto& b : p.blocks) b.predecessors.clear();
    std::uint64_t edges = 0;
    for (std::uint32_t b = 0; b < p.blocks.size(); ++b) {
        budget.step();
        for (auto target : successors(p.blocks[b].terminator)) {
            budget.step();
            if (++edges > p.limits.edges) fail("Structurization edge limit exceeded", {}, DiagnosticCode::limit_exceeded);
            p.blocks[target.value].predecessors.push_back(BlockId{b});
        }
    }
}
void destroy_phis(Program&, Budget&);
void build_regions(StructuredProgram&, Budget&);
Status validate_structured(const StructuredProgram&, std::uint64_t analysis_limit);
} // namespace armigo::shader::pipeline::structured_detail
