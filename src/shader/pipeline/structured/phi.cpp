// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo schedules whole edge bundles and
// splits their edges so copies cannot execute on an untaken successor.

#include "internal.h"

namespace armigo::shader::pipeline::structured_detail {
namespace {
struct Assignment { ValueId destination, source; SourceLocation location; };

void redirect(Terminator& terminator, BlockId old_target, BlockId new_target) {
    auto replace = [&](BlockId& target) { if (target == old_target) target = new_target; };
    if (auto* jump = std::get_if<Jump>(&terminator)) replace(jump->target);
    else if (auto* branch = std::get_if<Branch>(&terminator)) {
        replace(branch->when_true);
        replace(branch->when_false);
    } else if (auto* select = std::get_if<Switch>(&terminator)) {
        replace(select->fallback);
        for (auto& item : select->cases) replace(item.target);
    }
}

void parallel_copy(Program& p, BlockId edge, std::vector<Assignment> pending, Budget& budget) {
    std::erase_if(pending, [](const auto& assignment) { return assignment.destination == assignment.source; });
    while (!pending.empty()) {
        auto ready = pending.end();
        for (auto candidate = pending.begin(); candidate != pending.end(); ++candidate) {
            bool needed = false;
            for (const auto& other : pending) {
                budget.step();
                if (other.source == candidate->destination) { needed = true; break; }
            }
            if (!needed) { ready = candidate; break; }
        }
        if (ready != pending.end()) {
            copy(p, edge, ready->destination, ready->source, ready->location);
            pending.erase(ready);
            continue;
        }
        // Preserve an old destination before overwriting any member of its cycle.
        const auto saved = pending.front().destination;
        const auto temporary = value(p, p.values[saved.value].type);
        copy(p, edge, temporary, saved, pending.front().location);
        for (auto& assignment : pending) {
            budget.step();
            if (assignment.source == saved) assignment.source = temporary;
        }
    }
}
} // namespace

void destroy_phis(Program& p, Budget& budget) {
    const auto original_blocks = p.blocks.size();
    for (std::uint32_t target = 0; target < original_blocks; ++target) {
        budget.step();
        // Take ownership before appending blocks; arena growth invalidates references.
        auto phis = std::move(p.blocks[target].phis);
        p.blocks[target].phis.clear();
        if (phis.empty()) continue;
        const auto predecessors = p.blocks[target].predecessors;
        const auto function = p.blocks[target].function;
        for (const auto& phi : phis) p.values[phi.output.value].kind = ValueKind::mutable_local;
        for (auto predecessor : predecessors) {
            std::vector<Assignment> assignments;
            for (const auto& phi : phis) {
                for (const auto& input : phi.inputs) {
                    budget.step();
                    if (input.predecessor == predecessor) {
                        assignments.push_back({phi.output, input.value, phi.source});
                        break;
                    }
                }
            }
            const auto edge = block(p, function, p.blocks[predecessor.value].source);
            p.blocks[edge.value].terminator = Jump{BlockId{target}};
            redirect(p.blocks[predecessor.value].terminator, BlockId{target}, edge);
            parallel_copy(p, edge, std::move(assignments), budget);
        }
    }
    rebuild_predecessors(p, budget);
}
} // namespace armigo::shader::pipeline::structured_detail
