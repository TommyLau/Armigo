// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/ssa/dominance.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
bool charge(std::uint64_t& steps, const Limits& limits) {
    return ++steps <= limits.analysis_steps;
}

Diagnostic work_exhausted(std::string_view what) {
    return diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, {},
                    std::string(what) + " exceeded its analysis work limit");
}
} // namespace

Result<Dominators> find_dominators(const Program& program, const Function& function,
                                   Limits limits, std::uint64_t& steps) {
    if (function.entry.value >= program.blocks.size())
        return diagnose(Phase::construct_ssa, DiagnosticCode::invalid_ir, {}, "Function entry block is invalid");
    Dominators info;
    info.postorder_number.assign(program.blocks.size(), invalid_index);
    info.immediate_dominator.assign(program.blocks.size(), BlockId{});
    std::vector<std::vector<BlockId>> edges(program.blocks.size());
    for (std::size_t index = 0; index < program.blocks.size(); ++index)
        edges[index] = successors(program.blocks[index].terminator);

    // Depth-first postorder from the entry determines the CFG ranking.
    std::vector<bool> visited(program.blocks.size(), false);
    std::vector<std::pair<BlockId, std::size_t>> stack;
    visited[function.entry.value] = true;
    stack.emplace_back(function.entry, 0);
    while (!stack.empty()) {
        if (!charge(steps, limits)) return work_exhausted("Dominance search");
        auto& [block, next] = stack.back();
        const auto& targets = edges[block.value];
        if (next < targets.size()) {
            const auto target = targets[next++];
            if (!visited[target.value]) { visited[target.value] = true; stack.emplace_back(target, 0); }
        } else {
            info.postorder_number[block.value] = static_cast<std::uint32_t>(info.postorder.size());
            info.postorder.push_back(block);
            stack.pop_back();
        }
    }
    if (info.postorder.size() != function.blocks.size())
        return diagnose(Phase::construct_ssa, DiagnosticCode::invalid_ir, {}, "Function contains unreachable blocks");

    // Intersect walks climb toward the entry, whose postorder number is the
    // largest, so every loop terminates once the entry dominates itself.
    bool exhausted = false;
    auto intersect = [&](BlockId first, BlockId second) {
        while (first != second && !exhausted) {
            while (info.postorder_number[first.value] < info.postorder_number[second.value]) {
                if (!charge(steps, limits)) { exhausted = true; break; }
                first = info.immediate_dominator[first.value];
            }
            while (!exhausted && info.postorder_number[second.value] < info.postorder_number[first.value]) {
                if (!charge(steps, limits)) { exhausted = true; break; }
                second = info.immediate_dominator[second.value];
            }
        }
        return first;
    };

    info.immediate_dominator[function.entry.value] = function.entry;
    bool modified = true;
    while (modified && !exhausted) {
        modified = false;
        // The entry is the last postorder block and already has its dominator.
        for (std::size_t index = info.postorder.size() - 1; index-- > 0;) {
            if (!charge(steps, limits)) { exhausted = true; break; }
            const auto block = info.postorder[index];
            BlockId new_idom;
            for (const auto predecessor : program.blocks[block.value].predecessors) {
                if (!info.immediate_dominator[predecessor.value].valid()) continue;
                if (!new_idom.valid()) { new_idom = predecessor; continue; }
                new_idom = intersect(predecessor, new_idom);
            }
            if (exhausted) break;
            if (new_idom.valid() && new_idom != info.immediate_dominator[block.value]) {
                info.immediate_dominator[block.value] = new_idom;
                modified = true;
            }
        }
    }
    if (exhausted) return work_exhausted("Dominance iteration");
    return info;
}

Result<std::vector<std::vector<BlockId>>> find_dominance_frontiers(
    const Program& program, const Function& function, const Dominators& dominators,
    Limits limits, std::uint64_t& steps) {
    std::vector<std::vector<BlockId>> frontiers(program.blocks.size());
    for (const auto block : function.blocks) {
        const auto& predecessors = program.blocks[block.value].predecessors;
        if (predecessors.size() < 2) continue;
        for (const auto predecessor : predecessors) {
            auto cursor = predecessor;
            while (cursor != dominators.immediate_dominator[block.value]) {
                if (!charge(steps, limits)) return work_exhausted("Dominance frontier search");
                auto& cells = frontiers[cursor.value];
                if (std::find(cells.begin(), cells.end(), block) == cells.end()) cells.push_back(block);
                cursor = dominators.immediate_dominator[cursor.value];
            }
        }
    }
    return frontiers;
}
} // namespace armigo::shader::pipeline
