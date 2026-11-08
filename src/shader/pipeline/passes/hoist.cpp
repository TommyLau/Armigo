// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// A predicated instruction is lowered to a branch around its expansion, so a
// predicated texture sample sits alone in the guarded block. Sampling has no
// side effects: a filtered sample whose inputs all come from outside that block
// runs just as well before the branch, and only the uses of its result stay
// guarded. A blur of 63 predicated taps then issues its fetches together
// instead of waiting out each one behind its own branch.
//
// The win comes from many guarded taps in one stage, and a hoisted sample runs
// even where its guard is false, so the pass is limited to that pattern: a
// function needs at least min_guarded_samples eligible samples, and each guard
// receives at most max_moved_accesses native accesses from all the blocks it
// guards, taken in dominance order. A guarded block qualifies only when it
// reconverges with the guard's other side, so a sample never moves from a block
// that runs once (a loop exit) into one that may run many times.
//
// Explicit-LOD and gradient samples and gathers move; implicit-LOD and bias
// samples stay. On the host GPU a guarded implicit-LOD sample in a partial quad
// takes a finer mip than the same sample run by the whole quad, while the moved
// modes draw identical pixels either way (tests/shader/pipeline/
// guarded_sample_gpu.mm); see the project documentation. A title may opt in to
// moving implicit-LOD and bias samples as well, accepting whole-quad
// derivatives on the partial quads along a guard's edge.

#include "internal.h"
#include <algorithm>
#include <optional>
#include <vector>

namespace armigo::shader::pipeline::passes {
namespace {
using detail::Budget;

constexpr std::size_t min_guarded_samples = 8;
constexpr std::uint32_t max_moved_accesses = 4;

// Samples with an explicit LOD or gradients, and gathers: a texel fetch can
// read out of bounds when its guard is false, queries are cheap where they are,
// and an implicit-LOD or bias sample would take its LOD from lanes the guard
// leaves out, unless the title opted in (`implicit`).
bool hoistable(const Operation& op, bool implicit) {
    if (op.opcode != Opcode::texture_sample) return false;
    const auto* texture = std::get_if<TextureInfo>(&op.info);
    if (!texture || texture->atomic) return false;
    switch (texture->mode) {
    case TextureMode::sample_level: case TextureMode::sample_gradients: case TextureMode::gather: break;
    case TextureMode::sample_implicit: case TextureMode::sample_bias: if (implicit) break; return false;
    default: return false;
    }
    return (static_cast<std::uint32_t>(op.effects) &
            ~static_cast<std::uint32_t>(Effects::texture | Effects::derivative)) == 0;
}

// Native texture accesses one sample becomes: a gather with an offset per texel
// is four gathers, and a non-comparison sample may gain a custom-border
// coverage sample once resources are planned.
std::uint32_t native_accesses(const Operation& op) {
    const auto& texture = std::get<TextureInfo>(op.info);
    if (texture.mode == TextureMode::gather)
        return !texture.compare && texture.offsets.size() > 3 ? 4 : 1;
    return texture.compare ? 1 : 2;
}

// An operation a later sample must not move above: one that writes memory or
// I/O, discards, synchronizes or calls.
bool orders_samples(const Operation& op) {
    constexpr auto ordered = Effects::write_memory | Effects::write_io | Effects::discard | Effects::barrier |
        Effects::atomic | Effects::subgroup | Effects::call;
    return (op.effects & ordered) != Effects::none;
}

struct Hoister {
    Program& p;
    Budget& budget;
    bool implicit;
    std::vector<std::optional<BlockId>> defined_in;

    // The block's guard: its only predecessor, ending in a two-way branch whose
    // sides reconverge. The block jumps to the branch's other successor (the
    // shape of a predicated instruction) or to where that successor jumps (an
    // if/else); then it runs at most once per run of its guard.
    std::optional<BlockId> guard_of(std::uint32_t b) const {
        const auto& block = p.blocks[b];
        if (block.predecessors.size() != 1 || !block.phis.empty()) return std::nullopt;
        const auto guard = block.predecessors.front();
        if (guard.value == b) return std::nullopt;
        const auto* branch = std::get_if<Branch>(&p.blocks[guard.value].terminator);
        if (!branch || branch->when_true == branch->when_false ||
            (branch->when_true.value != b && branch->when_false.value != b)) return std::nullopt;
        const auto other = branch->when_true.value == b ? branch->when_false : branch->when_true;
        const auto* jump = std::get_if<Jump>(&block.terminator);
        if (!jump) return std::nullopt;
        if (jump->target != other) {
            const auto* next = std::get_if<Jump>(&p.blocks[other.value].terminator);
            if (!next || next->target != jump->target) return std::nullopt;
        }
        return guard;
    }
    // Every function's blocks in reverse post-order from its entry, following
    // each terminator's own successor order: a guard comes before the blocks
    // it guards, and sibling arms keep their branch order whatever their ids.
    std::vector<std::uint32_t> dominance_order() {
        std::vector<std::uint32_t> result;
        std::vector<bool> seen(p.blocks.size());
        const auto next_blocks = [](const Terminator& terminator) {
            std::vector<BlockId> next;
            if (const auto* jump = std::get_if<Jump>(&terminator)) next.push_back(jump->target);
            if (const auto* branch = std::get_if<Branch>(&terminator)) {
                next.push_back(branch->when_true);
                next.push_back(branch->when_false);
            }
            if (const auto* select = std::get_if<Switch>(&terminator)) {
                for (const auto& item : select->cases) next.push_back(item.target);
                next.push_back(select->fallback);
            }
            return next;
        };
        for (const auto& function : p.functions) {
            if (function.entry.value >= p.blocks.size() || seen[function.entry.value]) continue;
            std::vector<std::uint32_t> post;
            std::vector<std::pair<std::uint32_t, std::size_t>> stack{{function.entry.value, 0}};
            seen[function.entry.value] = true;
            while (!stack.empty()) {
                budget.spend();
                const auto block = stack.back().first;
                const auto next = next_blocks(p.blocks[block].terminator);
                if (stack.back().second == next.size()) {
                    post.push_back(block);
                    stack.pop_back();
                    continue;
                }
                const auto target = next[stack.back().second++].value;
                if (target < p.blocks.size() && !seen[target]) {
                    seen[target] = true;
                    stack.emplace_back(target, 0);
                }
            }
            result.insert(result.end(), post.rbegin(), post.rend());
        }
        return result;
    }
    // Samples of the block that could run before its guard, in order, up to the
    // first operation that orders them.
    std::vector<OperationId> eligible(std::uint32_t b) {
        std::vector<OperationId> result;
        for (const auto id : p.blocks[b].operations) {
            budget.spend();
            const auto& op = p.operations[id.value];
            const bool outside = std::all_of(op.inputs.begin(), op.inputs.end(), [&](const Operand& input) {
                const auto* value = std::get_if<ValueId>(&input);
                return !value || defined_in[value->value] != std::optional<BlockId>{BlockId{b}};
            });
            if (hoistable(op, implicit) && outside) result.push_back(id);
            else if (orders_samples(op)) break;
        }
        return result;
    }
    void run() {
        defined_in.assign(p.values.size(), std::nullopt);
        for (std::uint32_t b = 0; b < p.blocks.size(); ++b) {
            budget.spend(p.blocks[b].phis.size() + p.blocks[b].operations.size() + 1);
            for (const auto& phi : p.blocks[b].phis) defined_in[phi.output.value] = BlockId{b};
            for (const auto id : p.blocks[b].operations)
                for (const auto& output : p.operations[id.value].outputs)
                    if (const auto* value = std::get_if<ValueId>(&output)) defined_in[value->value] = BlockId{b};
        }
        std::vector<std::size_t> per_function(p.functions.size());
        for (std::uint32_t b = 0; b < p.blocks.size(); ++b)
            if (guard_of(b) && p.blocks[b].function.value < per_function.size())
                per_function[p.blocks[b].function.value] += eligible(b).size();
        // Native accesses each guard has received so far.
        std::vector<std::uint32_t> received(p.blocks.size());
        for (const auto b : dominance_order()) {
            const auto guard = guard_of(b);
            const auto function = p.blocks[b].function.value;
            if (!guard || function >= per_function.size() || per_function[function] < min_guarded_samples) continue;
            std::vector<OperationId> moved;
            auto& accesses = received[guard->value];
            for (const auto id : eligible(b)) {
                const auto cost = native_accesses(p.operations[id.value]);
                if (accesses + cost > max_moved_accesses) break;
                accesses += cost;
                moved.push_back(id);
            }
            if (moved.empty()) continue;
            auto& block = p.blocks[b];
            std::vector<OperationId> kept;
            for (const auto id : block.operations)
                if (std::find(moved.begin(), moved.end(), id) == moved.end()) kept.push_back(id);
            auto& before = p.blocks[guard->value].operations;
            before.insert(before.end(), moved.begin(), moved.end());
            block.operations = std::move(kept);
            for (const auto id : moved)
                for (const auto& output : p.operations[id.value].outputs)
                    if (const auto* value = std::get_if<ValueId>(&output)) defined_in[value->value] = *guard;
        }
    }
};
} // namespace

Status hoist_guarded_samples(Program& program, bool implicit) {
    Program candidate = program;
    try {
        Budget budget(candidate);
        Hoister{candidate, budget, implicit, {}}.run();
    } catch (const Diagnostic& error) { return error; }
    if (auto status = validate(candidate, IrForm::legal, Phase::legalize); std::holds_alternative<Diagnostic>(status))
        return status;
    program = std::move(candidate);
    return std::monostate{};
}
} // namespace armigo::shader::pipeline::passes
