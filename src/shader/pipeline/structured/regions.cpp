// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Pending-target predicates guard skipped ranges as in goto elimination. Armigo
// derives nested loops from graph components instead of assuming label order.

#include "internal.h"

#include <functional>
#include <map>
#include <set>

namespace armigo::shader::pipeline::structured_detail {
namespace {
Operand boolean(bool value) { return LiteralBits{value ? 1U : 0U, ValueType::predicate}; }

void hoist_cross_block_values(Program& p, Budget& budget) {
    std::vector<BlockId> definitions(p.values.size());
    std::vector<bool> parameters(p.values.size());
    for (const auto& function : p.functions)
        for (auto parameter : function.parameters) parameters[parameter.value] = true;
    for (std::uint32_t b = 0; b < p.blocks.size(); ++b)
        for (auto id : p.blocks[b].operations)
            for (const auto& output : p.operations[id.value].outputs) {
                budget.step();
                definitions[std::get<ValueId>(output).value] = BlockId{b};
            }
    for (std::uint32_t b = 0; b < p.blocks.size(); ++b) {
        auto use = [&](const Operand& operand) {
            budget.step();
            if (const auto* id = std::get_if<ValueId>(&operand); id && !parameters[id->value] &&
                definitions[id->value] != BlockId{b}) p.values[id->value].kind = ValueKind::mutable_local;
        };
        for (auto id : p.blocks[b].operations)
            for (const auto& input : p.operations[id.value].inputs) use(input);
        const auto& terminator = p.blocks[b].terminator;
        if (const auto* branch = std::get_if<Branch>(&terminator)) use(branch->condition);
        if (const auto* select = std::get_if<Switch>(&terminator)) use(select->selector);
        if (const auto* result = std::get_if<Return>(&terminator))
            for (const auto& input : result->values) use(input);
    }
}

// A multiway branch is a sequence of comparisons, not dispatch state. Phi
// copies already live on split edges, so introducing these test blocks cannot
// move a copy onto an untaken case.
void lower_switches(Program& p, Budget& budget) {
    const auto count = p.blocks.size();
    for (std::uint32_t b = 0; b < count; ++b) {
        budget.step();
        const auto* select = std::get_if<Switch>(&p.blocks[b].terminator);
        if (!select) continue;
        const auto selection = *select;
        const auto function = p.blocks[b].function;
        const auto source = p.blocks[b].source;
        std::map<BlockId, Operand> conditions;
        for (const auto& item : selection.cases) {
            budget.step();
            if (item.target == selection.fallback) continue;
            const auto equal = value(p, ValueType::predicate);
            operation(p, BlockId{b}, {source, Opcode::compare, Effects::none,
                {Domain::unsigned_integer, ScalarType::u32},
                {selection.selector, LiteralBits{item.bits, ValueType::word32}},
                {equal}, ComparisonInfo{Compare::equal, false}});
            const auto found = conditions.find(item.target);
            if (found == conditions.end()) conditions.emplace(item.target, equal);
            else {
                const auto combined = value(p, ValueType::predicate);
                operation(p, BlockId{b}, {source, Opcode::logical_or, Effects::none,
                    {Domain::boolean, ScalarType::boolean}, {found->second, equal}, {combined}, {}});
                found->second = combined;
            }
        }
        auto current = BlockId{b};
        for (const auto& [target, condition] : conditions) {
            const auto next = block(p, function, source);
            p.blocks[current.value].terminator = Branch{condition, target, next};
            current = next;
        }
        p.blocks[current.value].terminator = Jump{selection.fallback};
    }
    rebuild_predecessors(p, budget);
}

// Recognize single-entry regions before introducing any dispatch state. Each
// scope is a DAG after its child loops are collapsed and its own backedges are
// replaced by continues. Joins therefore cannot accidentally select a backedge.
class DirectBuilder {
public:
    DirectBuilder(StructuredProgram& result, Budget& budget) : result(result), p(result.ir), budget(budget) {}
    struct Unsupported {};
    RegionId run(FunctionId function) {
        const auto& f = p.functions[function.value];
        budget.step(p.blocks.size());
        emitted.assign(p.blocks.size(), false);
        auto root = scope(f.blocks, f.entry, {}, {}, 0);
        for (auto id : f.blocks) if (!emitted[id.value]) throw Unsupported{};
        return root;
    }
private:
    StructuredProgram& result;
    Program& p;
    Budget& budget;
    std::vector<bool> emitted;
    std::vector<RegionId> loops;
    RegionId add(Region payload) {
        budget.step();
        capacity(result.regions.size(), p.limits.regions, "region");
        RegionId id{static_cast<std::uint32_t>(result.regions.size())};
        result.regions.push_back(std::move(payload));
        return id;
    }
    RegionId sequence(std::vector<RegionId> children = {}) { return add(SequenceRegion{std::move(children)}); }
    void depth_check(std::uint32_t depth) {
        if (depth > p.limits.nesting_depth)
            fail("Direct region nesting limit exceeded", {}, DiagnosticCode::limit_exceeded);
    }
    RegionId scope(const std::vector<BlockId>& nodes, BlockId entry, BlockId backedge,
                   BlockId exit, std::uint32_t depth) {
        depth_check(depth);
        // Kosaraju on the induced graph, excluding this loop's backedges.
        budget.step(3 * static_cast<std::uint64_t>(p.blocks.size()));
        std::vector<bool> member(p.blocks.size()), seen(p.blocks.size());
        for (auto node : nodes) member[node.value] = true;
        auto edges = [&](BlockId node) {
            auto edges = successors(p.blocks[node.value].terminator);
            std::erase_if(edges, [&](BlockId target) { return target == backedge || !member[target.value]; });
            return edges;
        };
        std::vector<BlockId> finish;
        for (auto node : nodes) {
            if (seen[node.value]) continue;
            std::vector<std::pair<BlockId, bool>> stack{{node, false}};
            while (!stack.empty()) {
                budget.step();
                auto [current, leaving] = stack.back(); stack.pop_back();
                if (leaving) { finish.push_back(current); continue; }
                if (seen[current.value]) continue;
                seen[current.value] = true;
                stack.emplace_back(current, true);
                for (auto target : edges(current)) stack.emplace_back(target, false);
            }
        }
        std::vector<std::vector<BlockId>> groups;
        std::vector<std::size_t> owner(p.blocks.size(), nodes.size());
        for (auto cursor = finish.rbegin(); cursor != finish.rend(); ++cursor) {
            if (owner[cursor->value] != nodes.size()) continue;
            const auto index = groups.size();
            groups.emplace_back();
            std::vector<BlockId> stack{*cursor};
            owner[cursor->value] = index;
            while (!stack.empty()) {
                budget.step();
                const auto current = stack.back(); stack.pop_back();
                groups.back().push_back(current);
                if (current == backedge) continue;
                for (auto pred : p.blocks[current.value].predecessors) {
                    budget.step();
                    if (member[pred.value] && owner[pred.value] == nodes.size()) {
                        owner[pred.value] = index;
                        stack.push_back(pred);
                    }
                }
            }
        }
        struct Unit { BlockId header, next; bool loop{}; std::vector<std::size_t> outgoing; };
        std::vector<Unit> units(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            auto& unit = units[i];
            const auto& group = groups[i];
            unit.header = group.front();
            const auto successors = edges(unit.header);
            unit.loop = group.size() > 1 || std::find(successors.begin(), successors.end(), unit.header) != successors.end();
            std::set<BlockId> entries, exits;
            for (auto node : group) {
                if (node == entry) entries.insert(node);
                for (auto pred : p.blocks[node.value].predecessors) {
                    budget.step();
                    if (owner[pred.value] != i) entries.insert(node);
                }
                for (auto target : armigo::shader::pipeline::successors(p.blocks[node.value].terminator)) {
                    budget.step();
                    if (target == backedge || !member[target.value] || owner[target.value] != i)
                        exits.insert(target);
                }
            }
            if (unit.loop) {
                if (entries.size() != 1 || exits.size() > 1) throw Unsupported{};
                unit.header = *entries.begin();
                if (!exits.empty()) unit.next = *exits.begin();
            }
            for (auto target : exits) {
                if (target != backedge && member[target.value]) {
                    if (owner[target.value] <= i) throw Unsupported{};
                    unit.outgoing.push_back(owner[target.value]);
                }
            }
        }
        auto reachable = [&](BlockId target, BlockId stop) {
            budget.step(groups.size());
            std::vector<bool> reachable(groups.size());
            std::vector<std::size_t> stack;
            if (target != backedge && target.valid() && member[target.value]) stack.push_back(owner[target.value]);
            while (!stack.empty()) {
                budget.step();
                const auto index = stack.back(); stack.pop_back();
                if (reachable[index]) continue;
                reachable[index] = true;
                if (units[index].header != stop)
                    for (auto next : units[index].outgoing) stack.push_back(next);
            }
            return reachable;
        };
        std::function<RegionId(BlockId, BlockId, std::uint32_t)> walk;
        walk = [&](BlockId current, BlockId stop, std::uint32_t nesting) -> RegionId {
            depth_check(nesting);
            std::vector<RegionId> children;
            while (current.valid() && current != stop) {
                budget.step();
                if (current == backedge && emitted[current.value]) {
                    children.push_back(add(ContinueRegion{loops.back()}));
                    break;
                }
                if (current == exit) {
                    children.push_back(add(BreakRegion{loops.back()}));
                    break;
                }
                if (!member[current.value]) throw Unsupported{};
                const auto index = owner[current.value];
                const auto& unit = units[index];
                if (current != unit.header || emitted[current.value]) throw Unsupported{};
                if (unit.loop) {
                    const auto loop = sequence();
                    loops.push_back(loop);
                    const auto body = scope(groups[index], unit.header, unit.header, unit.next, nesting + 1);
                    loops.pop_back();
                    const auto continuing = sequence();
                    result.regions[loop.value] = LoopRegion{body, continuing};
                    children.push_back(loop);
                    current = unit.next;
                    continue;
                }
                emitted[current.value] = true;
                children.push_back(add(BlockRegion{current}));
                const auto& terminator = p.blocks[current.value].terminator;
                if (const auto* returned = std::get_if<Return>(&terminator)) {
                    children.push_back(add(ReturnRegion{returned->values}));
                    break;
                }
                if (const auto* jump = std::get_if<Jump>(&terminator)) {
                    current = jump->target;
                    continue;
                }
                const auto* branch = std::get_if<Branch>(&terminator);
                if (!branch) throw Unsupported{};
                if (branch->when_true == branch->when_false) {
                    current = branch->when_true;
                    continue;
                }
                const auto left = reachable(branch->when_true, stop), right = reachable(branch->when_false, stop);
                BlockId join = stop;
                // The first common DAG unit is the only possible shared tail.
                // Arms that terminate before it retain their return/loop transfer.
                // If other overlap exists, the single-owner check rejects the
                // attempt instead of duplicating operations or changing effects.
                for (std::size_t i = 0; i < groups.size(); ++i) {
                    budget.step();
                    if (left[i] && right[i]) { join = units[i].header; break; }
                }
                const auto yes = walk(branch->when_true, join, nesting + 1);
                const auto no = walk(branch->when_false, join, nesting + 1);
                children.push_back(add(IfRegion{branch->condition, yes, no}));
                current = join;
            }
            return sequence(std::move(children));
        };
        return walk(entry, {}, depth);
    }
};

class RegionBuilder {
public:
    RegionBuilder(StructuredProgram& program, Budget& budget)
        : result(program), p(program.ir), budget(budget), outgoing(p.blocks.size()), flags(p.blocks.size()) {
        for (std::uint32_t b = 0; b < p.blocks.size(); ++b) {
            budget.step();
            outgoing[b] = successors(p.blocks[b].terminator);
        }
    }
    void run();
private:
    StructuredProgram& result;
    Program& p;
    Budget& budget;
    std::vector<std::vector<BlockId>> outgoing;
    std::vector<ValueId> flags;
    RegionId region(Region payload) {
        budget.step();
        capacity(result.regions.size(), p.limits.regions, "region");
        RegionId id{static_cast<std::uint32_t>(result.regions.size())};
        result.regions.push_back(std::move(payload));
        return id;
    }
    RegionId sequence(std::vector<RegionId> children = {}) { return region(SequenceRegion{std::move(children)}); }
    RegionId guarded(ValueId flag, RegionId body) { return region(IfRegion{flag, body, sequence()}); }
    RegionId leaf(BlockId block, bool exhaustive = false) {
        std::vector<RegionId> children{region(BlockRegion{block})};
        if (const auto* returned = std::get_if<Return>(&p.blocks[block.value].terminator))
            children.push_back(region(ReturnRegion{returned->values}));
        const auto body = sequence(std::move(children));
        return exhaustive ? body : guarded(flags[block.value], body);
    }
    Operand logical(BlockId block, Opcode opcode, std::vector<Operand> inputs) {
        const auto output = value(p, ValueType::predicate);
        operation(p, block, {p.blocks[block.value].source, opcode, Effects::none,
                            {Domain::boolean, ScalarType::boolean}, std::move(inputs), {output}, {}});
        return output;
    }
    void transfer(BlockId);
    std::vector<std::vector<BlockId>> components(const std::vector<BlockId>&);
    RegionId schedule(const std::vector<BlockId>&, BlockId entry, std::uint32_t depth);
};

void RegionBuilder::transfer(BlockId block_id) {
    // Evaluate branch conditions in their original block before any edge copies.
    // Exactly one pending target is true between blocks; a return consumes it.
    const auto terminator = p.blocks[block_id.value].terminator;
    const auto source = p.blocks[block_id.value].source;
    copy(p, block_id, flags[block_id.value], boolean(false), source);
    if (const auto* jump = std::get_if<Jump>(&terminator)) {
        copy(p, block_id, flags[jump->target.value], boolean(true), source);
    } else if (const auto* branch = std::get_if<Branch>(&terminator)) {
        if (branch->when_true == branch->when_false) {
            copy(p, block_id, flags[branch->when_true.value], boolean(true), source);
        } else {
            copy(p, block_id, flags[branch->when_true.value], branch->condition, source);
            copy(p, block_id, flags[branch->when_false.value],
                 logical(block_id, Opcode::logical_not, {branch->condition}), source);
        }
    } else if (const auto* select = std::get_if<Switch>(&terminator)) {
        std::map<BlockId, Operand> targets;
        Operand matched = boolean(false);
        auto combine = [&](Operand left, Operand right) {
            return logical(block_id, Opcode::logical_or, {left, right});
        };
        for (const auto& item : select->cases) {
            budget.step();
            const auto equal = value(p, ValueType::predicate);
            operation(p, block_id, {source, Opcode::compare, Effects::none,
                {Domain::unsigned_integer, ScalarType::u32},
                {select->selector, LiteralBits{item.bits, ValueType::word32}}, {equal}, ComparisonInfo{Compare::equal, false}});
            matched = combine(matched, equal);
            const auto found = targets.find(item.target);
            if (found == targets.end()) targets.emplace(item.target, equal);
            else found->second = combine(found->second, equal);
        }
        const auto fallback = logical(block_id, Opcode::logical_not, {matched});
        const auto found = targets.find(select->fallback);
        if (found == targets.end()) targets.emplace(select->fallback, fallback);
        else found->second = combine(found->second, fallback);
        for (const auto& [target, condition] : targets) copy(p, block_id, flags[target.value], condition, source);
    }
}

std::vector<std::vector<BlockId>> RegionBuilder::components(const std::vector<BlockId>& nodes) {
    // Iterative Kosaraju traversal avoids host recursion proportional to CFG size.
    // Reverse finish order yields the condensation graph in execution order.
    budget.step(3 * static_cast<std::uint64_t>(outgoing.size()));
    std::vector<bool> member(outgoing.size()), visited(outgoing.size());
    for (auto node : nodes) { budget.step(); member[node.value] = true; }
    std::vector<BlockId> finish;
    for (auto node : nodes) {
        if (visited[node.value]) continue;
        visited[node.value] = true;
        std::vector<std::pair<BlockId, std::size_t>> stack{{node, 0}};
        while (!stack.empty()) {
            budget.step();
            auto& [current, next] = stack.back();
            if (next == outgoing[current.value].size()) {
                finish.push_back(current);
                stack.pop_back();
                continue;
            }
            const auto target = outgoing[current.value][next++];
            if (member[target.value] && !visited[target.value]) {
                visited[target.value] = true;
                stack.emplace_back(target, 0);
            }
        }
    }
    std::fill(visited.begin(), visited.end(), false);
    std::vector<std::vector<BlockId>> groups;
    for (auto cursor = finish.rbegin(); cursor != finish.rend(); ++cursor) {
        if (visited[cursor->value]) continue;
        auto& group = groups.emplace_back();
        std::vector<BlockId> stack{*cursor};
        visited[cursor->value] = true;
        while (!stack.empty()) {
            budget.step();
            const auto current = stack.back();
            stack.pop_back();
            group.push_back(current);
            for (auto predecessor : p.blocks[current.value].predecessors) {
                budget.step();
                if (predecessor.value < member.size() && member[predecessor.value] && !visited[predecessor.value]) {
                    visited[predecessor.value] = true;
                    stack.push_back(predecessor);
                }
            }
        }
    }
    return groups;
}

RegionId RegionBuilder::schedule(const std::vector<BlockId>& nodes, BlockId entry, std::uint32_t depth) {
    if (depth > p.limits.nesting_depth)
        fail("Loop decomposition nesting limit exceeded", {}, DiagnosticCode::limit_exceeded);
    if (nodes.empty()) return sequence();
    std::vector<RegionId> children;
    const auto groups = components(nodes);
    for (std::size_t index = 0; index < groups.size(); ++index) {
        const auto& group = groups[index];
        const bool exhaustive = depth == 0 && index + 1 == groups.size();
        const auto first = group.front();
        const bool cycle = group.size() > 1 ||
            std::find(outgoing[first.value].begin(), outgoing[first.value].end(), first) != outgoing[first.value].end();
        // At function scope the last component is exhaustive: earlier ones
        // have returned, diverged, or transferred forward to this component.
        // In particular, a final Return must not acquire an empty false path.
        if (!cycle) { children.push_back(leaf(first, exhaustive)); continue; }
        std::set<BlockId> members(group.begin(), group.end());
        std::set<BlockId> entries;
        if (members.contains(entry)) entries.insert(entry);
        for (auto node : group)
            for (auto predecessor : p.blocks[node.value].predecessors) {
                budget.step();
                if (!members.contains(predecessor)) entries.insert(node);
            }
        if (entries.size() != 1)
            fail("Irreducible control flow has no unique loop entry", p.blocks[first.value].source,
                 DiagnosticCode::unsupported_capability);
        const auto header = *entries.begin();
        std::vector<BlockId> rest;
        for (auto node : group) if (node != header) rest.push_back(node);
        // Reserve the loop ID before building transfers that refer to it.
        const auto loop = sequence();
        const auto header_region = leaf(header);
        const auto remainder = schedule(rest, entry, depth + 1);
        const auto repeat = region(ContinueRegion{loop});
        const auto tail = exhaustive ? repeat :
            region(IfRegion{flags[header.value], repeat, region(BreakRegion{loop})});
        const auto body = sequence({header_region, remainder, tail});
        const auto continuing = sequence();
        result.regions[loop.value] = LoopRegion{body, continuing};
        children.push_back(exhaustive ? loop : guarded(flags[header.value], loop));
    }
    return sequence(std::move(children));
}

void RegionBuilder::run() {
    for (std::uint32_t f = 0; f < p.functions.size(); ++f) {
        const auto saved_regions = result.regions.size();
        try {
            const auto root = DirectBuilder(result, budget).run(FunctionId{f});
            result.functions.push_back({FunctionId{f}, root});
            continue;
        } catch (const DirectBuilder::Unsupported&) {
            // The direct attempt only appends regions; discard it before adding
            // state for control flow with multiple structural exits.
            result.regions.resize(saved_regions);
        }
        auto nodes = p.functions[f].blocks;
        for (auto node : nodes) flags[node.value] = value(p, ValueType::predicate, ValueKind::mutable_local);
        for (auto node : nodes) transfer(node);
        std::sort(nodes.begin(), nodes.end());
        const auto entry = p.functions[f].entry;
        const auto init = block(p, FunctionId{f});
        p.blocks[init.value].terminator = Jump{entry};
        p.functions[f].entry = init;
        p.blocks[entry.value].predecessors.push_back(init);
        for (auto node : nodes) { budget.step(); copy(p, init, flags[node.value], boolean(node == entry)); }
        const auto initializer = region(BlockRegion{init});
        const auto body = schedule(nodes, entry, 0);
        result.functions.push_back({FunctionId{f}, sequence({initializer, body})});
    }
    rebuild_predecessors(p, budget);
}
} // namespace

void build_regions(StructuredProgram& program, Budget& budget) {
    lower_switches(program.ir, budget);
    hoist_cross_block_values(program.ir, budget);
    RegionBuilder(program, budget).run();
}
} // namespace armigo::shader::pipeline::structured_detail
