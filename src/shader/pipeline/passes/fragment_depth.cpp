// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "passes.h"
#include "../ssa/dominance.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace armigo::shader::pipeline::passes {
namespace {
using Roots = std::vector<std::set<std::uint32_t>>;
struct Definition { BlockId block; unsigned position{}; OperationId op; unsigned output{}; };
struct Position { OperationId op; unsigned component{}; IoSemantic semantic{}; };
class DepthSnap {
public:
    explicit DepthSnap(const Program& program) : p(program), definitions(p.values.size()) {}
    std::map<std::uint32_t, std::set<std::uint32_t>> pairs;
    std::map<std::uint32_t, std::vector<std::uint32_t>> source_pairs;
    std::map<std::uint32_t, Position> positions;
    void plan();
private:
    const Program& p;
    std::uint64_t steps{};
    std::vector<Definition> definitions;
    void spend(std::uint64_t count = 1) {
        steps += count;
        if (steps > p.limits.analysis_steps)
            throw diagnose(Phase::legalize, DiagnosticCode::limit_exceeded, {}, "Fragment depth analysis budget exceeded");
    }
    Roots trace(bool textures, const Roots& z);
};

bool texture_read(const Operation& op) {
    const auto* t = std::get_if<TextureInfo>(&op.info);
    if (!t || op.numeric.scalar != ScalarType::f32 || t->compare) return false;
    if (op.opcode == Opcode::image_load) return true;
    if (op.opcode != Opcode::texture_sample) return false;
    return t->mode == TextureMode::sample_implicit || t->mode == TextureMode::sample_level ||
        t->mode == TextureMode::sample_bias || t->mode == TextureMode::sample_gradients || t->mode == TextureMode::fetch;
}

Roots DepthSnap::trace(bool textures, const Roots& z) {
    Roots roots(p.values.size());
    for (const auto& [value, position] : positions) if (!textures) roots[value].insert(value);
    const auto unite = [&](std::set<std::uint32_t>& into, const std::set<std::uint32_t>& from) {
        bool changed = false;
        for (auto value : from) { spend(); changed |= into.insert(value).second; }
        return changed;
    };
    bool changed;
    do {
        changed = false;
        for (auto block_id : p.functions[p.entry.value].blocks) {
            const auto& block = p.blocks[block_id.value];
            for (const auto& phi : block.phis) for (const auto& input : phi.inputs) {
                spend(); changed |= unite(roots[phi.output.value], roots[input.value.value]);
            }
            for (auto id : block.operations) {
                spend(); const auto& op = p.operations[id.value];
                std::set<std::uint32_t> input;
                for (const auto& operand : op.inputs)
                    if (const auto* value = std::get_if<ValueId>(&operand); value && p.values[value->value].kind != ValueKind::mutable_local)
                        unite(input, roots[value->value]);
                for (const auto& output : op.outputs) {
                    const auto* value = std::get_if<ValueId>(&output);
                    if (!value || positions.contains(value->value) || p.values[value->value].kind == ValueKind::mutable_local) continue;
                    if (textures && texture_read(op) && z[value->value].empty())
                        changed |= roots[value->value].insert(value->value).second;
                    else changed |= unite(roots[value->value], input);
                }
            }
        }
    } while (changed);
    return roots;
}

void DepthSnap::plan() {
    for (auto block_id : p.functions[p.entry.value].blocks) {
        const auto& block = p.blocks[block_id.value];
        for (unsigned order = 0; order < block.operations.size(); ++order) {
            const auto id = block.operations[order]; const auto& op = p.operations[id.value];
            for (unsigned component = 0; component < op.outputs.size(); ++component)
                if (const auto* value = std::get_if<ValueId>(&op.outputs[component]))
                    definitions[value->value] = {block_id, order, id, component};
            const auto* io = std::get_if<IoInfo>(&op.info);
            if (op.opcode != Opcode::load_io || !io || io->output_storage || io->type != IoType{ScalarType::f32, 4} ||
                (io->semantic != IoSemantic::position && io->semantic != IoSemantic::fragment_coord) || !(io->component_mask & 4)) continue;
            unsigned index = 0;
            for (unsigned component = 0; component < 4; ++component) if (io->component_mask & (1u << component)) {
                if (component == 2) positions.emplace(std::get<ValueId>(op.outputs[index]).value, Position{id, index, io->semantic});
                ++index;
            }
        }
    }
    if (positions.empty()) return;
    const Roots unused;
    const auto z = trace(false, unused), textures = trace(true, z);
    auto result = find_dominators(p, p.functions[p.entry.value], p.limits, steps);
    if (const auto* error = std::get_if<Diagnostic>(&result)) throw *error;
    const auto& dom = std::get<Dominators>(result);
    const auto before = [&](std::uint32_t texture, std::uint32_t position) {
        const auto& a = definitions[texture]; const auto& b = definitions[position];
        if (a.block == b.block) return a.position < b.position;
        auto block = b.block;
        while (block != p.functions[p.entry.value].entry) {
            spend(); block = dom.immediate_dominator[block.value];
            if (block == a.block) return true;
        }
        return false;
    };
    for (const auto& block : p.blocks) if (block.function == p.entry) for (auto id : block.operations) {
        const auto& op = p.operations[id.value];
        const auto* comparison = std::get_if<ComparisonInfo>(&op.info);
        if (op.opcode != Opcode::compare || op.inputs.size() != 2 || !comparison ||
            comparison->relation == Compare::always || comparison->relation == Compare::never) continue;
        for (unsigned side = 0; side < 2; ++side) {
            const auto* a = std::get_if<ValueId>(&op.inputs[side]);
            const auto* b = std::get_if<ValueId>(&op.inputs[1-side]);
            if (!a || !b || z[a->value].empty() || !textures[a->value].empty() ||
                textures[b->value].empty() || !z[b->value].empty()) continue;
            for (auto position : z[a->value]) for (auto texture : textures[b->value])
                if (before(texture, position)) pairs[position].insert(texture);
        }
    }
    for (const auto& [position, depths] : pairs) {
        auto& ordered = source_pairs[position];
        ordered.assign(depths.begin(), depths.end());
        // Every paired definition dominates the position read, so their
        // blocks form one dominance chain. Value IDs carry no source order.
        std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
            const auto& a = definitions[left]; const auto& b = definitions[right];
            if (a.block == b.block) return std::tie(a.position, a.output) < std::tie(b.position, b.output);
            return before(left, right);
        });
    }
}

void rewrite(Program& p, const DepthSnap& plan) {
    const NumericMode fp{Domain::floating, ScalarType::f32, Rounding::nearest_even};
    const NumericMode bits{Domain::bits, ScalarType::u32};
    const NumericMode integer{Domain::unsigned_integer, ScalarType::u32};
    const auto literal = [](std::uint32_t value) -> Operand { return LiteralBits{value, ValueType::word32}; };
    const auto fresh = [&](ValueType type) {
        if (p.values.size() >= p.limits.values) throw diagnose(Phase::legalize, DiagnosticCode::limit_exceeded, {}, "Fragment depth value limit exceeded");
        ValueId value{static_cast<std::uint32_t>(p.values.size())}; p.values.push_back({type}); return value;
    };
    const auto emit = [&](std::vector<OperationId>& order, Opcode opcode, std::vector<Operand> inputs,
                          NumericMode mode, OperationInfo info = {}, std::optional<ValueId> output = {}) {
        const auto value = output ? *output : fresh(opcode == Opcode::compare || opcode == Opcode::logical_and ? ValueType::predicate : ValueType::word32);
        if (p.operations.size() >= p.limits.operations) throw diagnose(Phase::legalize, DiagnosticCode::limit_exceeded, {}, "Fragment depth operation limit exceeded");
        Operation op{{}, opcode, {}, mode, std::move(inputs), {value}, std::move(info)}; op.effects = required_effects(op);
        order.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())}); p.operations.push_back(std::move(op)); return value;
    };
    std::vector<OperationId> prefix;
    IoInfo io = std::get<IoInfo>(p.operations[plan.positions.at(plan.pairs.begin()->first).op.value].info);
    io.component_mask = 4;
    const auto raw = emit(prefix, Opcode::load_io, {}, bits, io);
    const auto dx = emit(prefix, Opcode::derivative_x, {raw}, fp), dy = emit(prefix, Opcode::derivative_y, {raw}, fp);
    const auto ax = emit(prefix, Opcode::absolute, {dx}, fp), ay = emit(prefix, Opcode::absolute, {dy}, fp);
    const auto width = emit(prefix, Opcode::add, {ax, ay}, fp);
    auto& entry = p.blocks[p.functions[p.entry.value].entry.value].operations;
    entry.insert(entry.begin(), prefix.begin(), prefix.end());
    std::map<std::uint32_t, std::uint32_t> by_operation;
    for (const auto& [value, depths] : plan.pairs) by_operation.emplace(plan.positions.at(value).op.value, value);
    for (auto& block : p.blocks) {
        std::vector<OperationId> order;
        for (auto id : block.operations) {
            order.push_back(id);
            const auto match = by_operation.find(id.value); if (match == by_operation.end()) continue;
            const auto original = match->second;
            const auto temporary = fresh(ValueType::word32);
            p.operations[id.value].outputs[plan.positions.at(original).component] = temporary;
            Operand current = temporary;
            const auto& depths = plan.source_pairs.at(original);
            unsigned remaining = depths.size();
            for (auto depth : depths) {
                // A positive depth's preceding bit pattern is one ULP lower.
                // The slope allowance covers interpolation and depth-store
                // rounding without merging a distinct farther surface.
                const auto previous = emit(order, Opcode::subtract, {current, literal(1)}, integer);
                const auto margin = emit(order, Opcode::multiply, {width, literal(0x35800000u)}, fp);
                const auto lower = emit(order, Opcode::subtract, {previous, margin}, fp);
                const auto positive = emit(order, Opcode::compare, {current, literal(0)}, fp, ComparisonInfo{Compare::greater, false});
                const auto behind = emit(order, Opcode::compare, {ValueId{depth}, current}, fp, ComparisonInfo{Compare::less, false});
                const auto close = emit(order, Opcode::compare, {ValueId{depth}, lower}, fp, ComparisonInfo{Compare::greater_equal, false});
                const NumericMode boolean{Domain::boolean, ScalarType::boolean};
                const auto match_depth = emit(order, Opcode::logical_and, {positive, behind}, boolean);
                const auto snap = emit(order, Opcode::logical_and, {match_depth, close}, boolean);
                current = emit(order, Opcode::select, {snap, ValueId{depth}, current}, bits, {},
                    --remaining ? std::nullopt : std::optional<ValueId>{ValueId{original}});
            }
        }
        block.operations = std::move(order);
    }
}
} // namespace

Status snap_fragment_depth(Program& program) {
    if (program.stage != Stage::fragment) return std::monostate{};
    if (auto status = validate(program, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return status;
    try {
        DepthSnap plan(program); plan.plan();
        if (plan.pairs.empty()) return std::monostate{};
        auto candidate = program; rewrite(candidate, plan);
        if (auto status = validate(candidate, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return status;
        program = std::move(candidate); return std::monostate{};
    } catch (const Diagnostic& error) { return error; }
}
} // namespace armigo::shader::pipeline::passes
