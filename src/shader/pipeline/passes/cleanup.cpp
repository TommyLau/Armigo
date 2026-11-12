// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "internal.h"
#include <algorithm>
#include <utility>

namespace armigo::shader::pipeline::passes {
namespace {
using detail::Budget;
template <class F> void term_inputs(Terminator& term, F&& visit) {
    if (auto* branch = std::get_if<Branch>(&term)) visit(branch->condition);
    else if (auto* selection = std::get_if<Switch>(&term)) visit(selection->selector);
    else if (auto* result = std::get_if<Return>(&term)) for (auto& value : result->values) visit(value);
}
class Cleanup {
public:
    explicit Cleanup(Program& program) : p(program), budget(program), discard_constant_reads(
        std::any_of(program.operations.begin(), program.operations.end(), [](const auto& op) {
            const auto* m = std::get_if<MemoryInfo>(&op.info);
            return m && m->direct_storage.has_value();
        })) {}
    void run();
private:
    Program& p;
    Budget budget;
    bool discard_constant_reads;
    bool propagate();
    bool branches();
    bool dead_code();
};

bool Cleanup::propagate() {
    budget.spend(p.values.size() + p.operations.size() + p.blocks.size());
    std::vector<std::optional<Operand>> replacements(p.values.size());
    for (const auto& op : p.operations) {
        if (op.opcode == Opcode::copy && detail::neutral(op))
            replacements[std::get<ValueId>(op.outputs[0]).value] = op.inputs[0];
    }
    auto resolve = [&](Operand value, bool value_only = false) {
        while (const auto* id = std::get_if<ValueId>(&value)) {
            budget.spend();
            const auto& next = replacements[id->value];
            // Phi inputs carry ValueIds, so a literal/CB copy remains at its definition.
            if (!next || (value_only && !std::holds_alternative<ValueId>(*next))) break;
            value = *next;
        }
        return value;
    };
    bool changed = false;
    for (auto& block : p.blocks) {
        for (const auto& phi : block.phis) {
            std::optional<ValueId> common;
            std::optional<LiteralBits> constant;
            bool equal = true, equal_constants = true;
            for (const auto& input : phi.inputs) {
                budget.spend();
                const auto value = std::get<ValueId>(resolve(input.value, true));
                if (value == phi.output) continue;
                if (common && *common != value) equal = false;
                common = value;
                const auto resolved = resolve(value);
                if (const auto* bits = std::get_if<LiteralBits>(&resolved)) {
                    if (constant && *constant != *bits) equal_constants = false;
                    constant = *bits;
                } else equal_constants = false;
            }
            // A self-only phi has no replacement; never invent a zero initializer.
            if (equal && common) replacements[phi.output.value] = *common;
            else if (equal_constants && constant) {
                // Phis cannot hold literals. Keep a definition at the merge for
                // other phi users; ordinary uses can still receive the raw bits.
                if (p.operations.size() >= p.limits.operations)
                    throw diagnose(Phase::legalize, DiagnosticCode::limit_exceeded, phi.source,
                        "Constant phi materialization exceeds operation limit");
                budget.spend(block.operations.size() + 1, phi.source);
                const OperationId id{static_cast<std::uint32_t>(p.operations.size())};
                p.operations.push_back({phi.source, Opcode::copy, Effects::none, {}, {*constant}, {phi.output}, {}});
                block.operations.insert(block.operations.begin(), id);
                replacements[phi.output.value] = *constant;
            }
        }
        const auto erased = std::erase_if(block.phis, [&](const Phi& phi) { return replacements[phi.output.value].has_value(); });
        changed |= erased != 0;
    }
    auto rewrite = [&](Operand& input) {
        budget.spend();
        const auto* id = std::get_if<ValueId>(&input);
        if (!id || !replacements[id->value]) return;
        input = resolve(input);
        changed = true;
    };
    for (auto& op : p.operations) {
        for (auto& input : op.inputs) rewrite(input);
        if (op.opcode != Opcode::copy && op.outputs.size() == 1) {
            const auto type = p.values[std::get<ValueId>(op.outputs[0]).value].type;
            if (auto replacement = detail::simplify(op, type)) {
                op.opcode = Opcode::copy;
                op.inputs = {*replacement};
                op.info = std::monostate{};
                // NumericMode is retained verbatim, including on a folded comparison.
                changed = true;
            }
        }
    }
    for (auto& block : p.blocks) {
        for (auto& phi : block.phis) for (auto& input : phi.inputs) {
            budget.spend();
            const auto value = std::get<ValueId>(resolve(input.value, true));
            changed |= input.value != value;
            input.value = value;
        }
        term_inputs(block.terminator, rewrite);
    }
    return changed;
}

bool Cleanup::branches() {
    bool changed = false;
    budget.spend(p.blocks.size());
    for (auto& block : p.blocks) {
        auto& term = block.terminator;
        if (const auto* branch = std::get_if<Branch>(&term)) {
            const auto* literal = std::get_if<LiteralBits>(&branch->condition);
            if (literal || branch->when_true == branch->when_false) {
                const auto target = literal && literal->bits == 0 ? branch->when_false : branch->when_true;
                term = Jump{target}; changed = true;
            }
        } else if (const auto* selection = std::get_if<Switch>(&term)) {
            const auto* literal = std::get_if<LiteralBits>(&selection->selector);
            auto target = selection->fallback;
            bool identical = true;
            for (const auto& item : selection->cases) {
                budget.spend();
                if (literal && literal->bits == item.bits) target = item.target;
                identical &= item.target == selection->fallback;
            }
            if (literal || identical) { term = Jump{target}; changed = true; }
        }
    }
    if (!changed) return false;
    std::vector<bool> reachable(p.blocks.size());
    std::vector<BlockId> pending;
    for (const auto& function : p.functions) { reachable[function.entry.value] = true; pending.push_back(function.entry); }
    while (!pending.empty()) {
        const auto id = pending.back(); pending.pop_back();
        budget.spend();
        for (const auto target : successors(p.blocks[id.value].terminator)) {
            budget.spend();
            if (!reachable[target.value]) { reachable[target.value] = true; pending.push_back(target); }
        }
    }
    std::vector<std::vector<BlockId>> incoming(p.blocks.size());
    for (std::uint32_t i = 0; i < p.blocks.size(); ++i) if (reachable[i]) {
        for (const auto target : successors(p.blocks[i].terminator)) {
            budget.spend(); incoming[target.value].push_back(BlockId{i});
        }
    }
    std::vector<BlockId> remap(p.blocks.size());
    std::uint32_t next = 0;
    for (std::size_t i = 0; i < reachable.size(); ++i) if (reachable[i]) remap[i] = BlockId{next++};
    std::vector<BasicBlock> blocks;
    blocks.reserve(next);
    for (std::size_t i = 0; i < p.blocks.size(); ++i) if (reachable[i]) {
        budget.spend();
        auto& block = p.blocks[i];
        for (auto& phi : block.phis) {
            std::erase_if(phi.inputs, [&](const PhiInput& input) {
                budget.spend();
                return !std::binary_search(incoming[i].begin(), incoming[i].end(), input.predecessor);
            });
            for (auto& input : phi.inputs) input.predecessor = remap[input.predecessor.value];
        }
        block.predecessors = std::move(incoming[i]);
        for (auto& predecessor : block.predecessors) predecessor = remap[predecessor.value];
        auto& term = block.terminator;
        if (auto* jump = std::get_if<Jump>(&term)) jump->target = remap[jump->target.value];
        else if (auto* branch = std::get_if<Branch>(&term)) {
            branch->when_true = remap[branch->when_true.value]; branch->when_false = remap[branch->when_false.value];
        } else if (auto* selection = std::get_if<Switch>(&term)) {
            selection->fallback = remap[selection->fallback.value];
            for (auto& item : selection->cases) { budget.spend(); item.target = remap[item.target.value]; }
        }
        blocks.push_back(std::move(block));
    }
    for (auto& function : p.functions) {
        function.entry = remap[function.entry.value];
        std::erase_if(function.blocks, [&](BlockId id) { budget.spend(); return !reachable[id.value]; });
        for (auto& id : function.blocks) id = remap[id.value];
    }
    p.blocks = std::move(blocks);
    return true;
}

bool Cleanup::dead_code() {
    budget.spend(p.values.size() + p.operations.size() + p.blocks.size());
    struct Definition { OperationId op; const Phi* phi{}; };
    std::vector<Definition> definitions(p.values.size());
    std::vector<bool> live_values(p.values.size()), live_ops(p.operations.size());
    std::vector<ValueId> pending;
    auto mark = [&](const Operand& operand) {
        budget.spend();
        if (const auto* id = std::get_if<ValueId>(&operand); id && !live_values[id->value]) {
            live_values[id->value] = true; pending.push_back(*id);
        }
    };
    auto mark_op = [&](OperationId id) {
        if (live_ops[id.value]) return;
        live_ops[id.value] = true;
        for (const auto& input : p.operations[id.value].inputs) mark(input);
    };
    for (auto& block : p.blocks) {
        for (const auto& phi : block.phis) { budget.spend(); definitions[phi.output.value].phi = &phi; }
        for (const auto id : block.operations) {
            budget.spend();
            const auto& op = p.operations[id.value];
            for (const auto& output : op.outputs) definitions[std::get<ValueId>(output).value].op = id;
            // Resolved storage bases leave immutable, non-faulting constant
            // reads behind. Remove those when unused; retain other effects.
            const auto* m = std::get_if<MemoryInfo>(&op.info);
            const bool dead_constant = discard_constant_reads && op.opcode == Opcode::load &&
                op.effects == Effects::read_memory && m && m->space == AddressSpace::constant_buffer && !m->is_volatile;
            if (op.effects != Effects::none && !dead_constant) mark_op(id);
        }
        term_inputs(block.terminator, mark);
    }
    while (!pending.empty()) {
        const auto value = pending.back(); pending.pop_back();
        budget.spend();
        const auto& definition = definitions[value.value];
        if (definition.op.valid()) mark_op(definition.op);
        else if (definition.phi) for (const auto& input : definition.phi->inputs) mark(input.value);
    }
    bool changed = false;
    std::vector<OperationId> remap(p.operations.size());
    std::vector<Operation> operations;
    for (std::size_t i = 0; i < p.operations.size(); ++i) if (live_ops[i]) {
        remap[i] = OperationId{static_cast<std::uint32_t>(operations.size())};
        operations.push_back(std::move(p.operations[i]));
    }
    changed |= operations.size() != p.operations.size();
    for (auto& block : p.blocks) {
        changed |= std::erase_if(block.phis, [&](const Phi& phi) { budget.spend(); return !live_values[phi.output.value]; }) != 0;
        std::erase_if(block.operations, [&](OperationId id) { budget.spend(); return !live_ops[id.value]; });
        for (auto& id : block.operations) id = remap[id.value];
    }
    p.operations = std::move(operations);
    return changed;
}
void Cleanup::run() {
    for (const auto& op : p.operations) {
        budget.spend();
        if (const auto* conversion = std::get_if<ConversionInfo>(&op.info);
            conversion && (conversion->source > ScalarType::boolean || conversion->destination > ScalarType::boolean))
            throw diagnose(Phase::legalize, DiagnosticCode::unsupported_modifier, op.source,
                "Conversion scalar is outside the defined contract");
        if (const auto* comparison = std::get_if<ComparisonInfo>(&op.info);
            comparison && comparison->relation > Compare::always)
            throw diagnose(Phase::legalize, DiagnosticCode::unsupported_modifier, op.source,
                "Comparison relation is outside the defined contract");
    }
    bool changed;
    do {
        budget.spend();
        changed = propagate();
        changed |= branches();
        changed |= dead_code();
    } while (changed);
}
} // namespace

Status cleanup(Program& program) {
    if (auto status = validate(program, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return status;
    Program candidate = program;
    try { Cleanup(candidate).run(); }
    catch (const Diagnostic& error) { return error; }
    if (auto status = validate(candidate, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return status;
    program = std::move(candidate);
    return std::monostate{};
}
} // namespace armigo::shader::pipeline::passes
