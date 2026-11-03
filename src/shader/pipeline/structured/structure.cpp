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

namespace armigo::shader::pipeline {
namespace {
Limits intersect(Limits inherited, const Limits& supplied) {
#define ARMIGO_MIN_LIMIT(member) inherited.member = std::min(inherited.member, supplied.member)
    ARMIGO_MIN_LIMIT(code_bytes); ARMIGO_MIN_LIMIT(instructions); ARMIGO_MIN_LIMIT(functions);
    ARMIGO_MIN_LIMIT(blocks); ARMIGO_MIN_LIMIT(edges); ARMIGO_MIN_LIMIT(operations);
    ARMIGO_MIN_LIMIT(values); ARMIGO_MIN_LIMIT(phi_inputs); ARMIGO_MIN_LIMIT(operands_per_operation);
    ARMIGO_MIN_LIMIT(token_depth); ARMIGO_MIN_LIMIT(token_states); ARMIGO_MIN_LIMIT(regions);
    ARMIGO_MIN_LIMIT(nesting_depth); ARMIGO_MIN_LIMIT(resources); ARMIGO_MIN_LIMIT(dependencies);
    ARMIGO_MIN_LIMIT(source_bytes); ARMIGO_MIN_LIMIT(analysis_steps);
#undef ARMIGO_MIN_LIMIT
    return inherited;
}
// Fold predicate literals while SSA still gives copies a single definition.
// Removing dead arms here keeps block/operation ownership and phi edges exact;
// the region builder never needs to hide an unreachable subtree from emission.
void fold_branches(Program& p, structured_detail::Budget& budget) {
    bool again = true;
    while (again) {
        again = false;
        budget.step(p.values.size());
        std::vector<std::optional<LiteralBits>> constants(p.values.size());
        auto literal = [&](const Operand& operand) -> std::optional<LiteralBits> {
            if (const auto* bits = std::get_if<LiteralBits>(&operand)) return *bits;
            if (const auto* id = std::get_if<ValueId>(&operand)) return constants[id->value];
            return {};
        };
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& op : p.operations) {
                budget.step();
                if (op.outputs.size() != 1 || op.inputs.size() != 1) continue;
                const auto output = std::get<ValueId>(op.outputs.front());
                if (constants[output.value]) continue;
                auto input = literal(op.inputs.front());
                if (!input) continue;
                if (op.opcode == Opcode::logical_not && input->type == ValueType::predicate)
                    input->bits = !input->bits;
                else if (op.opcode != Opcode::copy) continue;
                constants[output.value] = input;
                changed = true;
            }
            for (const auto& block : p.blocks) for (const auto& phi : block.phis) {
                budget.step();
                if (constants[phi.output.value] || phi.inputs.empty()) continue;
                auto common = constants[phi.inputs.front().value.value];
                for (const auto& input : phi.inputs) {
                    budget.step();
                    if (constants[input.value.value] != common) { common.reset(); break; }
                }
                if (common) { constants[phi.output.value] = common; changed = true; }
            }
        }
        for (auto& block : p.blocks) {
            budget.step();
            if (const auto* branch = std::get_if<Branch>(&block.terminator)) {
                const auto condition = literal(branch->condition);
                if (condition) {
                    block.terminator = Jump{condition->bits ? branch->when_true : branch->when_false};
                    again = true;
                }
            } else if (const auto* select = std::get_if<Switch>(&block.terminator)) {
                if (const auto selector = literal(select->selector)) {
                    auto target = select->fallback;
                    for (const auto& item : select->cases) {
                        budget.step();
                        if (item.bits == selector->bits) target = item.target;
                    }
                    block.terminator = Jump{target};
                    again = true;
                }
            }
        }
        if (!again) break;
        std::vector<bool> live(p.blocks.size());
        std::vector<BlockId> stack;
        for (const auto& function : p.functions) stack.push_back(function.entry);
        while (!stack.empty()) {
            budget.step();
            const auto id = stack.back(); stack.pop_back();
            if (live[id.value]) continue;
            live[id.value] = true;
            for (auto target : successors(p.blocks[id.value].terminator)) stack.push_back(target);
        }
        structured_detail::rebuild_predecessors(p, budget);
        std::vector<BlockId> remap(p.blocks.size());
        std::vector<BasicBlock> blocks;
        std::vector<Operation> operations;
        for (std::uint32_t old = 0; old < p.blocks.size(); ++old) {
            budget.step();
            if (!live[old]) continue;
            remap[old] = BlockId{static_cast<std::uint32_t>(blocks.size())};
            auto block = std::move(p.blocks[old]);
            for (auto& phi : block.phis) {
                std::erase_if(phi.inputs, [&](const auto& input) {
                    budget.step();
                    // Rebuilding visits source IDs in ascending order.
                    return !live[input.predecessor.value] ||
                        !std::binary_search(block.predecessors.begin(), block.predecessors.end(), input.predecessor,
                            [&](BlockId left, BlockId right) { budget.step(); return left < right; });
                });
            }
            for (auto& op : block.operations) {
                budget.step();
                auto payload = std::move(p.operations[op.value]);
                op = OperationId{static_cast<std::uint32_t>(operations.size())};
                operations.push_back(std::move(payload));
            }
            blocks.push_back(std::move(block));
        }
        p.blocks = std::move(blocks);
        p.operations = std::move(operations);
        for (auto& function : p.functions) {
            function.entry = remap[function.entry.value];
            std::erase_if(function.blocks, [&](auto id) { return !live[id.value]; });
            for (auto& id : function.blocks) id = remap[id.value];
        }
        for (auto& block : p.blocks) {
            auto target = [&](BlockId& id) { id = remap[id.value]; };
            if (auto* jump = std::get_if<Jump>(&block.terminator)) target(jump->target);
            if (auto* branch = std::get_if<Branch>(&block.terminator)) {
                target(branch->when_true); target(branch->when_false);
            }
            if (auto* select = std::get_if<Switch>(&block.terminator)) {
                target(select->fallback);
                for (auto& item : select->cases) target(item.target);
            }
            for (auto& phi : block.phis) for (auto& input : phi.inputs) target(input.predecessor);
        }
        structured_detail::rebuild_predecessors(p, budget);
    }
}
} // namespace

Result<StructuredProgram> structure(LegalProgram input, Limits limits) {
    input.ir.limits = intersect(input.ir.limits, limits);
    const auto effective_limits = input.ir.limits;
    // The shared validator does not expose its consumed work. Reserve disjoint
    // budget slices for input validation, transformation and output validation.
    const auto validation_budget = effective_limits.analysis_steps / 3;
    input.ir.limits.analysis_steps = validation_budget;
    auto valid = validate(input.ir, IrForm::legal, Phase::structure);
    if (const auto* error = std::get_if<Diagnostic>(&valid)) return *error;
    input.ir.limits = effective_limits;
    try {
        structured_detail::Budget budget(effective_limits.analysis_steps - 2 * validation_budget);
        for (const auto& function : input.ir.functions) {
            budget.step();
            if (!input.ir.blocks[function.entry.value].phis.empty())
                structured_detail::fail("Function-entry phi has no incoming value on invocation", input.ir.blocks[function.entry.value].source,
                                        DiagnosticCode::undefined_read);
        }
        fold_branches(input.ir, budget);
        structured_detail::destroy_phis(input.ir, budget);
        StructuredProgram result{std::move(input.ir), std::move(input.requirements), std::move(input.profile), {}, {}};
        structured_detail::build_regions(result, budget);
        valid = structured_detail::validate_structured(result, validation_budget);
        if (const auto* error = std::get_if<Diagnostic>(&valid)) return *error;
        return result;
    } catch (const Diagnostic& error) {
        return error;
    }
}
} // namespace armigo::shader::pipeline
