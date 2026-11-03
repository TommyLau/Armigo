// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Bounds, type and scope checks are Armigo code.

#include "internal.h"

namespace armigo::shader::pipeline {
namespace {
using structured_detail::fail;
using structured_detail::Budget;

class StructuredValidator {
public:
    StructuredValidator(const StructuredProgram& program, std::uint64_t analysis_limit)
        : s(program), p(program.ir), budget(analysis_limit), owners(p.values.size()),
          definitions(p.values.size()), available(p.values.size()), blocks(p.blocks.size()), regions(s.regions.size()) {}
    void run();
private:
    const StructuredProgram& s;
    const Program& p;
    Budget budget;
    std::vector<FunctionId> owners;
    std::vector<std::uint32_t> definitions;
    std::vector<bool> available, blocks, regions;
    std::vector<std::vector<ValueId>> scopes;
    std::vector<RegionId> loops;
    std::vector<bool> returned_blocks;
    ValueType operand(const Operand&, FunctionId, bool lexical = false);
    void payloads();
    void tree(const StructuredFunction&);
};

ValueType StructuredValidator::operand(const Operand& input, FunctionId function, bool lexical) {
    budget.step();
    if (const auto* id = std::get_if<ValueId>(&input)) {
        if (id->value >= p.values.size() || !definitions[id->value])
            fail("Structured operand has no definition", {}, DiagnosticCode::undefined_read);
        if (owners[id->value] != function) fail("Structured operand crosses a function boundary");
        if (lexical && !available[id->value])
            fail("Structured value is used before initialization or outside its defining scope", {}, DiagnosticCode::undefined_read);
        return p.values[id->value].type;
    }
    if (const auto* literal = std::get_if<LiteralBits>(&input)) {
        if (literal->type > ValueType::predicate ||
            (literal->type == ValueType::word32 && literal->bits > UINT32_MAX) ||
            (literal->type == ValueType::predicate && literal->bits > 1)) fail("Invalid structured literal");
        return literal->type;
    }
    if (std::holds_alternative<Undefined>(input)) fail("Undefined structured operand", {}, DiagnosticCode::undefined_read);
    fail("Register or unresolved constant-buffer operand remains in structured IR");
}

void StructuredValidator::payloads() {
    if (s.profile.identity.empty() || s.profile.version == 0 || !s.profile.buffer_slots ||
        !s.profile.texture_slots || !s.profile.sampler_slots) fail("Structured program has no usable Metal profile");
    if (s.requirements.uses.size() > p.limits.resources || s.requirements.io.size() > p.limits.resources)
        fail("Structured resource limit exceeded", {}, DiagnosticCode::limit_exceeded);
    auto define = [&](ValueId id, FunctionId function) {
        budget.step();
        if (id.value >= p.values.size()) fail("Structured definition is outside its arena");
        if (definitions[id.value] && owners[id.value] != function) fail("Structured value has multiple function owners");
        owners[id.value] = function;
        ++definitions[id.value];
        if (definitions[id.value] > 1 && p.values[id.value].kind != ValueKind::mutable_local)
            fail("Multiply assigned structured value is not a mutable local");
    };
    for (std::uint32_t f = 0; f < p.functions.size(); ++f) {
        for (auto parameter : p.functions[f].parameters) {
            if (p.values[parameter.value].kind != ValueKind::temporary) fail("Function parameter cannot be redeclared as a mutable local");
            define(parameter, FunctionId{f});
        }
        for (auto type : p.functions[f].results) if (type > ValueType::predicate) fail("Invalid function result type");
    }
    for (const auto& block : p.blocks) {
        budget.step();
        if (!block.phis.empty()) fail("Structured IR still contains phi functions", block.source);
        for (auto id : block.operations)
            for (const auto& output : p.operations[id.value].outputs) {
                const auto* value = std::get_if<ValueId>(&output);
                if (!value) fail("Structured destination is not a ValueId");
                define(*value, block.function);
            }
    }
    auto resource = [&](std::optional<ResourceId> id, std::size_t kind) {
        if (!id || id->value >= s.requirements.uses.size() || s.requirements.uses[id->value].index() != kind)
            fail("Structured operation references an unresolved or mismatched resource", {}, DiagnosticCode::unresolved_resource);
    };
    for (const auto& block : p.blocks) {
        for (auto id : block.operations) {
            const auto& op = p.operations[id.value];
            for (const auto& input : op.inputs) operand(input, block.function);
            if (const auto* memory = std::get_if<MemoryInfo>(&op.info)) {
                if ((memory->space == AddressSpace::storage || memory->space == AddressSpace::constant_buffer) &&
                    op.opcode != Opcode::barrier && op.opcode != Opcode::memory_barrier) resource(memory->resource, 0);
            }
            if (const auto* texture = std::get_if<TextureInfo>(&op.info)) {
                if (!texture->descriptor || std::holds_alternative<DynamicHandle>(texture->texture) ||
                    (texture->sampler && std::holds_alternative<DynamicHandle>(*texture->sampler)))
                    fail("Unresolved texture facts in structured IR", op.source, DiagnosticCode::unresolved_resource);
                resource(texture->resource, 1);
                const bool sampler = (op.opcode == Opcode::texture_sample && texture->mode != TextureMode::fetch) ||
                    (op.opcode == Opcode::texture_query && texture->mode == TextureMode::query_lod);
                if (sampler) resource(texture->sampler_resource, 2);
                else if (texture->sampler_resource) fail("Unexpected structured sampler resource", op.source);
            }
        }
        if (const auto* branch = std::get_if<Branch>(&block.terminator)) operand(branch->condition, block.function);
        if (const auto* select = std::get_if<Switch>(&block.terminator)) operand(select->selector, block.function);
        if (const auto* result = std::get_if<Return>(&block.terminator))
            for (const auto& input : result->values) operand(input, block.function);
    }
}

void StructuredValidator::tree(const StructuredFunction& function) {
    enum class Event { enter, scoped, leave_scope, leave_loop };
    struct Frame { RegionId region; std::uint32_t depth; Event event; };
    std::vector<Frame> stack{{function.root, 1, Event::enter}};
    returned_blocks.assign(p.blocks.size(), false);
    scopes.emplace_back();
    for (auto parameter : p.functions[function.function.value].parameters) {
        available[parameter.value] = true;
        scopes.back().push_back(parameter);
    }
    while (!stack.empty()) {
        budget.step();
        const auto frame = stack.back();
        stack.pop_back();
        if (frame.event == Event::leave_scope) {
            for (auto id : scopes.back()) available[id.value] = false;
            scopes.pop_back();
            continue;
        }
        if (frame.event == Event::leave_loop) { loops.pop_back(); continue; }
        if (frame.event == Event::scoped) {
            scopes.emplace_back();
            stack.push_back({{}, 0, Event::leave_scope});
        }
        if (frame.depth > p.limits.nesting_depth)
            fail("Structured region nesting limit exceeded", {}, DiagnosticCode::limit_exceeded);
        const auto id = frame.region;
        if (id.value >= s.regions.size() || regions[id.value]) fail("Region is cyclic, shared or outside its arena");
        regions[id.value] = true;
        auto child = [&](RegionId next, bool scoped = false) {
            stack.push_back({next, frame.depth + 1, scoped ? Event::scoped : Event::enter});
        };
        const auto& node = s.regions[id.value];
        if (const auto* sequence = std::get_if<SequenceRegion>(&node)) {
            for (auto cursor = sequence->children.rbegin(); cursor != sequence->children.rend(); ++cursor) child(*cursor);
        } else if (const auto* block = std::get_if<BlockRegion>(&node)) {
            if (block->block.value >= p.blocks.size() || blocks[block->block.value]) fail("BlockRegion ownership is invalid");
            const auto& payload = p.blocks[block->block.value];
            if (payload.function != function.function) fail("BlockRegion belongs to another function");
            blocks[block->block.value] = true;
            for (auto operation : payload.operations) {
                const auto& op = p.operations[operation.value];
                for (const auto& input : op.inputs) operand(input, function.function, true);
                for (const auto& output : op.outputs) {
                    budget.step();
                    const auto value = std::get<ValueId>(output);
                    available[value.value] = true;
                    if (p.values[value.value].kind == ValueKind::temporary) {
                        scopes.back().push_back(value);
                    }
                }
            }
        } else if (const auto* branch = std::get_if<IfRegion>(&node)) {
            if (operand(branch->condition, function.function, true) != ValueType::predicate) fail("IfRegion requires a predicate");
            child(branch->when_false, true);
            child(branch->when_true, true);
        } else if (const auto* loop = std::get_if<LoopRegion>(&node)) {
            loops.push_back(id);
            stack.push_back({{}, 0, Event::leave_loop});
            child(loop->continuing, true);
            child(loop->body, true);
        } else if (const auto* returned = std::get_if<ReturnRegion>(&node)) {
            const auto& expected = p.functions[function.function.value].results;
            if (returned->values.size() != expected.size()) fail("ReturnRegion signature is invalid");
            for (std::size_t i = 0; i < expected.size(); ++i)
                if (operand(returned->values[i], function.function, true) != expected[i]) fail("ReturnRegion value type is invalid");
            // Every original return needs a structural counterpart, just as every
            // operation needs a BlockRegion. This is coverage, not a proof of
            // path-sensitive equivalence between arbitrary region trees.
            auto same = [](const Operand& left, const Operand& right) {
                if (const auto* value = std::get_if<ValueId>(&left))
                    return std::holds_alternative<ValueId>(right) && *value == std::get<ValueId>(right);
                const auto* literal = std::get_if<LiteralBits>(&left);
                return literal && std::holds_alternative<LiteralBits>(right) && *literal == std::get<LiteralBits>(right);
            };
            bool matched = false;
            for (auto block : p.functions[function.function.value].blocks) {
                budget.step();
                const auto* original = std::get_if<Return>(&p.blocks[block.value].terminator);
                if (!original || returned_blocks[block.value] || original->values.size() != returned->values.size()) continue;
                if (std::equal(original->values.begin(), original->values.end(), returned->values.begin(), same)) {
                    returned_blocks[block.value] = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) fail("ReturnRegion has no matching retained return");
        } else {
            const auto target = std::holds_alternative<BreakRegion>(node) ? std::get<BreakRegion>(node).loop : std::get<ContinueRegion>(node).loop;
            if (std::find(loops.begin(), loops.end(), target) == loops.end()) fail("Loop transfer does not target an enclosing loop");
        }
    }
    for (auto id : scopes.back()) available[id.value] = false;
    scopes.pop_back();
    for (auto block : p.functions[function.function.value].blocks)
        if (std::holds_alternative<Return>(p.blocks[block.value].terminator) && !returned_blocks[block.value])
            fail("Retained return has no structural counterpart");
}

void StructuredValidator::run() {
    if (s.regions.size() > p.limits.regions) fail("Structured region limit exceeded", {}, DiagnosticCode::limit_exceeded);
    if (s.functions.size() != p.functions.size()) fail("Structured function coverage is incomplete");
    for (const auto& value : p.values) if (value.kind > ValueKind::mutable_local) fail("Invalid structured value kind");
    payloads();
    std::vector<bool> functions(p.functions.size());
    for (const auto& function : s.functions) {
        if (function.function.value >= p.functions.size() || functions[function.function.value]) fail("Structured function ownership is invalid");
        functions[function.function.value] = true;
        tree(function);
    }
    if (std::find(blocks.begin(), blocks.end(), false) != blocks.end()) fail("Structured block has no region owner");
    if (std::find(regions.begin(), regions.end(), false) != regions.end()) fail("Structured region has no parent or function owner");
}
} // namespace

Status structured_detail::validate_structured(const StructuredProgram& program, std::uint64_t analysis_limit) {
    // Reuse the shared ownership, schema, numeric and CFG checks in their mutable
    // mode. Only the shadow's resolved CB annotation is removed because register
    // IR forbids it; payloads() checks legal resource resolution on the original.
    auto shadow = program.ir;
    shadow.limits.analysis_steps = analysis_limit / 2;
    for (auto& op : shadow.operations)
        if (auto* memory = std::get_if<MemoryInfo>(&op.info); memory && memory->space == AddressSpace::constant_buffer)
            memory->resource.reset();
    auto status = validate(shadow, IrForm::registers, Phase::structure);
    if (const auto* error = std::get_if<Diagnostic>(&status)) return *error;
    try {
        StructuredValidator(program, analysis_limit - shadow.limits.analysis_steps).run();
        return std::monostate{};
    } catch (const Diagnostic& error) {
        return error;
    }
}
Status validate(const StructuredProgram& program) {
    return structured_detail::validate_structured(program, program.ir.limits.analysis_steps);
}
} // namespace armigo::shader::pipeline
