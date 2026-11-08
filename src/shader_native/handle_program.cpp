// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture handles read from guest memory at an address the stage computes
// from constant-buffer words: the same handle in every invocation of a draw,
// so the host computes it at draw time (pipeline/handle_program.h) and binds
// the texture it names. A handle is taken this way only when its value
// depends on nothing but constant-buffer words, literals, integer
// arithmetic, comparisons and selections over them, and at least one 32-bit
// guest word read at such an address; any other operand leaves the handle
// to the established resolution. A stage that writes guest memory (stores,
// atomics, reductions) takes no handle program: its own writes, or another
// invocation's, could change the word between the draw's start and the
// load, which a host evaluation before the draw cannot see. The pass charges
// its work to the program's analysis budget and keeps at most the resource
// limit of programs.

#include "builder.h"
#include "pipeline/handle_program.h"
#include <algorithm>
#include <map>

namespace armigo::shader_native::detail {
namespace {
using Code = std::vector<std::uint32_t>;

[[noreturn]] void exceeded() {
    throw Failure{ir::diagnose(ir::Phase::construct_ssa, ir::DiagnosticCode::limit_exceeded, {}, "handle program work or count limit exceeded")};
}

class ProgramBuilder {
public:
    explicit ProgramBuilder(const ir::Program& program) : budget_(program.limits.analysis_steps) {
        for (const auto& op : program.operations)
            for (std::uint32_t index = 0; index < op.outputs.size(); ++index) {
                charge();
                if (const auto* id = std::get_if<ir::ValueId>(&op.outputs[index])) definitions_[id->value] = {&op, index};
            }
    }
    void charge(std::uint64_t steps = 1) { if ((work_ += steps) > budget_) exceeded(); }
    // The program computing operand, or nothing when it is not such a value.
    std::optional<Code> build(const ir::Operand& operand) {
        code_.clear(); kept_.clear(); uses_.clear(); guest_ = false; failed_ = false;
        count(operand, 0);
        emit(operand, 0);
        if (failed_ || !guest_) return std::nullopt;
        // A program the host evaluator cannot run (too deep for its stack)
        // is not taken.
        const auto zero = [](auto...) { return std::uint32_t{0}; };
        if (!ir::evaluate_handle_program(code_, zero, zero)) return std::nullopt;
        return code_;
    }

private:
    void op(ir::HandleOp value) { code_.push_back(static_cast<std::uint32_t>(value)); }
    // How often each value is used in the computation, so only values used
    // more than once are kept.
    void count(const ir::Operand& operand, unsigned depth) {
        charge();
        const auto* id = std::get_if<ir::ValueId>(&operand);
        if (!id || depth > 64) return;
        if (uses_[id->value]++) return;
        if (const auto found = definitions_.find(id->value); found != definitions_.end())
            for (const auto& input : found->second.first->inputs) count(input, depth + 1);
    }
    void emit(const ir::Operand& operand, unsigned depth) {
        charge();
        if (failed_) return;
        if (depth > 64 || code_.size() > 4096) { failed_ = true; return; }
        if (const auto* literal = std::get_if<ir::LiteralBits>(&operand)) {
            op(ir::HandleOp::immediate); code_.push_back(static_cast<std::uint32_t>(literal->bits)); return;
        }
        if (const auto* word = std::get_if<ir::ConstantWord>(&operand)) {
            op(ir::HandleOp::cbuf); code_.push_back(word->slot); code_.push_back(word->byte_offset); return;
        }
        const auto* id = std::get_if<ir::ValueId>(&operand);
        const auto found = id ? definitions_.find(id->value) : definitions_.end();
        if (found == definitions_.end()) { failed_ = true; return; }
        // A value used twice is computed once and kept.
        if (const auto kept = kept_.find(id->value); kept != kept_.end()) {
            op(ir::HandleOp::recall); code_.push_back(kept->second); return;
        }
        compute(*found->second.first, found->second.second, depth);
        if (failed_ || uses_[id->value] < 2) return;
        if (kept_.size() >= ir::handle_program_slots) { failed_ = true; return; }
        const auto slot = static_cast<std::uint32_t>(kept_.size());
        kept_.emplace(id->value, slot);
        op(ir::HandleOp::keep); code_.push_back(slot);
    }
    // Output `element` of operation o: a vector load's element k is the word
    // k element sizes past its address.
    void compute(const ir::Operation& o, std::uint32_t element, unsigned depth) {
        const auto integer = (o.numeric.scalar == ir::ScalarType::u32 || o.numeric.scalar == ir::ScalarType::s32) &&
                             o.numeric.domain != ir::Domain::floating;
        const auto is_signed = o.numeric.domain == ir::Domain::signed_integer;
        const auto inputs = [&](std::size_t count) {
            if (o.inputs.size() != count) { failed_ = true; return; }
            for (const auto& input : o.inputs) emit(input, depth + 1);
        };
        switch (o.opcode) {
        case ir::Opcode::copy: if (o.inputs.size() != 1) failed_ = true; else emit(o.inputs[0], depth + 1); return;
        case ir::Opcode::load: {
            const auto& m = std::get<ir::MemoryInfo>(o.info);
            if (m.element_bytes != 4 || element >= m.elements || m.atomic || !m.data.empty()) { failed_ = true; return; }
            const auto offset = m.byte_offset + 4 * element;
            if (m.space == ir::AddressSpace::constant_buffer && m.constant_buffer_slot) {
                if (!m.dynamic_byte_offset) {
                    op(ir::HandleOp::cbuf); code_.push_back(*m.constant_buffer_slot); code_.push_back(offset); return;
                }
                emit(o.inputs[m.dynamic_byte_offset->value], depth + 1);
                if (offset) { op(ir::HandleOp::immediate); code_.push_back(offset); op(ir::HandleOp::add); }
                op(ir::HandleOp::cbuf_dynamic); code_.push_back(*m.constant_buffer_slot); return;
            }
            if (m.space == ir::AddressSpace::guest && m.dynamic_byte_offset && m.global_address_high) {
                emit(o.inputs[m.dynamic_byte_offset->value], depth + 1);
                emit(o.inputs[m.global_address_high->value], depth + 1);
                op(ir::HandleOp::guest); code_.push_back(offset);
                guest_ = true; return;
            }
            failed_ = true; return;
        }
        case ir::Opcode::compare: {
            const auto* c = std::get_if<ir::ComparisonInfo>(&o.info);
            if (!c || !integer || o.inputs.size() != 2) { failed_ = true; return; }
            using C = ir::Compare;
            if (c->relation == C::never || c->relation == C::always) {
                op(ir::HandleOp::immediate); code_.push_back(c->relation == C::always); return;
            }
            inputs(2);
            switch (c->relation) {
            case C::equal: op(ir::HandleOp::equal); break;
            case C::not_equal: op(ir::HandleOp::not_equal); break;
            case C::less: op(is_signed ? ir::HandleOp::less_signed : ir::HandleOp::less); break;
            case C::less_equal: op(is_signed ? ir::HandleOp::less_equal_signed : ir::HandleOp::less_equal); break;
            case C::greater: op(is_signed ? ir::HandleOp::greater_signed : ir::HandleOp::greater); break;
            case C::greater_equal: op(is_signed ? ir::HandleOp::greater_equal_signed : ir::HandleOp::greater_equal); break;
            default: failed_ = true; break;
            }
            return;
        }
        case ir::Opcode::select: inputs(3); op(ir::HandleOp::select); return;
        case ir::Opcode::logical_and: inputs(2); op(ir::HandleOp::logical_and); return;
        case ir::Opcode::logical_or: inputs(2); op(ir::HandleOp::logical_or); return;
        case ir::Opcode::logical_not: inputs(1); op(ir::HandleOp::logical_not); return;
        default: break;
        }
        if (!integer) { failed_ = true; return; }
        switch (o.opcode) {
        case ir::Opcode::add: inputs(2); op(ir::HandleOp::add); return;
        case ir::Opcode::subtract: inputs(2); op(ir::HandleOp::subtract); return;
        case ir::Opcode::multiply: inputs(2); op(ir::HandleOp::multiply); return;
        case ir::Opcode::bit_and: inputs(2); op(ir::HandleOp::bit_and); return;
        case ir::Opcode::bit_or: inputs(2); op(ir::HandleOp::bit_or); return;
        case ir::Opcode::bit_xor: inputs(2); op(ir::HandleOp::bit_xor); return;
        case ir::Opcode::bit_not: inputs(1); op(ir::HandleOp::bit_not); return;
        case ir::Opcode::shift_left: inputs(2); op(ir::HandleOp::shift_left); return;
        case ir::Opcode::shift_right: inputs(2); op(is_signed ? ir::HandleOp::shift_right_signed : ir::HandleOp::shift_right); return;
        case ir::Opcode::minimum: inputs(2); op(is_signed ? ir::HandleOp::minimum_signed : ir::HandleOp::minimum); return;
        case ir::Opcode::maximum: inputs(2); op(is_signed ? ir::HandleOp::maximum_signed : ir::HandleOp::maximum); return;
        default: failed_ = true; return;
        }
    }

    std::map<std::uint32_t, std::pair<const ir::Operation*, std::uint32_t>> definitions_;
    std::map<std::uint32_t, std::uint32_t> kept_, uses_;
    Code code_;
    bool guest_{}, failed_{};
    std::uint64_t work_ = 0, budget_;
};
} // namespace

std::vector<std::vector<std::uint32_t>> build_handle_programs(ir::Program& program) {
    for (const auto& o : program.operations)
        if (const auto* m = std::get_if<ir::MemoryInfo>(&o.info); m && m->space == ir::AddressSpace::guest && (m->access != ir::Access::read || m->atomic))
            return {};
    ProgramBuilder builder{program};
    std::vector<Code> programs;
    std::map<Code, std::uint32_t> indices;
    const auto key_of = [&](const Code& code) {
        builder.charge(code.size());
        auto [found, added] = indices.try_emplace(code, static_cast<std::uint32_t>(programs.size()));
        if (added) {
            if (programs.size() >= program.limits.resources) exceeded();
            programs.push_back(code);
        }
        return ir::TextureKey{ir::handle_program_slot, found->second * 4};
    };
    for (auto& o : program.operations) {
        auto* t = std::get_if<ir::TextureInfo>(&o.info);
        if (!t) continue;
        bool rewritten = false;
        for (auto* source : {&t->texture, t->sampler ? &*t->sampler : nullptr}) {
            const auto* dynamic = source ? std::get_if<ir::DynamicHandle>(source) : nullptr;
            if (!dynamic) continue;
            if (auto code = builder.build(o.inputs[dynamic->input.value])) { *source = key_of(*code); rewritten = true; }
        }
        if (!rewritten) continue;
        // The handle inputs no role names any more leave the operation; the
        // roles are renumbered over the inputs that stay.
        std::vector<bool> live(o.inputs.size());
        std::vector<ir::InputIndex*> roles;
        for (auto* list : {&t->coordinates, &t->gradients_x, &t->gradients_y, &t->offsets, &t->data})
            for (auto& index : *list) roles.push_back(&index);
        for (auto* optional : {&t->array_layer, &t->array_index, &t->level_or_bias, &t->compare, &t->sample_index, &t->lod_clamp})
            if (*optional) roles.push_back(&**optional);
        for (auto* source : {&t->texture, t->sampler ? &*t->sampler : nullptr})
            if (auto* dynamic = source ? std::get_if<ir::DynamicHandle>(source) : nullptr) roles.push_back(&dynamic->input);
        for (const auto* role : roles) live[role->value] = true;
        std::vector<std::uint16_t> remap(o.inputs.size());
        std::vector<ir::Operand> inputs;
        for (std::size_t i = 0; i < live.size(); ++i)
            if (live[i]) { remap[i] = static_cast<std::uint16_t>(inputs.size()); inputs.push_back(o.inputs[i]); }
        for (auto* role : roles) role->value = remap[role->value];
        o.inputs = std::move(inputs);
    }
    return programs;
}
} // namespace armigo::shader_native::detail
