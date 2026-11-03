// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses explicit block terminators and reconvergence selectors instead
// of chained labels, and rejects unsupported transfers explicitly.

#include "flow.h"
#include "../dispatch.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
constexpr NumericMode boolean_mode() { return {Domain::boolean, ScalarType::boolean}; }
constexpr NumericMode word_mode() { return {Domain::unsigned_integer, ScalarType::u32}; }

[[nodiscard]] Result<Operand> read_flag(EmitterContext& context, ConditionCode code) {
    return context.read(Register{RegisterFile::condition_code, static_cast<std::uint16_t>(code)});
}

[[nodiscard]] Result<Operand> combine(EmitterContext& context, Opcode opcode,
                                      Result<Operand> left, Result<Operand> right) {
    if (auto* error = std::get_if<Diagnostic>(&left)) return std::move(*error);
    if (auto* error = std::get_if<Diagnostic>(&right)) return std::move(*error);
    return context.binary(opcode, std::get<Operand>(std::move(left)), std::get<Operand>(std::move(right)),
        boolean_mode(), ValueType::predicate);
}

[[nodiscard]] Result<Operand> invert(EmitterContext& context, Result<Operand> value) {
    if (auto* error = std::get_if<Diagnostic>(&value)) return std::move(*error);
    return context.unary(Opcode::logical_not, std::get<Operand>(std::move(value)),
        boolean_mode(), ValueType::predicate);
}

// Condition-code selectors occupy values 0..31.
// Shading conditions (selectors 24..29) reject here; branch_condition gives
// EXIT its own answer for them.
[[nodiscard]] Result<Operand> cc_condition(EmitterContext& context, std::uint64_t selector, SourceLocation location) {
    const auto sign = [&] { return read_flag(context, ConditionCode::sign); };
    const auto zero = [&] { return read_flag(context, ConditionCode::zero); };
    const auto carry = [&] { return read_flag(context, ConditionCode::carry); };
    const auto overflow = [&] { return read_flag(context, ConditionCode::overflow); };
    switch (selector) {
    case 0: return Operand(LiteralBits{0, ValueType::predicate});
    case 1: return combine(context, Opcode::logical_xor,
        combine(context, Opcode::logical_and, sign(), invert(context, zero())), overflow());
    case 2: return combine(context, Opcode::logical_and, invert(context, sign()), zero());
    case 3: return combine(context, Opcode::logical_xor, sign(),
        combine(context, Opcode::logical_or, zero(), overflow()));
    case 4: return invert(context, combine(context, Opcode::logical_or,
        combine(context, Opcode::logical_xor, sign(), overflow()), zero()));
    case 5: return invert(context, zero());
    case 6: return invert(context, combine(context, Opcode::logical_xor, sign(), overflow()));
    case 7: return invert(context, combine(context, Opcode::logical_and, sign(), zero()));
    case 8: return combine(context, Opcode::logical_and, sign(), zero());
    case 9: return combine(context, Opcode::logical_xor, sign(), overflow());
    case 10: return zero();
    case 11: return combine(context, Opcode::logical_or,
        combine(context, Opcode::logical_xor, sign(), overflow()), zero());
    case 12: return combine(context, Opcode::logical_xor, invert(context, sign()),
        combine(context, Opcode::logical_or, overflow(), zero()));
    case 13: return combine(context, Opcode::logical_or, sign(), invert(context, zero()));
    case 14: return combine(context, Opcode::logical_xor,
        combine(context, Opcode::logical_or, invert(context, sign()), zero()), overflow());
    case 15: return Operand(LiteralBits{1, ValueType::predicate});
    case 16: return invert(context, overflow());
    case 17: return invert(context, carry());
    case 18: return invert(context, sign());
    case 19: return combine(context, Opcode::logical_or, zero(), invert(context, carry()));
    case 20: return combine(context, Opcode::logical_and, carry(), invert(context, zero()));
    case 21: return sign();
    case 22: return carry();
    case 23: return overflow();
    case 30: return combine(context, Opcode::logical_or, sign(), zero());
    case 31: return invert(context, combine(context, Opcode::logical_or, sign(), zero()));
    default:
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_modifier, location,
            "Condition code selector is not supported");
    }
}

struct BranchCondition {
    Operand value;
    bool constant_true{};
    bool constant_false{};
};

[[nodiscard]] bool predicate_literal(const Operand& operand, bool state) {
    const auto* literal = std::get_if<LiteralBits>(&operand);
    return literal != nullptr && literal->type == ValueType::predicate && literal->bits == (state ? 1u : 0u);
}

// Combines the instruction predicate with the condition-code selector the way
// branch emission. NoPred encodings branch unconditionally.
[[nodiscard]] Result<BranchCondition> branch_condition(EmitterContext& context, const DecodedInstruction& instruction) {
    const SourceLocation location{instruction.pc, instruction.name};
    if (!instruction.predicate.enabled) {
        return BranchCondition{LiteralBits{1, ValueType::predicate}, true, false};
    }
    auto read = context.read(Register{RegisterFile::predicate, instruction.predicate.index});
    if (auto* error = std::get_if<Diagnostic>(&read)) return *error;
    auto predicate = std::get<Operand>(std::move(read));
    if (instruction.predicate.inverted) {
        if (predicate_literal(predicate, true)) {
            predicate = LiteralBits{0, ValueType::predicate};
        } else if (predicate_literal(predicate, false)) {
            predicate = LiteralBits{1, ValueType::predicate};
        } else {
            auto inverted = invert(context, std::move(predicate));
            if (auto* error = std::get_if<Diagnostic>(&inverted)) return *error;
            predicate = std::get<Operand>(std::move(inverted));
        }
    }
    auto selector = decoded_field(instruction, DecodedField::condition);
    if (auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    const auto bits = std::get<std::uint64_t>(selector);
    // FCSM_TR (selector 28) guards the EXIT that follows a VOTE.VTG cull vote
    // in vertex stages. The vote is not translated, so the condition never holds
    // and the guarded EXIT falls through to the remaining attribute stores. The
    // other coverage-sample-mode selectors (24..27, 29) have no observed use and
    // stay unsupported below.
    if (instruction.name == InstructionName::exit && bits == 28)
        return BranchCondition{LiteralBits{0, ValueType::predicate}, false, true};
    if (bits == 15) { // Ccc T: the predicate alone decides.
        if (instruction.predicate.index == 7 && !instruction.predicate.inverted) {
            return BranchCondition{LiteralBits{1, ValueType::predicate}, true, false};
        }
        return BranchCondition{std::move(predicate), false, false};
    }
    auto condition = cc_condition(context, bits, location);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    auto value = std::get<Operand>(std::move(condition));
    if (instruction.predicate.index == 7 && !instruction.predicate.inverted) {
        return BranchCondition{std::move(value), predicate_literal(value, true), predicate_literal(value, false)};
    }
    auto combined = combine(context, Opcode::logical_and, std::move(predicate), std::move(value));
    if (auto* error = std::get_if<Diagnostic>(&combined)) return *error;
    return BranchCondition{std::get<Operand>(std::move(combined)), false, false};
}

// The SPH target mask assigns four consecutive registers to each active render
// target. Disabled targets consume no registers; disabled lanes retain their
// position within an active target's four-register group.
[[nodiscard]] Status export_fragment_outputs(EmitterContext& context) {
    const auto& program = context.decoded_program();
    if (program.stage != Stage::fragment) return std::monostate{};
    const auto header_word = [&](std::size_t offset) {
        std::uint32_t word = 0;
        for (unsigned byte = 0; byte < 4; ++byte) {
            word |= std::to_integer<std::uint32_t>(program.sph[offset + byte]) << (byte * 8);
        }
        return word;
    };
    const auto colors = header_word(0x48);
    const auto extra = header_word(0x4c);
    const bool sample_mask = (extra & 1u) != 0;
    const bool depth = (extra & 2u) != 0;
    if (sample_mask || depth) {
        for (unsigned target = 0; target < 8; ++target) {
            const auto mask = (colors >> (4 * target)) & 15u;
            if (mask != 0 && mask != 15) {
                return context.error(DiagnosticCode::unsupported_capability,
                    "Sparse color exports with depth or sample mask have no established register layout");
            }
        }
    }
    const auto store = [&](IoInfo info, std::uint16_t reg) -> Status {
        auto value = context.read(Register{RegisterFile::gpr, reg});
        if (const auto* error = std::get_if<Diagnostic>(&value)) return *error;
        const auto scalar = info.type.scalar;
        const auto domain = scalar == ScalarType::f32 ? Domain::floating :
            scalar == ScalarType::s32 ? Domain::signed_integer : Domain::unsigned_integer;
        info.data = {InputIndex{0}};
        info.output_storage = true;
        const std::array inputs{std::get<Operand>(value)};
        return context.store_io(std::move(info), inputs, NumericMode{domain, scalar, domain == Domain::floating ?
            Rounding::nearest_even : Rounding::not_applicable});
    };
    std::uint16_t base = 0;
    for (std::uint16_t target = 0; target < 8; ++target) {
        const auto mask = (colors >> (4 * target)) & 15u;
        if (mask == 0) continue;
        auto type = context.facts().require(FactQuery<FactKind::fragment_output_type>{target},
            Phase::emit_ir, {context.instruction().pc, context.instruction().name});
        if (const auto* error = std::get_if<Diagnostic>(&type)) return *error;
        const auto interface = std::get<IoType>(type);
        if (interface.components == 0 || interface.components > 4 ||
            (mask >> interface.components) != 0 ||
            (interface.scalar != ScalarType::f32 && interface.scalar != ScalarType::s32 &&
             interface.scalar != ScalarType::u32)) {
            return context.error(DiagnosticCode::unsupported_capability,
                "Fragment color output type cannot represent the SPH component mask");
        }
        for (unsigned component = 0; component < 4; ++component) {
            if ((mask & (1u << component)) == 0) continue;
            IoInfo info;
            info.semantic = IoSemantic::color;
            info.location = target;
            info.component_mask = static_cast<std::uint8_t>(1u << component);
            info.type = interface;
            if (auto outcome = store(info, static_cast<std::uint16_t>(base + component));
                std::holds_alternative<Diagnostic>(outcome)) return outcome;
        }
        base += 4;
    }
    if (sample_mask) {
        IoInfo info;
        info.semantic = IoSemantic::sample_mask;
        info.type = {ScalarType::u32, 1};
        if (auto outcome = store(info, base); std::holds_alternative<Diagnostic>(outcome)) return outcome;
    }
    if (depth) {
        IoInfo info;
        info.semantic = IoSemantic::fragment_depth;
        info.type = {ScalarType::f32, 1};
        if (auto outcome = store(info, base + 1); std::holds_alternative<Diagnostic>(outcome)) return outcome;
    }
    return std::monostate{};
}

[[nodiscard]] Status exit_stage(EmitterContext& context) {
    if (auto outcome = export_fragment_outputs(context); std::holds_alternative<Diagnostic>(outcome)) return outcome;
    return context.terminate(Return{});
}

// The block receiving the not-taken path: the decoded fallthrough when it
// exists, otherwise a synthesized end block returning from the stage.
[[nodiscard]] Result<BlockId> fallthrough_block(EmitterContext& context, const DecodedInstruction&) {
    const auto current = context.block();
    auto next = decoded_fallthrough(context.decoded_program(), context.decoded_block());
    if (auto* error = std::get_if<Diagnostic>(&next)) return *error;
    const auto decoded_next = std::get<BlockId>(next);
    if (decoded_next.valid()) {
        return context.label(context.decoded_program().blocks[decoded_next.value].begin_pc);
    }
    auto end = context.create_block();
    if (auto* error = std::get_if<Diagnostic>(&end)) return *error;
    const auto end_block = std::get<BlockId>(end);
    if (const auto outcome = context.enter_block(end_block); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.terminate(Return{}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.enter_block(current); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    return end_block;
}

[[nodiscard]] std::vector<ReconvergenceTarget> pop_targets(const DecodedProgram& program, BlockId block, Pc pop_pc) {
    std::vector<ReconvergenceTarget> targets;
    if (block.value < program.blocks.size()) {
        for (const auto& target : program.blocks[block.value].reconvergence) {
            if (target.pop_pc == pop_pc) targets.push_back(target);
        }
    }
    std::sort(targets.begin(), targets.end(),
        [](const ReconvergenceTarget& left, const ReconvergenceTarget& right) {
            return left.selector < right.selector;
        });
    return targets;
}

// Multi-target dispatch: compare the reconvergence selector against each
// alternative; the last alternative runs unconditionally.
[[nodiscard]] Status emit_dispatch(EmitterContext& context, BlockId block,
                                   const std::vector<ReconvergenceTarget>& targets) {
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const auto& target = targets[index];
        const auto& decoded = context.decoded_program();
        if (!target.target.valid() || target.target.value >= decoded.blocks.size()) {
            return context.error(DiagnosticCode::invalid_ir, "Unknown decoded reconvergence target");
        }
        auto label = context.label(decoded.blocks[target.target.value].begin_pc);
        if (auto* error = std::get_if<Diagnostic>(&label)) return *error;
        const auto destination = std::get<BlockId>(label);
        if (index + 1 == targets.size()) {
            return context.terminate(Jump{destination});
        }
        auto local = context.reconvergence_selector(block, target.pop_pc);
        if (auto* error = std::get_if<Diagnostic>(&local)) return *error;
        auto matched = context.compare(std::get<ValueId>(std::move(local)),
            LiteralBits{target.selector, ValueType::word32}, ComparisonInfo{Compare::equal, false}, word_mode());
        if (auto* error = std::get_if<Diagnostic>(&matched)) return *error;
        auto next = context.create_block();
        if (auto* error = std::get_if<Diagnostic>(&next)) return *error;
        if (const auto outcome = context.terminate(
                Branch{std::get<Operand>(std::move(matched)), destination, std::get<BlockId>(next)}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
        if (const auto outcome = context.enter_block(std::get<BlockId>(next)); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    }
    return std::monostate{};
}

[[nodiscard]] Status emit_pop(EmitterContext& context) {
    const auto& instruction = context.instruction();
    const auto block = context.decoded_block();
    const auto& program = context.decoded_program();
    const auto targets = pop_targets(program, block, instruction.pc);
    if (targets.empty()) {
        return context.error(DiagnosticCode::invalid_ir, "Pop instruction has no decoded reconvergence target");
    }
    auto condition = branch_condition(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    const BranchCondition& plan = std::get<BranchCondition>(condition);
    if (plan.constant_true) {
        return emit_dispatch(context, block, targets);
    }
    auto fallthrough = fallthrough_block(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&fallthrough)) return *error;
    if (plan.constant_false) {
        return context.terminate(Jump{std::get<BlockId>(fallthrough)});
    }
    auto dispatch = context.create_block();
    if (auto* error = std::get_if<Diagnostic>(&dispatch)) return *error;
    if (const auto outcome = context.terminate(
            Branch{plan.value, std::get<BlockId>(dispatch), std::get<BlockId>(fallthrough)}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.enter_block(std::get<BlockId>(dispatch)); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = emit_dispatch(context, block, targets); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    return context.enter_block(std::get<BlockId>(fallthrough));
}

[[nodiscard]] Status emit_push(EmitterContext& context) {
    const auto& instruction = context.instruction();
    const auto& program = context.decoded_program();
    std::optional<TokenId> token;
    for (std::size_t index = 0; index < program.tokens.size(); ++index) {
        if (program.tokens[index].push_pc == instruction.pc) {
            token = TokenId{static_cast<std::uint32_t>(index)};
            break;
        }
    }
    if (!token) {
        return context.error(DiagnosticCode::invalid_ir, "Push instruction has no decoded reconvergence token");
    }
    for (const auto consumer : program.tokens[token->value].consumers) {
        for (const auto& target : program.blocks[consumer.value].reconvergence) {
            if (target.token != *token) continue;
            auto local = context.reconvergence_selector(consumer, target.pop_pc);
            if (auto* error = std::get_if<Diagnostic>(&local)) return *error;
            const auto outcome = context.assign(std::get<ValueId>(std::move(local)),
                LiteralBits{target.selector, ValueType::word32});
            if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
        }
    }
    return std::monostate{};
}
} // namespace

Status emit_flow_bra(EmitterContext& context) {
    const auto& instruction = context.instruction();
    auto target = flow::flow_target(instruction);
    if (auto* error = std::get_if<Diagnostic>(&target)) return *error;
    auto label = context.label(std::get<Pc>(target));
    if (auto* error = std::get_if<Diagnostic>(&label)) return *error;
    auto condition = branch_condition(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    const BranchCondition& plan = std::get<BranchCondition>(condition);
    if (plan.constant_true) {
        return context.terminate(Jump{std::get<BlockId>(label)});
    }
    auto fallthrough = fallthrough_block(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&fallthrough)) return *error;
    if (plan.constant_false) {
        return context.terminate(Jump{std::get<BlockId>(fallthrough)});
    }
    return context.terminate(Branch{plan.value, std::get<BlockId>(label), std::get<BlockId>(fallthrough)});
}

Status emit_flow_sync(EmitterContext& context) { return emit_pop(context); }
Status emit_flow_brk(EmitterContext& context) { return emit_pop(context); }
Status emit_flow_cont(EmitterContext& context) { return emit_pop(context); }
Status emit_flow_ssy(EmitterContext& context) { return emit_push(context); }
Status emit_flow_pbk(EmitterContext& context) { return emit_push(context); }
Status emit_flow_pcnt(EmitterContext& context) { return emit_push(context); }

Status emit_flow_exit(EmitterContext& context) {
    if (context.function() != context.decoded_program().entry) {
        return context.error(DiagnosticCode::unsupported_instruction, "EXIT is only valid in the entry function");
    }
    const auto& instruction = context.instruction();
    auto condition = branch_condition(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    const BranchCondition& plan = std::get<BranchCondition>(condition);
    if (plan.constant_true) {
        return exit_stage(context);
    }
    auto fallthrough = fallthrough_block(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&fallthrough)) return *error;
    if (plan.constant_false) {
        return context.terminate(Jump{std::get<BlockId>(fallthrough)});
    }
    auto taken = context.create_block();
    if (auto* error = std::get_if<Diagnostic>(&taken)) return *error;
    if (const auto outcome = context.terminate(
            Branch{plan.value, std::get<BlockId>(taken), std::get<BlockId>(fallthrough)}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.enter_block(std::get<BlockId>(taken)); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = exit_stage(context); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    return context.enter_block(std::get<BlockId>(fallthrough));
}

Status emit_flow_kil(EmitterContext& context) {
    const auto& instruction = context.instruction();
    auto condition = branch_condition(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    const BranchCondition& plan = std::get<BranchCondition>(condition);
    auto fallthrough = fallthrough_block(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&fallthrough)) return *error;
    const auto fallthrough_block_id = std::get<BlockId>(fallthrough);
    if (plan.constant_false) {
        return context.terminate(Jump{fallthrough_block_id});
    }
    if (plan.constant_true) {
        if (const auto outcome = context.discard(); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
        return context.terminate(Jump{fallthrough_block_id});
    }
    auto guarded = context.create_block();
    if (auto* error = std::get_if<Diagnostic>(&guarded)) return *error;
    if (const auto outcome = context.terminate(
            Branch{plan.value, std::get<BlockId>(guarded), fallthrough_block_id}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.enter_block(std::get<BlockId>(guarded)); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.discard(); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    if (const auto outcome = context.terminate(Jump{fallthrough_block_id}); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    return context.enter_block(fallthrough_block_id);
}

Status emit_flow_cal(EmitterContext& context) {
    return context.error(DiagnosticCode::unimplemented, "Call ABI handling is not implemented");
}

Status emit_flow_ret(EmitterContext& context) {
    if (context.function() == context.decoded_program().entry) {
        return context.error(DiagnosticCode::unsupported_instruction, "RET cannot terminate the entry function");
    }
    const auto& instruction = context.instruction();
    auto condition = branch_condition(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&condition)) return *error;
    const auto& plan = std::get<BranchCondition>(condition);
    if (plan.constant_true) return context.terminate(Return{});
    auto fallthrough = fallthrough_block(context, instruction);
    if (auto* error = std::get_if<Diagnostic>(&fallthrough)) return *error;
    if (plan.constant_false) return context.terminate(Jump{std::get<BlockId>(fallthrough)});
    auto taken = context.create_block();
    if (auto* error = std::get_if<Diagnostic>(&taken)) return *error;
    if (auto outcome = context.terminate(Branch{plan.value, std::get<BlockId>(taken), std::get<BlockId>(fallthrough)});
        std::holds_alternative<Diagnostic>(outcome)) return outcome;
    if (auto outcome = context.enter_block(std::get<BlockId>(taken));
        std::holds_alternative<Diagnostic>(outcome)) return outcome;
    if (auto outcome = context.terminate(Return{}); std::holds_alternative<Diagnostic>(outcome)) return outcome;
    return context.enter_block(std::get<BlockId>(fallthrough));
}
} // namespace armigo::shader::pipeline
