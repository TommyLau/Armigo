// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Decoded blocks dispatch each instruction through the generated emitter
// table. Armigo's driver adds the frozen contract's
// predication rule: the instruction predicate is evaluated once and branches
// around the whole expansion. A final same-guard transfer reuses the flow
// lane's handling with
// its predicate already satisfied by the enclosing region.

#include "emit_ir.h"

#include "pipeline/emit/context_ir.h"
#include "pipeline/phases.h"

#include <algorithm>
#include <array>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
// Standalone flow instructions own their predication and control flow.
constexpr bool is_flow_instruction(InstructionName name) {
    switch (name) {
    case InstructionName::bra: case InstructionName::brx: case InstructionName::brk:
    case InstructionName::cont: case InstructionName::ret: case InstructionName::exit:
    case InstructionName::kil:
    case InstructionName::jmp: case InstructionName::jmx: case InstructionName::longjmp:
    case InstructionName::plongjmp: case InstructionName::ssy: case InstructionName::pbk:
    case InstructionName::pcnt: case InstructionName::sync: case InstructionName::pexit:
    case InstructionName::pret: case InstructionName::cal: case InstructionName::jcal:
        return true;
    default: return false;
    }
}

constexpr Limits min_limits(const Limits& inherited, const Limits& supplied) {
    return Limits{
        .code_bytes = std::min(inherited.code_bytes, supplied.code_bytes),
        .instructions = std::min(inherited.instructions, supplied.instructions),
        .functions = std::min(inherited.functions, supplied.functions),
        .blocks = std::min(inherited.blocks, supplied.blocks),
        .edges = std::min(inherited.edges, supplied.edges),
        .operations = std::min(inherited.operations, supplied.operations),
        .values = std::min(inherited.values, supplied.values),
        .phi_inputs = std::min(inherited.phi_inputs, supplied.phi_inputs),
        .operands_per_operation = std::min(inherited.operands_per_operation, supplied.operands_per_operation),
        .token_depth = std::min(inherited.token_depth, supplied.token_depth),
        .token_states = std::min(inherited.token_states, supplied.token_states),
        .regions = std::min(inherited.regions, supplied.regions),
        .nesting_depth = std::min(inherited.nesting_depth, supplied.nesting_depth),
        .resources = std::min(inherited.resources, supplied.resources),
        .dependencies = std::min(inherited.dependencies, supplied.dependencies),
        .source_bytes = std::min(inherited.source_bytes, supplied.source_bytes),
        .analysis_steps = std::min(inherited.analysis_steps, supplied.analysis_steps),
    };
}

Status check_local_bindings(std::span<const EmitterBinding> local) {
    std::array<bool, instruction_name_count> seen{};
    for (const auto& binding : local) {
        const auto index = static_cast<std::size_t>(binding.instruction);
        if (index == 0 || index >= seen.size() || seen[index] || binding.emit == nullptr) {
            return diagnose(Phase::emit_ir, DiagnosticCode::duplicate_dispatch, {},
                            "Local emitter bindings are null, invalid or duplicated");
        }
        seen[index] = true;
    }
    return std::monostate{};
}

Status dispatch_emitter(RegisterEmitter& context, std::span<const EmitterBinding> local) {
    const auto name = context.instruction().name;
    for (const auto& binding : local) {
        if (binding.instruction == name) return binding.emit(context);
    }
    return dispatch_instruction(context);
}

bool mergeable_guard(const DecodedInstruction& instruction) {
    if (!instruction.predicate.enabled ||
        (instruction.predicate.index == 7 && !instruction.predicate.inverted) ||
        is_flow_instruction(instruction.name)) return false;
    // These instructions can change execution or synchronization state even
    // without a branch terminator. Keep their individual guarded regions.
    switch (instruction.name) {
    case InstructionName::bar: case InstructionName::depbar: case InstructionName::membar:
    case InstructionName::bpt: case InstructionName::rtt: case InstructionName::out:
        return false;
    // R2P's dynamic write mask can target any predicate; its decoded register
    // accesses do not yet enumerate those conditional writes.
    case InstructionName::r2p: return false;
    default: break;
    }
    const Register guard{RegisterFile::predicate, instruction.predicate.index};
    return std::none_of(instruction.registers.begin(), instruction.registers.end(),
        [&](const RegisterAccess& access) { return access.reg == guard && access.access != Access::read; });
}

bool same_guard(const DecodedInstruction& left, const DecodedInstruction& right) {
    return left.predicate.index == right.predicate.index &&
           left.predicate.inverted == right.predicate.inverted;
}

constexpr bool is_guarded_transfer(InstructionName name) {
    switch (name) {
    case InstructionName::sync: case InstructionName::brk: case InstructionName::cont:
    case InstructionName::bra: case InstructionName::exit: case InstructionName::ret:
    case InstructionName::kil: return true;
    default: return false;
    }
}

std::size_t guard_run_end(std::span<const DecodedInstruction> instructions, std::size_t begin) {
    auto end = begin + 1;
    while (end < instructions.size() && mergeable_guard(instructions[end]) &&
           same_guard(instructions[begin], instructions[end])) ++end;
    // Decoded transfers end their block. Do not merge across block boundaries
    // or absorb any instruction after a transfer in malformed input.
    if (end + 1 == instructions.size() && is_guarded_transfer(instructions[end].name) &&
        instructions[end].predicate.enabled && same_guard(instructions[begin], instructions[end])) ++end;
    return end;
}

Result<BlockId> guard_continuation(RegisterEmitter& context, bool transfers) {
    if (!transfers) return context.create_block();
    auto next = decoded_fallthrough(context.decoded_program(), context.decoded_block());
    if (const auto* failure = std::get_if<Diagnostic>(&next)) return *failure;
    const auto target = std::get<BlockId>(next);
    if (target.valid()) return context.label(context.decoded_program().blocks[target.value].begin_pc);
    // Match the flow lane's end-of-stream fallthrough: an empty stage return.
    const auto current = context.block();
    auto end = context.create_block();
    if (const auto* failure = std::get_if<Diagnostic>(&end)) return *failure;
    if (auto status = context.enter_block(std::get<BlockId>(end)); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    if (auto status = context.terminate(Return{}); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    if (auto status = context.enter_block(current); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    return end;
}

// Evaluate the predicate before any destination write and branch around all
// expansions in a run. An adjacent complementary run occupies the other arm
// of the same branch: neither run can change the predicate being tested.
// A final transfer uses the same flow emitter with its outer guard satisfied.
Status emit_instructions(RegisterEmitter& context, std::span<const DecodedInstruction> instructions,
                         std::span<const DecodedInstruction> alternative,
                         std::span<const EmitterBinding> local) {
    context.begin_instruction(instructions.front());
    const auto& decoded = context.instruction();
    const bool always_true = decoded.predicate.index == 7 && !decoded.predicate.inverted;
    if (!decoded.predicate.enabled || always_true || is_flow_instruction(decoded.name)) {
        return dispatch_emitter(context, local);
    }
    auto condition = context.read(Register{RegisterFile::predicate, decoded.predicate.index});
    if (const auto* failure = std::get_if<Diagnostic>(&condition)) return *failure;
    Operand guard = std::get<Operand>(condition);
    if (decoded.predicate.inverted) {
        auto negated = context.unary(Opcode::logical_not, guard,
                                     NumericMode{Domain::boolean, ScalarType::boolean}, ValueType::predicate);
        if (const auto* failure = std::get_if<Diagnostic>(&negated)) return *failure;
        guard = std::get<Operand>(negated);
    }
    auto body = context.create_block();
    if (const auto* failure = std::get_if<Diagnostic>(&body)) return *failure;
    const bool transfers = is_guarded_transfer(instructions.back().name) ||
        (!alternative.empty() && is_guarded_transfer(alternative.back().name));
    auto continuation = guard_continuation(context, transfers);
    if (const auto* failure = std::get_if<Diagnostic>(&continuation)) return *failure;
    BlockId when_false = std::get<BlockId>(continuation);
    if (!alternative.empty()) {
        auto other_body = context.create_block();
        if (const auto* failure = std::get_if<Diagnostic>(&other_body)) return *failure;
        when_false = std::get<BlockId>(other_body);
    }
    if (auto failure = context.terminate(Branch{guard, std::get<BlockId>(body), when_false});
        std::holds_alternative<Diagnostic>(failure)) return failure;
    if (auto failure = context.enter_block(std::get<BlockId>(body));
        std::holds_alternative<Diagnostic>(failure)) return failure;
    for (const auto& instruction : instructions) {
        context.begin_instruction(instruction, is_guarded_transfer(instruction.name));
        if (auto failure = dispatch_emitter(context, local); std::holds_alternative<Diagnostic>(failure)) return failure;
    }
    if (!is_guarded_transfer(instructions.back().name)) {
        if (auto failure = context.terminate(Jump{std::get<BlockId>(continuation)});
            std::holds_alternative<Diagnostic>(failure)) return failure;
    }
    if (!alternative.empty()) {
        if (auto failure = context.enter_block(when_false);
            std::holds_alternative<Diagnostic>(failure)) return failure;
        for (const auto& instruction : alternative) {
            context.begin_instruction(instruction, is_guarded_transfer(instruction.name));
            if (auto failure = dispatch_emitter(context, local);
                std::holds_alternative<Diagnostic>(failure)) return failure;
        }
        if (!is_guarded_transfer(alternative.back().name)) {
            if (auto failure = context.terminate(Jump{std::get<BlockId>(continuation)});
                std::holds_alternative<Diagnostic>(failure)) return failure;
        }
    }
    return context.enter_block(std::get<BlockId>(continuation));
}
} // namespace

Result<RegisterProgram> emit_ir_with(const DecodedProgram& decoded, FactReader& facts, Limits limits,
                                     std::span<const EmitterBinding> local) {
    if (auto failure = check_local_bindings(local); std::holds_alternative<Diagnostic>(failure)) {
        return std::get<Diagnostic>(std::move(failure));
    }
    limits = min_limits(decoded.limits, limits);
    if (decoded.functions.empty() || !decoded.entry.valid() || decoded.entry.value >= decoded.functions.size()) {
        return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {}, "Decoded program has no entry function");
    }
    if (decoded.code_bytes > limits.code_bytes || decoded.functions.size() > limits.functions ||
        decoded.blocks.size() > limits.blocks) {
        return diagnose(Phase::emit_ir, DiagnosticCode::limit_exceeded, {}, "Decoded program exceeds the phase limits");
    }
    std::size_t instruction_count = 0;
    for (const auto& block : decoded.blocks) {
        // Compare before adding: neither narrowing nor total-count overflow
        // may allow a decoded program to bypass a tightened phase limit.
        if (block.instructions.size() > limits.instructions - instruction_count) {
            return diagnose(Phase::emit_ir, DiagnosticCode::limit_exceeded, {},
                            "Decoded instruction count exceeds the phase limit");
        }
        instruction_count += block.instructions.size();
    }
    // Decoded ownership must already be sound: every block is listed by exactly
    // one function, and each block names its owning function.
    std::vector<bool> owned(decoded.blocks.size());
    for (std::size_t index = 0; index < decoded.functions.size(); ++index) {
        const auto& function = decoded.functions[index];
        const auto owner = FunctionId{static_cast<std::uint32_t>(index)};
        if (function.blocks.empty() || !function.entry.valid() || function.entry.value >= decoded.blocks.size() ||
            decoded.blocks[function.entry.value].function != owner) {
            return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {}, "Decoded function entry is invalid");
        }
        for (const auto block : function.blocks) {
            if (!block.valid() || block.value >= decoded.blocks.size() || owned[block.value] ||
                decoded.blocks[block.value].function != owner) {
                return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {},
                                "Decoded block ownership is invalid");
            }
            owned[block.value] = true;
        }
    }

    RegisterEmitter context(decoded, facts, limits);
    for (std::size_t index = 0; index < decoded.functions.size(); ++index) {
        if (auto failure = context.begin_function(FunctionId{static_cast<std::uint32_t>(index)});
            std::holds_alternative<Diagnostic>(failure)) {
            return std::get<Diagnostic>(std::move(failure));
        }
        for (const auto decoded_block : decoded.functions[index].blocks) {
            if (auto failure = context.begin_block(decoded_block); std::holds_alternative<Diagnostic>(failure)) {
                return std::get<Diagnostic>(std::move(failure));
            }
            const auto& block = decoded.blocks[decoded_block.value];
            const std::span<const DecodedInstruction> instructions = block.instructions;
            for (std::size_t begin = 0; begin < instructions.size();) {
                auto end = begin + 1;
                const bool mergeable = mergeable_guard(instructions[begin]);
                if (mergeable) {
                    end = guard_run_end(instructions, begin);
                }
                auto pair_end = end;
                if (mergeable && end < instructions.size() &&
                    (mergeable_guard(instructions[end]) ||
                     (end + 1 == instructions.size() && instructions[end].predicate.enabled &&
                      is_guarded_transfer(instructions[end].name))) &&
                    instructions[begin].predicate.index == instructions[end].predicate.index &&
                    instructions[begin].predicate.inverted != instructions[end].predicate.inverted) {
                    // Pair one run at a time. Later alternating runs keep their
                    // original order and reevaluate the still-current predicate.
                    pair_end = guard_run_end(instructions, end);
                }
                if (auto failure = emit_instructions(context, instructions.subspan(begin, end - begin),
                                                     instructions.subspan(end, pair_end - end), local);
                    std::holds_alternative<Diagnostic>(failure)) {
                    return std::get<Diagnostic>(std::move(failure));
                }
                begin = pair_end;
            }
            if (context.block_open()) {
                auto decoded_next = decoded_fallthrough(decoded, decoded_block);
                if (const auto* failure = std::get_if<Diagnostic>(&decoded_next)) return *failure;
                const auto fallthrough = std::get<BlockId>(decoded_next);
                if (!fallthrough.valid()) {
                    return context.error(DiagnosticCode::invalid_ir, "Open decoded block has no fallthrough successor");
                }
                auto target = context.label(decoded.blocks[fallthrough.value].begin_pc);
                if (const auto* failure = std::get_if<Diagnostic>(&target)) return *failure;
                // Flow emitters may already enter the decoded successor after
                // wiring the edge. Do not turn that open successor into a loop.
                if (context.block() != std::get<BlockId>(target)) {
                    if (auto failure = context.terminate(Jump{std::get<BlockId>(target)});
                        std::holds_alternative<Diagnostic>(failure)) return std::get<Diagnostic>(std::move(failure));
                }
            }
        }
        if (auto failure = context.finish_function(context.emitted_block(decoded.functions[index].entry));
            std::holds_alternative<Diagnostic>(failure)) {
            return std::get<Diagnostic>(std::move(failure));
        }
    }

    auto program = context.take_program();
    for (const auto& block : program.blocks) {
        if (std::holds_alternative<Unreachable>(block.terminator)) {
            return diagnose(Phase::emit_ir, DiagnosticCode::invalid_ir, block.source,
                            "Decoded block was never given an executable terminator");
        }
    }
    for (std::size_t index = 0; index < decoded.blocks.size(); ++index) {
        if (!context.emitted_block(BlockId{static_cast<std::uint32_t>(index)}).valid()) {
            return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {},
                            "Decoded block was never emitted into the IR");
        }
    }

    // The phase carries the canonical union of inherited and newly consumed facts.
    auto dependencies = merge_dependencies(decoded.dependencies, facts.dependencies(), limits, Phase::emit_ir);
    if (const auto* failure = std::get_if<Diagnostic>(&dependencies)) return *failure;
    program.dependencies = std::get<std::vector<Dependency>>(std::move(dependencies));

    // Postcondition: the emitted register program passes contract validation.
    const auto checked = validate(program, IrForm::registers, Phase::emit_ir);
    if (const auto* failure = std::get_if<Diagnostic>(&checked)) return *failure;
    return RegisterProgram{std::move(program)};
}

// The frozen phase: identical walk, but only the checked global table dispatches.
Result<RegisterProgram> emit_ir(const DecodedProgram& decoded, FactReader& facts, Limits limits) {
    return emit_ir_with(decoded, facts, limits, {});
}
} // namespace armigo::shader::pipeline
