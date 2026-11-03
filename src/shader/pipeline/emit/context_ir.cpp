// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Labels are pseudo operations in the decoded stream, and RZ/PT reads fold to
// constants. The same behaviors back
// a block-based program with explicit terminators and checked arenas.

#include "context_ir.h"

#include <algorithm>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
Operand literal(std::uint64_t bits, ValueType type) {
    return Operand{LiteralBits{bits, type}};
}

bool is_pseudo_write(Register reg) {
    return reg == zero_register || reg == true_predicate;
}

bool writable_register(Register reg) {
    switch (reg.file) {
    case RegisterFile::gpr: return reg.index < 255; // R255 is the read-only zero pseudo-register.
    case RegisterFile::predicate: return reg.index < 7; // P7 is the read-only true predicate.
    case RegisterFile::condition_code: return reg.index < 4;
    }
    return false;
}
} // namespace

RegisterEmitter::RegisterEmitter(const DecodedProgram& decoded, FactReader& fact_reader, Limits limits)
    : decoded_(decoded), facts_(fact_reader), limits_(limits), block_map_(decoded.blocks.size()) {
    program_.stage = decoded.stage;
    program_.limits = limits;
    program_.entry = decoded.entry;
    program_.functions.reserve(decoded.functions.size());
}

const DecodedProgram& RegisterEmitter::decoded_program() const { return decoded_; }
const DecodedInstruction& RegisterEmitter::instruction() const { return *current_instruction_; }
FunctionId RegisterEmitter::function() const { return current_function_; }
BlockId RegisterEmitter::block() const { return current_block_; }
BlockId RegisterEmitter::decoded_block() const { return current_decoded_block_; }
FactReader& RegisterEmitter::facts() { return facts_; }
const Limits& RegisterEmitter::limits() const { return limits_; }

Result<Operand> RegisterEmitter::read(Register reg) {
    // Pseudo-register reads resolve to literals; other reads keep the physical
    // register as an operand and preserve its bits. SSA renaming happens later.
    if (reg == zero_register) return literal(0, ValueType::word32);
    if (reg == true_predicate) return literal(1, ValueType::predicate);
    if (!writable_register(reg)) {
        return error(DiagnosticCode::invalid_ir, "Read of an out-of-range register");
    }
    return Operand{reg};
}

Status RegisterEmitter::write(Register reg, Operand value) {
    // Writes to RZ/PT are observable no-ops, never errors.
    if (is_pseudo_write(reg)) return std::monostate{};
    if (!writable_register(reg)) return error(DiagnosticCode::invalid_ir, "Write to an out-of-range register");
    const auto storage = operand_storage(value);
    if (!storage) return error(DiagnosticCode::invalid_ir, "Write reads an unknown operand");
    const auto expected = reg.file == RegisterFile::gpr ? ValueType::word32 : ValueType::predicate;
    if (*storage != expected) return error(DiagnosticCode::invalid_ir, "Write operand storage disagrees with the register file");
    return append_copy(std::move(value), Operand{reg});
}

Result<ValueId> RegisterEmitter::temporary(ValueType type) {
    if (program_.values.size() >= limits_.values) {
        return error(DiagnosticCode::limit_exceeded, "Value limit exceeded");
    }
    const auto id = ValueId{static_cast<std::uint32_t>(program_.values.size())};
    program_.values.push_back(Value{type, ValueKind::temporary});
    return id;
}

Status RegisterEmitter::assign(ValueId local, Operand value) {
    if (!local.valid() || local.value >= program_.values.size()) {
        return error(DiagnosticCode::invalid_ir, "Assignment to an unknown value");
    }
    const auto storage = operand_storage(value);
    if (!storage) return error(DiagnosticCode::invalid_ir, "Assignment reads an unknown operand");
    if (*storage != program_.values[local.value].type) {
        return error(DiagnosticCode::invalid_ir, "Assignment operand storage disagrees with the local");
    }
    return append_copy(std::move(value), Operand{local});
}

bool RegisterEmitter::block_open() const {
    // An Unreachable terminator is the "not yet terminated" construction state.
    return current_block_.valid() &&
           std::holds_alternative<Unreachable>(program_.blocks[current_block_.value].terminator);
}

std::optional<ValueType> RegisterEmitter::operand_storage(const Operand& operand) const {
    if (const auto* id = std::get_if<ValueId>(&operand)) {
        if (!id->valid() || id->value >= program_.values.size()) return std::nullopt;
        return program_.values[id->value].type;
    }
    if (const auto* reg = std::get_if<Register>(&operand)) {
        return reg->file == RegisterFile::gpr ? ValueType::word32 : ValueType::predicate;
    }
    if (const auto* bits = std::get_if<LiteralBits>(&operand)) return bits->type;
    if (std::holds_alternative<ConstantWord>(operand)) return ValueType::word32;
    if (std::holds_alternative<Undefined>(operand)) return std::get<Undefined>(operand).type;
    return std::nullopt;
}

Status RegisterEmitter::append_copy(Operand value, Operand destination) {
    Operation copy;
    copy.source = current_location_;
    copy.opcode = Opcode::copy;
    copy.inputs.push_back(std::move(value));
    copy.outputs.push_back(std::move(destination));
    const auto appended = append(std::move(copy));
    if (const auto* failure = std::get_if<Diagnostic>(&appended)) return *failure;
    return std::monostate{};
}

Result<OperationId> RegisterEmitter::append(Operation operation) {
    const auto* sig = signature(operation.opcode);
    if (!sig) return error(DiagnosticCode::unsupported_instruction, "Unknown IR opcode");
    if (!sig->inputs.accepts(operation.inputs.size()) || !sig->outputs.accepts(operation.outputs.size())) {
        return error(DiagnosticCode::invalid_ir, "Operation signature does not match its operands");
    }
    if (operation.inputs.size() + operation.outputs.size() > limits_.operands_per_operation) {
        return error(DiagnosticCode::limit_exceeded, "Operation operand limit exceeded");
    }
    for (const auto& output : operation.outputs) {
        if (const auto* id = std::get_if<ValueId>(&output);
            id && (!id->valid() || id->value >= program_.values.size())) {
            return error(DiagnosticCode::invalid_ir, "Operation writes an unknown value");
        }
    }
    if (!block_open()) {
        return error(DiagnosticCode::invalid_ir, "Operation appended outside an open block");
    }
    if (program_.operations.size() >= limits_.operations) {
        return error(DiagnosticCode::limit_exceeded, "Operation limit exceeded");
    }
    if (!operation.source.pc) operation.source = current_location_;
    const auto id = OperationId{static_cast<std::uint32_t>(program_.operations.size())};
    program_.operations.push_back(std::move(operation));
    program_.blocks[current_block_.value].operations.push_back(id);
    return id;
}

Result<std::vector<ValueId>> RegisterEmitter::emit(Opcode opcode, NumericMode numeric,
    std::span<const Operand> inputs, std::span<const ValueType> output_types,
    OperationInfo info, Effects effects) {
    const auto* sig = signature(opcode);
    if (!sig) return error(DiagnosticCode::unsupported_instruction, "Unknown IR opcode");
    if (!sig->inputs.accepts(inputs.size()) || !sig->outputs.accepts(output_types.size())) {
        return error(DiagnosticCode::invalid_ir, "Emitter signature does not match the opcode arity");
    }
    if (!valid_numeric_mode(numeric)) {
        return error(DiagnosticCode::unsupported_modifier, "Numeric mode is not well formed");
    }
    if (info.index() != static_cast<std::size_t>(sig->info)) {
        return error(DiagnosticCode::invalid_ir, "Operation metadata kind does not match the opcode");
    }
    std::vector<ValueId> outputs;
    outputs.reserve(output_types.size());
    for (const auto type : output_types) {
        auto value = temporary(type);
        if (const auto* failure = std::get_if<Diagnostic>(&value)) return *failure;
        outputs.push_back(std::get<ValueId>(value));
    }
    Operation operation;
    operation.source = current_location_;
    operation.opcode = opcode;
    operation.numeric = numeric;
    operation.info = std::move(info);
    operation.inputs.assign(inputs.begin(), inputs.end());
    operation.outputs.assign(outputs.begin(), outputs.end());
    // Attach the required effect signature; caller-supplied effects are kept.
    operation.effects = effects | required_effects(operation);
    const auto appended = append(std::move(operation));
    if (const auto* failure = std::get_if<Diagnostic>(&appended)) return *failure;
    return outputs;
}

Result<Operand> RegisterEmitter::unary(Opcode opcode, Operand input, NumericMode numeric, ValueType output) {
    const std::array<Operand, 1> inputs{std::move(input)};
    const std::array<ValueType, 1> outputs{output};
    auto emitted = emit(opcode, numeric, inputs, outputs, {}, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<Operand> RegisterEmitter::binary(Opcode opcode, Operand left, Operand right,
                                        NumericMode numeric, ValueType output) {
    const std::array<Operand, 2> inputs{std::move(left), std::move(right)};
    const std::array<ValueType, 1> outputs{output};
    auto emitted = emit(opcode, numeric, inputs, outputs, {}, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<Operand> RegisterEmitter::ternary(Opcode opcode, Operand first, Operand second, Operand third,
                                         NumericMode numeric, ValueType output) {
    const std::array<Operand, 3> inputs{std::move(first), std::move(second), std::move(third)};
    const std::array<ValueType, 1> outputs{output};
    auto emitted = emit(opcode, numeric, inputs, outputs, {}, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<Operand> RegisterEmitter::convert(Operand input, ConversionInfo conversion,
                                         NumericMode numeric, ValueType output) {
    const std::array<Operand, 1> inputs{std::move(input)};
    const std::array<ValueType, 1> outputs{output};
    auto emitted = emit(Opcode::convert, numeric, inputs, outputs, conversion, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<Operand> RegisterEmitter::compare(Operand left, Operand right, ComparisonInfo relation,
                                         NumericMode numeric) {
    const std::array<Operand, 2> inputs{std::move(left), std::move(right)};
    const std::array<ValueType, 1> outputs{ValueType::predicate};
    auto emitted = emit(Opcode::compare, numeric, inputs, outputs, relation, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<Operand> RegisterEmitter::select(Operand condition, Operand when_true, Operand when_false,
                                        ValueType type) {
    const std::array<Operand, 3> inputs{std::move(condition), std::move(when_true), std::move(when_false)};
    const std::array<ValueType, 1> outputs{type};
    auto emitted = emit(Opcode::select, NumericMode{}, inputs, outputs, {}, Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return Operand{std::get<std::vector<ValueId>>(emitted).front()};
}

Result<std::vector<ValueId>> RegisterEmitter::load(MemoryInfo memory, std::span<const Operand> inputs,
                                                   NumericMode numeric, std::span<const ValueType> outputs) {
    return emit(Opcode::load, numeric, inputs, outputs, std::move(memory), Effects::none);
}

Status RegisterEmitter::store(MemoryInfo memory, std::span<const Operand> inputs, NumericMode numeric) {
    const std::span<const ValueType> no_outputs;
    auto emitted = emit(Opcode::store, numeric, inputs, no_outputs, std::move(memory), Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return std::monostate{};
}

Result<std::vector<ValueId>> RegisterEmitter::texture(Opcode opcode, TextureInfo info,
    std::span<const Operand> inputs, NumericMode numeric, std::span<const ValueType> outputs) {
    return emit(opcode, numeric, inputs, outputs, std::move(info), Effects::none);
}

Result<std::vector<ValueId>> RegisterEmitter::load_io(IoInfo io, std::span<const Operand> inputs,
                                                      NumericMode numeric, std::span<const ValueType> outputs) {
    return emit(Opcode::load_io, numeric, inputs, outputs, std::move(io), Effects::none);
}

Status RegisterEmitter::store_io(IoInfo io, std::span<const Operand> inputs, NumericMode numeric) {
    const std::span<const ValueType> no_outputs;
    auto emitted = emit(Opcode::store_io, numeric, inputs, no_outputs, std::move(io), Effects::none);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return std::monostate{};
}

Status RegisterEmitter::discard() {
    const std::span<const Operand> no_inputs;
    const std::span<const ValueType> no_outputs;
    auto emitted = emit(Opcode::discard, NumericMode{}, no_inputs, no_outputs, {}, Effects::discard);
    if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
    return std::monostate{};
}

Result<BlockId> RegisterEmitter::label(Pc pc) {
    if (const auto found = labels_.find(pc); found != labels_.end()) return found->second;
    auto created = create_block();
    if (const auto* failure = std::get_if<Diagnostic>(&created)) return *failure;
    labels_.emplace(pc, std::get<BlockId>(created));
    return std::get<BlockId>(created);
}

Result<BlockId> RegisterEmitter::create_block() {
    if (!current_function_.valid()) return error(DiagnosticCode::invalid_ir, "Block created outside a function");
    if (program_.blocks.size() >= limits_.blocks) return error(DiagnosticCode::limit_exceeded, "Block limit exceeded");
    BasicBlock created;
    created.function = current_function_;
    created.source = current_location_;
    const auto id = BlockId{static_cast<std::uint32_t>(program_.blocks.size())};
    program_.blocks.push_back(std::move(created));
    // Function block lists are the validator's ownership record; guard and
    // continuation blocks created mid-expansion are registered here too.
    program_.functions[current_function_.value].blocks.push_back(id);
    return id;
}

Status RegisterEmitter::enter_block(BlockId block) {
    if (!block.valid() || block.value >= program_.blocks.size()) {
        return error(DiagnosticCode::invalid_ir, "Enter of an unknown block");
    }
    if (program_.blocks[block.value].function != current_function_) {
        return error(DiagnosticCode::invalid_ir, "Block belongs to another function");
    }
    current_block_ = block;
    return std::monostate{};
}

Status RegisterEmitter::terminate(Terminator terminator) {
    if (!current_block_.valid()) return error(DiagnosticCode::invalid_ir, "Terminator installed outside a block");
    auto& block = program_.blocks[current_block_.value];
    if (!std::holds_alternative<Unreachable>(block.terminator)) {
        return error(DiagnosticCode::invalid_ir, "Block already has a terminator");
    }
    const auto targets = successors(terminator);
    for (const auto target : targets) {
        if (!target.valid() || target.value >= program_.blocks.size()) {
            return error(DiagnosticCode::invalid_ir, "Terminator targets an unknown block");
        }
        if (program_.blocks[target.value].function != current_function_) {
            return error(DiagnosticCode::invalid_ir, "Terminator crosses a function boundary");
        }
    }
    if (edge_count_ + targets.size() > limits_.edges) {
        return error(DiagnosticCode::limit_exceeded, "CFG edge limit exceeded");
    }
    edge_count_ += targets.size();
    for (const auto target : targets) {
        auto& predecessors = program_.blocks[target.value].predecessors;
        if (std::find(predecessors.begin(), predecessors.end(), current_block_) == predecessors.end()) {
            predecessors.push_back(current_block_);
        }
    }
    block.terminator = std::move(terminator);
    return std::monostate{};
}

Result<ValueId> RegisterEmitter::reconvergence_selector(BlockId consumer, Pc pop_pc) {
    // One stable selector value per (consumer block, pop pc), keyed the same way
    // as the decoded reconvergence targets. The flow lane defines it; interning
    // here keeps every consumer seeing the same compiler local.
    const auto key = (static_cast<std::uint64_t>(consumer.value) << 32) | pop_pc;
    if (const auto found = selectors_.find(key); found != selectors_.end()) return found->second;
    auto value = temporary(ValueType::word32);
    if (const auto* failure = std::get_if<Diagnostic>(&value)) return *failure;
    selectors_.emplace(key, std::get<ValueId>(value));
    return value;
}

Diagnostic RegisterEmitter::error(DiagnosticCode code, std::string message, std::string modifier) const {
    Diagnostic diagnostic;
    diagnostic.phase = Phase::emit_ir;
    diagnostic.code = code;
    diagnostic.pc = current_location_.pc;
    diagnostic.instruction = current_location_.instruction;
    diagnostic.modifier = std::move(modifier);
    diagnostic.message = std::move(message);
    return diagnostic;
}

Status RegisterEmitter::begin_function(FunctionId decoded_function) {
    if (!decoded_function.valid() || decoded_function.value >= decoded_.functions.size()) {
        return error(DiagnosticCode::malformed_input, "Decoded function is out of range");
    }
    if (program_.functions.size() >= limits_.functions) {
        return error(DiagnosticCode::limit_exceeded, "Function limit exceeded");
    }
    const auto& decoded = decoded_.functions[decoded_function.value];
    Function function;
    function.entry_pc = decoded.entry_pc;
    const auto id = FunctionId{static_cast<std::uint32_t>(program_.functions.size())};
    program_.functions.push_back(std::move(function));
    current_function_ = id;
    current_block_ = BlockId{};
    current_decoded_block_ = BlockId{};
    labels_.clear();
    selectors_.clear();
    return std::monostate{};
}

Status RegisterEmitter::begin_block(BlockId decoded_block) {
    if (!decoded_block.valid() || decoded_block.value >= decoded_.blocks.size()) {
        return error(DiagnosticCode::malformed_input, "Decoded block is out of range");
    }
    const auto& decoded = decoded_.blocks[decoded_block.value];
    current_location_ = SourceLocation{decoded.begin_pc, {}};
    BlockId block;
    if (const auto found = labels_.find(decoded.begin_pc); found != labels_.end()) {
        block = found->second;
        if (program_.blocks[block.value].function != current_function_) {
            return error(DiagnosticCode::malformed_input, "Decoded block start was labelled by another function");
        }
    } else {
        auto created = create_block();
        if (const auto* failure = std::get_if<Diagnostic>(&created)) return *failure;
        block = std::get<BlockId>(created);
        labels_.emplace(decoded.begin_pc, block);
    }
    current_block_ = block;
    block_map_[decoded_block.value] = block;
    current_decoded_block_ = decoded_block;
    return std::monostate{};
}

void RegisterEmitter::begin_instruction(const DecodedInstruction& decoded_instruction, bool guard_satisfied) {
    current_instruction_ = &decoded_instruction;
    if (guard_satisfied) {
        guarded_instruction_ = decoded_instruction;
        guarded_instruction_->predicate = {7, false, true};
        current_instruction_ = &*guarded_instruction_;
    } else {
        guarded_instruction_.reset();
    }
    current_location_ = SourceLocation{decoded_instruction.pc, decoded_instruction.name};
}

Status RegisterEmitter::finish_function(BlockId entry_block) {
    if (!entry_block.valid() || entry_block.value >= program_.blocks.size() ||
        program_.blocks[entry_block.value].function != current_function_) {
        return error(DiagnosticCode::malformed_input, "Function entry block is missing or foreign");
    }
    // A decoded entry can also be a loop header. Give its initial invocation
    // an ordinary edge so SSA can distinguish it from loop-carried definitions.
    if (!program_.blocks[entry_block.value].predecessors.empty()) {
        current_location_ = program_.blocks[entry_block.value].source;
        auto preheader = create_block();
        if (const auto* failure = std::get_if<Diagnostic>(&preheader)) return *failure;
        const auto entry = std::get<BlockId>(preheader);
        if (auto failure = enter_block(entry); std::holds_alternative<Diagnostic>(failure)) return failure;
        if (auto failure = terminate(Jump{entry_block}); std::holds_alternative<Diagnostic>(failure)) return failure;
        entry_block = entry;
    }
    program_.functions[current_function_.value].entry = entry_block;
    return std::monostate{};
}

BlockId RegisterEmitter::emitted_block(BlockId decoded_block) const {
    return decoded_block.valid() && decoded_block.value < block_map_.size() ? block_map_[decoded_block.value]
                                                                            : BlockId{};
}
} // namespace armigo::shader::pipeline
