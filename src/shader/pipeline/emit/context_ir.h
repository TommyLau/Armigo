// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// RZ/PT reads resolve to constants and writes to them are dropped when
// building register operands; read/write methods enforce that behavior.
// Block-based IR with explicit terminators replaces the flat operation list
// and label pseudo-operations.

#pragma once
#include "pipeline/emitter_context.h"

#include <optional>
#include <unordered_map>
#include <vector>

namespace armigo::shader::pipeline {
// Concrete EmitterContext behind emit_ir. It appends into one Program arena in
// register form, stamps every operation with the current guest instruction,
// checks limits at each allocation, and keeps block predecessors in step with
// installed terminators. Family callbacks reach it through the frozen
// EmitterContext interface; only the emit_ir driver uses the begin_* methods.
class RegisterEmitter final : public EmitterContext {
public:
    RegisterEmitter(const DecodedProgram& decoded, FactReader& facts, Limits limits);

    [[nodiscard]] const DecodedProgram& decoded_program() const override;
    [[nodiscard]] const DecodedInstruction& instruction() const override;
    [[nodiscard]] FunctionId function() const override;
    [[nodiscard]] BlockId block() const override;
    [[nodiscard]] FactReader& facts() override;
    [[nodiscard]] const Limits& limits() const override;

    [[nodiscard]] Result<Operand> read(Register reg) override;
    Status write(Register reg, Operand value) override;
    [[nodiscard]] Result<ValueId> temporary(ValueType type) override;
    Status assign(ValueId local, Operand value) override;
    [[nodiscard]] Result<OperationId> append(Operation operation) override;
    [[nodiscard]] Result<std::vector<ValueId>> emit(Opcode opcode, NumericMode numeric,
        std::span<const Operand> inputs, std::span<const ValueType> output_types,
        OperationInfo info, Effects effects) override;
    [[nodiscard]] Result<Operand> unary(Opcode opcode, Operand input, NumericMode numeric,
                                        ValueType output) override;
    [[nodiscard]] Result<Operand> binary(Opcode opcode, Operand left, Operand right,
                                         NumericMode numeric, ValueType output) override;
    [[nodiscard]] Result<Operand> ternary(Opcode opcode, Operand first, Operand second,
                                          Operand third, NumericMode numeric, ValueType output) override;
    [[nodiscard]] Result<Operand> convert(Operand input, ConversionInfo conversion,
                                          NumericMode numeric, ValueType output) override;
    [[nodiscard]] Result<Operand> compare(Operand left, Operand right, ComparisonInfo relation,
                                          NumericMode numeric) override;
    [[nodiscard]] Result<Operand> select(Operand condition, Operand when_true,
                                         Operand when_false, ValueType type) override;
    [[nodiscard]] Result<std::vector<ValueId>> load(MemoryInfo memory, std::span<const Operand> inputs,
                                                    NumericMode numeric, std::span<const ValueType> outputs) override;
    Status store(MemoryInfo memory, std::span<const Operand> inputs, NumericMode numeric) override;
    [[nodiscard]] Result<std::vector<ValueId>> texture(Opcode opcode, TextureInfo info,
        std::span<const Operand> inputs, NumericMode numeric, std::span<const ValueType> outputs) override;
    [[nodiscard]] Result<std::vector<ValueId>> load_io(IoInfo io, std::span<const Operand> inputs,
                                                       NumericMode numeric, std::span<const ValueType> outputs) override;
    Status store_io(IoInfo io, std::span<const Operand> inputs, NumericMode numeric) override;
    Status discard() override;

    [[nodiscard]] Result<BlockId> label(Pc pc) override;
    [[nodiscard]] Result<BlockId> create_block() override;
    Status enter_block(BlockId block) override;
    Status terminate(Terminator terminator) override;
    [[nodiscard]] Result<ValueId> reconvergence_selector(BlockId consumer, Pc pop_pc) override;
    [[nodiscard]] Diagnostic error(DiagnosticCode code, std::string message,
                                   std::string modifier = {}) const override;

    [[nodiscard]] BlockId decoded_block() const override;

    // Driver-only view. decoded_function indexes the decoded function arena.
    Status begin_function(FunctionId decoded_function);
    Status begin_block(BlockId decoded_block);
    // Installs the entry block once every block of the function has been emitted.
    Status finish_function(BlockId entry_block);
    // A satisfied outer guard replaces only the execution predicate with PT;
    // flow condition codes, raw encoding and decoded token identity are retained.
    void begin_instruction(const DecodedInstruction& decoded_instruction, bool guard_satisfied = false);
    // Decoded-to-IR block mapping, valid once begin_block has run for the block.
    [[nodiscard]] BlockId emitted_block(BlockId decoded_block) const;
    [[nodiscard]] bool block_open() const;
    [[nodiscard]] Program take_program() { return std::move(program_); }

private:
    [[nodiscard]] std::optional<ValueType> operand_storage(const Operand& operand) const;
    Status append_copy(Operand value, Operand destination);

    const DecodedProgram& decoded_;
    FactReader& facts_;
    Limits limits_;
    Program program_;
    FunctionId current_function_;
    BlockId current_block_;
    const DecodedInstruction* current_instruction_{};
    std::optional<DecodedInstruction> guarded_instruction_;
    SourceLocation current_location_; // Block begin pc, or the current instruction.
    std::unordered_map<Pc, BlockId> labels_;
    std::unordered_map<std::uint64_t, ValueId> selectors_;
    std::vector<BlockId> block_map_; // Decoded block arena index to emitted block.
    std::uint64_t edge_count_{};
    BlockId current_decoded_block_;
};
} // namespace armigo::shader::pipeline
