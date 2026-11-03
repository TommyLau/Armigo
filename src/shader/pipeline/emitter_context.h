// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "decoder.h"
#include "ir.h"

#include <span>

namespace armigo::shader::pipeline {
// Lane 4 implements a concrete context behind emit_ir. Family callbacks consume
// this interface; they do not mutate Program arenas or apply predication themselves.
class EmitterContext {
public:
    virtual ~EmitterContext() = default;
    [[nodiscard]] virtual const DecodedProgram& decoded_program() const = 0;
    [[nodiscard]] virtual const DecodedInstruction& instruction() const = 0;
    [[nodiscard]] virtual FunctionId function() const = 0;
    [[nodiscard]] virtual BlockId block() const = 0;
    [[nodiscard]] virtual FactReader& facts() = 0;
    [[nodiscard]] virtual const Limits& limits() const = 0;

    // Reads preserve bits. RZ/PT become literals; writes to RZ/PT are discarded.
    [[nodiscard]] virtual Result<Operand> read(Register) = 0;
    [[nodiscard]] virtual Status write(Register, Operand) = 0;
    [[nodiscard]] virtual Result<ValueId> temporary(ValueType) = 0;
    // Pre-SSA compiler locals may have multiple definitions. SSA renames these too.
    [[nodiscard]] virtual Status assign(ValueId, Operand) = 0;
    [[nodiscard]] virtual Result<OperationId> append(Operation) = 0;
    // Creates typed temporaries for all results, stamps source, validates signature,
    // and attaches required effects. Additional effects are never stripped.
    [[nodiscard]] virtual Result<std::vector<ValueId>> emit(
        Opcode, NumericMode, std::span<const Operand> inputs,
        std::span<const ValueType> outputs, OperationInfo, Effects) = 0;
    [[nodiscard]] virtual Result<Operand> unary(Opcode, Operand, NumericMode, ValueType) = 0;
    [[nodiscard]] virtual Result<Operand> binary(Opcode, Operand, Operand, NumericMode, ValueType) = 0;
    [[nodiscard]] virtual Result<Operand> ternary(Opcode, Operand, Operand, Operand, NumericMode, ValueType) = 0;
    [[nodiscard]] virtual Result<Operand> convert(Operand, ConversionInfo, NumericMode, ValueType) = 0;
    [[nodiscard]] virtual Result<Operand> compare(Operand, Operand, ComparisonInfo, NumericMode) = 0;
    [[nodiscard]] virtual Result<Operand> select(Operand condition, Operand when_true,
                                                Operand when_false, ValueType) = 0;
    [[nodiscard]] virtual Result<std::vector<ValueId>> load(MemoryInfo, std::span<const Operand>,
                                                            NumericMode, std::span<const ValueType>) = 0;
    [[nodiscard]] virtual Status store(MemoryInfo, std::span<const Operand>, NumericMode) = 0;
    [[nodiscard]] virtual Result<std::vector<ValueId>> texture(Opcode, TextureInfo,
        std::span<const Operand>, NumericMode, std::span<const ValueType>) = 0;
    [[nodiscard]] virtual Result<std::vector<ValueId>> load_io(IoInfo,
        std::span<const Operand>, NumericMode, std::span<const ValueType>) = 0;
    [[nodiscard]] virtual Status store_io(IoInfo, std::span<const Operand>, NumericMode) = 0;
    [[nodiscard]] virtual Status discard() = 0;

    // Flow lane alone uses these. Target mapping is explicit, not PC arithmetic.
    [[nodiscard]] virtual Result<BlockId> label(Pc) = 0;
    [[nodiscard]] virtual Result<BlockId> create_block() = 0;
    [[nodiscard]] virtual Status enter_block(BlockId) = 0;
    [[nodiscard]] virtual Status terminate(Terminator) = 0;
    [[nodiscard]] virtual Result<ValueId> reconvergence_selector(BlockId consumer, Pc pop_pc) = 0;
    [[nodiscard]] virtual Diagnostic error(DiagnosticCode, std::string message,
                                           std::string modifier = {}) const = 0;
    // Decoded identity is stable when predication creates extra emitted blocks.
    [[nodiscard]] virtual BlockId decoded_block() const {
        const auto& decoded = decoded_program();
        for (std::size_t index = 0; index < decoded.blocks.size(); ++index) {
            const auto& candidate = decoded.blocks[index];
            if (candidate.function == function() && candidate.begin_pc <= instruction().pc &&
                instruction().pc < candidate.end_pc) return BlockId{static_cast<std::uint32_t>(index)};
        }
        return {};
    }
};
using InstructionEmitter = Status (*)(EmitterContext&);
static_assert(std::is_abstract_v<EmitterContext>);
} // namespace armigo::shader::pipeline
