// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader_native/input.h>
#include <array>
#include <variant>

namespace armigo::shader_native {
struct Instruction {
    std::array<std::uint32_t, 4> words{};
    std::uint64_t bits(unsigned first, unsigned count) const;
    static Instruction read(Bytes bytes);
};
enum class RegisterFile { general, uniform };
struct Register {
    RegisterFile file{};
    std::uint8_t index{};
    bool is_zero() const { return index == (file == RegisterFile::general ? 255 : 63); }
};
enum class PredicateFile { general, uniform };
struct Predicate {
    PredicateFile file{};
    std::uint8_t index{};
    bool inverted{};
    bool is_constant() const { return index == 7; }
};
struct Immediate { std::uint32_t bits{}; };
struct Constant { std::uint8_t bank{}; std::uint16_t byte_offset{}; };
using Operand = std::variant<Register, Immediate, Constant>;
enum class Operation { move, uniform_move, nop, exit, branch };
struct DecodedInstruction {
    Instruction raw;
    std::uint64_t pc{};
    std::uint16_t opcode{};
    Operation operation{};
    Predicate execution;
    std::optional<Predicate> condition;
    std::optional<Register> destination;
    std::optional<Operand> source;
    std::optional<std::uint64_t> branch_target;
    // Scheduling bits are retained, without imposing another ISA's scheduler.
    std::uint32_t scheduling{};
};
enum class DecodeError { unknown_opcode, unvalidated_opcode, unvalidated_mode, invalid_pc, branch_target };
struct DecodeFailure {
    DecodeError code{};
    std::uint16_t opcode{};
    std::uint64_t pc{};
    const char* message{};
};
using DecodeResult = std::variant<DecodedInstruction, DecodeFailure>;
// PC and extent are instruction-relative bytes; a branch must land on a
// complete instruction inside that same span. No reachability pruning occurs.
DecodeResult decode_instruction(Instruction instruction, std::uint64_t pc, std::uint64_t extent);
} // namespace armigo::shader_native
