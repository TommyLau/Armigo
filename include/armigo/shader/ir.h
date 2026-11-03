// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Typed scalar shader intermediate representation.
//
// The IR describes a straight-line, per-invocation compute program in static
// single assignment form. Every value-producing instruction defines exactly one
// value whose identifier is the index of that instruction in Program::body.
// Operands may only refer to strictly earlier value-producing instructions.
//
// This header has no platform or framework dependencies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace armigo::shader {

// Scalar value types. Every type is 32 bits wide; values are carried as raw
// 32-bit words outside the IR (for example by the reference evaluator).
enum class ScalarType : std::uint8_t {
    f32 = 0,  // IEEE-754 binary32.
    u32 = 1,  // Unsigned 32-bit integer.
};

enum class Opcode : std::uint8_t {
    // Defines a value of `type` whose bit pattern is `immediate`.
    constant = 0,
    // Defines a value of `type` read from parameter slot `immediate`. Parameters
    // are uniform across all invocations of a dispatch.
    load_parameter = 1,
    // Defines a value of `type` read from input binding `immediate` at the
    // current invocation index.
    load_input = 2,
    // Defines the f32 sum `operand0 + operand1`.
    fadd = 3,
    // Defines the f32 product `operand0 * operand1`.
    fmul = 4,
    // Stores `operand0` into output binding `immediate` at the current
    // invocation index. Defines no value.
    store_output = 5,
};

// Identifier of an SSA value: the index of its defining instruction.
struct ValueId {
    std::uint32_t index = none().index;

    static constexpr ValueId none() noexcept {
        return ValueId{std::numeric_limits<std::uint32_t>::max()};
    }
    constexpr bool is_none() const noexcept { return index == none().index; }
    friend constexpr bool operator==(ValueId, ValueId) noexcept = default;
};

// One IR instruction. Fields that an opcode does not use must hold their
// canonical empty value (ValueId::none() for operands, zero for immediate);
// validation rejects anything else so that every program has one encoding.
//
//   opcode          type            operand0  operand1  immediate
//   constant        result type     none      none      value bits
//   load_parameter  result type     none      none      parameter slot
//   load_input      result type     none      none      input binding
//   fadd, fmul      f32             lhs       rhs       0
//   store_output    stored type     value     none      output binding
struct Instruction {
    Opcode opcode = Opcode::constant;
    ScalarType type = ScalarType::f32;
    ValueId operand0 = ValueId::none();
    ValueId operand1 = ValueId::none();
    std::uint32_t immediate = 0;

    friend bool operator==(const Instruction&, const Instruction&) = default;
};

// A complete compute program.
//
// Each declared output must be stored exactly once. Input and output bindings
// are distinct, non-aliasing arrays indexed by the invocation index.
struct Program {
    // Logical identifier; the emitter supplies the actual backend symbol.
    std::string entry_point;
    std::vector<ScalarType> parameters;
    std::vector<ScalarType> inputs;
    std::vector<ScalarType> outputs;
    std::vector<Instruction> body;

    friend bool operator==(const Program&, const Program&) = default;
};

// Structural limits enforced by validation. They keep every valid program
// within the buffer argument table of the Metal backend (31 slots: one
// parameter block, inputs, outputs, and the element count).
inline constexpr std::size_t kMaxParameters = 256;
inline constexpr std::size_t kMaxIoBindings = 29;
inline constexpr std::size_t kMaxInstructions = 65536;
inline constexpr std::size_t kMaxEntryPointLength = 64;

const char* to_string(ScalarType type) noexcept;
const char* to_string(Opcode opcode) noexcept;

// Convenience builder for constructing programs in tests and tools. It performs
// no validation; call validate() on the finished program.
class ProgramBuilder {
public:
    explicit ProgramBuilder(std::string entry_point);

    std::uint32_t add_parameter(ScalarType type);
    std::uint32_t add_input(ScalarType type);
    std::uint32_t add_output(ScalarType type);

    ValueId constant_f32(float value);
    ValueId constant_u32(std::uint32_t value);
    ValueId load_parameter(ScalarType type, std::uint32_t slot);
    ValueId load_input(ScalarType type, std::uint32_t binding);
    ValueId fadd(ValueId lhs, ValueId rhs);
    ValueId fmul(ValueId lhs, ValueId rhs);
    void store_output(ScalarType type, std::uint32_t binding, ValueId value);

    const Program& program() const noexcept { return program_; }
    Program take() { return std::move(program_); }

private:
    ValueId append(const Instruction& instruction);

    Program program_;
};

}  // namespace armigo::shader
