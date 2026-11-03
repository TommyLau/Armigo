// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/ir.h>

#include <bit>

namespace armigo::shader {

const char* to_string(ScalarType type) noexcept {
    switch (type) {
    case ScalarType::f32: return "f32";
    case ScalarType::u32: return "u32";
    }
    return "<unknown type>";
}

const char* to_string(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::constant: return "constant";
    case Opcode::load_parameter: return "load_parameter";
    case Opcode::load_input: return "load_input";
    case Opcode::fadd: return "fadd";
    case Opcode::fmul: return "fmul";
    case Opcode::store_output: return "store_output";
    }
    return "<unknown opcode>";
}

ProgramBuilder::ProgramBuilder(std::string entry_point) {
    program_.entry_point = std::move(entry_point);
}

std::uint32_t ProgramBuilder::add_parameter(ScalarType type) {
    program_.parameters.push_back(type);
    return static_cast<std::uint32_t>(program_.parameters.size() - 1);
}

std::uint32_t ProgramBuilder::add_input(ScalarType type) {
    program_.inputs.push_back(type);
    return static_cast<std::uint32_t>(program_.inputs.size() - 1);
}

std::uint32_t ProgramBuilder::add_output(ScalarType type) {
    program_.outputs.push_back(type);
    return static_cast<std::uint32_t>(program_.outputs.size() - 1);
}

ValueId ProgramBuilder::constant_f32(float value) {
    return append({Opcode::constant, ScalarType::f32, ValueId::none(),
                   ValueId::none(), std::bit_cast<std::uint32_t>(value)});
}

ValueId ProgramBuilder::constant_u32(std::uint32_t value) {
    return append({Opcode::constant, ScalarType::u32, ValueId::none(),
                   ValueId::none(), value});
}

ValueId ProgramBuilder::load_parameter(ScalarType type, std::uint32_t slot) {
    return append(
        {Opcode::load_parameter, type, ValueId::none(), ValueId::none(), slot});
}

ValueId ProgramBuilder::load_input(ScalarType type, std::uint32_t binding) {
    return append(
        {Opcode::load_input, type, ValueId::none(), ValueId::none(), binding});
}

ValueId ProgramBuilder::fadd(ValueId lhs, ValueId rhs) {
    return append({Opcode::fadd, ScalarType::f32, lhs, rhs, 0});
}

ValueId ProgramBuilder::fmul(ValueId lhs, ValueId rhs) {
    return append({Opcode::fmul, ScalarType::f32, lhs, rhs, 0});
}

void ProgramBuilder::store_output(ScalarType type, std::uint32_t binding,
                                  ValueId value) {
    append({Opcode::store_output, type, value, ValueId::none(), binding});
}

ValueId ProgramBuilder::append(const Instruction& instruction) {
    program_.body.push_back(instruction);
    return ValueId{static_cast<std::uint32_t>(program_.body.size() - 1)};
}

}  // namespace armigo::shader
