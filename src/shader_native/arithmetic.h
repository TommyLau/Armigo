// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/instruction.h>

namespace armigo::shader_native {
struct ArithmeticSource { Operand value; bool absolute{}, negate{}; };
struct Arithmetic {
    ir::Opcode operation{};
    ir::NumericMode numeric;
    std::uint8_t destination{};
    std::vector<ArithmeticSource> sources;
    bool integer_multiply_add{};
};
std::variant<Arithmetic, std::string> decode_arithmetic(Instruction);
bool arithmetic_opcode(unsigned opcode);
} // namespace armigo::shader_native
