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
    // A multiply's power-of-two scale, applied to source A first.
    std::optional<std::uint32_t> scale;
    // A directed FADD of RZ and one source: exact, so it is that source,
    // flushed under FTZ, with a zero result signed by the mode (true: toward
    // minus keeps -0). The RZ operand is not among the sources.
    std::optional<bool> zero_sum_toward_minus;
};
std::variant<Arithmetic, std::string> decode_arithmetic(Instruction);
bool arithmetic_opcode(unsigned opcode);
} // namespace armigo::shader_native
