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
#include <armigo/shader/maxwell_operands.h>

#include <variant>
namespace armigo::shader::maxwell {
// One FMNMX instruction. The B source is a
// register, a constant-buffer word, or a 20-bit float immediate depending on
// the decoded opcode window. The operand modifiers apply before the
// comparison, absolute first and negate second. The selecting predicate is
// admitted as PT only; negate_predicate then picks the operation (false is
// minimum, true is maximum).
struct FloatMinMax {
    RegisterOperand destination;
    RegisterOperand source_a;
    std::variant<RegisterOperand, ConstantBufferOperand, ShortFloatImmediate>
        source_b;
    bool absolute_a = false;
    bool negate_a = false;
    bool absolute_b = false;
    bool negate_b = false;
    bool negate_predicate = false;
    FlushMode flush = FlushMode::FlushToZero;
};
}  // namespace armigo::shader::maxwell
