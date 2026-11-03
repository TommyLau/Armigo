// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of SEL: a register-width selection between source A and source B
// by an instruction-encoded predicate. Raw words are selected; nothing is
// interpreted numerically.
#pragma once
#include <armigo/shader/maxwell_operands.h>
#include <cstdint>
#include <variant>
namespace armigo::shader::maxwell {
// The integer immediate is the 20-bit signed value Armigo decodes from bits
// 20..38 and the sign at bit 56, already sign-extended to 32 bits.
using SelectSource = std::variant<RegisterOperand, ConstantBufferOperand, std::uint32_t>;
struct SelectByPredicate {
    RegisterOperand destination;
    RegisterOperand source_a;
    SelectSource source_b;
    // Predicate register index 0..7 (7 is PT) and its inversion.
    std::uint8_t predicate = 7;
    bool negate_predicate = false;
};
}  // namespace armigo::shader::maxwell
