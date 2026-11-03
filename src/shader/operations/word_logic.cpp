// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bitwise logic on raw words and the zero test. WordLogic's immediate holds
// the operation in bits 0..1 (AND, OR, XOR, PASS_B) and the inversion of A
// (bit 2) and B (bit 3), applied before the operation.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {

bool admits_word_logic(std::uint32_t immediate) noexcept {
    return immediate < 16;
}

std::uint32_t evaluate_word_logic(const OperationWords& operands,
                                  std::uint32_t immediate) noexcept {
    const std::uint32_t a = (immediate & 4) != 0 ? ~operands[0] : operands[0];
    const std::uint32_t b = (immediate & 8) != 0 ? ~operands[1] : operands[1];
    switch (immediate & 3) {
        case 0: return a & b;
        case 1: return a | b;
        case 2: return a ^ b;
        default: return b;
    }
}

std::string emit_word_logic(const OperationText& operands, std::uint32_t immediate) {
    const std::string a = (immediate & 4) != 0 ? "(~" + operands[0] + ")" : operands[0];
    const std::string b = (immediate & 8) != 0 ? "(~" + operands[1] + ")" : operands[1];
    switch (immediate & 3) {
        case 0: return "(" + a + " & " + b + ")";
        case 1: return "(" + a + " | " + b + ")";
        case 2: return "(" + a + " ^ " + b + ")";
        default: return b;
    }
}

std::uint32_t evaluate_word_is_zero(const OperationWords& operands,
                                    std::uint32_t) noexcept {
    return operands[0] == 0 ? 1u : 0u;
}

std::string emit_word_is_zero(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " == 0u)";
}

}  // namespace armigo::shader::graphics::detail
