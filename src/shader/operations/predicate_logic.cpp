// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Two-input boolean combinations of predicates, the combining step of the
// Maxwell set-predicate instructions.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {

std::uint32_t evaluate_predicate_and(const OperationWords& operands,
                                     std::uint32_t) noexcept {
    return (operands[0] != 0 && operands[1] != 0) ? 1u : 0u;
}

std::string emit_predicate_and(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " && " + operands[1] + ")";
}

std::uint32_t evaluate_predicate_or(const OperationWords& operands,
                                    std::uint32_t) noexcept {
    return (operands[0] != 0 || operands[1] != 0) ? 1u : 0u;
}

std::string emit_predicate_or(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " || " + operands[1] + ")";
}

std::uint32_t evaluate_predicate_xor(const OperationWords& operands,
                                     std::uint32_t) noexcept {
    return ((operands[0] != 0) != (operands[1] != 0)) ? 1u : 0u;
}

std::string emit_predicate_xor(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " != " + operands[1] + ")";
}

}  // namespace armigo::shader::graphics::detail
