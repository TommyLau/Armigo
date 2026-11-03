// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Predicate-controlled selection of a raw word or of a predicate. Both
// candidates are already evaluated values; selection never evaluates or
// discards work, so it is the building block for predicated writes whose
// guard is not known at translation time.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {

std::uint32_t evaluate_select(const OperationWords& operands,
                              std::uint32_t) noexcept {
    return operands[0] != 0 ? operands[1] : operands[2];
}

std::string emit_select(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " ? " + operands[1] + " : " + operands[2] + ")";
}

std::uint32_t evaluate_predicate_select(const OperationWords& operands,
                                        std::uint32_t) noexcept {
    return operands[0] != 0 ? operands[1] : operands[2];
}

std::string emit_predicate_select(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " ? " + operands[1] + " : " + operands[2] + ")";
}

}  // namespace armigo::shader::graphics::detail
