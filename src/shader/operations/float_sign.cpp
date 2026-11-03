// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Sign-bit operations on raw binary32 words. They are exact bit operations,
// not arithmetic: no flush, no NaN canonicalization, and a NaN keeps its
// payload. Consumers that flush or canonicalize do so themselves, so a
// negated or absolute operand feeds an FTZ instruction exactly as the
// instruction's own operand modifier would.
#include "../graphics_operations.h"

namespace armigo::shader::graphics::detail {

std::uint32_t evaluate_float_negate(const OperationWords& operands,
                                    std::uint32_t) noexcept {
    return operands[0] ^ 0x80000000u;
}

std::string emit_float_negate(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " ^ 0x80000000u)";
}

std::uint32_t evaluate_float_absolute(const OperationWords& operands,
                                      std::uint32_t) noexcept {
    return operands[0] & 0x7fffffffu;
}

std::string emit_float_absolute(const OperationText& operands, std::uint32_t) {
    return "(" + operands[0] + " & 0x7fffffffu)";
}

}  // namespace armigo::shader::graphics::detail
