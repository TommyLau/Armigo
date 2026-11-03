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
#include <cstdint>
namespace armigo::shader::maxwell {

// FMUL32I: floating-point multiply of a register by a 32-bit immediate.
// Rounding is always nearest-even; the flush mode is carried so a forged
// payload outside the admitted profile stays detectable at lowering.
struct FloatMultiplyImmediate32 {
    RegisterOperand destination;
    RegisterOperand source_a;
    std::uint32_t immediate_bits = 0;
    FlushMode flush = FlushMode::FlushToZero;
    bool saturate = false;
};

// FADD32I: floating-point add of a register and a 32-bit immediate. Rounding
// is always nearest-even and flush-to-zero is implied by the family; the
// sign and absolute modifiers are carried verbatim. Negate-B and absolute-B
// apply to the immediate itself.
struct FloatAddImmediate32 {
    RegisterOperand destination;
    RegisterOperand source_a;
    std::uint32_t immediate_bits = 0;
    bool negate_a = false;
    bool absolute_a = false;
    bool negate_b = false;
    bool absolute_b = false;
};
}  // namespace armigo::shader::maxwell
