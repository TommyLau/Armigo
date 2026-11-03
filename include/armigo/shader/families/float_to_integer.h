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
#include <variant>
namespace armigo::shader::maxwell {
// F2I reads its source either from a register (the F2I_reg form) or from one
// constant-buffer word (the F2I_cbuf form).
using FloatToIntegerSource = std::variant<RegisterOperand, ConstantBufferOperand>;
// Float-to-integer conversion in the admitted profile: binary32 source under
// flush-to-zero with no absolute, negate, half, or condition-code modifier.
struct FloatToInteger {
    RegisterOperand destination;
    FloatToIntegerSource source;
    RoundingMode rounding = RoundingMode::Zero;
    FlushMode flush = FlushMode::FlushToZero;
    bool is_signed = false;
    // 16 or 32; narrower results are zero-extended into the register.
    std::uint8_t destination_width = 32;
};
}  // namespace armigo::shader::maxwell
