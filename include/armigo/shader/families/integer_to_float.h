// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of I2F: integer to binary32 conversion of a register or
// constant-buffer source. Only the game's admitted profiles are decoded:
// selector 0, F32 destination, CC clear, negate clear.
#pragma once
#include <armigo/shader/maxwell_operands.h>
#include <cstdint>
#include <variant>
namespace armigo::shader::maxwell {
struct IntegerToFloat {
    RegisterOperand destination;
    // The source word at bits 20..27 (register) or the constant-buffer word.
    std::variant<RegisterOperand, ConstantBufferOperand> source;
    bool is_signed = false;
    // Source width in bits after the selector-0 extraction: 16 or 32.
    std::uint8_t source_width = 32;
    RoundingMode rounding = RoundingMode::NearestEven;
    // Wraparound integer absolute value applied before the conversion; the
    // least value stays itself, so abs of the minimum stays negative.
    bool absolute = false;
};
}  // namespace armigo::shader::maxwell
