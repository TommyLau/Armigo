// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of F2F for equal F32 source and destination sizes: the source word
// rounded to an integral value, optionally saturated afterwards.
#pragma once
#include <armigo/shader/maxwell_operands.h>
#include <cstdint>
#include <variant>
namespace armigo::shader::maxwell {
// The same-size rounding operation of the F2F field at bits 39..42, mapped to
// its low two bits: Round=8, Floor=9, Ceil=10, Trunc=11.
enum class IntegralRounding : std::uint8_t {
    NearestEven = 0,
    Floor = 1,
    Ceil = 2,
    Trunc = 3,
};
struct FloatRoundIntegral {
    RegisterOperand destination;
    std::variant<RegisterOperand, ConstantBufferOperand> source;
    IntegralRounding rounding = IntegralRounding::Floor;
    bool saturate = false;
};
}  // namespace armigo::shader::maxwell
