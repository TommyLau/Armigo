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
namespace armigo::shader::maxwell {
struct FloatFmaConstantNegated {
    RegisterOperand destination;
    RegisterOperand source_a;
    ConstantBufferOperand multiplier;
    RegisterOperand addend;
    RoundingMode rounding = RoundingMode::NearestEven;
    FlushMode flush = FlushMode::FlushToZero;
};
}  // namespace armigo::shader::maxwell
