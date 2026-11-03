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
struct FloatFmaRegister {
    RegisterOperand destination;
    RegisterOperand source_a;
    RegisterOperand source_b;
    RegisterOperand source_c;
    RoundingMode rounding = RoundingMode::NearestEven;
    FlushMode flush = FlushMode::FlushToZero;
    bool negate_b = false;
    bool negate_c = false;
    bool saturate = false;
};
}  // namespace armigo::shader::maxwell
