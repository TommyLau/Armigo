// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of the constant-buffer register-move family. This header declares
// only the shared operand types it needs; the decoder and lowering
// declarations stay in the private registry headers.
#pragma once

#include <armigo/shader/maxwell_operands.h>

#include <cstdint>

namespace armigo::shader::maxwell {

// A full-word copy from one constant buffer word into a register. The source
// bits are preserved exactly as stored: no numeric interpretation, no NaN
// canonicalization, and no bit reinterpretation of any kind.
struct MovConstant {
    RegisterOperand destination;
    ConstantBufferOperand source;
    std::uint8_t write_mask = 15;
};

}  // namespace armigo::shader::maxwell
