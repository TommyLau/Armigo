// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payloads of the floating-point range-reduction family (RRO, both the
// register and constant-buffer forms) and of the dependency-barrier family
// (DEPBAR). This header declares only the shared operand types it needs; the
// decoder and lowering declarations stay in the private registry headers.
#pragma once

#include <armigo/shader/maxwell_operands.h>

#include <cstdint>
#include <variant>

namespace armigo::shader::maxwell {

// A range-reduction op: the result is the source with the absolute and
// negate modifiers applied and nothing else. The mode names the paired
// multi-function op (SINCOS = 0, EX2 = 1) that later consumes the result as
// an ordinary float; the mode itself performs no transformation here. The
// range-reduction result is an ordinary float operand of that op, never a
// hidden special-domain value.
struct RangeReduction {
    RegisterOperand destination;
    std::variant<RegisterOperand, ConstantBufferOperand> source;
    bool ex2_mode = false;
    bool negate = false;
    bool absolute = false;
};

// A dependency barrier: the hardware form waits on scoreboard dependencies,
// which have no meaning in Armigo's in-order translation, so the instruction
// lowers to no nodes and no register writes. The raw operand bits are kept
// for diagnostics only.
struct DependencyBarrier {
    std::uint64_t operand_bits = 0;
};

}  // namespace armigo::shader::maxwell
