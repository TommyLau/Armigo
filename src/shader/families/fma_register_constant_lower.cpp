// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../graphics_families.h"
namespace armigo::shader::graphics::detail {
void lower_fma_register_constant(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatFmaRegisterConstant>(&instruction.payload);
    if (!value) lower.fail(Error::Code::InvalidInstruction);
    if (value->rounding != maxwell::RoundingMode::NearestEven ||
        value->flush != maxwell::FlushMode::FlushToZero)
        lower.fail(Error::Code::UnsupportedModifier);
    // Read both multiplicands before writing an aliased destination. Keep the
    // constant-buffer operand in the addend position of the existing fused node.
    const auto lhs = lower.read(value->source_a);
    const auto rhs = lower.read(value->source_b);
    const auto addend = lower.constant(value->source_c);
    // The established FTZ FMA canonicalizes NaNs and uses only the XOR of
    // multiplicand signs. Flipping either multiplicand therefore represents
    // NegateB exactly, including signed zeros, without an intermediate operation.
    lower.write(value->destination,lower.arithmetic(FloatArithmetic{
        Arithmetic::FusedMultiplyAdd,lhs,rhs,addend,value->negate_b}));
}
}  // namespace armigo::shader::graphics::detail
