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
void lower_fma_constant_negated(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatFmaConstantNegated>(&instruction.payload);
    if (!value) lower.fail(Error::Code::InvalidInstruction);
    if (value->rounding != maxwell::RoundingMode::NearestEven ||
        value->flush != maxwell::FlushMode::FlushToZero)
        lower.fail(Error::Code::UnsupportedModifier);
    // Read both register sources before writing an aliased destination. Keep
    // the constant-buffer operand in the multiplier position of the existing
    // fused node.
    const auto lhs = lower.read(value->source_a);
    const auto addend = lower.read(value->addend);
    const auto multiplier = lower.constant(value->multiplier);
    // The established fused node's product-sign flip represents NegateB
    // exactly under the admitted FTZ/canonical-NaN contract: multiplicand
    // magnitudes and the XOR of their signs determine the product, including
    // signed zeros, with no intermediate operation.
    lower.write(value->destination,lower.arithmetic(FloatArithmetic{
        Arithmetic::FusedMultiplyAdd,lhs,multiplier,addend,true}));
}
}  // namespace armigo::shader::graphics::detail
