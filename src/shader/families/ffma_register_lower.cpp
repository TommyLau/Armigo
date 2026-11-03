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
void lower_ffma_register(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatFmaRegister>(&instruction.payload);
    if (!value) lower.fail(Error::Code::InvalidInstruction);
    if (value->rounding != maxwell::RoundingMode::NearestEven ||
        value->flush != maxwell::FlushMode::FlushToZero)
        lower.fail(Error::Code::UnsupportedModifier);
    // The established fused node flips only the product's sign, which carries
    // NegateB exactly (the product of the multiplicand magnitudes depends on
    // the XOR of their signs, including signed zeros). NegateC negates the
    // addend itself through the registered sign flip, applying the
    // modifier to the operand before the fused operation; -fma(-a, b, c)
    // would differ in the sign of an exact-zero result. The fused node
    // flushes its addend afterwards, and sign-preserving FTZ commutes with
    // the flip.
    // Read all three register sources before writing an aliased destination.
    const auto lhs = lower.read(value->source_a);
    const auto rhs = lower.read(value->source_b);
    auto addend = lower.read(value->source_c);
    if (value->negate_c) addend = lower.operation(PureKind::FloatNegate, {addend});
    const auto product = lower.arithmetic(FloatArithmetic{
        Arithmetic::FusedMultiplyAdd,lhs,rhs,addend,value->negate_b});
    if (!value->saturate) {
        lower.write(value->destination,product);
        return;
    }
    // Saturation is a separate checked node over the already-computed sum,
    // never fused into or reordered before it.
    lower.write(value->destination,lower.saturate(product));
}
}  // namespace armigo::shader::graphics::detail
