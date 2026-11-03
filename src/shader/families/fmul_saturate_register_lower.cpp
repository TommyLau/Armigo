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

#include <optional>
#include <variant>

namespace armigo::shader::graphics::detail {

// The saturating register family reuses the FloatMultiply payload with an
// implied saturation modifier, so every modifier mismatch on that payload is
// a forged instruction. The admitted profile is nearest-even rounding under
// flush-to-zero with a register multiplier; the product is lowered first and
// saturated by a separate checked node, matching the controlled multiply
// then distinct saturation order. Saturation is
// never fused into or reordered before the multiply, and no numeric
// saturation implementation is introduced here.
void lower_fmul_saturate_register(const maxwell::Instruction& instruction,
                                  FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatMultiply>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->rounding != maxwell::RoundingMode::NearestEven ||
        value->flush != maxwell::FlushMode::FlushToZero) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    const auto* source_b = std::get_if<maxwell::RegisterOperand>(&value->source_b);
    if (source_b == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    // Read both multiplicand registers before the destination write can
    // alias either one.
    const auto multiplicand_a = lower.read(value->source_a);
    const auto multiplicand_b = lower.read(*source_b);
    // MultiplyNearest carries no addend and no negation; the decode profile
    // admits no scale, condition code, or negate-B modifier to preserve.
    const auto product =
        lower.arithmetic(FloatArithmetic{Arithmetic::MultiplyNearest,
                                         multiplicand_a, multiplicand_b,
                                         std::nullopt, false});
    // Register zero is a discard target for the final write.
    lower.write(value->destination, lower.saturate(product));
}

}  // namespace armigo::shader::graphics::detail
