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

// The immediate multiplier becomes a literal; no sign modifier exists on this
// family, so every mismatch outside the admitted profile is a forged
// instruction. The register multiplicand is read before the destination write
// can alias it. The product is lowered as one MultiplyNearest node with no
// addend and no negation, then saturated by a separate checked node when the
// saturation modifier is set; saturation
// is never fused into or reordered before the multiply.
void lower_fmul_immediate32(const maxwell::Instruction& instruction,
                            FamilyLower& lower) {
    const auto* value =
        std::get_if<maxwell::FloatMultiplyImmediate32>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->flush != maxwell::FlushMode::FlushToZero) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    const auto multiplicand = lower.read(value->source_a);
    const auto immediate = lower.literal(value->immediate_bits);
    const auto product = lower.arithmetic(FloatArithmetic{
        Arithmetic::MultiplyNearest, multiplicand, immediate, std::nullopt,
        false});
    // Register zero is a discard target for the final write.
    lower.write(value->destination,
                value->saturate ? lower.saturate(product) : product);
}

}  // namespace armigo::shader::graphics::detail
