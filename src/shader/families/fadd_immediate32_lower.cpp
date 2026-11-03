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

// The immediate summand becomes a literal. Negate-B and absolute-B apply to
// the immediate itself: flipping or clearing bit 31 of the literal word is
// exact because the add flushes subnormal inputs and flushing preserves the
// sign, and applying absolute before negate matches the hardware ordering
// even when both modifiers are set. The register summand is read before the
// destination write can alias it, and negate-A is the add node's checked
// negation of that operand.
void lower_fadd_immediate32(const maxwell::Instruction& instruction,
                            FamilyLower& lower) {
    const auto* value =
        std::get_if<maxwell::FloatAddImmediate32>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    auto immediate_bits = value->immediate_bits;
    if (value->absolute_b) {
        immediate_bits &= 0x7fffffffu;
    }
    if (value->negate_b) {
        immediate_bits ^= 0x80000000u;
    }
    // Absolute A is the registered sign clear applied before the add node's
    // negate, matching the instruction's absolute-then-negate modifier order;
    // the add flushes afterwards and sign-preserving FTZ commutes with it.
    auto summand_a = lower.read(value->source_a);
    if (value->absolute_a) {
        summand_a = lower.operation(PureKind::FloatAbsolute, {summand_a});
    }
    const auto summand_b = lower.literal(immediate_bits);
    // Register zero is a discard target for the final write.
    lower.write(value->destination,
                lower.arithmetic(FloatArithmetic{Arithmetic::Add, summand_a,
                                                 summand_b, std::nullopt,
                                                 value->negate_a}));
}

}  // namespace armigo::shader::graphics::detail
