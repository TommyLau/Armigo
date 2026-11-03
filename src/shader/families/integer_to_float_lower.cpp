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
#include <variant>
namespace armigo::shader::graphics::detail {
// I2F reads its source word and writes the converted float to the
// destination; the source is fully read before the write, so aliasing is
// safe. The immediate mirrors the payload modifiers, which the decoder has
// already confined to the admitted profiles.
void lower_integer_to_float(const maxwell::Instruction& instruction, FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::IntegerToFloat>(&instruction.payload);
    if (value == nullptr ||
        (value->source_width != 16 && value->source_width != 32)) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    ScalarId source{0};
    if (const auto* reg = std::get_if<maxwell::RegisterOperand>(&value->source)) {
        source = lower.read(*reg);
    } else {
        source = lower.constant(std::get<maxwell::ConstantBufferOperand>(value->source));
    }
    const std::uint32_t immediate =
        (value->is_signed ? 1u : 0u) |
        ((value->source_width == 16 ? 1u : 2u) << 1) |
        (static_cast<std::uint32_t>(value->rounding) << 3) |
        (value->absolute ? 32u : 0u);
    lower.write(value->destination,
                lower.operation(PureKind::IntegerToFloat, {source}, immediate));
}
}  // namespace armigo::shader::graphics::detail
