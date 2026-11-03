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

// The register and constant-buffer F2I families share one payload and one
// lowering: the source becomes the single operand of the FloatToInteger
// registered operation, whose immediate carries the decoded profile. The
// decoder admits only the documented profiles, so every modifier mismatch
// here is a forged instruction, never a silently dropped modifier.
void lower_float_to_integer(const maxwell::Instruction& instruction,
                            FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatToInteger>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    if (value->flush != maxwell::FlushMode::FlushToZero ||
        (value->destination_width != 16 && value->destination_width != 32)) {
        lower.fail(Error::Code::UnsupportedModifier);
    }
    // Immediate layout of the FloatToInteger operation: bit 0 signedness,
    // bit 1 a 32-bit destination, bits 3..4 the instruction's rounding code.
    const std::uint32_t immediate =
        (value->is_signed ? 1u : 0u) |
        (value->destination_width == 32u ? 2u : 0u) |
        (static_cast<std::uint32_t>(value->rounding) << 3u);
    // Read the source before the destination write can alias it.
    const auto* source_register =
        std::get_if<maxwell::RegisterOperand>(&value->source);
    if ((instruction.family == maxwell::OpcodeFamily::FloatToIntegerRegister) !=
        (source_register != nullptr)) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    const auto source = source_register != nullptr
        ? lower.read(*source_register)
        : lower.constant(std::get<maxwell::ConstantBufferOperand>(value->source));
    lower.write(value->destination,
                lower.operation(PureKind::FloatToInteger, {source}, immediate));
}

}  // namespace armigo::shader::graphics::detail
