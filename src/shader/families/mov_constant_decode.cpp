// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../maxwell_families.h"

#include "../maxwell_decode_helpers.h"

#include <variant>

namespace armigo::shader::maxwell::detail {

// The constant-buffer register-move unit: admitted write masks, reserved
// bits, and the shared constant-buffer operand checks reproduce the same
// structured errors the register-move and legacy constant-buffer forms
// produce. The instruction predicate has already been validated as PT by the
// common gate.
DecodeOutcome decode_mov_constant(std::uint64_t word, std::uint32_t pc,
                                  Predicate predicate) {
    const OpcodeFamily family = OpcodeFamily::MovConstant;
    const std::uint64_t allowed = high_mask(0xfff8) | bit_mask(8, 0) |
                                  bit_mask(4, 16) | bit_mask(19, 20) |
                                  bit_mask(4, 39);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    const auto mask = static_cast<std::uint8_t>(field(word, 39, 4));
    if (mask != 15 && mask != 1) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::WriteMask, mask, 15);
    }
    std::variant<ConstantBufferOperand, DecodeError> source =
        constant_buffer(word, pc, family);
    if (auto* error = std::get_if<DecodeError>(&source)) {
        return *error;
    }
    return Instruction{
        pc, word, family, predicate,
        MovConstant{reg(word, 0), std::get<ConstantBufferOperand>(source),
                    mask}};
}

}  // namespace armigo::shader::maxwell::detail
