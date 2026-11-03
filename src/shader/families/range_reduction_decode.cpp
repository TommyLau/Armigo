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

// The floating-point range-reduction unit, register form: destination bits
// 0..7, source register B bits 20..27, mode bit 39, negate bit 45, and
// absolute bit 49 are admitted and carried; every other bit is rejected with
// the structured reserved-bits error, as the other families reject their
// unused fields. The instruction predicate has already been validated by the
// common gate.
DecodeOutcome decode_range_reduction_register(std::uint64_t word,
                                              std::uint32_t pc,
                                              Predicate predicate) {
    const OpcodeFamily family = OpcodeFamily::RangeReductionRegister;
    const std::uint64_t allowed = high_mask(0xfff8) | bit_mask(8, 0) |
                                  bit_mask(4, 16) | bit_mask(8, 20) |
                                  bit_mask(1, 39) | bit_mask(1, 45) |
                                  bit_mask(1, 49);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    return Instruction{
        pc, word, family, predicate,
        RangeReduction{reg(word, 0), reg(word, 20), field(word, 39, 1) != 0,
                       field(word, 45, 1) != 0, field(word, 49, 1) != 0}};
}

// The constant-buffer form: the source is the shared constant-buffer operand
// (word offset bits 20..33, binding bits 34..38) with the same mode, negate,
// and absolute admission, and the same shared binding bound check.
DecodeOutcome decode_range_reduction_constant(std::uint64_t word,
                                              std::uint32_t pc,
                                              Predicate predicate) {
    const OpcodeFamily family = OpcodeFamily::RangeReductionConstant;
    const std::uint64_t allowed = high_mask(0xfff8) | bit_mask(8, 0) |
                                  bit_mask(4, 16) | bit_mask(14, 20) |
                                  bit_mask(5, 34) | bit_mask(1, 39) |
                                  bit_mask(1, 45) | bit_mask(1, 49);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    std::variant<ConstantBufferOperand, DecodeError> source =
        constant_buffer(word, pc, family);
    if (auto* error = std::get_if<DecodeError>(&source)) {
        return *error;
    }
    return Instruction{
        pc, word, family, predicate,
        RangeReduction{reg(word, 0), std::get<ConstantBufferOperand>(source),
                       field(word, 39, 1) != 0, field(word, 45, 1) != 0,
                       field(word, 49, 1) != 0}};
}

}  // namespace armigo::shader::maxwell::detail
