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
#include <array>
namespace armigo::shader::maxwell::detail {
DecodeOutcome decode_fma_constant_negated(std::uint64_t word, std::uint32_t pc,
                                           Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatFmaConstantNegated;
    // FFMA with a constant-buffer multiplier (fields 20-38), a register
    // addend (fields 39-46), and NegateB (bit 48) set: the legacy classifier
    // defers exactly this high-half window (0xff81/0x4981) to this unit.
    const auto allowed = high_mask(0xff81) | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(19, 20) | bit_mask(8, 39) |
        bit_mask(8, 47) | bit_mask(1, 48);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::ConditionCode,47,1,0}, Rule{DecodeField::NegateC,49,1,0},
        Rule{DecodeField::Saturate,50,1,0}, Rule{DecodeField::Rounding,51,2,0},
        Rule{DecodeField::FlushMode,53,2,1}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc,family,rule.field,
                                     field(word,rule.shift,rule.width),rule.expected))
            return *error;
    const auto multiplier = constant_buffer(word,pc,family);
    if (const auto* error = std::get_if<DecodeError>(&multiplier)) return *error;
    return Instruction{pc,word,family,predicate,
        FloatFmaConstantNegated{reg(word,0),reg(word,8),
            std::get<ConstantBufferOperand>(multiplier),reg(word,39),
            RoundingMode::NearestEven,FlushMode::FlushToZero}};
}
}  // namespace armigo::shader::maxwell::detail
