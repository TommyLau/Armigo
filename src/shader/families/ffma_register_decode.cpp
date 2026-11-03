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
DecodeOutcome decode_ffma_register(std::uint64_t word, std::uint32_t pc,
                                   Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatFmaRegister;
    const auto allowed = high_mask(0xff80) | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(8, 20) | bit_mask(8, 39) |
        bit_mask(1, 47) | bit_mask(1, 48) | bit_mask(1, 49) |
        bit_mask(1, 50) | bit_mask(2, 51) | bit_mask(2, 53);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::ConditionCode,47,1,0}, Rule{DecodeField::Rounding,51,2,0},
        Rule{DecodeField::FlushMode,53,2,1}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc,family,rule.field,
                                     field(word,rule.shift,rule.width),rule.expected))
            return *error;
    return Instruction{pc,word,family,predicate,
        FloatFmaRegister{reg(word,0),reg(word,8),reg(word,20),reg(word,39),
            RoundingMode::NearestEven,FlushMode::FlushToZero,
            field(word,48,1)!=0,field(word,49,1)!=0,field(word,50,1)!=0}};
}
}  // namespace armigo::shader::maxwell::detail
