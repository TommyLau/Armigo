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
DecodeOutcome decode_fadd_immediate32(std::uint64_t word, std::uint32_t pc,
                                      Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatAddImmediate32;
    // FADD32I (window 0xfc00/0x0800): destination 0..7, source A 8..15, the
    // instruction predicate 16..19, the 32-bit immediate 20..51, condition
    // code 52, negate-B 53, absolute-A 54, flush-to-zero 55, negate-A 56,
    // and absolute-B 57; the opcode pattern's low bits coincide with the
    // immediate's top four. Rounding is implied nearest-even by the
    // encoding. Flush-to-zero is part of the admitted profile: it is a
    // one-bit modifier, so zero is the only rejected value. Sign and
    // absolute modifiers are carried verbatim.
    const auto allowed = high_mask(0xfc00) | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(32, 20) | bit_mask(6, 52);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::ConditionCode,52,1,0},
        Rule{DecodeField::FlushMode,55,1,1}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc,family,rule.field,
                                     field(word,rule.shift,rule.width),rule.expected))
            return *error;
    return Instruction{pc,word,family,predicate,
        FloatAddImmediate32{reg(word,0),reg(word,8),
            static_cast<std::uint32_t>(field(word,20,32)),
            field(word,56,1)!=0,field(word,54,1)!=0,
            field(word,53,1)!=0,field(word,57,1)!=0}};
}
}  // namespace armigo::shader::maxwell::detail
