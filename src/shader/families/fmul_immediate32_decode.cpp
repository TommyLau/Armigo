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
DecodeOutcome decode_fmul_immediate32(std::uint64_t word, std::uint32_t pc,
                                      Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatMultiplyImmediate32;
    // FMUL32I (window 0xff00/0x1e00): destination 0..7, source A 8..15, the
    // instruction predicate 16..19, the 32-bit immediate 20..51, condition
    // code 52, FMZ mode 53..54, and saturation 55; the opcode pattern's low
    // nibble coincides with the immediate's top four bits. The profile is
    // the conservative one: condition code off, FMZ flush-to-zero,
    // nearest-even rounding with no scale, and saturation carried either
    // way. Rounding is implied by the encoding, so it has no field here.
    const auto allowed = high_mask(0xff00) | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(32, 20) | bit_mask(1, 52) |
        bit_mask(2, 53) | bit_mask(1, 55);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::ConditionCode,52,1,0},
        Rule{DecodeField::FlushMode,53,2,1}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc,family,rule.field,
                                     field(word,rule.shift,rule.width),rule.expected))
            return *error;
    return Instruction{pc,word,family,predicate,
        FloatMultiplyImmediate32{reg(word,0),reg(word,8),
            static_cast<std::uint32_t>(field(word,20,32)),
            FlushMode::FlushToZero,field(word,55,1)!=0}};
}
}  // namespace armigo::shader::maxwell::detail
