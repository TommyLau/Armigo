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
DecodeOutcome decode_fmul_saturate_register(std::uint64_t word,
                                            std::uint32_t pc,
                                            Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatMultiplySaturateRegister;
    // FMUL with a register multiplier (fields 20-27) and the saturation
    // modifier (bit 50) set: the legacy classifier defers exactly this
    // high-half window (0xfffc/0x5c6c) to this unit. The admitted profile is
    // the conservative one: nearest-even rounding, FTZ flush, no scale,
    // condition code, or negation, so the payload reuses the existing
    // FloatMultiply type unchanged. The lowerer still rejects every
    // instruction of this family until the typed saturation hook lands.
    const auto allowed = high_mask(0xfffc) | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(8, 20) | bit_mask(7, 39) |
        bit_mask(1, 47) | bit_mask(1, 48);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::Rounding,39,2,0}, Rule{DecodeField::Scale,41,3,0},
        Rule{DecodeField::FlushMode,44,2,1}, Rule{DecodeField::ConditionCode,47,1,0},
        Rule{DecodeField::NegateB,48,1,0}, Rule{DecodeField::Saturate,50,1,1}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc,family,rule.field,
                                     field(word,rule.shift,rule.width),rule.expected))
            return *error;
    return Instruction{pc,word,family,predicate,
        FloatMultiply{reg(word,0),reg(word,8),reg(word,20),
                      RoundingMode::NearestEven,FlushMode::FlushToZero}};
}
}  // namespace armigo::shader::maxwell::detail
