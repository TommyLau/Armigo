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
DecodeOutcome decode_word_and_immediate(std::uint64_t word,std::uint32_t pc,Predicate predicate) {
    constexpr auto family=OpcodeFamily::WordAndImmediate;
    struct Rule { DecodeField field; unsigned shift,width; };
    constexpr std::array rules{
        Rule{DecodeField::ConditionCode,52,1},Rule{DecodeField::LogicOperation,53,2},
        Rule{DecodeField::InvertA,55,1},Rule{DecodeField::InvertB,56,1},
        Rule{DecodeField::ExtendedMode,57,1}};
    for(const auto& rule:rules)
        if(auto error=unsupported(pc,family,rule.field,field(word,rule.shift,rule.width),0))
            return *error;
    return Instruction{pc,word,family,predicate,WordAndImmediate{
        reg(word,0),reg(word,8),static_cast<std::uint32_t>(field(word,20,32))}};
}
}  // namespace armigo::shader::maxwell::detail
