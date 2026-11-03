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
#include <variant>

namespace armigo::shader::maxwell::detail {
namespace {

// F2F register form "0101 1100 1010 1---" and constant-buffer form
// "0100 1100 1010 1---": destination 0..7,
// destination size 8..9, source size 10..11, source register 20..27 or
// constant-buffer operand (word offset 20..33, binding 34..38), rounding
// operation 39..42 (the selector bit 41 is the F16 half selector, which must
// be clear for F32, so the admitted values are exactly 8..11), FTZ 44,
// negate 45, CC 47, absolute 49, saturate 50. The admitted profile is F32 to
// F32 with FTZ; the game census uses only this profile.
std::uint64_t allowed_fields(OpcodeFamily family) {
    const std::uint64_t source = family == OpcodeFamily::FloatRoundIntegralRegister
        ? bit_mask(8, 20)     // source register
        : bit_mask(19, 20);   // constant-buffer word offset and binding
    return high_mask(0xfff8) | bit_mask(12, 0) | bit_mask(4, 16) |
           source | bit_mask(4, 39) | bit_mask(1, 44) |
           bit_mask(1, 45) | bit_mask(1, 47) | bit_mask(1, 49) |
           bit_mask(1, 50);
}

DecodeOutcome decode_float_round_integral(std::uint64_t word, std::uint32_t pc,
                                          Predicate predicate,
                                          OpcodeFamily family) {
    if (auto error = reserved(pc, family, word, allowed_fields(family)))
        return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    const std::array rules{
        Rule{DecodeField::DestinationSize, 8, 2, 2},
        Rule{DecodeField::SourceSize, 10, 2, 2},
        Rule{DecodeField::FlushMode, 44, 2, 1},
        Rule{DecodeField::ConditionCode, 47, 1, 0},
        Rule{DecodeField::AbsoluteA, 49, 1, 0}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc, family, rule.field,
                                     field(word, rule.shift, rule.width),
                                     rule.expected))
            return *error;
    const auto operation = field(word, 39, 4);
    if (operation < 8 || operation > 11) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::RoundingOperation, operation, 8);
    }
    FloatRoundIntegral payload{};
    payload.destination = reg(word, 0);
    payload.rounding = static_cast<IntegralRounding>(operation - 8);
    payload.saturate = field(word, 50, 1) != 0;
    if (family == OpcodeFamily::FloatRoundIntegralRegister) {
        payload.source = reg(word, 20);
        return Instruction{pc, word, family, predicate, std::move(payload)};
    }
    const auto source = constant_buffer(word, pc, family);
    if (const auto* error = std::get_if<DecodeError>(&source)) return *error;
    payload.source = std::get<ConstantBufferOperand>(source);
    return Instruction{pc, word, family, predicate, std::move(payload)};
}

}  // namespace

DecodeOutcome decode_float_round_integral_register(std::uint64_t word,
                                                   std::uint32_t pc,
                                                   Predicate predicate) {
    return decode_float_round_integral(word, pc, predicate,
                                       OpcodeFamily::FloatRoundIntegralRegister);
}

DecodeOutcome decode_float_round_integral_constant(std::uint64_t word,
                                                   std::uint32_t pc,
                                                   Predicate predicate) {
    return decode_float_round_integral(word, pc, predicate,
                                       OpcodeFamily::FloatRoundIntegralConstant);
}

}  // namespace armigo::shader::maxwell::detail
