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
#include <utility>
namespace armigo::shader::maxwell::detail {
namespace {

// One FMNMX word: destination 0..7, A 8..15,
// B per window, guard predicate 16..19 (validated PT centrally), selecting
// predicate 39..41, negate predicate 42, FTZ 44, negate B 45, absolute A 46,
// CC 47, negate A 48, absolute B 49. Every other bit must be zero. The
// window contributes its own high-half mask and its B-field bits.
DecodeOutcome decode_float_min_max(std::uint64_t word, std::uint32_t pc,
                                   Predicate predicate, OpcodeFamily family,
                                   std::uint64_t window_allowed,
                                   FloatSource source_b) {
    const auto allowed = window_allowed | bit_mask(8, 0) | bit_mask(8, 8) |
        bit_mask(4, 16) | bit_mask(3, 39) | bit_mask(1, 42) |
        bit_mask(1, 44) | bit_mask(1, 45) | bit_mask(1, 46) |
        bit_mask(1, 47) | bit_mask(1, 48) | bit_mask(1, 49);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::SelectPredicate, 39, 3, 7},
        Rule{DecodeField::FlushMode, 44, 1, 1},
        Rule{DecodeField::ConditionCode, 47, 1, 0}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc, family, rule.field,
                                     field(word, rule.shift, rule.width),
                                     rule.expected))
            return *error;
    // Negation is admitted only on its own. The two absolute modifiers
    // may occur together, but neither may accompany a negation.
    if (field(word, 48, 1) || field(word, 45, 1)) {
        for (const auto& rule : {Rule{DecodeField::AbsoluteA, 46, 1, 0},
                                 Rule{DecodeField::AbsoluteB, 49, 1, 0}}) {
            if (auto error = unsupported(pc, family, rule.field,
                                         field(word, rule.shift, rule.width), 0))
                return *error;
        }
        if (field(word, 48, 1) && field(word, 45, 1))
            return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                           DecodeField::NegateB, 1, 0);
    }
    return Instruction{pc, word, family, predicate,
        FloatMinMax{reg(word, 0), reg(word, 8), std::move(source_b),
            field(word, 46, 1) != 0, field(word, 48, 1) != 0,
            field(word, 49, 1) != 0, field(word, 45, 1) != 0,
            field(word, 42, 1) != 0, FlushMode::FlushToZero}};
}

}  // namespace

DecodeOutcome decode_float_min_max_register(std::uint64_t word,
                                            std::uint32_t pc,
                                            Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatMinMaxRegister;
    return decode_float_min_max(word, pc, predicate, family,
                                high_mask(0xfff8) | bit_mask(8, 20),
                                FloatSource{reg(word, 20)});
}

DecodeOutcome decode_float_min_max_constant(std::uint64_t word,
                                            std::uint32_t pc,
                                            Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatMinMaxConstant;
    auto operand = constant_buffer(word, pc, family);
    if (const auto* error = std::get_if<DecodeError>(&operand)) return *error;
    return decode_float_min_max(word, pc, predicate, family,
                                high_mask(0xfff8) | bit_mask(19, 20),
                                FloatSource{std::get<ConstantBufferOperand>(
                                    std::move(operand))});
}

DecodeOutcome decode_float_min_max_immediate(std::uint64_t word,
                                             std::uint32_t pc,
                                             Predicate predicate) {
    constexpr auto family = OpcodeFamily::FloatMinMaxImmediate;
    // 19 exponent/fraction bits at 20 shifted into binary32 bits 12..30
    // with the sign at bit 56, exactly like the other float-immediate forms.
    const auto immediate = ShortFloatImmediate{
        static_cast<std::uint32_t>(field(word, 20, 19) << 12) |
        static_cast<std::uint32_t>(field(word, 56, 1) << 31)};
    return decode_float_min_max(word, pc, predicate, family,
                                high_mask(0xfef8) | bit_mask(19, 20) |
                                    bit_mask(1, 56),
                                FloatSource{immediate});
}
}  // namespace armigo::shader::maxwell::detail
