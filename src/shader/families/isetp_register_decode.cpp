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

// Integer set predicate, register form. The admitted profile is not-equal on
// two registers combined by AND with the PT source predicate: every other
// comparison, boolean operation, source predicate value, negation, extended
// compare, and all reserved bits are explicit errors. The paired predicate
// destinations 0..7 (PT discards downstream), both register operands
// including RZ, and the signed field are preserved as encoded.
DecodeOutcome decode_isetp_register(std::uint64_t word, std::uint32_t pc,
                                    Predicate predicate) {
    constexpr auto family = OpcodeFamily::IntegerSetPredicateRegister;
    const std::uint64_t allowed = high_mask(0xfff0) | bit_mask(6, 0) |
                                  bit_mask(8, 8) | bit_mask(4, 16) |
                                  bit_mask(8, 20) | bit_mask(5, 39) |
                                  bit_mask(2, 45) | bit_mask(1, 48) |
                                  bit_mask(3, 49);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    struct Rule {
        DecodeField field;
        unsigned shift, width;
        std::uint64_t expected;
    };
    constexpr std::array rules{
        Rule{DecodeField::BooleanPredicate, 39, 3, 7},
        Rule{DecodeField::BooleanPredicateInversion, 42, 1, 0},
        Rule{DecodeField::ExtendedMode, 43, 1, 0},
        Rule{DecodeField::BooleanOperation, 45, 2, 0},
        Rule{DecodeField::Comparison, 49, 3, 5},
    };
    for (const auto& rule : rules) {
        if (auto error = unsupported(pc, family, rule.field,
                                     field(word, rule.shift, rule.width),
                                     rule.expected)) {
            return *error;
        }
    }
    return Instruction{
        pc, word, family, predicate,
        IntegerSetPredicate{
            static_cast<std::uint8_t>(field(word, 3, 3)),
            static_cast<std::uint8_t>(field(word, 0, 3)), reg(word, 8),
            reg(word, 20),
            Predicate{static_cast<std::uint8_t>(field(word, 39, 3)), false},
            field(word, 48, 1) != 0}};
}

}  // namespace armigo::shader::maxwell::detail
