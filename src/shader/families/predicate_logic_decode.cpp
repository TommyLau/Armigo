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

// Predicate set predicate. The opcode pattern fixes thirteen high-half bits
// (63..51); the registry only routes words whose high half is 0x5090 through
// 0x5097 to this decoder, so the fixed bit 51 is an opcode boundary, not a
// reserved operand: a word with bit 51 set belongs to the CSET neighbor
// opcode and stays unregistered. Bits 50..48 are opcode-pattern wildcards
// the opcode table admits and the instruction semantics never
// read, so they are admitted here without operand meaning. The admitted
// profile keeps both 2-bit boolean operations inside AND/OR/XOR and treats
// every other undecoded field position as reserved except the three source
// predicates with their independent inversions, the paired 3-bit predicate
// destinations, and the common PT instruction predicate. Destinations and
// sources are preserved as encoded, including PT (7). The payload carries
// decoding facts only; evaluating the boolean combinations is downstream
// work that no path implements yet.
DecodeOutcome decode_predicate_set_predicate(std::uint64_t word,
                                             std::uint32_t pc,
                                             Predicate predicate) {
    constexpr auto family = OpcodeFamily::PredicateSetPredicate;
    const std::uint64_t allowed = high_mask(0xfff8) | bit_mask(3, 48) |
                                  bit_mask(6, 0) |
                                  bit_mask(3, 12) | bit_mask(1, 15) |
                                  bit_mask(4, 16) | bit_mask(2, 24) |
                                  bit_mask(3, 29) | bit_mask(1, 32) |
                                  bit_mask(3, 39) | bit_mask(1, 42) |
                                  bit_mask(2, 45);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    struct Rule {
        unsigned shift;
    };
    constexpr std::array rules{Rule{24}, Rule{45}};
    for (const auto& rule : rules) {
        const std::uint64_t operation = field(word, rule.shift, 2);
        if (operation > static_cast<std::uint64_t>(
                            PredicateBooleanOperation::Xor)) {
            return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                           DecodeField::BooleanOperation, operation,
                           static_cast<std::uint64_t>(
                               PredicateBooleanOperation::Xor));
        }
    }
    return Instruction{
        pc, word, family, predicate,
        PredicateSet{
            static_cast<std::uint8_t>(field(word, 3, 3)),
            static_cast<std::uint8_t>(field(word, 0, 3)),
            Predicate{static_cast<std::uint8_t>(field(word, 12, 3)),
                      field(word, 15, 1) != 0},
            Predicate{static_cast<std::uint8_t>(field(word, 29, 3)),
                      field(word, 32, 1) != 0},
            Predicate{static_cast<std::uint8_t>(field(word, 39, 3)),
                      field(word, 42, 1) != 0},
            static_cast<PredicateBooleanOperation>(field(word, 24, 2)),
            static_cast<PredicateBooleanOperation>(field(word, 45, 2))}};
}

}  // namespace armigo::shader::maxwell::detail
