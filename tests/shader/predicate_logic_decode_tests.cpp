// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"

#include <armigo/shader/graphics.h>
#include <armigo/shader/maxwell.h>

#include "maxwell_families.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

namespace mx = maxwell;
namespace gx = graphics;
using Family = mx::OpcodeFamily;
using BooleanOperation = mx::PredicateBooleanOperation;

// One synthetic predicate-set-predicate word with every decoded field
// settable. The defaults are the admitted profile: AND as both boolean
// operations, uninverted in-range sources with the third one reading PT, and
// a PT instruction predicate.
std::uint64_t psetp_word(std::uint64_t dest_a = 3, std::uint64_t dest_b = 6,
                         std::uint64_t lhs = 4, std::uint64_t rhs = 2,
                         std::uint64_t gate = 7,
                         std::uint64_t lhs_neg = 0,
                         std::uint64_t rhs_neg = 0,
                         std::uint64_t gate_neg = 0,
                         std::uint64_t first = 0,
                         std::uint64_t second = 0) {
    return (std::uint64_t{0x5090} << 48) | (std::uint64_t{7} << 16) |
           (dest_b << 0) | (dest_a << 3) | (lhs << 12) | (lhs_neg << 15) |
           (first << 24) | (rhs << 29) | (rhs_neg << 32) | (gate << 39) |
           (gate_neg << 42) | (second << 45);
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value,
         unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        bytes[at + i] = std::uint8_t(value >> (8 * i));
    }
}

// Minimal straight-line vertex program: the instruction under test at pc 8
// followed by an unconditional PT EXIT.
mx::DecodeResult decode(std::uint64_t instruction) {
    std::vector<std::uint8_t> control(0x718), code(0xa0);
    put(control, 0, 0x98761234, 4);
    put(control, 4, 1, 4);
    put(control, 8, 14, 4);
    put(control, 0x20, control.size(), 4);
    put(control, 0x700, code.size(), 4);
    put(control, 0x704, code.size(), 4);
    put(control, 0x708, 0x30, 4);
    put(control, 0x714, 0, 4);
    put(code, 0, 0x12345678, 4);
    put(code, 0x30,
        shader_input::sph_stage_type_for(shader_input::Stage::Vertex) << 10, 4);
    put(code, 0x88, instruction, 8);
    put(code, 0x90,
        (std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15, 8);
    return mx::decode({control.data(), control.size()},
                      {code.data(), code.size()});
}

const mx::PredicateSet* payload_of(const mx::DecodeResult& result) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return nullptr;
    }
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == Family::PredicateSetPredicate);
    ARMIGO_CHECK_EQ(instruction.pc, 8u);
    ARMIGO_CHECK(instruction.predicate.index == 7);
    ARMIGO_CHECK(!instruction.predicate.inverted);
    return std::get_if<mx::PredicateSet>(&instruction.payload);
}

bool expect_rejected(std::uint64_t instruction,
                     mx::DecodeField field,
                     std::uint64_t value,
                     std::uint64_t expected) {
    const auto result = decode(instruction);
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) {
        return false;
    }
    ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnsupportedModifier);
    ARMIGO_CHECK(error->family == Family::PredicateSetPredicate);
    ARMIGO_CHECK(error->field == field);
    ARMIGO_CHECK_EQ(error->value, value);
    ARMIGO_CHECK_EQ(error->expected, expected);
    ARMIGO_CHECK_EQ(error->pc, 8u);
    return true;
}

}  // namespace

ARMIGO_TEST(predicate_logic, decodes_admitted_profile_and_preserves_fields) {
    const std::uint64_t encoded = psetp_word();
    const auto result = decode(encoded);
    const auto* value = payload_of(result);
    ARMIGO_CHECK(value != nullptr);
    if (value == nullptr) {
        return;
    }
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(program->instructions.front().original_word, encoded);
    }
    ARMIGO_CHECK(value->destination_a == 3);
    ARMIGO_CHECK(value->destination_b == 6);
    ARMIGO_CHECK(value->lhs.index == 4);
    ARMIGO_CHECK(!value->lhs.inverted);
    ARMIGO_CHECK(value->rhs.index == 2);
    ARMIGO_CHECK(!value->rhs.inverted);
    ARMIGO_CHECK(value->gate.index == 7);
    ARMIGO_CHECK(!value->gate.inverted);
    ARMIGO_CHECK(value->first == BooleanOperation::And);
    ARMIGO_CHECK(value->second == BooleanOperation::And);
    // The operation encodings are the hardware order AND, OR, XOR.
    ARMIGO_CHECK_EQ(unsigned(BooleanOperation::And), 0u);
    ARMIGO_CHECK_EQ(unsigned(BooleanOperation::Or), 1u);
    ARMIGO_CHECK_EQ(unsigned(BooleanOperation::Xor), 2u);
}

ARMIGO_TEST(predicate_logic,
            preserves_field_endpoints_inversions_aliasing_and_pt) {
    const auto check = [](const mx::DecodeResult& result, unsigned dest_a,
                          unsigned dest_b, unsigned lhs, bool lhs_neg,
                          unsigned rhs, bool rhs_neg, unsigned gate,
                          bool gate_neg) {
        const auto* value = payload_of(result);
        ARMIGO_CHECK(value != nullptr);
        if (value == nullptr) {
            return;
        }
        ARMIGO_CHECK(unsigned(value->destination_a) == dest_a);
        ARMIGO_CHECK(unsigned(value->destination_b) == dest_b);
        ARMIGO_CHECK(unsigned(value->lhs.index) == lhs);
        ARMIGO_CHECK(value->lhs.inverted == lhs_neg);
        ARMIGO_CHECK(unsigned(value->rhs.index) == rhs);
        ARMIGO_CHECK(value->rhs.inverted == rhs_neg);
        ARMIGO_CHECK(unsigned(value->gate.index) == gate);
        ARMIGO_CHECK(value->gate.inverted == gate_neg);
    };
    // Endpoint destinations 0 and PT, endpoint sources 0 and PT.
    check(decode(psetp_word(0, 0, 0, 0, 0)), 0, 0, 0, false, 0, false, 0,
          false);
    check(decode(psetp_word(7, 7, 7, 7, 7)), 7, 7, 7, false, 7, false, 7,
          false);
    // Destinations may alias each other and sources may share one index.
    check(decode(psetp_word(5, 5, 2, 2, 2)), 5, 5, 2, false, 2, false, 2,
          false);
    // Each inversion bit is preserved on its own source predicate.
    check(decode(psetp_word(3, 6, 4, 2, 7, 1, 0, 0)), 3, 6, 4, true, 2, false,
          7, false);
    check(decode(psetp_word(3, 6, 4, 2, 7, 0, 1, 0)), 3, 6, 4, false, 2, true,
          7, false);
    check(decode(psetp_word(3, 6, 4, 2, 7, 0, 0, 1)), 3, 6, 4, false, 2, false,
          7, true);
    check(decode(psetp_word(1, 0, 0, 7, 5, 1, 1, 1)), 1, 0, 0, true, 7, true,
          5, true);
    // The all-PT sources profile with PT and p0 destinations decodes.
    check(decode(psetp_word(0, 7, 7, 7, 7)), 0, 7, 7, false, 7, false, 7,
          false);
}

ARMIGO_TEST(predicate_logic, admits_every_boolean_operation_combination) {
    const std::array<std::uint64_t, 3> encodings{0u, 1u, 2u};
    const std::array<BooleanOperation, 3> values{
        BooleanOperation::And, BooleanOperation::Or, BooleanOperation::Xor};
    for (std::size_t first = 0; first < encodings.size(); ++first) {
        for (std::size_t second = 0; second < encodings.size(); ++second) {
            const auto result = decode(psetp_word(
                3, 6, 4, 2, 7, 0, 0, 0, encodings[first], encodings[second]));
            const auto* value = payload_of(result);
            ARMIGO_CHECK(value != nullptr);
            if (value == nullptr) {
                continue;
            }
            ARMIGO_CHECK(value->first == values[first]);
            ARMIGO_CHECK(value->second == values[second]);
        }
    }
}

ARMIGO_TEST(predicate_logic, rejects_invalid_boolean_operation_encoding) {
    // The fourth 2-bit encoding is admitted by neither operation field; the
    // expected value carries the highest admitted encoding, XOR.
    expect_rejected(psetp_word(3, 6, 4, 2, 7, 0, 0, 0, 3, 0),
                    mx::DecodeField::BooleanOperation, 3, 2);
    expect_rejected(psetp_word(3, 6, 4, 2, 7, 0, 0, 0, 0, 3),
                    mx::DecodeField::BooleanOperation, 3, 2);
    expect_rejected(psetp_word(3, 6, 4, 2, 7, 0, 0, 0, 3, 3),
                    mx::DecodeField::BooleanOperation, 3, 2);
}

ARMIGO_TEST(predicate_logic, rejects_every_reserved_bit_explicitly) {
    // Reserved positions live only in the operand area: the thirteen fixed
    // opcode bits (63..51) are the registry's match pattern, and the three
    // wildcard bits 50..48 are admitted, so none of them is a reserved
    // operand here.
    for (unsigned shift : {6u, 7u, 8u, 9u, 10u, 11u, 20u, 21u, 22u, 23u,
                           26u, 27u, 28u, 33u, 34u, 35u, 36u, 37u, 38u,
                           43u, 44u, 47u}) {
        ARMIGO_CHECK(expect_rejected(
            psetp_word() | (std::uint64_t{1} << shift),
            mx::DecodeField::ReservedBits, std::uint64_t{1} << shift, 0));
    }
}

ARMIGO_TEST(predicate_logic, admits_opcode_wildcard_high_bits) {
    // Bits 50..48 are opcode-pattern wildcards: all eight high variants of
    // the admitted opcode decode, and the payload is identical under every
    // variant, including both boolean operations and the source inversions.
    for (std::uint64_t variant = 0; variant < 8; ++variant) {
        const std::uint64_t word = psetp_word(3, 6, 4, 2, 7, 1, 0, 1, 2, 1) |
                                   (variant << 48);
        const auto result = decode(word);
        const auto* value = payload_of(result);
        ARMIGO_CHECK(value != nullptr);
        if (value == nullptr) {
            continue;
        }
        ARMIGO_CHECK(value->destination_a == 3);
        ARMIGO_CHECK(value->destination_b == 6);
        ARMIGO_CHECK(value->lhs.index == 4);
        ARMIGO_CHECK(value->lhs.inverted);
        ARMIGO_CHECK(value->rhs.index == 2);
        ARMIGO_CHECK(!value->rhs.inverted);
        ARMIGO_CHECK(value->gate.index == 7);
        ARMIGO_CHECK(value->gate.inverted);
        ARMIGO_CHECK(value->first == BooleanOperation::Xor);
        ARMIGO_CHECK(value->second == BooleanOperation::Or);
    }
}

ARMIGO_TEST(predicate_logic, neighbors_are_never_admitted_by_the_registry) {
    // The MUFU, PSET, CSET, NOP, and FCHK register windows around the
    // admitted opcode belong to no registered family.
    for (std::uint16_t high : {0x5080u, 0x5088u, 0x5098u, 0x50a0u, 0x50b0u,
                               0x50c0u, 0x5c88u}) {
        ARMIGO_CHECK(maxwell::detail::find_registered_family(
                         std::uint64_t{high} << 48) == nullptr);
    }
    for (std::uint16_t high : {0x5088u, 0x5098u, 0x50a8u, 0x50b0u}) {
        const auto result = decode((std::uint64_t{high} << 48) |
                                   (std::uint64_t{7} << 16) |
                                   (std::uint64_t{9} << 29));
        const auto* error = std::get_if<mx::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnknownOpcode);
        }
    }
    // The admitted pattern's thirteen fixed opcode bits match exactly one
    // entry across its three wildcard bits (50..48), and every word of the
    // CSET window beyond fixed bit 51 matches no entry at all.
    const auto families = maxwell::detail::registered_families();
    const maxwell::detail::FamilyEntry* entry = nullptr;
    for (const maxwell::detail::FamilyEntry& registered : families) {
        if (registered.family == Family::PredicateSetPredicate) {
            entry = &registered;
        }
    }
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(entry->mask, 0xfff8u);
    ARMIGO_CHECK_EQ(entry->value, 0x5090u);
    ARMIGO_CHECK(entry->decode != nullptr);
    for (std::uint16_t low = 0x5090u; low <= 0x5097u; ++low) {
        ARMIGO_CHECK(maxwell::detail::find_registered_family(
                         std::uint64_t{low} << 48) == entry);
    }
    for (std::uint16_t low = 0x5098u; low <= 0x509fu; ++low) {
        ARMIGO_CHECK(maxwell::detail::find_registered_family(
                         std::uint64_t{low} << 48) == nullptr);
    }
}

ARMIGO_TEST(predicate_logic, rejects_non_pt_instruction_predicates) {
    for (std::uint64_t pred : {0u, 1u, 6u, 8u, 15u}) {
        const std::uint64_t word =
            (psetp_word() & ~(std::uint64_t{15} << 16)) | (pred << 16);
        const auto result = decode(word);
        const auto* error = std::get_if<mx::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code ==
                         mx::DecodeError::Code::UnsupportedPredicate);
            ARMIGO_CHECK(error->family == Family::PredicateSetPredicate);
            ARMIGO_CHECK(error->field ==
                         mx::DecodeField::InstructionPredicate);
            ARMIGO_CHECK_EQ(error->value, pred);
            ARMIGO_CHECK_EQ(error->expected, 7u);
        }
    }
}

ARMIGO_TEST(predicate_logic, lowering_still_rejects_the_family) {
    mx::DecodedProgram program;
    program.stage = shader_input::Stage::Vertex;
    const auto stage_bits =
        shader_input::sph_stage_type_for(program.stage) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        program.sph[i] = std::byte(stage_bits >> (8 * i));
    }
    program.sph[0x35] = std::byte{0xf0};
    const auto append = [&program](Family family,
                                   mx::InstructionPayload payload) {
        const auto index = program.instructions.size();
        program.instructions.push_back(
            {std::uint32_t((index / 3) * 32 + (index % 3 + 1) * 8), 0, family,
             {7, false}, std::move(payload)});
    };
    append(Family::Mov32Immediate, mx::Mov32Immediate{{1}, 15, 0});
    append(Family::PredicateSetPredicate,
           mx::PredicateSet{3, 6, {4, false}, {5, false}, {7, false},
                            BooleanOperation::And, BooleanOperation::Or});
    append(Family::Exit, mx::Exit{15});
    const auto lowered = gx::lower(program);
    const auto* error = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::InvalidInstruction);
        ARMIGO_CHECK_EQ(error->pc, 16u);
    }
}

}  // namespace armigo::shader::test
