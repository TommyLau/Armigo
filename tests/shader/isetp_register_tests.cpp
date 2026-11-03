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

// One synthetic ISETP register-form word with every admitted field settable.
// The defaults are the admitted profile: not-equal combined by AND with the
// PT source predicate, unsigned, PT instruction predicate.
std::uint64_t isetp_word(std::uint64_t dest_a = 3, std::uint64_t dest_b = 6,
                         std::uint64_t lhs = 4, std::uint64_t rhs = 9,
                         std::uint64_t source = 7, std::uint64_t neg = 0,
                         std::uint64_t extended = 0,
                         std::uint64_t bool_op = 0,
                         std::uint64_t is_signed = 0,
                         std::uint64_t compare = 5) {
    return (std::uint64_t{0x5b60} << 48) | (std::uint64_t{7} << 16) |
           (dest_b << 0) | (dest_a << 3) | (lhs << 8) | (rhs << 20) |
           (source << 39) | (neg << 42) | (extended << 43) |
           (bool_op << 45) | (is_signed << 48) | (compare << 49);
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
    ARMIGO_CHECK(error->code ==
                 mx::DecodeError::Code::UnsupportedModifier);
    ARMIGO_CHECK(error->family == Family::IntegerSetPredicateRegister);
    ARMIGO_CHECK(error->field == field);
    ARMIGO_CHECK_EQ(error->value, value);
    ARMIGO_CHECK_EQ(error->expected, expected);
    ARMIGO_CHECK_EQ(error->pc, 8u);
    return true;
}

}  // namespace

ARMIGO_TEST(isetp_register, decodes_admitted_profile_and_preserves_fields) {
    const std::uint64_t encoded = isetp_word();
    const auto result = decode(encoded);
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return;
    }
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == Family::IntegerSetPredicateRegister);
    ARMIGO_CHECK_EQ(instruction.pc, 8u);
    ARMIGO_CHECK_EQ(instruction.original_word, encoded);
    ARMIGO_CHECK(instruction.predicate.index == 7);
    ARMIGO_CHECK(!instruction.predicate.inverted);
    const auto* value =
        std::get_if<mx::IntegerSetPredicate>(&instruction.payload);
    ARMIGO_CHECK(value != nullptr);
    if (value == nullptr) {
        return;
    }
    ARMIGO_CHECK(value->destination_a == 3);
    ARMIGO_CHECK(value->destination_b == 6);
    ARMIGO_CHECK(value->lhs.index == 4);
    ARMIGO_CHECK(value->rhs.index == 9);
    ARMIGO_CHECK(value->source.index == 7);
    ARMIGO_CHECK(!value->source.inverted);
    ARMIGO_CHECK(!value->signed_compare);
}

ARMIGO_TEST(isetp_register,
            preserves_endpoint_aliasing_pt_and_zero_operands_and_signed) {
    const auto check = [](const mx::DecodeResult& result, unsigned dest_a,
                          unsigned dest_b, unsigned lhs, unsigned rhs,
                          bool signed_flag) {
        const auto* program = std::get_if<mx::DecodedProgram>(&result);
        ARMIGO_CHECK(program != nullptr);
        if (program == nullptr) {
            return;
        }
        const auto* value = std::get_if<mx::IntegerSetPredicate>(
            &program->instructions.front().payload);
        ARMIGO_CHECK(value != nullptr);
        if (value == nullptr) {
            return;
        }
        ARMIGO_CHECK(unsigned(value->destination_a) == dest_a);
        ARMIGO_CHECK(unsigned(value->destination_b) == dest_b);
        ARMIGO_CHECK(unsigned(value->lhs.index) == lhs);
        ARMIGO_CHECK(unsigned(value->rhs.index) == rhs);
        ARMIGO_CHECK(value->lhs.is_zero() == (lhs == 255));
        ARMIGO_CHECK(value->rhs.is_zero() == (rhs == 255));
        ARMIGO_CHECK(value->signed_compare == signed_flag);
    };
    // Endpoint predicate destinations 0 and PT, and endpoint registers 0/RZ.
    check(decode(isetp_word(0, 0, 0, 0)), 0, 0, 0, 0, false);
    check(decode(isetp_word(7, 7, 255, 255)), 7, 7, 255, 255, false);
    // Destinations may alias each other and operands may use one register.
    check(decode(isetp_word(5, 5, 12, 12)), 5, 5, 12, 12, false);
    // Both values of the signed field are admitted and carried raw.
    check(decode(isetp_word(3, 6, 4, 9, 7, 0, 0, 0, 1)), 3, 6, 4, 9, true);
    check(decode(isetp_word(7, 0, 255, 9, 7, 0, 0, 0, 1)), 7, 0, 255, 9,
          true);
}

ARMIGO_TEST(isetp_register, rejects_every_reserved_bit_explicitly) {
    for (unsigned shift : {6u, 7u, 28u, 29u, 30u, 31u, 32u, 33u, 34u, 35u,
                           36u, 37u, 38u, 44u, 47u}) {
        ARMIGO_CHECK(expect_rejected(
            isetp_word() | (std::uint64_t{1} << shift),
            mx::DecodeField::ReservedBits, std::uint64_t{1} << shift, 0));
    }
}

ARMIGO_TEST(isetp_register, rejects_every_disallowed_modifier_explicitly) {
    for (std::uint64_t compare : {0u, 1u, 2u, 3u, 4u, 6u, 7u}) {
        expect_rejected(isetp_word(3, 6, 4, 9, 7, 0, 0, 0, 0, compare),
                        mx::DecodeField::Comparison, compare, 5);
    }
    for (std::uint64_t bool_op : {1u, 2u, 3u}) {
        expect_rejected(isetp_word(3, 6, 4, 9, 7, 0, 0, bool_op),
                        mx::DecodeField::BooleanOperation, bool_op, 0);
    }
    for (std::uint64_t source : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
        expect_rejected(isetp_word(3, 6, 4, 9, source),
                        mx::DecodeField::BooleanPredicate, source, 7);
    }
    expect_rejected(isetp_word(3, 6, 4, 9, 7, 1),
                    mx::DecodeField::BooleanPredicateInversion, 1, 0);
    expect_rejected(isetp_word(3, 6, 4, 9, 7, 0, 1),
                    mx::DecodeField::ExtendedMode, 1, 0);
}

ARMIGO_TEST(isetp_register, rejects_non_pt_instruction_predicates) {
    for (std::uint64_t pred : {0u, 1u, 6u, 8u, 15u}) {
        const std::uint64_t word =
            (isetp_word() & ~(std::uint64_t{15} << 16)) | (pred << 16);
        const auto result = decode(word);
        const auto* error = std::get_if<mx::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code ==
                         mx::DecodeError::Code::UnsupportedPredicate);
            ARMIGO_CHECK(error->family ==
                         Family::IntegerSetPredicateRegister);
            ARMIGO_CHECK(error->field ==
                         mx::DecodeField::InstructionPredicate);
        }
    }
}

ARMIGO_TEST(isetp_register, neighbors_are_never_admitted_by_the_registry) {
    // The constant-buffer and immediate ISETP forms and their high
    // neighbors belong to no registered family.
    for (std::uint16_t high : {0x4b60u, 0x4b6fu, 0x3660u, 0x5b70u}) {
        ARMIGO_CHECK(maxwell::detail::find_registered_family(
                         std::uint64_t{high} << 48) == nullptr);
    }
    for (std::uint16_t high : {0x4b60u, 0x3660u}) {
        const auto result = decode((std::uint64_t{high} << 48) |
                                   (std::uint64_t{7} << 16) |
                                   (std::uint64_t{9} << 20));
        const auto* error = std::get_if<mx::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnknownOpcode);
        }
    }
    // The admitted pattern's own low nibble variants match the one entry.
    const auto families = maxwell::detail::registered_families();
    const maxwell::detail::FamilyEntry* entry = nullptr;
    for (const maxwell::detail::FamilyEntry& registered : families) {
        if (registered.family == Family::IntegerSetPredicateRegister) {
            entry = &registered;
        }
    }
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(entry->mask, 0xfff0u);
    ARMIGO_CHECK_EQ(entry->value, 0x5b60u);
    ARMIGO_CHECK(entry->decode != nullptr);
    ARMIGO_CHECK(maxwell::detail::find_registered_family(
                     std::uint64_t{0x5b6f} << 48) == entry);
}

ARMIGO_TEST(isetp_register, lowering_still_rejects_the_family) {
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
    append(Family::IntegerSetPredicateRegister,
           mx::IntegerSetPredicate{3, 6, {4}, {9}, {7, false}, false});
    append(Family::Exit, mx::Exit{15});
    const auto lowered = gx::lower(program);
    const auto* error = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::InvalidInstruction);
    }
}

}  // namespace armigo::shader::test
