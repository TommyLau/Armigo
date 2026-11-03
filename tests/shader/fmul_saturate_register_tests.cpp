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

std::uint64_t set(std::uint64_t word, unsigned shift, unsigned width,
                  std::uint64_t value) {
    const std::uint64_t field_mask =
        ((std::uint64_t{1} << width) - 1) << shift;
    return (word & ~field_mask) | ((value << shift) & field_mask);
}

// One synthetic saturating register-FMUL word with every decoded field
// settable. The defaults are the admitted profile: nearest-even rounding,
// FTZ flush, no scale, condition code, or negation, a PT instruction
// predicate, and the saturation modifier set.
std::uint64_t fmul_saturate_word(std::uint64_t destination = 3,
                                 std::uint64_t source_a = 2,
                                 std::uint64_t source_b = 9,
                                 std::uint64_t rounding = 0,
                                 std::uint64_t predicate = 7) {
    return (std::uint64_t{0x5c6c} << 48) | (predicate << 16) | destination |
           (source_a << 8) | (source_b << 20) | (rounding << 39) |
           (std::uint64_t{1} << 44);
}

std::uint64_t fmul_register_word(std::uint64_t destination = 3,
                                 std::uint64_t source_a = 2,
                                 std::uint64_t source_b = 9,
                                 std::uint64_t rounding = 0) {
    return set(set(set(set(set((std::uint64_t{0x5c68} << 48) |
                                   (std::uint64_t{7} << 16),
                               0, 8, destination),
                          8, 8, source_a),
                     20, 8, source_b),
                39, 2, rounding),
           44, 2, 1);
}

std::uint64_t fmul_constant_saturate_word(std::uint64_t offset = 0x31,
                                          std::uint64_t binding = 4) {
    return set(set(set(set(set((std::uint64_t{0x4c68} << 48) |
                                   (std::uint64_t{7} << 16),
                               0, 8, 3),
                          8, 8, 2),
                     20, 14, offset),
                34, 5, binding),
           44, 2, 1) |
           (std::uint64_t{1} << 50);
}

std::uint64_t fmul_immediate_saturate_word() {
    return set(set(set(set((std::uint64_t{0x3868} << 48) |
                               (std::uint64_t{7} << 16),
                           0, 8, 3),
                      8, 8, 2),
                 20, 19, 0x3f800),
            44, 2, 1) |
           (std::uint64_t{1} << 50);
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

const mx::FloatMultiply* admitted(const mx::DecodeResult& result,
                                  std::uint64_t encoded) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return nullptr;
    }
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == Family::FloatMultiplySaturateRegister);
    ARMIGO_CHECK_EQ(instruction.pc, 8u);
    ARMIGO_CHECK_EQ(instruction.original_word, encoded);
    ARMIGO_CHECK(instruction.predicate.index == 7);
    ARMIGO_CHECK(!instruction.predicate.inverted);
    return std::get_if<mx::FloatMultiply>(&instruction.payload);
}

bool expect_rejected(std::uint64_t instruction,
                     mx::DecodeError::Code code,
                     Family family,
                     mx::DecodeField field,
                     std::uint64_t value,
                     std::uint64_t expected) {
    const auto result = decode(instruction);
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) {
        return false;
    }
    ARMIGO_CHECK(error->code == code);
    ARMIGO_CHECK(error->family == family);
    if (field != mx::DecodeField::None) {
        ARMIGO_CHECK(error->field == field);
        ARMIGO_CHECK_EQ(error->value, value);
        ARMIGO_CHECK_EQ(error->expected, expected);
    }
    ARMIGO_CHECK_EQ(error->pc, 8u);
    return true;
}

}  // namespace

ARMIGO_TEST(fmul_saturate_register, decodes_admitted_profile_and_preserves_fields) {
    const std::uint64_t encoded = fmul_saturate_word();
    const auto result = decode(encoded);
    const auto* value = admitted(result, encoded);
    ARMIGO_CHECK(value != nullptr);
    if (value == nullptr) {
        return;
    }
    ARMIGO_CHECK(value->destination.index == 3);
    ARMIGO_CHECK(value->source_a.index == 2);
    const auto* source_b = std::get_if<mx::RegisterOperand>(&value->source_b);
    ARMIGO_CHECK(source_b != nullptr);
    if (source_b != nullptr) {
        ARMIGO_CHECK(source_b->index == 9);
    }
    ARMIGO_CHECK(value->rounding == mx::RoundingMode::NearestEven);
    ARMIGO_CHECK(value->flush == mx::FlushMode::FlushToZero);
}

ARMIGO_TEST(fmul_saturate_register, admits_zero_and_aliased_registers) {
    for (const auto [destination, source_a, source_b] :
         {std::array{5u, 255u, 9u}, std::array{5u, 5u, 5u},
          std::array{3u, 255u, 255u}, std::array{255u, 2u, 9u}}) {
        const std::uint64_t encoded =
            fmul_saturate_word(destination, source_a, source_b);
        const auto result = decode(encoded);
        const auto* value = admitted(result, encoded);
        ARMIGO_CHECK(value != nullptr);
        if (value == nullptr) {
            continue;
        }
        ARMIGO_CHECK(value->destination.index == destination);
        ARMIGO_CHECK(value->source_a.index == source_a);
        const auto* source_b_operand =
            std::get_if<mx::RegisterOperand>(&value->source_b);
        ARMIGO_CHECK(source_b_operand != nullptr);
        if (source_b_operand != nullptr) {
            ARMIGO_CHECK(source_b_operand->index == source_b);
        }
    }
}

ARMIGO_TEST(fmul_saturate_register, rejects_non_nearest_even_rounding) {
    for (const std::uint64_t rounding : {1u, 2u, 3u}) {
        expect_rejected(fmul_saturate_word(3, 2, 9, rounding),
                        mx::DecodeError::Code::UnsupportedModifier,
                        Family::FloatMultiplySaturateRegister,
                        mx::DecodeField::Rounding, rounding, 0);
    }
}

ARMIGO_TEST(fmul_saturate_register,
            rejects_scale_flush_condition_negation_and_reserved_bits) {
    using Code = mx::DecodeError::Code;
    using Field = mx::DecodeField;
    const auto word = fmul_saturate_word();
    const auto bit = [&word](unsigned position) {
        return word | (std::uint64_t{1} << position);
    };
    expect_rejected(set(word, 41, 3, 1), Code::UnsupportedModifier,
                    Family::FloatMultiplySaturateRegister, Field::Scale, 1, 0);
    expect_rejected(set(word, 44, 2, 0), Code::UnsupportedModifier,
                    Family::FloatMultiplySaturateRegister, Field::FlushMode,
                    0, 1);
    expect_rejected(set(word, 44, 2, 2), Code::UnsupportedModifier,
                    Family::FloatMultiplySaturateRegister, Field::FlushMode,
                    2, 1);
    expect_rejected(set(word, 47, 1, 1), Code::UnsupportedModifier,
                    Family::FloatMultiplySaturateRegister,
                    Field::ConditionCode, 1, 0);
    expect_rejected(set(word, 48, 1, 1), Code::UnsupportedModifier,
                    Family::FloatMultiplySaturateRegister, Field::NegateB, 1,
                    0);
    for (const unsigned reserved : {46u, 49u, 28u, 38u}) {
        expect_rejected(bit(reserved), Code::UnsupportedModifier,
                        Family::FloatMultiplySaturateRegister,
                        Field::ReservedBits, std::uint64_t{1} << reserved, 0);
    }
    // The instruction predicate gate stays central and strict: a non-PT
    // index and an inverted PT (raw field 15) are both rejected.
    expect_rejected(fmul_saturate_word(3, 2, 9, 0, 6),
                    Code::UnsupportedPredicate,
                    Family::FloatMultiplySaturateRegister,
                    Field::InstructionPredicate, 6, 7);
    expect_rejected(fmul_saturate_word(3, 2, 9, 0, 15),
                    Code::UnsupportedPredicate,
                    Family::FloatMultiplySaturateRegister,
                    Field::InstructionPredicate, 15, 7);
    // Opcode bits are not modifiers: flipping a fixed opcode bit leaves the
    // family windows entirely.
    expect_rejected(set(fmul_saturate_word(), 48, 16, 0x5c88),
                    Code::UnknownOpcode, Family::Unknown, Field::None, 0, 0);
    expect_rejected(fmul_saturate_word() | (std::uint64_t{1} << 52),
                    Code::UnknownOpcode, Family::Unknown, Field::None, 0, 0);
}

ARMIGO_TEST(fmul_saturate_register, legacy_unsaturated_paths_stay_disjoint) {
    using Code = mx::DecodeError::Code;
    using Field = mx::DecodeField;
    // The unsaturated register form keeps its legacy family, including the
    // retained round-toward-zero profile.
    for (const std::uint64_t rounding : {0u, 3u}) {
        const auto result = decode(fmul_register_word(3, 2, 9, rounding));
        const auto* program = std::get_if<mx::DecodedProgram>(&result);
        ARMIGO_CHECK(program != nullptr);
        if (program == nullptr) {
            continue;
        }
        const auto& instruction = program->instructions.front();
        ARMIGO_CHECK(instruction.family == Family::FloatMultiplyRegister);
        const auto* value =
            std::get_if<mx::FloatMultiply>(&instruction.payload);
        ARMIGO_CHECK(value != nullptr);
        if (value != nullptr) {
            ARMIGO_CHECK(value->rounding == (rounding == 0
                                                 ? mx::RoundingMode::NearestEven
                                                 : mx::RoundingMode::Zero));
        }
    }
    // Saturation in the constant and immediate windows stays rejected by
    // their legacy families; only the register window is deferred.
    expect_rejected(fmul_constant_saturate_word(), Code::UnsupportedModifier,
                    Family::FloatMultiplyConstant, Field::Saturate, 1, 0);
    expect_rejected(fmul_immediate_saturate_word(), Code::UnsupportedModifier,
                    Family::FloatMultiplyImmediate, Field::Saturate, 1, 0);
}

ARMIGO_TEST(fmul_saturate_register, registry_entry_matches_the_deferred_window) {
    const mx::detail::FamilyEntry* entry = nullptr;
    for (const mx::detail::FamilyEntry& candidate :
         mx::detail::registered_families()) {
        if (candidate.family == Family::FloatMultiplySaturateRegister) {
            entry = &candidate;
        }
    }
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(entry->mask, std::uint16_t{0xfffc});
    ARMIGO_CHECK_EQ(entry->value, std::uint16_t{0x5c6c});
    ARMIGO_CHECK(entry->decode != nullptr);
    ARMIGO_CHECK(mx::detail::find_registered_family(
                     std::uint64_t{0x5c6f} << 48) == entry);
}

ARMIGO_TEST(fmul_saturate_register, lowering_accepts_the_family) {
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
    append(Family::Mov32Immediate, mx::Mov32Immediate{{2}, 15, 0});
    append(Family::Mov32Immediate, mx::Mov32Immediate{{9}, 15, 0});
    // The registered lowerer composes the multiply with the typed saturate
    // node; detailed lowering behavior lives in the dedicated lower tests.
    append(Family::FloatMultiplySaturateRegister,
           mx::FloatMultiply{{3}, {1}, mx::RegisterOperand{2},
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero});
    for (unsigned c = 28; c < 32; ++c) {
        append(Family::AttributeStore,
               mx::AttributeAccess{{3}, {255}, {255},
                                   std::uint16_t(c * 4), 0, false});
    }
    append(Family::Exit, mx::Exit{15});
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (std::holds_alternative<gx::Error>(lowered)) {
        record_failure(__FILE__, __LINE__,
                       std::string("lower failed: ") +
                           std::get<gx::Error>(lowered).message());
    }
}

}  // namespace armigo::shader::test
