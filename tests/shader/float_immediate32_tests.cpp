// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic tests for the 32-bit-immediate float multiply and add families
// (FMUL32I and FADD32I). Every expected result word below is authored from
// nearest-even rounding under flush-to-zero, with saturation applied only
// after the arithmetic; no oracle recomputes
// any result word.

#include "test_support.h"

#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>
#include <armigo/shader/maxwell_forward_program.h>

#include "maxwell_families.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

namespace mx = maxwell;
namespace gx = graphics;
namespace gm = graphics_msl;
using Family = mx::OpcodeFamily;
using Stage = shader_input::Stage;

std::uint64_t set(std::uint64_t word, unsigned shift, unsigned width,
                  std::uint64_t value) {
    const std::uint64_t field_mask =
        ((std::uint64_t{1} << width) - 1) << shift;
    return (word & ~field_mask) | ((value << shift) & field_mask);
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value,
         unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        bytes[at + i] = std::uint8_t(value >> (8 * i));
    }
}

// One synthetic FMUL32I word (window 0xff00/0x1e00): destination 0..7,
// source A 8..15, predicate 16..19, immediate 20..51, condition code 52,
// FMZ 53..54, saturation 55. Defaults are the admitted profile.
std::uint64_t fmul32i_word(std::uint64_t destination = 3,
                           std::uint64_t source_a = 2,
                           std::uint32_t immediate = 0x3f800000u,
                           std::uint64_t fmz = 1, std::uint64_t saturate = 0,
                           std::uint64_t predicate = 7) {
    return (std::uint64_t{0x1e00} << 48) | (predicate << 16) | destination |
           (source_a << 8) | (std::uint64_t{immediate} << 20) |
           (fmz << 53) | (saturate << 55);
}

// One synthetic FADD32I word (window 0xfc00/0x0800): destination 0..7,
// source A 8..15, predicate 16..19, immediate 20..51, condition code 52,
// negate-B 53, absolute-A 54, flush-to-zero 55, negate-A 56, absolute-B 57.
// Defaults are the admitted profile.
std::uint64_t fadd32i_word(std::uint64_t destination = 3,
                           std::uint64_t source_a = 2,
                           std::uint32_t immediate = 0x40400000u,
                           bool negate_a = false, bool absolute_a = false,
                           bool negate_b = false, bool absolute_b = false,
                           std::uint64_t ftz = 1,
                           std::uint64_t predicate = 7) {
    return (std::uint64_t{0x0800} << 48) | (predicate << 16) | destination |
           (source_a << 8) | (std::uint64_t{immediate} << 20) |
           (std::uint64_t{negate_b} << 53) |
           (std::uint64_t{absolute_a} << 54) | (ftz << 55) |
           (std::uint64_t{negate_a} << 56) |
           (std::uint64_t{absolute_b} << 57);
}

struct Container {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

// Synthetic straight-line vertex program: the supplied words beginning at
// pc 8, then an unconditional PT EXIT.
Container container(std::vector<std::uint64_t> instructions) {
    instructions.push_back((std::uint64_t{0xe300} << 48) |
                           (std::uint64_t{7} << 16) | 15);
    const std::size_t groups = (instructions.size() + 2) / 3;
    Container data{std::vector<std::uint8_t>(0x718),
                   std::vector<std::uint8_t>(0x80 + groups * 32)};
    put(data.control, 0, 0x98761234, 4);
    put(data.control, 4, 1, 4);
    put(data.control, 8, 14, 4);
    put(data.control, 0x20, data.control.size(), 4);
    put(data.control, 0x700, data.code.size(), 4);
    put(data.control, 0x704, data.code.size(), 4);
    put(data.control, 0x708, 0x30, 4);
    put(data.control, 0x714, 0, 4);
    put(data.code, 0, 0x12345678, 4);
    put(data.code, 0x30,
        shader_input::sph_stage_type_for(Stage::Vertex) << 10, 4);
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const std::size_t offset =
            0x80 + (index / 3) * 32 + 8 + (index % 3) * 8;
        put(data.code, std::uint32_t(offset), instructions[index], 8);
    }
    return data;
}

mx::DecodeResult decode(std::uint64_t instruction) {
    const auto data = container({instruction});
    return mx::decode({data.control.data(), data.control.size()},
                      {data.code.data(), data.code.size()});
}

mx::ForwardDecodeResult decode_forward(std::vector<std::uint64_t> instructions) {
    const auto data = container(std::move(instructions));
    return mx::decode_forward_program(
        {data.control.data(), data.control.size()},
        {data.code.data(), data.code.size()});
}

const mx::FloatMultiplyImmediate32* admitted_fmul(
    const mx::DecodeResult& result, std::uint64_t encoded) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return nullptr;
    }
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == Family::FloatMultiplyImmediate32);
    ARMIGO_CHECK_EQ(instruction.pc, 8u);
    ARMIGO_CHECK_EQ(instruction.original_word, encoded);
    ARMIGO_CHECK(instruction.predicate.index == 7);
    ARMIGO_CHECK(!instruction.predicate.inverted);
    return std::get_if<mx::FloatMultiplyImmediate32>(&instruction.payload);
}

const mx::FloatAddImmediate32* admitted_fadd(const mx::DecodeResult& result,
                                             std::uint64_t encoded) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return nullptr;
    }
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == Family::FloatAddImmediate32);
    ARMIGO_CHECK_EQ(instruction.pc, 8u);
    ARMIGO_CHECK_EQ(instruction.original_word, encoded);
    ARMIGO_CHECK(instruction.predicate.index == 7);
    ARMIGO_CHECK(!instruction.predicate.inverted);
    return std::get_if<mx::FloatAddImmediate32>(&instruction.payload);
}

bool expect_rejected(std::uint64_t instruction,
                     mx::DecodeError::Code code, Family family,
                     mx::DecodeField field, std::uint64_t value,
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

// Fragment programs expose register zero's lanes as the color output, so a
// result in register zero reads back as output component 0.
mx::InstructionPayload fmul32i_payload(unsigned destination,
                                       unsigned source_a,
                                       std::uint32_t immediate,
                                       bool saturate = false) {
    return mx::FloatMultiplyImmediate32{{std::uint8_t(destination)},
                                        {std::uint8_t(source_a)}, immediate,
                                        mx::FlushMode::FlushToZero, saturate};
}

mx::InstructionPayload fadd32i_payload(unsigned destination,
                                       unsigned source_a,
                                       std::uint32_t immediate,
                                       bool negate_a = false,
                                       bool absolute_a = false,
                                       bool negate_b = false,
                                       bool absolute_b = false) {
    return mx::FloatAddImmediate32{{std::uint8_t(destination)},
                                   {std::uint8_t(source_a)}, immediate,
                                   negate_a, absolute_a, negate_b,
                                   absolute_b};
}

mx::DecodedProgram program(
    std::vector<std::pair<Family, mx::InstructionPayload>> body,
    std::vector<std::pair<unsigned, std::uint32_t>> moves = {},
    std::uint8_t color_mask = 1) {
    mx::DecodedProgram p;
    p.stage = Stage::Fragment;
    const auto header = shader_input::sph_stage_type_for(Stage::Fragment) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        p.sph[i] = std::byte(header >> (i * 8));
    }
    // Each declared color lane reads the next consecutive register, so a
    // full four-lane target also requires registers one to three written.
    p.sph[0x48] = std::byte{color_mask};
    const auto append = [&p](Family family, mx::InstructionPayload payload) {
        const auto i = p.instructions.size();
        p.instructions.push_back(
            {std::uint32_t((i / 3) * 32 + (i % 3 + 1) * 8), 0, family,
             {7, false}, std::move(payload)});
    };
    for (const auto& [reg, bits] : moves) {
        append(Family::Mov32Immediate,
               mx::Mov32Immediate{{std::uint8_t(reg)}, 15, bits});
    }
    for (auto& [family, payload] : body) {
        append(family, std::move(payload));
    }
    append(Family::Exit, mx::Exit{15});
    return p;
}

std::uint32_t evaluate(const mx::DecodedProgram& p) {
    const auto lowered = gx::lower(p);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        const auto& failure = std::get<gx::Error>(lowered);
        record_failure(__FILE__, __LINE__,
                       std::string("lower failed: ") + failure.message() +
                           " pc=" + std::to_string(failure.pc));
        return 0;
    }
    const auto result = gx::evaluate(std::get<gx::Program>(lowered), {});
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated != nullptr);
    if (evaluated == nullptr) {
        return 0;
    }
    for (const auto& output : evaluated->outputs) {
        if (output.component == 0) {
            return output.bits;
        }
    }
    ARMIGO_CHECK(false);
    return 0;
}

void expect_error(const mx::DecodedProgram& p, gx::Error::Code expected) {
    const auto lowered = gx::lower(p);
    const auto* e = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(e != nullptr);
    if (e != nullptr) {
        ARMIGO_CHECK(e->code == expected);
    }
}

std::optional<std::size_t> find_literal(std::span<const gx::Node> nodes,
                                        std::uint32_t bits) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (const auto* literal = std::get_if<gx::Literal>(&nodes[i].operation)) {
            if (literal->bits == bits) {
                return i;
            }
        }
    }
    return std::nullopt;
}

}  // namespace

ARMIGO_TEST(float_immediate32, fmul32i_decodes_the_admitted_profile) {
    const std::uint64_t encoded = fmul32i_word();
    const auto result = decode(encoded);
    const auto* value = admitted_fmul(result, encoded);
    ARMIGO_CHECK(value != nullptr);
    if (value == nullptr) {
        return;
    }
    ARMIGO_CHECK(value->destination.index == 3);
    ARMIGO_CHECK(value->source_a.index == 2);
    ARMIGO_CHECK_EQ(value->immediate_bits, 0x3f800000u);
    ARMIGO_CHECK(value->flush == mx::FlushMode::FlushToZero);
    ARMIGO_CHECK(!value->saturate);
    // Saturation, register zero, aliasing, and immediate bit patterns
    // (negative, subnormal, infinity, all-ones) are preserved verbatim.
    const std::uint64_t saturated = fmul32i_word(1, 255, 0x7f800000u, 1, 1);
    const auto saturated_result = decode(saturated);
    const auto* saturating = admitted_fmul(saturated_result, saturated);
    ARMIGO_CHECK(saturating != nullptr);
    if (saturating != nullptr) {
        ARMIGO_CHECK(saturating->destination.index == 1);
        ARMIGO_CHECK(saturating->source_a.is_zero());
        ARMIGO_CHECK_EQ(saturating->immediate_bits, 0x7f800000u);
        ARMIGO_CHECK(saturating->saturate);
    }
    for (const std::uint32_t immediate :
         {0x00000000u, 0xbf800000u, 0x00000001u, 0x7f800000u, 0xffffffffu}) {
        const std::uint64_t word = fmul32i_word(5, 6, immediate);
        const auto immediate_result = decode(word);
        const auto* decoded = admitted_fmul(immediate_result, word);
        ARMIGO_CHECK(decoded != nullptr);
        if (decoded != nullptr) {
            ARMIGO_CHECK_EQ(decoded->immediate_bits, immediate);
        }
    }
    const std::uint64_t aliased = fmul32i_word(4, 4);
    const auto aliased_result = decode(aliased);
    const auto* aliased_value = admitted_fmul(aliased_result, aliased);
    ARMIGO_CHECK(aliased_value != nullptr);
    if (aliased_value != nullptr) {
        ARMIGO_CHECK(aliased_value->destination.index == 4);
        ARMIGO_CHECK(aliased_value->source_a.index == 4);
    }
}

ARMIGO_TEST(float_immediate32, fadd32i_decodes_the_admitted_profile) {
    const std::uint64_t encoded = fadd32i_word();
    const auto result = decode(encoded);
    const auto* value = admitted_fadd(result, encoded);
    ARMIGO_CHECK(value != nullptr);
    if (value == nullptr) {
        return;
    }
    ARMIGO_CHECK(value->destination.index == 3);
    ARMIGO_CHECK(value->source_a.index == 2);
    ARMIGO_CHECK_EQ(value->immediate_bits, 0x40400000u);
    ARMIGO_CHECK(!value->negate_a);
    ARMIGO_CHECK(!value->absolute_a);
    ARMIGO_CHECK(!value->negate_b);
    ARMIGO_CHECK(!value->absolute_b);
    // Each sign or absolute modifier maps to its own payload flag: negate-A
    // is bit 56, absolute-A 54, negate-B 53, and absolute-B 57.
    struct Case {
        bool negate_a, absolute_a, negate_b, absolute_b;
    };
    for (const auto [negate_a, absolute_a, negate_b, absolute_b] :
         {std::array{true, false, false, false},
          std::array{false, true, false, false},
          std::array{false, false, true, false},
          std::array{false, false, false, true},
          std::array{true, true, true, true}}) {
        const std::uint64_t word =
            fadd32i_word(1, 255, 0x00000001u, negate_a, absolute_a, negate_b,
                         absolute_b);
        const auto modifier_result = decode(word);
        const auto* decoded = admitted_fadd(modifier_result, word);
        ARMIGO_CHECK(decoded != nullptr);
        if (decoded == nullptr) {
            continue;
        }
        ARMIGO_CHECK(decoded->negate_a == negate_a);
        ARMIGO_CHECK(decoded->absolute_a == absolute_a);
        ARMIGO_CHECK(decoded->negate_b == negate_b);
        ARMIGO_CHECK(decoded->absolute_b == absolute_b);
        ARMIGO_CHECK(decoded->source_a.is_zero());
        ARMIGO_CHECK_EQ(decoded->immediate_bits, 0x00000001u);
    }
}

ARMIGO_TEST(float_immediate32, fmul32i_rejects_unadmitted_modifiers) {
    using Code = mx::DecodeError::Code;
    using Field = mx::DecodeField;
    const auto word = fmul32i_word();
    expect_rejected(set(word, 52, 1, 1), Code::UnsupportedModifier,
                    Family::FloatMultiplyImmediate32,
                    Field::ConditionCode, 1, 0);
    for (const std::uint64_t fmz : {0u, 2u, 3u}) {
        expect_rejected(set(word, 53, 2, fmz), Code::UnsupportedModifier,
                        Family::FloatMultiplyImmediate32, Field::FlushMode,
                        fmz, 1);
    }
    // Bits 48..51 are the top four immediate bits, not reserved: they are
    // admitted and preserved inside the immediate. Fixed opcode bits are
    // not modifiers: flipping one leaves the window.
    expect_rejected(word ^ (std::uint64_t{1} << 62), Code::UnknownOpcode,
                    Family::Unknown, Field::None, 0, 0);
    expect_rejected(word ^ (std::uint64_t{1} << 57), Code::UnknownOpcode,
                    Family::Unknown, Field::None, 0, 0);
    // The instruction predicate gate stays central and strict.
    expect_rejected(fmul32i_word(3, 2, 0x3f800000u, 1, 0, 6),
                    Code::UnsupportedPredicate,
                    Family::FloatMultiplyImmediate32,
                    Field::InstructionPredicate, 6, 7);
    expect_rejected(fmul32i_word(3, 2, 0x3f800000u, 1, 0, 15),
                    Code::UnsupportedPredicate,
                    Family::FloatMultiplyImmediate32,
                    Field::InstructionPredicate, 15, 7);
}

ARMIGO_TEST(float_immediate32, fadd32i_rejects_unadmitted_modifiers) {
    using Code = mx::DecodeError::Code;
    using Field = mx::DecodeField;
    const auto word = fadd32i_word();
    expect_rejected(set(word, 52, 1, 1), Code::UnsupportedModifier,
                    Family::FloatAddImmediate32, Field::ConditionCode, 1, 0);
    // Flush-to-zero is a one-bit modifier; zero is the only rejected value.
    expect_rejected(set(word, 55, 1, 0), Code::UnsupportedModifier,
                    Family::FloatAddImmediate32, Field::FlushMode, 0, 1);
    // Bits 48..51 are the top four immediate bits, not reserved: they are
    // admitted and preserved inside the immediate. Fixed opcode bits are
    // not modifiers: flipping one leaves the window.
    expect_rejected(word ^ (std::uint64_t{1} << 62), Code::UnknownOpcode,
                    Family::Unknown, Field::None, 0, 0);
    expect_rejected(word ^ (std::uint64_t{1} << 59), Code::UnknownOpcode,
                    Family::Unknown, Field::None, 0, 0);
    expect_rejected(fadd32i_word(3, 2, 0x40400000u, false, false, false,
                                 false, 1, 6),
                    Code::UnsupportedPredicate,
                    Family::FloatAddImmediate32,
                    Field::InstructionPredicate, 6, 7);
    expect_rejected(fadd32i_word(3, 2, 0x40400000u, false, false, false,
                                 false, 1, 15),
                    Code::UnsupportedPredicate,
                    Family::FloatAddImmediate32,
                    Field::InstructionPredicate, 15, 7);
}

ARMIGO_TEST(float_immediate32, registry_windows_stay_disjoint) {
    const mx::detail::FamilyEntry* fmul_entry = nullptr;
    const mx::detail::FamilyEntry* fadd_entry = nullptr;
    for (const mx::detail::FamilyEntry& entry :
         mx::detail::registered_families()) {
        if (entry.family == Family::FloatMultiplyImmediate32) {
            fmul_entry = &entry;
        }
        if (entry.family == Family::FloatAddImmediate32) {
            fadd_entry = &entry;
        }
    }
    ARMIGO_CHECK(fmul_entry != nullptr && fadd_entry != nullptr);
    if (fmul_entry == nullptr || fadd_entry == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(fmul_entry->mask, std::uint16_t{0xff00});
    ARMIGO_CHECK_EQ(fmul_entry->value, std::uint16_t{0x1e00});
    ARMIGO_CHECK_EQ(fadd_entry->mask, std::uint16_t{0xfc00});
    ARMIGO_CHECK_EQ(fadd_entry->value, std::uint16_t{0x0800});
    ARMIGO_CHECK(mx::detail::find_registered_family(
                     std::uint64_t{0x1e55} << 48) == fmul_entry);
    ARMIGO_CHECK(mx::detail::find_registered_family(
                     std::uint64_t{0x0a00} << 48) == fadd_entry);
    // The short-immediate word-and window (0xfc00/0x0400) does not claim
    // FADD32I words, and neither window overlaps the legacy immediate
    // families (0xfef8/0x3858 and 0xfef8/0x3868), which keep decoding.
    const auto* word_and = mx::detail::find_registered_family(
        std::uint64_t{0x04ab} << 48);
    ARMIGO_CHECK(word_and != nullptr);
    if (word_and != nullptr) {
        ARMIGO_CHECK(word_and->family == Family::WordAndImmediate);
    }
    const std::uint64_t legacy_add =
        (std::uint64_t{0x3858} << 48) | (std::uint64_t{7} << 16) | 3 |
        (std::uint64_t{2} << 8) | (std::uint64_t{0x40000} << 20) |
        (std::uint64_t{1} << 44);
    const auto add_result = decode(legacy_add);
    const auto* add_program = std::get_if<mx::DecodedProgram>(&add_result);
    ARMIGO_CHECK(add_program != nullptr);
    if (add_program != nullptr) {
        ARMIGO_CHECK(add_program->instructions.front().family ==
                     Family::FloatAddImmediate);
        const auto* add_value =
            std::get_if<mx::FloatAdd>(&add_program->instructions.front().payload);
        ARMIGO_CHECK(add_value != nullptr);
        if (add_value != nullptr) {
            ARMIGO_CHECK(std::holds_alternative<mx::ShortFloatImmediate>(
                add_value->source_b));
        }
    }
    const std::uint64_t legacy_mul =
        (std::uint64_t{0x3868} << 48) | (std::uint64_t{7} << 16) | 3 |
        (std::uint64_t{2} << 8) | (std::uint64_t{0x3f800} << 20) |
        (std::uint64_t{1} << 44);
    const auto mul_result = decode(legacy_mul);
    const auto* mul_program = std::get_if<mx::DecodedProgram>(&mul_result);
    ARMIGO_CHECK(mul_program != nullptr);
    if (mul_program != nullptr) {
        ARMIGO_CHECK(mul_program->instructions.front().family ==
                     Family::FloatMultiplyImmediate);
        const auto* mul_value = std::get_if<mx::FloatMultiply>(
            &mul_program->instructions.front().payload);
        ARMIGO_CHECK(mul_value != nullptr);
        if (mul_value != nullptr) {
            ARMIGO_CHECK(std::holds_alternative<mx::ShortFloatImmediate>(
                mul_value->source_b));
        }
    }
}

ARMIGO_TEST(float_immediate32, lowering_fmul32i_exposes_literal_and_saturate) {
    const auto lowered = gx::lower(program(
        {{Family::FloatMultiplyImmediate32, fmul32i_payload(0, 1, 0x40400000u)},
         {Family::FloatMultiplyImmediate32,
          fmul32i_payload(3, 1, 0x3f800000u, true)}},
        {{1, 0x40000000u}}));
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        return;
    }
    const auto nodes = std::get<gx::Program>(lowered).nodes();
    // The immediate becomes its own literal node, distinct from the move
    // literal; each multiply reads the register before writing.
    const auto immediate = find_literal(nodes, 0x40400000u);
    const auto saturated_immediate = find_literal(nodes, 0x3f800000u);
    ARMIGO_CHECK(immediate.has_value() && saturated_immediate.has_value());
    if (!immediate.has_value() || !saturated_immediate.has_value()) {
        return;
    }
    std::optional<std::size_t> multiply, saturated_multiply;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (const auto* arithmetic =
                std::get_if<gx::FloatArithmetic>(&nodes[i].operation)) {
            if (arithmetic->operation != gx::Arithmetic::MultiplyNearest) {
                continue;
            }
            ARMIGO_CHECK(!arithmetic->addend.has_value());
            ARMIGO_CHECK(!arithmetic->negate_lhs);
            if (arithmetic->rhs.index ==
                static_cast<std::uint32_t>(*immediate)) {
                ARMIGO_CHECK(!multiply.has_value());
                multiply = i;
            }
            if (arithmetic->rhs.index ==
                static_cast<std::uint32_t>(*saturated_immediate)) {
                saturated_multiply = i;
            }
        }
    }
    // The unsaturated multiply writes the destination directly: exactly one
    // saturate node exists, and it saturates the other product.
    ARMIGO_CHECK(multiply.has_value());
    ARMIGO_CHECK(saturated_multiply.has_value());
    if (!multiply.has_value() || !saturated_multiply.has_value()) {
        return;
    }
    ARMIGO_CHECK(*multiply < *saturated_multiply);
    std::optional<std::size_t> saturate;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (const auto* saturation =
                std::get_if<gx::Saturate>(&nodes[i].operation)) {
            ARMIGO_CHECK(!saturate.has_value());
            saturate = i;
            ARMIGO_CHECK_EQ(saturation->operand.index,
                            static_cast<std::uint32_t>(*saturated_multiply));
        }
    }
    ARMIGO_CHECK(saturate.has_value());
    if (saturate.has_value()) {
        ARMIGO_CHECK(*saturated_multiply < *saturate);
    }
}

ARMIGO_TEST(float_immediate32, lowering_fadd32i_folds_signs_into_the_literal) {
    // Plain and negate-A forms keep the immediate bits verbatim and drive
    // the add node's checked negation of the register operand.
    for (const bool negate_a : {false, true}) {
        const auto lowered = gx::lower(program(
            {{Family::FloatAddImmediate32,
              fadd32i_payload(0, 1, 0x40400000u, negate_a)}},
            {{1, 0x3f800000u}}));
        ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
        if (!std::holds_alternative<gx::Program>(lowered)) {
            continue;
        }
        const auto nodes = std::get<gx::Program>(lowered).nodes();
        const auto immediate = find_literal(nodes, 0x40400000u);
        ARMIGO_CHECK(immediate.has_value());
        if (!immediate.has_value()) {
            continue;
        }
        bool found = false;
        for (const auto& node : nodes) {
            const auto* arithmetic =
                std::get_if<gx::FloatArithmetic>(&node.operation);
            if (arithmetic == nullptr ||
                arithmetic->operation != gx::Arithmetic::Add) {
                continue;
            }
            ARMIGO_CHECK(!arithmetic->addend.has_value());
            ARMIGO_CHECK(arithmetic->negate_lhs == negate_a);
            ARMIGO_CHECK_EQ(arithmetic->rhs.index,
                            static_cast<std::uint32_t>(*immediate));
            found = true;
        }
        ARMIGO_CHECK(found);
    }
    // Negate-B flips the literal sign bit, absolute-B clears it, and both
    // modifiers together negate the absolute value (absolute first).
    struct SignCase {
        std::uint32_t immediate, expected;
        bool negate_b, absolute_b;
    };
    for (const auto& c : std::array<SignCase, 4>{{
             {0x40400000u, 0xc0400000u, true, false},
             {0xc0400000u, 0x40400000u, false, true},
             {0x40400000u, 0xc0400000u, true, true},
             {0xc0400000u, 0x40400000u, true, false},
         }}) {
        const auto lowered = gx::lower(program(
            {{Family::FloatAddImmediate32,
              fadd32i_payload(0, 1, c.immediate, false, false, c.negate_b,
                              c.absolute_b)}},
            {{1, 0x3f800000u}}));
        ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
        if (!std::holds_alternative<gx::Program>(lowered)) {
            continue;
        }
        const auto& nodes = std::get<gx::Program>(lowered).nodes();
        const auto transformed = find_literal(nodes, c.expected);
        ARMIGO_CHECK(transformed.has_value());
    }
    // Absolute A clears the register operand's sign before the add node's
    // negate (the instruction's absolute-then-negate order): |-1| + 3 = 4 and
    // -|-1| + 3 = 2. A forged payload keeps the invalid-instruction error.
    ARMIGO_CHECK_EQ(evaluate(program({{Family::FloatAddImmediate32,
                                       fadd32i_payload(0, 1, 0x40400000u, false, true)}},
                                     {{1, 0xbf800000u}})),
                    0x40800000u);
    ARMIGO_CHECK_EQ(evaluate(program({{Family::FloatAddImmediate32,
                                       fadd32i_payload(0, 1, 0x40400000u, true, true)}},
                                     {{1, 0xbf800000u}})),
                    0x40000000u);
    expect_error(program({{Family::FloatAddImmediate32,
                           mx::MovConstant{{0}, {5, 0}, 15}}}),
                 gx::Error::Code::InvalidInstruction);
}

// Authored exact-bit results: the multiply rounds nearest-even under
// flush-to-zero, subnormal immediates flush, and saturation clamps after
// the multiply.
ARMIGO_TEST(float_immediate32, evaluates_fmul32i) {
    struct Case {
        std::uint32_t a, immediate, expected;
        bool saturate;
    };
    const std::array<Case, 9> cases{{
        {0x40000000u, 0x40000000u, 0x40800000u, false},  // 2*2 = 4
        {0x3f000000u, 0xbf800000u, 0xbf000000u, false},  // 0.5*-1 = -0.5
        {0x3f800000u, 0x00000001u, 0x00000000u, false},  // subnormal flushes: 1*0 = +0
        {0x7f800000u, 0x00000001u, 0x7fc00000u, false},  // inf times the flushed zero is NaN
        {0x40000000u, 0x7f800000u, 0x7f800000u, false},  // 2*inf = inf
        {0x40000000u, 0x3f800000u, 0x3f800000u, true},   // 2*1 stays 1
        {0x40000000u, 0x40000000u, 0x3f800000u, true},   // 4 clamps to one
        {0xc0800000u, 0x40000000u, 0x00000000u, true},   // -8 clamps to +0
        {0x3f800000u, 0x40000000u, 0x40000000u, false},  // 1*2 = 2, unsaturated
    }};
    for (const auto& c : cases) {
        ARMIGO_CHECK_EQ(evaluate(program(
                            {{Family::FloatMultiplyImmediate32,
                              fmul32i_payload(0, 1, c.immediate, c.saturate)}},
                            {{1, c.a}})),
                        c.expected);
    }
    // A register-zero destination discards the result without clobbering
    // the color output.
    ARMIGO_CHECK_EQ(evaluate(program(
                            {{Family::FloatMultiplyImmediate32,
                              fmul32i_payload(255, 1, 0x40000000u)}},
                            {{0, 0x3f800000u}, {1, 0x40000000u}})),
                        0x3f800000u);
}

// Authored exact-bit results: the add rounds nearest-even under
// flush-to-zero, subnormal immediates flush, and sign modifiers fold into
// the literal before the operation.
ARMIGO_TEST(float_immediate32, evaluates_fadd32i) {
    struct Case {
        std::uint32_t a, immediate, expected;
        bool negate_a, negate_b, absolute_b;
    };
    const std::array<Case, 7> cases{{
        {0x3f800000u, 0x40000000u, 0x40400000u, false, false, false},  // 1+2 = 3
        {0x3f800000u, 0x00000001u, 0x3f800000u, false, false, false},  // subnormal flushes: 1+0
        {0x3f800000u, 0x40000000u, 0x3f800000u, true, false, false},   // -1+2 = 1
        {0x3f800000u, 0x40000000u, 0xbf800000u, false, true, false},   // 1+(-2) = -1
        {0x3f800000u, 0xc0000000u, 0x40400000u, false, false, true},   // 1+|-2| = 3
        {0x3f800000u, 0x40000000u, 0xbf800000u, false, true, true},    // 1+(-|2|) = -1
        {0xc0000000u, 0x00000001u, 0xc0000000u, false, false, false},  // -2+flush = -2
    }};
    for (const auto& c : cases) {
        ARMIGO_CHECK_EQ(evaluate(program(
                            {{Family::FloatAddImmediate32,
                              fadd32i_payload(0, 1, c.immediate, c.negate_a,
                                              false, c.negate_b,
                                              c.absolute_b)}},
                            {{1, c.a}})),
                        c.expected);
    }
}

ARMIGO_TEST(float_immediate32, forward_decoder_admits_both_with_guards) {
    const auto result = decode_forward(
        {fmul32i_word(3, 2, 0x3f800000u, 1, 0, 3),
         fadd32i_word(4, 5, 0x40400000u, true, false, false, false, 1, 11)});
    const auto* program = std::get_if<mx::ForwardProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(program->instructions.size(), 3u);
    const mx::Instruction* fmul = nullptr;
    const mx::Instruction* fadd = nullptr;
    unsigned exits = 0;
    for (const auto& node : program->instructions) {
        const auto* instruction = std::get_if<mx::Instruction>(&node);
        if (instruction == nullptr) {
            continue;
        }
        if (instruction->pc == 8) {
            fmul = instruction;
        }
        if (instruction->pc == 0x10) {
            fadd = instruction;
        }
        if (instruction->family == Family::Exit) {
            ++exits;
        }
    }
    ARMIGO_CHECK(fmul != nullptr && fadd != nullptr);
    if (fmul == nullptr || fadd == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(exits, 1u);
    // Guard raw 3 admits the multiply with predicate 3; guard raw 11
    // inverts predicate 3 on the add. Both preserve their words verbatim.
    ARMIGO_CHECK(fmul->family == Family::FloatMultiplyImmediate32);
    ARMIGO_CHECK_EQ(fmul->predicate.index, std::uint8_t{3});
    ARMIGO_CHECK_EQ(fmul->predicate.inverted, false);
    const auto* fmul_value =
        std::get_if<mx::FloatMultiplyImmediate32>(&fmul->payload);
    ARMIGO_CHECK(fmul_value != nullptr);
    if (fmul_value != nullptr) {
        ARMIGO_CHECK_EQ(fmul_value->immediate_bits, 0x3f800000u);
        ARMIGO_CHECK(fmul_value->saturate == false);
    }
    ARMIGO_CHECK(fadd->family == Family::FloatAddImmediate32);
    ARMIGO_CHECK_EQ(fadd->predicate.index, std::uint8_t{3});
    ARMIGO_CHECK_EQ(fadd->predicate.inverted, true);
    const auto* fadd_value =
        std::get_if<mx::FloatAddImmediate32>(&fadd->payload);
    ARMIGO_CHECK(fadd_value != nullptr);
    if (fadd_value != nullptr) {
        ARMIGO_CHECK(fadd_value->negate_a);
    }
}

// MSL emission renders the shared nearest-even helpers around both results
// and adds no stage resources.
ARMIGO_TEST(float_immediate32, emits_shared_helpers_and_literals) {
    const auto lowered = gx::lower(program(
        {{Family::FloatMultiplyImmediate32,
          fmul32i_payload(0, 1, 0x40400000u, true)},
         {Family::FloatAddImmediate32, fadd32i_payload(0, 0, 0x40400000u)}},
        {{1, 0x40000000u}, {2, 0u}, {3, 0u}}, 0xf));
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        return;
    }
    const auto emitted = gm::emit(std::get<gx::Program>(lowered), {});
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) {
        record_failure(__FILE__, __LINE__,
                       std::string("emit failed: ") +
                           std::get<gm::EmitError>(emitted).message());
        return;
    }
    ARMIGO_CHECK(stage->source.find("= armigo_mul_rn_ftz(") !=
                 std::string::npos);
    ARMIGO_CHECK(stage->source.find("armigo_saturate_float32(") !=
                 std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_add_rn_ftz(") !=
                 std::string::npos);
    // The 32-bit immediate appears as its own literal word.
    ARMIGO_CHECK(stage->source.find("0x40400000u") != std::string::npos);
    ARMIGO_CHECK(stage->constant_buffers.empty());
}

}  // namespace armigo::shader::test
