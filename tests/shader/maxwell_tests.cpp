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

#include <armigo/shader/maxwell.h>

#include "family_registry_validation.h"
#include "maxwell_families.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

using namespace maxwell;
using shader_input::ByteSpan;
using shader_input::Stage;

constexpr std::uint64_t mask(unsigned width, unsigned shift) {
    return ((std::uint64_t{1} << width) - 1) << shift;
}

std::uint64_t set(std::uint64_t word, unsigned shift, unsigned width,
                  std::uint64_t value) {
    const std::uint64_t field_mask = mask(width, shift);
    return (word & ~field_mask) | ((value << shift) & field_mask);
}

std::uint64_t instruction(std::uint16_t high) {
    return (static_cast<std::uint64_t>(high) << 48) |
           (std::uint64_t{7} << 16);
}

std::uint64_t mov(std::uint8_t destination, std::uint32_t immediate) {
    std::uint64_t word = instruction(0x0100);
    word = set(word, 0, 8, destination);
    word = set(word, 12, 4, 15);
    return set(word, 20, 32, immediate);
}

std::uint64_t mov_register(std::uint8_t destination, std::uint8_t source,
                           std::uint8_t write_mask = 15) {
    std::uint64_t word = instruction(0x5c98);
    word = set(word, 0, 8, destination);
    word = set(word, 20, 8, source);
    return set(word, 39, 4, write_mask);
}

std::uint64_t attribute(std::uint16_t opcode, std::uint8_t value,
                        std::uint16_t byte_offset) {
    std::uint64_t word = instruction(opcode);
    word = set(word, 0, 8, value);
    word = set(word, 8, 8, 255);
    word = set(word, 20, 10, byte_offset);
    return set(word, 39, 8, 255);
}

std::uint64_t fset(std::uint8_t destination, std::uint8_t source_a,
                   std::uint8_t source_b, std::uint8_t comparison,
                   bool negate_a = false) {
    std::uint64_t word = instruction(0x5800);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = set(word, 20, 8, source_b);
    word = set(word, 39, 3, 7);
    word = set(word, 43, 1, negate_a);
    word = set(word, 48, 4, comparison);
    word = set(word, 52, 1, 1);
    return set(word, 55, 1, 1);
}

std::uint64_t float_binary(std::uint16_t opcode, std::uint8_t destination,
                           std::uint8_t source_a, std::uint8_t source_b,
                           std::uint8_t rounding = 0) {
    std::uint64_t word = instruction(opcode);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = set(word, 20, 8, source_b);
    word = set(word, 39, 2, rounding);
    if (opcode == 0x5c58 || opcode == 0x4c58) {
        word = set(word, 44, 1, 1);
    } else {
        word = set(word, 44, 2, 1);
    }
    return word;
}

std::uint64_t fadd_immediate(std::uint8_t destination, std::uint8_t source_a,
                             std::uint32_t field_bits, bool negative) {
    std::uint64_t word = instruction(0x3858);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = set(word, 20, 19, field_bits);
    word = set(word, 44, 1, 1);
    return set(word, 56, 1, negative);
}

std::uint64_t fmul_immediate(std::uint8_t destination, std::uint8_t source_a,
                             std::uint32_t field_bits, std::uint8_t rounding,
                             bool negative) {
    std::uint64_t word = instruction(0x3868);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = set(word, 20, 19, field_bits);
    word = set(word, 39, 2, rounding);
    word = set(word, 44, 2, 1);
    return set(word, 56, 1, negative);
}

std::uint64_t cbuf(std::uint64_t word, std::uint8_t binding,
                   std::uint16_t word_offset) {
    word = set(word, 20, 14, word_offset);
    return set(word, 34, 5, binding);
}

std::uint64_t ffma_immediate(std::uint8_t destination,
                             std::uint8_t source_a,
                             std::uint32_t short_bits,
                             std::uint8_t source_c, bool negative) {
    std::uint64_t word = instruction(0x3280);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = set(word, 20, 19, short_bits);
    word = set(word, 39, 8, source_c);
    word = set(word, 53, 2, 1);
    return set(word, 56, 1, negative);
}

std::uint64_t ffma_constant(std::uint8_t destination,
                            std::uint8_t source_a, std::uint8_t binding,
                            std::uint16_t word_offset,
                            std::uint8_t source_c) {
    std::uint64_t word = instruction(0x4980);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source_a);
    word = cbuf(word, binding, word_offset);
    word = set(word, 39, 8, source_c);
    return set(word, 53, 2, 1);
}

std::uint64_t mufu(std::uint8_t destination, std::uint8_t source) {
    std::uint64_t word = instruction(0x5080);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, source);
    return set(word, 20, 4, 4);
}

std::uint64_t ipa(std::uint8_t destination, std::uint8_t attribute_component) {
    std::uint64_t word = instruction(0xe000);
    word = set(word, 0, 8, destination);
    word = set(word, 8, 8, 255);
    word = set(word, 20, 8, 255);
    word = set(word, 30, 8, attribute_component);
    return set(word, 39, 11, 0x7ff);
}

std::uint64_t texs() {
    std::uint64_t word = instruction(0xd000);
    word = set(word, 0, 8, 28);
    word = set(word, 8, 8, 32);
    word = set(word, 20, 8, 33);
    word = set(word, 28, 8, 30);
    word = set(word, 36, 13, 0x37);
    word = set(word, 49, 1, 1);
    word = set(word, 50, 3, 4);
    word = set(word, 53, 4, 1);
    return set(word, 59, 1, 1);
}

std::uint64_t exit_word() {
    return set(instruction(0xe300), 0, 5, 15);
}

void put32(std::vector<std::uint8_t>& bytes, std::size_t offset,
           std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] =
            static_cast<std::uint8_t>(value >> (index * 8));
    }
}

void put64(std::vector<std::uint8_t>& bytes, std::size_t offset,
           std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        bytes[offset + index] =
            static_cast<std::uint8_t>(value >> (index * 8));
    }
}

struct Fixture {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

Fixture fixture(Stage stage, const std::vector<std::uint64_t>& words) {
    const std::size_t groups = (words.size() + 2) / 3;
    const std::uint32_t c1_offset =
        static_cast<std::uint32_t>(0x80 + groups * 32);
    Fixture result{std::vector<std::uint8_t>(0x718),
                   std::vector<std::uint8_t>(c1_offset)};
    put32(result.control, 0, 0x98761234);
    put32(result.control, 4, 1);
    put32(result.control, 8, 14);
    put32(result.control, 0x20, static_cast<std::uint32_t>(result.control.size()));
    put32(result.control, 0x700, c1_offset);
    put32(result.control, 0x704, c1_offset);
    put32(result.control, 0x708, 0x30);
    put32(result.control, 0x714, static_cast<std::uint32_t>(stage));
    put32(result.code, 0, 0x12345678);
    for (std::size_t index = 0; index < shader_input::format::sph_size;
         ++index) {
        result.code[0x30 + index] =
            static_cast<std::uint8_t>(0xa0u + (index * 7u) % 0x50u);
    }
    put32(result.code, 0x30,
          shader_input::sph_stage_type_for(stage) <<
              shader_input::format::sph_stage_shift);
    for (std::size_t group = 0; group < groups; ++group) {
        put64(result.code, 0x80 + group * 32,
              0x1122334455667700ull + group);
    }
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::size_t offset =
            0x80 + (index / 3) * 32 + 8 + (index % 3) * 8;
        put64(result.code, offset, words[index]);
    }
    return result;
}

ByteSpan span(const std::vector<std::uint8_t>& bytes) {
    return {bytes.data(), bytes.size()};
}

const DecodedProgram* accepted(const DecodeResult& result) {
    const auto* program = std::get_if<DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    return program;
}

std::optional<DecodeError> rejected(
    const DecodeResult& result, DecodeError::Code code,
    OpcodeFamily family = OpcodeFamily::Unknown) {
    const auto* error = std::get_if<DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == code);
        if (family != OpcodeFamily::Unknown) {
            ARMIGO_CHECK(error->family == family);
        }
        ARMIGO_CHECK(error->message()[0] != '\0');
        return *error;
    }
    return std::nullopt;
}

DecodeResult decode_one(std::uint64_t word, Stage stage = Stage::Vertex) {
    const Fixture data = fixture(stage, {word, exit_word()});
    return decode(span(data.control), span(data.code));
}

}  // namespace

ARMIGO_TEST(maxwell, decodes_the_bounded_family_set_and_skips_schedules) {
    std::uint64_t fadd_reg = float_binary(0x5c58, 9, 9, 10);
    fadd_reg = set(fadd_reg, 48, 1, 1);
    std::uint64_t fadd_cbuf = float_binary(0x4c58, 11, 12, 0);
    fadd_cbuf = cbuf(fadd_cbuf, 5, 0x123);
    const std::uint64_t fmul_reg = float_binary(0x5c68, 13, 14, 15, 3);
    std::uint64_t fmul_cbuf = float_binary(0x4c68, 16, 17, 0);
    fmul_cbuf = cbuf(fmul_cbuf, 2, 0x55);
    const std::vector<std::uint64_t> words = {
        mov(17, 0xdeadbeef),
        attribute(0xefd8, 6, 0x24),
        attribute(0xeff0, 7, 0x34),
        fset(3, 4, 255, 1),
        fset(5, 6, 8, 4, true),
        fadd_reg,
        fadd_cbuf,
        fmul_reg,
        fmul_cbuf,
        fmul_immediate(40, 41, 0x15555, 3, true),
        ffma_immediate(18, 19, 0x15555, 20, true),
        ffma_constant(21, 22, 6, 0x33, 23),
        mufu(24, 25),
        ipa(26, 0x2a),
        texs(),
        exit_word(),
    };
    Fixture data = fixture(Stage::Fragment, words);
    const DecodeResult result = decode(span(data.control), span(data.code));
    const DecodedProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK(program->stage == Stage::Fragment);
    ARMIGO_CHECK_EQ(program->instructions.size(), words.size());
    const std::array expected_families = {
        OpcodeFamily::Mov32Immediate,
        OpcodeFamily::AttributeLoad,
        OpcodeFamily::AttributeStore,
        OpcodeFamily::FloatSetRegister,
        OpcodeFamily::FloatSetRegister,
        OpcodeFamily::FloatAddRegister,
        OpcodeFamily::FloatAddConstant,
        OpcodeFamily::FloatMultiplyRegister,
        OpcodeFamily::FloatMultiplyConstant,
        OpcodeFamily::FloatMultiplyImmediate,
        OpcodeFamily::FloatFmaImmediate,
        OpcodeFamily::FloatFmaConstantRegister,
        OpcodeFamily::MultiFunction,
        OpcodeFamily::InterpolateAttribute,
        OpcodeFamily::TextureSample,
        OpcodeFamily::Exit,
    };
    for (std::size_t index = 0; index < words.size(); ++index) {
        const Instruction& decoded = program->instructions[index];
        ARMIGO_CHECK(decoded.family == expected_families[index]);
        ARMIGO_CHECK_EQ(decoded.original_word, words[index]);
        ARMIGO_CHECK_EQ(decoded.pc,
                        static_cast<std::uint32_t>((index / 3) * 32 + 8 +
                                                   (index % 3) * 8));
        ARMIGO_CHECK(decoded.predicate.index == 7);
        ARMIGO_CHECK(!decoded.predicate.inverted);
    }
    const auto& moved =
        std::get<Mov32Immediate>(program->instructions[0].payload);
    ARMIGO_CHECK(moved.destination.index == 17);
    ARMIGO_CHECK_EQ(moved.immediate_bits, 0xdeadbeefu);
    const auto& loaded =
        std::get<AttributeAccess>(program->instructions[1].payload);
    ARMIGO_CHECK_EQ(loaded.byte_offset, 0x24u);
    ARMIGO_CHECK(loaded.index.is_zero());
    ARMIGO_CHECK(loaded.vertex.is_zero());
    ARMIGO_CHECK(!loaded.patch);
    const auto& set_lt =
        std::get<FloatSetRegister>(program->instructions[3].payload);
    ARMIGO_CHECK(set_lt.source_b.is_zero());
    ARMIGO_CHECK(set_lt.comparison == OrderedComparison::LessThan);
    const auto& set_gt =
        std::get<FloatSetRegister>(program->instructions[4].payload);
    ARMIGO_CHECK(set_gt.comparison == OrderedComparison::GreaterThan);
    ARMIGO_CHECK(set_gt.negate_a);
    const auto& add_buffer =
        std::get<FloatAdd>(program->instructions[6].payload);
    const auto& add_cbuf =
        std::get<ConstantBufferOperand>(add_buffer.source_b);
    ARMIGO_CHECK(add_cbuf.binding == 5);
    ARMIGO_CHECK_EQ(add_cbuf.byte_offset, 0x48cu);
    const auto& multiply =
        std::get<FloatMultiply>(program->instructions[7].payload);
    ARMIGO_CHECK(multiply.rounding == RoundingMode::Zero);
    const auto& multiply_buffer = std::get<ConstantBufferOperand>(
        std::get<FloatMultiply>(program->instructions[8].payload).source_b);
    ARMIGO_CHECK(multiply_buffer.binding == 2);
    ARMIGO_CHECK_EQ(multiply_buffer.byte_offset, 0x154u);
    const auto& multiply_immediate = std::get<FloatMultiply>(
        program->instructions[9].payload);
    ARMIGO_CHECK(multiply_immediate.rounding == RoundingMode::Zero);
    ARMIGO_CHECK_EQ(std::get<ShortFloatImmediate>(multiply_immediate.source_b)
                        .raw_bits,
                    0x95555000u);
    const auto& immediate =
        std::get<ShortFloatImmediate>(
            std::get<FloatFma>(program->instructions[10].payload).source_b);
    ARMIGO_CHECK_EQ(immediate.raw_bits, 0x95555000u);
    const auto& fma_buffer = std::get<ConstantBufferOperand>(
        std::get<FloatFma>(program->instructions[11].payload).source_b);
    ARMIGO_CHECK(fma_buffer.binding == 6);
    ARMIGO_CHECK_EQ(fma_buffer.byte_offset, 0xccu);
    const auto& sample =
        std::get<TextureSample2D>(program->instructions[14].payload);
    ARMIGO_CHECK_EQ(sample.handle_byte_offset, 0xdcu);
    ARMIGO_CHECK(sample.no_dependency);
    const auto& interpolated =
        std::get<InterpolateAttribute>(program->instructions[13].payload);
    ARMIGO_CHECK(interpolated.encoding_control == 0x7ff);
    for (std::size_t index = 0; index < program->sph.size(); ++index) {
        ARMIGO_CHECK(std::to_integer<std::uint8_t>(program->sph[index]) ==
                     data.code[0x30 + index]);
    }
}

ARMIGO_TEST(maxwell, decodes_short_immediate_float_add_exact_bits) {
    struct Case {
        std::uint32_t field_bits;
        bool negative;
        std::uint32_t raw_bits;
    };
    const std::array cases = {
        Case{0x3f800, false, 0x3f800000u},  // +1.0
        Case{0x3f800, true, 0xbf800000u},   // -1.0
        Case{0x00000, false, 0x00000000u},  // +0.0
        Case{0x00000, true, 0x80000000u},   // -0.0
        Case{0x40200, false, 0x40200000u},  // +2.5
    };
    for (const Case& test_case : cases) {
        const DecodeResult result = decode_one(
            fadd_immediate(9, 10, test_case.field_bits, test_case.negative));
        const DecodedProgram* program = accepted(result);
        if (program == nullptr) {
            return;
        }
        ARMIGO_CHECK(program->instructions[0].family ==
                     OpcodeFamily::FloatAddImmediate);
        const auto& add =
            std::get<FloatAdd>(program->instructions[0].payload);
        ARMIGO_CHECK(add.destination.index == 9);
        ARMIGO_CHECK(add.source_a.index == 10);
        ARMIGO_CHECK(add.rounding == RoundingMode::NearestEven);
        ARMIGO_CHECK(add.flush == FlushMode::FlushToZero);
        ARMIGO_CHECK(!add.negate_a);
        const auto& immediate = std::get<ShortFloatImmediate>(add.source_b);
        ARMIGO_CHECK_EQ(immediate.raw_bits, test_case.raw_bits);
    }

    // Negating A is part of the existing FADD semantics and is independent of
    // the immediate's encoded sign.
    std::uint64_t negated_word = fadd_immediate(1, 2, 0x3f800, false);
    negated_word = set(negated_word, 48, 1, 1);
    const DecodeResult negated_result = decode_one(negated_word);
    const DecodedProgram* negated = accepted(negated_result);
    if (negated != nullptr) {
        const auto& add = std::get<FloatAdd>(negated->instructions[0].payload);
        ARMIGO_CHECK(add.negate_a);
        ARMIGO_CHECK_EQ(std::get<ShortFloatImmediate>(add.source_b).raw_bits,
                        0x3f800000u);
    }
}

ARMIGO_TEST(maxwell, decodes_short_immediate_float_multiply_exact_bits) {
    struct Case {
        std::uint32_t field_bits;
        bool negative;
        std::uint32_t raw_bits;
    };
    const std::array cases = {
        Case{0x3f800, false, 0x3f800000u},  // +1.0
        Case{0x3f800, true, 0xbf800000u},   // -1.0
        Case{0x00000, false, 0x00000000u},  // +0.0
        Case{0x00000, true, 0x80000000u},   // -0.0
        Case{0x40200, false, 0x40200000u},  // +2.5
        Case{0x00001, false, 0x00001000u},  // Smallest positive subnormal
        Case{0x7f800, false, 0x7f800000u},  // +Infinity
        Case{0x7fc00, false, 0x7fc00000u},  // Quiet NaN
        Case{0x7fc00, true, 0xffc00000u},   // Sign-preserved NaN payload
    };
    for (const Case& test_case : cases) {
        for (const std::uint8_t rounding : {std::uint8_t{0}, std::uint8_t{3}}) {
            const DecodeResult result = decode_one(fmul_immediate(
                9, 10, test_case.field_bits, rounding, test_case.negative));
            const DecodedProgram* program = accepted(result);
            if (program == nullptr) {
                return;
            }
            ARMIGO_CHECK(program->instructions[0].family ==
                         OpcodeFamily::FloatMultiplyImmediate);
            const auto& multiply =
                std::get<FloatMultiply>(program->instructions[0].payload);
            ARMIGO_CHECK(multiply.destination.index == 9);
            ARMIGO_CHECK(multiply.source_a.index == 10);
            ARMIGO_CHECK(multiply.rounding ==
                         (rounding == 0 ? RoundingMode::NearestEven
                                        : RoundingMode::Zero));
            ARMIGO_CHECK(multiply.flush == FlushMode::FlushToZero);
            const auto& immediate =
                std::get<ShortFloatImmediate>(multiply.source_b);
            ARMIGO_CHECK_EQ(immediate.raw_bits, test_case.raw_bits);
            ARMIGO_CHECK_EQ(immediate.raw_bits & 4095u, 0u);
        }
    }

    // RZ and the zero register are accepted as destination and multiplicand.
    const DecodeResult zero_registers =
        decode_one(fmul_immediate(255, 255, 0x3f800, 0, false));
    const DecodedProgram* zero = accepted(zero_registers);
    if (zero != nullptr) {
        const auto& multiply =
            std::get<FloatMultiply>(zero->instructions[0].payload);
        ARMIGO_CHECK(multiply.destination.is_zero());
        ARMIGO_CHECK(multiply.source_a.is_zero());
        ARMIGO_CHECK_EQ(std::get<ShortFloatImmediate>(multiply.source_b)
                            .raw_bits,
                        0x3f800000u);
    }

    // The register, buffer and immediate forms of the same family remain
    // distinct payload alternatives.
    Fixture data = fixture(Stage::Vertex,
                           {float_binary(0x5c68, 1, 2, 3),
                            cbuf(float_binary(0x4c68, 4, 5, 0), 3, 6),
                            fmul_immediate(7, 8, 0x40000, 0, false),
                            exit_word()});
    const DecodeResult mixed_result = decode(span(data.control), span(data.code));
    const DecodedProgram* mixed = accepted(mixed_result);
    if (mixed != nullptr) {
        ARMIGO_CHECK(mixed->instructions[0].family ==
                     OpcodeFamily::FloatMultiplyRegister);
        ARMIGO_CHECK(std::holds_alternative<RegisterOperand>(
            std::get<FloatMultiply>(mixed->instructions[0].payload).source_b));
        ARMIGO_CHECK(mixed->instructions[1].family ==
                     OpcodeFamily::FloatMultiplyConstant);
        ARMIGO_CHECK(std::holds_alternative<ConstantBufferOperand>(
            std::get<FloatMultiply>(mixed->instructions[1].payload).source_b));
        ARMIGO_CHECK(mixed->instructions[2].family ==
                     OpcodeFamily::FloatMultiplyImmediate);
        ARMIGO_CHECK(std::holds_alternative<ShortFloatImmediate>(
            std::get<FloatMultiply>(mixed->instructions[2].payload).source_b));
        ARMIGO_CHECK_EQ(std::get<ShortFloatImmediate>(
                            std::get<FloatMultiply>(
                                mixed->instructions[2].payload)
                                .source_b)
                            .raw_bits,
                        0x40000000u);
    }
}

ARMIGO_TEST(maxwell, metadata_is_owned_and_preserves_operand_overlap) {
    const std::uint64_t overlap = float_binary(0x5c58, 42, 42, 42);
    Fixture data = fixture(Stage::Vertex, {overlap, exit_word()});
    DecodeResult result = decode(span(data.control), span(data.code));
    const DecodedProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    const auto& add = std::get<FloatAdd>(program->instructions[0].payload);
    ARMIGO_CHECK(add.destination.index == 42);
    ARMIGO_CHECK(add.source_a.index == 42);
    ARMIGO_CHECK(std::get<RegisterOperand>(add.source_b).index == 42);
    const auto owned_word = program->instructions[0].original_word;
    const auto owned_sph = program->sph[17];
    std::fill(data.control.begin(), data.control.end(), 0);
    std::fill(data.code.begin(), data.code.end(), 0);
    ARMIGO_CHECK_EQ(program->instructions[0].original_word, owned_word);
    ARMIGO_CHECK(program->sph[17] == owned_sph);
}

ARMIGO_TEST(maxwell, rejects_input_stage_limits_and_missing_exit) {
    Fixture malformed = fixture(Stage::Vertex, {exit_word()});
    malformed.control[0] = 0;
    const DecodeResult malformed_result =
        decode(span(malformed.control), span(malformed.code));
    const std::optional<DecodeError> parse =
        rejected(malformed_result, DecodeError::Code::ShaderParse);
    if (parse.has_value()) {
        ARMIGO_CHECK(parse->parse_error.has_value());
    }

    Fixture compute = fixture(Stage::Compute, {exit_word()});
    // Rebuild the compute layout expected by the structural parser.
    put32(compute.control, 0x700, 8);
    put32(compute.control, 0x704, 8);
    put32(compute.control, 0x708, 0);
    compute.code.assign(8, 0);
    rejected(decode(span(compute.control), span(compute.code)),
             DecodeError::Code::UnsupportedStage);

    Fixture no_exit = fixture(Stage::Vertex, {mov(1, 2), mov(3, 4), mov(5, 6)});
    rejected(decode(span(no_exit.control), span(no_exit.code)),
             DecodeError::Code::MissingExit);

    Fixture limited = fixture(Stage::Vertex, {mov(1, 2), exit_word()});
    rejected(decode(span(limited.control), span(limited.code), {1}),
             DecodeError::Code::InstructionLimit);
}

ARMIGO_TEST(maxwell, rejects_reachable_predicates_flow_and_unknown_opcodes) {
    for (const std::uint8_t predicate : {std::uint8_t{6}, std::uint8_t{15}}) {
        std::uint64_t word = mov(1, 2);
        word = set(word, 16, 4, predicate);
        const std::optional<DecodeError> error =
            rejected(decode_one(word), DecodeError::Code::UnsupportedPredicate,
                     OpcodeFamily::Mov32Immediate);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->pc, 8u);
        }
    }
    rejected(decode_one(instruction(0xe240)),
             DecodeError::Code::UnsupportedControlFlow,
             OpcodeFamily::Branch);
    rejected(decode_one(instruction(0x7abc)), DecodeError::Code::UnknownOpcode);
    rejected(decode_one(set(exit_word(), 0, 5, 14)),
             DecodeError::Code::UnsupportedControlFlow, OpcodeFamily::Exit);
}

ARMIGO_TEST(maxwell, ignores_every_word_after_unconditional_exit) {
    const Fixture data = fixture(
        Stage::Vertex,
        {mov(4, 0x12345678), exit_word(), instruction(0xe240),
         instruction(0x7abc), 0xffffffffffffffffull});
    const DecodeResult result = decode(span(data.control), span(data.code));
    const DecodedProgram* program = accepted(result);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
        ARMIGO_CHECK(program->instructions.back().family == OpcodeFamily::Exit);
    }
}

ARMIGO_TEST(maxwell, rejects_unsupported_modifier_forms) {
    struct Case {
        std::uint64_t word;
        OpcodeFamily family;
        DecodeField field;
    };
    std::uint64_t add = float_binary(0x5c58, 1, 2, 3);
    std::uint64_t add_constant =
        cbuf(float_binary(0x4c58, 1, 2, 0), 4, 7);
    const std::uint64_t add_immediate = fadd_immediate(1, 2, 0x3f800, false);
    std::uint64_t multiply = float_binary(0x5c68, 1, 2, 3);
    const std::uint64_t multiply_immediate =
        fmul_immediate(1, 2, 0x3f800, 0, false);
    std::uint64_t fused = ffma_immediate(1, 2, 0x12345, 3, false);
    std::uint64_t interpolation = ipa(1, 7);
    std::uint64_t texture = texs();
    const std::vector<Case> cases = {
        {set(mov(1, 2), 12, 4, 1), OpcodeFamily::Mov32Immediate,
         DecodeField::WriteMask},
        {set(mov(1, 2), 8, 1, 1), OpcodeFamily::Mov32Immediate,
         DecodeField::ReservedBits},
        {set(attribute(0xefd8, 1, 0x20), 8, 8, 2),
         OpcodeFamily::AttributeLoad, DecodeField::IndexRegister},
        {set(attribute(0xefd8, 1, 0x20), 39, 8, 3),
         OpcodeFamily::AttributeLoad, DecodeField::VertexRegister},
        {set(attribute(0xefd8, 1, 0x20), 47, 2, 1),
         OpcodeFamily::AttributeLoad, DecodeField::AttributeWidth},
        {set(attribute(0xeff0, 1, 0x20), 31, 1, 1),
         OpcodeFamily::AttributeStore, DecodeField::Patch},
        {set(attribute(0xefd8, 1, 0x20), 30, 1, 1),
         OpcodeFamily::AttributeLoad, DecodeField::AddressMode},
        {set(attribute(0xefd8, 1, 0x20), 32, 1, 1),
         OpcodeFamily::AttributeLoad, DecodeField::AddressMode},
        {attribute(0xeff0, 1, 0x22), OpcodeFamily::AttributeStore,
         DecodeField::AttributeOffset},
        {set(fset(1, 2, 3, 1), 39, 3, 6),
         OpcodeFamily::FloatSetRegister, DecodeField::BooleanPredicate},
        {set(fset(1, 2, 3, 1), 42, 1, 1),
         OpcodeFamily::FloatSetRegister,
         DecodeField::BooleanPredicateInversion},
        {set(fset(1, 2, 3, 1), 44, 1, 1),
         OpcodeFamily::FloatSetRegister, DecodeField::AbsoluteB},
        {set(fset(1, 2, 3, 1), 45, 2, 1),
         OpcodeFamily::FloatSetRegister, DecodeField::BooleanOperation},
        {set(fset(1, 2, 3, 1), 47, 1, 1),
         OpcodeFamily::FloatSetRegister, DecodeField::ConditionCode},
        {set(fset(1, 2, 3, 1), 48, 4, 2),
         OpcodeFamily::FloatSetRegister, DecodeField::Comparison},
        {set(fset(1, 2, 3, 1), 52, 1, 0),
         OpcodeFamily::FloatSetRegister, DecodeField::BooleanFloat},
        {set(fset(1, 2, 3, 1), 53, 1, 1),
         OpcodeFamily::FloatSetRegister, DecodeField::NegateB},
        {set(fset(1, 2, 3, 1), 54, 1, 1),
         OpcodeFamily::FloatSetRegister, DecodeField::AbsoluteA},
        {set(fset(1, 2, 3, 1), 55, 1, 0),
         OpcodeFamily::FloatSetRegister, DecodeField::FlushMode},
        {set(add, 39, 2, 3), OpcodeFamily::FloatAddRegister,
         DecodeField::Rounding},
        {set(add, 44, 1, 0), OpcodeFamily::FloatAddRegister,
         DecodeField::FlushMode},
        {set(add, 45, 1, 1), OpcodeFamily::FloatAddRegister,
         DecodeField::NegateB},
        {set(add, 46, 1, 1), OpcodeFamily::FloatAddRegister,
         DecodeField::AbsoluteA},
        {set(add, 47, 1, 1), OpcodeFamily::FloatAddRegister,
         DecodeField::ConditionCode},
        {set(add, 49, 1, 1), OpcodeFamily::FloatAddRegister,
         DecodeField::AbsoluteB},
        {set(add, 50, 1, 1), OpcodeFamily::FloatAddRegister,
         DecodeField::Saturate},
        {set(add_constant, 48, 1, 1), OpcodeFamily::FloatAddConstant,
         DecodeField::NegateA},
        {set(add_immediate, 39, 2, 3), OpcodeFamily::FloatAddImmediate,
         DecodeField::Rounding},
        {set(add_immediate, 44, 1, 0), OpcodeFamily::FloatAddImmediate,
         DecodeField::FlushMode},
        {set(add_immediate, 45, 1, 1), OpcodeFamily::FloatAddImmediate,
         DecodeField::NegateB},
        {set(add_immediate, 46, 1, 1), OpcodeFamily::FloatAddImmediate,
         DecodeField::AbsoluteA},
        {set(add_immediate, 47, 1, 1), OpcodeFamily::FloatAddImmediate,
         DecodeField::ConditionCode},
        {set(add_immediate, 49, 1, 1), OpcodeFamily::FloatAddImmediate,
         DecodeField::AbsoluteB},
        {set(add_immediate, 50, 1, 1), OpcodeFamily::FloatAddImmediate,
         DecodeField::Saturate},
        {set(multiply, 39, 2, 1), OpcodeFamily::FloatMultiplyRegister,
         DecodeField::Rounding},
        {set(multiply, 41, 3, 1), OpcodeFamily::FloatMultiplyRegister,
         DecodeField::Scale},
        {set(multiply, 44, 2, 2), OpcodeFamily::FloatMultiplyRegister,
         DecodeField::FlushMode},
        {set(multiply, 47, 1, 1), OpcodeFamily::FloatMultiplyRegister,
         DecodeField::ConditionCode},
        {set(multiply, 48, 1, 1), OpcodeFamily::FloatMultiplyRegister,
         DecodeField::NegateB},
        // The register saturation window is the registered
        // FloatMultiplySaturateRegister family now; fmul_saturate_register
        // tests cover its admission and rejection profile.
        {set(multiply_immediate, 39, 2, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::Rounding},
        {set(multiply_immediate, 39, 2, 2),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::Rounding},
        {set(multiply_immediate, 41, 3, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::Scale},
        {set(multiply_immediate, 41, 3, 7),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::Scale},
        {set(multiply_immediate, 44, 2, 0),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::FlushMode},
        {set(multiply_immediate, 44, 2, 2),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::FlushMode},
        {set(multiply_immediate, 44, 2, 3),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::FlushMode},
        {set(multiply_immediate, 47, 1, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::ConditionCode},
        {set(multiply_immediate, 48, 1, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::NegateB},
        {set(multiply_immediate, 50, 1, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::Saturate},
        {set(multiply_immediate, 46, 1, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::ReservedBits},
        {set(multiply_immediate, 49, 1, 1),
         OpcodeFamily::FloatMultiplyImmediate, DecodeField::ReservedBits},
        {set(fused, 47, 1, 1), OpcodeFamily::FloatFmaImmediate,
         DecodeField::ConditionCode},
        {set(fused, 48, 1, 1), OpcodeFamily::FloatFmaImmediate,
         DecodeField::NegateB},
        {set(fused, 49, 1, 1), OpcodeFamily::FloatFmaImmediate,
         DecodeField::NegateC},
        {set(fused, 50, 1, 1), OpcodeFamily::FloatFmaImmediate,
         DecodeField::Saturate},
        {set(fused, 51, 2, 3), OpcodeFamily::FloatFmaImmediate,
         DecodeField::Rounding},
        {set(fused, 53, 2, 0), OpcodeFamily::FloatFmaImmediate,
         DecodeField::FlushMode},
        {set(mufu(1, 2), 20, 4, 5), OpcodeFamily::MultiFunction,
         DecodeField::Function},
        {set(mufu(1, 2), 46, 1, 1), OpcodeFamily::MultiFunction,
         DecodeField::AbsoluteA},
        {set(mufu(1, 2), 48, 1, 1), OpcodeFamily::MultiFunction,
         DecodeField::NegateA},
        {set(mufu(1, 2), 50, 1, 1), OpcodeFamily::MultiFunction,
         DecodeField::Saturate},
        {set(interpolation, 8, 8, 3),
         OpcodeFamily::InterpolateAttribute, DecodeField::IndexRegister},
        {set(interpolation, 20, 8, 4),
         OpcodeFamily::InterpolateAttribute,
         DecodeField::MultiplierRegister},
        {set(interpolation, 38, 1, 1),
         OpcodeFamily::InterpolateAttribute, DecodeField::Indexed},
        {set(interpolation, 51, 1, 1),
         OpcodeFamily::InterpolateAttribute, DecodeField::Saturate},
        {set(interpolation, 52, 2, 1),
         OpcodeFamily::InterpolateAttribute, DecodeField::SampleMode},
        {set(interpolation, 54, 2, 2),
         OpcodeFamily::InterpolateAttribute, DecodeField::InterpolationMode},
        {set(interpolation, 50, 1, 1),
         OpcodeFamily::InterpolateAttribute, DecodeField::ReservedBits},
        {set(texture, 59, 1, 0), OpcodeFamily::TextureSample,
         DecodeField::TexturePrecision},
        {set(texture, 53, 4, 2), OpcodeFamily::TextureSample,
         DecodeField::TextureEncoding},
        {set(texture, 50, 3, 3), OpcodeFamily::TextureSample,
         DecodeField::TextureSwizzle},
        {set(texture, 0, 8, 29), OpcodeFamily::TextureSample,
         DecodeField::TextureDestination},
        {set(texture, 28, 8, 255), OpcodeFamily::TextureSample,
         DecodeField::TextureDestination},
        {set(texture, 0, 8, 254), OpcodeFamily::TextureSample,
         DecodeField::TextureDestination},
        {set(texture, 28, 8, 254), OpcodeFamily::TextureSample,
         DecodeField::TextureDestination},
    };
    for (const Case& test_case : cases) {
        const std::optional<DecodeError> error =
            rejected(decode_one(test_case.word),
                     DecodeError::Code::UnsupportedModifier,
                     test_case.family);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == test_case.field);
            ARMIGO_CHECK_EQ(error->pc, 8u);
        }
    }

    for (std::uint16_t encoding = 0; encoding < 0x7ff; ++encoding) {
        const std::optional<DecodeError> error = rejected(
            decode_one(set(interpolation, 39, 11, encoding)),
            DecodeError::Code::UnsupportedModifier,
            OpcodeFamily::InterpolateAttribute);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field ==
                         DecodeField::InterpolationEncodingControl);
        }
    }

    const DecodeResult destination_a_boundary =
        decode_one(set(texture, 0, 8, 252));
    const DecodedProgram* accepted_a = accepted(destination_a_boundary);
    if (accepted_a != nullptr) {
        const auto& sample =
            std::get<TextureSample2D>(accepted_a->instructions[0].payload);
        ARMIGO_CHECK(sample.destination_a.index == 252);
    }
    const DecodeResult destination_b_boundary =
        decode_one(set(texture, 28, 8, 252));
    const DecodedProgram* accepted_b = accepted(destination_b_boundary);
    if (accepted_b != nullptr) {
        const auto& sample =
            std::get<TextureSample2D>(accepted_b->instructions[0].payload);
        ARMIGO_CHECK(sample.destination_b.index == 252);
    }

    const std::array fadd_unused_bits = {41u, 42u, 43u};
    for (const std::uint16_t opcode : {std::uint16_t{0x5c58},
                                      std::uint16_t{0x4c58}}) {
        for (const unsigned bit : fadd_unused_bits) {
            std::uint64_t word = float_binary(opcode, 8, 9, 10);
            if (opcode == 0x4c58) {
                word = cbuf(word, 3, 0x27);
            }
            const std::optional<DecodeError> error = rejected(
                decode_one(word | (std::uint64_t{1} << bit)),
                DecodeError::Code::UnsupportedModifier,
                opcode == 0x4c58 ? OpcodeFamily::FloatAddConstant
                                 : OpcodeFamily::FloatAddRegister);
            if (error.has_value()) {
                ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
            }
        }
    }

    const std::array fadd_immediate_unused_bits = {41u, 42u, 43u};
    for (const unsigned bit : fadd_immediate_unused_bits) {
        const std::optional<DecodeError> error = rejected(
            decode_one(add_immediate | (std::uint64_t{1} << bit)),
            DecodeError::Code::UnsupportedModifier,
            OpcodeFamily::FloatAddImmediate);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
        }
    }
    // Bit 55 is a fixed zero in the FADD immediate opcode pattern, unlike its
    // flush-mode role in the separate FADD32I encoding.
    rejected(decode_one(add_immediate | (std::uint64_t{1} << 55)),
             DecodeError::Code::UnknownOpcode);
    // The FMUL immediate pattern also fixes bit 55 to zero and bit 51 to one.
    // Flipping either opcode bit must not be treated as a modifier.
    rejected(decode_one(multiply_immediate | (std::uint64_t{1} << 55)),
             DecodeError::Code::UnknownOpcode);
    rejected(decode_one(set(multiply_immediate, 48, 16, 0x5c88)),
             DecodeError::Code::UnknownOpcode);

    const std::array fmul_unused_bits = {46u, 49u};
    for (const std::uint16_t opcode : {std::uint16_t{0x5c68},
                                      std::uint16_t{0x4c68}}) {
        std::uint64_t word = float_binary(opcode, 11, 12, 13);
        if (opcode == 0x4c68) {
            word = cbuf(word, 4, 0x31);
        }
        const OpcodeFamily family =
            opcode == 0x4c68 ? OpcodeFamily::FloatMultiplyConstant
                             : OpcodeFamily::FloatMultiplyRegister;
        for (const unsigned bit : fmul_unused_bits) {
            const std::optional<DecodeError> error = rejected(
                decode_one(word | (std::uint64_t{1} << bit)),
                DecodeError::Code::UnsupportedModifier, family);
            if (error.has_value()) {
                ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
            }
        }
        // The immediate sign bit is outside these opcode patterns, so it can
        // never turn a register or buffer form into the immediate form.
        rejected(decode_one(word | (std::uint64_t{1} << 56)),
                 DecodeError::Code::UnknownOpcode);
    }

    std::uint64_t invalid_cbuf = float_binary(0x4c58, 1, 2, 0);
    invalid_cbuf = cbuf(invalid_cbuf, 31, 4);
    const std::optional<DecodeError> binding = rejected(
        decode_one(invalid_cbuf), DecodeError::Code::UnsupportedModifier,
        OpcodeFamily::FloatAddConstant);
    if (binding.has_value()) {
        ARMIGO_CHECK(binding->field == DecodeField::ConstantBinding);
    }
}

ARMIGO_TEST(maxwell, decodes_register_move_endpoints_masks_and_zero) {
    struct Case {
        std::uint8_t destination;
        std::uint8_t source;
        std::uint8_t write_mask;
    };
    const std::array cases = {
        Case{0, 0, 15},   Case{254, 254, 15}, Case{255, 255, 1},
        Case{0, 254, 1},  Case{254, 0, 15},   Case{255, 0, 1},
    };
    for (const Case& test_case : cases) {
        const std::uint64_t word = mov_register(
            test_case.destination, test_case.source, test_case.write_mask);
        const DecodeResult result = decode_one(word);
        const DecodedProgram* program = accepted(result);
        if (program == nullptr) {
            return;
        }
        const Instruction& decoded = program->instructions[0];
        ARMIGO_CHECK(decoded.family == OpcodeFamily::MovRegister);
        ARMIGO_CHECK_EQ(decoded.original_word, word);
        const auto* move = std::get_if<MovRegister>(&decoded.payload);
        ARMIGO_CHECK(move != nullptr);
        if (move != nullptr) {
            ARMIGO_CHECK(move->destination.index == test_case.destination);
            ARMIGO_CHECK(move->source.index == test_case.source);
            ARMIGO_CHECK(move->write_mask == test_case.write_mask);
        }
    }
}

ARMIGO_TEST(maxwell, rejects_register_move_masks_outside_full_copy) {
    for (unsigned write_mask = 0; write_mask < 15; ++write_mask) {
        if (write_mask == 1) {
            continue;
        }
        const std::optional<DecodeError> error = rejected(
            decode_one(mov_register(3, 4, static_cast<std::uint8_t>(write_mask))),
            DecodeError::Code::UnsupportedModifier, OpcodeFamily::MovRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == DecodeField::WriteMask);
            ARMIGO_CHECK_EQ(error->value, write_mask);
            ARMIGO_CHECK_EQ(error->expected, 15u);
            ARMIGO_CHECK_EQ(error->pc, 8u);
        }
    }
}

ARMIGO_TEST(maxwell, rejects_register_move_unused_fields_predicates_and_neighbors) {
    for (const unsigned bit : {8u, 15u, 28u, 38u, 43u, 47u, 48u, 49u, 50u}) {
        const std::optional<DecodeError> error = rejected(
            decode_one(mov_register(1, 2) | (std::uint64_t{1} << bit)),
            DecodeError::Code::UnsupportedModifier, OpcodeFamily::MovRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
        }
    }
    for (const std::uint8_t predicate : {std::uint8_t{6}, std::uint8_t{15}}) {
        std::uint64_t word = mov_register(1, 2);
        word = set(word, 16, 4, predicate);
        const std::optional<DecodeError> error = rejected(
            decode_one(word), DecodeError::Code::UnsupportedPredicate,
            OpcodeFamily::MovRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == DecodeField::InstructionPredicate);
            ARMIGO_CHECK_EQ(error->value, predicate);
        }
    }
    // Neighbouring 16-bit patterns are not admitted by any move family.
    for (const std::uint16_t high : {std::uint16_t{0x5c88},
                                     std::uint16_t{0x5c8f},
                                     std::uint16_t{0x5c80},
                                     std::uint16_t{0x5d98}}) {
        rejected(decode_one(instruction(high)), DecodeError::Code::UnknownOpcode);
    }
    // The low three wildcard bits of the pattern are unused by this form.
    const std::optional<DecodeError> extension = rejected(
        decode_one(instruction(0x5c9f)),
        DecodeError::Code::UnsupportedModifier, OpcodeFamily::MovRegister);
    if (extension.has_value()) {
        ARMIGO_CHECK(extension->field == DecodeField::ReservedBits);
    }
}

ARMIGO_TEST(maxwell, decoder_registry_admits_only_move_families_today) {
    const std::span<const maxwell::detail::FamilyEntry> families =
        maxwell::detail::registered_families();
    ARMIGO_CHECK(!families.empty());
    const maxwell::detail::FamilyEntry* move_entry = nullptr;
    for (const maxwell::detail::FamilyEntry& registered : families) {
        if (registered.family == OpcodeFamily::MovRegister) {
            move_entry = &registered;
        }
    }
    ARMIGO_CHECK(move_entry != nullptr);
    if (move_entry == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(move_entry->mask, 0xfff8u);
    ARMIGO_CHECK_EQ(move_entry->value, 0x5c98u);
    ARMIGO_CHECK(move_entry->decode != nullptr);
    for (std::size_t i = 0; i < families.size(); ++i) {
        for (std::size_t j = i + 1; j < families.size(); ++j) {
            ARMIGO_CHECK(families[i].family != families[j].family);
        }
    }
    // Families the legacy classifier owns are never registered: the registry
    // may only extend recognition, never shadow a legacy pattern.
    ARMIGO_CHECK(maxwell::detail::find_registered_family(instruction(0x5c58)) ==
                 nullptr);
    ARMIGO_CHECK(maxwell::detail::find_registered_family(instruction(0xefd8)) ==
                 nullptr);
    ARMIGO_CHECK(maxwell::detail::find_registered_family(instruction(0x0100)) ==
                 nullptr);
    // The registered entry matches the register-move pattern and its decoder
    // produces the same instruction the legacy case produced.
    const auto* entry = maxwell::detail::find_registered_family(
        mov_register(3, 4));
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) {
        return;
    }
    const std::uint64_t word = mov_register(3, 4, 1);
    const maxwell::detail::DecodeOutcome outcome =
        entry->decode(word, 8, Predicate{7, false});
    const auto* decoded = std::get_if<Instruction>(&outcome);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded != nullptr) {
        ARMIGO_CHECK(decoded->family == OpcodeFamily::MovRegister);
        ARMIGO_CHECK_EQ(decoded->original_word, word);
        ARMIGO_CHECK_EQ(decoded->pc, 8u);
        const auto* move = std::get_if<MovRegister>(&decoded->payload);
        ARMIGO_CHECK(move != nullptr);
        if (move != nullptr) {
            ARMIGO_CHECK(move->destination.index == 3);
            ARMIGO_CHECK(move->source.index == 4);
            ARMIGO_CHECK(move->write_mask == 1);
        }
    }
}

ARMIGO_TEST(maxwell, family_registry_validator_rejects_ambiguous_tables) {
    using FamilyEntry = maxwell::detail::FamilyEntry;
    const auto valid = [](std::span<const FamilyEntry> table) {
        return shader::detail::family_registry_valid(
            table, [](const FamilyEntry& entry) {
                return entry.decode != nullptr &&
                       entry.family != OpcodeFamily::Unknown;
            });
    };
    // The production table satisfies the shared validator.
    ARMIGO_CHECK(valid(maxwell::detail::registered_families()));

    // Positive: patterns that disagree on a bit both masks fix can never
    // match the same word, so distinct families stay unambiguous.
    const std::array<FamilyEntry, 2> disjoint{{
        {0xfff8, 0x5c98, OpcodeFamily::MovRegister,
         maxwell::detail::decode_mov_register},
        {0xffff, 0x5c90, OpcodeFamily::FloatAddRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!shader::detail::patterns_intersect(0xfff8, 0x5c98, 0xffff,
                                                     0x5c90));
    ARMIGO_CHECK(valid(disjoint));

    // Negative: a value bit outside its own mask is unreachable.
    const std::array<FamilyEntry, 1> impossible_value{{
        {0x00f0, 0x0100, OpcodeFamily::MovRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(impossible_value));

    // Negative: masks that partially overlap while agreeing on their common
    // fixed bits both admit the same word, here 0x5c00.
    const std::array<FamilyEntry, 2> partial_overlap{{
        {0xff00, 0x5c00, OpcodeFamily::MovRegister,
         maxwell::detail::decode_mov_register},
        {0x0f00, 0x0c00, OpcodeFamily::FloatAddRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(partial_overlap));

    // Negative: a family may not register twice.
    const std::array<FamilyEntry, 2> duplicate_family{{
        {0xfff8, 0x5c98, OpcodeFamily::MovRegister,
         maxwell::detail::decode_mov_register},
        {0xffff, 0x5c90, OpcodeFamily::MovRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(duplicate_family));

    // Negative: a missing decoder has no registration.
    const std::array<FamilyEntry, 2> null_decoder{{
        {0xfff8, 0x5c98, OpcodeFamily::MovRegister, nullptr},
        {0xffff, 0x5c90, OpcodeFamily::FloatAddRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(null_decoder));

    // Negative: a zero mask would swallow every unknown word.
    const std::array<FamilyEntry, 1> zero_mask{{
        {0x0000, 0x0000, OpcodeFamily::FloatAddRegister,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(zero_mask));

    // Negative: Unknown is never a registrable family.
    const std::array<FamilyEntry, 1> unknown_family{{
        {0xfff8, 0x5c98, OpcodeFamily::Unknown,
         maxwell::detail::decode_mov_register},
    }};
    ARMIGO_CHECK(!valid(unknown_family));
}


ARMIGO_TEST(maxwell, ipa_multiply_preserves_operand_and_rejects_other_modes) {
    for (const unsigned source : {0u, 9u, 254u, 255u}) {
        auto word = set(ipa(9, 38), 54, 2, 1);
        word = set(word, 20, 8, source);
        const auto result = decode_one(word, Stage::Fragment);
        const auto* program = std::get_if<DecodedProgram>(&result);
        ARMIGO_CHECK(program != nullptr);
        if (program) {
            const auto& value = std::get<InterpolateAttribute>(program->instructions[0].payload);
            ARMIGO_CHECK(value.multiply);
            ARMIGO_CHECK_EQ(value.multiplier.index, source);
            ARMIGO_CHECK_EQ(value.destination.index, 9u);
        }
    }
    for (const unsigned mode : {2u, 3u}) {
        const auto error = rejected(decode_one(set(ipa(9, 38), 54, 2, mode)),
                 DecodeError::Code::UnsupportedModifier, OpcodeFamily::InterpolateAttribute);
        if (error) ARMIGO_CHECK(error->field == DecodeField::InterpolationMode);
    }
}

}  // namespace armigo::shader::test
