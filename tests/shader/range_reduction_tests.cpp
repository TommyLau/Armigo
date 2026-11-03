// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic decode, lowering, and evaluation tests for the range-reduction
// (RRO, register and constant-buffer forms) and dependency-barrier (DEPBAR)
// families. Every instruction word and every expected output word below is
// independently authored from the documented field layout; no captured game
// bytes take part.

#include <armigo/shader/families/range_reduction.h>
#include <armigo/shader/graphics.h>
#include <armigo/shader/maxwell.h>

#include "maxwell_families.h"
#include "maxwell_forward_decode.h"
#include "test_support.h"

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
using Family = mx::OpcodeFamily;
using shader_input::ByteSpan;
using shader_input::Stage;

// Synthetic register-form word: destination bits 0..7, instruction guard
// bits 16..19, source register B bits 20..27, mode bit 39, negate bit 45,
// and absolute bit 49, under the fixed opcode high half.
std::uint64_t rro_register(std::uint8_t destination, std::uint8_t source,
                           bool ex2_mode = false, bool negate = false,
                           bool absolute = false) {
    std::uint64_t word = (std::uint64_t{0x5c90} << 48) |
                         (std::uint64_t{7} << 16) | destination |
                         (std::uint64_t{source} << 20);
    if (ex2_mode) {
        word |= std::uint64_t{1} << 39;
    }
    if (negate) {
        word |= std::uint64_t{1} << 45;
    }
    if (absolute) {
        word |= std::uint64_t{1} << 49;
    }
    return word;
}

// Synthetic constant-buffer-form word: destination bits 0..7, word offset
// bits 20..33, binding bits 34..38, and the same mode, negate, and absolute
// bits as the register form.
std::uint64_t rro_constant(std::uint8_t destination, std::uint8_t binding,
                           std::uint32_t word_offset, bool ex2_mode = false,
                           bool negate = false, bool absolute = false) {
    std::uint64_t word = (std::uint64_t{0x4c90} << 48) |
                         (std::uint64_t{7} << 16) | destination |
                         (std::uint64_t{word_offset & 0x3fff} << 20) |
                         (static_cast<std::uint64_t>(binding & 0x1f) << 34);
    if (ex2_mode) {
        word |= std::uint64_t{1} << 39;
    }
    if (negate) {
        word |= std::uint64_t{1} << 45;
    }
    if (absolute) {
        word |= std::uint64_t{1} << 49;
    }
    return word;
}

// Synthetic dependency-barrier word with the requested operand bits. The
// guard field keeps the PT value so the common gate admits the word.
std::uint64_t depbar(std::uint64_t operand_bits) {
    constexpr std::uint64_t guard = std::uint64_t{0xf} << 16;
    return (std::uint64_t{0xf0f0} << 48) |
           (operand_bits & ~guard) | (std::uint64_t{7} << 16);
}

// Synthetic register-move words for the window-disjointness checks.
std::uint64_t mov_register(std::uint8_t destination, std::uint8_t source) {
    return (std::uint64_t{0x5c98} << 48) | (std::uint64_t{7} << 16) |
           destination | (std::uint64_t{source} << 20) |
           (std::uint64_t{15} << 39);
}

std::uint64_t mov_constant_word(std::uint8_t destination) {
    return (std::uint64_t{0x4c98} << 48) | (std::uint64_t{7} << 16) |
           destination | (std::uint64_t{15} << 39);
}

std::uint64_t exit_word() {
    return (std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15;
}

std::uint64_t set_field(std::uint64_t word, unsigned shift, unsigned width,
                        std::uint64_t value) {
    const std::uint64_t field_mask =
        ((std::uint64_t{1} << width) - 1) << shift;
    return (word & ~field_mask) | ((value << shift) & field_mask);
}

void put32(std::vector<std::uint8_t>& bytes, std::size_t offset,
           std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        bytes[offset + index] =
            static_cast<std::uint8_t>(value >> (index * 8));
    }
}

void put64(std::vector<std::uint8_t>& bytes, std::size_t offset,
           std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
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

std::optional<mx::DecodeError> rejected_decode(
    const mx::DecodeResult& result, mx::DecodeError::Code code,
    Family family, mx::DecodeField field = mx::DecodeField::None) {
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == code);
        ARMIGO_CHECK(error->family == family);
        if (field != mx::DecodeField::None) {
            ARMIGO_CHECK(error->field == field);
        }
        ARMIGO_CHECK(error->message()[0] != '\0');
        return *error;
    }
    return std::nullopt;
}

// Borrows the decode result: the caller must keep the result object alive
// while it uses the returned instruction.
const mx::Instruction* decoded_instruction(const mx::DecodeResult& result,
                                           Family family) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return nullptr;
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const mx::Instruction& decoded = program->instructions[0];
    ARMIGO_CHECK(decoded.family == family);
    if (decoded.family != family) return nullptr;
    ARMIGO_CHECK(decoded.predicate.index == 7 && !decoded.predicate.inverted);
    return &decoded;
}

mx::DecodeResult decode_one(std::uint64_t word, Stage stage = Stage::Vertex) {
    const Fixture data = fixture(stage, {word, exit_word()});
    return mx::decode(span(data.control), span(data.code));
}

void put(mx::DecodedProgram& program, unsigned at, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        program.sph[at + i] = std::byte(value >> (i * 8));
    }
}

mx::DecodedProgram base(Stage stage) {
    mx::DecodedProgram result;
    result.stage = stage;
    put(result, 0, shader_input::sph_stage_type_for(stage) << 10);
    return result;
}

void declare_output(mx::DecodedProgram& program, unsigned component) {
    unsigned at = 0;
    unsigned bit = 0;
    if (component >= 28 && component < 32) {
        at = 0x35;
        bit = component - 28 + 4;
    } else {
        at = 0x36 + (component - 32) / 8;
        bit = (component - 32) % 8;
    }
    program.sph[at] |= std::byte(1u << bit);
}

void append(mx::DecodedProgram& program, Family family,
            mx::InstructionPayload payload) {
    const auto i = program.instructions.size();
    program.instructions.push_back(
        {static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8), 0, family,
         {7, false}, std::move(payload)});
}

void load_immediate(mx::DecodedProgram& program, std::uint8_t reg,
                    std::uint32_t bits) {
    append(program, Family::Mov32Immediate,
           mx::Mov32Immediate{{reg}, 15, bits});
}

void range_reduce_register(mx::DecodedProgram& program, std::uint8_t destination,
                           std::uint8_t source, bool ex2_mode = false) {
    append(program, Family::RangeReductionRegister,
           mx::RangeReduction{{destination}, mx::RegisterOperand{source},
                              ex2_mode, false, false});
}

void range_reduce_constant(mx::DecodedProgram& program, std::uint8_t destination,
                           std::uint8_t binding, std::uint32_t byte_offset,
                           bool ex2_mode = false) {
    append(program, Family::RangeReductionConstant,
           mx::RangeReduction{{destination},
                              mx::ConstantBufferOperand{binding, byte_offset},
                              ex2_mode, false, false});
}

void dependency_barrier(mx::DecodedProgram& program, std::uint64_t operand_bits) {
    append(program, Family::DependencyBarrier,
           mx::DependencyBarrier{operand_bits});
}

void store(mx::DecodedProgram& program, std::uint8_t reg, unsigned component) {
    append(program, Family::AttributeStore,
           mx::AttributeAccess{{reg}, {255}, {255},
               static_cast<std::uint16_t>(component * 4), 0, false});
}

void exit_program(mx::DecodedProgram& program) {
    append(program, Family::Exit, mx::Exit{15});
}

gx::EvaluateResult run(const mx::DecodedProgram& program,
                       const gx::Inputs& inputs = {}) {
    const auto lowered = gx::lower(program);
    if (const auto* error = std::get_if<gx::Error>(&lowered)) return *error;
    return gx::evaluate(std::get<gx::Program>(lowered), inputs);
}

void output(const gx::EvaluateResult& result, gx::OutputKind kind,
            unsigned component, std::uint32_t expected) {
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    if (!evaluated) {
        record_failure(__FILE__, __LINE__, std::get<gx::Error>(result).message());
        return;
    }
    for (const auto& value : evaluated->outputs) {
        if (value.kind == kind && value.component == component) {
            ARMIGO_CHECK_EQ(value.bits, expected);
            return;
        }
    }
    record_failure(__FILE__, __LINE__, "Expected graphics output was not exported");
}

void rejected_lower(const gx::LowerResult& result, gx::Error::Code code) {
    const auto* error = std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
}

}  // namespace

ARMIGO_TEST(range_reduction, decodes_register_form_fields_exactly) {
    const std::uint64_t word = rro_register(3, 1);
    const mx::DecodeResult result = decode_one(word);
    const mx::Instruction* decoded =
        decoded_instruction(result, Family::RangeReductionRegister);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded == nullptr) return;
    ARMIGO_CHECK_EQ(decoded->original_word, word);
    ARMIGO_CHECK_EQ(decoded->pc, 8u);
    const auto* payload =
        std::get_if<mx::RangeReduction>(&decoded->payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload == nullptr) return;
    ARMIGO_CHECK(payload->destination.index == 3);
    const auto* source = std::get_if<mx::RegisterOperand>(&payload->source);
    ARMIGO_CHECK(source != nullptr);
    if (source != nullptr) {
        ARMIGO_CHECK(source->index == 1);
    }
    ARMIGO_CHECK(!payload->ex2_mode);
    ARMIGO_CHECK(!payload->negate);
    ARMIGO_CHECK(!payload->absolute);
}

ARMIGO_TEST(range_reduction, decodes_register_rz_and_ex2_mode) {
    const mx::DecodeResult result =
        decode_one(rro_register(255, 255, true));
    const mx::Instruction* decoded =
        decoded_instruction(result, Family::RangeReductionRegister);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded == nullptr) return;
    const auto* payload =
        std::get_if<mx::RangeReduction>(&decoded->payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload == nullptr) return;
    ARMIGO_CHECK(payload->destination.is_zero());
    const auto* source = std::get_if<mx::RegisterOperand>(&payload->source);
    ARMIGO_CHECK(source != nullptr);
    if (source != nullptr) {
        ARMIGO_CHECK(source->is_zero());
    }
    ARMIGO_CHECK(payload->ex2_mode);
}

ARMIGO_TEST(range_reduction, decodes_constant_form_fields_exactly) {
    const std::uint64_t word = rro_constant(9, 3, 0x123);
    const mx::DecodeResult result = decode_one(word);
    const mx::Instruction* decoded =
        decoded_instruction(result, Family::RangeReductionConstant);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded == nullptr) return;
    ARMIGO_CHECK_EQ(decoded->original_word, word);
    const auto* payload =
        std::get_if<mx::RangeReduction>(&decoded->payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload == nullptr) return;
    ARMIGO_CHECK(payload->destination.index == 9);
    const auto* source =
        std::get_if<mx::ConstantBufferOperand>(&payload->source);
    ARMIGO_CHECK(source != nullptr);
    if (source != nullptr) {
        ARMIGO_CHECK(source->binding == 3);
        ARMIGO_CHECK_EQ(source->byte_offset, 0x123u * 4u);
    }
}

ARMIGO_TEST(range_reduction, decodes_constant_maximal_offset_and_binding) {
    const mx::DecodeResult result =
        decode_one(rro_constant(255, 17, 0x3fff, true));
    const mx::Instruction* decoded =
        decoded_instruction(result, Family::RangeReductionConstant);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded == nullptr) return;
    const auto* payload =
        std::get_if<mx::RangeReduction>(&decoded->payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload == nullptr) return;
    const auto* source =
        std::get_if<mx::ConstantBufferOperand>(&payload->source);
    ARMIGO_CHECK(source != nullptr);
    if (source != nullptr) {
        ARMIGO_CHECK(source->binding == 17);
        ARMIGO_CHECK_EQ(source->byte_offset, 65532u);
    }
    ARMIGO_CHECK(payload->ex2_mode);
}

ARMIGO_TEST(range_reduction, rejects_stray_bits_in_both_forms) {
    // Bits 8..15 are unused fields, bits 40..44 and 46..48 sit outside the
    // admitted mode, negate, and absolute singletons, and bits 48..50 are
    // wildcard in the match pattern but must be zero in the admitted form.
    // The register form keeps the stray bits 28 and 38 of the unused span
    // between its source and mode fields; in the constant-buffer form those
    // bit positions are admitted offset and binding bits instead, exercised
    // by the field and binding tests above.
    for (const unsigned bit : {8u, 15u, 28u, 38u, 40u, 44u, 46u, 48u, 50u}) {
        const std::uint64_t register_word = rro_register(3, 1) | (1ull << bit);
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(register_word),
            mx::DecodeError::Code::UnsupportedModifier,
            Family::RangeReductionRegister, mx::DecodeField::ReservedBits);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, 1ull << bit);
        }
        if (bit == 28u || bit == 38u) {
            continue;
        }
        const std::uint64_t constant_word = rro_constant(3, 5, 1) | (1ull << bit);
        const std::optional<mx::DecodeError> constant_error = rejected_decode(
            decode_one(constant_word),
            mx::DecodeError::Code::UnsupportedModifier,
            Family::RangeReductionConstant, mx::DecodeField::ReservedBits);
        if (constant_error.has_value()) {
            ARMIGO_CHECK_EQ(constant_error->value, 1ull << bit);
        }
    }
}

ARMIGO_TEST(range_reduction, rejects_out_of_range_constant_binding) {
    for (const std::uint8_t binding : {18, 19, 24, 31}) {
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(rro_constant(3, binding, 1)),
            mx::DecodeError::Code::UnsupportedModifier,
            Family::RangeReductionConstant,
            mx::DecodeField::ConstantBinding);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, static_cast<std::uint64_t>(binding));
            ARMIGO_CHECK_EQ(error->expected, 17u);
        }
    }
}

ARMIGO_TEST(range_reduction, leaves_the_immediate_form_unregistered) {
    // The immediate opcode window stays unclaimed because the immediate form
    // is unimplemented: a reachable word must fail as an unknown
    // opcode, never decode as a range reduction.
    for (const std::uint16_t high : {std::uint16_t{0x3890},
                                     std::uint16_t{0x3c90}}) {
        rejected_decode(decode_one((std::uint64_t{high} << 48) |
                                   (std::uint64_t{7} << 16)),
                        mx::DecodeError::Code::UnknownOpcode, Family::Unknown);
    }
}

ARMIGO_TEST(range_reduction, depbar_decodes_arbitrary_operand_bits) {
    for (const std::uint64_t operands :
         {0ull, 1ull, 0xffull, 0x800000000000ull, 0xdeadbeefull}) {
        const std::uint64_t word = depbar(operands);
        const mx::DecodeResult result = decode_one(word);
        const mx::Instruction* decoded =
            decoded_instruction(result, Family::DependencyBarrier);
        ARMIGO_CHECK(decoded != nullptr);
        if (decoded == nullptr) continue;
        ARMIGO_CHECK_EQ(decoded->original_word, word);
        const auto* payload =
            std::get_if<mx::DependencyBarrier>(&decoded->payload);
        ARMIGO_CHECK(payload != nullptr);
        if (payload != nullptr) {
            // The raw low bits, guard field included, are preserved for
            // diagnostics only.
            ARMIGO_CHECK_EQ(payload->operand_bits, word & 0xffffffffffffull);
        }
    }
}

ARMIGO_TEST(range_reduction, windows_stay_disjoint_from_the_move_windows) {
    // The range-reduction and move windows differ in opcode bit 51 only;
    // each word must land in its own family.
    const auto* register_entry =
        mx::detail::find_registered_family(rro_register(3, 1));
    ARMIGO_CHECK(register_entry != nullptr);
    if (register_entry != nullptr) {
        ARMIGO_CHECK(register_entry->family == Family::RangeReductionRegister);
        ARMIGO_CHECK_EQ(register_entry->mask, 0xfff8u);
        ARMIGO_CHECK_EQ(register_entry->value, 0x5c90u);
    }
    const auto* constant_entry =
        mx::detail::find_registered_family(rro_constant(3, 5, 1));
    ARMIGO_CHECK(constant_entry != nullptr);
    if (constant_entry != nullptr) {
        ARMIGO_CHECK(constant_entry->family == Family::RangeReductionConstant);
        ARMIGO_CHECK_EQ(constant_entry->mask, 0xfff8u);
        ARMIGO_CHECK_EQ(constant_entry->value, 0x4c90u);
    }
    const auto* barrier_entry = mx::detail::find_registered_family(depbar(0));
    ARMIGO_CHECK(barrier_entry != nullptr);
    if (barrier_entry != nullptr) {
        ARMIGO_CHECK(barrier_entry->family == Family::DependencyBarrier);
        ARMIGO_CHECK_EQ(barrier_entry->mask, 0xfff8u);
        ARMIGO_CHECK_EQ(barrier_entry->value, 0xf0f0u);
    }
    const auto* mov_entry =
        mx::detail::find_registered_family(mov_register(3, 1));
    ARMIGO_CHECK(mov_entry != nullptr);
    if (mov_entry != nullptr) {
        ARMIGO_CHECK(mov_entry->family == Family::MovRegister);
        ARMIGO_CHECK_EQ(mov_entry->value, 0x5c98u);
    }

    // Full-container decode agrees with the registry: bit 51 alone decides
    // between the move and range-reduction families.
    const mx::DecodeResult moved_result = decode_one(mov_register(3, 1));
    const mx::Instruction* moved =
        decoded_instruction(moved_result, Family::MovRegister);
    ARMIGO_CHECK(moved != nullptr);
    const mx::DecodeResult reduced_result = decode_one(rro_register(3, 1));
    const mx::Instruction* reduced =
        decoded_instruction(reduced_result, Family::RangeReductionRegister);
    ARMIGO_CHECK(reduced != nullptr);
    const mx::DecodeResult moved_constant_result =
        decode_one(mov_constant_word(3));
    const mx::Instruction* moved_constant =
        decoded_instruction(moved_constant_result, Family::MovConstant);
    ARMIGO_CHECK(moved_constant != nullptr);
    const mx::DecodeResult reduced_constant_result =
        decode_one(rro_constant(3, 5, 1));
    const mx::Instruction* reduced_constant = decoded_instruction(
        reduced_constant_result, Family::RangeReductionConstant);
    ARMIGO_CHECK(reduced_constant != nullptr);

    // Neighbouring opcode halves match no family and fail as unknown
    // opcodes, and the barrier window keeps its neighbours out.
    for (const std::uint16_t high : {std::uint16_t{0x5c88},
                                     std::uint16_t{0x4c88},
                                     std::uint16_t{0xf0e8},
                                     std::uint16_t{0xf0e0}}) {
        rejected_decode(decode_one((std::uint64_t{high} << 48) |
                                   (std::uint64_t{7} << 16)),
                        mx::DecodeError::Code::UnknownOpcode, Family::Unknown);
    }
}

ARMIGO_TEST(range_reduction, lowers_register_source_unchanged_in_both_modes) {
    for (const bool ex2_mode : {false, true}) {
        auto program = base(Stage::Vertex);
        declare_output(program, 28);
        load_immediate(program, 1, 0x7fc12345u);
        range_reduce_register(program, 2, 1, ex2_mode);
        store(program, 2, 28);
        exit_program(program);
        output(run(program), gx::OutputKind::Position, 0, 0x7fc12345u);
    }
}

ARMIGO_TEST(range_reduction, lowering_adds_no_node_beyond_the_source) {
    auto with_reduction = base(Stage::Vertex);
    declare_output(with_reduction, 28);
    load_immediate(with_reduction, 1, 0x7fc12345u);
    range_reduce_register(with_reduction, 2, 1, false);
    store(with_reduction, 2, 28);
    exit_program(with_reduction);

    auto without_reduction = base(Stage::Vertex);
    declare_output(without_reduction, 28);
    load_immediate(without_reduction, 1, 0x7fc12345u);
    store(without_reduction, 1, 28);
    exit_program(without_reduction);

    const auto lowered = gx::lower(with_reduction);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        return;
    }
    const auto& program = std::get<gx::Program>(lowered);
    const auto baseline = gx::lower(without_reduction);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(baseline));
    if (!std::holds_alternative<gx::Program>(baseline)) {
        return;
    }
    // The reduction consumes no nodes of its own: the node list is exactly
    // the move literal, with no arithmetic, reciprocal, or saturate node.
    ARMIGO_CHECK_EQ(program.nodes().size(),
                    std::get<gx::Program>(baseline).nodes().size());
    std::optional<std::uint32_t> literal_index;
    for (std::size_t i = 0; i < program.nodes().size(); ++i) {
        const auto* literal =
            std::get_if<gx::Literal>(&program.nodes()[i].operation);
        ARMIGO_CHECK(literal != nullptr);
        if (literal != nullptr && literal->bits == 0x7fc12345u) {
            literal_index = static_cast<std::uint32_t>(i);
        }
    }
    ARMIGO_CHECK(literal_index.has_value());
    if (!literal_index.has_value()) {
        return;
    }
    // The export consumes the source scalar itself, the SSA alias, not a
    // copy produced by the reduction.
    bool exported = false;
    for (const auto& value : program.exports()) {
        if (value.kind == gx::OutputKind::Position && value.component == 0) {
            ARMIGO_CHECK_EQ(value.value.index, *literal_index);
            exported = true;
        }
    }
    ARMIGO_CHECK(exported);
}

ARMIGO_TEST(range_reduction, lowers_constant_source_unchanged) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    range_reduce_constant(program, 1, 3, 4);
    store(program, 1, 28);
    exit_program(program);
    const std::array<std::uint32_t, 2> buffer = {0xdeadbeefu, 0x7f811234u};
    gx::Inputs inputs;
    inputs.constant_buffers[3] = buffer;
    output(run(program, inputs), gx::OutputKind::Position, 0, 0x7f811234u);
}

ARMIGO_TEST(range_reduction, negate_and_absolute_are_exact_sign_operations) {
    for (const bool ex2_mode : {false, true}) {
        for (const bool negate : {false, true}) {
            for (const bool absolute : {false, true}) {
                if (!negate && !absolute) {
                    continue;
                }
                mx::DecodedProgram program = base(Stage::Vertex);
                declare_output(program, 28);
                load_immediate(program, 1, 0xffc12345u);  // negative NaN, payload kept
                append(program, Family::RangeReductionRegister,
                       mx::RangeReduction{{2}, mx::RegisterOperand{1},
                                          ex2_mode, negate, absolute});
                store(program, 2, 28);
                exit_program(program);
                // Absolute first, then negate, as exact bit operations.
                std::uint32_t expected = 0xffc12345u;
                if (absolute) expected &= 0x7fffffffu;
                if (negate) expected ^= 0x80000000u;
                output(run(program), gx::OutputKind::Position, 0, expected);

                mx::DecodedProgram constant_program = base(Stage::Vertex);
                declare_output(constant_program, 28);
                append(constant_program, Family::RangeReductionConstant,
                       mx::RangeReduction{{1},
                                          mx::ConstantBufferOperand{3, 0},
                                          ex2_mode, negate, absolute});
                store(constant_program, 1, 28);
                exit_program(constant_program);
                const std::array<std::uint32_t, 1> words{0x80000000u};  // -0.0
                gx::Inputs inputs;
                inputs.constant_buffers[3] = words;
                std::uint32_t constant_expected = 0x80000000u;
                if (absolute) constant_expected &= 0x7fffffffu;
                if (negate) constant_expected ^= 0x80000000u;
                output(run(constant_program, inputs), gx::OutputKind::Position, 0,
                       constant_expected);
            }
        }
    }
}

ARMIGO_TEST(range_reduction, forged_payloads_fail_the_checked_paths) {
    auto forged = base(Stage::Vertex);
    declare_output(forged, 28);
    load_immediate(forged, 1, 0x40000000u);
    append(forged, Family::RangeReductionRegister,
           mx::MovRegister{{2}, {1}, 15});
    store(forged, 2, 28);
    exit_program(forged);
    rejected_lower(gx::lower(forged), gx::Error::Code::InvalidInstruction);

    auto forged_barrier = base(Stage::Vertex);
    declare_output(forged_barrier, 28);
    load_immediate(forged_barrier, 1, 0x40000000u);
    append(forged_barrier, Family::DependencyBarrier,
           mx::MovRegister{{1}, {1}, 15});
    store(forged_barrier, 1, 28);
    exit_program(forged_barrier);
    rejected_lower(gx::lower(forged_barrier),
                   gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(range_reduction, depbar_lowers_to_nothing) {
    auto with_barriers = base(Stage::Vertex);
    declare_output(with_barriers, 28);
    load_immediate(with_barriers, 1, 0x40490fdbu);
    dependency_barrier(with_barriers, 0x1);
    range_reduce_register(with_barriers, 2, 1, false);
    dependency_barrier(with_barriers, 0x800000000000ull);
    store(with_barriers, 2, 28);
    exit_program(with_barriers);

    auto without_barriers = base(Stage::Vertex);
    declare_output(without_barriers, 28);
    load_immediate(without_barriers, 1, 0x40490fdbu);
    range_reduce_register(without_barriers, 2, 1, false);
    store(without_barriers, 2, 28);
    exit_program(without_barriers);

    const auto lowered = gx::lower(with_barriers);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    const auto baseline = gx::lower(without_barriers);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(baseline));
    if (!std::holds_alternative<gx::Program>(lowered) ||
        !std::holds_alternative<gx::Program>(baseline)) {
        return;
    }
    // The barriers add no node: both lowerings produce identical node
    // counts, and the routed value still reaches the output exactly.
    ARMIGO_CHECK_EQ(std::get<gx::Program>(lowered).nodes().size(),
                    std::get<gx::Program>(baseline).nodes().size());
    output(run(with_barriers), gx::OutputKind::Position, 0, 0x40490fdbu);
}

ARMIGO_TEST(range_reduction, routes_bits_through_the_full_chain_exactly) {
    // A synthetic program routes a NaN payload through a barrier, both RRO
    // forms, and a second barrier to the output: every hop must reproduce
    // the bits exactly, with no arithmetic reinterpretation in between.
    auto nan_program = base(Stage::Vertex);
    declare_output(nan_program, 28);
    load_immediate(nan_program, 1, 0x7fc12345u);
    dependency_barrier(nan_program, 0x2a);
    range_reduce_register(nan_program, 2, 1, false);
    range_reduce_constant(nan_program, 3, 3, 8, true);
    dependency_barrier(nan_program, 0x40);
    store(nan_program, 3, 28);
    exit_program(nan_program);
    const std::array<std::uint32_t, 3> buffer = {0x12345678u, 0x12345678u,
                                                 0x7fc12345u};
    gx::Inputs inputs;
    inputs.constant_buffers[3] = buffer;
    output(run(nan_program, inputs), gx::OutputKind::Position, 0, 0x7fc12345u);

    // Signed zero keeps its sign bit through the same chain.
    auto zero_program = base(Stage::Vertex);
    declare_output(zero_program, 28);
    load_immediate(zero_program, 1, 0x80000000u);
    dependency_barrier(zero_program, 0);
    range_reduce_register(zero_program, 2, 1, true);
    range_reduce_constant(zero_program, 3, 3, 0, false);
    store(zero_program, 3, 28);
    exit_program(zero_program);
    const std::array<std::uint32_t, 1> zero_buffer = {0x80000000u};
    gx::Inputs zero_inputs;
    zero_inputs.constant_buffers[3] = zero_buffer;
    output(run(zero_program, zero_inputs), gx::OutputKind::Position, 0,
           0x80000000u);
}

ARMIGO_TEST(range_reduction, forward_decoder_admits_families_with_guards) {
    const std::uint64_t register_word = rro_register(4, 1);
    const std::uint64_t constant_word = rro_constant(4, 2, 8);
    const std::uint64_t barrier_word = depbar(0xdeadbeefull);
    for (const std::uint8_t guard : {0, 5, 9, 15}) {
        const auto decode_guarded = [guard](std::uint64_t word) {
            return mx::decode_guarded_instruction(
                set_field(word, 16, 4, guard), 8);
        };
        const auto register_result = decode_guarded(register_word);
        const auto* register_instruction =
            std::get_if<mx::Instruction>(&register_result);
        ARMIGO_CHECK(register_instruction != nullptr);
        if (register_instruction != nullptr) {
            ARMIGO_CHECK(register_instruction->family ==
                         Family::RangeReductionRegister);
            ARMIGO_CHECK_EQ(register_instruction->predicate.index,
                            static_cast<std::uint8_t>(guard & 7));
            ARMIGO_CHECK_EQ(register_instruction->predicate.inverted,
                            guard >= 8);
        }
        const auto constant_result = decode_guarded(constant_word);
        const auto* constant_instruction =
            std::get_if<mx::Instruction>(&constant_result);
        ARMIGO_CHECK(constant_instruction != nullptr);
        if (constant_instruction != nullptr) {
            ARMIGO_CHECK(constant_instruction->family ==
                         Family::RangeReductionConstant);
            ARMIGO_CHECK_EQ(constant_instruction->predicate.inverted,
                            guard >= 8);
        }
        const auto barrier_result = decode_guarded(barrier_word);
        const auto* barrier_instruction =
            std::get_if<mx::Instruction>(&barrier_result);
        ARMIGO_CHECK(barrier_instruction != nullptr);
        if (barrier_instruction != nullptr) {
            ARMIGO_CHECK(barrier_instruction->family ==
                         Family::DependencyBarrier);
            ARMIGO_CHECK_EQ(barrier_instruction->predicate.inverted,
                            guard >= 8);
        }
    }
    // The strict straight-line path keeps requiring PT for all three.
    const std::array<std::pair<std::uint64_t, Family>, 3> strict_words{
        {{register_word, Family::RangeReductionRegister},
         {constant_word, Family::RangeReductionConstant},
         {barrier_word, Family::DependencyBarrier}}};
    for (const auto& [word, family] : strict_words) {
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(set_field(word, 16, 4, 5)),
            mx::DecodeError::Code::UnsupportedPredicate, family,
            mx::DecodeField::InstructionPredicate);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, 5u);
        }
    }
}

}  // namespace armigo::shader::test
