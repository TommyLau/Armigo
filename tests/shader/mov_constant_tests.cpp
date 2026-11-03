// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/families/mov_constant.h>
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>

#include "maxwell_families.h"
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
namespace gm = graphics_msl;
using Family = mx::OpcodeFamily;
using shader_input::ByteSpan;
using shader_input::Stage;

// Synthetic word builder for the constant-buffer register-move encoding:
// destination bits 0..7, word offset bits 20..33, binding bits 34..38, and
// write mask bits 39..42, under the fixed opcode high half.
std::uint64_t mov_constant(std::uint8_t destination, std::uint8_t binding,
                           std::uint32_t word_offset,
                           std::uint8_t write_mask = 15) {
    std::uint64_t word = (std::uint64_t{0x4c98} << 48) |
                         (std::uint64_t{7} << 16);
    word |= static_cast<std::uint64_t>(destination);
    word |= static_cast<std::uint64_t>(word_offset & 0x3fff) << 20;
    word |= static_cast<std::uint64_t>(binding & 0x1f) << 34;
    word |= static_cast<std::uint64_t>(write_mask & 0xf) << 39;
    return word;
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

void cb_move(mx::DecodedProgram& program, std::uint8_t destination,
             std::uint8_t binding, std::uint32_t byte_offset,
             std::uint8_t write_mask = 15) {
    append(program, Family::MovConstant,
           mx::MovConstant{{destination}, {binding, byte_offset}, write_mask});
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

ARMIGO_TEST(mov_constant, decodes_word_fields_exactly) {
    const std::uint64_t word = mov_constant(3, 5, 0x123);
    const mx::DecodeResult result = decode_one(word);
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const mx::Instruction& decoded = program->instructions[0];
    ARMIGO_CHECK(decoded.family == Family::MovConstant);
    ARMIGO_CHECK_EQ(decoded.original_word, word);
    ARMIGO_CHECK_EQ(decoded.pc, 8u);
    ARMIGO_CHECK(decoded.predicate.index == 7 && !decoded.predicate.inverted);
    const auto* payload = std::get_if<mx::MovConstant>(&decoded.payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload != nullptr) {
        ARMIGO_CHECK(payload->destination.index == 3);
        ARMIGO_CHECK(payload->source.binding == 5);
        ARMIGO_CHECK_EQ(payload->source.byte_offset, 0x123u * 4u);
        ARMIGO_CHECK(payload->write_mask == 15);
    }
}

ARMIGO_TEST(mov_constant, decodes_maximal_offset_and_last_binding) {
    const mx::DecodeResult result = decode_one(mov_constant(9, 17, 0x3fff));
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    const auto* payload =
        std::get_if<mx::MovConstant>(&program->instructions[0].payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload != nullptr) {
        ARMIGO_CHECK(payload->destination.index == 9);
        ARMIGO_CHECK(payload->source.binding == 17);
        ARMIGO_CHECK_EQ(payload->source.byte_offset, 65532u);
    }
}

ARMIGO_TEST(mov_constant, admits_single_lane_mask_and_rz_destination) {
    const mx::DecodeResult result = decode_one(mov_constant(255, 1, 2, 1));
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    const auto* payload =
        std::get_if<mx::MovConstant>(&program->instructions[0].payload);
    ARMIGO_CHECK(payload != nullptr);
    if (payload != nullptr) {
        ARMIGO_CHECK(payload->destination.is_zero());
        ARMIGO_CHECK(payload->write_mask == 1);
    }
}

ARMIGO_TEST(mov_constant, rejects_masks_outside_full_or_single_lane) {
    for (const std::uint8_t mask :
         {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) {
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(mov_constant(3, 5, 1, mask)),
            mx::DecodeError::Code::UnsupportedModifier, Family::MovConstant,
            mx::DecodeField::WriteMask);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, static_cast<std::uint64_t>(mask));
            ARMIGO_CHECK_EQ(error->expected, 15u);
        }
    }
}

ARMIGO_TEST(mov_constant, rejects_bindings_18_through_31) {
    for (const std::uint8_t binding : {18, 19, 24, 31}) {
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(mov_constant(3, binding, 1)),
            mx::DecodeError::Code::UnsupportedModifier, Family::MovConstant,
            mx::DecodeField::ConstantBinding);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, static_cast<std::uint64_t>(binding));
            ARMIGO_CHECK_EQ(error->expected, 17u);
        }
    }
}

ARMIGO_TEST(mov_constant, rejects_unused_and_reserved_bits) {
    // Bits 8..15 are unused fields; bits 43..47 are reserved; bits 48..50 are
    // wildcard in the match pattern but must be zero in the admitted form.
    for (const unsigned bit : {8u, 15u, 43u, 47u, 48u, 50u}) {
        const std::uint64_t word = mov_constant(3, 5, 1) | (1ull << bit);
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(word), mx::DecodeError::Code::UnsupportedModifier,
            Family::MovConstant, mx::DecodeField::ReservedBits);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, 1ull << bit);
        }
    }
}

ARMIGO_TEST(mov_constant, keeps_the_common_predicate_gate) {
    // Every predicate field value except PT (7), including inverted forms,
    // stays rejected by the shared gate before the family decoder runs.
    for (const std::uint64_t predicate : {0ull, 1ull, 6ull, 14ull, 15ull}) {
        const std::uint64_t word =
            set_field(mov_constant(3, 5, 1), 16, 4, predicate);
        const std::optional<mx::DecodeError> error = rejected_decode(
            decode_one(word), mx::DecodeError::Code::UnsupportedPredicate,
            Family::MovConstant, mx::DecodeField::InstructionPredicate);
        if (error.has_value()) {
            ARMIGO_CHECK_EQ(error->value, predicate);
        }
    }
}

ARMIGO_TEST(mov_constant, neighbours_and_legacy_forms_keep_old_behavior) {
    // Neighbouring opcode halves match neither the legacy classifier nor the
    // registered pattern.
    for (const std::uint16_t high :
         {std::uint16_t{0x4c88}, std::uint16_t{0x4c8f},
          std::uint16_t{0x4c80}, std::uint16_t{0x4d98}}) {
        rejected_decode(decode_one((std::uint64_t{high} << 48) |
                                   (std::uint64_t{7} << 16)),
                        mx::DecodeError::Code::UnknownOpcode,
                        Family::Unknown);
    }
    // The registered table serves the constant-move pattern directly.
    const auto* entry = mx::detail::find_registered_family(
        mov_constant(3, 5, 0x123));
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) return;
    ARMIGO_CHECK(entry->family == Family::MovConstant);
    ARMIGO_CHECK_EQ(entry->mask, 0xfff8u);
    ARMIGO_CHECK_EQ(entry->value, 0x4c98u);
    const std::uint64_t word = mov_constant(3, 5, 0x123, 1);
    const mx::detail::DecodeOutcome outcome =
        entry->decode(word, 8, mx::Predicate{7, false});
    const auto* decoded = std::get_if<mx::Instruction>(&outcome);
    ARMIGO_CHECK(decoded != nullptr);
    if (decoded != nullptr) {
        ARMIGO_CHECK(decoded->family == Family::MovConstant);
        ARMIGO_CHECK_EQ(decoded->original_word, word);
        ARMIGO_CHECK_EQ(decoded->pc, 8u);
        ARMIGO_CHECK(std::holds_alternative<mx::MovConstant>(decoded->payload));
    }
}

ARMIGO_TEST(mov_constant, lowers_raw_bits_without_reinterpretation) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    declare_output(program, 29);
    declare_output(program, 30);
    declare_output(program, 31);
    declare_output(program, 32);
    declare_output(program, 33);
    declare_output(program, 34);
    declare_output(program, 35);
    cb_move(program, 1, 3, 0);
    store(program, 1, 28);
    cb_move(program, 2, 3, 4);
    store(program, 2, 29);
    cb_move(program, 3, 3, 8);
    store(program, 3, 30);
    cb_move(program, 4, 3, 12);
    store(program, 4, 31);
    cb_move(program, 5, 4, 0);
    store(program, 5, 32);
    cb_move(program, 6, 4, 4);
    store(program, 6, 33);
    cb_move(program, 7, 4, 8);
    store(program, 7, 34);
    cb_move(program, 8, 4, 12);
    store(program, 8, 35);
    exit_program(program);
    // Signed zero, a NaN with payload, infinities, and subnormals must pass
    // through as the exact stored words.
    const std::array<std::uint32_t, 4> buffer3 = {
        0x00000000u, 0x80000000u, 0x7fc12345u, 0x7f800000u};
    const std::array<std::uint32_t, 4> buffer4 = {
        0x00000001u, 0x80000001u, 0x7f811234u, 0xff800000u};
    gx::Inputs inputs;
    inputs.constant_buffers[3] = buffer3;
    inputs.constant_buffers[4] = buffer4;
    const auto result = run(program, inputs);
    output(result, gx::OutputKind::Position, 0, 0x00000000u);
    output(result, gx::OutputKind::Position, 1, 0x80000000u);
    output(result, gx::OutputKind::Position, 2, 0x7fc12345u);
    output(result, gx::OutputKind::Position, 3, 0x7f800000u);
    output(result, gx::OutputKind::Generic, 0, 0x00000001u);
    output(result, gx::OutputKind::Generic, 1, 0x80000001u);
    output(result, gx::OutputKind::Generic, 2, 0x7f811234u);
    output(result, gx::OutputKind::Generic, 3, 0xff800000u);
}

ARMIGO_TEST(mov_constant, single_lane_mask_copies_the_same_raw_word) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    cb_move(program, 1, 6, 8, 1);
    store(program, 1, 28);
    exit_program(program);
    const std::array<std::uint32_t, 3> buffer = {0xdeadbeefu, 0x7f811234u,
                                                 0x007fffffu};
    gx::Inputs inputs;
    inputs.constant_buffers[6] = buffer;
    output(run(program, inputs), gx::OutputKind::Position, 0, 0x007fffffu);
}

ARMIGO_TEST(mov_constant, rz_destination_lowering_stays_defined) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    cb_move(program, 255, 3, 0);
    cb_move(program, 1, 3, 4);
    store(program, 1, 28);
    exit_program(program);
    const std::array<std::uint32_t, 2> buffer = {0xffffffffu, 0x12345678u};
    gx::Inputs inputs;
    inputs.constant_buffers[3] = buffer;
    output(run(program, inputs), gx::OutputKind::Position, 0, 0x12345678u);
}

ARMIGO_TEST(mov_constant, forged_payload_and_masks_fail_the_checked_paths) {
    auto forged = base(Stage::Vertex);
    declare_output(forged, 28);
    append(forged, Family::MovRegister,
           mx::MovConstant{{1}, {3, 0}, 15});
    store(forged, 1, 28);
    exit_program(forged);
    rejected_lower(gx::lower(forged), gx::Error::Code::InvalidInstruction);

    auto wrong_payload = base(Stage::Vertex);
    declare_output(wrong_payload, 28);
    append(wrong_payload, Family::MovConstant, mx::MovRegister{{1}, {255}, 15});
    store(wrong_payload, 1, 28);
    exit_program(wrong_payload);
    rejected_lower(gx::lower(wrong_payload), gx::Error::Code::InvalidInstruction);

    auto masked = base(Stage::Vertex);
    declare_output(masked, 28);
    cb_move(masked, 1, 3, 0, 3);
    store(masked, 1, 28);
    exit_program(masked);
    rejected_lower(gx::lower(masked), gx::Error::Code::UnsupportedModifier);

    auto binding = base(Stage::Vertex);
    declare_output(binding, 28);
    cb_move(binding, 1, 18, 0);
    store(binding, 1, 28);
    exit_program(binding);
    rejected_lower(gx::lower(binding), gx::Error::Code::UnsupportedModifier);

    auto unaligned = base(Stage::Vertex);
    declare_output(unaligned, 28);
    cb_move(unaligned, 1, 3, 2);
    store(unaligned, 1, 28);
    exit_program(unaligned);
    rejected_lower(gx::lower(unaligned), gx::Error::Code::UnsupportedModifier);

    auto oversized = base(Stage::Vertex);
    declare_output(oversized, 28);
    cb_move(oversized, 1, 3, 65536);
    store(oversized, 1, 28);
    exit_program(oversized);
    rejected_lower(gx::lower(oversized), gx::Error::Code::UnsupportedModifier);
}

ARMIGO_TEST(mov_constant, maximal_offset_reports_constant_bounds) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    cb_move(program, 1, 17, 65532);
    store(program, 1, 28);
    exit_program(program);
    // Lowering admits the maximal in-range word; evaluation reports the
    // buffer bound through the shared ConstantBounds diagnostic.
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    gx::Inputs inputs;
    const std::array<std::uint32_t, 1> buffer = {0x12345678u};
    inputs.constant_buffers[17] = buffer;
    const auto result = run(program, inputs);
    const auto* error = std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::ConstantBounds);
    }
    // Covering the requested word yields the raw value.
    std::array<std::uint32_t, 16384> full{};
    full[16383] = 0x12345678u;
    gx::Inputs covering;
    covering.constant_buffers[17] = full;
    output(run(program, covering), gx::OutputKind::Position, 0, 0x12345678u);
}

ARMIGO_TEST(mov_constant, reflection_and_msl_match_the_constant_path) {
    auto program = base(Stage::Vertex);
    declare_output(program, 28);
    declare_output(program, 29);
    declare_output(program, 30);
    declare_output(program, 31);
    cb_move(program, 1, 3, 0);
    store(program, 1, 28);
    cb_move(program, 2, 3, 4);
    store(program, 2, 29);
    cb_move(program, 3, 3, 8);
    store(program, 3, 30);
    cb_move(program, 4, 3, 12);
    store(program, 4, 31);
    exit_program(program);
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const gm::EmitResult result =
        gm::emit(std::get<gx::Program>(lowered), options);
    const auto* stage = std::get_if<gm::EmittedStage>(&result);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) return;
    // Reflection reports the minimal extent covering the loaded words.
    ARMIGO_CHECK(stage->constant_buffers.size() == 1);
    ARMIGO_CHECK(stage->constant_buffers[0].guest_binding == 3);
    ARMIGO_CHECK_EQ(stage->constant_buffers[0].minimum_size, 16u);
    // The emitted body is a raw load feeding the export through the SSA
    // alias: the helper library is always present in the prelude, so the
    // assertion is that no arithmetic call site touches the copied bits.
    ARMIGO_CHECK(stage->source.find("cb3[0]") != std::string::npos);
    ARMIGO_CHECK(stage->source.find("cb3[3]") != std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_add_rn_ftz(") == std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_mul_rn_ftz(") == std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_fma_rne_ftz(") ==
                 std::string::npos);
    ARMIGO_CHECK(stage->source.find("out.position.x = as_type<float>(v1);") !=
                 std::string::npos);
    ARMIGO_CHECK(stage->source.find("out.position.w = as_type<float>(v4);") !=
                 std::string::npos);
}

}  // namespace armigo::shader::test
