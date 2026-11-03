// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Float-to-integer tests: the CPU reference against authored binary32 edge
// tables, immediate admission, the F2I register and constant-buffer decoder
// profiles and rejections, lowering structure in both stages, and emitted
// MSL text. Every instruction word and expected result below is synthetic
// and authored from the F2I contract: flush, round, clamp, and map every
// NaN to zero; no oracle recomputes
// any result word.

#include "test_support.h"

#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>

#include "maxwell_families.h"
#include "graphics_operations.h"

#include <array>
#include <cstddef>
#include <cstdint>
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

// Immediate encoding of the FloatToInteger operation: bit 0 signedness,
// bit 1 a 32-bit destination, bits 3..4 the rounding code (0 round-even,
// 1 floor, 2 ceil, 3 trunc).
constexpr std::uint32_t i16_unsigned_even = 0x00u;
constexpr std::uint32_t i32_signed_floor = 0x0bu;
constexpr std::uint32_t i32_signed_ceil = 0x13u;
constexpr std::uint32_t i32_unsigned_trunc = 0x1au;
constexpr std::uint32_t i32_signed_trunc = 0x1bu;

std::uint64_t set(std::uint64_t word, unsigned shift, unsigned width,
                  std::uint64_t value) {
    const std::uint64_t field_mask = ((std::uint64_t{1} << width) - 1) << shift;
    return (word & ~field_mask) | ((value << shift) & field_mask);
}

// One synthetic F2I word in the admitted profile: F32 source, half=0, FTZ
// input, no absolute, CC, or negate, PT instruction predicate.
std::uint64_t f2i_register_word(std::uint64_t destination = 3,
                                std::uint64_t source = 2,
                                std::uint64_t dest_format = 2,
                                std::uint64_t is_signed = 1,
                                std::uint64_t rounding = 3) {
    return (std::uint64_t{0x5cb0} << 48) | (std::uint64_t{7} << 16) |
           destination | (dest_format << 8) | (2u << 10) | (is_signed << 12) |
           (source << 20) | (rounding << 39) | (std::uint64_t{1} << 44);
}

std::uint64_t f2i_constant_word(std::uint64_t destination = 3,
                                std::uint64_t offset_words = 9,
                                std::uint64_t binding = 4,
                                std::uint64_t dest_format = 2,
                                std::uint64_t is_signed = 1,
                                std::uint64_t rounding = 3) {
    return (std::uint64_t{0x4cb0} << 48) | (std::uint64_t{7} << 16) |
           destination | (dest_format << 8) | (2u << 10) | (is_signed << 12) |
           (offset_words << 20) | (binding << 34) | (rounding << 39) |
           (std::uint64_t{1} << 44);
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value,
         unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        bytes[at + i] = std::uint8_t(value >> (8 * i));
    }
}

// Full container decode of one instruction at pc 8 followed by an
// unconditional PT EXIT.
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
    put(code, 0x30, shader_input::sph_stage_type_for(Stage::Vertex) << 10, 4);
    put(code, 0x88, instruction, 8);
    put(code, 0x90,
        (std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15, 8);
    return mx::decode({control.data(), control.size()},
                      {code.data(), code.size()});
}

std::optional<mx::FloatToInteger> admitted(const mx::DecodeResult& result,
                                   Family family) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return std::nullopt;
    ARMIGO_CHECK_EQ(program->instructions.size(), std::size_t{2});
    if (program->instructions.size() != 2) return std::nullopt;
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == family);
    if (instruction.family != family) return std::nullopt;
    const auto* value = std::get_if<mx::FloatToInteger>(&instruction.payload);
    if (value == nullptr) return std::nullopt;
    return *value;
}

std::optional<mx::DecodeError> rejected(const mx::DecodeResult& result,
                                        mx::DecodeError::Code code,
                                        Family family = Family::Unknown) {
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) return std::nullopt;
    ARMIGO_CHECK(error->code == code);
    if (family != Family::Unknown) {
        ARMIGO_CHECK(error->family == family);
    }
    return *error;
}

// Straight-line fragment program: the color registers are initialised, the
// conversion writes register zero (color lane 0), then an unconditional
// EXIT. For the constant-buffer form the value comes from buffer 0.
mx::DecodedProgram program(mx::FloatToInteger value,
                           std::optional<std::uint32_t> register_source) {
    mx::DecodedProgram p;
    p.stage = Stage::Fragment;
    const auto header = shader_input::sph_stage_type_for(Stage::Fragment) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        p.sph[i] = std::byte(header >> (i * 8));
    }
    p.sph[0x48] = std::byte{0xf};
    const auto append = [&p](Family family, mx::InstructionPayload payload) {
        const auto index = p.instructions.size();
        p.instructions.push_back({std::uint32_t((index / 3) * 32 + (index % 3 + 1) * 8), 0, family,
                                  {7, false}, std::move(payload)});
    };
    const auto mov = [&append](unsigned reg, std::uint32_t bits) {
        append(Family::Mov32Immediate,
               mx::Mov32Immediate{{std::uint8_t(reg)}, 15, bits});
    };
    mov(0, 0);
    mov(1, 0);
    mov(2, 0);
    mov(3, 0x3f800000u);
    if (register_source.has_value()) {
        mov(2, *register_source);
    }
    append(value.source.index() == 0 ? Family::FloatToIntegerRegister
                                     : Family::FloatToIntegerConstant,
           std::move(value));
    append(Family::Exit, mx::Exit{15});
    return p;
}

mx::FloatToInteger payload(mx::FloatToIntegerSource source, bool is_signed,
                           std::uint8_t width, mx::RoundingMode rounding) {
    return mx::FloatToInteger{{0}, std::move(source), rounding,
                              mx::FlushMode::FlushToZero, is_signed, width};
}

std::uint32_t evaluate(const mx::DecodedProgram& p,
                       std::vector<std::uint32_t> constants = {}) {
    const auto lowered = gx::lower(p);
    auto* lowered_program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(lowered_program != nullptr);
    if (lowered_program == nullptr) {
        const auto& failure = std::get<gx::Error>(lowered);
        record_failure(__FILE__, __LINE__,
                       std::string("lower failed: ") + failure.message() +
                           " pc=" + std::to_string(failure.pc) +
                           " detail=" + std::to_string(failure.detail));
        return 0;
    }
    gx::Inputs inputs;
    inputs.constant_buffers[0] = std::move(constants);
    const auto result = gx::evaluate(*lowered_program, inputs);
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated != nullptr);
    if (evaluated == nullptr) return 0;
    for (const auto& output : evaluated->outputs) {
        if (output.component == 0) return output.bits;
    }
    ARMIGO_CHECK(false);
    return 0;
}

void lowering_error(const mx::DecodedProgram& p, gx::Error::Code expected) {
    const auto lowered = gx::lower(p);
    const auto* error = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == expected);
    }
}

std::string emit_fragment(const gx::Program& lowered) {
    const auto emitted = gm::emit(lowered, gm::EmitOptions{});
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    return stage == nullptr ? std::string{} : stage->source;
}

bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}

// One-operand FloatToInteger pure node over a literal word, exported as all
// four position lanes of a synthetic vertex graph.
gx::GraphSpec conversion_spec(std::uint32_t word, std::uint32_t immediate) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    const auto header = shader_input::sph_stage_type_for(Stage::Vertex) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        spec.sph[i] = std::byte(header >> (i * 8));
    }
    spec.sph[0x35] = std::byte{0xf0};
    spec.nodes = {{8, gx::Literal{word}},
                  {16, gx::Pure{gx::PureKind::FloatToInteger, {0, 0, 0},
                                immediate}}};
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane, gx::ScalarId{1}});
    }
    spec.blocks = {{0, 2, gx::Return{}}};
    return spec;
}

std::uint32_t convert(const gx::GraphSpec& spec) {
    const auto graph = gx::make_graph(spec);
    const auto* program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return 0;
    const auto result = gx::evaluate(*program, {});
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated != nullptr);
    if (evaluated == nullptr) return 0;
    return evaluated->outputs.front().bits;
}

}  // namespace

// Authored binary32 edges of the signed I32 truncation profile: both zeros,
// subnormals under FTZ, the smallest normal, halves, exact ones, the 2^31
// window, infinities, and NaNs of both signs.
ARMIGO_TEST(float_to_integer, cpu_reference_signed_i32_trunc_edges) {
    struct Case { std::uint32_t word, expected; };
    const std::array<Case, 27> cases{{
        {0x00000000u, 0x00000000u},  // +0
        {0x80000000u, 0x00000000u},  // -0
        {0x00000001u, 0x00000000u},  // smallest subnormal, flushed
        {0x80000001u, 0x00000000u},  // negative smallest subnormal
        {0x007fffffu, 0x00000000u},  // largest subnormal, flushed
        {0x807fffffu, 0x00000000u},  // negative largest subnormal
        {0x00800000u, 0x00000000u},  // smallest normal, 2^-126
        {0x3f000000u, 0x00000000u},  // +0.5 truncates to zero
        {0xbf000000u, 0x00000000u},  // -0.5 truncates to zero
        {0x3f800000u, 0x00000001u},  // 1.0
        {0xbf800000u, 0xffffffffu},  // -1.0
        {0x3fe00000u, 0x00000001u},  // 1.75 truncates toward zero
        {0xbfe00000u, 0xffffffffu},  // -1.75 truncates toward zero
        {0x4effffffu, 0x7fffff80u},  // 2147483520
        {0xceffffffu, 0x80000080u},  // -2147483520
        {0xcf000000u, 0x80000000u},  // -2^31 exactly
        {0xcf000001u, 0x80000000u},  // below -2^31 saturates
        {0x4f000000u, 0x7fffffffu},  // 2^31 saturates positively
        {0x4f000001u, 0x7fffffffu},  // above 2^31 saturates positively
        {0x7f7fffffu, 0x7fffffffu},  // largest finite
        {0x7f800000u, 0x7fffffffu},  // +infinity
        {0xff7fffffu, 0x80000000u},  // most negative finite
        {0xff800000u, 0x80000000u},  // -infinity
        {0x7fc00000u, 0x00000000u},  // quiet NaN
        {0xffc00000u, 0x00000000u},  // negative quiet NaN
        {0x7f800001u, 0x00000000u},  // signalling NaN
        {0x7fffffffu, 0x00000000u},  // NaN with a varied payload
    }};
    for (const auto& case_item : cases) {
        ARMIGO_CHECK_EQ(convert(conversion_spec(case_item.word, i32_signed_trunc)),
                        case_item.expected);
    }
}

// The remaining admitted profiles against their own authored edges.
ARMIGO_TEST(float_to_integer, cpu_reference_remaining_profiles) {
    struct Case { std::uint32_t immediate, word, expected; };
    const std::array<Case, 41> cases{{
        // I32 unsigned, trunc: negative and NaN sources give zero.
        {i32_unsigned_trunc, 0x3f800000u, 0x00000001u},  // 1.0
        {i32_unsigned_trunc, 0x40490fdbu, 0x00000003u},  // pi truncates to 3
        {i32_unsigned_trunc, 0xbf800000u, 0x00000000u},  // -1.0
        {i32_unsigned_trunc, 0xbfe00000u, 0x00000000u},  // -1.75
        {i32_unsigned_trunc, 0x4effffffu, 0x7fffff80u},  // 2147483520
        {i32_unsigned_trunc, 0x4f000000u, 0x80000000u},  // 2^31 stays in range
        {i32_unsigned_trunc, 0x4f7fffffu, 0xffffff00u},  // 2^32 - 256
        {i32_unsigned_trunc, 0x4f800000u, 0xffffffffu},  // 2^32 saturates
        {i32_unsigned_trunc, 0x7f800000u, 0xffffffffu},  // +infinity
        {i32_unsigned_trunc, 0xff800000u, 0x00000000u},  // -infinity
        {i32_unsigned_trunc, 0x7fc00000u, 0x00000000u},  // NaN
        {i32_unsigned_trunc, 0xcf000000u, 0x00000000u},  // -2^31
        {i32_unsigned_trunc, 0x00800000u, 0x00000000u},  // smallest normal
        // I32 signed, floor.
        {i32_signed_floor, 0x3fe00000u, 0x00000001u},  // floor(1.75)
        {i32_signed_floor, 0x3fc00000u, 0x00000001u},  // floor(1.5)
        {i32_signed_floor, 0xbfc00000u, 0xfffffffeu},  // floor(-1.5) = -2
        {i32_signed_floor, 0xbf000000u, 0xffffffffu},  // floor(-0.5) = -1
        {i32_signed_floor, 0x3f000000u, 0x00000000u},  // floor(0.5)
        {i32_signed_floor, 0xbf800000u, 0xffffffffu},  // floor(-1.0)
        {i32_signed_floor, 0x40000000u, 0x00000002u},  // floor(2.0)
        {i32_signed_floor, 0xc0000000u, 0xfffffffeu},  // floor(-2.0)
        {i32_signed_floor, 0x4f000000u, 0x7fffffffu},  // 2^31 saturates
        {i32_signed_floor, 0xceffffffu, 0x80000080u},  // -2147483520
        {i32_signed_floor, 0xff800000u, 0x80000000u},  // -infinity
        {i32_signed_floor, 0x7fc00000u, 0x00000000u},  // NaN
        {i32_signed_floor, 0x00800000u, 0x00000000u},  // smallest normal
        // I32 signed, ceil.
        {i32_signed_ceil, 0x3fc00000u, 0x00000002u},  // ceil(1.5)
        {i32_signed_ceil, 0xbfc00000u, 0xffffffffu},  // ceil(-1.5) = -1
        {i32_signed_ceil, 0xbf000000u, 0x00000000u},  // ceil(-0.5) = -0
        {i32_signed_ceil, 0x3f000000u, 0x00000001u},  // ceil(0.5)
        {i32_signed_ceil, 0xbfe00000u, 0xffffffffu},  // ceil(-1.75) = -1
        {i32_signed_ceil, 0x40000000u, 0x00000002u},  // ceil(2.0)
        {i32_signed_ceil, 0xc0000000u, 0xfffffffeu},  // ceil(-2.0)
        {i32_signed_ceil, 0xff800000u, 0x80000000u},  // -infinity
        {i32_signed_ceil, 0x7fc00000u, 0x00000000u},  // NaN
        // I16 unsigned, round-even: results zero-extend into the register.
        {i16_unsigned_even, 0x3f000000u, 0x00000000u},  // 0.5 ties to even 0
        {i16_unsigned_even, 0x40600000u, 0x00000004u},  // 3.5 ties to even 4
        {i16_unsigned_even, 0x477ffe80u, 0x0000fffeu},  // 65534.5 ties to even
        {i16_unsigned_even, 0x477fffc0u, 0x0000ffffu},  // 65535.75 clamps
        {i16_unsigned_even, 0x7f800000u, 0x0000ffffu},  // +infinity clamps
        {i16_unsigned_even, 0xc0000000u, 0x00000000u},  // -2.0
    }};
    for (const auto& case_item : cases) {
        ARMIGO_CHECK_EQ(convert(conversion_spec(case_item.word, case_item.immediate)),
                        case_item.expected);
    }
}

// The operation admits exactly the five game-profile immediates.
ARMIGO_TEST(float_to_integer, immediate_admission_is_exactly_the_profiles) {
    const gx::detail::OperationEntry* entry = gx::detail::find_operation(
        gx::PureKind::FloatToInteger);
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) return;
    for (std::uint32_t immediate = 0; immediate < 32u; ++immediate) {
        const bool admitted =
            immediate == i16_unsigned_even || immediate == i32_signed_floor ||
            immediate == i32_signed_ceil || immediate == i32_unsigned_trunc ||
            immediate == i32_signed_trunc;
        ARMIGO_CHECK_EQ(entry->admits(immediate), admitted);
    }
    // An unadmitted immediate is a graph format error.
    gx::GraphSpec spec = conversion_spec(0x3f800000u, i32_signed_trunc);
    std::get<gx::Pure>(spec.nodes[1].operation).immediate = 0x15u;
    const auto graph = gx::make_graph(spec);
    const auto* error = std::get_if<gx::Error>(&graph);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::GraphFormat);
    }
}

// The register decoder admits the profile words and rejects every other
// field value with its structured field.
ARMIGO_TEST(float_to_integer, register_decoder_admits_and_rejects) {
    const auto value = admitted(decode(f2i_register_word()),
                                 Family::FloatToIntegerRegister);
    ARMIGO_CHECK(value.has_value());
    if (!value) return;
    ARMIGO_CHECK(value->destination.index == 3);
    const auto* source = std::get_if<mx::RegisterOperand>(&value->source);
    ARMIGO_CHECK(source != nullptr && source->index == 2);
    ARMIGO_CHECK(value->is_signed);
    ARMIGO_CHECK(value->destination_width == 32);
    ARMIGO_CHECK(value->rounding == mx::RoundingMode::Zero);
    ARMIGO_CHECK(value->flush == mx::FlushMode::FlushToZero);

    // The 16-bit unsigned round-even profile decodes its width.
    const auto i16_value = admitted(
        decode(f2i_register_word(6, 5, 1, 0, 0)), Family::FloatToIntegerRegister);
    ARMIGO_CHECK(i16_value.has_value());
    if (i16_value.has_value()) {
        ARMIGO_CHECK(!i16_value->is_signed);
        ARMIGO_CHECK(i16_value->destination_width == 16);
        ARMIGO_CHECK(i16_value->rounding == mx::RoundingMode::NearestEven);
    }
    // The floor and ceil profiles carry their rounding mode.
    const auto floor_value = admitted(
        decode(f2i_register_word(3, 2, 2, 1, 1)), Family::FloatToIntegerRegister);
    ARMIGO_CHECK(floor_value.has_value());
    if (floor_value.has_value()) {
        ARMIGO_CHECK(floor_value->rounding == mx::RoundingMode::NegativeInfinity);
    }
    const auto ceil_value = admitted(
        decode(f2i_register_word(3, 2, 2, 1, 2)), Family::FloatToIntegerRegister);
    ARMIGO_CHECK(ceil_value.has_value());
    if (ceil_value.has_value()) {
        ARMIGO_CHECK(ceil_value->rounding == mx::RoundingMode::PositiveInfinity);
    }

    struct Rule { unsigned shift, width; std::uint64_t value; };
    const mx::DecodeError::Code modifier = mx::DecodeError::Code::UnsupportedModifier;
    // Source format must be F32; other encodings are rejected.
    for (const std::uint64_t src_format : {std::uint64_t{0}, std::uint64_t{1},
                                           std::uint64_t{3}}) {
        const auto error = rejected(
            decode(set(f2i_register_word(), 10, 2, src_format)), modifier,
            Family::FloatToIntegerRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == mx::DecodeField::SourceFormat);
        }
    }
    // Destination format must be I16 or I32; invalid and I64 are rejected.
    for (const std::uint64_t dest_format : {std::uint64_t{0}, std::uint64_t{3}}) {
        const auto error = rejected(
            decode(set(f2i_register_word(), 8, 2, dest_format)), modifier,
            Family::FloatToIntegerRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == mx::DecodeField::DestinationFormat);
        }
    }
    // Modifier profile rejections with their fields.
    const std::array<Rule, 5> modifiers{{
        Rule{41, 1, 1},  // half select
        Rule{44, 1, 0},  // flush-to-zero required
        Rule{45, 1, 1},  // absolute
        Rule{47, 1, 1},  // condition code
        Rule{49, 1, 1},  // negate
    }};
    const std::array<mx::DecodeField, 5> fields{
        mx::DecodeField::SourceHalf, mx::DecodeField::FlushMode,
        mx::DecodeField::AbsoluteA, mx::DecodeField::ConditionCode,
        mx::DecodeField::NegateA};
    for (std::size_t index = 0; index < modifiers.size(); ++index) {
        const auto& rule = modifiers[index];
        const auto error = rejected(
            decode(set(f2i_register_word(), rule.shift, rule.width, rule.value)),
            modifier, Family::FloatToIntegerRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == fields[index]);
        }
    }
    // Stray bits anywhere outside the admitted fields are reserved-bit
    // errors, including the wildcard low bits of the opcode window.
    for (const unsigned bit : {13u, 14u, 15u, 28u, 29u, 30u, 31u, 32u, 33u, 34u, 35u,
                               36u, 37u, 38u, 42u, 43u, 46u, 48u, 50u}) {
        const auto error = rejected(
            decode(f2i_register_word() | (std::uint64_t{1} << bit)), modifier,
            Family::FloatToIntegerRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field == mx::DecodeField::ReservedBits);
        }
    }
}

// Only the five (destination, signedness, rounding) combinations are
// admitted; neighbouring combinations are explicit modifier errors.
ARMIGO_TEST(float_to_integer, decoder_rejects_unlisted_profiles) {
    const mx::DecodeError::Code modifier = mx::DecodeError::Code::UnsupportedModifier;
    struct Case { std::uint64_t dest_format, is_signed, rounding; };
    const std::array<Case, 6> cases{{
        {2, 1, 0},  // I32 signed round-even
        {2, 0, 0},  // I32 unsigned round-even
        {2, 0, 1},  // I32 unsigned floor
        {2, 0, 2},  // I32 unsigned ceil
        {1, 1, 0},  // I16 signed round-even
        {1, 0, 3},  // I16 unsigned trunc
    }};
    for (const auto& case_item : cases) {
        const auto error = rejected(
            decode(f2i_register_word(3, 2, case_item.dest_format,
                                     case_item.is_signed, case_item.rounding)),
            modifier, Family::FloatToIntegerRegister);
        if (error.has_value()) {
            ARMIGO_CHECK(error->field ==
                (case_item.dest_format == 1 && case_item.is_signed
                     ? mx::DecodeField::Signedness : mx::DecodeField::Rounding));
        }
    }
}

// The constant-buffer decoder reads one buffer word and shares the register
// profile's modifier and combination rejections.
ARMIGO_TEST(float_to_integer, constant_decoder_admits_and_rejects) {
    const auto value = admitted(decode(f2i_constant_word()),
                                 Family::FloatToIntegerConstant);
    ARMIGO_CHECK(value.has_value());
    if (!value) return;
    ARMIGO_CHECK(value->destination.index == 3);
    const auto* source = std::get_if<mx::ConstantBufferOperand>(&value->source);
    ARMIGO_CHECK(source != nullptr);
    if (source != nullptr) {
        ARMIGO_CHECK(source->binding == 4);
        ARMIGO_CHECK(source->byte_offset == 9 * 4);
    }
    ARMIGO_CHECK(value->is_signed);
    ARMIGO_CHECK(value->destination_width == 32);
    ARMIGO_CHECK(value->rounding == mx::RoundingMode::Zero);

    const mx::DecodeError::Code modifier = mx::DecodeError::Code::UnsupportedModifier;
    // Out-of-bounds constant buffer bindings are rejected.
    const auto binding_error = rejected(
        decode(f2i_constant_word(3, 9, 18)), modifier,
        Family::FloatToIntegerConstant);
    if (binding_error.has_value()) {
        ARMIGO_CHECK(binding_error->field == mx::DecodeField::ConstantBinding);
    }
    // The shared modifier profile: half, flush, absolute, CC, negate, and
    // stray bits reject exactly as on the register form.
    const auto half_error = rejected(
        decode(set(f2i_constant_word(), 41, 1, 1)), modifier,
        Family::FloatToIntegerConstant);
    if (half_error.has_value()) {
        ARMIGO_CHECK(half_error->field == mx::DecodeField::SourceHalf);
    }
    const auto flush_error = rejected(
        decode(set(f2i_constant_word(), 44, 1, 0)), modifier,
        Family::FloatToIntegerConstant);
    if (flush_error.has_value()) {
        ARMIGO_CHECK(flush_error->field == mx::DecodeField::FlushMode);
    }
    const auto stray_error = rejected(
        decode(f2i_constant_word() | (std::uint64_t{1} << 46)), modifier,
        Family::FloatToIntegerConstant);
    if (stray_error.has_value()) {
        ARMIGO_CHECK(stray_error->field == mx::DecodeField::ReservedBits);
    }
    // An unlisted profile combination rejects on the rounding field.
    const auto profile_error = rejected(
        decode(f2i_constant_word(3, 9, 4, 2, 1, 0)), modifier,
        Family::FloatToIntegerConstant);
    if (profile_error.has_value()) {
        ARMIGO_CHECK(profile_error->field == mx::DecodeField::Rounding);
    }
}

// Lowering of both source forms runs the registered operation and honours
// the authored conversion edges in a decoded fragment program.
ARMIGO_TEST(float_to_integer, lowering_evaluates_both_source_forms) {
    const mx::FloatToInteger trunc_register =
        payload(mx::RegisterOperand{2}, true, 32, mx::RoundingMode::Zero);
    ARMIGO_CHECK_EQ(evaluate(program(trunc_register, 0x3fe00000u), {}),
                    0x00000001u);  // trunc(1.75)
    ARMIGO_CHECK_EQ(evaluate(program(trunc_register, 0xbfe00000u), {}),
                    0xffffffffu);  // trunc(-1.75)
    ARMIGO_CHECK_EQ(evaluate(program(trunc_register, 0x4f000000u), {}),
                    0x7fffffffu);  // 2^31 saturates positively
    ARMIGO_CHECK_EQ(evaluate(program(trunc_register, 0x7fc00000u), {}),
                    0x00000000u);  // NaN gives zero

    const mx::FloatToInteger floor_register =
        payload(mx::RegisterOperand{2}, true, 32, mx::RoundingMode::NegativeInfinity);
    ARMIGO_CHECK_EQ(evaluate(program(floor_register, 0xbfc00000u), {}),
                    0xfffffffeu);  // floor(-1.5) = -2

    const mx::FloatToInteger even_i16 =
        payload(mx::RegisterOperand{2}, false, 16, mx::RoundingMode::NearestEven);
    ARMIGO_CHECK_EQ(evaluate(program(even_i16, 0x477ffe80u), {}),
                    0x0000fffeu);  // 65534.5 ties to even

    // The constant-buffer form reads the word at its decoded offset.
    const mx::FloatToInteger trunc_constant = payload(
        mx::ConstantBufferOperand{0, 36}, true, 32, mx::RoundingMode::Zero);
    ARMIGO_CHECK_EQ(evaluate(program(trunc_constant, std::nullopt),
                             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3f800000u}),
                    0x00000001u);  // trunc(buffer word 1.0)
    ARMIGO_CHECK_EQ(evaluate(program(trunc_constant, std::nullopt),
                             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7f800000u}),
                    0x7fffffffu);  // +infinity saturates
}

// Forged payloads fail with structured errors instead of dropping a
// modifier.
ARMIGO_TEST(float_to_integer, forged_payloads_are_structured_errors) {
    mx::FloatToInteger wrong_flush =
        payload(mx::RegisterOperand{2}, true, 32, mx::RoundingMode::Zero);
    wrong_flush.flush = mx::FlushMode::None;
    lowering_error(program(wrong_flush, 0x3f800000u),
                   gx::Error::Code::UnsupportedModifier);

    mx::FloatToInteger unknown_width =
        payload(mx::RegisterOperand{2}, true, 8, mx::RoundingMode::Zero);
    lowering_error(program(unknown_width, 0x3f800000u),
                   gx::Error::Code::UnsupportedModifier);

    // A payload that admits no operation immediate is an invalid
    // instruction at its pc.
    mx::FloatToInteger unlisted_profile =
        payload(mx::RegisterOperand{2}, true, 32, mx::RoundingMode::NearestEven);
    lowering_error(program(unlisted_profile, 0x3f800000u),
                   gx::Error::Code::InvalidInstruction);
}

// The emitted MSL names the shared helper and a program without the
// operation keeps its source free of it.
ARMIGO_TEST(float_to_integer, emitted_msl_uses_the_shared_helper) {
    mx::FloatToInteger trunc_register =
        payload(mx::RegisterOperand{2}, true, 32, mx::RoundingMode::Zero);
    const auto lowered = gx::lower(program(trunc_register, 0x3fe00000u));
    auto* lowered_program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(lowered_program != nullptr);
    if (lowered_program == nullptr) return;
    const std::string source = emit_fragment(*lowered_program);
    ARMIGO_CHECK(contains(
        source, "inline uint armigo_float_to_integer(uint word, uint profile)"));
    ARMIGO_CHECK(contains(source, ", 0x0000001bu)"));

    // A fragment program without the operation does not ship the helper.
    mx::DecodedProgram plain;
    plain.stage = Stage::Fragment;
    const auto header = shader_input::sph_stage_type_for(Stage::Fragment) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        plain.sph[i] = std::byte(header >> (i * 8));
    }
    plain.sph[0x48] = std::byte{0xf};
    for (unsigned reg = 0; reg < 4; ++reg) {
        plain.instructions.push_back({(reg / 3) * 32 + (reg % 3 + 1) * 8, 0, Family::Mov32Immediate,
            {7, false}, mx::Mov32Immediate{{std::uint8_t(reg)}, 15, 0x3f800000u}});
    }
    plain.instructions.push_back({48, 0, Family::Exit, {7, false}, mx::Exit{15}});
    const auto plain_lowered = gx::lower(plain);
    auto* plain_program = std::get_if<gx::Program>(&plain_lowered);
    ARMIGO_CHECK(plain_program != nullptr);
    if (plain_program == nullptr) return;
    ARMIGO_CHECK(!contains(emit_fragment(*plain_program),
                           "armigo_float_to_integer"));
}

ARMIGO_TEST(float_to_integer, tiny_normals_and_saturation_exponent_sweep) {
    const auto* entry = gx::detail::find_operation(gx::PureKind::FloatToInteger);
    ARMIGO_CHECK(entry != nullptr);
    if (!entry) return;
    for (const auto profile : {0u, 11u, 19u, 26u, 27u}) {
        for (std::uint32_t exponent = 0; exponent < 126; ++exponent) {
            for (const auto fraction : {0u, 1u, 0x3fffffu, 0x7fffffu}) {
                for (const auto sign : {0u, 0x80000000u}) {
                    const auto word = sign | (exponent << 23) | fraction;
                    const auto expected = exponent == 0 ? 0u
                        : profile == 11 && sign != 0 ? 0xffffffffu
                        : profile == 19 && sign == 0 ? 1u : 0u;
                    ARMIGO_CHECK_EQ(entry->evaluate({word, 0, 0}, profile), expected);
                }
            }
        }
        for (std::uint32_t exponent = 159; exponent <= 255; ++exponent) {
            for (const auto fraction : {0u, 1u, 0x400001u, 0x7fffffu}) {
                for (const auto sign : {0u, 0x80000000u}) {
                    const auto word = sign | (exponent << 23) | fraction;
                    const bool nan = exponent == 255 && fraction != 0;
                    const auto expected = nan ? 0u : (profile & 1u) != 0
                        ? (sign ? 0x80000000u : 0x7fffffffu)
                        : sign ? 0u : profile == 0 ? 65535u : 0xffffffffu;
                    ARMIGO_CHECK_EQ(entry->evaluate({word, 0, 0}, profile), expected);
                }
            }
        }
    }
    for (const auto word : {0x3effffffu, 0x3f000000u, 0x3f000001u,
                            0x3fc00000u, 0x40200000u, 0x40600000u}) {
        const auto expected = word <= 0x3f000000u ? 0u
            : word == 0x3f000001u ? 1u : word == 0x40600000u ? 4u : 2u;
        ARMIGO_CHECK_EQ(convert(conversion_spec(word, 0)), expected);
    }
}

ARMIGO_TEST(float_to_integer, complete_profile_matrix_and_unknown_neighbors) {
    for (const bool constant : {false, true}) {
        const auto family = constant ? Family::FloatToIntegerConstant
                                     : Family::FloatToIntegerRegister;
        for (unsigned width = 0; width < 4; ++width) {
            for (unsigned sign = 0; sign < 2; ++sign) {
                for (unsigned rounding = 0; rounding < 4; ++rounding) {
                    const auto word = constant
                        ? f2i_constant_word(3, 9, 4, width, sign, rounding)
                        : f2i_register_word(3, 2, width, sign, rounding);
                    const bool valid = (width == 1 && sign == 0 && rounding == 0) ||
                        (width == 2 && (sign ? rounding != 0 : rounding == 3));
                    if (valid) {
                        ARMIGO_CHECK(admitted(decode(word), family).has_value());
                    } else {
                        const auto error = rejected(decode(word),
                            mx::DecodeError::Code::UnsupportedModifier, family);
                        if (error) {
                            ARMIGO_CHECK(error->field ==
                                (width == 0 || width == 3 ? mx::DecodeField::DestinationFormat
                                 : width == 1 && sign ? mx::DecodeField::Signedness
                                 : mx::DecodeField::Rounding));
                            ARMIGO_CHECK_EQ(error->pc, 8u);
                        }
                    }
                }
            }
        }
        const auto base = constant ? f2i_constant_word() : f2i_register_word();
        for (const auto format : {0u, 1u, 3u}) {
            const auto error = rejected(decode(set(base, 10, 2, format)),
                mx::DecodeError::Code::UnsupportedModifier, family);
            if (error) ARMIGO_CHECK(error->field == mx::DecodeField::SourceFormat);
        }
        for (const auto window : {0x5c88u, 0x4c88u, 0x38b0u}) {
            rejected(decode(set(base, 48, 16, window)),
                     mx::DecodeError::Code::UnknownOpcode);
        }
    }
    for (std::uint32_t bit = 5; bit < 32; ++bit) {
        const auto* entry = gx::detail::find_operation(gx::PureKind::FloatToInteger);
        ARMIGO_CHECK(!entry->admits(i32_signed_trunc | (1u << bit)));
    }
}

ARMIGO_TEST(float_to_integer, lowering_alias_rz_and_source_validation) {
    auto conversion = payload(mx::RegisterOperand{0}, true, 32, mx::RoundingMode::Zero);
    auto alias = program(conversion, std::nullopt);
    std::get<mx::Mov32Immediate>(alias.instructions[0].payload).immediate_bits = 0x3fe00000u;
    ARMIGO_CHECK_EQ(evaluate(alias), 1u);

    conversion.source = mx::RegisterOperand{255};
    ARMIGO_CHECK_EQ(evaluate(program(conversion, std::nullopt)), 0u);
    conversion.destination = {255};
    const auto discarded = gx::lower(program(conversion, std::nullopt));
    const auto* discarded_program = std::get_if<gx::Program>(&discarded);
    ARMIGO_CHECK(discarded_program != nullptr);
    if (discarded_program) {
        std::size_t conversions = 0;
        for (const auto& node : discarded_program->nodes()) {
            if (const auto* op = std::get_if<gx::Pure>(&node.operation)) {
                if (op->kind == gx::PureKind::FloatToInteger) ++conversions;
            }
        }
        ARMIGO_CHECK_EQ(conversions, std::size_t{1});
    }
    conversion.destination = {0};
    conversion.source = mx::RegisterOperand{9};
    lowering_error(program(conversion, std::nullopt), gx::Error::Code::UninitializedRegister);

    auto mismatch = program(conversion, std::nullopt);
    mismatch.instructions[mismatch.instructions.size() - 2].family = Family::FloatToIntegerConstant;
    lowering_error(mismatch, gx::Error::Code::InvalidInstruction);
    conversion.source = mx::ConstantBufferOperand{0, 0};
    mismatch = program(conversion, std::nullopt);
    mismatch.instructions[mismatch.instructions.size() - 2].family = Family::FloatToIntegerRegister;
    lowering_error(mismatch, gx::Error::Code::InvalidInstruction);
}

}  // namespace armigo::shader::test
