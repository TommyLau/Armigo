// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// I2F: decoding of the register and constant-buffer forms, every rejected
// modifier field, the CPU reference against an independent oracle over
// power-of-two boundaries and a fixed pseudo-random set, lowering through
// the decoded-program path, and the emitted MSL. Synthetic words, programs,
// and graphs only.
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/maxwell.h>
#include "test_support.h"

#include "graphics_operations.h"

#include <armigo/shader/graphics_msl.h>

#include <cmath>
#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace mx = maxwell;
namespace gx = graphics;
namespace gm = graphics_msl;
namespace gd = graphics::detail;
namespace si = shader_input;
using Family = mx::OpcodeFamily;

// Pure-operation immediates of the five admitted profiles: bit 0 signed,
// bits 1..2 width code, bits 3..4 rounding, bit 5 absolute.
constexpr std::uint32_t kSigned32Rn = 0x05u;
constexpr std::uint32_t kUnsigned32Rz = 0x1cu;
constexpr std::uint32_t kUnsigned32Rn = 0x04u;
constexpr std::uint32_t kSigned32RnAbsolute = 0x25u;
constexpr std::uint32_t kSigned16Rn = 0x03u;

std::uint64_t i2f_word(std::uint16_t high, unsigned destination, unsigned float_format,
                       unsigned int_format, bool is_signed, std::uint64_t source_field,
                       unsigned rounding, unsigned selector, bool negate, bool cc,
                       bool absolute) {
    return (std::uint64_t{high} << 48) | (std::uint64_t{7} << 16) |
           (std::uint64_t{absolute ? 1u : 0u} << 49) | (std::uint64_t{cc ? 1u : 0u} << 47) |
           (std::uint64_t{negate ? 1u : 0u} << 45) | (std::uint64_t{selector} << 41) |
           (std::uint64_t{rounding} << 39) | ((source_field & 0xfffffu) << 20) |
           (std::uint64_t{int_format} << 10) | (std::uint64_t{float_format} << 8) |
           (std::uint64_t{is_signed ? 1u : 0u} << 13) | destination;
}

std::uint64_t mov32i(unsigned destination, std::uint32_t bits) {
    return (std::uint64_t{0x0100} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{15} << 12) |
           (std::uint64_t{bits} << 20) | destination;
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

// Vertex program: R1 = seed word, the middle instructions, then R3 stored to
// the four position lanes and EXIT.
mx::DecodeResult decode(const std::vector<std::uint64_t>& middle) {
    std::vector<std::uint64_t> words{mov32i(1, 0x12345678u)};
    words.insert(words.end(), middle.begin(), middle.end());
    for (std::uint64_t offset : {0x70u, 0x74u, 0x78u, 0x7cu}) {
        words.push_back((std::uint64_t{0xeff0} << 48) | (std::uint64_t{7} << 16) |
                        (offset << 20) | (std::uint64_t{0xff} << 8) |
                        (std::uint64_t{0xff} << 39) | 3);
    }
    words.push_back((std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15);
    const unsigned end = 0x80 + static_cast<unsigned>((words.size() + 2) / 3) * 32;
    std::vector<std::uint8_t> control(0x718), code(end);
    put(control, 0, si::format::control_magic, 4);
    put(control, 4, si::format::control_tag, 4);
    put(control, 8, 14, 4);
    put(control, 0x20, control.size(), 4);
    put(control, 0x700, end, 4);
    put(control, 0x704, end, 4);
    put(control, 0x708, 0x30, 4);
    put(code, 0, 0x12345678, 4);
    put(code, 0x30, si::sph_stage_type_for(si::Stage::Vertex) << 10, 4);
    code[0x30 + 0x35] = 0xf0;
    for (std::size_t i = 0; i < words.size(); ++i)
        put(code, static_cast<unsigned>(0x80 + (i / 3) * 32 + 8 + (i % 3) * 8), words[i], 8);
    return mx::decode({control.data(), control.size()}, {code.data(), code.size()});
}

// The payload is copied out: the decoded program dies with this scope.
std::optional<mx::IntegerToFloat> decode_one(std::uint16_t high, std::uint64_t source_field) {
    const auto decoded = decode({i2f_word(high, 3, 2, 2, true, source_field, 0, 0, false,
                                          false, false)});
    const auto* program = std::get_if<mx::DecodedProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return std::nullopt;
    const auto* value = std::get_if<mx::IntegerToFloat>(&program->instructions[1].payload);
    return value == nullptr ? std::nullopt : std::optional{*value};
}

// Decodes one instruction with an arbitrary modifier combination.
std::optional<mx::IntegerToFloat> decode_modifiers(unsigned int_format, bool is_signed,
                                                   unsigned rounding, bool absolute) {
    const auto decoded = decode({i2f_word(0x5cb8, 3, 2, int_format, is_signed, 1, rounding, 0,
                                          false, false, absolute)});
    const auto* program = std::get_if<mx::DecodedProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return std::nullopt;
    const auto* value = std::get_if<mx::IntegerToFloat>(&program->instructions[1].payload);
    return value == nullptr ? std::nullopt : std::optional{*value};
}

// Vertex SPH: stage header and a position-output mask.
std::array<std::byte, 80> vertex_sph(std::uint8_t position_mask) {
    std::array<std::byte, 80> data{};
    const auto header = si::sph_stage_type_for(si::Stage::Vertex) << 10;
    for (unsigned byte = 0; byte < 4; ++byte) {
        data[byte] = std::byte(static_cast<unsigned char>(header >> (8 * byte)));
    }
    data[0x35] = std::byte{position_mask};
    return data;
}

// The position-x word after decoding, lowering, and evaluating the program.
std::uint32_t run(const std::vector<std::uint64_t>& middle,
                  std::vector<std::uint32_t> constants = {}) {
    const auto decoded = decode(middle);
    const auto* program = std::get_if<mx::DecodedProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return 0xdeadbeefu;
    const auto lowered = gx::lower(*program);
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (graph == nullptr) return 0xdeadbeefu;
    gx::Inputs inputs;
    if (!constants.empty()) inputs.constant_buffers[0] = constants;
    const auto evaluated = gx::evaluate(*graph, inputs);
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return 0xdeadbeefu;
    for (const auto& output : outputs->outputs)
        if (output.kind == gx::OutputKind::Position && output.component == 0) return output.bits;
    return 0xdeadbeefu;
}

std::uint32_t convert(std::uint32_t word, std::uint32_t immediate) {
    return gd::evaluate_integer_to_float({word, 0, 0}, immediate);
}

// Independent oracle: decompose the magnitude with frexp and floor on
// doubles (exact below 2^53), round in the requested mode, and assemble the
// IEEE bits. Shares no algorithm with the shift-based reference.
std::uint32_t oracle_bits(std::uint32_t magnitude, bool round_to_zero, bool negative) {
    if (magnitude == 0) return 0;
    int exponent = 0;
    const double fraction = std::frexp(static_cast<double>(magnitude), &exponent);
    const double scaled = fraction * 16777216.0;  // The mantissa in [2^23, 2^24).
    double units = std::floor(scaled);
    if (!round_to_zero) {
        const double rest = scaled - units;
        if (rest > 0.5 || (rest == 0.5 && std::fmod(units, 2.0) == 1.0)) units += 1.0;
    }
    unsigned p = static_cast<unsigned>(exponent) - 1;
    if (units == 16777216.0) {
        p += 1;  // Rounding carried past the mantissa.
        units = 8388608.0;
    }
    const std::uint32_t bits =
        ((127u + p) << 23) | static_cast<std::uint32_t>(units - 8388608.0);
    return negative ? bits | 0x80000000u : bits;
}

// Signed conversion of the raw word, with the optional wraparound absolute.
std::uint32_t oracle_signed(std::uint32_t word, bool round_to_zero, bool absolute) {
    if ((word & 0x80000000u) == 0) return oracle_bits(word, round_to_zero, false);
    if (absolute) {
        const std::uint32_t flipped = 0u - word;
        const bool still_negative = (flipped & 0x80000000u) != 0;  // Only the minimum.
        return oracle_bits(flipped, round_to_zero, still_negative);
    }
    return oracle_bits(0u - word, round_to_zero, true);
}

std::uint32_t oracle_unsigned(std::uint32_t word, bool round_to_zero) {
    return oracle_bits(word, round_to_zero, false);
}

std::vector<std::uint32_t> samples() {
    std::vector<std::uint32_t> result{0u, 1u, 2u, 0xfffffffeu, 0xffffffffu};
    for (unsigned p = 0; p < 32; ++p) {
        const std::uint64_t center = std::uint64_t{1} << p;
        for (std::uint64_t offset : {1u, 2u, 3u}) {
            if (center > offset) result.push_back(static_cast<std::uint32_t>(center - offset));
        }
        for (std::uint64_t offset : {0u, 1u, 2u, 3u}) {
            const std::uint64_t value = center + offset;
            if (value <= 0xffffffffull) result.push_back(static_cast<std::uint32_t>(value));
        }
    }
    // Fixed linear congruential stream, independent of the reference.
    std::uint32_t state = 0x12345678u;
    for (unsigned i = 0; i < 8192; ++i) {
        state = state * 1664525u + 1013904223u;
        result.push_back(state);
    }
    return result;
}

std::string emit_graph(const gx::GraphSpec& spec) {
    const auto graph = gx::make_graph(spec);
    const auto* program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return {};
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted = gm::emit(*program, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    return stage == nullptr ? std::string{} : stage->source;
}

}  // namespace

ARMIGO_TEST(integer_to_float, decodes_register_and_constant_forms) {
    const auto value = decode_one(0x5cb8, 1);
    ARMIGO_CHECK(value.has_value());
    if (value.has_value()) {
        ARMIGO_CHECK(value->destination.index == 3);
        ARMIGO_CHECK(std::get<mx::RegisterOperand>(value->source).index == 1);
        ARMIGO_CHECK(value->is_signed && value->source_width == 32 &&
                     value->rounding == mx::RoundingMode::NearestEven && !value->absolute);
    }
    // Unsigned toward-zero profile.
    const auto unsigned_rz = decode_modifiers(2, false, 3, false);
    ARMIGO_CHECK(unsigned_rz.has_value());
    if (unsigned_rz.has_value()) {
        ARMIGO_CHECK(!unsigned_rz->is_signed &&
                     unsigned_rz->rounding == mx::RoundingMode::Zero &&
                     unsigned_rz->source_width == 32);
    }
    // Constant-buffer form: binding 1, byte offset 8.
    const auto buffer = decode_one(0x4cb8, (1u << 14) | 2u);
    ARMIGO_CHECK(buffer.has_value());
    if (buffer.has_value()) {
        const auto& operand = std::get<mx::ConstantBufferOperand>(buffer->source);
        ARMIGO_CHECK(operand.binding == 1 && operand.byte_offset == 8);
    }
    // Signed 16-bit profile (integer format 1, selector 0).
    const auto half = decode_modifiers(1, true, 0, false);
    ARMIGO_CHECK(half.has_value() && half->is_signed && half->source_width == 16);
    // Absolute inside the signed 32-bit nearest-even profile.
    const auto absolute = decode_modifiers(2, true, 0, true);
    ARMIGO_CHECK(absolute.has_value() && absolute->absolute);
}

ARMIGO_TEST(integer_to_float, rejects_every_unadmitted_modifier) {
    struct Rejection {
        std::uint16_t high;
        unsigned float_format, int_format, rounding, selector;
        bool is_signed, negate, cc, absolute;
        mx::DecodeField field;
    };
    const std::uint64_t source = 1;
    const std::vector<Rejection> cases{
        // Float format: 0 is undefined, F16 and F64 are not decoded.
        {0x5cb8, 0, 2, 0, 0, true, false, false, false, mx::DecodeField::FloatFormat},
        {0x5cb8, 1, 2, 0, 0, true, false, false, false, mx::DecodeField::FloatFormat},
        {0x5cb8, 3, 2, 0, 0, true, false, false, false, mx::DecodeField::FloatFormat},
        // Integer format: U8 and U64 are not decoded.
        {0x5cb8, 2, 0, 0, 0, true, false, false, false, mx::DecodeField::IntegerFormat},
        {0x5cb8, 2, 3, 0, 0, true, false, false, false, mx::DecodeField::IntegerFormat},
        // Rounding: only nearest-even and toward-zero are decoded.
        {0x5cb8, 2, 2, 1, 0, true, false, false, false, mx::DecodeField::Rounding},
        {0x5cb8, 2, 2, 2, 0, true, false, false, false, mx::DecodeField::Rounding},
        // Selector: only zero.
        {0x5cb8, 2, 2, 0, 1, true, false, false, false, mx::DecodeField::IntegerSelector},
        {0x5cb8, 2, 2, 0, 3, true, false, false, false, mx::DecodeField::IntegerSelector},
        // Negate and CC are rejected outright.
        {0x5cb8, 2, 2, 0, 0, true, true, false, false, mx::DecodeField::NegateB},
        {0x5cb8, 2, 2, 0, 0, true, false, true, false, mx::DecodeField::ConditionCode},
        // Absolute only inside signed 32-bit nearest-even.
        {0x5cb8, 2, 2, 0, 0, false, false, false, true, mx::DecodeField::AbsoluteB},
        {0x5cb8, 2, 2, 3, 0, true, false, false, true, mx::DecodeField::AbsoluteB},
        {0x5cb8, 2, 1, 0, 0, true, false, false, true, mx::DecodeField::AbsoluteB},
    };
    for (const auto& item : cases) {
        const auto decoded =
            decode({i2f_word(item.high, 3, item.float_format, item.int_format, item.is_signed,
                             source, item.rounding, item.selector, item.negate, item.cc,
                             item.absolute)});
        const auto* error = std::get_if<mx::DecodeError>(&decoded);
        ARMIGO_CHECK(error != nullptr &&
                     error->code == mx::DecodeError::Code::UnsupportedModifier);
        if (error != nullptr) {
            ARMIGO_CHECK(error->field == item.field);
        }
    }
    // Stray bits outside any field.
    for (const unsigned bit : {12u, 14u, 28u, 43u, 44u, 46u, 48u}) {
        const auto decoded = decode({i2f_word(0x5cb8, 3, 2, 2, true, source, 0, 0, false,
                                              false, false) |
                                     (std::uint64_t{1} << bit)});
        const auto* error = std::get_if<mx::DecodeError>(&decoded);
        ARMIGO_CHECK(error != nullptr &&
                     error->code == mx::DecodeError::Code::UnsupportedModifier &&
                     error->field == mx::DecodeField::ReservedBits);
    }
    // The immediate form is not registered.
    const auto imm = decode({i2f_word(0x39b8, 3, 2, 2, true, 1, 0, 0, false, false, false)});
    ARMIGO_CHECK(std::get_if<mx::DecodeError>(&imm) != nullptr);
}

ARMIGO_TEST(integer_to_float, reference_matches_the_oracle_everywhere) {
    const auto all = samples();
    for (const std::uint32_t word : all) {
        ARMIGO_CHECK_EQ(convert(word, kUnsigned32Rn), oracle_unsigned(word, false));
        ARMIGO_CHECK_EQ(convert(word, kUnsigned32Rz), oracle_unsigned(word, true));
        ARMIGO_CHECK_EQ(convert(word, kSigned32Rn), oracle_signed(word, false, false));
        ARMIGO_CHECK_EQ(convert(word, kSigned32RnAbsolute), oracle_signed(word, false, true));
        // Signed 16 sees the low halfword only.
        const std::uint32_t half = word & 0xffffu;
        const bool negative = (half & 0x8000u) != 0;
        const std::uint32_t value = negative ? (0xffff0000u | half) : half;
        const std::uint32_t magnitude = negative ? 0u - value : value;
        ARMIGO_CHECK_EQ(convert(word, kSigned16Rn),
                        oracle_bits(magnitude, false, negative));
    }
    // The minimum integer converts to itself under the absolute modifier,
    // matching the wraparound absolute value of the hardware profile.
    ARMIGO_CHECK_EQ(convert(0x80000000u, kSigned32RnAbsolute), 0xcf000000u);
    ARMIGO_CHECK_EQ(convert(0x80000000u, kSigned32Rn), 0xcf000000u);
    ARMIGO_CHECK_EQ(convert(0x80000001u, kSigned32RnAbsolute), 0x4f000000u);
    ARMIGO_CHECK_EQ(convert(0xffffffffu, kSigned32Rn), 0xbf800000u);
    // Rounding pins the 2^32 boundary: toward-zero truncates, nearest-even
    // carries to the next power of two.
    ARMIGO_CHECK_EQ(convert(0xffffffffu, kUnsigned32Rz), 0x4f7fffffu);
    ARMIGO_CHECK_EQ(convert(0xffffffffu, kUnsigned32Rn), 0x4f800000u);
    ARMIGO_CHECK_EQ(convert(0x00000000u, kSigned32Rn), 0x00000000u);
    ARMIGO_CHECK_EQ(convert(0x00008000u, kSigned16Rn), 0xc7000000u);
}

ARMIGO_TEST(integer_to_float, graphs_reject_unadmitted_profiles) {
    const auto spec = [&](std::uint32_t immediate) {
        gx::GraphSpec value;
        value.stage = si::Stage::Vertex;
        value.sph = vertex_sph(0x10);  // Position x only.
        value.nodes = {{8, gx::Literal{7}},
                       {16, gx::Pure{gx::PureKind::IntegerToFloat, {0, 0, 0}, immediate}}};
        value.exports = {{gx::OutputKind::Position, 0, gx::ScalarId{1}}};
        value.blocks = {{0, 2, gx::Return{}}};
        return value;
    };
    for (const std::uint32_t immediate :
         {0u, 1u, 2u, 7u, 0x0du, 0x1du, 0x24u, 0x3cu, 0x40u, 0x80000000u}) {
        const auto graph = gx::make_graph(spec(immediate));
        const auto* error = std::get_if<gx::Error>(&graph);
        ARMIGO_CHECK(error != nullptr && error->code == gx::Error::Code::GraphFormat);
    }
    // The admitted profiles build, and the evaluator agrees with the oracle.
    for (const std::uint32_t immediate :
         {kSigned32Rn, kUnsigned32Rz, kUnsigned32Rn, kSigned32RnAbsolute, kSigned16Rn}) {
        const auto graph = gx::make_graph(spec(immediate));
        const auto* program = std::get_if<gx::Program>(&graph);
        if (program == nullptr) {
            const auto* failure = std::get_if<gx::Error>(&graph);
            std::fprintf(stderr, "DBG imm %u code %d pc %u detail %u\n", immediate,
                         failure ? int(failure->code) : -1, failure ? failure->pc : 0,
                         failure ? failure->detail : 0);
        }
        ARMIGO_CHECK(program != nullptr);
        if (program == nullptr) continue;
        const auto evaluated = gx::evaluate(*program, gx::Inputs{});
        const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
        ARMIGO_CHECK(outputs != nullptr && outputs->outputs.size() == 1);
        if (outputs == nullptr || outputs->outputs.size() != 1) continue;
        ARMIGO_CHECK_EQ(outputs->outputs[0].bits, convert(7, immediate));
    }
}

ARMIGO_TEST(integer_to_float, lowers_decoded_programs) {
    // R1 = 0xffffffff (seeded by the fixture), R3 = I2F of R1.
    const auto rz = i2f_word(0x5cb8, 3, 2, 2, false, 1, 3, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({rz}), convert(0x12345678u, kUnsigned32Rz));
    const auto rn = i2f_word(0x5cb8, 3, 2, 2, false, 1, 0, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({rn}), convert(0x12345678u, kUnsigned32Rn));
    const auto signed_rn = i2f_word(0x5cb8, 3, 2, 2, true, 1, 0, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({signed_rn}), convert(0x12345678u, kSigned32Rn));
    const auto signed_abs = i2f_word(0x5cb8, 3, 2, 2, true, 1, 0, 0, false, false, true);
    ARMIGO_CHECK_EQ(run({signed_abs}), convert(0x12345678u, kSigned32RnAbsolute));
    const auto half = i2f_word(0x5cb8, 3, 2, 1, true, 1, 0, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({half}), convert(0x12345678u, kSigned16Rn));
    // Constant-buffer source: binding 1, byte offset 0 reads word 0.
    const auto buffer = i2f_word(0x4cb8, 3, 2, 2, true, 0, 0, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({buffer}, {0x80000000u}), convert(0x80000000u, kSigned32Rn));
    // The source is read before the destination is written, so converting a
    // register into itself is defined.
    const std::uint64_t seed_r3 = mov32i(3, 0x12345678u);
    const auto self = i2f_word(0x5cb8, 3, 2, 2, false, 3, 3, 0, false, false, false);
    ARMIGO_CHECK_EQ(run({seed_r3, self}), convert(0x12345678u, kUnsigned32Rz));
}

ARMIGO_TEST(integer_to_float, msl_carries_the_exact_helper) {
    gx::GraphSpec spec;
    spec.stage = si::Stage::Vertex;
    spec.sph = vertex_sph(0xf0);  // All four position lanes are exported.
    spec.nodes = {{8, gx::Literal{0x12345678u}},
                  {16, gx::Pure{gx::PureKind::IntegerToFloat, {0, 0, 0}, kSigned32Rn}},
                  {24, gx::Pure{gx::PureKind::IntegerToFloat, {0, 0, 0}, kUnsigned32Rz}},
                  {32, gx::Pure{gx::PureKind::IntegerToFloat, {0, 0, 0}, kSigned32RnAbsolute}},
                  {40, gx::Pure{gx::PureKind::IntegerToFloat, {0, 0, 0}, kSigned16Rn}}};
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane,
                                gx::ScalarId{static_cast<std::uint32_t>(1 + lane)}});
    }
    spec.blocks = {{0, 5, gx::Return{}}};
    const auto source = emit_graph(spec);
    ARMIGO_CHECK(source.find("armigo_i2f_round(") != std::string::npos);
    ARMIGO_CHECK(source.find("v1 = (armigo_i2f(v0, true, 32u, false, false))") !=
                 std::string::npos);
    ARMIGO_CHECK(source.find("v2 = (armigo_i2f(v0, false, 32u, true, false))") !=
                 std::string::npos);
    ARMIGO_CHECK(source.find("v3 = (armigo_i2f(v0, true, 32u, false, true))") !=
                 std::string::npos);
    ARMIGO_CHECK(source.find("v4 = (armigo_i2f(v0, true, 16u, false, false))") !=
                 std::string::npos);
}

}  // namespace armigo::shader::test
