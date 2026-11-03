// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// F2F F32-to-F32 rounding to an integral value: the registered
// FloatRoundIntegral CPU reference over edge inputs, the register and
// constant-buffer decoder profiles and rejections, lowering of decoded
// programs, and the emitted MSL helper text. Synthetic words and programs
// only.
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>
#include "graphics_operations.h"
#include "test_support.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace mx = maxwell;
namespace gx = graphics;
namespace gm = graphics_msl;
namespace si = shader_input;
using Family = mx::OpcodeFamily;
using gx::PureKind;

constexpr std::uint32_t kNearestEven = 0;
constexpr std::uint32_t kFloor = 1;
constexpr std::uint32_t kCeil = 2;
constexpr std::uint32_t kTrunc = 3;

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

// Vertex container: MOV32I R1, MOV32I R2, the instruction under test, four
// position stores of R3 and EXIT, in 32-byte scheduling groups from 0x80.
struct Container {
    std::vector<std::uint8_t> control, code;
};
Container container(std::uint64_t instruction, std::uint32_t source_word) {
    const std::vector<std::uint64_t> words{
        (std::uint64_t{0x0100} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{15} << 12) |
            (std::uint64_t{source_word} << 20) | 1,  // MOV32I R1, source_word
        (std::uint64_t{0x0100} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{15} << 12) |
            (std::uint64_t{0x40000000} << 20) | 2,  // MOV32I R2, 2.0
        instruction,
        // AST position x..w from R3 (attribute offsets 0x70..0x7c).
        (std::uint64_t{0xeff0} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{0x70} << 20) |
            (std::uint64_t{0xff} << 8) | (std::uint64_t{0xff} << 39) | 3,
        (std::uint64_t{0xeff0} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{0x74} << 20) |
            (std::uint64_t{0xff} << 8) | (std::uint64_t{0xff} << 39) | 3,
        (std::uint64_t{0xeff0} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{0x78} << 20) |
            (std::uint64_t{0xff} << 8) | (std::uint64_t{0xff} << 39) | 3,
        (std::uint64_t{0xeff0} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{0x7c} << 20) |
            (std::uint64_t{0xff} << 8) | (std::uint64_t{0xff} << 39) | 3,
        (std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15,
    };
    const unsigned groups = static_cast<unsigned>((words.size() + 2) / 3);
    const unsigned end = 0x80 + groups * 32;
    Container result{std::vector<std::uint8_t>(0x718), std::vector<std::uint8_t>(end)};
    put(result.control, 0, si::format::control_magic, 4);
    put(result.control, 4, si::format::control_tag, 4);
    put(result.control, 8, 14, 4);
    put(result.control, 0x20, result.control.size(), 4);
    put(result.control, 0x700, end, 4);
    put(result.control, 0x704, end, 4);
    put(result.control, 0x708, 0x30, 4);
    put(result.control, 0x714, 0, 4);
    put(result.code, 0, 0x12345678, 4);
    put(result.code, 0x30, si::sph_stage_type_for(si::Stage::Vertex) << 10, 4);
    result.code[0x30 + 0x35] = 0xf0;  // position outputs declared
    for (std::size_t i = 0; i < words.size(); ++i) {
        put(result.code, static_cast<unsigned>(0x80 + (i / 3) * 32 + 8 + (i % 3) * 8), words[i], 8);
    }
    return result;
}

mx::DecodeResult decode(std::uint64_t instruction,
                        std::uint32_t source_word = 0x40490fdbu /* pi */) {
    const auto c = container(instruction, source_word);
    return mx::decode({c.control.data(), c.control.size()}, {c.code.data(), c.code.size()});
}

// F2F word with the admitted F32 profile: destination 0..7, sizes 8..11 set
// to F32, guard predicate PT, source field 20.., rounding operation 39..42,
// FTZ 44, saturate 50, and the rejected modifier bits at 45, 47 and 49.
std::uint64_t f2f_word(std::uint16_t high, unsigned destination,
                       std::uint64_t source_field, unsigned operation,
                       bool saturate = false, bool negate = false,
                       bool condition_code = false, bool absolute = false,
                       bool flush = true) {
    return (std::uint64_t{high} << 48) | destination | (std::uint64_t{7} << 16) |
           (std::uint64_t{2} << 8) | (std::uint64_t{2} << 10) |
           (source_field << 20) | (std::uint64_t{operation} << 39) |
           (std::uint64_t{flush ? 1u : 0u} << 44) |
           (std::uint64_t{negate ? 1u : 0u} << 45) |
           (std::uint64_t{condition_code ? 1u : 0u} << 47) |
           (std::uint64_t{absolute ? 1u : 0u} << 49) |
           (std::uint64_t{saturate ? 1u : 0u} << 50);
}

const mx::FloatRoundIntegral* payload(const mx::DecodeResult& result, Family family) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return nullptr;
    const auto& instruction = program->instructions[2];
    ARMIGO_CHECK(instruction.family == family);
    return std::get_if<mx::FloatRoundIntegral>(&instruction.payload);
}

void rejected(const mx::DecodeResult& result, mx::DecodeError::Code code) {
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) ARMIGO_CHECK(error->code == code);
}

std::uint32_t position_x(const gx::LowerResult& lowered,
                         std::span<const std::uint32_t> constants = {}) {
    const auto* program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return 0xdeadbeefu;
    gx::Inputs inputs;
    if (!constants.empty()) inputs.constant_buffers[0] = constants;
    const auto evaluated = gx::evaluate(*program, inputs);
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return 0xdeadbeefu;
    for (const auto& output : outputs->outputs)
        if (output.kind == gx::OutputKind::Position && output.component == 0) return output.bits;
    return 0xdeadbeefu;
}

void put_word(std::array<std::byte, 80>& data, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        data[offset + i] = std::byte(static_cast<unsigned char>(value >> (i * 8)));
    }
}

gx::Node pure(std::uint32_t pc, PureKind kind, std::array<std::uint32_t, 3> operands,
              std::uint32_t immediate = 0) {
    return {pc, gx::Pure{kind, operands, immediate}};
}

// Vertex graph exporting four Raw32 nodes as position x/y/z/w.
gx::GraphSpec vertex_spec(std::vector<gx::Node> nodes, std::array<std::uint32_t, 4> outputs) {
    gx::GraphSpec spec;
    spec.stage = si::Stage::Vertex;
    std::array<std::byte, 80> header{};
    put_word(header, 0, si::sph_stage_type_for(si::Stage::Vertex) << 10);
    header[0x35] = std::byte{0xf0};
    spec.sph = header;
    spec.nodes = std::move(nodes);
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane, gx::ScalarId{outputs[lane]}});
    }
    spec.blocks = {{0, static_cast<std::uint32_t>(spec.nodes.size()), gx::Return{}}};
    return spec;
}

std::array<std::uint32_t, 4> evaluate_modes(std::uint32_t input) {
    const auto graph = gx::make_graph(vertex_spec({
        {8, gx::Literal{input}},
        pure(16, PureKind::FloatRoundIntegral, {0}, kNearestEven),
        pure(24, PureKind::FloatRoundIntegral, {0}, kFloor),
        pure(32, PureKind::FloatRoundIntegral, {0}, kCeil),
        pure(40, PureKind::FloatRoundIntegral, {0}, kTrunc),
    }, {1, 2, 3, 4}));
    const auto* program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(program != nullptr);
    std::array<std::uint32_t, 4> result{};
    if (program == nullptr) return result;
    const auto evaluated = gx::evaluate(*program, {});
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return result;
    for (const auto& output : outputs->outputs) result[output.component] = output.bits;
    return result;
}

void check_modes(std::uint32_t input, std::uint32_t nearest_even, std::uint32_t floor,
                 std::uint32_t ceil, std::uint32_t trunc) {
    const auto result = evaluate_modes(input);
    ARMIGO_CHECK_EQ(result[0], nearest_even);
    ARMIGO_CHECK_EQ(result[1], floor);
    ARMIGO_CHECK_EQ(result[2], ceil);
    ARMIGO_CHECK_EQ(result[3], trunc);
}

}  // namespace

ARMIGO_TEST(float_round_integral, rounds_each_direction_exactly) {
    // Nearest-even, floor, ceil, trunc of exactly representable halves and
    // near-integers, derived from the IEEE roundToIntegral definitions.
    check_modes(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);  // +0
    check_modes(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);  // -0
    check_modes(0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u);  // 1.0
    check_modes(0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u);  // -1.0
    check_modes(0x3f000000u, 0x00000000u, 0x00000000u, 0x3f800000u, 0x00000000u);  // +0.5
    check_modes(0xbf000000u, 0x80000000u, 0xbf800000u, 0x80000000u, 0x80000000u);  // -0.5
    check_modes(0x3fc00000u, 0x40000000u, 0x3f800000u, 0x40000000u, 0x3f800000u);  // +1.5
    check_modes(0xbfc00000u, 0xc0000000u, 0xc0000000u, 0xbf800000u, 0xbf800000u);  // -1.5
    check_modes(0x40200000u, 0x40000000u, 0x40000000u, 0x40400000u, 0x40000000u);  // +2.5
    check_modes(0xc0200000u, 0xc0000000u, 0xc0400000u, 0xc0000000u, 0xc0000000u);  // -2.5
    check_modes(0x40600000u, 0x40800000u, 0x40400000u, 0x40800000u, 0x40400000u);  // +3.5
    check_modes(0x3dcccccdu, 0x00000000u, 0x00000000u, 0x3f800000u, 0x00000000u);  // +0.1
    check_modes(0xbdcccccdu, 0x80000000u, 0xbf800000u, 0x80000000u, 0x80000000u);  // -0.1
    check_modes(0x3e800000u, 0x00000000u, 0x00000000u, 0x3f800000u, 0x00000000u);  // +0.25
    check_modes(0x3fffffffu, 0x40000000u, 0x3f800000u, 0x40000000u, 0x3f800000u);  // just under 2
    check_modes(0x4affffffu, 0x4b000000u, 0x4afffffeu, 0x4b000000u, 0x4afffffeu);  // 2^23-0.5
}

ARMIGO_TEST(float_round_integral, flushes_subnormals_and_preserves_specials) {
    // FTZ flushes subnormal inputs to the sign-preserving zero; infinities
    // and integral words pass through; NaNs keep sign and payload.
    check_modes(0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);
    check_modes(0x80000001u, 0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
    check_modes(0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u);  // +inf
    check_modes(0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u);  // -inf
    check_modes(0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u);
    check_modes(0xffc12345u, 0xffc12345u, 0xffc12345u, 0xffc12345u, 0xffc12345u);
    check_modes(0x7f800001u, 0x7f800001u, 0x7f800001u, 0x7f800001u, 0x7f800001u);
    check_modes(0x4b000000u, 0x4b000000u, 0x4b000000u, 0x4b000000u, 0x4b000000u);  // 2^23
    check_modes(0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu);
}

ARMIGO_TEST(float_round_integral, admits_only_four_modes) {
    const auto* entry = gx::detail::find_operation(PureKind::FloatRoundIntegral);
    ARMIGO_CHECK(entry != nullptr);
    if (entry == nullptr) return;
    ARMIGO_CHECK(entry->kind == PureKind::FloatRoundIntegral);
    ARMIGO_CHECK(gx::pure_result_type(PureKind::FloatRoundIntegral) == gx::ValueType::Raw32);
    for (std::uint32_t immediate = 0; immediate < 4; ++immediate) {
        ARMIGO_CHECK(entry->admits(immediate));
    }
    ARMIGO_CHECK(!entry->admits(4));
    const auto rejected_graph = gx::make_graph(vertex_spec({
        {8, gx::Literal{0x3f800000u}},
        pure(16, PureKind::FloatRoundIntegral, {0}, 4),
    }, {1, 0, 0, 0}));
    const auto* error = std::get_if<gx::Error>(&rejected_graph);
    ARMIGO_CHECK(error != nullptr && error->code == gx::Error::Code::GraphFormat);
}

ARMIGO_TEST(float_round_integral, decodes_register_and_constant_forms) {
    const auto reg_result = decode(f2f_word(0x5ca8, 3, 1, 9));
    const auto reg = payload(reg_result, Family::FloatRoundIntegralRegister);
    ARMIGO_CHECK(reg != nullptr);
    if (reg != nullptr) {
        ARMIGO_CHECK(reg->destination.index == 3);
        ARMIGO_CHECK(std::get<mx::RegisterOperand>(reg->source).index == 1);
        ARMIGO_CHECK(reg->rounding == mx::IntegralRounding::Floor);
        ARMIGO_CHECK(!reg->saturate);
    }
    const struct { unsigned operation; mx::IntegralRounding rounding; } modes[] = {
        {8, mx::IntegralRounding::NearestEven},
        {9, mx::IntegralRounding::Floor},
        {10, mx::IntegralRounding::Ceil},
        {11, mx::IntegralRounding::Trunc},
    };
    for (const auto& mode : modes) {
        const auto result = decode(f2f_word(0x5ca8, 3, 1, mode.operation));
        const auto value = payload(result, Family::FloatRoundIntegralRegister);
        ARMIGO_CHECK(value != nullptr);
        if (value != nullptr) ARMIGO_CHECK(value->rounding == mode.rounding);
    }
    const auto saturated = decode(f2f_word(0x5ca8, 3, 1, 9, true));
    const auto saturate = payload(saturated, Family::FloatRoundIntegralRegister);
    ARMIGO_CHECK(saturate != nullptr && saturate->saturate);
    // Constant buffer: word offset 0x10 (bits 20..33), binding 3 (34..38).
    const auto cbuf_result = decode(f2f_word(0x4ca8, 3, 0x10 | (3u << 14), 9));
    const auto cbuf = payload(cbuf_result, Family::FloatRoundIntegralConstant);
    ARMIGO_CHECK(cbuf != nullptr);
    if (cbuf != nullptr) {
        const auto& operand = std::get<mx::ConstantBufferOperand>(cbuf->source);
        ARMIGO_CHECK(operand.binding == 3 && operand.byte_offset == 0x40);
    }
}

ARMIGO_TEST(float_round_integral, rejects_unadmitted_fields) {
    // Wrong source or destination format, rounding operations outside the
    // same-size integral set (including the F16 selector bit), disabled FTZ,
    // and every unadmitted modifier bit.
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9) | (std::uint64_t{3} << 10)),
             mx::DecodeError::Code::UnsupportedModifier);  // F64 source
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9) | (std::uint64_t{3} << 8)),
             mx::DecodeError::Code::UnsupportedModifier);  // F64 destination
    for (unsigned operation : {0u, 3u, 12u, 13u, 14u, 15u}) {
        rejected(decode(f2f_word(0x5ca8, 3, 1, operation)),
                 mx::DecodeError::Code::UnsupportedModifier);
    }
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9, false, false, false, false, false)),
             mx::DecodeError::Code::UnsupportedModifier);  // FTZ clear
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9, false, true)),
             mx::DecodeError::Code::UnsupportedModifier);  // negate
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9, false, false, true)),
             mx::DecodeError::Code::UnsupportedModifier);  // condition code
    rejected(decode(f2f_word(0x5ca8, 3, 1, 9, false, false, false, true)),
             mx::DecodeError::Code::UnsupportedModifier);  // absolute
    for (unsigned bit : {12u, 28u, 43u, 46u, 48u}) {
        rejected(decode(f2f_word(0x5ca8, 3, 1, 9) | (std::uint64_t{1} << bit)),
                 mx::DecodeError::Code::UnsupportedModifier);
    }
    // Constant-buffer binding outside the evaluated range.
    rejected(decode(f2f_word(0x4ca8, 3, (18u << 14), 9)),
             mx::DecodeError::Code::UnsupportedModifier);
}

ARMIGO_TEST(float_round_integral, lowers_decoded_programs) {
    // floor(pi) = 3, ceil(pi) = 4, rne(pi) = 3, trunc(pi) = 3; saturation
    // applies after rounding; floor(-2.5) = -3; the constant-buffer form
    // reads c0[0].
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 9))))),
                    0x40400000u);
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 10))))),
                    0x40800000u);
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 11))))),
                    0x40400000u);
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 8))))),
                    0x40400000u);
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 9, true))))),
                    0x3f800000u);
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(
                        decode(f2f_word(0x5ca8, 3, 1, 9), 0xc0200000u)))),
                    0xc0400000u);
    const auto cbuf = gx::lower(std::get<mx::DecodedProgram>(
        decode(f2f_word(0x4ca8, 3, 0, 9))));
    const std::array<std::uint32_t, 1> constants{0xc0200000u};  // -2.5
    ARMIGO_CHECK_EQ(position_x(cbuf, constants), 0xc0400000u);
}

ARMIGO_TEST(float_round_integral, emits_helper_only_when_used) {
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto lowered = gx::lower(std::get<mx::DecodedProgram>(
        decode(f2f_word(0x5ca8, 3, 1, 9))));
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (graph == nullptr) return;
    const auto emitted = gm::emit(*graph, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) return;
    ARMIGO_CHECK(stage->source.find(
        "inline uint armigo_round_integral(uint x, uint mode)") != std::string::npos);
    ARMIGO_CHECK(stage->source.find("armigo_round_integral(v") != std::string::npos);
    ARMIGO_CHECK(stage->source.find(", 1u)") != std::string::npos);
    // A program without the kind keeps its source bytes free of the helper.
    // SEL R3, R1, R2, PT lowers through the registered Select operation,
    // which ships no helper.
    const auto plain = decode((std::uint64_t{0x5ca0} << 48) | (std::uint64_t{7} << 16) |
                              (std::uint64_t{1} << 8) | (std::uint64_t{2} << 20) |
                              (std::uint64_t{7} << 39) | 3);
    const auto plain_lowered = gx::lower(std::get<mx::DecodedProgram>(plain));
    const auto* plain_graph = std::get_if<gx::Program>(&plain_lowered);
    ARMIGO_CHECK(plain_graph != nullptr);
    if (plain_graph == nullptr) return;
    const auto plain_emitted = gm::emit(*plain_graph, options);
    const auto* plain_stage = std::get_if<gm::EmittedStage>(&plain_emitted);
    ARMIGO_CHECK(plain_stage != nullptr);
    if (plain_stage == nullptr) return;
    ARMIGO_CHECK(plain_stage->source.find("armigo_round_integral") == std::string::npos);
}

// Registered under the compile group so shader.forward_lower_msl_compile
// compiles the float rounding fixture with the Metal toolchain.
ARMIGO_TEST(forward_lower_msl, float_round_integral_fixture) {
    const auto decoded = decode(f2f_word(0x5ca8, 3, 1, 9));
    const auto lowered = gx::lower(std::get<mx::DecodedProgram>(decoded));
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (graph == nullptr) return;
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted = gm::emit(*graph, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    const char* directory = std::getenv("ARMIGO_FORWARD_MSL_DIR");
    if (stage == nullptr || directory == nullptr || *directory == '\0') return;
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    std::ofstream output{std::string{directory} + "/float_round_integral.metal",
                         std::ios::binary | std::ios::trunc};
    output << stage->source;
}

}  // namespace armigo::shader::test
