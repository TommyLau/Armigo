// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// SEL (select by predicate): decoding of the register, constant-buffer and
// immediate forms, raw-word selection on the straight-line and guarded
// forward paths, and the straight-line predicate declaration in MSL.
// Synthetic words and programs only.
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_forward.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>
#include <armigo/shader/maxwell_forward_program.h>
#include "test_support.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::uint64_t sel_word(std::uint16_t high, unsigned destination, unsigned a,
                       std::uint64_t b_field, unsigned predicate, bool negate) {
    return (std::uint64_t{high} << 48) | (std::uint64_t{7} << 16) |
           (b_field << 20) | (std::uint64_t{predicate} << 39) |
           (std::uint64_t{negate ? 1u : 0u} << 42) | (std::uint64_t{a} << 8) |
           destination;
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

// Vertex container: MOV32I R1, MOV32I R2, the instruction under test, four
// position stores of R3 and EXIT, in 32-byte scheduling groups from 0x80.
struct Container {
    std::vector<std::uint8_t> control, code;
};
Container container(std::uint64_t instruction) {
    const std::vector<std::uint64_t> words{
        (std::uint64_t{0x0100} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{15} << 12) |
            (std::uint64_t{0x3f800000} << 20) | 1,  // MOV32I R1, 1.0
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

mx::DecodeResult decode(std::uint64_t instruction) {
    const auto c = container(instruction);
    return mx::decode({c.control.data(), c.control.size()}, {c.code.data(), c.code.size()});
}

const mx::SelectByPredicate* payload(const mx::DecodeResult& result, Family family) {
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return nullptr;
    const auto& instruction = program->instructions[2];
    ARMIGO_CHECK(instruction.family == family);
    return std::get_if<mx::SelectByPredicate>(&instruction.payload);
}

void rejected(const mx::DecodeResult& result, mx::DecodeError::Code code) {
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) ARMIGO_CHECK(error->code == code);
}

std::uint32_t position_x(const gx::LowerResult& lowered, std::uint32_t condition = 0) {
    const auto* program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return 0xdeadbeefu;
    const std::array<std::uint32_t, 1> words{condition};
    gx::Inputs inputs;
    inputs.constant_buffers[0] = words;
    const auto evaluated = gx::evaluate(*program, inputs);
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return 0xdeadbeefu;
    for (const auto& output : outputs->outputs)
        if (output.kind == gx::OutputKind::Position && output.component == 0) return output.bits;
    return 0xdeadbeefu;
}

mx::Instruction at(std::uint32_t pc, Family family, mx::Predicate guard,
                   mx::InstructionPayload value) {
    return mx::Instruction{pc, 0, family, guard, std::move(value)};
}

// Forged guarded program: R1 = 1.0, R2 = 2.0, R4 = c0[0], R5 = 0,
// P0 = (R4 != R5), then the SEL into R3 (guarded by `guard`), position
// stores and EXIT.
mx::ForwardProgram forward(mx::SelectByPredicate select, mx::Predicate guard) {
    std::array<std::byte, 80> sph{};
    const std::uint32_t word0 = si::sph_stage_type_for(si::Stage::Vertex) << 10;
    for (unsigned i = 0; i < 4; ++i) sph[i] = std::byte(word0 >> (i * 8));
    sph[0x35] = std::byte{0xf0};
    const mx::Predicate pt{7, false};
    std::vector<mx::ForwardInstruction> nodes{
        at(8, Family::Mov32Immediate, pt, mx::Mov32Immediate{{1}, 15, 0x3f800000u}),
        at(16, Family::Mov32Immediate, pt, mx::Mov32Immediate{{2}, 15, 0x40000000u}),
        at(24, Family::Mov32Immediate, pt, mx::Mov32Immediate{{3}, 15, 0x40400000u}),
        at(40, Family::MovConstant, pt, mx::MovConstant{{4}, {0, 0}, 15}),
        at(48, Family::Mov32Immediate, pt, mx::Mov32Immediate{{5}, 15, 0}),
        at(56, Family::IntegerSetPredicateRegister, pt,
           mx::IntegerSetPredicate{0, 7, {4}, {5}, pt, false}),
        at(72, Family::SelectByPredicateRegister, guard, select),
    };
    std::uint32_t pc = 80;
    for (std::uint16_t offset : {0x70, 0x74, 0x78, 0x7c}) {
        nodes.push_back(at(pc, Family::AttributeStore, pt,
                           mx::AttributeAccess{{3}, {255}, {255}, offset, 0, false}));
        pc += (pc % 32 == 24) ? 16 : 8;
    }
    nodes.push_back(at(pc, Family::Exit, pt, mx::Exit{15}));
    mx::ForwardProgram program;
    program.stage = si::Stage::Vertex;
    program.sph = sph;
    program.program_size = (pc / 32 + 1) * 32;
    program.instructions = std::move(nodes);
    return program;
}

}  // namespace

ARMIGO_TEST(select_by_predicate, decodes_every_form) {
    const auto reg_result = decode(sel_word(0x5ca0, 3, 1, 2, 5, true));
    const auto reg = payload(reg_result, Family::SelectByPredicateRegister);
    ARMIGO_CHECK(reg != nullptr);
    if (reg != nullptr) {
        ARMIGO_CHECK(reg->destination.index == 3 && reg->source_a.index == 1);
        ARMIGO_CHECK(std::get<mx::RegisterOperand>(reg->source_b).index == 2);
        ARMIGO_CHECK(reg->predicate == 5 && reg->negate_predicate);
    }
    // Constant buffer: word offset 0x10 (bits 20..33), binding 3 (34..38).
    const auto cbuf_result = decode(sel_word(0x4ca0, 3, 1, 0x10 | (3u << 14), 7, false));
    const auto cbuf = payload(cbuf_result, Family::SelectByPredicateConstant);
    ARMIGO_CHECK(cbuf != nullptr);
    if (cbuf != nullptr) {
        const auto& operand = std::get<mx::ConstantBufferOperand>(cbuf->source_b);
        ARMIGO_CHECK(operand.binding == 3 && operand.byte_offset == 0x40);
    }
    // Immediate: 19-bit magnitude with the sign at bit 56.
    const struct { std::uint16_t high; std::uint64_t field; std::uint32_t value; } cases[] = {
        {0x38a0, 0x12345, 0x12345u}, {0x39a0, 0x7ffff, 0xffffffffu},
        {0x39a0, 0, 0xfff80000u}, {0x38a0, 0x7ffff, 0x7ffffu}};
    for (const auto& c : cases) {
        const auto imm_result = decode(sel_word(c.high, 3, 1, c.field, 0, false));
        const auto imm = payload(imm_result, Family::SelectByPredicateImmediate);
        ARMIGO_CHECK(imm != nullptr);
        if (imm != nullptr) ARMIGO_CHECK_EQ(std::get<std::uint32_t>(imm->source_b), c.value);
    }
}

ARMIGO_TEST(select_by_predicate, rejects_reserved_bits) {
    rejected(decode(sel_word(0x5ca0, 3, 1, 2, 0, false) | (std::uint64_t{1} << 28)),
             mx::DecodeError::Code::UnsupportedModifier);
    rejected(decode(sel_word(0x5ca0, 3, 1, 2, 0, false) | (std::uint64_t{1} << 45)),
             mx::DecodeError::Code::UnsupportedModifier);
    rejected(decode(sel_word(0x38a0, 3, 1, 2, 0, false) | (std::uint64_t{1} << 43)),
             mx::DecodeError::Code::UnsupportedModifier);
}

ARMIGO_TEST(select_by_predicate, straight_line_pt_selects_and_emits_bool) {
    // SEL R3, R1, R2, PT picks A; !PT picks B. These need no prior predicate
    // write, so they stay on the straight-line path and its MSL declares the
    // PT literal as a const bool.
    const auto always = decode(sel_word(0x5ca0, 3, 1, 2, 7, false));
    const auto* program = std::get_if<mx::DecodedProgram>(&always);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    const auto lowered = gx::lower(*program);
    ARMIGO_CHECK_EQ(position_x(lowered), 0x3f800000u);
    const auto never = decode(sel_word(0x5ca0, 3, 1, 2, 7, true));
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(never))), 0x40000000u);
    const auto immediate = decode(sel_word(0x39a0, 3, 1, 0x7fffe, 7, true));
    ARMIGO_CHECK_EQ(position_x(gx::lower(std::get<mx::DecodedProgram>(immediate))),
                    0xfffffffeu);

    const auto* graph = std::get_if<gx::Program>(&lowered);
    if (graph == nullptr) return;
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted = gm::emit(*graph, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) return;
    ARMIGO_CHECK(stage->source.find("const bool p") != std::string::npos);
    ARMIGO_CHECK(stage->source.find(" = true;") != std::string::npos);
    const char* directory = std::getenv("ARMIGO_FORWARD_MSL_DIR");
    if (directory == nullptr || *directory == '\0') return;
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    std::ofstream output{std::string{directory} + "/select_straight_line.metal",
                         std::ios::binary | std::ios::trunc};
    output << stage->source;
}

// Registered under the compile group so shader.forward_lower_msl_compile
// compiles the straight-line predicate fixture with the Metal toolchain.
ARMIGO_TEST(forward_lower_msl, select_straight_line_fixture) {
    const auto decoded = decode(sel_word(0x5ca0, 3, 1, 2, 7, false));
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
    std::ofstream output{std::string{directory} + "/select_straight_line.metal",
                         std::ios::binary | std::ios::trunc};
    output << stage->source;
}

ARMIGO_TEST(select_by_predicate, guarded_forward_selection) {
    const mx::Predicate pt{7, false};
    const mx::SelectByPredicate plain{{3}, {1}, mx::RegisterOperand{2}, 0, false};
    // Runtime P0 picks A when c0[0] != 0 and B otherwise.
    const auto picked = gx::lower_forward(forward(plain, pt));
    ARMIGO_CHECK_EQ(position_x(picked, 1), 0x3f800000u);
    ARMIGO_CHECK_EQ(position_x(picked, 0), 0x40000000u);
    // Negated predicate swaps the candidates.
    auto negated = plain;
    negated.negate_predicate = true;
    const auto swapped = gx::lower_forward(forward(negated, pt));
    ARMIGO_CHECK_EQ(position_x(swapped, 1), 0x40000000u);
    ARMIGO_CHECK_EQ(position_x(swapped, 0), 0x3f800000u);
    // Guarded by P0 itself: when the guard is false R3 keeps 3.0.
    const auto guarded = gx::lower_forward(forward(plain, mx::Predicate{0, false}));
    ARMIGO_CHECK_EQ(position_x(guarded, 1), 0x3f800000u);
    ARMIGO_CHECK_EQ(position_x(guarded, 0), 0x40400000u);
    // A forged predicate index is invalid even under a false guard.
    auto forged = plain;
    forged.predicate = 9;
    const auto rejected_program = gx::lower_forward(forward(forged, mx::Predicate{7, true}));
    const auto* error = std::get_if<gx::Error>(&rejected_program);
    ARMIGO_CHECK(error != nullptr && error->code == gx::Error::Code::InvalidInstruction);
}

}  // namespace armigo::shader::test
