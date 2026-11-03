// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// LOP (bitwise logic): decoding of the three forms, rejected X/CC/reserved
// bits, every operation with every inversion on the CPU reference, and the
// predicate result consumed by a following SEL on the straight-line path.
// Synthetic words and programs only.
#include <armigo/shader/graphics.h>
#include <armigo/shader/maxwell.h>
#include "test_support.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace mx = maxwell;
namespace gx = graphics;
namespace si = shader_input;
using Family = mx::OpcodeFamily;

constexpr std::uint32_t kA = 0xf0f0a5a5u;
constexpr std::uint32_t kB = 0xff00c33cu;

std::uint64_t lop_word(std::uint16_t high, unsigned destination, unsigned a, std::uint64_t b,
                       unsigned operation, bool invert_a, bool invert_b,
                       unsigned predicate_operation = 0, unsigned predicate_destination = 7) {
    return (std::uint64_t{high} << 48) | (std::uint64_t{predicate_destination} << 48) |
           (std::uint64_t{predicate_operation} << 44) | (std::uint64_t{operation} << 41) |
           (std::uint64_t{invert_b ? 1u : 0u} << 40) | (std::uint64_t{invert_a ? 1u : 0u} << 39) |
           (b << 20) | (std::uint64_t{7} << 16) | (std::uint64_t{a} << 8) | destination;
}

std::uint64_t mov32i(unsigned destination, std::uint32_t bits) {
    return (std::uint64_t{0x0100} << 48) | (std::uint64_t{7} << 16) | (std::uint64_t{15} << 12) |
           (std::uint64_t{bits} << 20) | destination;
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

// Vertex program: R1 = A, R2 = B, the middle instructions, then R3 stored
// to the four position lanes and EXIT.
mx::DecodeResult decode(const std::vector<std::uint64_t>& middle) {
    std::vector<std::uint64_t> words{mov32i(1, kA), mov32i(2, kB)};
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

std::uint32_t result(const std::vector<std::uint64_t>& middle) {
    const auto decoded = decode(middle);
    const auto* program = std::get_if<mx::DecodedProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return 0xdeadbeefu;
    const auto lowered = gx::lower(*program);
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (graph == nullptr) return 0xdeadbeefu;
    const auto evaluated = gx::evaluate(*graph, {});
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return 0xdeadbeefu;
    for (const auto& output : outputs->outputs)
        if (output.kind == gx::OutputKind::Position && output.component == 0) return output.bits;
    return 0xdeadbeefu;
}

}  // namespace

ARMIGO_TEST(word_logic, decodes_every_form) {
    const auto reg_result = decode({lop_word(0x5c40, 3, 1, 2, 2, true, false, 3, 4)});
    const auto* program = std::get_if<mx::DecodedProgram>(&reg_result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        const auto& instruction = program->instructions[2];
        ARMIGO_CHECK(instruction.family == Family::WordLogicRegister);
        const auto* value = std::get_if<mx::WordLogic>(&instruction.payload);
        ARMIGO_CHECK(value != nullptr);
        if (value != nullptr) {
            ARMIGO_CHECK(value->operation == mx::LogicOperation::Xor && value->invert_a &&
                         !value->invert_b);
            ARMIGO_CHECK(value->predicate_operation == mx::LogicPredicate::NonZero &&
                         value->predicate_destination == 4);
            ARMIGO_CHECK(std::get<mx::RegisterOperand>(value->source_b).index == 2);
        }
    }
    const auto cbuf_result = decode({lop_word(0x4c40, 3, 1, 0x8 | (2u << 14), 0, false, false)});
    const auto* cbuf_program = std::get_if<mx::DecodedProgram>(&cbuf_result);
    ARMIGO_CHECK(cbuf_program != nullptr &&
                 cbuf_program->instructions[2].family == Family::WordLogicConstant);
    // Immediate: the sign at bit 56 is the low bit of the 0x3940 window.
    const auto negative = decode({lop_word(0x3940, 3, 1, 0x7fff0, 0, false, false)});
    const auto* imm_program = std::get_if<mx::DecodedProgram>(&negative);
    ARMIGO_CHECK(imm_program != nullptr);
    if (imm_program != nullptr) {
        const auto* value = std::get_if<mx::WordLogic>(&imm_program->instructions[2].payload);
        ARMIGO_CHECK(value != nullptr &&
                     std::get<std::uint32_t>(value->source_b) == 0xfffffff0u);
    }
}

ARMIGO_TEST(word_logic, rejects_extended_cc_and_reserved_bits) {
    for (unsigned bit : {43u, 47u, 46u}) {
        const auto rejected =
            decode({lop_word(0x5c40, 3, 1, 2, 0, false, false) | (std::uint64_t{1} << bit)});
        const auto* error = std::get_if<mx::DecodeError>(&rejected);
        ARMIGO_CHECK(error != nullptr &&
                     error->code == mx::DecodeError::Code::UnsupportedModifier);
    }
}

ARMIGO_TEST(word_logic, every_operation_and_inversion) {
    for (unsigned operation = 0; operation < 4; ++operation) {
        for (const bool invert_a : {false, true}) {
            for (const bool invert_b : {false, true}) {
                const std::uint32_t a = invert_a ? ~kA : kA;
                const std::uint32_t b = invert_b ? ~kB : kB;
                const std::uint32_t expected = operation == 0 ? (a & b)
                    : operation == 1 ? (a | b) : operation == 2 ? (a ^ b) : b;
                ARMIGO_CHECK_EQ(result({lop_word(0x5c40, 3, 1, 2, operation, invert_a,
                                                 invert_b)}),
                                expected);
            }
        }
    }
    // Immediate B: AND with the sign-extended 20-bit value 0xfffffff0.
    ARMIGO_CHECK_EQ(result({lop_word(0x3940, 3, 1, 0x7fff0, 0, false, false)}),
                    kA & 0xfffffff0u);
}

ARMIGO_TEST(word_logic, predicate_result_drives_a_following_select) {
    // LOP.OR with both inputs inverted writes NZ to P2; SEL R3 picks R1 (A)
    // when P2 holds, else R2 (B). ~A | ~B is nonzero for these words.
    const std::uint64_t select = (std::uint64_t{0x5ca0} << 48) | (std::uint64_t{7} << 16) |
                                 (std::uint64_t{2} << 20) | (std::uint64_t{2} << 39) |
                                 (std::uint64_t{1} << 8) | 3;
    const std::uint64_t nonzero = lop_word(0x5c40, 4, 1, 2, 1, true, true, 3, 2);
    ARMIGO_CHECK_EQ(result({nonzero, select}), kA);
    // Zero test on A AND ~A (always zero): Z is true, NZ false.
    const std::uint64_t zero = lop_word(0x5c40, 4, 1, 1, 0, false, true, 2, 2);
    ARMIGO_CHECK_EQ(result({zero, select}), kA);
    const std::uint64_t nonzero_of_zero = lop_word(0x5c40, 4, 1, 1, 0, false, true, 3, 2);
    ARMIGO_CHECK_EQ(result({nonzero_of_zero, select}), kB);
    // F and T predicate operations.
    ARMIGO_CHECK_EQ(result({lop_word(0x5c40, 4, 1, 2, 0, false, false, 0, 2), select}), kB);
    ARMIGO_CHECK_EQ(result({lop_word(0x5c40, 4, 1, 2, 0, false, false, 1, 2), select}), kA);
}

}  // namespace armigo::shader::test
