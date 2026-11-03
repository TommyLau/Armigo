// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/maxwell_forward_program.h>

#include "test_support.h"

#include <armigo/shader/maxwell_flow.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

using namespace armigo::shader::maxwell;
using shader_input::ByteSpan;
using shader_input::Stage;

std::uint64_t field_bits(unsigned shift, unsigned width, std::uint64_t value) {
    return (value << shift) &
           (((std::uint64_t{1} << width) - 1) << shift);
}

// Instruction word skeleton with the raw 4-bit predicate guard.
std::uint64_t guarded_word(std::uint16_t high, std::uint8_t predicate_raw) {
    return (static_cast<std::uint64_t>(high) << 48) |
           field_bits(16, 4, predicate_raw);
}

std::uint64_t mov_word(std::uint8_t destination, std::uint8_t predicate_raw) {
    return guarded_word(0x0100, predicate_raw) |
           field_bits(0, 8, destination) | field_bits(12, 4, 15) |
           field_bits(20, 32, 0xc0ffee00u + destination);
}

std::uint64_t exit_word(std::uint8_t predicate_raw) {
    return guarded_word(0xe300, predicate_raw) | field_bits(0, 5, 15);
}

// Direct relative branch word: shared opcode window, always-true flow test,
// raw guard, raw 24-bit displacement field.
std::uint64_t branch_word(std::uint32_t displacement_field,
                          std::uint8_t predicate_raw) {
    return guarded_word(0xe240, predicate_raw) | field_bits(0, 5, 15) |
           field_bits(20, 24, displacement_field);
}

constexpr std::uint32_t disp24(std::int32_t displacement) {
    return static_cast<std::uint32_t>(displacement) & 0xffffff;
}

// An opcode window no admitted family claims; a reachable occurrence must be
// an explicit error, never a silent NOP.
constexpr std::uint64_t unknown_word() {
    return std::uint64_t{0xdead} << 48;
}

struct Fixture {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

// Synthetic graphics shader container: 0x30-byte wrapper, 0x50-byte SPH, then
// 32-byte scheduling groups carrying the supplied instruction words.
Fixture make_fixture(Stage stage, const std::vector<std::uint64_t>& words) {
    const std::size_t groups = (words.size() + 2) / 3;
    const std::uint32_t program_size =
        static_cast<std::uint32_t>(groups * 32);
    const std::uint32_t c1_offset = 0x80 + program_size;
    Fixture result{std::vector<std::uint8_t>(0x718),
                   std::vector<std::uint8_t>(c1_offset)};
    auto put32 = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned index = 0; index < 4; ++index) {
            result.control[offset + index] =
                static_cast<std::uint8_t>(value >> (index * 8));
        }
    };
    put32(0, 0x98761234);
    put32(4, 1);
    put32(8, 14);
    put32(0x20, static_cast<std::uint32_t>(result.control.size()));
    put32(0x700, c1_offset);
    put32(0x704, c1_offset);
    put32(0x708, 0x30);
    put32(0x714, static_cast<std::uint32_t>(stage));
    for (unsigned index = 0; index < 4; ++index) {
        result.code[index] = static_cast<std::uint8_t>(0x12345678u >>
                                                       (index * 8));
    }
    for (std::size_t index = 0; index < shader_input::format::sph_size;
         ++index) {
        result.code[0x30 + index] =
            static_cast<std::uint8_t>(0x50u + (index * 3u) % 0x40u);
    }
    // The parser verifies the SPH stage-type nibble against the control
    // stage field.
    for (unsigned index = 0; index < 4; ++index) {
        result.code[0x30 + index] = static_cast<std::uint8_t>(
            (shader_input::sph_stage_type_for(stage) <<
             shader_input::format::sph_stage_shift) >>
            (index * 8));
    }
    for (std::size_t group = 0; group < groups; ++group) {
        for (unsigned index = 0; index < 8; ++index) {
            result.code[0x80 + group * 32 + index] =
                static_cast<std::uint8_t>(0x11u * (group + 1));
        }
    }
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::size_t offset = 0x80 + (index / 3) * 32 + 8 + (index % 3) * 8;
        for (unsigned byte = 0; byte < 8; ++byte) {
            result.code[offset + byte] =
                static_cast<std::uint8_t>(words[index] >> (byte * 8));
        }
    }
    return result;
}

ByteSpan span(const std::vector<std::uint8_t>& bytes) {
    return {bytes.data(), bytes.size()};
}

ForwardDecodeResult run(const Fixture& data, DecodeLimits limits = {}) {
    return decode_forward_program(span(data.control), span(data.code), limits);
}

const ForwardProgram* accepted(const ForwardDecodeResult& result) {
    const auto* program = std::get_if<ForwardProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    return program;
}

std::optional<DecodeError> rejected(const ForwardDecodeResult& result,
                                    DecodeError::Code code) {
    const auto* error = std::get_if<DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == code);
        ARMIGO_CHECK(error->message()[0] != '\0');
        return *error;
    }
    return std::nullopt;
}

// Renders the node pc list for failure messages.
std::string describe_pcs(const std::vector<std::uint32_t>& pcs) {
    std::string text = "{";
    for (std::uint32_t pc : pcs) {
        text += std::to_string(pc) + ",";
    }
    return text + "}";
}

std::vector<std::uint32_t> node_pcs(const ForwardProgram& program) {
    std::vector<std::uint32_t> pcs;
    for (const ForwardInstruction& node : program.instructions) {
        if (const auto* instruction = std::get_if<Instruction>(&node)) {
            pcs.push_back(instruction->pc);
        } else {
            pcs.push_back(std::get<ForwardBranch>(node).source_pc);
        }
    }
    return pcs;
}

const Instruction* find_instruction(const ForwardProgram& program,
                                    std::uint32_t pc) {
    for (const ForwardInstruction& node : program.instructions) {
        if (const auto* instruction = std::get_if<Instruction>(&node);
            instruction != nullptr && instruction->pc == pc) {
            return instruction;
        }
    }
    ARMIGO_CHECK(false);
    return nullptr;
}

const ForwardBranch* find_branch(const ForwardProgram& program,
                                 std::uint32_t source) {
    for (const ForwardInstruction& node : program.instructions) {
        if (const auto* branch = std::get_if<ForwardBranch>(&node);
            branch != nullptr && branch->source_pc == source) {
            return branch;
        }
    }
    ARMIGO_CHECK(false);
    return nullptr;
}

void expect_pt(const Instruction& instruction) {
    ARMIGO_CHECK_EQ(instruction.predicate.index, std::uint8_t{7});
    ARMIGO_CHECK_EQ(instruction.predicate.inverted, false);
}

}  // namespace

ARMIGO_TEST(forward_program, straight_line_matches_strict_shape) {
    const Fixture data = make_fixture(Stage::Vertex, {mov_word(4, 7),
                                                      exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK(program->stage == Stage::Vertex);
    ARMIGO_CHECK_EQ(program->program_size, std::uint32_t{32});
    ARMIGO_CHECK(std::memcmp(program->sph.data(), data.code.data() + 0x30,
                             program->sph.size()) == 0);
    ARMIGO_CHECK_EQ(program->instructions.size(), std::size_t{2});
    const Instruction* first = find_instruction(*program, 8);
    const Instruction* last = find_instruction(*program, 16);
    if (first != nullptr && last != nullptr) {
        ARMIGO_CHECK(first->family == OpcodeFamily::Mov32Immediate);
        ARMIGO_CHECK(last->family == OpcodeFamily::Exit);
        expect_pt(*first);
        expect_pt(*last);
        ARMIGO_CHECK_EQ(first->original_word, mov_word(4, 7));
    }
}

ARMIGO_TEST(forward_program, pt_branch_prunes_unsupported_fallthrough) {
    const Fixture data = make_fixture(
        Stage::Vertex,
        {mov_word(4, 7), branch_word(16, 7), unknown_word(), exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    // The PT branch's fallthrough lands on the unknown word at pc 24; the
    // decode only succeeds because that slot is never fetched.
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,40,}"});
    const ForwardBranch* branch = find_branch(*program, 16);
    if (branch != nullptr) {
        ARMIGO_CHECK_EQ(branch->target_pc, std::uint32_t{40});
        ARMIGO_CHECK_EQ(branch->predicate.index, std::uint8_t{7});
        ARMIGO_CHECK_EQ(branch->predicate.inverted, false);
    }
    ARMIGO_CHECK(find_instruction(*program, 40) != nullptr);
}

ARMIGO_TEST(forward_program, inverted_pt_branch_falls_through_only) {
    const Fixture data = make_fixture(
        Stage::Vertex,
        {mov_word(4, 7), branch_word(16, 15), exit_word(7), unknown_word()});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    // The !PT branch never admits its target; the unknown word at pc 40 is
    // structurally pruned.
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,24,}"});
    const ForwardBranch* branch = find_branch(*program, 16);
    if (branch != nullptr) {
        ARMIGO_CHECK_EQ(branch->predicate.index, std::uint8_t{7});
        ARMIGO_CHECK_EQ(branch->predicate.inverted, true);
        ARMIGO_CHECK_EQ(branch->target_pc, std::uint32_t{40});
    }
}

ARMIGO_TEST(forward_program, guarded_branch_admits_both_edges) {
    const Fixture data = make_fixture(
        Stage::Vertex,
        {mov_word(4, 7), branch_word(16, 1), mov_word(5, 7), exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    // Both arms of the diamond join at the PT EXIT at pc 40; the fallthrough
    // from pc 24 skips the scheduling word at pc 32.
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,24,40,}"});
    const ForwardBranch* branch = find_branch(*program, 16);
    if (branch != nullptr) {
        ARMIGO_CHECK_EQ(branch->predicate.index, std::uint8_t{1});
        ARMIGO_CHECK_EQ(branch->predicate.inverted, false);
    }
}

ARMIGO_TEST(forward_program, guarded_branch_preserves_inversion) {
    const Fixture data = make_fixture(
        Stage::Vertex,
        {branch_word(16, 9), exit_word(7), exit_word(7), exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    // Guard raw 9 is inverted P1: both the target (pc 40) and the
    // fallthrough (pc 16) terminate, and pc 24 stays unreachable.
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,40,}"});
    const ForwardBranch* branch = find_branch(*program, 8);
    if (branch != nullptr) {
        ARMIGO_CHECK_EQ(branch->predicate.index, std::uint8_t{1});
        ARMIGO_CHECK_EQ(branch->predicate.inverted, true);
    }
}

ARMIGO_TEST(forward_program, guarded_exit_falls_through) {
    const Fixture data = make_fixture(Stage::Vertex, {exit_word(1),
                                                      mov_word(6, 7),
                                                      exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,24,}"});
    const Instruction* guarded = find_instruction(*program, 8);
    if (guarded != nullptr) {
        ARMIGO_CHECK(guarded->family == OpcodeFamily::Exit);
        ARMIGO_CHECK_EQ(guarded->predicate.index, std::uint8_t{1});
        ARMIGO_CHECK_EQ(guarded->predicate.inverted, false);
    }
}

ARMIGO_TEST(forward_program, guarded_exit_chain_reaches_end) {
    const Fixture data = make_fixture(Stage::Vertex, {exit_word(1),
                                                      exit_word(2),
                                                      exit_word(3)});
    const std::optional<DecodeError> error = rejected(run(data),
                                                      DecodeError::Code::MissingExit);
    if (error.has_value()) {
        ARMIGO_CHECK_EQ(error->pc, std::uint32_t{24});
    }
}

ARMIGO_TEST(forward_program, inverted_pt_exit_falls_through) {
    const Fixture data = make_fixture(Stage::Vertex, {exit_word(15),
                                                      mov_word(4, 7),
                                                      exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    // An inverted-PT EXIT can never end a path, so the walk continues into
    // its fallthrough and only the last PT EXIT terminates.
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,16,24,}"});
    const Instruction* terminal = find_instruction(*program, 8);
    if (terminal != nullptr) {
        ARMIGO_CHECK(terminal->family == OpcodeFamily::Exit);
        ARMIGO_CHECK_EQ(terminal->predicate.index, std::uint8_t{7});
        ARMIGO_CHECK_EQ(terminal->predicate.inverted, true);
    }
}

ARMIGO_TEST(forward_program, fallthrough_past_end_is_missing_exit) {
    const Fixture data = make_fixture(Stage::Vertex, {mov_word(4, 7),
                                                      mov_word(5, 7),
                                                      mov_word(6, 7)});
    const std::optional<DecodeError> error = rejected(run(data),
                                                      DecodeError::Code::MissingExit);
    if (error.has_value()) {
        ARMIGO_CHECK_EQ(error->pc, std::uint32_t{24});
    }
}

ARMIGO_TEST(forward_program, pt_exit_on_first_slot_terminates) {
    const Fixture data = make_fixture(Stage::Vertex, {exit_word(7),
                                                      mov_word(4, 7),
                                                      mov_word(5, 7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(describe_pcs(node_pcs(*program)),
                    std::string{"{8,}"});
}

ARMIGO_TEST(forward_program, conditional_branch_word_is_explicit_error) {
    // Flow test 1 instead of the always-true test.
    const Fixture conditional = make_fixture(
        Stage::Vertex,
        {(guarded_word(0xe240, 7) | field_bits(0, 5, 1) |
          field_bits(20, 24, 16)),
         exit_word(7)});
    (void)rejected(run(conditional), DecodeError::Code::UnsupportedControlFlow);

    // Reserved low bits outside the admitted direct-branch profile.
    const Fixture reserved = make_fixture(
        Stage::Vertex,
        {(guarded_word(0xe240, 7) | field_bits(0, 5, 15) |
          field_bits(6, 1, 1) | field_bits(20, 24, 16)),
         exit_word(7)});
    (void)rejected(run(reserved), DecodeError::Code::UnsupportedControlFlow);
}

ARMIGO_TEST(forward_program, branch_unsupported_field_is_classified) {
    // Raw word 0xe240000001070001: flow test 1 instead of the always-true
    // test, so the flow test is the failing field.
    const Fixture conditional = make_fixture(
        Stage::Vertex, {std::uint64_t{0xe240000001070001ull}, exit_word(7)});
    std::optional<DecodeError> error =
        rejected(run(conditional), DecodeError::Code::UnsupportedControlFlow);
    if (error.has_value()) {
        ARMIGO_CHECK_EQ(error->pc, std::uint32_t{8});
        ARMIGO_CHECK(error->field == DecodeField::FlowTest);
        ARMIGO_CHECK_EQ(error->value, std::uint64_t{1});
        ARMIGO_CHECK_EQ(error->expected, std::uint64_t{15});
    }

    // Raw word 0xe24000000107002f: always-true flow test with bit 5 set, so
    // the constant-buffer target address mode is the failing field.
    const Fixture address_mode = make_fixture(
        Stage::Vertex, {std::uint64_t{0xe24000000107002full}, exit_word(7)});
    error = rejected(run(address_mode),
                     DecodeError::Code::UnsupportedControlFlow);
    if (error.has_value()) {
        ARMIGO_CHECK(error->field == DecodeField::AddressMode);
        ARMIGO_CHECK_EQ(error->value, std::uint64_t{1});
        ARMIGO_CHECK_EQ(error->expected, std::uint64_t{0});
    }

    // Reserved bits: the low range 6..15 is reported when any of its bits is
    // set; the high range 44..47 only when the low range is clear. Both
    // expect zero.
    const Fixture reserved_low = make_fixture(
        Stage::Vertex,
        {branch_word(16, 7) | field_bits(6, 10, 0x2a5), exit_word(7)});
    error = rejected(run(reserved_low),
                     DecodeError::Code::UnsupportedControlFlow);
    if (error.has_value()) {
        ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
        ARMIGO_CHECK_EQ(error->value, (std::uint64_t{0x2a5} << 6));
        ARMIGO_CHECK_EQ(error->expected, std::uint64_t{0});
    }

    const Fixture reserved_high = make_fixture(
        Stage::Vertex,
        {branch_word(16, 7) | field_bits(44, 4, 0x9), exit_word(7)});
    error = rejected(run(reserved_high),
                     DecodeError::Code::UnsupportedControlFlow);
    if (error.has_value()) {
        ARMIGO_CHECK(error->field == DecodeField::ReservedBits);
        ARMIGO_CHECK_EQ(error->value, (std::uint64_t{0x9} << 44));
        ARMIGO_CHECK_EQ(error->expected, std::uint64_t{0});
    }
}

ARMIGO_TEST(forward_program, backward_target_is_explicit_error) {
    // pc 8 + 8 - 16 normalizes to 8: a reachable backward (self) edge is
    // rejected even though the guard could be false at run time. This is the
    // documented conservative limit, not dynamic reachability.
    const Fixture data = make_fixture(Stage::Vertex,
                                      {branch_word(disp24(-16), 7),
                                       mov_word(4, 7),
                                       exit_word(7)});
    (void)rejected(run(data), DecodeError::Code::UnsupportedControlFlow);
}

ARMIGO_TEST(forward_program, inverted_pt_backward_edge_conservative_error) {
    // The branch helper validates every admitted branch as forward-only
    // regardless of its guard: a self edge behind the always-false
    // inverted-PT guard is rejected too, never silently dropped as dead.
    const Fixture data = make_fixture(Stage::Vertex,
                                      {branch_word(disp24(-16), 15),
                                       mov_word(4, 7),
                                       exit_word(7)});
    (void)rejected(run(data), DecodeError::Code::UnsupportedControlFlow);
}

ARMIGO_TEST(forward_program, malformed_target_is_explicit_error) {
    // pc 8 + 8 + 256 lands far past the 64-byte program region.
    const Fixture data = make_fixture(Stage::Vertex,
                                      {branch_word(256, 7),
                                       mov_word(4, 7),
                                       exit_word(7)});
    (void)rejected(run(data), DecodeError::Code::UnsupportedControlFlow);
}

ARMIGO_TEST(forward_program, unknown_word_is_never_a_nop) {
    const Fixture unguarded = make_fixture(Stage::Vertex, {unknown_word(),
                                                           exit_word(7)});
    (void)rejected(run(unguarded), DecodeError::Code::UnknownOpcode);

    // A common-looking guard on a dead-looking word must not turn it into a
    // NOP either.
    const Fixture guarded = make_fixture(Stage::Vertex, {unknown_word() |
                                                             field_bits(16, 4, 3),
                                                         exit_word(7)});
    (void)rejected(run(guarded), DecodeError::Code::UnknownOpcode);
}

ARMIGO_TEST(forward_program, inverted_pt_unknown_is_explicit_error) {
    // The always-false inverted-PT guard must not soften an unknown word
    // into a NOP either.
    const Fixture data = make_fixture(Stage::Vertex,
                                      {unknown_word() | field_bits(16, 4, 15),
                                       exit_word(7)});
    (void)rejected(run(data), DecodeError::Code::UnknownOpcode);
}

ARMIGO_TEST(forward_program, instruction_guards_preserved_all_16) {
    for (std::uint32_t raw = 0; raw < 16; ++raw) {
        const Fixture data = make_fixture(Stage::Vertex,
                                          {mov_word(4, static_cast<std::uint8_t>(raw)),
                                           exit_word(7)});
        const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
        if (program == nullptr) {
            return;
        }
        const Instruction* instruction = find_instruction(*program, 8);
        if (instruction == nullptr) {
            return;
        }
        ARMIGO_CHECK_EQ(instruction->predicate.index,
                        static_cast<std::uint8_t>(raw & 7));
        ARMIGO_CHECK_EQ(instruction->predicate.inverted, raw >= 8);
        // The original word, including the guard bits, survives verbatim.
        ARMIGO_CHECK_EQ(instruction->original_word,
                        mov_word(4, static_cast<std::uint8_t>(raw)));
    }
}

ARMIGO_TEST(forward_program, instruction_limit_is_explicit) {
    const Fixture data = make_fixture(Stage::Vertex, {mov_word(4, 7),
                                                      mov_word(5, 7),
                                                      exit_word(7)});
    const std::optional<DecodeError> error =
        rejected(run(data, DecodeLimits{2}), DecodeError::Code::InstructionLimit);
    if (error.has_value()) {
        ARMIGO_CHECK_EQ(error->pc, std::uint32_t{24});
        ARMIGO_CHECK_EQ(error->value, std::uint64_t{3});
        ARMIGO_CHECK_EQ(error->expected, std::uint64_t{2});
    }
}

ARMIGO_TEST(forward_program, instruction_limit_zero_and_exact_caps) {
    const Fixture data = make_fixture(Stage::Vertex, {mov_word(4, 7),
                                                      exit_word(7)});
    // A zero cap rejects the first reachable slot as an explicit limit
    // error, never as an empty success.
    const std::optional<DecodeError> zero =
        rejected(run(data, DecodeLimits{0}),
                 DecodeError::Code::InstructionLimit);
    if (zero.has_value()) {
        ARMIGO_CHECK_EQ(zero->pc, std::uint32_t{8});
        ARMIGO_CHECK_EQ(zero->value, std::uint64_t{1});
        ARMIGO_CHECK_EQ(zero->expected, std::uint64_t{0});
    }
    // A cap equal to the reachable node count accepts the whole program.
    const ForwardDecodeResult result = run(data, DecodeLimits{2});
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK_EQ(program->instructions.size(), std::size_t{2});
}

ARMIGO_TEST(forward_program, parse_failure_propagates) {
    Fixture data = make_fixture(Stage::Vertex, {mov_word(4, 7), exit_word(7)});
    data.control[0] = 0;
    const ForwardDecodeResult result = run(data);
    const auto* error = std::get_if<DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == DecodeError::Code::ShaderParse);
        ARMIGO_CHECK(error->parse_error.has_value());
    }
}

ARMIGO_TEST(forward_program, fragment_stage_preserved) {
    const Fixture data = make_fixture(Stage::Fragment, {mov_word(4, 7),
                                                        exit_word(7)});
    const ForwardDecodeResult result = run(data);
    const ForwardProgram* program = accepted(result);
    if (program == nullptr) {
        return;
    }
    ARMIGO_CHECK(program->stage == Stage::Fragment);
}

ARMIGO_TEST(forward_program, strict_decode_still_rejects_same_guards) {
    const Fixture data = make_fixture(Stage::Vertex, {exit_word(1),
                                                      mov_word(6, 7),
                                                      exit_word(7)});
    const DecodeResult strict = decode(span(data.control), span(data.code));
    const auto* error = std::get_if<DecodeError>(&strict);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == DecodeError::Code::UnsupportedPredicate);
        ARMIGO_CHECK_EQ(error->pc, std::uint32_t{8});
        ARMIGO_CHECK(error->family == OpcodeFamily::Exit);
    }
}

ARMIGO_TEST(forward_program, strict_decode_still_rejects_branch_words) {
    const Fixture data = make_fixture(Stage::Vertex,
                                      {branch_word(16, 7),
                                       mov_word(4, 7),
                                       exit_word(7)});
    const DecodeResult strict = decode(span(data.control), span(data.code));
    const auto* error = std::get_if<DecodeError>(&strict);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code ==
                     DecodeError::Code::UnsupportedControlFlow);
        ARMIGO_CHECK(error->family == OpcodeFamily::Branch);
    }
}

}  // namespace armigo::shader::test
