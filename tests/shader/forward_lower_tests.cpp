// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Dedicated tests for guarded forward-program lowering: synthetic programs
// only, no game-derived bytes. CPU graph results are evaluated exactly; one
// fixture additionally emits Metal Shading Language for compilation.

#include <armigo/shader/graphics_forward.h>

#include "test_support.h"

#include <armigo/shader/families/predicate_logic.h>
#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell_flow.h>
#include <armigo/shader/maxwell_forward_program.h>
#include <armigo/shader/maxwell_predicates.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

namespace gx = armigo::shader::graphics;
namespace mw = armigo::shader::maxwell;
namespace si = armigo::shader_input;
namespace gx_msl = armigo::shader::graphics_msl;

using si::Stage;

constexpr std::uint32_t kFragmentR0 = 0x40490fdb;  // synthetic literal

std::uint64_t field_bits(unsigned shift, unsigned width, std::uint64_t value) {
    return (value << shift) & (((std::uint64_t{1} << width) - 1) << shift);
}

std::uint64_t guarded_word(std::uint16_t high, std::uint8_t predicate_raw) {
    return (static_cast<std::uint64_t>(high) << 48) |
           field_bits(16, 4, predicate_raw);
}

std::uint64_t mov32i_word(std::uint8_t destination, std::uint32_t bits) {
    return guarded_word(0x0100, 7) | field_bits(0, 8, destination) |
           field_bits(12, 4, 15) | field_bits(20, 32, bits);
}

std::uint64_t exit_word() { return guarded_word(0xe300, 7) | 15; }

std::array<std::byte, 80> stage_sph(Stage stage) {
    std::array<std::byte, 80> sph{};
    const std::uint32_t word0 =
        si::sph_stage_type_for(stage) << si::format::sph_stage_shift;
    for (unsigned index = 0; index < 4; ++index) {
        sph[index] = static_cast<std::byte>(word0 >> (index * 8));
    }
    return sph;
}

std::array<std::byte, 80> fragment_sph(std::uint32_t color_mask) {
    auto sph = stage_sph(Stage::Fragment);
    for (unsigned index = 0; index < 4; ++index) {
        sph[0x48 + index] = static_cast<std::byte>(color_mask >> (index * 8));
    }
    return sph;
}

std::array<std::byte, 80> vertex_sph_position_output() {
    auto sph = stage_sph(Stage::Vertex);
    sph[0x35] = std::byte{0x10};  // position x output declared
    return sph;
}

// Synthetic graphics shader container: control prefix, SPH at 0x30, program
// words in 32-byte scheduling groups starting at 0x80.
struct Container {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

Container make_container(Stage stage, const std::array<std::byte, 80>& sph,
                         const std::vector<std::uint64_t>& words) {
    const std::size_t groups = (words.size() + 2) / 3;
    const std::uint32_t program_size =
        static_cast<std::uint32_t>(groups * si::format::scheduling_group_size);
    const std::uint32_t code_offset = 0x80 + program_size;
    Container result{std::vector<std::uint8_t>(0x718),
                     std::vector<std::uint8_t>(code_offset)};
    auto put32 = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned index = 0; index < 4; ++index) {
            result.control[offset + index] =
                static_cast<std::uint8_t>(value >> (index * 8));
        }
    };
    put32(0, si::format::control_magic);
    put32(4, si::format::control_tag);
    put32(si::format::control_version_offset, 14);
    put32(si::format::control_size_offset,
          static_cast<std::uint32_t>(result.control.size()));
    put32(si::format::c1_offset_field, code_offset);
    put32(si::format::code_size_field, code_offset);
    put32(si::format::code_offset_field, 0x30);
    put32(si::format::stage_field, static_cast<std::uint32_t>(stage));
    for (unsigned index = 0; index < 4; ++index) {
        result.code[index] =
            static_cast<std::uint8_t>(0x12345678u >> (index * 8));
    }
    for (std::size_t index = 0; index < sph.size(); ++index) {
        result.code[0x30 + index] = std::to_integer<std::uint8_t>(sph[index]);
    }
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::size_t offset =
            0x80 + (index / 3) * si::format::scheduling_group_size +
            si::format::instruction_word_size +
            (index % 3) * si::format::instruction_word_size;
        for (unsigned byte = 0; byte < 8; ++byte) {
            result.code[offset + byte] =
                static_cast<std::uint8_t>(words[index] >> (byte * 8));
        }
    }
    return result;
}

si::ByteSpan span_bytes(const std::vector<std::uint8_t>& bytes) {
    return si::ByteSpan{bytes.data(), bytes.size()};
}

// Forged-input helper: builds the public ForwardProgram view directly, so
// metadata validation and lowering are exercised independently of the
// decoder.
mw::Instruction instruction(std::uint32_t pc, mw::OpcodeFamily family,
                            mw::Predicate guard, auto payload) {
    return mw::Instruction{pc, 0, family, guard, std::move(payload)};
}

mw::ForwardProgram forged(Stage stage, const std::array<std::byte, 80>& sph,
                          std::uint32_t program_size,
                          std::vector<mw::ForwardInstruction> nodes) {
    mw::ForwardProgram program;
    program.stage = stage;
    program.sph = sph;
    program.program_size = program_size;
    program.instructions = std::move(nodes);
    return program;
}

mw::ForwardBranch forward_branch(std::uint32_t source, std::uint32_t target,
                                 mw::Predicate guard) {
    return mw::ForwardBranch{source, target, guard};
}

const gx::Error* error_of(const gx::LowerResult& result) {
    return std::get_if<gx::Error>(&result);
}

const gx::Program* program_of(const gx::LowerResult& result) {
    return std::get_if<gx::Program>(&result);
}

std::string error_text(const gx::Error& error) {
    std::ostringstream stream;
    stream << error.message() << " (code " << static_cast<int>(error.code)
           << " pc " << error.pc << " detail " << error.detail << ")";
    return stream.str();
}

std::uint32_t color_output(const gx::Program& program,
                           const gx::Inputs& inputs = {}) {
    const gx::EvaluateResult result = gx::evaluate(program, inputs);
    const auto* evaluation = std::get_if<gx::Evaluation>(&result);
    if (evaluation == nullptr) return 0xdeadbeefu;
    if (evaluation->outputs.empty()) return 0xdeadbeefu;
    for (const gx::Output& output : evaluation->outputs) {
        if (output.kind == gx::OutputKind::Color && output.component == 0) {
            return output.bits;
        }
    }
    return 0xdeadbeefu;
}

bool has_branch_terminator(const gx::Program& program) {
    for (const gx::BasicBlock& block : program.blocks()) {
        if (std::holds_alternative<gx::BranchIf>(block.terminator)) return true;
    }
    return false;
}

mw::Predicate pt() { return mw::Predicate{7, false}; }
mw::Predicate pred(unsigned index, bool inverted = false) {
    return mw::Predicate{static_cast<std::uint8_t>(index), inverted};
}

mw::Instruction mov32i(std::uint32_t pc, std::uint8_t destination,
                       std::uint32_t bits, mw::Predicate guard = pt()) {
    return instruction(pc, mw::OpcodeFamily::Mov32Immediate, guard,
                       mw::Mov32Immediate{mw::RegisterOperand{destination}, 15,
                                          bits});
}

mw::Instruction fadd_reg(std::uint32_t pc, std::uint8_t destination,
                         std::uint8_t source_a, std::uint8_t source_b,
                         mw::Predicate guard = pt()) {
    return instruction(pc, mw::OpcodeFamily::FloatAddRegister, guard,
                       mw::FloatAdd{mw::RegisterOperand{destination},
                                    mw::RegisterOperand{source_a},
                                    mw::RegisterOperand{source_b}});
}

mw::Instruction mov_constant(std::uint32_t pc, std::uint8_t destination,
                             std::uint8_t binding, std::uint32_t byte_offset,
                             mw::Predicate guard = pt()) {
    return instruction(pc, mw::OpcodeFamily::MovConstant, guard,
                       mw::MovConstant{mw::RegisterOperand{destination},
                                       mw::ConstantBufferOperand{binding,
                                                                 byte_offset},
                                       15});
}

mw::Instruction attribute_store(std::uint32_t pc, std::uint8_t source,
                                std::uint16_t component,
                                mw::Predicate guard = pt()) {
    return instruction(
        pc, mw::OpcodeFamily::AttributeStore, guard,
        mw::AttributeAccess{mw::RegisterOperand{source},
                            mw::RegisterOperand{255},
                            mw::RegisterOperand{255},
                            static_cast<std::uint16_t>(component * 4), 0,
                            false});
}

mw::Instruction isetp(std::uint32_t pc, std::uint8_t destination_a,
                      std::uint8_t destination_b, std::uint8_t lhs,
                      std::uint8_t rhs, mw::Predicate guard = pt(),
                      bool signed_compare = false) {
    return instruction(pc, mw::OpcodeFamily::IntegerSetPredicateRegister,
                       guard,
                       mw::IntegerSetPredicate{
                           destination_a, destination_b,
                           mw::RegisterOperand{lhs}, mw::RegisterOperand{rhs},
                           mw::Predicate{7, false}, signed_compare});
}

mw::Instruction exit_at(std::uint32_t pc, mw::Predicate guard = pt()) {
    return instruction(pc, mw::OpcodeFamily::Exit, guard, mw::Exit{15});
}

mw::ForwardInstruction branch_at(std::uint32_t source, std::uint32_t target,
                                 mw::Predicate guard) {
    return forward_branch(source, target, guard);
}

// Forged fragment program over whole scheduling groups.
struct ForwardFixture {
    mw::ForwardProgram program;

    ForwardFixture(std::uint32_t groups,
                   std::vector<mw::ForwardInstruction> nodes) {
        program = forged(Stage::Fragment, fragment_sph(1),
                         groups * si::format::scheduling_group_size,
                         std::move(nodes));
    }
};

void advance_pc(std::uint32_t& pc) {
    pc += si::format::instruction_word_size;
    if ((pc % si::format::scheduling_group_size) == 0) {
        pc += si::format::instruction_word_size;
    }
}

std::uint32_t program_size_for(std::uint32_t pc) {
    return ((pc / si::format::scheduling_group_size) + 1) *
           si::format::scheduling_group_size;
}

mw::ForwardProgram guarded_write_chain(unsigned rounds) {
    std::vector<mw::ForwardInstruction> nodes;
    std::uint32_t pc = 8;
    const auto append = [&](mw::ForwardInstruction node) {
        nodes.push_back(std::move(node));
        advance_pc(pc);
    };
    append(mov32i(pc, 0, 0u));
    append(mov32i(pc, 1, 0u));
    append(mov32i(pc, 2, 0u));
    append(mov32i(pc, 3, 0u));
    for (unsigned round = 0; round < rounds; ++round) {
        append(mov_constant(pc, 17, 0, round * 4));
        append(isetp(pc, static_cast<std::uint8_t>(round % 7), 7, 17,
                     255));
        append(mov32i(pc, 0, round + 1,
                      pred(static_cast<unsigned>(round % 7))));
    }
    append(exit_at(pc));
    return forged(Stage::Fragment, fragment_sph(0xf), program_size_for(pc),
                  std::move(nodes));
}

mw::ForwardProgram branch_diamond_chain(unsigned diamonds) {
    std::vector<mw::ForwardInstruction> nodes;
    std::uint32_t pc = 8;
    const auto append = [&](mw::ForwardInstruction node) {
        nodes.push_back(std::move(node));
        advance_pc(pc);
    };
    append(mov32i(pc, 0, 0u));
    append(mov32i(pc, 1, 0u));
    append(mov32i(pc, 2, 0u));
    append(mov32i(pc, 3, 0u));
    for (unsigned round = 0; round < diamonds; ++round) {
        const std::uint8_t predicate = static_cast<std::uint8_t>(round % 7);
        append(mov_constant(pc, 17, 0, round * 4));
        append(isetp(pc, predicate, 7, 17, 255));

        const std::size_t conditional = nodes.size();
        append(branch_at(pc, 0, pred(predicate)));
        append(mov32i(pc, 0, round * 2 + 1));

        const std::size_t skip_taken = nodes.size();
        append(branch_at(pc, 0, pt()));
        const std::uint32_t taken_pc = pc;
        append(mov32i(pc, 0, round * 2 + 2));
        const std::uint32_t join_pc = pc;

        std::get<mw::ForwardBranch>(nodes[conditional]).target_pc = taken_pc;
        std::get<mw::ForwardBranch>(nodes[skip_taken]).target_pc = join_pc;
    }
    append(exit_at(pc));
    return forged(Stage::Fragment, fragment_sph(0xf), program_size_for(pc),
                  std::move(nodes));
}

enum class PredicatePattern { AllFalse, AllTrue, Alternating };

std::vector<std::uint32_t> predicate_words(unsigned count,
                                           PredicatePattern pattern) {
    std::vector<std::uint32_t> words(count);
    for (unsigned index = 0; index < count; ++index) {
        words[index] = pattern == PredicatePattern::AllTrue ||
                               (pattern == PredicatePattern::Alternating &&
                                (index % 2) == 0)
                           ? 1u
                           : 0u;
    }
    return words;
}

void write_msl_fixture(const char* name, const std::string& source) {
    const char* directory = std::getenv("ARMIGO_FORWARD_MSL_DIR");
    if (directory == nullptr || *directory == '\0') return;
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    std::ofstream output{std::string{directory} + "/" + name + ".metal",
                         std::ios::binary | std::ios::trunc};
    ARMIGO_CHECK(static_cast<bool>(output));
    output << source;
}

ARMIGO_TEST(forward_lower, straight_line_matches_legacy_path) {
    const std::array<std::byte, 80> sph = fragment_sph(1);
    const Container container = make_container(
        Stage::Fragment, sph, {mov32i_word(0, kFragmentR0), exit_word()});
    const mw::DecodeResult strict =
        mw::decode(span_bytes(container.control), span_bytes(container.code));
    const mw::ForwardDecodeResult reachable = mw::decode_forward_program(
        span_bytes(container.control), span_bytes(container.code));
    const auto decoded = std::get_if<mw::DecodedProgram>(&strict);
    const auto forward = std::get_if<mw::ForwardProgram>(&reachable);
    ARMIGO_CHECK(decoded != nullptr);
    ARMIGO_CHECK(forward != nullptr);
    if (decoded == nullptr || forward == nullptr) return;

    const gx::LowerResult legacy_result = gx::lower(*decoded);
    const auto* legacy_program = program_of(legacy_result);
    const gx::LowerResult graph_result = gx::lower_forward(*forward);
    const auto* graph_program = program_of(graph_result);
    ARMIGO_CHECK(legacy_program != nullptr);
    ARMIGO_CHECK(graph_program != nullptr);
    if (legacy_program == nullptr || graph_program == nullptr) return;
    ARMIGO_CHECK(!graph_program->blocks().empty());
    ARMIGO_CHECK_EQ(color_output(*graph_program), kFragmentR0);
    ARMIGO_CHECK_EQ(color_output(*graph_program), color_output(*legacy_program));
}

ARMIGO_TEST(forward_lower, isetp_ne_drives_both_outcomes) {
    // 1 != RZ is true: the guarded write executes.
    const ForwardFixture taken(2, {mov32i(8, 1, 0x3f800000u),
                                   isetp(16, 0, 7, 1, 255),
                                   mov32i(24, 0, 0x40u, pred(0)),
                                   exit_at(40)});
    // 0 != RZ is false even with the signed field set: the guarded write is
    // skipped and the earlier value survives.
    const ForwardFixture skipped(2, {mov32i(8, 0, 7u),
                                     mov32i(16, 1, 0u),
                                     isetp(24, 0, 7, 1, 255, pt(), true),
                                     mov32i(40, 0, 0x40u, pred(0)),
                                     exit_at(48)});
    const gx::LowerResult taken_result = gx::lower_forward(taken.program);
    const gx::LowerResult skipped_result = gx::lower_forward(skipped.program);
    const auto* taken_program = program_of(taken_result);
    const auto* skipped_program = program_of(skipped_result);
    ARMIGO_CHECK(taken_program != nullptr);
    ARMIGO_CHECK(skipped_program != nullptr);
    if (taken_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*taken_program), 0x40u);
    }
    if (skipped_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*skipped_program), 7u);
    }
}

ARMIGO_TEST(forward_lower, isetp_ne_high_bits_ignores_signedness) {
    // 0x80000000 != RZ as raw words; the signed field must not invert it.
    const ForwardFixture high(2, {mov32i(8, 1, 0x80000000u),
                                  isetp(16, 0, 7, 1, 255, pt(), true),
                                  mov32i(24, 0, 0x40u, pred(0)),
                                  exit_at(40)});
    const gx::LowerResult result = gx::lower_forward(high.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 0x40u);
    }
}

ARMIGO_TEST(forward_lower, isetp_alias_and_pt_discard) {
    // A and B write the same destination: the second (inverted) write wins.
    const ForwardFixture aliased(2, {mov32i(8, 1, 0x3f800000u),
                                     isetp(16, 1, 1, 1, 255),
                                     mov32i(24, 0, 0x11u, pred(1)),
                                     mov32i(40, 0, 0x22u, pred(1, true)),
                                     exit_at(48)});
    const ForwardFixture discarded(2, {mov32i(8, 1, 0x3f800000u),
                                       isetp(16, 0, 7, 1, 255),
                                       mov32i(24, 0, 0x11u, pred(0)),
                                       exit_at(40)});
    const gx::LowerResult aliased_result = gx::lower_forward(aliased.program);
    const gx::LowerResult kept_result = gx::lower_forward(discarded.program);
    const auto* aliased_program = program_of(aliased_result);
    const auto* kept = program_of(kept_result);
    ARMIGO_CHECK(aliased_program != nullptr);
    ARMIGO_CHECK(kept != nullptr);
    if (aliased_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*aliased_program), 0x22u);
    }
    if (kept != nullptr) {
        // The B write targeted PT and discarded; A stays the guarded value.
        ARMIGO_CHECK_EQ(color_output(*kept), 0x11u);
    }
}

ARMIGO_TEST(forward_lower, conditional_write_supports_guarded_read) {
    // R3 is written only under the forked predicate and read only under the
    // same predicate; no path reads an undefined value.
    const ForwardFixture fixture(3, {mov32i(8, 0, 7u),
                                     mov32i(16, 1, 0x3f800000u),
                                     fadd_reg(24, 2, 1, 1),
                                     isetp(40, 0, 7, 2, 255),
                                     mov32i(48, 3, 0x42c80000u, pred(0)),
                                     fadd_reg(56, 0, 3, 3, pred(0)),
                                     exit_at(72)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        // Taken path: 100.0f + 100.0f = 200.0f.
        ARMIGO_CHECK_EQ(color_output(*program), 0x43480000u);
    }
    // With a zero addend the comparison is false: both guarded writes skip
    // and the preset 7 survives.
    ForwardFixture zero = fixture;
    zero.program.instructions[1] = mov32i(16, 1, 0u);
    const gx::LowerResult zero_result = gx::lower_forward(zero.program);
    const auto* zero_program = program_of(zero_result);
    ARMIGO_CHECK(zero_program != nullptr);
    if (zero_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*zero_program), 7u);
    }
}

ARMIGO_TEST(forward_lower, unguarded_undefined_read_is_rejected) {
    const ForwardFixture fixture(1, {fadd_reg(8, 0, 5, 5), exit_at(16)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UninitializedRegister);
        ARMIGO_CHECK_EQ(error->detail, std::uint32_t{5});
    }
}

ARMIGO_TEST(forward_lower, uninitialized_predicate_guard_is_rejected) {
    const ForwardFixture fixture(1, {mov32i(8, 0, 1u, pred(3)), exit_at(16)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UninitializedRegister);
        ARMIGO_CHECK_EQ(error->detail, std::uint32_t{3});
    }
}

ARMIGO_TEST(forward_lower, known_facts_prune_correlated_branches) {
    // Literal comparison operands resolve both destinations immediately, so
    // the correlated branch lowers without a BranchIf terminator.
    const ForwardFixture equal(2, {mov32i(8, 1, 1u),
                                   isetp(16, 0, 1, 1, 255),
                                   branch_at(24, 48, pred(0)),
                                   mov32i(40, 0, 0xccu),
                                   mov32i(48, 0, 0x11u),
                                   exit_at(56)});
    const gx::LowerResult equal_result = gx::lower_forward(equal.program);
    const auto* equal_program = program_of(equal_result);
    ARMIGO_CHECK(equal_program != nullptr);
    if (equal_program != nullptr) {
        ARMIGO_CHECK(!has_branch_terminator(*equal_program));
        ARMIGO_CHECK_EQ(color_output(*equal_program), 0x11u);
    }
    // The inverted condition is known false: only the fallthrough is live
    // and the structurally reachable target is fact-pruned.
    const ForwardFixture inverted(2, {mov32i(8, 1, 1u),
                                      isetp(16, 0, 1, 1, 255),
                                      branch_at(24, 48, pred(0, true)),
                                      mov32i(40, 0, 0xccu),
                                      exit_at(48)});
    const gx::LowerResult inverted_result = gx::lower_forward(inverted.program);
    const auto* inverted_program = program_of(inverted_result);
    ARMIGO_CHECK(inverted_program != nullptr);
    if (inverted_program != nullptr) {
        ARMIGO_CHECK(!has_branch_terminator(*inverted_program));
        ARMIGO_CHECK_EQ(color_output(*inverted_program), 0xccu);
    }
}

ARMIGO_TEST(forward_lower, isetp_complement_guards_share_one_fork) {
    // Runtime-unknown comparison (the operand is an add result, not a
    // tracked literal): P0 and !P1 are the same condition, so both guarded
    // writes resolve against one fork and the compare-false path never
    // reads the R3 that only the compare-true side defines.
    const ForwardFixture inner(3, {mov32i(8, 0, 0x3f800000u),
                                   fadd_reg(16, 1, 0, 0),
                                   isetp(24, 0, 1, 1, 255),
                                   mov32i(40, 3, 0x40000000u, pred(0)),
                                   fadd_reg(48, 0, 3, 3, pred(1, true)),
                                   exit_at(56)});
    const gx::LowerResult inner_result = gx::lower_forward(inner.program);
    const auto* inner_program = program_of(inner_result);
    ARMIGO_CHECK(inner_program != nullptr);
    if (inner_program != nullptr) {
        // Compare-true side: R0 = R3 + R3 = 2.0f + 2.0f = 4.0f.
        ARMIGO_CHECK_EQ(color_output(*inner_program), 0x40800000u);
    }
    // With a zero addend the comparison is literal-known false: both
    // guarded writes skip and the preset 1.0f survives.
    ForwardFixture inner_false = inner;
    inner_false.program.instructions[1] = mov32i(16, 1, 0u);
    const gx::LowerResult inner_false_result =
        gx::lower_forward(inner_false.program);
    const auto* inner_false_program = program_of(inner_false_result);
    ARMIGO_CHECK(inner_false_program != nullptr);
    if (inner_false_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*inner_false_program), 0x3f800000u);
    }
    // The same pair through the B alias orientation: @P1 is the complement
    // of @P0, so the guarded pair runs on the compare-false side only.
    const ForwardFixture alias(3, {mov32i(8, 0, 0x3f800000u),
                                   fadd_reg(16, 1, 0, 0),
                                   isetp(24, 0, 1, 1, 255),
                                   mov32i(40, 3, 0x40000000u, pred(1)),
                                   fadd_reg(48, 0, 3, 3, pred(1)),
                                   exit_at(56)});
    const gx::LowerResult alias_result = gx::lower_forward(alias.program);
    const auto* alias_program = program_of(alias_result);
    ARMIGO_CHECK(alias_program != nullptr);
    if (alias_program != nullptr) {
        // Compare-true side: the alias-guarded pair is skipped.
        ARMIGO_CHECK_EQ(color_output(*alias_program), 0x3f800000u);
    }
    ForwardFixture alias_false = alias;
    alias_false.program.instructions[1] = mov32i(16, 1, 0u);
    const gx::LowerResult alias_false_result =
        gx::lower_forward(alias_false.program);
    const auto* alias_false_program = program_of(alias_false_result);
    ARMIGO_CHECK(alias_false_program != nullptr);
    if (alias_false_program != nullptr) {
        // Compare-false side: R0 = R3 + R3 = 4.0f.
        ARMIGO_CHECK_EQ(color_output(*alias_false_program), 0x40800000u);
    }
}

ARMIGO_TEST(forward_lower, overflowing_pcs_are_rejected) {
    // Containment must add the instruction width in wide arithmetic: a
    // uint32 pc + 8 that wraps to zero must not admit an out-of-region
    // branch target or node.
    const auto check = [](gx::LowerResult result) {
        const auto* error = error_of(result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code ==
                         gx::Error::Code::InvalidInstruction);
        }
    };
    // Always-taken branch to a target past the region whose encoded pc
    // wraps program_size 32.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 32,
                                   {branch_at(8, 0xfffffff8u, pt()),
                                    exit_at(0xfffffff8u)})));
    // Target absent from the list must also reject, not wrap into success.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 32,
                                   {branch_at(8, 0xfffffff8u, pt()),
                                    exit_at(16)})));
}

ARMIGO_TEST(forward_lower, redefined_predicate_uses_fresh_identity) {
    // The second ISETP rewrites predicate 0 with a false comparison; the
    // guarded write must observe the new identity, not the stale fact.
    const ForwardFixture fixture(2, {mov32i(8, 0, 5u),
                                     mov32i(16, 1, 1u),
                                     isetp(24, 0, 7, 1, 255),
                                     isetp(40, 0, 7, 1, 1),
                                     mov32i(48, 0, 1u, pred(0)),
                                     exit_at(56)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 5u);
    }
}

ARMIGO_TEST(forward_lower, nested_branch_guarded_exit_and_scheduler_skip) {
    // The branch sits at the third slot, so its fallthrough crosses the
    // scheduling word of the next group. A guarded EXIT forks one exit; the
    // other path exits unconditionally, and the output joins through a
    // two-arm phi.
    ForwardFixture fixture(3, {mov32i(8, 1, 0x3f800000u),
                               fadd_reg(16, 2, 1, 1),
                               isetp(24, 0, 7, 2, 255),
                               branch_at(40, 72, pred(0)),
                               mov32i(48, 0, 1u),
                               exit_at(56, pred(0, true)),
                               mov32i(72, 0, 2u),
                               exit_at(80)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK(has_branch_terminator(*program));
        ARMIGO_CHECK_EQ(color_output(*program), 2u);
    }
    // With a zero addend the comparison is false and the guarded EXIT takes
    // the first snapshot instead.
    fixture.program.instructions[0] = mov32i(8, 1, 0u);
    const gx::LowerResult zero_result = gx::lower_forward(fixture.program);
    const auto* zero_program = program_of(zero_result);
    ARMIGO_CHECK(zero_program != nullptr);
    if (zero_program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*zero_program), 1u);
    }
}

ARMIGO_TEST(forward_lower, vertex_missing_output_is_rejected) {
    const mw::ForwardProgram program =
        forged(Stage::Vertex, vertex_sph_position_output(),
               si::format::scheduling_group_size, {exit_at(8)});
    const gx::LowerResult result = gx::lower_forward(program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::MissingOutput);
    }
}

ARMIGO_TEST(forward_lower, forged_missing_exit_is_rejected) {
    // The only instruction falls off the program region without an EXIT.
    const mw::ForwardProgram program = forged(Stage::Fragment, fragment_sph(1),
                                              16, {mov32i(8, 0, 1u)});
    const gx::LowerResult result = gx::lower_forward(program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::MissingExit);
    }
}

ARMIGO_TEST(forward_lower, node_limit_is_respected) {
    const mw::ForwardProgram program =
        forged(Stage::Fragment, fragment_sph(1), 48,
               {mov32i(8, 0, 1u), mov32i(16, 0, 2u), mov32i(24, 0, 3u),
                exit_at(40)});
    const gx::LowerResult result = gx::lower_forward(program, gx::Limits{2});
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::Limit);
    }
}

ARMIGO_TEST(forward_lower, forged_payload_is_rejected) {
    // Family and typed payload disagree: forged metadata never lowers.
    const ForwardFixture fixture(
        1, {instruction(8, mw::OpcodeFamily::TextureSample, pt(),
                        mw::Mov32Immediate{}),
            exit_at(16)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::InvalidInstruction);
    }
}

ARMIGO_TEST(forward_lower, psetp_is_unsupported_even_under_false_guard) {
    const ForwardFixture fixture(
        1, {instruction(8, mw::OpcodeFamily::PredicateSetPredicate,
                        mw::Predicate{7, true}, mw::PredicateSet{}),
            exit_at(16)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UnsupportedModifier);
    }
}

ARMIGO_TEST(forward_lower, false_guard_invalid_metadata_is_still_rejected) {
    // Metadata validation runs before any guard decision: an instruction a
    // statically false guard would skip is rejected with the Builder's own
    // code and pc, never lowered as a silent no-op.
    const auto rejected = [](gx::LowerResult result,
                             gx::Error::Code code, std::uint32_t pc) {
        const auto* error = error_of(result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            if (error->code != code || error->pc != pc) {
                std::cout << error_text(*error) << "\n";
            }
            ARMIGO_CHECK(error->code == code);
            ARMIGO_CHECK(error->pc == pc);
        }
    };
    const mw::Predicate never = pred(7, true);
    // MOV32I write mask 0 under !PT.
    rejected(gx::lower_forward(ForwardFixture(
                 1, {instruction(8, mw::OpcodeFamily::Mov32Immediate, never,
                                 mw::Mov32Immediate{mw::RegisterOperand{0}, 0,
                                                    1u}),
                     exit_at(16)})
                 .program),
             gx::Error::Code::UnsupportedModifier, 8);
    // Forged rounding and flush modes on a register add under !PT.
    mw::FloatAdd forged_round =
        std::get<mw::FloatAdd>(fadd_reg(8, 2, 1, 1, never).payload);
    forged_round.rounding = mw::RoundingMode::Zero;
    rejected(gx::lower_forward(ForwardFixture(
                 1, {instruction(8, mw::OpcodeFamily::FloatAddRegister, never,
                                 std::move(forged_round)),
                     exit_at(16)})
                 .program),
             gx::Error::Code::UnsupportedModifier, 8);
    mw::FloatAdd forged_flush =
        std::get<mw::FloatAdd>(fadd_reg(8, 2, 1, 1, never).payload);
    forged_flush.flush = mw::FlushMode::None;
    rejected(gx::lower_forward(ForwardFixture(
                 1, {instruction(8, mw::OpcodeFamily::FloatAddRegister, never,
                                 std::move(forged_flush)),
                     exit_at(16)})
                 .program),
             gx::Error::Code::UnsupportedModifier, 8);
    // Wrong nested FloatSource variant: the register family carries an
    // immediate instead.
    rejected(gx::lower_forward(ForwardFixture(
                 1, {instruction(
                         8, mw::OpcodeFamily::FloatAddRegister, never,
                         mw::FloatAdd{mw::RegisterOperand{2},
                                      mw::RegisterOperand{1},
                                      mw::ShortFloatImmediate{0}}),
                     exit_at(16)})
                 .program),
             gx::Error::Code::InvalidInstruction, 8);
    // Odd texture destinations are rejected before the skipped sample.
    rejected(gx::lower_forward(ForwardFixture(
                 1, {instruction(8, mw::OpcodeFamily::TextureSample, never,
                                 mw::TextureSample2D{
                                     mw::RegisterOperand{3},
                                     mw::RegisterOperand{4},
                                     mw::RegisterOperand{5},
                                     mw::RegisterOperand{6}}),
                     exit_at(16)})
                 .program),
             gx::Error::Code::UnsupportedModifier, 8);
}

ARMIGO_TEST(forward_lower, false_guard_valid_metadata_skips_without_reads) {
    // A metadata-valid instruction under !PT is skipped without reading its
    // undefined sources: operand definedness stays a property of executed
    // paths only.
    const ForwardFixture fixture(
        1, {fadd_reg(8, 2, 250, 251, pred(7, true)), mov32i(16, 0, 7u),
            exit_at(24)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 7u);
    }
}

ARMIGO_TEST(forward_lower, malformed_metadata_is_rejected) {
    const auto check = [](gx::LowerResult result, gx::Error::Code code) {
        const auto* error = error_of(result);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            if (error->code != code) {
                std::cout << error_text(*error) << "\n";
            }
            ARMIGO_CHECK(error->code == code);
        }
    };
    // Misaligned slot.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 32,
                                   {mov32i(12, 0, 1u), exit_at(16)})),
          gx::Error::Code::InvalidInstruction);
    // Unsorted pcs.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 64,
                                   {mov32i(40, 0, 1u), mov32i(8, 0, 2u),
                                    exit_at(48)})),
          gx::Error::Code::InvalidInstruction);
    // Missing branch target (contained, aligned, but absent from the list).
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 64,
                                   {branch_at(8, 56, pred(0)),
                                    mov32i(40, 0, 1u), exit_at(48)})),
          gx::Error::Code::InvalidInstruction);
    // Backward branch target.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 64,
                                   {mov32i(8, 0, 1u), branch_at(40, 8, pred(0)),
                                    exit_at(48)})),
          gx::Error::Code::InvalidInstruction);
    // Extra node outside the structural closure.
    check(gx::lower_forward(forged(Stage::Fragment, fragment_sph(1), 64,
                                   {mov32i(8, 0, 1u), exit_at(16),
                                    mov32i(40, 0, 2u)})),
          gx::Error::Code::InvalidInstruction);
    // Guard index outside the predicate file.
    check(gx::lower_forward(
              forged(Stage::Fragment, fragment_sph(1), 32,
                     {mov32i(8, 0, 1u, mw::Predicate{9, false}), exit_at(16)})),
          gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(forward_lower, runtime_guard_chain_has_linear_graph_and_msl) {
    constexpr unsigned rounds = 64;
    const gx::LowerResult result =
        gx::lower_forward(guarded_write_chain(rounds));
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;

    const auto check = [&](PredicatePattern pattern, std::uint32_t expected) {
        const std::vector<std::uint32_t> words =
            predicate_words(rounds, pattern);
        gx::Inputs inputs;
        inputs.constant_buffers[0] = words;
        ARMIGO_CHECK_EQ(color_output(*program, inputs), expected);
    };
    check(PredicatePattern::AllTrue, rounds);
    check(PredicatePattern::AllFalse, 0u);
    check(PredicatePattern::Alternating, rounds - 1);
    ARMIGO_CHECK(program->nodes().size() <= rounds * 16 + 32);
    ARMIGO_CHECK(program->blocks().size() <= rounds * 4 + 4);

    gx_msl::EmitOptions options;
    options.fragment_position =
        gx_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    const gx_msl::EmitResult emitted = gx_msl::emit(*program, options);
    const auto* stage = std::get_if<gx_msl::EmittedStage>(&emitted);
    if (const auto* error = std::get_if<gx_msl::EmitError>(&emitted)) {
        std::cout << error->message() << " (code "
                  << static_cast<int>(error->code) << " node " << error->node
                  << " detail " << error->detail << ")\n";
    }
    ARMIGO_CHECK(stage != nullptr);
    if (stage != nullptr) {
        ARMIGO_CHECK(stage->source.size() <= rounds * 4096 + 16384);
    }
}

ARMIGO_TEST(forward_lower, explicit_diamond_chain_has_linear_graph_and_msl) {
    constexpr unsigned diamonds = 32;
    const gx::LowerResult result =
        gx::lower_forward(branch_diamond_chain(diamonds));
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;

    const auto check = [&](PredicatePattern pattern, std::uint32_t expected) {
        const std::vector<std::uint32_t> words =
            predicate_words(diamonds, pattern);
        gx::Inputs inputs;
        inputs.constant_buffers[0] = words;
        ARMIGO_CHECK_EQ(color_output(*program, inputs), expected);
    };
    check(PredicatePattern::AllTrue, diamonds * 2);
    check(PredicatePattern::AllFalse, diamonds * 2 - 1);
    check(PredicatePattern::Alternating, diamonds * 2 - 1);
    ARMIGO_CHECK(program->nodes().size() <= diamonds * 16 + 32);
    ARMIGO_CHECK(program->blocks().size() <= diamonds * 4 + 4);

    gx_msl::EmitOptions options;
    options.fragment_position =
        gx_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    const gx_msl::EmitResult emitted = gx_msl::emit(*program, options);
    const auto* stage = std::get_if<gx_msl::EmittedStage>(&emitted);
    if (const auto* error = std::get_if<gx_msl::EmitError>(&emitted)) {
        std::cout << error->message() << " (code "
                  << static_cast<int>(error->code) << " node " << error->node
                  << " detail " << error->detail << ")\n";
    }
    ARMIGO_CHECK(stage != nullptr);
    if (stage != nullptr) {
        ARMIGO_CHECK(stage->source.size() <= diamonds * 4096 + 16384);
    }
}

ARMIGO_TEST(forward_lower, nested_joins_preserve_each_path_value) {
    std::vector<mw::ForwardInstruction> nodes;
    std::uint32_t pc = 8;
    const auto append = [&](mw::ForwardInstruction node) {
        nodes.push_back(std::move(node));
        advance_pc(pc);
    };
    append(mov32i(pc, 0, 0u));
    append(mov32i(pc, 3, 0u));
    append(mov_constant(pc, 16, 0, 0));
    append(mov_constant(pc, 17, 0, 4));
    append(isetp(pc, 0, 7, 16, 255));
    append(isetp(pc, 1, 7, 17, 255));

    const std::size_t outer_condition = nodes.size();
    append(branch_at(pc, 0, pred(0)));
    append(mov32i(pc, 0, 0x3f800000u));
    const std::size_t skip_outer_true = nodes.size();
    append(branch_at(pc, 0, pt()));

    const std::uint32_t outer_true_pc = pc;
    const std::size_t inner_condition = nodes.size();
    append(branch_at(pc, 0, pred(1)));
    append(mov32i(pc, 0, 0x40000000u));
    const std::size_t skip_inner_true = nodes.size();
    append(branch_at(pc, 0, pt()));

    const std::uint32_t inner_true_pc = pc;
    append(mov32i(pc, 0, 0x40400000u));
    const std::uint32_t inner_join_pc = pc;
    append(fadd_reg(pc, 0, 0, 3));
    const std::uint32_t outer_join_pc = pc;
    append(exit_at(pc));

    std::get<mw::ForwardBranch>(nodes[outer_condition]).target_pc =
        outer_true_pc;
    std::get<mw::ForwardBranch>(nodes[skip_outer_true]).target_pc =
        outer_join_pc;
    std::get<mw::ForwardBranch>(nodes[inner_condition]).target_pc =
        inner_true_pc;
    std::get<mw::ForwardBranch>(nodes[skip_inner_true]).target_pc =
        inner_join_pc;
    const mw::ForwardProgram input =
        forged(Stage::Fragment, fragment_sph(1), program_size_for(pc),
               std::move(nodes));
    const gx::LowerResult result = gx::lower_forward(input);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;

    const auto check = [&](std::array<std::uint32_t, 2> words,
                           std::uint32_t expected) {
        gx::Inputs inputs;
        inputs.constant_buffers[0] = words;
        ARMIGO_CHECK_EQ(color_output(*program, inputs), expected);
    };
    check({0u, 0u}, 0x3f800000u);
    check({1u, 0u}, 0x40000000u);
    check({1u, 1u}, 0x40400000u);
}

ARMIGO_TEST(forward_lower, conditional_predicate_is_valid_in_same_scope) {
    const ForwardFixture fixture(
        4, {mov32i(8, 0, 0u), mov_constant(16, 16, 0, 0),
            isetp(24, 0, 7, 16, 255),
            isetp(40, 1, 7, 16, 255, pred(0)),
            branch_at(48, 72, pred(0)), branch_at(56, 88, pt()),
            mov32i(72, 0, 9u, pred(1)), branch_at(80, 88, pt()),
            exit_at(88)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    bool has_partial_phi = false;
    for (const gx::Node& node : program->nodes()) {
        has_partial_phi = has_partial_phi ||
                          std::holds_alternative<gx::PredicatePhi>(
                              node.operation);
    }
    ARMIGO_CHECK(has_partial_phi);

    const auto check = [&](std::uint32_t word, std::uint32_t expected) {
        const std::array words{word};
        gx::Inputs inputs;
        inputs.constant_buffers[0] = words;
        ARMIGO_CHECK_EQ(color_output(*program, inputs), expected);
    };
    check(0u, 0u);
    check(1u, 9u);
}

ARMIGO_TEST(forward_lower, conditional_predicate_is_invalid_after_join) {
    const ForwardFixture fixture(
        2, {mov32i(8, 0, 0u), mov_constant(16, 16, 0, 0),
            isetp(24, 0, 7, 16, 255),
            isetp(40, 1, 7, 16, 255, pred(0)),
            mov32i(48, 0, 9u, pred(1)), exit_at(56)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UninitializedRegister);
        ARMIGO_CHECK_EQ(error->detail, std::uint32_t{1});
    }
}

ARMIGO_TEST(forward_lower, predicate_phi_selects_both_definitions) {
    const ForwardFixture fixture(
        4, {mov32i(8, 0, 7u), mov32i(16, 18, 0u),
            mov32i(24, 19, 1u), mov_constant(40, 16, 0, 0),
            isetp(48, 0, 7, 16, 255), branch_at(56, 88, pred(0)),
            isetp(72, 1, 7, 18, 255), branch_at(80, 104, pt()),
            isetp(88, 1, 7, 19, 255), mov32i(104, 0, 9u, pred(1)),
            exit_at(112)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    bool has_phi = false;
    for (const gx::Node& node : program->nodes()) {
        has_phi = has_phi ||
                  std::holds_alternative<gx::PredicatePhi>(node.operation);
    }
    ARMIGO_CHECK(has_phi);

    const auto check = [&](std::uint32_t word, std::uint32_t expected) {
        const std::array words{word};
        gx::Inputs inputs;
        inputs.constant_buffers[0] = words;
        ARMIGO_CHECK_EQ(color_output(*program, inputs), expected);
    };
    check(0u, 7u);
    check(1u, 9u);
}

ARMIGO_TEST(forward_lower, partial_vertex_output_after_join_is_rejected) {
    const mw::ForwardProgram program = forged(
        Stage::Vertex, vertex_sph_position_output(),
        2 * si::format::scheduling_group_size,
        {mov32i(8, 0, 0x3f800000u), mov_constant(16, 16, 0, 0),
         isetp(24, 0, 7, 16, 255),
         attribute_store(40, 0, 28, pred(0)), exit_at(48)});
    const gx::LowerResult result = gx::lower_forward(program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::MissingOutput);
    }
}

ARMIGO_TEST(forward_lower, branch_with_same_target_has_one_continuation) {
    const ForwardFixture fixture(
        3, {mov32i(8, 0, 3u), mov32i(16, 16, 0x3f800000u),
            fadd_reg(24, 17, 16, 16), isetp(40, 0, 7, 17, 255),
            branch_at(48, 56, pred(0)), mov32i(56, 0, 9u), exit_at(72)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 9u);
        ARMIGO_CHECK(program->blocks().size() <= 8);
    }
}

ARMIGO_TEST(forward_lower, paths_without_program_join_merge_at_exit) {
    ForwardFixture fixture(
        3, {mov32i(8, 0, 5u), mov32i(16, 16, 0x3f800000u),
            fadd_reg(24, 17, 16, 16), isetp(40, 0, 7, 17, 255),
            branch_at(48, 80, pred(0)), mov32i(56, 0, 11u), exit_at(72),
            mov32i(80, 0, 22u), exit_at(88)});
    const gx::LowerResult taken_result = gx::lower_forward(fixture.program);
    const auto* taken = program_of(taken_result);
    ARMIGO_CHECK(taken != nullptr);
    if (taken != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*taken), 22u);
    }

    fixture.program.instructions[1] = mov32i(16, 16, 0u);
    const gx::LowerResult skipped_result = gx::lower_forward(fixture.program);
    const auto* skipped = program_of(skipped_result);
    ARMIGO_CHECK(skipped != nullptr);
    if (skipped != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*skipped), 11u);
    }
}

ARMIGO_TEST(forward_lower, partial_initialization_after_join_is_rejected) {
    const ForwardFixture fixture(
        3, {mov32i(8, 0, 3u), mov32i(16, 16, 0x3f800000u),
            fadd_reg(24, 17, 16, 16), isetp(40, 0, 7, 17, 255),
            mov32i(48, 5, 0x40000000u, pred(0)),
            fadd_reg(56, 0, 5, 5), exit_at(72)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UninitializedRegister);
        ARMIGO_CHECK_EQ(error->detail, std::uint32_t{5});
    }
}

ARMIGO_TEST(forward_lower, skipped_guarded_load_does_not_access_resource) {
    ForwardFixture fixture(
        2, {mov32i(8, 0, 7u), mov32i(16, 16, 0u),
            fadd_reg(24, 17, 16, 16), isetp(40, 0, 7, 17, 255),
            mov_constant(48, 0, 5, 0, pred(0)), exit_at(56)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 7u);
    }

    fixture.program.instructions[1] = mov32i(16, 16, 0x3f800000u);
    const gx::LowerResult taken_result = gx::lower_forward(fixture.program);
    const auto* taken = program_of(taken_result);
    ARMIGO_CHECK(taken != nullptr);
    if (taken != nullptr) {
        const gx::EvaluateResult evaluation = gx::evaluate(*taken, gx::Inputs{});
        const auto* error = std::get_if<gx::Error>(&evaluation);
        ARMIGO_CHECK(error != nullptr);
        if (error != nullptr) {
            ARMIGO_CHECK(error->code == gx::Error::Code::ConstantBounds);
        }
    }
}

ARMIGO_TEST(forward_lower, predicate_redefinition_after_join_is_fresh) {
    const ForwardFixture fixture(
        5, {mov32i(8, 0, 5u), mov32i(16, 16, 0x3f800000u),
            fadd_reg(24, 17, 16, 16), isetp(40, 0, 7, 17, 255),
            branch_at(48, 80, pred(0)), mov32i(56, 0, 11u),
            branch_at(72, 88, pt()), mov32i(80, 0, 22u),
            mov32i(88, 18, 0u), fadd_reg(104, 19, 18, 18),
            isetp(112, 0, 7, 19, 255), mov32i(120, 0, 99u, pred(0)),
            exit_at(136)});
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program != nullptr) {
        ARMIGO_CHECK_EQ(color_output(*program), 22u);
    }
}

ARMIGO_TEST(forward_lower, block_cap_rejects_large_explicit_cfg) {
    constexpr unsigned diamonds =
        static_cast<unsigned>(gx::max_blocks / 2 + 1);
    const gx::LowerResult result =
        gx::lower_forward(branch_diamond_chain(diamonds));
    const auto* error = error_of(result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::Limit);
    }
}

ARMIGO_TEST(forward_lower_msl, emits_compilable_metal_source) {
    // Fragment MSL emission requires every lane of a color target, so the
    // fixture declares and defines R0..R3 and lets the fork choose R0.
    struct MslFixture {
        mw::ForwardProgram program;
        MslFixture() {
            program = forged(Stage::Fragment, fragment_sph(0xf),
                             4 * si::format::scheduling_group_size,
                             {mov32i(8, 0, 5u),
                              mov32i(16, 1, 0x3f800000u),
                              mov32i(24, 3, 6u),
                              fadd_reg(40, 2, 1, 1),
                              isetp(48, 0, 7, 2, 255),
                              branch_at(56, 88, pred(0)),
                              mov32i(72, 0, 1u),
                              exit_at(80, pred(0, true)),
                              mov32i(88, 0, 2u),
                              exit_at(104)});
        }
    };
    const MslFixture fixture;
    const gx::LowerResult result = gx::lower_forward(fixture.program);
    const auto* program = program_of(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    gx_msl::EmitOptions options;
    options.fragment_position =
        gx_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    const gx_msl::EmitResult emitted = gx_msl::emit(*program, options);
    const auto* stage = std::get_if<gx_msl::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) return;
    ARMIGO_CHECK(!stage->source.empty());
    write_msl_fixture("forward_lower_graph", stage->source);
}

ARMIGO_TEST(forward_lower_msl, emits_scaling_metal_sources) {
    const std::array fixtures{
        std::pair{"forward_runtime_guards", guarded_write_chain(64)},
        std::pair{"forward_branch_diamonds", branch_diamond_chain(32)},
    };
    for (const auto& [name, input] : fixtures) {
        const gx::LowerResult result = gx::lower_forward(input);
        const auto* program = program_of(result);
        ARMIGO_CHECK(program != nullptr);
        if (program == nullptr) continue;
        gx_msl::EmitOptions options;
        options.fragment_position = gx_msl::FragmentPositionConvention::
            MetalPositionWMatchesGuestPositionW;
        const gx_msl::EmitResult emitted = gx_msl::emit(*program, options);
        const auto* stage = std::get_if<gx_msl::EmittedStage>(&emitted);
        ARMIGO_CHECK(stage != nullptr);
        if (stage == nullptr) continue;
        ARMIGO_CHECK(!stage->source.empty());
        write_msl_fixture(name, stage->source);
    }
}

}  // namespace
}  // namespace armigo::shader::test
