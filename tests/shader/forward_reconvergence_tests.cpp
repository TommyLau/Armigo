// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics_forward.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell_forward_program.h>

#include "forward_reconvergence.h"
#include "test_support.h"

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace mw = maxwell;
namespace gx = graphics;
namespace si = shader_input;

constexpr std::uint32_t pc(std::size_t index) {
    return static_cast<std::uint32_t>(8 + (index / 3) * 32 + (index % 3) * 8);
}
constexpr std::uint64_t word(std::uint16_t opcode, unsigned guard = 7) {
    return (std::uint64_t{opcode} << 48) | (std::uint64_t{guard} << 16);
}
constexpr std::uint64_t mov(unsigned reg, std::uint32_t bits) {
    return word(0x0100) | reg | (15ull << 12) | (std::uint64_t{bits} << 20);
}
constexpr std::uint64_t exit_word() { return word(0xe300) | 15; }
constexpr std::uint64_t sync(unsigned guard = 7) { return word(0xf0f8, guard) | 15; }
constexpr std::uint64_t relative(std::uint16_t opcode, std::size_t source,
                                 std::int64_t target, unsigned guard) {
    const auto displacement = static_cast<std::uint64_t>(target - pc(source) - 8);
    return word(opcode, guard) | ((displacement & 0xffffff) << 20);
}
constexpr std::uint64_t ssy(std::size_t source, std::size_t target) {
    return relative(0xe290, source, pc(target), 0);
}
constexpr std::uint64_t bra(std::size_t source, std::size_t target, unsigned guard = 7) {
    return relative(0xe240, source, pc(target), guard) | 15;
}
constexpr std::uint64_t constant(unsigned reg, unsigned byte_offset) {
    return word(0x4c98) | reg | (15ull << 39) | (std::uint64_t{byte_offset / 4} << 20);
}
constexpr std::uint64_t predicate(unsigned pred, unsigned reg) {
    return word(0x5b60) | 7 | (std::uint64_t{pred} << 3) |
           (std::uint64_t{reg} << 8) | (255ull << 20) | (7ull << 39) | (5ull << 49);
}

// Both predicate inputs are runtime constant-buffer words; R0 is the color.
std::vector<std::uint64_t> prefix() {
    return {constant(1, 0), predicate(0, 1), constant(2, 4), predicate(1, 2), mov(0, 10)};
}

struct Container {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};
Container container(const std::vector<std::uint64_t>& words) {
    const auto size = static_cast<std::uint32_t>(((words.size() + 2) / 3) * 32);
    Container data{std::vector<std::uint8_t>(0x718), std::vector<std::uint8_t>(0x80 + size)};
    const auto put32 = [](auto& bytes, std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte < 4; ++byte) bytes[offset + byte] = value >> (byte * 8);
    };
    put32(data.control, 0, si::format::control_magic);
    put32(data.control, 4, si::format::control_tag);
    put32(data.control, si::format::control_version_offset, 14);
    put32(data.control, si::format::control_size_offset, data.control.size());
    put32(data.control, si::format::c1_offset_field, data.code.size());
    put32(data.control, si::format::code_size_field, data.code.size());
    put32(data.control, si::format::code_offset_field, 0x30);
    put32(data.control, si::format::stage_field, static_cast<std::uint32_t>(si::Stage::Fragment));
    put32(data.code, 0, 0x12345678);
    put32(data.code, 0x30, si::sph_stage_type_for(si::Stage::Fragment) << si::format::sph_stage_shift);
    put32(data.code, 0x30 + 0x48, 1);
    for (std::size_t index = 0; index < words.size(); ++index) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            data.code[0x80 + pc(index) + byte] = words[index] >> (byte * 8);
        }
    }
    return data;
}
si::ByteSpan bytes(const std::vector<std::uint8_t>& data) { return {data.data(), data.size()}; }
mw::ForwardDecodeResult decode(const std::vector<std::uint64_t>& words,
                                mw::DecodeLimits limits = {}) {
    const auto data = container(words);
    return mw::decode_forward_program(bytes(data.control), bytes(data.code), limits);
}

void check_error(const std::vector<std::uint64_t>& words, mw::DecodeField field,
                 std::uint32_t source) {
    const auto result = decode(words);
    const auto* error = std::get_if<mw::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (!error) return;
    ARMIGO_CHECK(error->code == mw::DecodeError::Code::UnsupportedControlFlow);
    ARMIGO_CHECK(error->field == field);
    ARMIGO_CHECK_EQ(error->pc, source);
}

graphics_msl::EmitResult emit(const gx::Program& program) {
    graphics_msl::EmitOptions options;
    options.fragment_position =
        graphics_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    return graphics_msl::emit(program, options);
}

void check_program(const std::vector<std::uint64_t>& words,
                   const std::array<std::uint32_t, 4>& expected,
                   const std::vector<std::pair<std::size_t, std::size_t>>& branches = {}) {
    const auto decoded = decode(words);
    const auto* program = std::get_if<mw::ForwardProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    for (const auto [source, target] : branches) {
        const mw::ForwardBranch* found = nullptr;
        for (const auto& node : program->instructions) {
            const auto* branch = std::get_if<mw::ForwardBranch>(&node);
            if (branch && branch->source_pc == pc(source)) found = branch;
        }
        ARMIGO_CHECK(found != nullptr);
        if (found) {
            ARMIGO_CHECK_EQ(found->target_pc, pc(target));
            ARMIGO_CHECK_EQ(found->predicate.index, (words[source] >> 16) & 7);
            ARMIGO_CHECK(found->predicate.inverted == (((words[source] >> 19) & 1) != 0));
        }
    }
    const auto lowered = gx::lower_forward(*program);
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (!graph) return;
    for (unsigned outcome = 0; outcome < 4; ++outcome) {
        const std::array<std::uint32_t, 2> conditions{outcome & 1u, (outcome >> 1) & 1u};
        gx::Inputs inputs;
        inputs.constant_buffers[0] = conditions;
        const auto result = gx::evaluate(*graph, inputs);
        const auto* evaluation = std::get_if<gx::Evaluation>(&result);
        ARMIGO_CHECK(evaluation != nullptr && evaluation->outputs.size() == 1);
        if (evaluation && evaluation->outputs.size() == 1) {
            ARMIGO_CHECK_EQ(evaluation->outputs.front().bits, expected[outcome]);
        }
    }
}
}  // namespace

ARMIGO_TEST(forward_reconvergence, diamond) {
    auto words = prefix();
    words.insert(words.end(), {ssy(5, 11), bra(6, 9, 0), mov(0, 20), sync(),
                               mov(0, 30), sync(), exit_word(), bra(12, 12)});
    check_program(words, {20, 30, 20, 30}, {{8, 11}, {10, 11}});
}

ARMIGO_TEST(forward_reconvergence, nested_and_conditional_stacks) {
    auto words = prefix();
    // Inner SYNC consumes only the inner token. Outer SYNC can still resolve.
    words.insert(words.end(), {ssy(5, 14), ssy(6, 11), sync(0), mov(0, 20), sync(),
                               0, sync(1), mov(0, 30), sync(), exit_word()});
    check_program(words, {30, 30, 20, 10}, {{7, 11}, {9, 11}, {11, 14}, {13, 14}});
}

ARMIGO_TEST(forward_reconvergence, conditional_sync_retains_fallthrough_token) {
    auto words = prefix();
    words.insert(words.end(), {ssy(5, 9), sync(0), mov(0, 20), sync(), exit_word()});
    check_program(words, {20, 10, 20, 10}, {{6, 9}, {8, 9}});
    words[6] = sync(8);
    check_program(words, {10, 20, 10, 20}, {{6, 9}, {8, 9}});
    words[6] = sync(15);
    check_program(words, {20, 20, 20, 20}, {{6, 9}, {8, 9}});
    words[6] = sync();
    check_program(words, {10, 10, 10, 10}, {{6, 9}});
}

ARMIGO_TEST(forward_reconvergence, missing_token_and_incompatible_joins) {
    check_error({sync(), exit_word()}, mw::DecodeField::ReconvergenceStack, pc(0));
    check_error({sync(15), exit_word()}, mw::DecodeField::ReconvergenceStack, pc(0));
    check_error({bra(0, 2, 0), ssy(1, 3), sync(), exit_word()},
                mw::DecodeField::ReconvergenceStack, pc(2));
    // Equal top targets do not make different complete stacks compatible.
    check_error({ssy(0, 8), bra(1, 5, 0), ssy(2, 7), ssy(3, 8), bra(4, 6),
                 ssy(5, 8), sync(), sync(), exit_word()},
                mw::DecodeField::ReconvergenceStack, pc(6));
    // Both edges name the same slot, but SYNC pops only on the taken edge.
    check_error({ssy(0, 2), sync(0), exit_word()},
                mw::DecodeField::ReconvergenceStack, pc(2));
}

ARMIGO_TEST(forward_reconvergence, equivalent_stacks_and_scheduling_aliases) {
    auto words = prefix();
    // Different pushes of equal complete stacks must intern identically.
    words.insert(words.end(), {bra(5, 8, 0), ssy(6, 12), bra(7, 9),
                               relative(0xe290, 8, pc(12) - 8, 0),
                               mov(0, 20), sync(), 0, exit_word()});
    check_program(words, {20, 20, 20, 20}, {{10, 12}});
}

ARMIGO_TEST(forward_reconvergence, rejects_invalid_targets_and_unconsumed_target) {
    check_error({relative(0xe290, 0, 0, 0), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(0));
    check_error({ssy(0, 1), mov(0, 10), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(2));
    check_error({ssy(0, 1), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(1));
    check_error({relative(0xe290, 0, -8, 0), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(0));
    check_error({relative(0xe290, 0, 26, 0), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(0));
    check_error({relative(0xe290, 0, 32, 0), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(0));
    check_error({relative(0xe290, 0, 0x7ffff8, 0), sync(), exit_word()},
                mw::DecodeField::ReconvergenceTarget, pc(0));
}

ARMIGO_TEST(forward_reconvergence, rejects_unsupported_forms) {
    for (const auto bit : {0, 6, 16, 19, 44, 48, 51}) {
        check_error({ssy(0, 2) | (1ull << bit), sync(), exit_word()},
                    mw::DecodeField::ReservedBits, pc(0));
    }
    check_error({ssy(0, 2) | (1ull << 5), sync(), exit_word()},
                mw::DecodeField::AddressMode, pc(0));
    for (unsigned flow = 0; flow < 32; ++flow) {
        if (flow == 15) continue;
        check_error({ssy(0, 2), word(0xf0f8) | flow, exit_word()},
                    mw::DecodeField::FlowTest, pc(1));
    }
    check_error({ssy(0, 2), sync() | (1ull << 5), exit_word()},
                mw::DecodeField::AddressMode, pc(1));
    for (const auto bit : {6, 20, 44, 48, 50}) {
        check_error({ssy(0, 2), sync() | (1ull << bit), exit_word()},
                    mw::DecodeField::ReservedBits, pc(1));
    }
    for (const auto opcode : {0xe2a0, 0xe340}) {
        check_error({word(opcode), exit_word()}, mw::DecodeField::None, pc(0));
    }
}

ARMIGO_TEST(forward_reconvergence, reachability_limits_and_marker_validation) {
    auto words = prefix();
    words.insert(words.end(), {ssy(5, 7), sync(), exit_word(), bra(8, 8)});
    check_program(words, {10, 10, 10, 10}, {{6, 7}});
    const auto limited = decode(words, mw::DecodeLimits{7});
    const auto* limit = std::get_if<mw::DecodeError>(&limited);
    ARMIGO_CHECK(limit && limit->code == mw::DecodeError::Code::InstructionLimit);
    check_error({ssy(0, 3), bra(1, 1), sync(), exit_word()}, mw::DecodeField::None, pc(1));
    const auto missing_exit = decode({ssy(0, 1), mov(0, 10), mov(0, 20)});
    const auto* error = std::get_if<mw::DecodeError>(&missing_exit);
    ARMIGO_CHECK(error && error->code == mw::DecodeError::Code::MissingExit);
    auto decoded = decode(words);
    auto* program = std::get_if<mw::ForwardProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    auto& marker = std::get<mw::Instruction>(program->instructions[5]);
    ARMIGO_CHECK(marker.family == mw::OpcodeFamily::ReconvergenceSetup);
    ARMIGO_CHECK(marker.original_word == words[5]);
    marker.payload = mw::Exit{};
    ARMIGO_CHECK(std::holds_alternative<gx::Error>(gx::lower_forward(*program)));
}

ARMIGO_TEST(forward_reconvergence, ordered_token_stack_unwinds_younger_kinds) {
    // PBK is reserved internally only; no guest PBK/BRK word is admitted.
    mw::detail::ReconvergenceStacks stacks;
    const auto outer = stacks.push(0, mw::detail::TokenKind::Ssy, 160);
    const auto younger = stacks.push(outer, mw::detail::TokenKind::Pbk, 120);
    const auto popped = stacks.pop(younger, mw::detail::TokenKind::Ssy);
    ARMIGO_CHECK(popped && popped->remaining == 0 && popped->target_pc == 160);
    ARMIGO_CHECK(stacks.push(0, mw::detail::TokenKind::Ssy, 160) == outer);
}

ARMIGO_TEST(forward_reconvergence, resolved_sync_emits_full_color_msl) {
    const std::vector<std::uint64_t> words{ssy(0, 4), mov(0, 0x3f800000), sync(), 0,
                                           mov(1, 0), mov(2, 0), mov(3, 0), exit_word()};
    auto data = container(words);
    data.code[0x30 + 0x48] = 15;
    const auto decoded = mw::decode_forward_program(bytes(data.control), bytes(data.code));
    const auto* program = std::get_if<mw::ForwardProgram>(&decoded);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    const auto lowered = gx::lower_forward(*program);
    const auto* graph = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(graph != nullptr);
    if (!graph) return;
    ARMIGO_CHECK(std::holds_alternative<graphics_msl::EmittedStage>(emit(*graph)));
    const auto evaluated = gx::evaluate(*graph, gx::Inputs{});
    const auto* output = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(output && output->outputs.size() == 4);
    if (output && output->outputs.size() == 4) {
        ARMIGO_CHECK_EQ(output->outputs[0].bits, 0x3f800000u);
        for (unsigned i = 1; i < 4; ++i) ARMIGO_CHECK_EQ(output->outputs[i].bits, 0u);
    }
}

ARMIGO_TEST(forward_reconvergence, unchanged_direct_branch_decode_and_msl) {
    auto words = prefix();
    words.insert(words.end(), {bra(5, 8, 0), mov(0, 20), bra(7, 9), mov(0, 30),
                               mov(3, 0), exit_word(), bra(11, 11)});
    check_program(words, {20, 30, 20, 30}, {{5, 8}, {7, 9}});
    auto data = container(words);
    data.code[0x30 + 0x48] = 15;
    const auto decoded = mw::decode_forward_program(bytes(data.control), bytes(data.code));
    const auto* actual = std::get_if<mw::ForwardProgram>(&decoded);
    ARMIGO_CHECK(actual != nullptr);
    if (!actual) return;

    // The legacy strict decoder supplies unchanged ordinary payloads; branch
    // metadata and reachable slot ordering are specified independently here.
    auto straight_words = prefix();
    straight_words.push_back(exit_word());
    const auto straight_data = container(straight_words);
    const auto strict = mw::decode(bytes(straight_data.control), bytes(straight_data.code));
    const auto* ordinary = std::get_if<mw::DecodedProgram>(&strict);
    ARMIGO_CHECK(ordinary != nullptr);
    if (!ordinary) return;
    mw::ForwardProgram expected{actual->stage, actual->sph, 128, {}};
    for (unsigned i = 0; i < 5; ++i) expected.instructions.emplace_back(ordinary->instructions[i]);
    expected.instructions.emplace_back(mw::ForwardBranch{pc(5), pc(8), {0, false}});
    const auto append_mov = [&](std::size_t index, std::uint8_t reg, std::uint32_t bits) {
        expected.instructions.emplace_back(mw::Instruction{
            pc(index), words[index], mw::OpcodeFamily::Mov32Immediate, {7, false},
            mw::Mov32Immediate{{reg}, 15, bits}});
    };
    append_mov(6, 0, 20);
    expected.instructions.emplace_back(mw::ForwardBranch{pc(7), pc(9), {7, false}});
    append_mov(8, 0, 30);
    append_mov(9, 3, 0);
    expected.instructions.emplace_back(mw::Instruction{
        pc(10), words[10], mw::OpcodeFamily::Exit, {7, false}, mw::Exit{15}});
    ARMIGO_CHECK_EQ(actual->program_size, expected.program_size);
    ARMIGO_CHECK_EQ(actual->instructions.size(), expected.instructions.size());
    if (actual->instructions.size() != expected.instructions.size()) return;
    for (std::size_t i = 0; i < expected.instructions.size(); ++i) {
        const auto& lhs = actual->instructions[i];
        const auto& rhs = expected.instructions[i];
        ARMIGO_CHECK(lhs.index() == rhs.index());
        if (lhs.index() != rhs.index()) continue;
        if (const auto* branch = std::get_if<mw::ForwardBranch>(&lhs)) {
            const auto& other = std::get<mw::ForwardBranch>(rhs);
            ARMIGO_CHECK(branch->source_pc == other.source_pc && branch->target_pc == other.target_pc);
            ARMIGO_CHECK(branch->predicate.index == other.predicate.index &&
                         branch->predicate.inverted == other.predicate.inverted);
        } else {
            const auto& current = std::get<mw::Instruction>(lhs);
            const auto& other = std::get<mw::Instruction>(rhs);
            ARMIGO_CHECK(current.pc == other.pc && current.original_word == other.original_word &&
                         current.family == other.family && current.payload.index() == other.payload.index());
            ARMIGO_CHECK(current.predicate.index == other.predicate.index &&
                         current.predicate.inverted == other.predicate.inverted);
        }
    }
    const auto expected_lower = gx::lower_forward(expected);
    const auto actual_lower = gx::lower_forward(*actual);
    const auto* expected_graph = std::get_if<gx::Program>(&expected_lower);
    const auto* actual_graph = std::get_if<gx::Program>(&actual_lower);
    ARMIGO_CHECK(expected_graph && actual_graph);
    if (!expected_graph || !actual_graph) return;
    const auto expected_msl = emit(*expected_graph);
    const auto actual_msl = emit(*actual_graph);
    const auto* lhs = std::get_if<graphics_msl::EmittedStage>(&expected_msl);
    const auto* rhs = std::get_if<graphics_msl::EmittedStage>(&actual_msl);
    ARMIGO_CHECK(lhs && rhs);
    if (lhs && rhs) ARMIGO_CHECK_EQ(lhs->source, rhs->source);
}

ARMIGO_TEST(forward_reconvergence, unchanged_straight_line_decode_and_msl) {
    const std::vector<std::uint64_t> words{
        mov(0, 0x3f800000), mov(1, 0), mov(2, 0), mov(3, 0), exit_word(), bra(5, 5)};
    auto data = container(words);
    data.code[0x30 + 0x48] = 15;
    const auto strict = mw::decode(bytes(data.control), bytes(data.code));
    const auto decoded = mw::decode_forward_program(bytes(data.control), bytes(data.code));
    const auto* before = std::get_if<mw::DecodedProgram>(&strict);
    const auto* after = std::get_if<mw::ForwardProgram>(&decoded);
    ARMIGO_CHECK(before && after);
    if (!before || !after) return;
    ARMIGO_CHECK(before->stage == after->stage && before->sph == after->sph);
    ARMIGO_CHECK_EQ(before->instructions.size(), after->instructions.size());
    mw::ForwardProgram expected{before->stage, before->sph, 64, {}};
    for (std::size_t index = 0; index < before->instructions.size(); ++index) {
        const auto& old = before->instructions[index];
        const auto& now = std::get<mw::Instruction>(after->instructions[index]);
        ARMIGO_CHECK(old.pc == now.pc && old.original_word == now.original_word);
        ARMIGO_CHECK(old.family == now.family && old.payload.index() == now.payload.index());
        ARMIGO_CHECK(old.predicate.index == now.predicate.index &&
                     old.predicate.inverted == now.predicate.inverted);
        if (old.family == mw::OpcodeFamily::Mov32Immediate) {
            const auto& lhs = std::get<mw::Mov32Immediate>(old.payload);
            const auto& rhs = std::get<mw::Mov32Immediate>(now.payload);
            ARMIGO_CHECK(lhs.destination.index == rhs.destination.index &&
                         lhs.write_mask == rhs.write_mask && lhs.immediate_bits == rhs.immediate_bits);
        } else {
            ARMIGO_CHECK(std::get<mw::Exit>(old.payload).flow_test == std::get<mw::Exit>(now.payload).flow_test);
        }
        expected.instructions.emplace_back(old);
    }
    const auto expected_lower = gx::lower_forward(expected);
    const auto actual_lower = gx::lower_forward(*after);
    const auto* expected_graph = std::get_if<gx::Program>(&expected_lower);
    const auto* actual_graph = std::get_if<gx::Program>(&actual_lower);
    ARMIGO_CHECK(expected_graph && actual_graph);
    if (!expected_graph || !actual_graph) return;
    const auto expected_msl = emit(*expected_graph);
    const auto actual_msl = emit(*actual_graph);
    const auto* lhs = std::get_if<graphics_msl::EmittedStage>(&expected_msl);
    const auto* rhs = std::get_if<graphics_msl::EmittedStage>(&actual_msl);
    ARMIGO_CHECK(lhs && rhs);
    if (lhs && rhs) ARMIGO_CHECK_EQ(lhs->source, rhs->source);
    check_program(words, {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000});
    // The strict decoder must report reconvergence words as unsupported
    // control flow (not unknown opcodes) so translation falls back to the
    // forward decoder, which accepts this program.
    const auto setup_data = container({ssy(0, 2), sync(), exit_word()});
    const auto strict_setup = mw::decode(bytes(setup_data.control), bytes(setup_data.code));
    const auto* strict_error = std::get_if<mw::DecodeError>(&strict_setup);
    ARMIGO_CHECK(strict_error != nullptr &&
                 strict_error->code == mw::DecodeError::Code::UnsupportedControlFlow);
    ARMIGO_CHECK(std::holds_alternative<mw::ForwardProgram>(
        mw::decode_forward_program(bytes(setup_data.control), bytes(setup_data.code))));
    for (const std::uint16_t high : {std::uint16_t{0xe290}, std::uint16_t{0xf0f8},
                                     std::uint16_t{0xe2a0}, std::uint16_t{0xe340}}) {
        const auto data = container({(std::uint64_t{high} << 48) | (std::uint64_t{7} << 16) | 15,
                                     exit_word()});
        const auto result = mw::decode(bytes(data.control), bytes(data.code));
        const auto* error = std::get_if<mw::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr &&
                     error->code == mw::DecodeError::Code::UnsupportedControlFlow);
    }
}

}  // namespace armigo::shader::test
