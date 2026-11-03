// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/maxwell_flow.h>

#include "test_support.h"

#include <variant>

namespace armigo::shader::test {
namespace {

using armigo::shader::maxwell::ForwardBranch;
using armigo::shader::maxwell::ForwardBranchError;
using armigo::shader::maxwell::ForwardBranchErrorCode;
using armigo::shader::maxwell::ForwardBranchResult;
using armigo::shader::maxwell::decode_forward_branch;

// A 128-byte program region holds four scheduling groups; instruction slots
// sit at 8, 16, 24, 40, 48, 56, 72, 80, 88, 104, 112, and 120.
constexpr std::uint32_t kProgram128 = 128;

// Builds a direct relative branch word: opcode window in the top half, the
// always-true flow test, the raw 4-bit predicate guard, and the raw 24-bit
// displacement field.
constexpr std::uint64_t direct_branch_word(std::uint32_t displacement_field,
                                           std::uint8_t predicate_raw) {
    return (std::uint64_t{0xe240} << 48) | 15u |
           (std::uint64_t{predicate_raw} << 16) |
           (std::uint64_t{displacement_field} << 20);
}

constexpr std::uint64_t direct_branch_word(std::uint32_t displacement_field) {
    return direct_branch_word(displacement_field, 7);
}

// Renders a signed displacement as its raw 24-bit field.
constexpr std::uint32_t disp24(std::int32_t displacement) {
    return static_cast<std::uint32_t>(displacement) & 0xffffff;
}

const ForwardBranch* success(const ForwardBranchResult& result) {
    return std::get_if<ForwardBranch>(&result);
}

const ForwardBranchError* failure(const ForwardBranchResult& result) {
    return std::get_if<ForwardBranchError>(&result);
}

void expect_forward(const ForwardBranchResult& result, std::uint32_t source,
                    std::uint32_t target, std::uint8_t predicate_raw) {
    const ForwardBranch* branch = success(result);
    if (branch == nullptr) {
        ARMIGO_CHECK(false);
        return;
    }
    ARMIGO_CHECK_EQ(branch->source_pc, source);
    ARMIGO_CHECK_EQ(branch->target_pc, target);
    ARMIGO_CHECK_EQ(branch->predicate.index,
                    static_cast<std::uint8_t>(predicate_raw & 7));
    ARMIGO_CHECK_EQ(branch->predicate.inverted, predicate_raw >= 8);
}

void expect_error(const ForwardBranchResult& result,
                  ForwardBranchErrorCode code) {
    const ForwardBranchError* error = failure(result);
    if (error == nullptr) {
        ARMIGO_CHECK(false);
        return;
    }
    ARMIGO_CHECK_EQ(static_cast<unsigned>(error->code),
                    static_cast<unsigned>(code));
    ARMIGO_CHECK(error->message() != nullptr);
}

}  // namespace

ARMIGO_TEST(forward_branch, forward_from_first_slot) {
    // pc 8 + 8 + 40 = 56, a slot of the first group pair.
    expect_forward(decode_forward_branch(direct_branch_word(40), 8, kProgram128),
                   8, 56, 7);
}

ARMIGO_TEST(forward_branch, forward_across_groups) {
    // pc 24 + 8 + 16 = 48, a slot of the second group.
    expect_forward(
        decode_forward_branch(direct_branch_word(16), 24, kProgram128), 24, 48,
        7);
}

ARMIGO_TEST(forward_branch, max_positive_displacement_exact_fit) {
    // pc 8 + 8 + 8'388'600 = 8'388'616 (0x800008); the program ends exactly
    // after the target word at 8'388'624 (0x800010).
    constexpr std::uint32_t displacement = 0x7ffff8;
    constexpr std::uint32_t program_size = 0x800010;
    expect_forward(
        decode_forward_branch(direct_branch_word(displacement), 8, program_size),
        8, 0x800008, 7);
}

ARMIGO_TEST(forward_branch, target_is_last_word_of_exact_program) {
    // Two full groups: slots end at 56, and 56 + 8 equals the program size.
    constexpr std::uint32_t program_size = 64;
    expect_forward(
        decode_forward_branch(direct_branch_word(40), 8, program_size), 8, 56,
        7);
}

ARMIGO_TEST(forward_branch, scheduler_normalization) {
    // pc 40 + 8 + 16 = 64, a scheduling word; the target advances to 72.
    expect_forward(
        decode_forward_branch(direct_branch_word(16), 40, kProgram128), 40, 72,
        7);
}

ARMIGO_TEST(forward_branch, free_low_opcode_bits_admitted) {
    // Bits 51..48 are free in the shared opcode window; such a word still
    // decodes and targets like any other direct branch.
    constexpr std::uint64_t word = (std::uint64_t{0xe241} << 48) | 15u |
                                   (std::uint64_t{7} << 16) |
                                   (std::uint64_t{32} << 20);
    expect_forward(decode_forward_branch(word, 8, kProgram128), 8, 48, 7);
}

ARMIGO_TEST(forward_branch, forward_near_uint32_max) {
    // pc 0x7fffffc8 + 8 + 16 = 0x7fffffe0, a scheduling word; the normalized
    // target 0x7fffffe8 ends exactly at the 0x7ffffff0 program size.
    constexpr std::uint32_t source = 0x7fffffc8;
    constexpr std::uint32_t program_size = 0x7ffffff0;
    expect_forward(decode_forward_branch(direct_branch_word(0x10), source,
                                         program_size),
                   source, 0x7fffffe8, 7);
}

ARMIGO_TEST(forward_branch, forward_above_int32_max_near_uint32_max) {
    // A source literally above INT32_MAX, near UINT32_MAX. pc 0xffffffc8 +
    // 8 + 8 = 0xffffffd8, a slot (offset 24 in its group); the target word
    // ends exactly at the 0xffffffe0 program size.
    constexpr std::uint32_t source = 0xffffffc8;
    constexpr std::uint32_t program_size = 0xffffffe0;
    expect_forward(
        decode_forward_branch(direct_branch_word(0x8), source, program_size),
        source, 0xffffffd8, 7);
}

ARMIGO_TEST(forward_branch, guard_preservation_all_16) {
    for (std::uint32_t raw = 0; raw < 16; ++raw) {
        expect_forward(
            decode_forward_branch(direct_branch_word(32, static_cast<std::uint8_t>(raw)),
                                  8, kProgram128),
            8, 48, static_cast<std::uint8_t>(raw));
    }
}

ARMIGO_TEST(forward_branch, self_edge_rejected) {
    // pc 40 + 8 - 8 = 40: the normalized target equals the source.
    expect_error(
        decode_forward_branch(direct_branch_word(disp24(-8)), 40, kProgram128),
        ForwardBranchErrorCode::backward_edge);
}

ARMIGO_TEST(forward_branch, backward_edge_rejected) {
    // pc 72 + 8 - 32 = 48, behind the source.
    expect_error(
        decode_forward_branch(direct_branch_word(disp24(-32)), 72, kProgram128),
        ForwardBranchErrorCode::backward_edge);
}

ARMIGO_TEST(forward_branch, scheduler_advanced_self_edge_rejected) {
    // pc 8 + 8 - 16 = 0, a scheduling word; normalization advances to 8,
    // which equals the source.
    expect_error(
        decode_forward_branch(direct_branch_word(disp24(-16)), 8, kProgram128),
        ForwardBranchErrorCode::backward_edge);
}

ARMIGO_TEST(forward_branch, scheduler_advanced_backward_near_max_rejected) {
    // pc 0xffffffc8 + 8 - 16 = 0xffffffc0, a scheduling word; normalization
    // advances it back onto the source itself.
    constexpr std::uint32_t source = 0xffffffc8;
    constexpr std::uint32_t program_size = 0xffffffd0;
    expect_error(decode_forward_branch(direct_branch_word(disp24(-16)), source,
                                       program_size),
                 ForwardBranchErrorCode::backward_edge);
}

ARMIGO_TEST(forward_branch, negative_displacement_backward_rejected) {
    // pc 24 + 8 - 16 = 16, behind the source.
    expect_error(
        decode_forward_branch(direct_branch_word(disp24(-16)), 24, kProgram128),
        ForwardBranchErrorCode::backward_edge);
}

ARMIGO_TEST(forward_branch, source_misaligned_rejected) {
    expect_error(decode_forward_branch(direct_branch_word(32), 12, kProgram128),
                 ForwardBranchErrorCode::invalid_pc);
}

ARMIGO_TEST(forward_branch, source_scheduler_slot_rejected) {
    expect_error(decode_forward_branch(direct_branch_word(32), 0, kProgram128),
                 ForwardBranchErrorCode::invalid_pc);
    expect_error(decode_forward_branch(direct_branch_word(32), 32, kProgram128),
                 ForwardBranchErrorCode::invalid_pc);
}

ARMIGO_TEST(forward_branch, source_beyond_program_rejected) {
    expect_error(decode_forward_branch(direct_branch_word(32), 120, 64),
                 ForwardBranchErrorCode::invalid_pc);
}

ARMIGO_TEST(forward_branch, source_word_beyond_uint32_max_rejected) {
    // The source word spans [0xfffffff8, 0x100000000); 32-bit wraparound
    // would compute pc + 8 as zero and wrongly admit it.
    expect_error(decode_forward_branch(direct_branch_word(0), 0xfffffff8,
                                       0xfffffff8),
                 ForwardBranchErrorCode::invalid_pc);
}

ARMIGO_TEST(forward_branch, target_misaligned_rejected) {
    // pc 8 + 8 + 4 = 20 is not a multiple of 8.
    const ForwardBranchResult result =
        decode_forward_branch(direct_branch_word(4), 8, kProgram128);
    expect_error(result, ForwardBranchErrorCode::invalid_target);
    ARMIGO_CHECK_EQ(failure(result)->detail, std::uint64_t{20});
}

ARMIGO_TEST(forward_branch, target_at_program_end_rejected) {
    // pc 8 + 8 + 48 = 64 equals the program size, one past the last word.
    const ForwardBranchResult result =
        decode_forward_branch(direct_branch_word(48), 8, 64);
    expect_error(result, ForwardBranchErrorCode::invalid_target);
    ARMIGO_CHECK_EQ(failure(result)->detail, std::uint64_t{64});
}

ARMIGO_TEST(forward_branch, target_past_bare_scheduler_group_rejected) {
    // A 72-byte program ends with a bare scheduling word at 64. The target
    // 64 normalizes to 72, and 72 + 8 exceeds the program size.
    const ForwardBranchResult result =
        decode_forward_branch(direct_branch_word(48), 8, 72);
    expect_error(result, ForwardBranchErrorCode::invalid_target);
    ARMIGO_CHECK_EQ(failure(result)->detail, std::uint64_t{72});
}

ARMIGO_TEST(forward_branch, negative_target_below_zero_rejected) {
    // pc 8 + 8 - 24 = -8 leaves the program region below zero.
    expect_error(
        decode_forward_branch(direct_branch_word(disp24(-24)), 8, kProgram128),
        ForwardBranchErrorCode::invalid_target);
}

ARMIGO_TEST(forward_branch, max_negative_displacement_rejected) {
    // pc 8 + 8 - 8'388'608 is far below zero.
    expect_error(decode_forward_branch(direct_branch_word(0x800000), 8,
                                       kProgram128),
                 ForwardBranchErrorCode::invalid_target);
}

ARMIGO_TEST(forward_branch, target_word_beyond_uint32_max_rejected) {
    // The target word spans [0xfffffff8, 0x100000000); 32-bit wraparound
    // would compute target + 8 as zero and wrongly admit it.
    expect_error(
        decode_forward_branch(direct_branch_word(0), 0xfffffff0, 0xffffffff),
        ForwardBranchErrorCode::invalid_target);
}

ARMIGO_TEST(forward_branch, opcode_neighbors_not_branch) {
    // BRK, BRX, EXIT, and zero windows are not direct relative branches.
    expect_error(decode_forward_branch((std::uint64_t{0xe340} << 48) | 15u, 8,
                                       kProgram128),
                 ForwardBranchErrorCode::not_branch);
    expect_error(decode_forward_branch((std::uint64_t{0xe250} << 48) | 15u, 8,
                                       kProgram128),
                 ForwardBranchErrorCode::not_branch);
    expect_error(decode_forward_branch((std::uint64_t{0xe300} << 48) | 15u, 8,
                                       kProgram128),
                 ForwardBranchErrorCode::not_branch);
    expect_error(decode_forward_branch(0, 8, kProgram128),
                 ForwardBranchErrorCode::not_branch);
}

ARMIGO_TEST(forward_branch, conditional_flow_tests_rejected) {
    constexpr std::uint64_t base = std::uint64_t{0xe240} << 48;
    // Flow test 1 (a comparison test) instead of the always-true test.
    expect_error(decode_forward_branch(base | 1u | (std::uint64_t{32} << 20), 8,
                                       kProgram128),
                 ForwardBranchErrorCode::unsupported_field);
    // Flow test 16 (off) does not exist in the admitted profile either.
    expect_error(decode_forward_branch(base | 16u | (std::uint64_t{32} << 20),
                                       8, kProgram128),
                 ForwardBranchErrorCode::unsupported_field);
}

ARMIGO_TEST(forward_branch, absolute_target_mode_rejected) {
    constexpr std::uint64_t word = (std::uint64_t{0xe240} << 48) | 15u |
                                   (std::uint64_t{1} << 5) |
                                   (std::uint64_t{32} << 20);
    expect_error(decode_forward_branch(word, 8, kProgram128),
                 ForwardBranchErrorCode::unsupported_field);
}

ARMIGO_TEST(forward_branch, reserved_low_bits_rejected) {
    constexpr std::uint64_t word = (std::uint64_t{0xe240} << 48) | 15u |
                                   (std::uint64_t{1} << 6) |
                                   (std::uint64_t{32} << 20);
    const ForwardBranchResult result = decode_forward_branch(word, 8, kProgram128);
    expect_error(result, ForwardBranchErrorCode::unsupported_field);
    ARMIGO_CHECK_EQ(failure(result)->detail, std::uint64_t{1});
}

ARMIGO_TEST(forward_branch, reserved_high_bits_rejected) {
    constexpr std::uint64_t word = (std::uint64_t{0xe240} << 48) | 15u |
                                   (std::uint64_t{1} << 44) |
                                   (std::uint64_t{32} << 20);
    const ForwardBranchResult result = decode_forward_branch(word, 8, kProgram128);
    expect_error(result, ForwardBranchErrorCode::unsupported_field);
    ARMIGO_CHECK_EQ(failure(result)->detail, std::uint64_t{1});
}

ARMIGO_TEST(forward_branch, messages_cover_all_codes) {
    for (const ForwardBranchErrorCode code :
         {ForwardBranchErrorCode::not_branch,
          ForwardBranchErrorCode::unsupported_field,
          ForwardBranchErrorCode::invalid_pc,
          ForwardBranchErrorCode::invalid_target,
          ForwardBranchErrorCode::backward_edge}) {
        const ForwardBranchError error{code, 0, 0};
        ARMIGO_CHECK(error.message() != nullptr);
    }
}

}  // namespace armigo::shader::test
