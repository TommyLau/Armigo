// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/maxwell_flow.h>

#include "maxwell_decode_helpers.h"

#include <armigo/shader_input/shader_input.h>

namespace armigo::shader::maxwell {
namespace {

// The opcode window shared with the bounded decoder's Branch classification.
constexpr std::uint16_t kDirectBranchWindowMask = 0xfff0;
constexpr std::uint16_t kDirectBranchWindowValue = 0xe240;
// Flow test 15 is the always-true test; the only admitted condition.
constexpr std::uint64_t kAlwaysFlowTest = 15;

constexpr std::uint64_t branch_window(std::uint64_t word) {
    return word >> 48;
}

// Wide signed displacement from bits 20..43, sign-extended through 64 bits.
constexpr std::int64_t displacement(std::uint64_t word) {
    const auto raw = static_cast<std::int64_t>(detail::field(word, 20, 24));
    return (raw << 40) >> 40;
}

ForwardBranchError error(ForwardBranchErrorCode code, std::uint32_t pc,
                         std::uint64_t detail) {
    return ForwardBranchError{code, pc, detail};
}

}  // namespace

const char* ForwardBranchError::message() const noexcept {
    switch (code) {
        case ForwardBranchErrorCode::not_branch:
            return "word is not a direct relative branch instruction";
        case ForwardBranchErrorCode::unsupported_field:
            return "branch field is outside the direct forward-branch profile";
        case ForwardBranchErrorCode::invalid_pc:
            return "branch source pc is not a fully contained instruction slot";
        case ForwardBranchErrorCode::invalid_target:
            return "branch target is misaligned or not fully contained";
        case ForwardBranchErrorCode::backward_edge:
            return "branch target is not forward of the source pc";
    }
    return "unknown forward branch error";
}

ForwardBranchResult decode_forward_branch(std::uint64_t word, std::uint32_t pc,
                                          std::uint32_t program_size) {
    if ((branch_window(word) & kDirectBranchWindowMask) !=
        kDirectBranchWindowValue) {
        return error(ForwardBranchErrorCode::not_branch, pc,
                     branch_window(word));
    }
    const auto flow_test = detail::field(word, 0, 5);
    if (flow_test != kAlwaysFlowTest) {
        return error(ForwardBranchErrorCode::unsupported_field, pc, flow_test);
    }
    if (detail::field(word, 5, 1) != 0) {
        return error(ForwardBranchErrorCode::unsupported_field, pc,
                     detail::field(word, 5, 1));
    }
    const std::uint64_t reserved_low = detail::field(word, 6, 10);
    if (reserved_low != 0) {
        return error(ForwardBranchErrorCode::unsupported_field, pc,
                     reserved_low);
    }
    const std::uint64_t reserved_high = detail::field(word, 44, 4);
    if (reserved_high != 0) {
        return error(ForwardBranchErrorCode::unsupported_field, pc,
                     reserved_high);
    }

    return resolve_forward_target(
        static_cast<std::int64_t>(pc) + 8 + displacement(word), pc,
        program_size,
        Predicate{static_cast<std::uint8_t>(detail::field(word, 16, 3)),
                  detail::field(word, 19, 1) != 0});
}

ForwardBranchResult resolve_forward_target(std::int64_t raw, std::uint32_t pc,
                                           std::uint32_t program_size,
                                           Predicate predicate) {
    const std::uint64_t source = pc;
    if (source % shader_input::format::instruction_word_size != 0 ||
        source % shader_input::format::scheduling_group_size == 0) {
        return error(ForwardBranchErrorCode::invalid_pc, pc, source);
    }
    if (source + shader_input::format::instruction_word_size >
        program_size) {
        return error(ForwardBranchErrorCode::invalid_pc, pc,
                     source + shader_input::format::instruction_word_size);
    }

    if (raw < 0 ||
        raw % static_cast<std::int64_t>(
                   shader_input::format::instruction_word_size) != 0) {
        return error(ForwardBranchErrorCode::invalid_target, pc,
                     static_cast<std::uint64_t>(raw));
    }
    const auto raw_target = static_cast<std::uint64_t>(raw);
    if (raw_target >= program_size) {
        return error(ForwardBranchErrorCode::invalid_target, pc, raw_target);
    }
    const std::uint64_t target =
        raw_target % shader_input::format::scheduling_group_size == 0
            ? raw_target + shader_input::format::instruction_word_size
            : raw_target;
    if (target + shader_input::format::instruction_word_size > program_size) {
        return error(ForwardBranchErrorCode::invalid_target, pc, target);
    }
    if (target <= source) {
        return error(ForwardBranchErrorCode::backward_edge, pc, target);
    }

    return ForwardBranch{pc, static_cast<std::uint32_t>(target), predicate};
}

}  // namespace armigo::shader::maxwell
