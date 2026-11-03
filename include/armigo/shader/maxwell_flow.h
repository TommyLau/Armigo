// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Frontend helper for the direct relative Maxwell BRA instruction. Decodes
// one synthetic-forward branch candidate and validates its target against the
// instruction program layout. This is a per-word frontend service for the
// bounded decoder; it does not by itself enable branches in decode() or
// claim guest control-flow support.
#pragma once
#include <armigo/shader/maxwell_operands.h>

#include <cstdint>
#include <variant>

namespace armigo::shader::maxwell {

// A decoded direct relative branch whose target has been validated and
// normalized. target_pc is the scheduling-normalized program offset of the
// first instruction slot at the destination.
struct ForwardBranch {
    std::uint32_t source_pc = 0;
    std::uint32_t target_pc = 0;
    Predicate predicate;
};

enum class ForwardBranchErrorCode : std::uint8_t {
    // The word does not match the shared direct-branch opcode window.
    not_branch,
    // The word is a direct branch but carries a field outside the admitted
    // always-tested direct relative profile.
    unsupported_field,
    // The source pc is not an 8-byte-aligned, fully contained instruction
    // slot of the program region.
    invalid_pc,
    // The computed or normalized target is misaligned or not a fully
    // contained instruction slot of the program region.
    invalid_target,
    // The normalized target is at or before the source pc; the validated
    // branch profile is forward-only and acyclic.
    backward_edge,
};

struct ForwardBranchError {
    ForwardBranchErrorCode code = ForwardBranchErrorCode::not_branch;
    // Source pc of the rejected query.
    std::uint32_t pc = 0;
    // Offending value, meaning depends on code: not_branch and
    // unsupported_field carry the offending field or window value,
    // invalid_pc and invalid_target carry the offending byte offset, and
    // backward_edge carries the normalized target.
    std::uint64_t detail = 0;

    [[nodiscard]] const char* message() const noexcept;
};

using ForwardBranchResult = std::variant<ForwardBranch, ForwardBranchError>;

// Decodes `word` as a direct relative branch whose source instruction slot
// starts at byte offset `pc` of the instruction program. `program_size` is
// the size of the instruction program region in bytes; it excludes the
// graphics wrapper and SPH header, and offsets are relative to that region.
//
// Admitted profile: the shared direct-branch opcode window, an always-true
// flow test, the direct (non-constant-buffer) target mode, zero reserved
// fields, and any 4-bit instruction predicate guard, preserved verbatim.
// Target arithmetic uses pc + 8 + signed 24-bit displacement in wide signed
// arithmetic, then advances a scheduling-word target by one instruction word
// and rechecks containment. Checks run in order: word shape, fields, source
// pc, target, edge direction. A rejection is always a structured error and
// never a success or NOP.
[[nodiscard]] ForwardBranchResult decode_forward_branch(
    std::uint64_t word, std::uint32_t pc, std::uint32_t program_size);

// Shared BRA/SSY/SYNC target validation, including scheduling-word aliases.
// The raw target is already computed in wide signed arithmetic.
[[nodiscard]] ForwardBranchResult resolve_forward_target(
    std::int64_t raw_target, std::uint32_t pc, std::uint32_t program_size,
    Predicate predicate);

}  // namespace armigo::shader::maxwell
