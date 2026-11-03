// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "../decoder.h"

#include <span>
#include <vector>

namespace armigo::shader::pipeline::flow {
// Flow-lane classification of decoded instruction names. These classes drive
// block termination, push/pop token propagation and call discovery; they are
// lane-3 ownership, independent of the encoding tables published by lanes 1-2.
enum class FlowClass : std::uint8_t {
    plain,        // Stays inside its block; emitted by later lanes or rejected there.
    push,         // SSY/PBK/PCNT: pushes a typed reconvergence token.
    pop,          // SYNC/BRK/CONT: pops the topmost token of a matching kind.
    branch,       // BRA: static displacement target, predicable.
    call,         // CAL: enqueues a callee function; execution returns to the next block.
    exit_like,    // EXIT/KIL/RET: terminates with no displacement target.
    unsupported,  // BRX/JMP/JCAL/LONGJMP/PLONGJMP/PRET/PEXIT: explicit decode failure.
};
[[nodiscard]] FlowClass classify_flow(InstructionName name);
[[nodiscard]] TokenKind push_token_kind(InstructionName name);
[[nodiscard]] TokenKind pop_token_kind(InstructionName name);
// True when the predicate (and, when present, the condition-code selector) is
// statically always true, so no fallthrough alternative exists.
[[nodiscard]] bool flow_unconditional(const DecodedInstruction& instruction);
// Lane-3 displacement convention: the branch_displacement field stores the
// signed byte displacement from the instruction's own PC to the target (the
// sign-extended immediate plus the "+8 next instruction" bias). Lane 2
// must publish displacement fields in the same normalized form.
[[nodiscard]] Result<Pc> flow_target(const DecodedInstruction& instruction);

// Merges the lane 1/2 rule tables, rejecting null callbacks, empty masks and
// overlapping (ambiguously matching) mask/value pairs.
[[nodiscard]] Result<std::vector<DecodeRule>> merge_decode_rules(
    std::span<const DecodeRule> first, std::span<const DecodeRule> second);

// Decode entry over an explicit rule table. The public decode() binds this to
// the merged published tables; tests bind synthetic local rule sets.
[[nodiscard]] Result<DecodedProgram> decode_stream(
    const StageImage& image, FactReader& facts, Limits limits, std::span<const DecodeRule> rules);

// Structural self-check for decoded programs: ownership, edges, token stacks,
// transition coverage and unique pop selectors. Run at the end of decode; also
// usable on hand-built programs.
[[nodiscard]] Status validate_decoded_cfg(const DecodedProgram& program);
} // namespace armigo::shader::pipeline::flow
