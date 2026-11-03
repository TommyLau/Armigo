// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Checked substitution of one general register in one A64 instruction word.
//
// substitute_register() renames every explicit operand field that names
// general register `from` (X0..X30) to `to` (X0..X30). It works only from the
// operand fields reported by decode_register_effects(); it never searches the
// word for bit patterns, so immediates, prefetch operations, SIMD&FP register
// numbers and register-31 operands are never touched. The rewritten word is
// decoded again and must have the same operation, operand layout, views,
// accesses, roles, flags, control flow, memory effect and notes, with only the
// renamed operands naming `to`.
//
// This is a single-instruction field rewrite. It does not establish that
// `to` is dead, choose or spill scratch registers, relocate PC-relative code,
// keep exclusive-monitor windows intact, or prove that rewritten code is safe
// to execute natively.

#pragma once

#include <armigo/aarch64/register_effects.h>

#include <cstdint>

namespace armigo::aarch64 {

enum class SubstitutionError : std::uint8_t {
    none,
    // from or to is above 30. Field value 31 is the zero register or SP,
    // never a general register this API can rename to or from.
    invalid_register,
    // The original word is not EffectStatus::decoded.
    unknown_encoding,
    reserved_encoding,
    unpredictable_encoding,
    // Exception generation: the handler's register use is not described.
    handler_defined_registers,
    // from is used by an implicit operand (X30 written by BL and BLR), which
    // no field rewrite can rename.
    implicit_operand,
    // from == to under UnchangedWord::reject.
    same_register,
    // The instruction does not name from, under UnchangedWord::reject.
    register_absent,
    // to is already named by another operand (explicit or implicit). The
    // rewrite would merge two independently named registers.
    register_collision,
    // verify_register_substitution(): the rewritten word differs from the
    // original outside the fields that name from.
    changed_outside_fields,
    // The rewritten word is unknown, reserved or CONSTRAINED UNPREDICTABLE.
    output_not_decoded,
    // The rewritten word's effects are not those of the original with from
    // renamed to to.
    effects_mismatch,
};

// Whether a substitution that leaves the word unchanged (from == to, or an
// instruction that does not name from) is a success or an error. The original
// word must still be completely decoded in either case.
enum class UnchangedWord : std::uint8_t {
    reject,
    accept,
};

struct SubstitutedWord {
    // The rewritten word; the original word when it is unchanged and
    // accepted. Zero on failure.
    std::uint32_t word = 0;
    SubstitutionError error = SubstitutionError::none;
    // Word bits that were rewritten (the original fields naming from). Zero
    // when the word is unchanged or on failure.
    std::uint32_t fields = 0;
    // Effects of word, decoded again after the rewrite. Default on failure.
    RegisterEffects effects;

    bool ok() const noexcept { return error == SubstitutionError::none; }
    bool changed() const noexcept { return fields != 0; }
};

SubstitutedWord substitute_register(std::uint32_t word, unsigned from, unsigned to,
                                    UnchangedWord unchanged) noexcept;

// Checks a supplied rewrite against the same contract: original must be a
// valid substitution input, rewritten may differ only in the fields of
// original that name from, it must be decoded, and its effects must equal
// those of original with from renamed to to. On success the result describes
// rewritten. substitute_register() applies this check to its own output.
SubstitutedWord verify_register_substitution(std::uint32_t original, std::uint32_t rewritten,
                                             unsigned from, unsigned to,
                                             UnchangedWord unchanged) noexcept;

} // namespace armigo::aarch64
