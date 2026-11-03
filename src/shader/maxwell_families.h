// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Private instruction-family registry of the Maxwell decoder. Families that
// live in their own translation unit register one immutable entry here; the
// legacy classifier in maxwell.cpp always runs first and the registry is
// consulted only when that classifier returns Unknown, so a registered entry
// can never override a legacy family.
#pragma once

#include <armigo/shader/maxwell.h>

#include <span>
#include <variant>

#include "maxwell_decode_helpers.h"

namespace armigo::shader::maxwell::detail {

// What a family decoder returns: the decoded instruction, or the same
// structured error the legacy decoder path would have produced.
using DecodeOutcome = std::variant<Instruction, DecodeError>;
[[nodiscard]] DecodeOutcome decode_word_and_immediate(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_fma_register_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_fma_constant_negated(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_fmul_saturate_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_ffma_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_fmul_immediate32(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_fadd_immediate32(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_range_reduction_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_range_reduction_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_dependency_barrier(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_select_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_select_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_select_immediate(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_logic_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_logic_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_logic_immediate(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_float_to_integer_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_float_to_integer_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_float_min_max_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_float_min_max_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_float_min_max_immediate(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_integer_to_float_register(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);
[[nodiscard]] DecodeOutcome decode_integer_to_float_constant(std::uint64_t word,
    std::uint32_t pc, Predicate predicate);

// One admitted instruction family. `mask` and `value` match bits 48..63 of
// the instruction word. The decoder hands over a word whose instruction
// predicate is already validated as PT; a family decoder still reports its
// own field and reserved-bit errors exactly as the legacy path did.
struct FamilyEntry {
    std::uint16_t mask;
    std::uint16_t value;
    OpcodeFamily family;
    DecodeOutcome (*decode)(std::uint64_t word, std::uint32_t pc,
                            Predicate predicate);
};

// Immutable, append-only table of registered decoder families.
[[nodiscard]] std::span<const FamilyEntry> registered_families() noexcept;

// First registered family matching the word's high half, or null.
[[nodiscard]] inline const FamilyEntry* find_registered_family(
    std::uint64_t word) noexcept {
    for (const FamilyEntry& entry : registered_families()) {
        if (high_matches(word, entry.mask, entry.value)) {
            return &entry;
        }
    }
    return nullptr;
}

// Unit decoders; each is defined beside the family it serves and declared
// here so the table below stays the only registration point.
[[nodiscard]] DecodeOutcome decode_mov_register(std::uint64_t word,
                                                std::uint32_t pc,
                                                Predicate predicate);

[[nodiscard]] DecodeOutcome decode_mov_constant(std::uint64_t word,
                                                std::uint32_t pc,
                                                Predicate predicate);

[[nodiscard]] DecodeOutcome decode_isetp_register(std::uint64_t word,
                                                  std::uint32_t pc,
                                                  Predicate predicate);

[[nodiscard]] DecodeOutcome decode_predicate_set_predicate(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_float_set_predicate_register(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_float_set_predicate_constant(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_float_set_predicate_immediate(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_float_round_integral_register(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

[[nodiscard]] DecodeOutcome decode_float_round_integral_constant(
    std::uint64_t word, std::uint32_t pc, Predicate predicate);

}  // namespace armigo::shader::maxwell::detail
