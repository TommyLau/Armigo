// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Moving one classified instruction from its original address to a new one
// while it keeps referring to the same address.
//
// Relocation never truncates a displacement, never substitutes a NOP, and
// never copies an unclassified word: an unclassified word is not proven
// position independent. Every refusal is a typed error.

#pragma once

#include <armigo/aarch64/instruction.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::aarch64 {

enum class RelocationError : std::uint8_t {
    none,
    unclassified_instruction,  // not examined, so not proven movable
    misaligned_address,        // from or to is not 4-byte aligned
    address_overflow,          // the original reference leaves the address space
    out_of_range,              // the reference is unreachable from the new address
    unsupported_operand,       // an operand form the expansion cannot preserve
    scratch_register_required, // the expansion needs a caller-provided scratch register
    invalid_scratch_register,  // scratch is above 30, is x18, or conflicts with operands
};

// Single-word relocation: the result has the same opcode and operands as the
// original and differs only within displacement_field_mask(opcode).
// Position-independent classified instructions are returned unchanged.
struct RelocatedWord {
    std::uint32_t word = 0;
    RelocationError error = RelocationError::none;
    // BL and BLR at the new address write to + 4, not from + 4, into x30.
    bool link_value_changed = false;
    bool ok() const noexcept { return error == RelocationError::none; }
};

RelocatedWord relocate_instruction(std::uint32_t word, std::uint64_t from,
                                   std::uint64_t to) noexcept;

inline constexpr std::uint8_t no_scratch_register = 0xFF;

struct SequenceOptions {
    // Expansions model ordinary completion under a baseline execution policy.
    // The caller must exclude active GCS, incompatible BTI guarding, and MTE
    // tag checks introduced by literal-load expansion. Fault-time register
    // state is not preserved. This API does not inspect host feature policy.
    // A register the caller guarantees is dead at the instruction. Required
    // only for far SIMD/FP literal loads and far PRFM, which have no general
    // destination to build the address in. Never x18 or register 31.
    std::uint8_t scratch_register = no_scratch_register;
    // For BL/BLR, set x30 to the original return address (from + 4) and
    // branch, so the callee returns to the original instruction stream and
    // return-address based unwinding still finds the original call site.
    bool preserve_link_address = false;
};

inline constexpr std::size_t max_sequence_words = 3;

// Relocation that may expand to a short sequence. The sequence occupies
// count consecutive words starting at to; falling through its last word
// continues at to + 4 * count, which corresponds to from + 4. A call with a
// preserved link address has no fall-through: its callee returns to from + 4.
//
// Expansions, used only when the single-word form cannot reach:
//   B.cond/BC.cond     b.<inverted> +8 ; b target   (AL/NV become b target)
//   CBZ/CBNZ, TBZ/TBNZ inverted test over +8 ; b target
//   ADR                adrp rd ; add rd, rd, #lo12   (rd != 31)
//   LDR/LDRSW literal  adrp rt ; ldr rt, [rt, #lo12]  or adrp ; add ; ldr [rt]
//   SIMD/FP, PRFM      as above through the scratch register
//   BL/BLR (preserve_link_address) adrp x30 ; add x30, x30, #lo12 ; b|br
// B, BL without preserve_link_address, and ADRP have no expansion: a target
// beyond their single-word reach needs a branch island or a register the
// caller has proven dead, and is reported as out_of_range.
struct RelocatedSequence {
    std::array<std::uint32_t, max_sequence_words> words{};
    std::size_t count = 0;
    RelocationError error = RelocationError::none;
    bool link_value_changed = false;
    bool ok() const noexcept { return error == RelocationError::none; }
    std::span<const std::uint32_t> view() const noexcept { return {words.data(), count}; }
};

RelocatedSequence relocate_instruction_sequence(std::uint32_t word, std::uint64_t from,
                                                std::uint64_t to,
                                                const SequenceOptions& options = {}) noexcept;

} // namespace armigo::aarch64
