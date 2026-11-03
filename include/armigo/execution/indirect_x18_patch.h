// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/scalar_x18_patch.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace armigo::execution {

struct IndirectX18Target {
    std::uint64_t guest_address = 0;
    std::uint64_t prepared_address = 0;
};

enum class IndirectX18PatchError {
    none,
    invalid_layout,
    overlapping_regions,
    invalid_thread_slot,
    missing_execution_contract,
    unsupported_instruction,
    register_not_used,
    missing_targets,
    invalid_target,
    duplicate_target,
    unreachable_branch,
};

struct IndirectX18Patch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    std::vector<std::uint32_t> trampoline;
};

struct IndirectX18PatchResult {
    IndirectX18PatchError error = IndirectX18PatchError::none;
    std::optional<IndirectX18Patch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Prepares an in-place BR/BLR X18 using a finite mapping from guest pointer
// values to prepared native destinations. All registers and NZCV survive a
// successful BR; BLR changes only x30 to instruction_address + 4. Hardware
// x18 and the virtual x18 cell are never written. Unknown targets restore
// scratch registers and SP, then trap with BRK instead of transferring.
//
// The caller establishes the scalar execution contract, binds the register
// cell in the probed native TSD slot, proves every mapped destination prepared,
// and verifies global overlap and preimages before writing/sealing code.
// Guest addresses are comparison values, never implicitly rebased. Prepared
// addresses, instruction_address and trampoline_address are native addresses.
// A relocated branch site with a different architectural link address is not
// supported. Destination copies are supported when the caller proves their
// semantics and supplies a consistent map for every incoming edge.
// Ordinary completion only: BTI/GCS/MTE, asynchronous reentry and fault-time
// state reconstruction require separate contracts. No target discovery or
// module-wide preparation is performed here.
IndirectX18PatchResult make_indirect_x18_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, std::uint32_t tsd_byte_offset,
    std::span<const IndirectX18Target> targets, ScalarX18Policy policy = {});

// Prepares an in-place BR X18 whose target is a word inside a table that
// starts at `table_base` and holds `entries` instructions (a power of two),
// the shape of a switch that adds a scaled byte index to an ADR base. The
// trampoline reads virtual x18, checks that the target lies inside the table
// and is word-aligned without changing NZCV, then restores x0/x1 and SP and
// branches directly to the matching table word; every integer register and
// NZCV survive. A target outside the table restores the scratch pair and SP
// and executes BRK #indirect_x18_table_trap. Table words are guest code at their native addresses
// (code is not relocated). The caller proves the table bound from the
// instructions that compute x18 and establishes the scalar contract.
IndirectX18PatchResult make_indirect_x18_table_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, std::uint32_t tsd_byte_offset,
    std::uint64_t table_base, std::uint32_t entries, ScalarX18Policy policy = {});

// The BRK immediate of a table trampoline's out-of-table trap.
inline constexpr std::uint16_t indirect_x18_table_trap = 0xae22;

// Words the table trampoline occupies for `entries` table instructions.
[[nodiscard]] constexpr std::size_t indirect_x18_table_words(std::uint32_t entries) noexcept {
    return 16 + 2 * std::size_t{entries} + 2;
}

} // namespace armigo::execution
