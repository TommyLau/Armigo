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

#include <array>
#include <cstdint>
#include <optional>

namespace armigo::execution {

enum class ScalarX18PatchError {
    none,
    invalid_layout,
    overlapping_regions,
    invalid_thread_slot,
    missing_execution_contract,
    unsupported_instruction,
    stack_pointer_operand,
    exclusive_instruction,
    register_not_used,
    substitution_failed,
    unreachable_branch,
};

struct ScalarX18Policy {
    // The caller establishes these for every execution of the patched site.
    // Defaults deliberately establish neither prerequisite.
    bool exclusive_monitor_clear = false;
    // The 16 bytes immediately below guest SP are writable, aligned, unused,
    // disjoint from the original access and private to this execution.
    bool stack_scratch_available = false;
    // The original memory access cannot reach the guest-register cell, native
    // TSD slot, or other runtime storage used by the trampoline. This builder
    // cannot prove address isolation and does not provide a memory sandbox.
    bool runtime_storage_disjoint = false;
};

struct ScalarX18Patch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    // Only the first trampoline_word_count entries describe the plan.
    std::array<std::uint32_t, 12> trampoline{};
    std::uint8_t trampoline_word_count = 0;
    std::uint8_t value_register = 0;
    std::uint8_t cell_register = 0;
};

struct ScalarX18PatchResult {
    ScalarX18PatchError error = ScalarX18PatchError::none;
    std::optional<ScalarX18Patch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Plans one decoded non-branch x18/W18 instruction, or ADR/ADRP X18, using a native
// thread slot that publishes GuestRegisterCell. The slot offset must come from
// a successful platform probe; the correct cell must be bound before entry.
// The builder checks encoding, complete register effects, scratch conflicts,
// branch reach and region overlap. ADRP retains the original page target and
// is re-encoded for the trampoline PC; unrepresentable targets fail substitution.
// It does not inspect process memory, prove
// the execution policy, or discover exclusive-monitor windows.
//
// Unsigned-offset and unscaled LDR/STR/LDUR/STUR of any transfer width through
// SP, pair loads and stores (LDP/STP/LDNP/STNP, LDPSW), and their pre-index
// and post-index forms, compensate for the 16-byte spill. Pre-index and
// post-index become one plain offset access followed by an SP update in the
// trampoline epilogue. Immediate ADD/ADDS/SUB/SUBS W18/X18 from WSP/SP instead
// reconstruct the original SP in the value scratch, preserving the original
// immediate, width and flag behavior. AdvSIMD structures through general bases,
// general/vector copies and integer/FP conversions use checked substitution;
// post-index base writeback updates the virtual cell. Other SP operands,
// PC-relative instructions other than ADR/ADRP, branches, system operations,
// exclusive instructions, register-indexed
// SP accesses and unknown encodings are refused; so is any stack access that
// could read or write the 16 spill bytes themselves. The trampoline
// temporarily decrements SP by 16 and preserves both scratch registers;
// injected instructions leave x30 and NZCV unchanged.
// The original instruction may still write x30. Only ordinary completion is
// modeled: exception-time register state, asynchronous reentry, BTI/GCS/MTE
// execution policies and fault recovery require separate preparation.
// Addresses share one namespace; original word and continuation must remain
// outside the caller-owned trampoline. Verify preimages and the complete
// module plan before writing nonexecutable code, then seal it.
ScalarX18PatchResult make_scalar_x18_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, std::uint32_t tsd_byte_offset,
    ScalarX18Policy policy = {}) noexcept;

} // namespace armigo::execution
