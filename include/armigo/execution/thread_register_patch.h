// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Inline thread-pointer reads: MRS Xt, TPIDR_EL0 and MRS Xt, TPIDRRO_EL0 read
// the guest register straight from the thread's GuestRegisterCell instead of
// entering the host gate.
//
// The replacement is a branch to a four-word trampoline that loads the native
// TSD base into the destination register, follows the published cell pointer
// and loads the cell word. The destination register is the only register
// written; no scratch register, stack slot or flag is used, so the site needs
// no spill and cannot disturb NZCV. Writes to TPIDR_EL0 and every other system
// instruction keep using the gate. The result is the value the gate handler
// would have stored (execution::GuestThreadContext::thread_pointer /
// read_only_thread_pointer), because both address the same cell.

#pragma once

#include <armigo/aarch64/instruction.h>
#include <armigo/execution/guest_register_cell.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace armigo::execution {

enum class ThreadRegisterPatchError {
    none,
    unsupported_instruction,  // not MRS of TPIDR_EL0/TPIDRRO_EL0, or Xt is XZR or x18
    invalid_thread_slot,
    invalid_layout,           // misaligned or overflowing addresses
    overlapping_regions,
    unreachable_branch,
};

struct ThreadRegisterReadPatch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    std::array<std::uint32_t, 4> trampoline{};
    std::uint8_t destination_register = 0;
};

struct ThreadRegisterReadPatchResult {
    ThreadRegisterPatchError error = ThreadRegisterPatchError::none;
    std::optional<ThreadRegisterReadPatch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Plans one MRS of TPIDR_EL0 or TPIDRRO_EL0 using the native thread slot that
// publishes GuestRegisterCell (tsd_byte_offset comes from a successful
// platform probe, and the correct cell must be bound before entry). Xt may be
// any general register except XZR and x18, including x29 and x30. The caller
// keeps exclusive-monitor windows out of scope exactly as for gate sites: the
// trampoline performs ordinary loads.
inline ThreadRegisterReadPatchResult make_thread_register_read_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, std::uint32_t tsd_byte_offset) noexcept {
    using namespace aarch64;
    const auto decoded = decode(instruction);
    if (decoded.opcode != Opcode::mrs || decoded.reg >= 31 || decoded.reg == 18) {
        return {.error = ThreadRegisterPatchError::unsupported_instruction};
    }
    std::size_t cell_offset;
    switch (identify_system_register(decoded.system_register)) {
    case SystemRegister::tpidr_el0: cell_offset = kGuestRegisterCellThreadPointerOffset; break;
    case SystemRegister::tpidrro_el0:
        cell_offset = kGuestRegisterCellReadOnlyThreadPointerOffset;
        break;
    default: return {.error = ThreadRegisterPatchError::unsupported_instruction};
    }
    if (tsd_byte_offset % 8 != 0 || tsd_byte_offset > 4095 * 8) {
        return {.error = ThreadRegisterPatchError::invalid_thread_slot};
    }
    constexpr std::uint64_t trampoline_bytes = 16;
    constexpr std::uint64_t maximum = ~std::uint64_t{0};
    if (instruction_address % 4 != 0 || trampoline_address % 4 != 0 ||
        instruction_address > maximum - 7 || trampoline_address > maximum - (trampoline_bytes - 1)) {
        return {.error = ThreadRegisterPatchError::invalid_layout};
    }
    if (instruction_address <= trampoline_address + (trampoline_bytes - 1) &&
        trampoline_address <= instruction_address + 7) {
        return {.error = ThreadRegisterPatchError::overlapping_regions};
    }
    const auto enter = encode_b(instruction_address, trampoline_address);
    const auto resume = encode_b(trampoline_address + 12, instruction_address + 4);
    if (!enter.ok() || !resume.ok()) {
        return {.error = ThreadRegisterPatchError::unreachable_branch};
    }
    const unsigned rt = decoded.reg;
    const auto read_tsd = encode_mrs(rt, system_register_encoding(3, 3, 13, 0, 3));
    if (!read_tsd.ok()) return {.error = ThreadRegisterPatchError::unsupported_instruction};
    ThreadRegisterReadPatch patch{.instruction_address = instruction_address,
                                  .trampoline_address = trampoline_address,
                                  .expected_instruction = instruction,
                                  .replacement_instruction = enter.word,
                                  .destination_register = static_cast<std::uint8_t>(rt)};
    // LDR Xt, [Xt, #imm]: the cell pointer, then the cell word.
    patch.trampoline = {read_tsd.word,
                        0xf9400000u | ((tsd_byte_offset / 8) << 10) | (rt << 5) | rt,
                        0xf9400000u | (static_cast<std::uint32_t>(cell_offset / 8) << 10) |
                            (rt << 5) | rt,
                        resume.word};
    return {.patch = patch};
}

}  // namespace armigo::execution
