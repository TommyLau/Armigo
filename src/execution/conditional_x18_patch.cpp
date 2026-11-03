// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/conditional_x18_patch.h>

#include <armigo/aarch64/instruction.h>
#include <armigo/execution/guest_register_cell.h>

#include <limits>

namespace armigo::execution {

ConditionalX18PatchResult make_conditional_x18_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint32_t slot, ScalarX18Policy policy) noexcept {
    using namespace aarch64;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (address % 4 != 0 || trampoline % 4 != 0 || address > maximum - 7 ||
        trampoline > maximum - 35) {
        return {.error = ScalarX18PatchError::invalid_layout};
    }
    if (address <= trampoline + 35 && trampoline <= address + 7) {
        return {.error = ScalarX18PatchError::overlapping_regions};
    }
    if (slot % 8 != 0 || slot > 4095 * 8) {
        return {.error = ScalarX18PatchError::invalid_thread_slot};
    }
    if (!policy.exclusive_monitor_clear || !policy.stack_scratch_available ||
        !policy.runtime_storage_disjoint) {
        return {.error = ScalarX18PatchError::missing_execution_contract};
    }
    const auto decoded = decode(instruction);
    const bool compare = decoded.opcode == Opcode::cbz || decoded.opcode == Opcode::cbnz;
    const bool test = decoded.opcode == Opcode::tbz || decoded.opcode == Opcode::tbnz;
    if (!compare && !test) {
        return {.error = ScalarX18PatchError::unsupported_instruction};
    }
    if (decoded.reg != platform_register) {
        return {.error = ScalarX18PatchError::register_not_used};
    }
    const auto target = referenced_address(decoded, address);
    if (!target) return {.error = ScalarX18PatchError::invalid_layout};
    if (*target >= trampoline && *target <= trampoline + 35) {
        return {.error = ScalarX18PatchError::overlapping_regions};
    }

    // The original branch reads only x18, so both scratch registers are free
    // after saving their original values. No injected instruction changes flags.
    constexpr unsigned value = 0, cell = 1;
    const auto enter = encode_b(address, trampoline);
    const auto resume = encode_b(trampoline + 24, address + 4);
    const auto taken = encode_b(trampoline + 32, *target);
    const auto condition = compare
        ? encode_cbz(decoded.opcode == Opcode::cbnz, decoded.wide, value,
                     trampoline + 16, trampoline + 28)
        : encode_tbz(decoded.opcode == Opcode::tbnz, value, decoded.test_bit,
                     trampoline + 16, trampoline + 28);
    if (!enter.ok() || !resume.ok() || !taken.ok() || !condition.ok()) {
        return {.error = ScalarX18PatchError::unreachable_branch};
    }
    static_assert(kGuestRegisterCellX18Offset == 0);
    const auto read_tsd = encode_mrs(cell, system_register_encoding(3, 3, 13, 0, 3));
    constexpr auto save = 0xa9bf0000u | (cell << 10) | (31u << 5) | value;
    constexpr auto restore = 0xa8c10000u | (cell << 10) | (31u << 5) | value;
    ConditionalX18Patch patch{address, trampoline, instruction, enter.word, {}, value, cell};
    patch.trampoline = {
        save,
        read_tsd.word,
        0xf9400000u | ((slot / 8) << 10) | (cell << 5) | cell,
        0xf9400000u | (cell << 5) | value,
        condition.word,
        restore,
        resume.word,
        restore,
        taken.word,
    };
    return {.patch = patch};
}

} // namespace armigo::execution
