// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/indirect_x18_patch.h>
#include <armigo/aarch64/instruction.h>
#include <armigo/execution/guest_register_cell.h>
#include <algorithm>
#include <limits>

namespace armigo::execution {

IndirectX18PatchResult make_indirect_x18_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint32_t slot, std::span<const IndirectX18Target> targets, ScalarX18Policy policy) {
    using namespace aarch64;
    using Error = IndirectX18PatchError;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (targets.empty()) return {.error = Error::missing_targets};
    // Four prologue words, eight per comparison, and restore/BRK on failure.
    if (targets.size() > (maximum / 4 - 6) / 8 ||
        targets.size() > (std::vector<std::uint32_t>{}.max_size() - 6) / 8) {
        return {.error = Error::invalid_layout};
    }
    const std::size_t word_count = 6 + 8 * targets.size();
    const std::uint64_t bytes = std::uint64_t{word_count} * 4;
    if (address % 4 != 0 || trampoline % 4 != 0 || address > maximum - 7 ||
        trampoline > maximum - (bytes - 1)) {
        return {.error = Error::invalid_layout};
    }
    const auto end = trampoline + bytes - 1;
    if (address <= end && trampoline <= address + 7) {
        return {.error = Error::overlapping_regions};
    }
    if (slot % 8 != 0 || slot > 4095 * 8) return {.error = Error::invalid_thread_slot};
    if (!policy.exclusive_monitor_clear || !policy.stack_scratch_available ||
        !policy.runtime_storage_disjoint) return {.error = Error::missing_execution_contract};
    const auto decoded = decode(instruction);
    if (decoded.opcode != Opcode::br && decoded.opcode != Opcode::blr) {
        return {.error = Error::unsupported_instruction};
    }
    if (decoded.reg != platform_register) return {.error = Error::register_not_used};
    std::vector<std::uint64_t> keys;
    keys.reserve(targets.size());
    for (const auto& target : targets) {
        if (target.guest_address == 0 || target.prepared_address == 0 ||
            target.guest_address % 4 != 0 || target.prepared_address % 4 != 0) {
            return {.error = Error::invalid_target};
        }
        if (target.prepared_address >= trampoline && target.prepared_address <= end) {
            return {.error = Error::overlapping_regions};
        }
        keys.push_back(target.guest_address);
    }
    std::sort(keys.begin(), keys.end());
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
        return {.error = Error::duplicate_target};
    }
    const auto enter = decoded.opcode == Opcode::blr ? encode_bl(address, trampoline)
                                                    : encode_b(address, trampoline);
    if (!enter.ok()) return {.error = Error::unreachable_branch};
    IndirectX18Patch patch{address, trampoline, instruction, enter.word, {}};
    auto& words = patch.trampoline;
    words.reserve(word_count);
    static_assert(kGuestRegisterCellX18Offset == 0);
    constexpr std::uint32_t save = 0xa9bf07e0u; // STP X0,X1,[SP,#-16]!
    constexpr std::uint32_t restore = 0xa8c107e0u; // LDP X0,X1,[SP],#16
    words.push_back(save);
    words.push_back(encode_mrs(1, system_register_encoding(3, 3, 13, 0, 3)).word);
    words.push_back(0xf9400021u | ((slot / 8) << 10)); // LDR X1,[X1,#slot]
    words.push_back(0xf9400020u); // LDR X0,[X1]
    for (const auto& target : targets) {
        for (unsigned half = 0; half != 4; ++half) {
            const auto part = static_cast<std::uint32_t>((target.guest_address >> (16 * half)) & 0xffffu);
            words.push_back((half == 0 ? 0xd2800000u : 0xf2800000u) |
                            (half << 21) | (part << 5) | 1u);
        }
        words.push_back(0xca010001u); // EOR X1,X0,X1; equality without changing NZCV.
        const auto condition_at = trampoline + 4 * words.size();
        const auto mismatch = encode_cbz(true, true, 1, condition_at, condition_at + 12);
        const auto transfer = encode_b(condition_at + 8, target.prepared_address);
        if (!mismatch.ok() || !transfer.ok()) return {.error = Error::unreachable_branch};
        words.push_back(mismatch.word);
        words.push_back(restore);
        words.push_back(transfer.word);
    }
    words.push_back(restore);
    words.push_back(0xd4200000u); // BRK #0: unmapped virtual target must never transfer.
    return {.patch = std::move(patch)};
}

IndirectX18PatchResult make_indirect_x18_table_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint32_t slot, std::uint64_t base, std::uint32_t entries, ScalarX18Policy policy) {
    using namespace aarch64;
    using Error = IndirectX18PatchError;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (entries == 0 || entries > 65536 || (entries & (entries - 1)) != 0) return {.error = Error::missing_targets};
    const std::size_t word_count = indirect_x18_table_words(entries);
    const std::uint64_t bytes = std::uint64_t{word_count} * 4;
    if (address % 4 != 0 || trampoline % 4 != 0 || address > maximum - 7 ||
        trampoline > maximum - (bytes - 1)) {
        return {.error = Error::invalid_layout};
    }
    const auto end = trampoline + bytes - 1;
    if (address <= end && trampoline <= address + 7) return {.error = Error::overlapping_regions};
    if (base == 0 || base % 4 != 0 || base > maximum - 4 * std::uint64_t{entries})
        return {.error = Error::invalid_target};
    if (base <= end && trampoline < base + 4 * std::uint64_t{entries})
        return {.error = Error::overlapping_regions};
    if (slot % 8 != 0 || slot > 4095 * 8) return {.error = Error::invalid_thread_slot};
    if (!policy.exclusive_monitor_clear || !policy.stack_scratch_available ||
        !policy.runtime_storage_disjoint) return {.error = Error::missing_execution_contract};
    const auto decoded = decode(instruction);
    if (decoded.opcode != Opcode::br) return {.error = Error::unsupported_instruction};
    if (decoded.reg != platform_register) return {.error = Error::register_not_used};
    const auto enter = encode_b(address, trampoline);
    if (!enter.ok()) return {.error = Error::unreachable_branch};
    IndirectX18Patch patch{address, trampoline, instruction, enter.word, {}};
    auto& words = patch.trampoline;
    words.reserve(word_count);
    static_assert(kGuestRegisterCellX18Offset == 0);
    constexpr std::uint32_t save = 0xa9bf07e0u;    // STP X0,X1,[SP,#-16]!
    constexpr std::uint32_t restore = 0xa8c107e0u; // LDP X0,X1,[SP],#16
    unsigned shift = 0;
    while ((std::uint64_t{1} << shift) < 4 * std::uint64_t{entries}) ++shift;
    words.push_back(save);
    words.push_back(encode_mrs(1, system_register_encoding(3, 3, 13, 0, 3)).word);
    words.push_back(0xf9400021u | ((slot / 8) << 10)); // LDR X1,[X1,#slot]
    words.push_back(0xf9400020u);                      // LDR X0,[X1]: virtual x18
    for (unsigned half = 0; half != 4; ++half) {       // X1 = table base
        const auto part = static_cast<std::uint32_t>((base >> (16 * half)) & 0xffffu);
        words.push_back((half == 0 ? 0xd2800000u : 0xf2800000u) | (half << 21) | (part << 5) | 1u);
    }
    // None of the bound checks changes NZCV.
    words.push_back(0xcb010000u);                           // SUB X0,X0,X1: byte offset
    words.push_back(0xd340fc01u | (shift << 16));           // LSR X1,X0,#shift: outside the table
    const auto fail = trampoline + 4 * (word_count - 2);
    const auto outside = encode_cbz(true, true, 1, trampoline + 4 * words.size(), fail);
    if (!outside.ok()) return {.error = Error::unreachable_branch};
    words.push_back(outside.word);                          // CBNZ X1,fail
    words.push_back(0x92400401u);                           // AND X1,X0,#3: misaligned
    const auto misaligned = encode_cbz(true, true, 1, trampoline + 4 * words.size(), fail);
    if (!misaligned.ok()) return {.error = Error::unreachable_branch};
    words.push_back(misaligned.word);                       // CBNZ X1,fail
    words.push_back(0x10000061u);                           // ADR X1,table (three words on)
    words.push_back(0x8b000421u);                           // ADD X1,X1,X0,LSL #1: 8-byte entries
    words.push_back(0xd61f0020u);                           // BR X1
    for (std::uint32_t index = 0; index != entries; ++index) {
        words.push_back(restore);
        const auto transfer = encode_b(trampoline + 4 * words.size(), base + 4 * std::uint64_t{index});
        if (!transfer.ok()) return {.error = Error::unreachable_branch};
        words.push_back(transfer.word);
    }
    words.push_back(restore);
    // A target outside the table must never transfer. The immediate names this
    // trap in a crash report, apart from the preparation's stop words.
    words.push_back(encode_brk(indirect_x18_table_trap).word);
    return {.patch = std::move(patch)};
}

} // namespace armigo::execution
