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
#include <vector>

namespace armigo::execution {

inline constexpr std::size_t embedded_key_page_size = 16384;
inline constexpr std::size_t embedded_thread_slot_offset = 0;
inline constexpr std::size_t embedded_gate_dispatcher_offset = 8;
inline constexpr std::size_t embedded_nvn_bootstrap_offset = 16;

struct EmbeddedPatch {
    std::uint32_t replacement_instruction = 0;
    std::vector<std::uint32_t> trampoline;
};

// Addresses share a module-relative planning namespace with a page-aligned
// base. key_slot names an aligned 64-bit word containing the runtime TSD byte
// offset, not the cell pointer. The slot is immutable during guest execution.
// All functions retain the ordinary builders' contracts and additionally
// require the key slot to be outside executable code and guest storage.
[[nodiscard]] std::optional<EmbeddedPatch> make_embedded_scalar_x18_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key_slot, ScalarX18Policy policy);
[[nodiscard]] std::optional<EmbeddedPatch> make_embedded_conditional_x18_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key_slot, ScalarX18Policy policy);
[[nodiscard]] std::optional<EmbeddedPatch> make_embedded_indirect_x18_table_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key_slot, std::uint64_t table_base, std::uint32_t entries,
    ScalarX18Policy policy);
[[nodiscard]] std::optional<EmbeddedPatch> make_embedded_thread_register_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key_slot, ScalarX18Policy policy);

// The gate preserves the ordinary GateScope ABI. dispatcher_key names a
// separate aligned RW word containing gate_dispatcher_address() at runtime.
[[nodiscard]] std::optional<std::vector<std::byte>> make_embedded_gate_code(
    std::uint64_t gate_address, std::uint64_t dispatcher_key);

// Tail-transfer to a host ABI entry through a key-page pointer. Clobbers only
// x16, preserving LR so the target returns directly to the original caller.
[[nodiscard]] std::optional<EmbeddedPatch> make_embedded_host_branch(
    std::uint64_t address, std::uint64_t trampoline, std::uint64_t target_key);

} // namespace armigo::execution
