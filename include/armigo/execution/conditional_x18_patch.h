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

namespace armigo::execution {

struct ConditionalX18Patch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    std::array<std::uint32_t, 9> trampoline{};
    std::uint8_t value_register = 0;
    std::uint8_t cell_register = 0;
};

struct ConditionalX18PatchResult {
    ScalarX18PatchError error = ScalarX18PatchError::none;
    std::optional<ConditionalX18Patch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Plans CBZ/CBNZ/TBZ/TBNZ reading virtual W18/X18 through a probed native TSD
// slot. The correct GuestRegisterCell must be bound throughout execution.
// Both outcomes restore scratch registers and SP before branching to the
// original taken target or site+4. NZCV, x30 and the virtual cell are unchanged;
// hardware x18 is never used. All nine emitted words are ordinary native code.
//
// The ScalarX18Policy contract applies: a clear exclusive monitor, 16 bytes
// of aligned writable private stack scratch, and disjoint runtime storage.
// The caller owns a guarded stack with adequate headroom, proves both original
// destinations prepared, verifies preimages and the whole preparation plan,
// writes only nonexecutable code, and seals afterward. Neither the original
// site/continuation nor the taken target may overlap injected storage.
//
// Addresses share one namespace. Only ordinary completion is modeled;
// exceptions, asynchronous reentry and BTI/GCS/MTE policies need separate work.
ConditionalX18PatchResult make_conditional_x18_patch(
    std::uint32_t instruction, std::uint64_t instruction_address,
    std::uint64_t trampoline_address, std::uint32_t tsd_byte_offset,
    ScalarX18Policy policy = {}) noexcept;

} // namespace armigo::execution
