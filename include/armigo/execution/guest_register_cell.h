// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared storage for the guest registers that prepared code cannot keep in
// host hardware: the guest's x18 and its two EL0 thread-pointer registers.
//
// Apple reserves the host x18, and the host's TPIDR_EL0 and TPIDRRO_EL0 hold
// platform state, so none of them can carry guest values. A guest thread's
// values live instead in one GuestRegisterCell owned by its
// GuestThreadContext (guest_dispatch.h). Gate handlers and native code that
// is prepared to address the cell directly read and write the same three
// words; there is no second copy to synchronize.
//
// The cell is plain data with a fixed layout so that handwritten or emitted
// arm64 code can address its fields with immediate offsets. It defines no
// guest TLS layout and carries no other CPU state.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace armigo::execution {

struct alignas(16) GuestRegisterCell {
    std::uint64_t x18 = 0;                       // guest x18
    std::uint64_t thread_pointer = 0;            // guest TPIDR_EL0
    std::uint64_t read_only_thread_pointer = 0;  // guest TPIDRRO_EL0
};

// Byte offsets for code that addresses the cell through a base register.
inline constexpr std::size_t kGuestRegisterCellX18Offset = 0;
inline constexpr std::size_t kGuestRegisterCellThreadPointerOffset = 8;
inline constexpr std::size_t kGuestRegisterCellReadOnlyThreadPointerOffset = 16;
inline constexpr std::size_t kGuestRegisterCellSize = 32;
inline constexpr std::size_t kGuestRegisterCellAlignment = 16;

static_assert(std::is_standard_layout_v<GuestRegisterCell>);
static_assert(std::is_trivially_copyable_v<GuestRegisterCell>);
static_assert(offsetof(GuestRegisterCell, x18) == kGuestRegisterCellX18Offset);
static_assert(offsetof(GuestRegisterCell, thread_pointer) ==
              kGuestRegisterCellThreadPointerOffset);
static_assert(offsetof(GuestRegisterCell, read_only_thread_pointer) ==
              kGuestRegisterCellReadOnlyThreadPointerOffset);
static_assert(sizeof(GuestRegisterCell) == kGuestRegisterCellSize);
static_assert(alignof(GuestRegisterCell) == kGuestRegisterCellAlignment);

}  // namespace armigo::execution
