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

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace armigo::execution {

struct LocalGateTemplate {
    std::vector<std::byte> code;
    std::size_t dispatcher_offset = 0;
};

// Returns the relocatable gate bytes with an all-zero dispatcher binding.
// This representation is deterministic and safe to persist. The offset names
// one aligned uintptr_t-sized field in bytes.
LocalGateTemplate make_local_gate_template();

// Binds an explicitly supplied dispatcher into a copied template. The whole
// field and alignment are checked before any byte is changed.
bool bind_local_gate_dispatcher(std::span<std::byte> bytes, std::size_t dispatcher_offset,
                                std::uintptr_t dispatcher) noexcept;

// Address used by this process's ordinary local gate. Offline preparation
// must persist the unbound template rather than this process-specific value.
std::uintptr_t gate_dispatcher_address() noexcept;

// Instantiates an owned copy of the native gate with a dispatcher binding for
// this process. The entry is byte zero. The code can be placed anywhere on an
// aligned executable page; it has no external PC-relative branch requirement.
//
// The caller must copy the returned bytes into writable, nonexecutable storage,
// seal it and invalidate instruction caches before entry. The gate obeys the
// GateScope/GateCall ABI, including 16-byte stack alignment and no live data
// below sp. Entering it still requires an installed GateScope.
//
// The result is process-specific and must not be persisted or shared. It is not
// an iOS code-signing solution or proof that arbitrary guest code is prepared.
// Throws std::bad_alloc on allocation failure and std::logic_error if linked
// template labels do not describe a valid layout.
std::vector<std::byte> make_local_gate_code();

} // namespace armigo::execution
