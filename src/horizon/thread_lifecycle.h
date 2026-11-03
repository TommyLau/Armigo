// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>
#include <optional>

namespace armigo::horizon {

// Pure firmware-20.5.0 argument normalization. Object lookup precedes the
// SetThreadCoreMask checks. These helpers neither schedule nor mutate a thread.
struct ThreadCoreRequest {
    std::uint32_t result = 0;
    std::int32_t ideal_core = 0;
    std::uint64_t affinity_mask = 0;
};
ThreadCoreRequest resolve_thread_creation_core(std::int32_t requested,
    std::int32_t process_default, std::uint64_t allowed) noexcept;
ThreadCoreRequest resolve_thread_core_mask(std::int32_t requested, std::uint64_t mask,
    std::int32_t previous_ideal, std::int32_t process_default,
    std::uint64_t allowed) noexcept;

// Runtime policy: prefer an explicit ideal, otherwise retain an admitted
// current assignment, otherwise select the lowest admitted virtual processor.
// This is virtual identity, not host CPU affinity or scheduling exclusivity.
std::optional<std::uint32_t> select_virtual_processor(std::int32_t ideal,
    std::uint64_t mask, std::optional<std::uint32_t> current = {}) noexcept;

} // namespace armigo::horizon
