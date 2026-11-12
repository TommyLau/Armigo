// ============================================================================
// Armigo Platform - Android Guest Memory Mapping
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/3ds/android_memory.h"

extern "C" std::int32_t armigo_3ds_map_memory(std::uint32_t address, std::uint32_t size,
                                            std::uint32_t permissions) noexcept;

namespace armigo::n3ds {

HostMemoryMapping MapAndroidGuestMemory(void*, std::uint32_t address, std::uint32_t size,
                                       std::uint32_t permissions) noexcept {
    static_assert(sizeof(void*) == 4);
    const auto result = armigo_3ds_map_memory(address, size, permissions);
    if (static_cast<std::uint32_t>(result) >= 0xfffff001U) {
        return {0, -result};
    }
    return {static_cast<std::uint32_t>(result), 0};
}

} // namespace armigo::n3ds
