// ============================================================================
// Armigo Platform - Android Guest Memory Mapping
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "armigo/3ds/guest_memory.h"

namespace armigo::n3ds {

// Available in the Android AArch32 target. Creates an exact anonymous mapping.
HostMemoryMapping MapAndroidGuestMemory(void* context, std::uint32_t address,
                                       std::uint32_t size,
                                       std::uint32_t permissions) noexcept;

} // namespace armigo::n3ds
