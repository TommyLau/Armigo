// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>

namespace armigo::platform::metal::nvn_driver {

// Captured under one registration lock. A zero descriptor key is an empty
// ID; its empty epoch survives a later occupied replacement.
struct TextureReadIdentity {
    std::uint64_t descriptor_key = 0;
    std::uint64_t serial = 0;
    std::uint64_t storage_key = 0;
    std::uint64_t empty_epoch = 0;
    bool depth = false;
};

} // namespace armigo::platform::metal::nvn_driver
