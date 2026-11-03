// ============================================================================
// Armigo Services - Shared Graphics Buffer Backing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace armigo::services {

// A retained mapping in one application's exported NvMap namespace. The owner
// pins the runtime backing and callback context; no host pointer is a guest or
// GPU address. Releasing a client handle does not invalidate this reference.
struct GraphicsBufferBacking {
    std::uint32_t export_id = 0;
    // Allocation metadata does not require the supplied CPU VA to share this alignment.
    std::uint32_t alignment = 0;
    std::uint64_t guest_address = 0;
    std::uint64_t size = 0;
    std::shared_ptr<void> owner;
};

// The provider is bound to the application's driver client. It returns only a
// live, exported, allocated object and keeps its mapping alive through owner.
using GraphicsBufferBackingLookup =
    std::function<std::optional<GraphicsBufferBacking>(std::uint32_t)>;

}  // namespace armigo::services
