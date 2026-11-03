// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <vector>

#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Represents an array of bits used for nodes and edges for the mixing graph.
struct BitArray {
    void reset() {
        buffer.assign(buffer.size(), false);
    }

    // Bits
    std::vector<bool> buffer{};
    // Size of the buffer
    u32 size{};
};

} // namespace armigo::audio::renderer
