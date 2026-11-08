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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace armigo::platform::metal::nvn_driver {

// Metal has no quads or triangle fans; both draw as triangle lists. Each
// quad's indices a, b, c, d become a, b, c and a, c, d, and a trailing partial
// quad draws nothing; a fan's v0, v1, ... become v0, vi, vi+1. A restart index
// ends the current quad or fan. The pieces between restarts join into one
// list, which needs no restart of its own.

// The most indices expand_triangles writes for `count` input indices.
constexpr std::size_t expanded_capacity(bool fan, std::size_t count) {
    return fan ? (count >= 3 ? (count - 2) * 3 : 0) : count / 4 * 6;
}

// The most indices one Metal draw takes: its index count is 32 bits. A fan
// of more than 1,431,655,767 vertices expands past it.
inline constexpr std::size_t max_expanded_indices = std::numeric_limits<std::uint32_t>::max();

// expand_triangles' result when an index other than the restart index is
// UINT32_MAX, the value Metal itself restarts on.
inline constexpr std::size_t expansion_collision = std::numeric_limits<std::size_t>::max();

// Reads `count` little-endian indices of `element_size` bytes (1, 2 or 4) from
// `source`, or the sequence first, first + 1, ... when `source` is null, and
// writes the triangles to `out`, which holds expanded_capacity(fan, count)
// entries. Returns the number of indices written.
inline std::size_t expand_triangles(std::uint32_t* out, const std::uint8_t* source, unsigned element_size,
                                    std::uint32_t first, std::size_t count, bool fan, bool restart,
                                    std::uint32_t restart_index) {
    std::size_t used = 0;
    unsigned have = 0;
    std::uint32_t corner[4]{};
    for (std::size_t position = 0; position != count; ++position) {
        std::uint32_t index = first + static_cast<std::uint32_t>(position);
        if (source) {
            index = 0;
            std::memcpy(&index, source + position * element_size, element_size);
        }
        if (restart && index == restart_index) {
            have = 0;
            continue;
        }
        if (index == std::numeric_limits<std::uint32_t>::max()) return expansion_collision;
        if (fan) {
            // corner[0] is the hub, corner[1] the previous vertex.
            if (!have) corner[0] = index;
            else if (have == 2) {
                out[used++] = corner[0];
                out[used++] = corner[1];
                out[used++] = index;
            }
            corner[1] = index;
            if (have < 2) ++have;
        } else {
            corner[have++] = index;
            if (have == 4) {
                out[used++] = corner[0];
                out[used++] = corner[1];
                out[used++] = corner[2];
                out[used++] = corner[0];
                out[used++] = corner[2];
                out[used++] = corner[3];
                have = 0;
            }
        }
    }
    return used;
}

} // namespace armigo::platform::metal::nvn_driver
