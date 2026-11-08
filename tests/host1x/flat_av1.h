// ============================================================================
// Armigo Tests - Synthetic Fixtures
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

namespace armigo::test {

// A 64x64 flat-colour (RGB 0x50a050) clip of three frames, encoded with
// SVT-AV1 as low-overhead OBUs: a temporal delimiter starts each temporal unit.
inline constexpr std::uint8_t flat_av1[] = {
    0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x00, 0x40, 0x32,
    0x13, 0x10, 0x00, 0xb0, 0x82, 0x05, 0x14, 0x20, 0x81, 0x00, 0x00, 0x03, 0x2b, 0x91, 0x22, 0x1f,
    0x0d, 0x67, 0x77, 0x40, 0x12, 0x00, 0x32, 0x11, 0x28, 0x02, 0x00, 0x40, 0x00, 0x00, 0x11, 0x8c,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x9c, 0x10, 0x32, 0x11, 0x30, 0x02, 0x00, 0x00, 0x00,
    0x49, 0x23, 0x18, 0x00, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x9c, 0xe8, 0x12, 0x00, 0x1a, 0x01,
    0x98,
};

// The clip's temporal units without their temporal delimiters, as an MP4
// sample holds them.
inline std::vector<std::vector<std::uint8_t>> flat_av1_temporal_units() {
    const std::span<const std::uint8_t> stream(flat_av1);
    std::vector<std::vector<std::uint8_t>> units;
    for (std::size_t offset = 0; offset < stream.size();) {
        const auto header = stream[offset];
        std::size_t cursor = offset + 1 + ((header >> 2) & 1);
        std::size_t size = 0;
        for (unsigned shift = 0;; shift += 7) {
            const auto byte = stream[cursor++];
            size |= std::size_t{byte & 0x7fu} << shift;
            if (!(byte & 0x80)) break;
        }
        if (((header >> 3) & 15) == 2) units.emplace_back();
        else units.back().insert(units.back().end(), stream.begin() + static_cast<std::ptrdiff_t>(offset),
                                 stream.begin() + static_cast<std::ptrdiff_t>(cursor + size));
        offset = cursor + size;
    }
    return units;
}

} // namespace armigo::test
