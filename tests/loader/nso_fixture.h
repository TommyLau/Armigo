// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Builder for independently authored synthetic NSO0 files. It writes only
// the fields the reader consumes; every value is chosen by the test.

#pragma once

#include "sha256.h"
#include "test_support.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace armigo::loader::test {

struct FixtureSegment {
    std::vector<std::byte> content;
    std::uint32_t image_offset = 0;
    bool compress = false;
    bool verify_hash = false;
};

struct FixtureSpec {
    std::array<FixtureSegment, 3> segments;
    std::uint32_t bss_size = 0;
    std::array<std::array<std::uint32_t, 2>, 3> rodata_extents{};
};

inline void put_u32(std::vector<std::byte>& file, std::size_t offset, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        file[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xFF);
    }
}

inline std::uint32_t get_u32(const std::vector<std::byte>& file, std::size_t offset) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(file[offset + i]) << (8 * i);
    }
    return value;
}

// Header field offsets used by tests that corrupt specific fields.
inline constexpr std::size_t kFieldVersion = 0x04;
inline constexpr std::size_t kFieldFlags = 0x0C;
inline constexpr std::size_t field_file_offset(std::size_t segment) { return 0x10 + 0x10 * segment; }
inline constexpr std::size_t field_image_offset(std::size_t segment) { return 0x14 + 0x10 * segment; }
inline constexpr std::size_t field_size(std::size_t segment) { return 0x18 + 0x10 * segment; }
inline constexpr std::size_t kFieldBssSize = 0x3C;
inline constexpr std::size_t field_file_size(std::size_t segment) { return 0x60 + 4 * segment; }
inline constexpr std::size_t field_extent(std::size_t extent) { return 0x88 + 8 * extent; }
inline constexpr std::size_t field_hash(std::size_t segment) { return 0xA0 + 0x20 * segment; }

inline std::vector<std::byte> build_nso(const FixtureSpec& spec) {
    std::vector<std::byte> file(0x100);
    file[0] = std::byte{'N'};
    file[1] = std::byte{'S'};
    file[2] = std::byte{'O'};
    file[3] = std::byte{'0'};
    std::uint32_t flags = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const FixtureSegment& segment = spec.segments[i];
        const std::vector<std::byte> stored =
            segment.compress ? lz4_encode(segment.content) : segment.content;
        put_u32(file, field_file_offset(i), static_cast<std::uint32_t>(file.size()));
        put_u32(file, field_image_offset(i), segment.image_offset);
        put_u32(file, field_size(i), static_cast<std::uint32_t>(segment.content.size()));
        put_u32(file, field_file_size(i), static_cast<std::uint32_t>(stored.size()));
        const auto digest = detail::sha256(segment.content);
        std::memcpy(file.data() + field_hash(i), digest->data(), digest->size());
        flags |= (segment.compress ? 1u : 0u) << i;
        flags |= (segment.verify_hash ? 1u : 0u) << (3 + i);
        file.insert(file.end(), stored.begin(), stored.end());
    }
    put_u32(file, kFieldFlags, flags);
    put_u32(file, kFieldBssSize, spec.bss_size);
    for (std::size_t i = 0; i < 3; ++i) {
        put_u32(file, field_extent(i), spec.rodata_extents[i][0]);
        put_u32(file, field_extent(i) + 4, spec.rodata_extents[i][1]);
    }
    return file;
}

// Content with long repeats and short-period runs so that compressed
// fixtures exercise overlapping match copies.
inline std::vector<std::byte> patterned(std::size_t size, std::uint8_t seed) {
    std::vector<std::byte> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t block = i / 97;
        if (block % 3 == 0) {
            data[i] = static_cast<std::byte>(seed + (i % 3));
        } else if (block % 3 == 1) {
            data[i] = static_cast<std::byte>(seed ^ static_cast<std::uint8_t>(i * 31 + block));
        } else {
            data[i] = static_cast<std::byte>(seed + 0x40);
        }
    }
    return data;
}

// A small valid layout: text 0x1800 bytes, rodata 0x900 bytes, data 0x400
// bytes followed by 0x2345 bytes of BSS.
inline FixtureSpec default_spec(bool compress, bool verify_hash) {
    FixtureSpec spec;
    spec.segments[0] = {patterned(0x1800, 0x11), 0x0000, compress, verify_hash};
    spec.segments[1] = {patterned(0x900, 0x22), 0x2000, compress, verify_hash};
    spec.segments[2] = {patterned(0x400, 0x33), 0x3000, compress, verify_hash};
    spec.bss_size = 0x2345;
    spec.rodata_extents = {{{0x10, 0x20}, {0x100, 0x80}, {0x200, 0x300}}};
    return spec;
}

} // namespace armigo::loader::test
