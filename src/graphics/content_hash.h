// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Coherence certificates use a fast 64-bit content hash as a fallback proof.
// Equal hashes show that bytes are unchanged; they do not prove that no CPU
// store occurred and never waive registry version checks.

#pragma once

#include <cstring>
#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::graphics {

// Buffer caches certify page-aligned slices for CPU revalidation.
inline constexpr std::uint64_t revalidation_page_bytes = 4096;

inline std::uint64_t content_hash64(const std::byte* bytes, std::uint64_t size) noexcept {
    const auto* p = reinterpret_cast<const std::uint8_t*>(bytes);
    auto rotl = [](std::uint64_t x, int r) { return x << r | x >> (64 - r); };
    auto lane = [&rotl](std::uint64_t a, std::uint64_t w) {
        return rotl(a + w * 0xc2b2ae3d27d4eb4full, 31) * 0x9e3779b97f4a7c15ull;
    };
    std::uint64_t a = 0x60ea27eeadc0b5d6ull ^ size, b = 0xc2b2ae3d27d4eb4full,
                  c = 0x165667b19e3779f9ull, e = 0x27d4eb2f165667c5ull;
    auto read_word = [](const std::uint8_t* from) {
        std::uint64_t word = 0;
        std::memcpy(&word, from, sizeof word);
        return word;
    };
    for (; size >= 32; p += 32, size -= 32) {
        a = lane(a, read_word(p));
        b = lane(b, read_word(p + 8));
        c = lane(c, read_word(p + 16));
        e = lane(e, read_word(p + 24));
    }
    std::uint64_t h = rotl(a, 1) + rotl(b, 7) + rotl(c, 12) + rotl(e, 18);
    for (; size >= 8; p += 8, size -= 8) h = lane(h, read_word(p));
    if (size) {
        std::uint64_t word = 0;
        std::memcpy(&word, p, static_cast<std::size_t>(size));
        h = lane(h ^ size, word);
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

inline std::uint64_t content_hash64(std::span<const std::byte> bytes) noexcept {
    return content_hash64(bytes.data(), static_cast<std::uint64_t>(bytes.size()));
}

} // namespace armigo::graphics
