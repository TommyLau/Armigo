// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared helpers for the loader tests: a minimal check facility and a
// deliberately simple greedy LZ4 block encoder used only to produce
// synthetic compressed fixtures. The encoder favors clarity over ratio.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <span>
#include <vector>

namespace armigo::loader::test {

inline int g_failures = 0;

inline void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
        ++g_failures;
    }
}

#define ARMIGO_CHECK(expression) \
    ::armigo::loader::test::check((expression), #expression, __FILE__, __LINE__)

inline int finish(const char* suite) {
    if (g_failures != 0) {
        std::fprintf(stderr, "%s: %d check(s) failed\n", suite, g_failures);
        return 1;
    }
    std::printf("%s: all checks passed\n", suite);
    return 0;
}

inline std::vector<std::byte> bytes_of(std::initializer_list<int> values) {
    std::vector<std::byte> result;
    result.reserve(values.size());
    for (int value : values) {
        result.push_back(static_cast<std::byte>(value));
    }
    return result;
}

inline void append_length(std::vector<std::byte>& out, std::size_t remainder) {
    while (remainder >= 255) {
        out.push_back(std::byte{0xFF});
        remainder -= 255;
    }
    out.push_back(static_cast<std::byte>(remainder));
}

inline void append_sequence(std::vector<std::byte>& out, std::span<const std::byte> literals,
                            std::size_t match_offset, std::size_t match_length) {
    const std::size_t literal_nibble = literals.size() < 15 ? literals.size() : 15;
    std::size_t match_nibble = 0;
    if (match_length != 0) {
        const std::size_t encoded = match_length - 4;
        match_nibble = encoded < 15 ? encoded : 15;
    }
    out.push_back(static_cast<std::byte>((literal_nibble << 4) | match_nibble));
    if (literal_nibble == 15) {
        append_length(out, literals.size() - 15);
    }
    out.insert(out.end(), literals.begin(), literals.end());
    if (match_length != 0) {
        out.push_back(static_cast<std::byte>(match_offset & 0xFF));
        out.push_back(static_cast<std::byte>(match_offset >> 8));
        if (match_nibble == 15) {
            append_length(out, match_length - 4 - 15);
        }
    }
}

// Greedy encoder: at each position, searches a bounded window for the
// longest earlier match (overlap allowed) and emits it when at least four
// bytes long. The final five bytes are always emitted as literals.
inline std::vector<std::byte> lz4_encode(std::span<const std::byte> input,
                                         std::size_t window = 4096) {
    std::vector<std::byte> out;
    std::size_t anchor = 0;
    std::size_t position = 0;
    const std::size_t match_limit = input.size() > 5 ? input.size() - 5 : 0;
    while (position < match_limit) {
        std::size_t best_length = 0;
        std::size_t best_offset = 0;
        const std::size_t first = position > window ? position - window : 0;
        const std::size_t first_bounded = position > 65535 ? std::max(first, position - 65535) : first;
        for (std::size_t candidate = first_bounded; candidate < position; ++candidate) {
            std::size_t length = 0;
            while (position + length < match_limit && input[candidate + length] == input[position + length]) {
                ++length;
            }
            if (length > best_length) {
                best_length = length;
                best_offset = position - candidate;
            }
        }
        if (best_length >= 4) {
            append_sequence(out, input.subspan(anchor, position - anchor), best_offset, best_length);
            position += best_length;
            anchor = position;
        } else {
            ++position;
        }
    }
    append_sequence(out, input.subspan(anchor), 0, 0);
    return out;
}

} // namespace armigo::loader::test
