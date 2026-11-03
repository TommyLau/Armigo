// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Minimal checks shared by the AArch64 preparation tests. Each test program
// calls its checks from main() and returns finish().

#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace armigo::aarch64::test {

inline unsigned& failures() {
    static unsigned count = 0;
    return count;
}

inline bool check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        ++failures();
        std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    }
    return condition;
}

inline bool check_equal(std::uint64_t actual, std::uint64_t expected, const char* expression,
                        const char* file, int line) {
    if (actual != expected) {
        ++failures();
        std::fprintf(stderr, "FAIL %s:%d: %s: got 0x%" PRIx64 ", expected 0x%" PRIx64 "\n", file,
                     line, expression, actual, expected);
        return false;
    }
    return true;
}

inline int finish(const char* program) {
    if (failures() != 0) {
        std::fprintf(stderr, "%s: %u check(s) failed\n", program, failures());
        return 1;
    }
    std::printf("%s: all checks passed\n", program);
    return 0;
}

// Deterministic generator for broad but reproducible operand coverage.
struct Lcg {
    std::uint64_t state;
    std::uint64_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return state >> 11;
    }
};

} // namespace armigo::aarch64::test

#define CHECK(condition) ::armigo::aarch64::test::check((condition), #condition, __FILE__, __LINE__)
#define CHECK_EQ(actual, expected)                                                            \
    ::armigo::aarch64::test::check_equal(static_cast<std::uint64_t>(actual),                 \
                                         static_cast<std::uint64_t>(expected),               \
                                         #actual " == " #expected, __FILE__, __LINE__)
