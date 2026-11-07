// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_write_ranges.h"

#include <array>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace {

using armigo::platform::metal::nvn_driver::WriteRanges;
using Range = std::pair<std::uint64_t, std::uint64_t>;
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #value); ++failures; \
} } while (false)

bool legacy_overlap(const Range& write, const Range& read) {
    return read.first < write.first + write.second &&
           write.first < read.first + read.second;
}

void test_query_holes_and_aliases() {
    WriteRanges index;
    CHECK(!index.may_intersect(0, 64));
    // Descending insertion prevents this from testing only an append path.
    for (unsigned i = 1024; i-- != 0;) index.add(0x10000 + i * 32, 16);
    for (unsigned i = 0; i != 1024; ++i) {
        const auto address = 0x10000 + i * 32;
        CHECK(index.may_intersect(address, 8));
        CHECK(index.intersection(address, 8).value_or(false));
        CHECK(index.may_intersect(address + 8, 8));
        CHECK(index.may_intersect(address + 15, 2));
        CHECK(!index.may_intersect(address + 16, 8));
        CHECK(!index.may_intersect(address + 24, 8));
    }
    // An event writes part of a formerly empty reference operand.
    index.add(0x10012, 4);
    CHECK(index.may_intersect(0x10010, 8));
    CHECK(!index.may_intersect(0x10016, 8));
    // A covering alias merges earlier ranges without losing either endpoint.
    index.add(0x0ffff, 0x9000);
    CHECK(index.may_intersect(0x0ffff, 1));
    CHECK(index.may_intersect(0x18ffe, 1));
    CHECK(!index.may_intersect(0x18fff, 1));
    index.clear();
    index.add(0x80000, 16);
    CHECK(!index.may_intersect(0x10000, 16));
    CHECK(index.may_intersect(0x80008, 8));
}

void test_conservative_edge_cases() {
    WriteRanges index;
    index.add(100, 0);
    CHECK(index.may_intersect(99, 2));
    CHECK(!index.intersection(100, 1).has_value());
    index.add(200, 16);
    CHECK(index.may_intersect(208, 0));
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    CHECK(index.may_intersect(max - 1, 4));
    index.add(max - 3, 8);
    CHECK(index.may_intersect(0, 1));
    index.clear();
    for (unsigned i = 0; i != 4097; ++i) index.add(i * 32, 16);
    CHECK(index.may_intersect(0x100000, 1));
    index.clear();
    CHECK(!index.may_intersect(0x100000, 1));
}

void test_against_byte_range_oracle() {
    std::uint32_t random = 0x729c34a1;
    const auto next = [&] {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        return random;
    };
    for (unsigned trial = 0; trial != 100; ++trial) {
        WriteRanges index;
        std::vector<Range> writes;
        for (unsigned count = 0; count != 128; ++count) {
            const Range write{next() % 8192, (trial % 3 ? 1 : 0) + next() % 65};
            writes.push_back(write);
            index.add(write.first, write.second);
            for (unsigned query = 0; query != 128; ++query) {
                const Range read{next() % 8192, next() % 65};
                bool overlap = false;
                for (const auto& candidate : writes) overlap |= legacy_overlap(candidate, read);
                CHECK(!overlap || index.may_intersect(read.first, read.second));
                if (const auto exact = index.intersection(read.first, read.second))
                    CHECK(*exact == overlap);
            }
        }
    }
}

} // namespace

int main() {
    test_query_holes_and_aliases();
    test_conservative_edge_cases();
    test_against_byte_range_oracle();
    std::printf("NVN write-range failures: %u\n", failures);
    return failures ? 1 : 0;
}
