// ============================================================================
// Armigo Tests - Process Memory Coverage Query
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/horizon/process_memory.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using namespace armigo::horizon;
namespace pm = armigo::horizon::process_memory;

// The definition the production query must agree with: walk every block in order.
bool reference_covered(const std::vector<MemoryBlock>& blocks, AddressRange range, std::uint8_t permissions) {
    if (!pm::valid_range(range)) return false;
    auto cursor = range.address;
    for (const auto& b : blocks) {
        if (!pm::overlaps(b.range, range)) continue;
        if (b.range.address > cursor || b.type == MemoryType::free ||
            (b.permissions & permissions) != permissions) return false;
        cursor = std::min(range.address + range.size, b.range.address + b.range.size);
        if (cursor == range.address + range.size) return true;
    }
    return false;
}
}  // namespace

int main() {
    std::mt19937_64 rng(0x5eed);
    unsigned failures = 0, true_results = 0, false_results = 0;
    for (unsigned map = 0; map < 400; ++map) {
        // An ordered, non-overlapping map with gaps, free blocks and mixed permissions.
        std::vector<MemoryBlock> blocks;
        std::uint64_t at = 0x200000 + (rng() % 64) * 0x1000;
        const unsigned count = 1 + rng() % 60;
        for (unsigned i = 0; i < count; ++i) {
            if (rng() % 4 == 0) at += (1 + rng() % 8) * 0x1000;  // gap
            MemoryBlock b;
            b.range = {at, (1 + rng() % 16) * 0x1000};
            b.type = rng() % 5 == 0 ? MemoryType::free : MemoryType::code_data;
            b.permissions = static_cast<std::uint8_t>(rng() % 8);
            at += b.range.size;
            blocks.push_back(b);
        }
        const std::uint64_t end = at;
        for (unsigned q = 0; q < 400; ++q) {
            AddressRange range;
            switch (rng() % 6) {
            case 0: range = {0, 4}; break;
            case 1: range = {end - 4, 4 + rng() % 16}; break;
            case 2: range = {end + rng() % 0x10000, 1 + rng() % 8}; break;
            default: range = {0x200000 + rng() % (end - 0x200000 + 1), 1 + rng() % 0x6000}; break;
            }
            const std::uint8_t permissions = static_cast<std::uint8_t>(rng() % 4);
            const bool expected = reference_covered(blocks, range, permissions);
            const bool actual = pm::covered(blocks, range, permissions);
            (expected ? true_results : false_results)++;
            if (expected != actual) {
                if (failures++ < 5)
                    std::fprintf(stderr, "covered mismatch: range %llx+%llx perms %u expected %d\n",
                                 (unsigned long long)range.address, (unsigned long long)range.size,
                                 permissions, expected);
            }
        }
    }
    std::fprintf(stderr, "covered queries: %u true, %u false, %u mismatches\n", true_results, false_results, failures);
    // Empty map and an unaligned single-word range.
    if (pm::covered({}, {0x200000, 4}, 0)) ++failures;
    return failures ? 1 : 0;
}
