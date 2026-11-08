// ============================================================================
// Armigo Tests - NVN Guest Memory Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_guest_memory.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

using armigo::platform::metal::nvn_driver::GuestMemoryTable;
using armigo::platform::metal::nvn_driver::GuestMemoryWindow;
using armigo::platform::metal::nvn_driver::build_guest_memory_table;
using armigo::platform::metal::nvn_driver::guest_memory_max_entries;
using armigo::platform::metal::nvn_driver::guest_memory_table_bytes;
using armigo::platform::metal::nvn_driver::guest_memory_write_log_bytes;
using armigo::platform::metal::nvn_driver::lookup_guest_memory;
using armigo::platform::metal::nvn_driver::visit_guest_memory_writes;

unsigned failures = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);      \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

constexpr std::uint64_t kChunk = 1ull << 21;

std::uint64_t pair(const std::vector<std::uint32_t>& words, std::size_t entry, std::size_t word) {
    const std::size_t index = 8 + entry * 8 + word;
    return words[index] | (std::uint64_t{words[index + 1]} << 32);
}

std::uint32_t hint_of(const std::vector<std::uint32_t>& words, std::size_t entry) {
    return words[8 + entry * 8 + 6];
}

std::uint32_t partial_of(const std::vector<std::uint32_t>& words, std::size_t entry) {
    return words[8 + entry * 8 + 7];
}

bool device_is(std::optional<std::uint64_t> found, std::uint64_t device) {
    return found && *found == device;
}

} // namespace

int main() {
    // Unsorted input. Empty and contained windows drop. The partial overlap is
    // trimmed, and a device address below the guest start wraps the delta.
    const GuestMemoryWindow mixed[] = {
        {100, 50, 1},
        {200, 200, 2},
        {5000, 5016, 100},
        {1000, 1300, 111},
        {1000, 2000, 5000},
        {1200, 1500, 9999},
        {1800, 3000, 8000},
        {4000, 4500, 7000},
    };
    auto mixed_table = build_guest_memory_table(mixed);
    CHECK(mixed_table.has_value());
    if (mixed_table) {
        const auto& words = mixed_table->words;
        CHECK(mixed_table->dropped == 4 && mixed_table->trimmed == 1);
        CHECK(mixed_table->entries == 4 && words.size() == guest_memory_table_bytes / 4);
        CHECK(words[0] == 5);
        for (std::size_t word = 1; word < 8; ++word) CHECK(words[word] == 0);
        CHECK(pair(words, 0, 0) == 0 && pair(words, 0, 2) == 0);
        CHECK(pair(words, 1, 0) == 1000 && pair(words, 1, 2) == 2000);
        CHECK(pair(words, 1, 4) == 5000 - 1000);
        CHECK(pair(words, 2, 0) == 2000 && pair(words, 2, 2) == 3000);
        CHECK(pair(words, 2, 4) == 8000 - 1800);
        CHECK(pair(words, 3, 0) == 4000 && pair(words, 3, 2) == 4500);
        CHECK(pair(words, 4, 0) == 5000 && pair(words, 4, 2) == 5016);
        CHECK(pair(words, 4, 4) == std::uint64_t{100} - 5000);
        for (std::size_t entry = 0; entry < guest_memory_max_entries; ++entry) {
            CHECK(hint_of(words, entry) == 0);
            if (entry < 5) continue;
            CHECK(pair(words, entry, 0) == ~0ull && pair(words, entry, 2) == ~0ull);
            CHECK(pair(words, entry, 4) == 0);
        }
        CHECK(device_is(lookup_guest_memory(words, 1000, 4), 5000));
        CHECK(device_is(lookup_guest_memory(words, 1800, 4), 5800));
        CHECK(device_is(lookup_guest_memory(words, 2000, 4), 8200));
        CHECK(device_is(lookup_guest_memory(words, 2500, 4), 8700));
        CHECK(device_is(lookup_guest_memory(words, 4000, 4), 7000));
        CHECK(device_is(lookup_guest_memory(words, 5000, 4), 100));
        CHECK(device_is(lookup_guest_memory(words, 5008, 8), 108));
        CHECK(!lookup_guest_memory(words, 0, 1));
        CHECK(!lookup_guest_memory(words, 996, 4));
        CHECK(!lookup_guest_memory(words, 3000, 1));
        CHECK(!lookup_guest_memory(words, 3600, 4));
        CHECK(!lookup_guest_memory(words, 4500, 1));
        CHECK(!lookup_guest_memory(words, 5016, 1));
        CHECK(!lookup_guest_memory(words, 1001, 4));
        CHECK(!lookup_guest_memory(words, 5008, 16));
    }

    // Chunks wholly inside an entry get a hint. A chunk that crosses either
    // edge does not. Entry 0 can carry a hint and stays an empty range.
    const GuestMemoryWindow hinted[] = {
        {0, 2 * kChunk, 0x100000000ull},
        {10 * kChunk + 100, 12 * kChunk + 100, 0x200000000ull},
    };
    auto hint_table = build_guest_memory_table(hinted);
    CHECK(hint_table && hint_table->entries == 2 && hint_table->hint_collisions == 0);
    if (hint_table) {
        const auto& words = hint_table->words;
        CHECK(hint_of(words, 0) == 1 && pair(words, 0, 0) == 0 && pair(words, 0, 2) == 0);
        CHECK(hint_of(words, 1) == ((1u << 13) | 1u));
        CHECK(hint_of(words, 2) == 0 && hint_of(words, 10) == 0 && hint_of(words, 12) == 0);
        CHECK(hint_of(words, 11) == ((11u << 13) | 2u));
        CHECK(device_is(lookup_guest_memory(words, 0, 16), 0x100000000ull));
        CHECK(device_is(lookup_guest_memory(words, kChunk, 4), 0x100000000ull + kChunk));
        const std::uint64_t partial = 10 * kChunk + 100;
        CHECK(device_is(lookup_guest_memory(words, partial, 4), 0x200000000ull));
        CHECK(device_is(lookup_guest_memory(words, 11 * kChunk, 4),
                        0x200000000ull + (11 * kChunk - partial)));
        CHECK(!lookup_guest_memory(words, partial - 4, 4));
    }

    // A chunk that entries start or end inside names the first entry
    // overlapping it: here entry 1 ends in chunk 5, entries 2 and 3 lie in
    // it, and entry 4 starts in it and ends in chunk 7. Chunk 6 lies wholly
    // inside entry 4 and keeps only its hint.
    const GuestMemoryWindow shared[] = {
        {4 * kChunk + 100, 5 * kChunk + 0x300, 0x10000000ull},
        {5 * kChunk + 0x400, 5 * kChunk + 0x500, 0x20000000ull},
        {5 * kChunk + 0x800, 5 * kChunk + 0x900, 0x30000000ull},
        {5 * kChunk + 0x1000, 7 * kChunk + 0x40, 0x40000000ull},
    };
    auto shared_table = build_guest_memory_table(shared);
    CHECK(shared_table && shared_table->entries == 4 && shared_table->partial_collisions == 0);
    if (shared_table) {
        const auto& words = shared_table->words;
        CHECK(partial_of(words, 4) == ((4u << 13) | 1u));
        CHECK(partial_of(words, 5) == ((5u << 13) | 1u));
        CHECK(partial_of(words, 6) == 0 && hint_of(words, 6) == ((6u << 13) | 4u));
        CHECK(partial_of(words, 7) == ((7u << 13) | 4u));
        CHECK(device_is(lookup_guest_memory(words, 5 * kChunk + 0x880, 4), 0x30000080ull));
    }

    // A pool holding exactly one aligned chunk takes the chunk hint only.
    const GuestMemoryWindow aligned[] = {{kChunk, 2 * kChunk, 0x50000000ull}};
    auto aligned_table = build_guest_memory_table(aligned);
    CHECK(aligned_table && hint_of(aligned_table->words, 1) == ((1u << 13) | 1u) && partial_of(aligned_table->words, 1) == 0);
    // Partial hints follow the final entries: a contained window drops and a
    // partial overlap is trimmed before hinting.
    const GuestMemoryWindow overlapping[] = {
        {9 * kChunk + 0x100, 9 * kChunk + 0x900, 0x60000000ull},
        {9 * kChunk + 0x200, 9 * kChunk + 0x300, 0x61000000ull},
        {9 * kChunk + 0x800, 9 * kChunk + 0x1000, 0x62000000ull},
    };
    auto overlap_table = build_guest_memory_table(overlapping);
    CHECK(overlap_table && overlap_table->entries == 2 && overlap_table->dropped == 1 && overlap_table->trimmed == 1);
    if (overlap_table) {
        CHECK(partial_of(overlap_table->words, 9) == ((9u << 13) | 1u));
        CHECK(pair(overlap_table->words, 2, 0) == 9 * kChunk + 0x900);
        CHECK(device_is(lookup_guest_memory(overlap_table->words, 9 * kChunk + 0x900, 4), 0x62000100ull));
    }

    // Two chunks 8192 slots apart share a hint slot, so neither keeps it.
    // Both ranges still resolve through the binary search.
    const std::uint64_t far = (3ull + 8192ull) * kChunk;
    const GuestMemoryWindow colliding[] = {
        {3 * kChunk, 4 * kChunk, 0x3000},
        {far, far + kChunk, 0x4000},
    };
    auto collision = build_guest_memory_table(colliding);
    CHECK(collision && collision->hint_collisions == 1 && hint_of(collision->words, 3) == 0);
    // Whole chunks take no partial hints. Two partly covered chunks 8192
    // slots apart share a partial slot, so neither keeps it.
    CHECK(collision && collision->partial_collisions == 0 && partial_of(collision->words, 3) == 0);
    const std::uint64_t far_partial = (5ull + 8192ull) * kChunk;
    const GuestMemoryWindow colliding_partial[] = {
        {5 * kChunk + 0x10, 5 * kChunk + 0x20, 0x5000},
        {far_partial + 0x10, far_partial + 0x20, 0x6000},
    };
    auto partial_collision = build_guest_memory_table(colliding_partial);
    CHECK(partial_collision && partial_collision->partial_collisions == 1 && partial_of(partial_collision->words, 5) == 0);
    if (partial_collision) {
        CHECK(device_is(lookup_guest_memory(partial_collision->words, 5 * kChunk + 0x18, 4), 0x5008));
        CHECK(device_is(lookup_guest_memory(partial_collision->words, far_partial + 0x10, 8), 0x6000));
    }
    if (collision) {
        CHECK(device_is(lookup_guest_memory(collision->words, 3 * kChunk + 16, 4), 0x3010));
        CHECK(device_is(lookup_guest_memory(collision->words, far + 32, 4), 0x4020));
    }

    // The last chunk below 2^40 is hinted. An address at 2^40 is not.
    const std::uint64_t limit = 1ull << 40;
    const GuestMemoryWindow high[] = {
        {limit - kChunk, limit, 0x1111},
        {limit, limit + kChunk, 0x2222},
    };
    auto high_table = build_guest_memory_table(high);
    CHECK(high_table && high_table->entries == 2 && high_table->hint_collisions == 0);
    if (high_table) {
        CHECK(hint_of(high_table->words, 8191) == ((524287u << 13) | 1u));
        CHECK(hint_of(high_table->words, 0) == 0);
        // Neither the whole chunk below 2^40 nor the pool at 2^40 gets a partial hint.
        CHECK(partial_of(high_table->words, 8191) == 0 && partial_of(high_table->words, 0) == 0);
        CHECK(device_is(lookup_guest_memory(high_table->words, limit - kChunk, 4), 0x1111));
        CHECK(device_is(lookup_guest_memory(high_table->words, limit + 8, 8), 0x222a));
    }

    std::vector<GuestMemoryWindow> many;
    many.reserve(8192);
    for (std::uint32_t index = 0; index < 8192; ++index) {
        const std::uint64_t start = std::uint64_t{index} * 32;
        many.push_back({start, start + 16, 0x80000000ull + start});
    }
    CHECK(!build_guest_memory_table(many));
    many.pop_back();
    auto exact = build_guest_memory_table(many);
    CHECK(exact && exact->entries == 8191 && exact->dropped == 0 && exact->trimmed == 0);
    if (exact) {
        CHECK(exact->words[0] == 8192 && exact->words.size() == guest_memory_table_bytes / 4);
        // All 8,191 pools share chunk 0: its partial hint names the first.
        CHECK(partial_of(exact->words, 0) == 1u && exact->partial_collisions == 0);
        CHECK(device_is(lookup_guest_memory(exact->words, 0, 4), 0x80000000ull));
        CHECK(!lookup_guest_memory(exact->words, 16, 4));
        const std::uint64_t middle = 4000ull * 32;
        CHECK(device_is(lookup_guest_memory(exact->words, middle, 4), 0x80000000ull + middle));
        const std::uint64_t last = 8190ull * 32;
        CHECK(device_is(lookup_guest_memory(exact->words, last, 16), 0x80000000ull + last));
        CHECK(pair(exact->words, 8191, 0) == last);
        CHECK(!lookup_guest_memory(exact->words, last + 16, 1));

        // The write log names entries by bit: 31 and 32 straddle a word, 8191
        // is the last entry; entry 0 names no pool.
        std::vector<std::uint32_t> log(guest_memory_write_log_bytes / 4, 0);
        log[0] = 1u | 1u << 31;
        log[1] = 1u;
        log[255] = 1u << 31;
        std::vector<std::uint64_t> written;
        visit_guest_memory_writes(exact->words, log, [&](std::uint64_t start, std::uint64_t end) {
            CHECK(end == start + 16);
            written.push_back(start);
        });
        CHECK((written == std::vector<std::uint64_t>{30ull * 32, 31ull * 32, 8190ull * 32}));
    }
    {
        // Entries at or past the table's count name no pool.
        const std::array<GuestMemoryWindow, 2> two{{{0x1000, 0x2000, 0x9000}, {0x4000, 0x5000, 0xa000}}};
        auto table = build_guest_memory_table(two);
        CHECK(table && table->words[0] == 3);
        if (table) {
            std::vector<std::uint32_t> log(guest_memory_write_log_bytes / 4, 0);
            log[0] = 0b1110;
            log[200] = ~0u;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> written;
            visit_guest_memory_writes(table->words, log, [&](std::uint64_t start, std::uint64_t end) {
                written.emplace_back(start, end);
            });
            CHECK((written == std::vector<std::pair<std::uint64_t, std::uint64_t>>{{0x1000, 0x2000}, {0x4000, 0x5000}}));
            std::size_t none = 0;
            visit_guest_memory_writes(std::span(table->words).first(8), log, [&](std::uint64_t, std::uint64_t) { ++none; });
            CHECK(none == 0);
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "%u checks failed\n", failures);
        return 1;
    }
    std::puts("guest memory table tests passed");
    return 0;
}
