// ============================================================================
// Armigo Platform - NVN Guest Memory Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// Host side of the armigo-guest-memory shader ABI, version 3: one 8-word
// header and 8192 8-word entries, little-endian, then the write log: one bit
// per entry (bit e % 32 of word e / 32), set by every store, atomic or
// reduction that reaches entry e.

inline constexpr std::size_t guest_memory_max_entries = 8192;
inline constexpr std::size_t guest_memory_table_bytes = 32 + guest_memory_max_entries * 32;
inline constexpr std::size_t guest_memory_write_log_bytes = guest_memory_max_entries / 8;
inline constexpr std::size_t guest_memory_buffer_bytes = guest_memory_table_bytes + guest_memory_write_log_bytes;

struct GuestMemoryWindow {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint64_t gpu_address = 0;
};

// `entries` is N - 1. The other counters record drops, trims, and hint and
// partial-hint collisions.
struct GuestMemoryTable {
    std::vector<std::uint32_t> words;
    std::size_t entries = 0;
    std::size_t dropped = 0;
    std::size_t trimmed = 0;
    std::size_t hint_collisions = 0;
    std::size_t partial_collisions = 0;
};

namespace detail {
inline constexpr std::uint64_t chunk_bytes = 1ull << 21;
inline constexpr std::uint64_t hint_address_limit = 1ull << 40;
inline std::size_t entry_word(std::size_t entry) { return 8 + entry * 8; }
inline std::uint64_t load_u64(std::span<const std::uint32_t> words, std::size_t index) {
    return std::uint64_t{words[index]} | (std::uint64_t{words[index + 1]} << 32);
}
inline void store_u64(std::vector<std::uint32_t>& words, std::size_t index, std::uint64_t value) {
    words[index] = static_cast<std::uint32_t>(value);
    words[index + 1] = static_cast<std::uint32_t>(value >> 32);
}
} // namespace detail

// Drops empty and contained windows, then trims a partial overlap onto the
// previous end without changing its delta. Fails above 8191 kept windows.
// Entry 0 is empty; entries N..8191 use 0xffffffff as start and end. A 2 MiB
// chunk wholly inside an entry below 2^40 stores (c << 13) | e at slot
// c & 8191 (word 6), and a second chunk clears that slot permanently. A chunk
// below 2^40 that an entry starts or ends inside without wholly holding it
// stores the first entry overlapping it as its partial hint (word 7), cleared
// the same way.
inline std::optional<GuestMemoryTable> build_guest_memory_table(
    std::span<const GuestMemoryWindow> windows) {
    struct Range { std::uint64_t start, end, gpu; };
    std::vector<Range> pending;
    pending.reserve(windows.size());
    std::size_t dropped = 0;
    for (const GuestMemoryWindow& window : windows) {
        if (window.end <= window.start) { ++dropped; continue; }
        pending.push_back({window.start, window.end, window.gpu_address});
    }
    std::stable_sort(pending.begin(), pending.end(), [](const Range& a, const Range& b) {
        return a.start < b.start || (a.start == b.start && a.end > b.end);
    });

    std::vector<Range> kept;
    kept.reserve(pending.size());
    std::size_t trimmed = 0;
    for (Range window : pending) {
        if (!kept.empty() && window.start < kept.back().end) {
            const std::uint64_t previous_end = kept.back().end;
            if (window.end <= previous_end) { ++dropped; continue; }
            window.gpu += previous_end - window.start;
            window.start = previous_end;
            ++trimmed;
        }
        kept.push_back(window);
    }
    if (kept.size() > guest_memory_max_entries - 1) return std::nullopt;

    GuestMemoryTable table;
    table.words.assign(guest_memory_table_bytes / 4, 0);
    table.entries = kept.size();
    table.dropped = dropped;
    table.trimmed = trimmed;
    const std::uint32_t count = static_cast<std::uint32_t>(kept.size() + 1);
    table.words[0] = count;
    for (std::size_t index = 0; index < kept.size(); ++index) {
        const std::size_t base = detail::entry_word(index + 1);
        detail::store_u64(table.words, base, kept[index].start);
        detail::store_u64(table.words, base + 2, kept[index].end);
        detail::store_u64(table.words, base + 4, kept[index].gpu - kept[index].start);
    }
    for (std::size_t entry = count; entry < guest_memory_max_entries; ++entry) {
        const std::size_t base = detail::entry_word(entry);
        table.words[base] = table.words[base + 1] = 0xffffffffu;
        table.words[base + 2] = table.words[base + 3] = 0xffffffffu;
    }

    std::vector<std::uint8_t> slot_state(guest_memory_max_entries); // 0 empty, 1 set, 2 blocked
    for (std::size_t entry = 1; entry < count; ++entry) {
        const std::size_t base = detail::entry_word(entry);
        const std::uint64_t start = detail::load_u64(table.words, base);
        const std::uint64_t end = detail::load_u64(table.words, base + 2);
        if (start >= detail::hint_address_limit || end < detail::chunk_bytes) continue;
        const std::uint64_t first = (start + detail::chunk_bytes - 1) & ~(detail::chunk_bytes - 1);
        const std::uint64_t limit = std::min(end, detail::hint_address_limit);
        if (limit < detail::chunk_bytes || first > limit - detail::chunk_bytes) continue;
        for (std::uint64_t at = first; at <= limit - detail::chunk_bytes; at += detail::chunk_bytes) {
            const std::uint64_t chunk = at >> 21;
            const std::size_t slot = static_cast<std::size_t>(chunk & 8191u);
            if (slot_state[slot] == 2) continue;
            std::uint32_t& hint = table.words[detail::entry_word(slot) + 6];
            if (slot_state[slot] == 1) {
                hint = 0;
                slot_state[slot] = 2;
                ++table.hint_collisions;
                continue;
            }
            hint = static_cast<std::uint32_t>((chunk << 13) | entry);
            slot_state[slot] = 1;
        }
    }
    // Partial hints: a chunk an entry starts or ends inside, and does not
    // wholly hold, names the first entry overlapping it (entries are sorted,
    // so the first to claim it). A second chunk at the same slot clears it
    // permanently.
    std::vector<std::uint8_t> partial_state(guest_memory_max_entries); // 0 empty, 1 set, 2 blocked
    for (std::size_t entry = 1; entry < count; ++entry) {
        const std::size_t base = detail::entry_word(entry);
        const std::uint64_t start = detail::load_u64(table.words, base);
        const std::uint64_t end = detail::load_u64(table.words, base + 2);
        for (const std::uint64_t at : {start, end - 1}) {
            if (at >= detail::hint_address_limit) continue;
            const std::uint64_t chunk = at >> 21;
            const std::uint64_t chunk_start = chunk << 21;
            if (start <= chunk_start && end - chunk_start >= detail::chunk_bytes) continue;  // the chunk hint's
            const std::size_t slot = static_cast<std::size_t>(chunk & 8191u);
            std::uint32_t& partial = table.words[detail::entry_word(slot) + 7];
            if (partial_state[slot] == 2 || (partial_state[slot] == 1 && (partial >> 13) == chunk)) continue;
            if (partial_state[slot] == 1) {
                partial = 0;
                partial_state[slot] = 2;
                ++table.partial_collisions;
                continue;
            }
            partial = static_cast<std::uint32_t>((chunk << 13) | entry);
            partial_state[slot] = 1;
        }
    }
    return table;
}

// Device address of 1..16 naturally aligned bytes at `address`, or nullopt.
// Tries the hint, then the last entry in 1..N-1 with start <= address.
inline std::optional<std::uint64_t> lookup_guest_memory(std::span<const std::uint32_t> words,
                                                        std::uint64_t address, std::uint32_t size) {
    if (size < 1 || size > 16 || address % size != 0 || words.size() < 8 ||
        address > std::numeric_limits<std::uint64_t>::max() - size) {
        return std::nullopt;
    }
    const std::uint32_t count = words[0];
    if (count < 1 || count > guest_memory_max_entries) return std::nullopt;
    if (words.size() < detail::entry_word(count - 1) + 8) return std::nullopt;

    auto contained = [&](std::uint32_t entry) -> std::optional<std::uint64_t> {
        if (entry == 0 || entry >= count) return std::nullopt;
        const std::size_t base = detail::entry_word(entry);
        const std::uint64_t start = detail::load_u64(words, base);
        const std::uint64_t end = detail::load_u64(words, base + 2);
        if (address < start || address > end || end - address < size) return std::nullopt;
        return address + detail::load_u64(words, base + 4);
    };

    const std::uint64_t chunk = address >> 21;
    if (chunk < (detail::hint_address_limit >> 21)) {
        const std::size_t slot = static_cast<std::size_t>(chunk & 8191u);
        if (words.size() >= detail::entry_word(slot) + 7) {
            const std::uint32_t hint = words[detail::entry_word(slot) + 6];
            if (static_cast<std::uint64_t>(hint >> 13) == chunk) {
                if (auto found = contained(hint & 8191u)) return found;
            }
        }
    }

    std::uint32_t lower = 1;
    std::uint32_t upper = count;
    while (lower < upper) {
        const std::uint32_t mid = lower + (upper - lower) / 2;
        if (detail::load_u64(words, detail::entry_word(mid)) <= address) lower = mid + 1;
        else upper = mid;
    }
    return contained(lower - 1);
}

// Calls visit(start, end) for each pool the write log marks, in entry order.
// Entry 0 and entries at or past the table's count name no pool.
template <typename Visit>
void visit_guest_memory_writes(std::span<const std::uint32_t> words, std::span<const std::uint32_t> log,
                               Visit&& visit) {
    if (words.size() < guest_memory_table_bytes / 4) return;
    const std::size_t entries = std::min<std::size_t>(words[0], guest_memory_max_entries);
    for (std::size_t word = 0; word < log.size() && word * 32 < entries; ++word) {
        for (auto bits = log[word]; bits; bits &= bits - 1) {
            const std::size_t entry = word * 32 + static_cast<std::size_t>(std::countr_zero(bits));
            if (!entry || entry >= entries) continue;
            const std::uint64_t start = detail::load_u64(words, detail::entry_word(entry));
            const std::uint64_t end = detail::load_u64(words, detail::entry_word(entry) + 2);
            if (end > start) visit(start, end);
        }
    }
}

} // namespace armigo::platform::metal::nvn_driver
