// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/system_resource_pool.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <set>
#include <thread>
#include <vector>

namespace {
using Pool = armigo::horizon::SystemResourcePool;
using Kind = Pool::Kind;
std::atomic<unsigned> failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
std::uint64_t low(void*) noexcept { return 0; }
std::uint64_t high(void*) noexcept { return UINT64_MAX; }

void geometry() {
    CHECK(!Pool::layout(0));
    CHECK(!Pool::layout(4095));
    CHECK(!Pool::layout(4096));
    CHECK(!Pool::layout(8192));
    const auto small = Pool::layout(65536);
    CHECK(small && small->reference_bytes == 4096 && small->bitmap_bytes == 4096 && small->usable_pages == 14);
    const auto medium = Pool::layout(2 * 1024 * 1024);
    CHECK(medium && medium->reference_bytes == 4096 && medium->bitmap_bytes == 4096 && medium->usable_pages == 510);
    const auto requested = Pool::layout(48 * 1024 * 1024);
    CHECK(requested && requested->reference_bytes == 24576 && requested->bitmap_bytes == 4096 && requested->usable_pages == 12281);
    std::string error;
    CHECK(!Pool::create(65536, nullptr, nullptr, error) && !error.empty());
}

void pages(Pool::Entropy entropy, std::size_t capacity = 65536, unsigned count = 14) {
    std::string error;
    auto pool = Pool::create(capacity, entropy, nullptr, error);
    CHECK(pool && pool->used_bytes() == 0);
    if (!pool) return;
    std::vector<Pool::Allocation> pages;
    std::set<std::uintptr_t> addresses;
    for (unsigned i = 0; i < count; ++i) {
        auto allocation = pool->allocate(Kind::page_table);
        CHECK(allocation.has_value());
        if (!allocation) return;
        auto bytes = allocation->bytes();
        CHECK(bytes.size() == 4096 && reinterpret_cast<std::uintptr_t>(bytes.data()) % 4096 == 0);
        CHECK(std::all_of(bytes.begin(), bytes.end(), [](auto v) { return v == std::byte{}; }));
        CHECK(addresses.insert(reinterpret_cast<std::uintptr_t>(bytes.data())).second);
        std::fill(bytes.begin(), bytes.end(), std::byte((i + 1) & 255));
        pages.push_back(std::move(*allocation));
        CHECK(allocation->bytes().empty());
    }
    CHECK(*addresses.rbegin() - *addresses.begin() == (count - 1) * 4096);
    CHECK(pool->used_bytes() == count * 4096 && pool->live_objects(Kind::page_table) == count);
    CHECK(!pool->allocate(Kind::page_table));
    CHECK(!pool->allocate(Kind::memory_block));
    for (unsigned i = 0; i < pages.size(); ++i)
        CHECK(pages[i].bytes().front() == std::byte((i + 1) & 255) && pages[i].bytes().back() == std::byte((i + 1) & 255));
    pages.clear();
    CHECK(pool->live_objects(Kind::page_table) == 0 && pool->used_bytes() == count * 4096);
    // Empty slab capacity remains with its original consumer.
    CHECK(!pool->allocate(Kind::memory_block));
    auto reused = pool->allocate(Kind::page_table);
    CHECK(reused && addresses.contains(reinterpret_cast<std::uintptr_t>(reused->bytes().data())));
    CHECK(reused && std::all_of(reused->bytes().begin(), reused->bytes().end(), [](auto v) { return v == std::byte{}; }));
    std::weak_ptr<Pool> retained = pool;
    pool.reset();
    CHECK(!retained.expired());
    reused.reset();
    CHECK(retained.expired());
}

std::size_t size_of(Kind kind) {
    return kind == Kind::memory_block ? 64 : kind == Kind::block_info ? 16 : 4096;
}

void full_refill(Kind kind) {
    std::string error;
    auto pool = Pool::create(65536, high, nullptr, error);
    CHECK(pool != nullptr); if (!pool) return;
    const auto count = pool->geometry().usable_pages * (4096 / size_of(kind));
    std::set<std::uintptr_t> original;
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        std::vector<Pool::Allocation> live;
        std::set<std::uintptr_t> addresses;
        for (std::size_t i = 0; i < count; ++i) {
            auto item = pool->allocate(kind);
            CHECK(item.has_value()); if (!item) return;
            CHECK(item->bytes().size() == size_of(kind));
            CHECK(std::all_of(item->bytes().begin(), item->bytes().end(), [](auto b) { return b == std::byte{}; }));
            CHECK(addresses.insert(reinterpret_cast<std::uintptr_t>(item->bytes().data())).second);
            std::fill(item->bytes().begin(), item->bytes().end(), std::byte{0x95});
            live.push_back(std::move(*item));
        }
        CHECK(!pool->allocate(kind));
        CHECK(pool->live_objects(kind) == count);
        CHECK(pool->used_bytes() == 14 * 4096);
        if (!cycle) original = addresses;
        else CHECK(addresses == original);
        // Interleave releases so a lost free-list tail cannot hide behind LIFO reuse.
        for (std::size_t i = 0; i < count; i += 2) live[i].reset();
        for (std::size_t i = 1; i < count; i += 2) live[i].reset();
        CHECK(pool->live_objects(kind) == 0 && pool->used_bytes() == 14 * 4096);
    }
}

void mixed_exhaustion() {
    std::string error;
    auto pool = Pool::create(5 * 4096, low, nullptr, error);
    CHECK(pool && pool->geometry().usable_pages == 3); if (!pool) return;
    const std::array kinds{Kind::memory_block, Kind::block_info, Kind::page_table};
    const std::array<std::size_t, 3> counts{64, 256, 1};
    std::array<std::set<std::uintptr_t>, 3> first;
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        std::array<std::vector<Pool::Allocation>, 3> live;
        std::set<std::uintptr_t> all;
        for (unsigned k = 0; k < kinds.size(); ++k) {
            std::set<std::uintptr_t> addresses;
            for (std::size_t i = 0; i < counts[k]; ++i) {
                auto item = pool->allocate(kinds[k]);
                CHECK(item.has_value()); if (!item) return;
                const auto address = reinterpret_cast<std::uintptr_t>(item->bytes().data());
                CHECK(addresses.insert(address).second && all.insert(address).second);
                live[k].push_back(std::move(*item));
            }
            if (!cycle) first[k] = addresses;
            else CHECK(first[k] == addresses);
        }
        CHECK(pool->used_bytes() == 3 * 4096);
        for (auto kind : kinds) CHECK(!pool->allocate(kind));
        // Cached pages remain attached to their own slabs even when another kind is exhausted.
        live[0].clear();
        CHECK(!pool->allocate(Kind::page_table));
        for (auto& items : live) items.clear();
        for (auto kind : kinds) CHECK(pool->live_objects(kind) == 0);
    }
}

void repeated_reset_and_move() {
    std::string error;
    auto pool = Pool::create(65536, low, nullptr, error);
    CHECK(pool != nullptr); if (!pool) return;
    auto a = pool->allocate(Kind::memory_block), b = pool->allocate(Kind::block_info);
    CHECK(a && b); if (!a || !b) return;
    const auto address = a->bytes().data();
    a->bytes()[0] = std::byte{0x63};
    auto moved = std::move(*a);
    CHECK(a->bytes().empty() && moved.bytes().data() == address);
    CHECK(pool->live_objects(Kind::memory_block) == 1);
    a->reset(); a->reset();
    auto* same = &moved;
    moved = std::move(*same);
    CHECK(moved.bytes().data() == address && moved.bytes()[0] == std::byte{0x63});
    CHECK(pool->live_objects(Kind::memory_block) == 1);
    moved = std::move(*b);
    CHECK(b->bytes().empty() && moved.bytes().size() == 16);
    CHECK(pool->live_objects(Kind::memory_block) == 0 && pool->live_objects(Kind::block_info) == 1);
    b->reset(); b->reset(); moved.reset(); moved.reset();
    CHECK(pool->live_objects(Kind::block_info) == 0 && pool->used_bytes() == 8192);
}

void bitmap_levels() {
    // These independently specified capacities leave the listed usable pages after overhead.
    struct Case { std::size_t input_pages; std::size_t usable; unsigned levels; };
    for (auto c : std::array<Case, 6>{{{65, 63, 1}, {66, 64, 2}, {67, 65, 2},
                                    {4099, 4095, 2}, {4100, 4096, 3}, {4101, 4097, 3}}}) {
        std::string error;
        auto pool = Pool::create(c.input_pages * 4096, high, nullptr, error);
        CHECK(pool != nullptr); if (!pool) continue;
        CHECK(pool->geometry().usable_pages == c.usable);
        // geometry reports the allocated bitmap levels, not only the layout planner's result.
        CHECK(pool->geometry().bitmap_levels == c.levels);
    }
}

void slabs_and_lifetime() {
    std::string error;
    auto pool = Pool::create(2 * 1024 * 1024, low, nullptr, error);
    CHECK(pool != nullptr);
    if (!pool) return;
    std::vector<Pool::Allocation> blocks, infos;
    for (unsigned i = 0; i < 65; ++i) {
        auto value = pool->allocate(Kind::memory_block);
        CHECK(value && value->bytes().size() == 64);
        blocks.push_back(std::move(*value));
    }
    CHECK(pool->used_bytes() == 2 * 4096);
    for (unsigned i = 0; i < 257; ++i) {
        auto value = pool->allocate(Kind::block_info);
        CHECK(value && value->bytes().size() == 16);
        infos.push_back(std::move(*value));
    }
    CHECK(pool->used_bytes() == 4 * 4096);
    auto page = pool->allocate(Kind::page_table);
    CHECK(page && pool->used_bytes() == 5 * 4096);
    const auto reused = blocks.back().bytes().data();
    blocks.back().reset();
    auto again = pool->allocate(Kind::memory_block);
    CHECK(again && again->bytes().data() == reused && pool->used_bytes() == 5 * 4096);
    blocks.front() = std::move(*again);
    CHECK(again->bytes().empty() && pool->live_objects(Kind::memory_block) == 64);
    CHECK(!pool->allocate(static_cast<Kind>(99)));
}

void concurrent_ownership() {
    std::string error;
    auto pool = Pool::create(2 * 1024 * 1024, high, nullptr, error);
    CHECK(pool != nullptr);
    if (!pool) return;
    std::mutex mutex;
    std::set<std::byte*> live;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < 4; ++t) threads.emplace_back([&, t] {
        for (unsigned i = 0; i < 1000; ++i) {
            auto node = pool->allocate(Kind::memory_block);
            CHECK(node.has_value());
            if (!node) return;
            { std::lock_guard lock(mutex); CHECK(live.insert(node->bytes().data()).second); }
            std::fill(node->bytes().begin(), node->bytes().end(), std::byte(t + 1));
            std::this_thread::yield();
            CHECK(node->bytes().front() == std::byte(t + 1) && node->bytes().back() == std::byte(t + 1));
            { std::lock_guard lock(mutex); CHECK(live.erase(node->bytes().data()) == 1); }
        }
    });
    for (auto& thread : threads) thread.join();
    CHECK(live.empty() && pool->live_objects(Kind::memory_block) == 0);
    CHECK(pool->used_bytes() == 4096);
}
} // namespace

int main() {
    geometry(); pages(low); pages(high);
    pages(high, 65 * 4096, 63);
    pages(high, 66 * 4096, 64);
    pages(high, 67 * 4096, 65);
    pages(high, 4099 * 4096, 4095);
    pages(high, 4100 * 4096, 4096);
    pages(high, 4101 * 4096, 4097);
    bitmap_levels();
    for (auto kind : {Kind::memory_block, Kind::block_info, Kind::page_table}) full_refill(kind);
    mixed_exhaustion(); repeated_reset_and_move();
    slabs_and_lifetime(); concurrent_ownership();
    if (failures) return 1;
    std::puts("Private system-resource geometry, ownership, exhaustion and reuse checks passed.");
    return 0;
}
