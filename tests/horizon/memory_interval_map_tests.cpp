// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/horizon/memory_interval_map.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>

namespace allocation_fault {
bool refuse = false;
std::size_t largest = 0;
}
void* operator new(std::size_t size) {
    allocation_fault::largest = std::max(allocation_fault::largest, size);
    if (allocation_fault::refuse) { allocation_fault::refuse = false; throw std::bad_alloc(); }
    if (auto* data = std::malloc(size ? size : 1)) return data;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* data) noexcept { std::free(data); }
void operator delete[](void* data) noexcept { std::free(data); }
void operator delete(void* data, std::size_t) noexcept { std::free(data); }
void operator delete[](void* data, std::size_t) noexcept { std::free(data); }

namespace {
using namespace armigo::horizon;
using Map = MemoryIntervalMap;
using Error = IntervalMapError;
unsigned failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); ++failures; } } while (false)
std::uint64_t entropy(void*) noexcept { return 0; }
std::shared_ptr<SystemResourcePool> pool(std::size_t bytes) {
    std::string error;
    auto result = SystemResourcePool::create(bytes, entropy, nullptr, error);
    CHECK(result != nullptr); return result;
}
bool accept(void*) noexcept { return true; }
bool refuse(void*) noexcept { return false; }
Error update(Map& map, std::uint64_t first, std::uint64_t count, unsigned permission,
    bool (*apply)(void*) noexcept = accept) {
    const AddressRange range{first * 4096, count * 4096};
    const Map::Update change{range, MemoryBlock{range, permission ? MemoryType::normal : MemoryType::free,
        0, static_cast<std::uint8_t>(permission)}};
    return map.transact(std::span(&change, 1), apply, nullptr);
}
void differential() {
    constexpr unsigned pages = 256;
    auto arena = pool(65536);
    auto map = Map::create(arena, pages * 4096);
    CHECK(map && map->valid_tree()); if (!map) return;
    std::array<unsigned, pages> model{};
    std::mt19937 random(91831);
    for (unsigned trial = 0; trial < 5000; ++trial) {
        const unsigned first = random() % pages;
        const unsigned count = 1 + random() % (pages - first);
        const unsigned permission = (random() % 3) * 2;
        const unsigned bits = permission == 4 ? 3 : permission == 2 ? 1 : 0;
        const bool rejected = trial % 13 == 0;
        CHECK(update(*map, first, count, bits, rejected ? refuse : accept) ==
            (rejected ? Error::native_failure : Error::none));
        if (!rejected) std::fill(model.begin() + first, model.begin() + first + count, bits);
        CHECK(map->valid_tree());
        unsigned expected_nodes = 1;
        for (unsigned page = 0; page < pages; ++page) {
            if (page && model[page] != model[page - 1]) ++expected_nodes;
            const auto block = map->query(page * 4096);
            CHECK(block.permissions == model[page]);
            CHECK(block.type == (model[page] ? MemoryType::normal : MemoryType::free));
            CHECK(map->free({page * 4096, 4096}) == !model[page]);
            const auto uniform = map->uniform({page * 4096, 4096});
            CHECK(uniform.has_value() == bool(model[page]));
            if (uniform) CHECK(uniform->permissions == model[page]);
            CHECK(map->covered({page * 4096, 4096}, 1) == bool(model[page]));
            CHECK(map->covered({page * 4096, 4096}, 2) == (model[page] == 3));
        }
        CHECK(map->node_count() == expected_nodes);
        CHECK(arena->live_objects(SystemResourcePool::Kind::memory_block) == expected_nodes);
    }
    map.reset();
    CHECK(arena->live_objects(SystemResourcePool::Kind::memory_block) == 0);
}
void bounded_split() {
    constexpr unsigned pages = 2048;
    auto arena = pool(256 * 1024);
    auto map = Map::create(arena, pages * 4096);
    for (unsigned i = 0; i < pages; ++i) CHECK(update(*map, i, 1, i % 2 ? 1 : 3) == Error::none);
    CHECK(map->node_count() == pages && map->valid_tree());
    CHECK(map->snapshot({1000 * 4096, 4096}).size() == 1);
    allocation_fault::largest = 0;
    CHECK(update(*map, 1000, 1, 1) == Error::none);
    CHECK(allocation_fault::largest < 2048); // No whole-map allocation for one-page change.
    CHECK(map->node_count() == pages - 2 && map->valid_tree());
    allocation_fault::refuse = true;
    CHECK(update(*map, 1000, 1, 3) == Error::resource_limit);
    CHECK(map->node_count() == pages - 2 && map->query(1000 * 4096).permissions == 1);
    allocation_fault::largest = 0;
    CHECK(update(*map, 1000, 1, 3) == Error::none);
    CHECK(allocation_fault::largest < 2048); // The successful split is bounded too.
    CHECK(map->node_count() == pages && map->valid_tree());
}
void exhaustion_and_retry() {
    // One usable slab page, exactly 64 records including free intervals.
    auto arena = pool(3 * 4096);
    auto map = Map::create(arena, 128 * 4096);
    for (unsigned i = 0; i < 63; ++i) CHECK(update(*map, i, 1, i % 2 ? 1 : 3) == Error::none);
    CHECK(map->node_count() == 64 && map->valid_tree());
    bool called = false;
    const Map::Update split{{100 * 4096, 4096}, MemoryBlock{{}, MemoryType::normal, 0, 3}};
    const auto apply = [](void* p) noexcept { *static_cast<bool*>(p) = true; return true; };
    CHECK(map->transact(std::span(&split, 1), apply, &called) == Error::resource_limit);
    CHECK(!called && map->node_count() == 64 && map->query(100 * 4096).type == MemoryType::free);
    CHECK(arena->live_objects(SystemResourcePool::Kind::memory_block) == 64);
    CHECK(update(*map, 0, 4, 3) == Error::none);
    CHECK(map->transact(std::span(&split, 1), apply, &called) == Error::none);
    CHECK(called && map->covered(split.range, 3) && map->valid_tree());
}
void batch_and_barriers() {
    auto arena = pool(65536);
    auto map = Map::create(arena, 32 * 4096);
    CHECK(update(*map, 0, 32, 3) == Error::none);
    const std::array<Map::Update, 2> changes{{
        {{4096, 4096}, MemoryBlock{{}, MemoryType::normal, 0, 1}},
        {{30 * 4096, 4096}, MemoryBlock{{}, MemoryType::normal, 0, 1}}}};
    CHECK(map->transact(changes, accept, nullptr) == Error::none);
    CHECK(map->node_count() == 5 && map->valid_tree());
    const Map::Update barrier{{8 * 4096, 4096}, MemoryBlock{{}, MemoryType::normal, 0, 3}, 1, 0};
    CHECK(map->transact(std::span(&barrier, 1), accept, nullptr) == Error::none);
    CHECK(map->query(8 * 4096).merge_disable == 1 && map->valid_tree());
    auto clear = barrier; clear.set_merge = 0; clear.clear_merge = 1;
    CHECK(map->transact(std::span(&clear, 1), accept, nullptr) == Error::none);
    CHECK(map->node_count() == 5 && map->valid_tree());
    auto invalid = barrier; invalid.range.size = 1;
    CHECK(map->transact(std::span(&invalid, 1), accept, nullptr) == Error::invalid_map);
}
}
int main() {
    differential(); bounded_split(); exhaustion_and_retry(); batch_and_barriers();
    return failures ? 1 : 0;
}
