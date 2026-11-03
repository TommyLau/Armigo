// ============================================================================
// Armigo Tests - NVN Flat Hash Map
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "flat_map.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>

namespace {

using armigo::nvn::FlatMap64;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

void check_basic() {
    FlatMap64<int> map;
    CHECK(map.find(1) == nullptr);
    CHECK(map.emplace(0, 5).first == nullptr);
    auto [first, inserted] = map.emplace(7, 70);
    CHECK(inserted && first && *first == 70);
    auto [again, inserted_again] = map.emplace(7, 71);
    CHECK(!inserted_again && again == first && *again == 70);
    map.insert_or_assign(7, 72);
    CHECK(*map.find(7) == 72);
    map.insert_or_assign(8, 80);
    CHECK(map.size() == 2 && *map.find(8) == 80);
    CHECK(map.erase(7) && !map.erase(7) && map.find(7) == nullptr && *map.find(8) == 80);
    CHECK(!map.erase(0) && map.size() == 1);
}

// Random inserts and erases against a reference map: every key that remains
// is still found after backward-shift deletions across wrapped probe runs.
void check_against_reference() {
    FlatMap64<std::uint64_t> map;
    std::unordered_map<std::uint64_t, std::uint64_t> reference;
    std::uint64_t state = 0x243f6a8885a308d3ull;
    const auto random = [&] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (int step = 0; step < 200000; ++step) {
        // A small key space forces collisions, long probe runs and wraparound.
        const auto key = 1 + random() % 512;
        if (random() % 3) {
            const auto value = random();
            map.insert_or_assign(key, value);
            reference[key] = value;
        } else {
            CHECK(map.erase(key) == (reference.erase(key) == 1));
        }
        if (step % 997 == 0) {
            CHECK(map.size() == reference.size());
            for (std::uint64_t probe = 1; probe <= 512; ++probe) {
                const auto* found = map.find(probe);
                const auto expected = reference.find(probe);
                CHECK((found != nullptr) == (expected != reference.end()));
                if (found && expected != reference.end()) CHECK(*found == expected->second);
            }
        }
    }
}

void check_reserve_and_owned_values() {
    FlatMap64<std::shared_ptr<int>> map;
    map.reserve(1000);
    for (std::uint64_t key = 1; key <= 1000; ++key) map.emplace(key * 4096, std::make_shared<int>(int(key)));
    CHECK(map.size() == 1000);
    for (std::uint64_t key = 1; key <= 1000; ++key) {
        const auto* found = map.find(key * 4096);
        CHECK(found && **found == int(key));
    }
    auto held = *map.find(4096);
    CHECK(map.erase(4096) && held.use_count() == 1 && *held == 1);
}

} // namespace

int main() {
    check_basic();
    check_against_reference();
    check_reserve_and_owned_values();
    if (failures) {
        std::fprintf(stderr, "%u flat map checks failed\n", failures);
        return 1;
    }
    std::puts("NVN flat map checks passed");
    return 0;
}
