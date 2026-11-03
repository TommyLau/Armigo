// ============================================================================
// Armigo Tests - Shared Memory Object
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/handle_table.h>
#include <armigo/horizon/shared_memory.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)
constexpr std::size_t page = 4096;
}  // namespace

int main() {
    using armigo::horizon::HandleTable;
    using armigo::horizon::SharedMemory;

    // Only complete pages are accepted, as for the read-only shared-memory object.
    bool refused = false;
    try {
        SharedMemory empty(0);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
    refused = false;
    try {
        SharedMemory partial(page + 4);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);

    SharedMemory memory(0x40000);
    CHECK(memory.size() == 0x40000);
    // A guest mapping aliases this base, so it must be page-aligned.
    CHECK(reinterpret_cast<std::uintptr_t>(memory.data()) % page == 0);
    bool zeroed = true;
    for (const auto value : memory.bytes()) zeroed = zeroed && value == std::byte{};
    CHECK(zeroed);

    // Writes through the object are the bytes a mapping aliases.
    memory.data()[0] = std::byte{0x12};
    memory.data()[0x3ffff] = std::byte{0x34};
    CHECK(memory.bytes()[0] == std::byte{0x12});
    CHECK(memory.bytes()[0x3ffff] == std::byte{0x34});
    const auto& const_memory = memory;
    CHECK(const_memory.bytes()[0] == std::byte{0x12});

    // Rephasing moves the bytes to an offset inside a larger page, keeping them,
    // and the new region spans whole pages past the last byte.
    constexpr std::size_t large = 4 * page;
    CHECK(memory.spans(0, page));
    CHECK(!memory.rephase(page / 2, large) && !memory.rephase(large, large) && !memory.rephase(0, 3 * page));
    CHECK(memory.rephase(2 * page, large));
    CHECK(reinterpret_cast<std::uintptr_t>(memory.data()) % large == 2 * page);
    CHECK(memory.spans(2 * page, large) && !memory.spans(0, large) && !memory.spans(page, large));
    CHECK(memory.bytes()[0] == std::byte{0x12} && memory.bytes()[0x3ffff] == std::byte{0x34});
    memory.update([](std::span<std::byte> bytes) { bytes[1] = std::byte{0x78}; });
    CHECK(memory.data()[1] == std::byte{0x78});
    CHECK(memory.rephase(0, large) && memory.spans(0, large));
    CHECK(memory.data()[1] == std::byte{0x78} && memory.bytes()[0x3ffff] == std::byte{0x34});

    // Two handles to the same object share its storage, and the table keeps the object
    // alive after the creator's reference is gone.
    HandleTable handles(8);
    auto object = std::make_shared<SharedMemory>(page);
    object->data()[8] = std::byte{0x56};
    const auto first = handles.insert(object);
    const auto second = handles.insert(object);
    CHECK(first.has_value() && second.has_value() && *first != *second);
    object.reset();
    const auto pinned = handles.lookup_as<SharedMemory>(*first);
    CHECK(pinned != nullptr);
    CHECK(pinned->data()[8] == std::byte{0x56});
    const auto other = handles.lookup_as<SharedMemory>(*second);
    CHECK(other == pinned);
    CHECK(handles.lookup_as<SharedMemory>(*first)->size() == page);
    CHECK(handles.close(*first));
    CHECK(handles.lookup_as<SharedMemory>(*first) == nullptr);
    CHECK(handles.lookup_as<SharedMemory>(*second) == pinned);

    if (failures == 0) std::fprintf(stderr, "Shared memory failures: 0\n");
    return failures == 0 ? 0 : 1;
}
