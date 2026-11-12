// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/3ds/guest_memory.h"

#include <cstdio>

namespace {
using armigo::n3ds::GuestMemory;
using armigo::n3ds::HostMemoryMapping;
struct Mapper {
    unsigned calls{};
    std::uint32_t permissions{};
    int error{};
    bool wrong_address{};
};
HostMemoryMapping Map(void* context, std::uint32_t address, std::uint32_t,
                      std::uint32_t permissions) noexcept {
    auto& mapper = *static_cast<Mapper*>(context);
    ++mapper.calls;
    mapper.permissions = permissions;
    return {mapper.wrong_address ? address + 4096 : address, mapper.error};
}
unsigned checks{};
unsigned failures{};
void Check(bool condition, int line) {
    ++checks;
    if (!condition) {
        ++failures;
        std::printf("FAIL line %d\n", line);
    }
}
#define CHECK(condition) Check((condition), __LINE__)

void ExistingMappingsAndCommit() {
    GuestMemory memory;
    Mapper mapper;
    CHECK(!memory.Initialize(0, Map, &mapper));
    CHECK(!memory.Initialize(4097, Map, &mapper));
    CHECK(!memory.Initialize(1ULL << 32, Map, &mapper));
    CHECK(!memory.Initialize(4096, nullptr, &mapper));
    CHECK(memory.Initialize(8 * 4096, Map, &mapper));
    CHECK(memory.RegisterExisting({0x100000, 4096, 5}, true));
    CHECK(memory.RegisterExisting({0x200000, 4096, 1}, false));
    CHECK(!memory.RegisterExisting({0x100000, 4096, 5}, true));
    CHECK(!memory.RegisterExisting({0x300001, 4096, 3}, true));
    CHECK(!memory.RegisterExisting({0x300000, 4095, 3}, true));
    CHECK(!memory.RegisterExisting({0x300000, 4096, 8}, true));
    CHECK(!memory.RegisterExisting({0x300000, 0, 3}, true));
    CHECK(!memory.RegisterExisting({0xfffff000, 8192, 3}, true));
    CHECK(memory.CommittedBytes() == 4096);
    CHECK(memory.MappingCount() == 2);
    std::uint32_t output = 0xabcdef;
    CHECK(memory.Commit(3, 0x08000000, 0, 2 * 4096, 3, output) == 0);
    CHECK(output == 0x08000000);
    CHECK(memory.CommittedBytes() == 3 * 4096);
    CHECK(memory.MappingCount() == 3);
    CHECK(memory.CommitCount() == 1);
    CHECK(mapper.calls == 1 && mapper.permissions == 3);
    CHECK(memory.Commit(3, 0x08001000, 0, 4096, 3, output) == 0xe0a01bf5);
    CHECK(memory.Commit(3, 0x08002000, 0, 4096, 1, output) == 0);
    CHECK(mapper.permissions == 1);
    CHECK(memory.CommittedBytes() == 4 * 4096);
    CHECK(memory.Initialize(4096, Map, &mapper));
    CHECK(memory.CommittedBytes() == 0 && memory.MappingCount() == 0);
}

void AdmissionAndFailureTransactions() {
    GuestMemory memory;
    Mapper mapper;
    CHECK(memory.Initialize(4096, Map, &mapper));
    std::uint32_t output = 0x1234;
    CHECK(memory.Commit(3, 0x08000001, 0, 4096, 3, output) == 0xe0e01bf1);
    CHECK(memory.Commit(3, 0x08000000, 1, 4096, 3, output) == 0xe0e01bf1);
    CHECK(memory.Commit(3, 0x08000000, 0, 4095, 3, output) == 0xe0e01bf2);
    CHECK(memory.Commit(3, 0x08000000, 0, 4096, 4, output) == 0xe0e01bee);
    CHECK(memory.Commit(0x10003, 0x08000000, 0, 4096, 3, output) == 0xe0e01bf4);
    CHECK(memory.Commit(3, 0x08000000, 0, 0, 3, output) == 0xe0e01bf4);
    CHECK(memory.Commit(3, 0x07fff000, 0, 4096, 3, output) == 0xe0e01bf5);
    CHECK(memory.Commit(3, 0x10000000, 0, 4096, 3, output) == 0xe0e01bf5);
    CHECK(memory.Commit(3, 0x0ffff000, 0, 8192, 3, output) == 0xe0e01bf5);
    CHECK(memory.Commit(3, 0xfffff000, 0, 8192, 3, output) == 0xe0e01bf5);
    CHECK(memory.Commit(3, 0x08000000, 0, 8192, 3, output) == 0xc860180a);
    CHECK(output == 0x1234 && mapper.calls == 0);
    CHECK(memory.CommittedBytes() == 0 && memory.MappingCount() == 0);
    mapper.error = 12;
    CHECK(memory.Commit(3, 0x08000000, 0, 4096, 3, output) == 0xc860180a);
    CHECK(memory.LastHostError() == 12);
    CHECK(output == 0x1234 && memory.CommittedBytes() == 0 && memory.MappingCount() == 0);
    mapper.error = 0;
    mapper.wrong_address = true;
    CHECK(memory.Commit(3, 0x08000000, 0, 4096, 3, output) == 0xc860180a);
    CHECK(memory.LastHostError() == -1 && output == 0x1234);
    mapper.wrong_address = false;
    CHECK(memory.Commit(3, 0x08000000, 0x2000, 4096, 0, output) == 0);
    CHECK(mapper.permissions == 0 && memory.LastHostError() == 0);
    CHECK(memory.CommittedBytes() == 4096);
    output = 0x5678;
    CHECK(memory.Commit(3, 0x08001000, 0, 4096, 3, output) == 0xc860180a);
    CHECK(output == 0x5678 && mapper.calls == 3);
}

void CapacityAndAddressBoundary() {
    GuestMemory memory;
    Mapper mapper;
    CHECK(memory.Initialize(128 * 4096, Map, &mapper));
    CHECK(memory.RegisterExisting({0xfffff000, 4096, 1}, false));
    for (std::uint32_t i = 0; i < 63; ++i) {
        CHECK(memory.RegisterExisting({0x100000 + i * 4096, 4096, 3}, true));
    }
    CHECK(!memory.RegisterExisting({0x200000, 4096, 3}, true));
    std::uint32_t output = 7;
    CHECK(memory.Commit(3, 0x08000000, 0, 4096, 3, output) == 0xe0e01bf4);
    CHECK(output == 7 && mapper.calls == 0 && memory.CommittedBytes() == 63 * 4096);
    CHECK(memory.Initialize(4096, Map, &mapper));
    CHECK(memory.Commit(3, 0x0ffff000, 0, 4096, 2, output) == 0);
    CHECK(output == 0x0ffff000 && mapper.permissions == 2);
}
void LinearAllocationAndRollback() {
    GuestMemory memory;
    Mapper mapper;
    CHECK(!memory.ConfigureLinearHeap(0x14000000, 16 * 4096));
    CHECK(memory.Initialize(8 * 4096, Map, &mapper));
    CHECK(!memory.ConfigureLinearHeap(0x14000001, 16 * 4096));
    CHECK(!memory.ConfigureLinearHeap(0x14000000, 4096));
    CHECK(!memory.ConfigureLinearHeap(0x0f000000, 16 * 4096));
    CHECK(memory.ConfigureLinearHeap(0x14000000, 16 * 4096));
    CHECK(!memory.ConfigureLinearHeap(0x30000000, 16 * 4096));
    CHECK(!memory.RegisterExisting({0x14000000, 4096, 3}, false));
    CHECK(memory.RegisterExisting({0x100000, 4096, 5}, true));
    std::uint32_t output = 0;
    CHECK(memory.Commit(3, 0x08000000, 0, 2 * 4096, 3, output) == 0);
    CHECK(memory.Commit(0x10003, 0, 0, 3 * 4096, 3, output) == 0);
    CHECK(output == 0x14000000 && memory.LinearCommittedBytes() == 3 * 4096);
    CHECK(memory.Commit(0x10003, 0, 0, 2 * 4096, 3, output) == 0);
    CHECK(output == 0x14003000 && memory.LinearCommittedBytes() == 5 * 4096);
    CHECK(memory.CommittedBytes() == 8 * 4096 && memory.CommitCount() == 3);
    output = 0x1234;
    CHECK(memory.Commit(0x10003, 0, 0, 4096, 3, output) == 0xc860180a);
    CHECK(output == 0x1234 && mapper.calls == 3);
    CHECK(memory.Initialize(8 * 4096, Map, &mapper));
    CHECK(memory.LinearCommittedBytes() == 0);
    CHECK(memory.ConfigureLinearHeap(0x30000000, 16 * 4096));
    mapper.error = 12;
    CHECK(memory.Commit(0x10003, 0, 0, 4096, 3, output) == 0xc860180a);
    CHECK(output == 0x1234 && memory.LinearCommittedBytes() == 0 && memory.CommittedBytes() == 0);
    mapper.error = 0;
    CHECK(memory.Commit(0x10003, 0, 0, 4096, 3, output) == 0);
    CHECK(output == 0x30000000 && memory.LinearCommittedBytes() == 4096);
    CHECK(memory.Commit(0x10003, 0x30001000, 0, 4096, 3, output) == 0xe0e01bf4);
    CHECK(memory.LinearCommittedBytes() == 4096 && memory.CommittedBytes() == 4096);
}
} // namespace

int main() {
    ExistingMappingsAndCommit();
    AdmissionAndFailureTransactions();
    CapacityAndAddressBoundary();
    LinearAllocationAndRollback();
    std::printf("%s: %u checks, %u failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
