// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/startup_queries.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <future>
#include <thread>

namespace {
using namespace armigo;
using namespace armigo::horizon;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #c); ++failures; } } while (false)
constexpr std::uint64_t mib = 1024 * 1024;
constexpr std::uint64_t code = 0x10000000;
ProcessConfiguration config(std::uint64_t reserved = 2 * mib) {
    ProcessConfiguration c;
    c.platform = GuestPlatform::switch2; c.baseline = Baseline::horizon_23_0_1;
    c.private_system_resource_bytes = reserved;
    c.allowed_core_mask = 0x3f;
    c.heap_address = 0x20000000; c.heap_size = 16 * mib;
    c.alias_address = 0x30000000; c.base_alias_size = 16 * mib;
    c.stack_address = 0x40000000; c.stack_size = 16 * mib;
    return c;
}
MemoryAccounting usage(std::uint64_t reserved) {
    MemoryAccounting a;
    a.code_bytes = 4096; a.resource_limit_bytes = 64 * mib;
    a.resource_limit_used = a.code_bytes + reserved;
    return a;
}
std::vector<MemoryBlock> blocks(unsigned count) {
    std::vector<MemoryBlock> result;
    for (unsigned i = 0; i < count; ++i)
        result.push_back({{code + i * 8192, 4096}, MemoryType::code, 0, 1});
    return result;
}
bool applied(void* value) noexcept { ++*static_cast<unsigned*>(value); return true; }
bool refused(void* value) noexcept { ++*static_cast<unsigned*>(value); return false; }

void accounting_and_queries() {
    const auto c = config(); auto a = usage(c.private_system_resource_bytes);
    auto input = blocks(1);
    auto built = ProcessRecord::create(c, a, input);
    CHECK(built.ok()); if (!built.ok()) return;
    auto process = built.process;
    CHECK(process->total_memory() == 62 * mib);
    CHECK(process->used_memory() == 4096);
    CHECK(process->used_system_resource() == 4096);
    input[0].permissions = 3;
    CHECK(!process->writable(code, 1));
    CHECK(process->query_memory(code).permissions == 1);
    CHECK(process->query_memory(code + 4096).type == MemoryType::free);
    CHECK(process->query_memory(ProcessRecord::query_end).type == MemoryType::reserved);
    const auto larger_config = config(48 * mib);
    auto larger = ProcessRecord::create(larger_config, usage(48 * mib), blocks(1));
    CHECK(larger.ok());
    if (larger.ok()) {
        CHECK(larger.process->total_memory() == 16 * mib);
        CHECK(larger.process->used_system_resource() == 4096);
        CHECK(larger.process->query_memory(code).range.address == code);
    }
    auto missing_charge = a; missing_charge.resource_limit_used = a.code_bytes;
    CHECK(!ProcessRecord::create(c, missing_charge, blocks(1)).ok());
    auto tiny_limit = a; tiny_limit.resource_limit_bytes = c.private_system_resource_bytes;
    CHECK(!ProcessRecord::create(c, tiny_limit, blocks(1)).ok());
    auto cap = a; cap.resource_limit_bytes = (9809 + 16) * mib;
    CHECK(process->set_accounting(cap) == ProcessStateError::none);
    CHECK(process->total_memory() == 9809 * mib - c.private_system_resource_bytes);
    CHECK(process->set_accounting(a) == ProcessStateError::none);

    // The application pool is independent of the heap VA capacity. This
    // synthetic accounting snapshot claims no host allocation of these bytes.
    auto large = a;
    large.resource_limit_bytes = 9809 * mib;
    large.other_accounted_mapping_bytes = 9ull << 30;
    large.resource_limit_used = large.code_bytes + large.other_accounted_mapping_bytes +
                               c.private_system_resource_bytes;
    CHECK(process->set_accounting(large) == ProcessStateError::none);
    CHECK(process->total_memory() == 9809 * mib - c.private_system_resource_bytes);
    CHECK(process->used_memory() == large.code_bytes + large.other_accounted_mapping_bytes);
    auto exhausted = large;
    exhausted.resource_limit_used = exhausted.resource_limit_bytes + 1;
    CHECK(process->set_accounting(exhausted) == ProcessStateError::invalid_accounting);
    CHECK(process->used_memory() == large.code_bytes + large.other_accounted_mapping_bytes);
    exhausted = large;
    exhausted.other_accounted_mapping_bytes = exhausted.resource_limit_bytes;
    exhausted.resource_limit_used = exhausted.resource_limit_bytes + exhausted.code_bytes +
                                    c.private_system_resource_bytes;
    exhausted.resource_limit_bytes *= 2;
    CHECK(process->set_accounting(exhausted) == ProcessStateError::invalid_accounting);
    CHECK(process->set_accounting(a) == ProcessStateError::none);

    HandleTable handles(64);
    auto thread = std::make_shared<ThreadRecord>(process, 9, 44);
    StartupQueries queries(process, handles, thread);
    QueryRegisters legacy_permission{};
    legacy_permission[0] = code; legacy_permission[1] = 4096; legacy_permission[2] = 3;
    const auto unchanged = legacy_permission;
    CHECK(queries.dispatch(2, legacy_permission) == QueryDisposition::unsupported_operation);
    CHECK(legacy_permission == unchanged && process->query_memory(code).permissions == 1);
    CHECK(process->replace_memory_map(input) == ProcessStateError::unsupported_profile);
    CHECK(process->query_memory(code).permissions == 1);
    for (const auto [id, expected] : std::array<std::pair<unsigned, std::uint64_t>, 4>{{
             {16, 2 * mib}, {17, 4096}, {21, 62 * mib}, {22, 4096}}}) {
        QueryRegisters r{}; r[1] = id; r[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled);
        CHECK(r[0] == 0 && r[1] == expected);
    }
    QueryRegisters r{}; r[1] = 16; r[2] = 0;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xe401 && r[1] == 16);
    r = {}; r[1] = 16; r[2] = 0; r[3] = 1;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xe801);
    r = {}; r[1] = 8; r[2] = current_process_handle; r[3] = 1;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xe401);
    r = {}; r[1] = 8;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0 && r[1] == 0);
    r = {}; r[1] = 33; r[2] = current_process_handle;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0 && r[1] == 0x1000);
    for (unsigned id : {29u, 30u, 36u, 37u, 38u}) {
        r = {}; r[1] = id; r[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, r) == QueryDisposition::unsupported_operation);
    }
    // Remote memory size and use report zero as provisional values.
    for (unsigned id : {31u, 32u}) {
        r = {}; r[1] = id; r[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0 && r[1] == 0);
    }
    r = {}; r[1] = 35;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xf001);
}

void switch1_horizon_23_profile() {
    auto c = config(0);
    c.platform = GuestPlatform::switch1;
    c.baseline = Baseline::horizon_23_0_1;
    auto a = usage(0);
    auto built = ProcessRecord::create(c, a, blocks(1));
    CHECK(built.ok()); if (!built.ok()) return;
    CHECK(built.process->total_memory() == 64 * mib && built.process->used_system_resource() == 0);
    HandleTable handles(64);
    auto thread = std::make_shared<ThreadRecord>(built.process, 3, 44);
    StartupQueries queries(built.process, handles, thread);
    QueryRegisters r{};
    r[1] = 16; r[2] = current_process_handle;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0 && r[1] == 0);
    r = {}; r[1] = 16; r[2] = 0;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xe401 && r[1] == 16);
    r = {}; r[1] = 17; r[2] = current_process_handle;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0 && r[1] == 0);
    r = {}; r[1] = 20; r[2] = 0;
    CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xe401);
    for (unsigned id : {29u, 30u, 31u, 32u, 33u}) {
        r = {}; r[1] = id; r[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, r) == QueryDisposition::handled && r[0] == 0xf001);
    }
    for (unsigned id : {36u, 37u, 38u}) {
        r = {}; r[1] = id; r[2] = current_process_handle;
        const auto before = r;
        CHECK(queries.dispatch(0x29, r) == QueryDisposition::unsupported_operation);
        CHECK(r == before);
    }
    c.private_system_resource_bytes = 2 * mib;
    CHECK(ProcessRecord::create(c, usage(2 * mib), blocks(1)).error == ProcessStateError::unsupported_profile);
    c.private_system_resource_bytes = 0;
    // A platform value outside the enumeration is not admitted as Switch 1.
    c.baseline = Baseline::horizon_23_0_1;
    c.platform = static_cast<GuestPlatform>(2);
    CHECK(ProcessRecord::create(c, usage(0), blocks(1)).error == ProcessStateError::unsupported_profile);
}

void publication_refusal_accounting() {
    const auto c = config(); auto a = usage(c.private_system_resource_bytes);
    auto built = ProcessRecord::create(c, a, blocks(1));
    CHECK(built.ok()); if (!built.ok()) return;
    const auto original = built.process->memory_map();
    std::vector<MemoryMapUpdate> updates;
    for (unsigned i = 0; i < 40; ++i) {
        const AddressRange range{code + 4 * mib + i * 8192, 4096};
        updates.push_back({range, MemoryBlock{range, MemoryType::code, 0, 1}});
    }
    auto changed = a; changed.code_bytes += 40 * 4096; changed.resource_limit_used += 40 * 4096;
    unsigned calls = 0;
    CHECK(built.process->commit_memory_updates(updates, changed, refused, &calls) == ProcessStateError::backend_failure);
    CHECK(calls == 1 && built.process->used_memory() == a.code_bytes);
    CHECK(built.process->total_memory() == 62 * mib && built.process->used_system_resource() == 8192);
    const auto after = built.process->memory_map();
    CHECK(after.size() == original.size());
    for (std::size_t i = 0; i < std::min(after.size(), original.size()); ++i)
        CHECK(encode_memory_info(after[i]) == encode_memory_info(original[i]));
    CHECK(built.process->commit_memory_updates(updates, changed, applied, &calls) == ProcessStateError::none);
    CHECK(calls == 2 && built.process->used_memory() == changed.code_bytes);
    CHECK(built.process->total_memory() == 62 * mib && built.process->used_system_resource() == 8192);
    auto invalid = changed; invalid.resource_limit_used = 0;
    CHECK(built.process->commit_memory_updates(updates, invalid, applied, &calls) == ProcessStateError::invalid_accounting);
    CHECK(calls == 2 && built.process->used_memory() == changed.code_bytes);
    CHECK(built.process->commit_memory_state(original, a, applied, &calls) == ProcessStateError::unsupported_profile);
    CHECK(calls == 2);
}
}
int main() {
    accounting_and_queries(); switch1_horizon_23_profile(); publication_refusal_accounting();
    return failures ? 1 : 0;
}
