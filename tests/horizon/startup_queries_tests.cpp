// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/horizon/startup_queries.h>
#include "../../src/horizon/process_memory.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <memory>

namespace {
using namespace armigo::horizon;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; } } while (false)

// A missed handshake or grant would leave a worker blocked in an unbounded SVC
// wait and its std::async future would block in its destructor. The test
// records the failure and aborts instead.
[[noreturn]] void bail(unsigned line, const char* what) {
    std::fprintf(stderr, "Check failed at line %u: %s; aborting so no worker can hang the test\n",
                 line, what);
    std::fflush(stderr);
    std::abort();
}

// Bounded steady-clock handshake on a live condition such as the waiters count
// of the shared condition-variable table.
template <typename Ready>
bool settles_within(std::chrono::steady_clock::duration budget, Ready&& ready) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

#define REQUIRE_SETTLES(condition)                                              \
    do {                                                                        \
        if (!settles_within(std::chrono::seconds(5), [&] { return (condition); })) \
            bail(__LINE__, #condition);                                         \
    } while (false)

#define REQUIRE_READY(future)                                                     \
    do {                                                                          \
        if ((future).wait_for(std::chrono::seconds(5)) != std::future_status::ready) \
            bail(__LINE__, "worker future did not become ready");                 \
    } while (false)

// Last-resort watchdog for waits the gates above cannot see: abort with a
// diagnostic instead of hanging the run.
void arm_watchdog() {
    const auto deadline = std::make_shared<std::atomic<std::chrono::steady_clock::time_point>>(
        std::chrono::steady_clock::now() + std::chrono::seconds(45));
    std::thread([deadline] {
        while (true) {
            if (std::chrono::steady_clock::now() > deadline->load()) {
                std::fprintf(stderr, "Check failed: watchdog budget expired; aborting instead of hanging\n");
                std::fflush(stderr);
                std::abort();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }).detach();
}

constexpr std::uint64_t code_address = 0x10000000;
constexpr std::uint64_t output_address = 0x20000000;
constexpr std::uint64_t stack_address = 0x6000400000;
constexpr std::uint64_t upper = 0xface123400000000ull;

ProcessConfiguration configuration() {
    ProcessConfiguration result;
    result.heap_address = 0x1000000000;
    result.alias_address = 0x2000000000;
    result.stack_address = 0x6000000000;
    result.allowed_core_mask = 0x5;
    result.entropy = {0x1122334455667788, 0x9900aabbccddeeff, 0x12345678, 0xabcdef};
    return result;
}

MemoryAccounting accounting() {
    return {0x4000, 0x1000, 0x6000, 0x2000, 0x100000, 0x20000};
}

std::vector<MemoryBlock> blocks() {
    // Deliberately unordered: the metadata builder owns and sorts its copy.
    return {
        {{stack_address, 0x4000}, MemoryType::stack, 0x18, 0x1b, 0x1234, 0xabcd},
        {{code_address, 0x2000}, MemoryType::code, 0, 5},
        {{output_address, 0x2000}, MemoryType::normal, 0, 3},
    };
}

bool same_blocks(std::span<const MemoryBlock> left, std::span<const MemoryBlock> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index != left.size(); ++index) {
        const auto& a = left[index];
        const auto& b = right[index];
        if (a.range.address != b.range.address || a.range.size != b.range.size ||
            a.type != b.type || a.attributes != b.attributes ||
            a.permissions != b.permissions || a.ipc_references != b.ipc_references ||
            a.device_references != b.device_references ||
            a.merge_disable != b.merge_disable) return false;
    }
    return true;
}

std::shared_ptr<ProcessRecord> process() {
    const auto built = ProcessRecord::create(configuration(), accounting(), blocks());
    CHECK(built.ok());
    return built.process;
}

QueryRegisters registers() {
    QueryRegisters result{};
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = upper + index + 1;
    return result;
}

void check_reply(const QueryRegisters& result, std::uint32_t code, std::uint64_t value) {
    CHECK(result[0] == code);
    CHECK(result[1] == value);
    for (std::size_t index = 2; index < result.size(); ++index) CHECK(result[index] == 0);
}

QueryRegisters get_info(const StartupQueries& kernel, std::uint64_t type,
                        std::uint64_t handle = current_process_handle, std::uint64_t sub = 0) {
    auto input = registers();
    input[1] = type;
    input[2] = handle;
    input[3] = sub;
    CHECK(kernel.dispatch(0x29, input) == QueryDisposition::handled);
    return input;
}

struct Output {
    std::array<std::byte, 8192> bytes;
    std::size_t completed = 0;
    std::size_t allowed = 10;
    // Makes the compare-exchange report an inaccessible word, as a revocation
    // between the wait and the signal would.
    bool force_cas_invalid = false;
    // Recorded compare-exchange traffic, for assertions on the handoff path.
    std::size_t cas_calls = 0;
    std::uint64_t cas_address = 0;
    std::uint32_t cas_desired = 0;

    void reset(std::size_t successful_words = 10) {
        bytes.fill(std::byte{0xcd});
        completed = 0;
        allowed = successful_words;
        force_cas_invalid = false;
        cas_calls = 0;
        cas_address = 0;
        cas_desired = 0;
    }

    // Arbitration reads the word a waiter observed; in this test the guest address is a host
    // address, so a direct copy is the honest stand-in for the guest reader.
    static bool read(void* raw, std::uint64_t address, std::span<std::byte, 4> value) noexcept {
        auto& output = *static_cast<Output*>(raw);
        if (address >= output_address &&
            address - output_address <= output.bytes.size() - value.size()) {
            std::memcpy(value.data(), output.bytes.data() + (address - output_address), value.size());
            return true;
        }
        std::memcpy(value.data(), reinterpret_cast<const void*>(address), value.size());
        return true;
    }
    // This fixture has no concurrent guest memory writer; read the wide value in one
    // callback. The concurrent synchronization tests use native atomic backing instead.
    static bool read_double_word(void* raw, std::uint64_t address,
                                 std::uint64_t& value) noexcept {
        auto& output = *static_cast<Output*>(raw);
        if (address < output_address || address - output_address > output.bytes.size() - 8)
            return false;
        value = output.word(static_cast<std::size_t>(address - output_address), 8);
        return true;
    }
    static bool write(void* raw, std::uint64_t address,
                      std::span<const std::byte, 4> value) noexcept {
        auto& output = *static_cast<Output*>(raw);
        if (output.completed >= output.allowed || address < output_address ||
            address - output_address > output.bytes.size() - value.size()) return false;
        std::memcpy(output.bytes.data() + (address - output_address), value.data(), value.size());
        ++output.completed;
        return true;
    }
    // The condition-variable handoff exchange: an atomic compare-and-set of one little-endian
    // word, honest about mismatch by reporting the observed value.
    static ProcessWideKey::ExchangeResult compare_exchange(void* raw, std::uint64_t address,
                                                           std::uint32_t& expected,
                                                           std::uint32_t desired) noexcept {
        auto& output = *static_cast<Output*>(raw);
        ++output.cas_calls;
        output.cas_address = address;
        output.cas_desired = desired;
        if (output.force_cas_invalid) return ProcessWideKey::ExchangeResult::invalid_address;
        if (address < output_address || address - output_address > output.bytes.size() - 4)
            return ProcessWideKey::ExchangeResult::invalid_address;
        const auto at = static_cast<std::size_t>(address - output_address);
        const auto observed = static_cast<std::uint32_t>(output.word(at, 4));
        if (observed != expected) {
            expected = observed;
            return ProcessWideKey::ExchangeResult::mismatch;
        }
        for (unsigned index = 0; index != 4; ++index)
            output.bytes[at + index] = static_cast<std::byte>((desired >> (8 * index)) & 255);
        return ProcessWideKey::ExchangeResult::exchanged;
    }

    std::uint64_t word(std::size_t offset, unsigned size) const {
        std::uint64_t value = 0;
        for (unsigned byte = 0; byte < size; ++byte) {
            value |= std::to_integer<std::uint64_t>(bytes.at(offset + byte)) << (byte * 8);
        }
        return value;
    }
};

void test_process_state() {
    auto config = configuration();
    auto usage = accounting();
    auto map = blocks();
    auto state = process();
    CHECK(state->used_memory() == 0xd000);
    CHECK(state->total_memory() == 0xed000);
    CHECK(state->query_memory(0).range.address == 0);
    CHECK(state->query_memory(0).range.size == code_address);
    CHECK(state->query_memory(code_address + 0x3001).range.address == code_address + 0x2000);
    CHECK(state->query_memory(code_address + 0x3001).range.size ==
          output_address - code_address - 0x2000);
    CHECK(state->query_memory(stack_address + 19).range.size == 0x4000);
    CHECK(state->query_memory(stack_address + 19).type == MemoryType::stack);
    CHECK(state->query_memory(ProcessRecord::query_end).type == MemoryType::reserved);
    CHECK(state->query_memory(UINT64_MAX).range.size == std::uint64_t{0} - ProcessRecord::query_end);
    CHECK(state->writable(output_address + 1, 40));
    CHECK(!state->writable(code_address, 4));
    CHECK(!state->writable(UINT64_MAX - 1, 4));
    CHECK(!state->writable(output_address + 0x1fff, 4));

    config.address_bits = 36;
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::unsupported_profile);
    config = configuration();
    config.private_system_resource_bytes = 4096;
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::unsupported_profile);
    config = configuration();
    config.allowed_core_mask = 0;
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::unsupported_profile);
    config = configuration();
    config.baseline = static_cast<Baseline>(99);
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::unsupported_profile);
    config = configuration();
    config.alias_address = config.heap_address;
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::invalid_layout);
    config = configuration();
    ++config.heap_address;
    CHECK(ProcessRecord::create(config, usage).error == ProcessStateError::invalid_layout);
    config = configuration();
    config.max_memory_blocks = 1;
    CHECK(ProcessRecord::create(config, usage, map).error == ProcessStateError::resource_limit);
    config = configuration();

    map.push_back(map[0]);
    CHECK(state->replace_memory_map(map) == ProcessStateError::invalid_memory_map);
    CHECK(state->query_memory(stack_address).range.size == 0x4000);
    map = blocks();
    map[0].range.address = ProcessRecord::query_end;
    CHECK(state->replace_memory_map(map) == ProcessStateError::invalid_memory_map);
    CHECK(state->query_memory(stack_address).range.size == 0x4000);
    map = blocks();
    map[0].range.size = 0x1000;
    map.push_back({{stack_address + 0x1000, 0x1000}, MemoryType::stack, 0x18, 0x1b, 0x1234, 0xabcd});
    CHECK(state->replace_memory_map(map) == ProcessStateError::none);
    CHECK(state->query_memory(stack_address + 0x1001).range.address == stack_address);
    CHECK(state->query_memory(stack_address + 1).range.size == 0x2000);
    CHECK(state->writable(stack_address + 0xffe, 4));

    usage.resource_limit_bytes = ProcessRecord::heap_capacity;
    usage.resource_limit_used = 0xd000;
    CHECK(state->set_accounting(usage) == ProcessStateError::none);
    CHECK(state->total_memory() == ProcessRecord::heap_capacity);
    usage.resource_limit_bytes *= 4;
    CHECK(state->set_accounting(usage) == ProcessStateError::none);
    CHECK(state->total_memory() == ProcessRecord::heap_capacity);
    usage.resource_limit_bytes = ProcessRecord::heap_capacity;
    usage.resource_limit_used += 0x4000;
    CHECK(state->set_accounting(usage) == ProcessStateError::none);
    CHECK(state->total_memory() == ProcessRecord::heap_capacity - 0x4000);
    const auto previous = state->total_memory();
    usage.resource_limit_used = 0;
    CHECK(state->set_accounting(usage) == ProcessStateError::invalid_accounting);
    CHECK(state->total_memory() == previous);
    usage = accounting();
    usage.other_accounted_mapping_bytes = UINT64_MAX;
    CHECK(state->set_accounting(usage) == ProcessStateError::invalid_accounting);
}

void test_compatible_memory_spans() {
    namespace pm = armigo::horizon::process_memory;
    constexpr std::uint64_t base = output_address + 0x100000;
    constexpr auto page = guest_page_size;
    const MemoryBlock common{{base, 2 * page}, MemoryType::normal, 0, 3, 7, 9};

    // ProcessRecord canonicalizes a plain metadata boundary when all state and
    // reference fields agree.
    std::array adjacent{common, common};
    adjacent[0].range.size = page;
    adjacent[1].range = {base + page, page};
    auto built = ProcessRecord::create(configuration(), accounting(), adjacent);
    CHECK(built.ok());
    if (built.ok()) {
        const auto merged = built.process->query_memory(base);
        CHECK(merged.range.address == base && merged.range.size == 2 * page);
        CHECK(merged.ipc_references == 7 && merged.device_references == 9);
    }

    // Restoring an attribute-split section publishes the original full range
    // again; stale fragment boundaries must not constrain later allocations.
    MemoryBlock whole{{base, 4 * page}, MemoryType::normal, 0, 3, 7, 9};
    built = ProcessRecord::create(configuration(), accounting(), std::span{&whole, 1});
    CHECK(built.ok());
    if (built.ok()) {
        std::array fragmented{
            MemoryBlock{{base, page}, MemoryType::normal, 0, 3, 7, 9},
            MemoryBlock{{base + page, 2 * page}, MemoryType::normal, 8, 3, 7, 9},
            MemoryBlock{{base + 3 * page, page}, MemoryType::normal, 0, 3, 7, 9},
        };
        CHECK(built.process->replace_memory_map(fragmented) == ProcessStateError::none);
        CHECK(built.process->query_memory(base).range.size == page);
        fragmented[1].attributes = 0;
        CHECK(built.process->replace_memory_map(fragmented) == ProcessStateError::none);
        const auto restored = built.process->query_memory(base);
        CHECK(restored.range.size == 4 * page && restored.attributes == 0);
        CHECK(restored.ipc_references == 7 && restored.device_references == 9);
    }

    // Either side may retain an internal boundary. Reference mismatches also
    // remain separate even without a barrier.
    auto boundary = adjacent;
    boundary[0].merge_disable = pm::merge_disable_right_mask;
    built = ProcessRecord::create(configuration(), accounting(), boundary);
    CHECK(built.ok() && built.process->memory_map().size() == 2);
    boundary[0].merge_disable = 0;
    boundary[1].merge_disable = pm::merge_disable_mapping;
    built = ProcessRecord::create(configuration(), accounting(), boundary);
    CHECK(built.ok() && built.process->memory_map().size() == 2);
    boundary[1].merge_disable = 0;
    ++boundary[1].ipc_references;
    built = ProcessRecord::create(configuration(), accounting(), boundary);
    CHECK(built.ok() && built.process->memory_map().size() == 2);

    // A retained mapping barrier keeps two compatible blocks distinct. A
    // permission update may still span them, and partial splitting must retain
    // both references and the internal barrier.
    adjacent[0] = common;
    adjacent[1] = common;
    adjacent[0].merge_disable = pm::merge_disable_right_mask;
    adjacent[1].range.address = base + 2 * page;
    built = ProcessRecord::create(configuration(), accounting(), adjacent);
    CHECK(built.ok());
    if (!built.ok()) return;
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(built.process, 1, 17);
    StartupQueries kernel(built.process, handles, thread);
    auto protect = registers();
    protect[0] = base + page;
    protect[1] = 2 * page;
    protect[2] = 1;
    CHECK(kernel.dispatch(0x02, protect) == QueryDisposition::handled);
    CHECK(protect[0] == 0);
    const auto updated = built.process->memory_map();
    CHECK(updated.size() == 4);
    if (updated.size() == 4) {
        CHECK(updated[1].permissions == 1 && updated[2].permissions == 1);
        CHECK(updated[1].merge_disable == pm::merge_disable_right_mask);
        for (const auto& block : updated)
            CHECK(block.ipc_references == 7 && block.device_references == 9);
    }
    auto transfer = registers();
    transfer[1] = base + page;
    transfer[2] = 2 * page;
    transfer[3] = 0;
    CHECK(kernel.dispatch(0x15, transfer) == QueryDisposition::handled);
    // The range carries IPC and device references, so the kernel cannot lock it.
    CHECK(transfer[0] == 0xd401);
    CHECK(handles.size() == 0);

    const auto rejected = [&](std::array<MemoryBlock, 2> map) {
        const auto made = ProcessRecord::create(configuration(), accounting(), map);
        CHECK(made.ok());
        if (!made.ok()) return;
        HandleTable local_handles(4);
        auto local_thread = std::make_shared<ThreadRecord>(made.process, 2, 17);
        StartupQueries local(made.process, local_handles, local_thread);
        const auto before = made.process->memory_map();
        auto permission = registers();
        permission[0] = base;
        permission[1] = 2 * page;
        permission[2] = 1;
        CHECK(local.dispatch(0x02, permission) == QueryDisposition::handled);
        CHECK(permission[0] != 0);
        CHECK(same_blocks(made.process->memory_map(), before));
        auto allocation = registers();
        allocation[1] = base;
        allocation[2] = 2 * page;
        allocation[3] = 0;
        CHECK(local.dispatch(0x15, allocation) == QueryDisposition::handled);
        CHECK(allocation[0] != 0);
        CHECK(local_handles.size() == 0);
        CHECK(same_blocks(made.process->memory_map(), before));
    };

    const auto incompatible = [&](unsigned field) {
        auto map = adjacent;
        map[0].range = {base, page};
        map[1].range = {base + page, page};
        if (field == 0) map[1].range.address += page;
        if (field == 1) map[1].type = MemoryType::code_data;
        if (field == 2) map[1].attributes = 1;
        if (field == 3) map[1].permissions = 1;
        if (field == 4) ++map[1].ipc_references;
        if (field == 5) ++map[1].device_references;
        rejected(map);
    };
    for (unsigned field = 0; field != 6; ++field) incompatible(field);
}

void test_get_info() {
    auto current = process();
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(current, 0x1122334455667788, 17);
    const auto process_handle = *handles.insert(current);
    const auto thread_handle = *handles.insert(thread);
    StartupQueries kernel(current, handles, thread);
    const std::array<std::pair<std::uint32_t, std::uint64_t>, 13> cases = {{
        {0, 5}, {2, 0x2000000000}, {3, 64ull << 30}, {4, 0x1000000000}, {5, 8ull << 30},
        {12, 0x08000000}, {13, 0x7ff8000000}, {14, 0x6000000000}, {15, 2ull << 30},
        {16, 0}, {21, 0xed000}, {22, 0xd000}, {28, 0},
    }};
    for (auto [type, value] : cases) {
        check_reply(get_info(kernel, upper | type), 0, value);
        check_reply(get_info(kernel, type, process_handle), 0, value);
        check_reply(get_info(kernel, upper | type, 0, 1), 0xe801, upper | type);
        check_reply(get_info(kernel, upper | type, 0), 0xe401, upper | type);
        check_reply(get_info(kernel, type, 0), 0xe401, type);
        check_reply(get_info(kernel, type, thread_handle), 0xe401, type);
        check_reply(get_info(kernel, type, current_thread_handle), 0xe401, type);
    }
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        for (std::uint64_t index = 0; index < 4; ++index) {
            check_reply(get_info(kernel, 11, 0, index), 0, configuration().entropy[index]);
            check_reply(get_info(kernel, 11, upper, index), 0, configuration().entropy[index]);
        }
    }
    check_reply(get_info(kernel, upper | 11, process_handle, UINT64_MAX), 0xe401, upper | 11);
    check_reply(get_info(kernel, upper | 11, 0, 4), 0xe801, upper | 11);
    check_reply(get_info(kernel, upper | 11, 0, UINT64_MAX), 0xe801, upper | 11);
    for (auto type : {19u, 29u, 33u, 35u, UINT32_MAX}) {
        check_reply(get_info(kernel, upper | type, 0, UINT64_MAX), 0xf001, upper | type);
    }

    // svcGetInfo(UserExceptionContextAddress) resolves a process and reports the
    // address the runtime reserved for its user exception context. Without one the
    // query is an explicit unsupported capability, never a fabricated address.
    {
        auto input = registers();
        input[1] = 20;
        input[2] = current_process_handle;
        input[3] = 0;
        CHECK(kernel.dispatch(0x29, input) == QueryDisposition::unsupported_operation);
    }
    current->set_user_exception_context(0x12345000);
    check_reply(get_info(kernel, 20), 0, 0x12345000);
    check_reply(get_info(kernel, 20, current_process_handle), 0, 0x12345000);
    check_reply(get_info(kernel, 20, process_handle), 0, 0x12345000);
    check_reply(get_info(kernel, 20, thread_handle), 0xe401, 20);
    check_reply(get_info(kernel, upper | 20, 0, 1), 0xe801, upper | 20);

    auto input = registers();
    input[1] = 9; // Recognized by the kernel, not implemented in this slice.
    const auto before = input;
    CHECK(kernel.dispatch(0x29, input) == QueryDisposition::unsupported_operation);
    CHECK(input == before);
    CHECK(kernel.dispatch(0x99, input) == QueryDisposition::unsupported_operation);
    CHECK(input == before);

    // Switch 1 on Horizon 23.0.1. Type 17 reports used private system resources.
    check_reply(get_info(kernel, 17), 0, 0);
    check_reply(get_info(kernel, 17, process_handle), 0, 0);
    // Types 29-33 are invalid information on Switch 1, whatever the baseline.
    for (auto type : {29u, 30u, 31u, 32u, 33u}) {
        check_reply(get_info(kernel, type), 0xf001, type);
    }
    // Types 36-38 are recognized and unimplemented: unsupported, not invalid info.
    for (auto type : {36u, 37u, 38u}) {
        input = registers();
        input[1] = type;
        input[2] = current_process_handle;
        const auto unchanged = input;
        CHECK(kernel.dispatch(0x29, input) == QueryDisposition::unsupported_operation);
        CHECK(input == unchanged);
    }
    // A zero handle is not a current-process alias. Type 8 checks the handle
    // before the sub-id.
    check_reply(get_info(kernel, 20, 0), 0xe401, 20);
    check_reply(get_info(kernel, 17, 0), 0xe401, 17);
    check_reply(get_info(kernel, 8, 0, 0), 0, 0);
    check_reply(get_info(kernel, 8, process_handle, 1), 0xe401, 8);
    check_reply(get_info(kernel, 8, 0, 1), 0xe801, 8);

    auto config = configuration();
    config.alias_extra_enabled = true;
    config.allowed_core_mask = 0xa;
    auto other = ProcessRecord::create(config, accounting(), blocks()).process;
    CHECK(other != nullptr);
    const auto other_handle = *handles.insert(other);
    check_reply(get_info(kernel, 0, other_handle), 0, 0xa);
    check_reply(get_info(kernel, 3, other_handle), 0, 128ull << 30);
    check_reply(get_info(kernel, 28, other_handle), 0, 64ull << 30);
    check_reply(get_info(kernel, 22, other_handle), 0, 0xd000);
    CHECK(handles.close(other_handle));
    check_reply(get_info(kernel, 2, other_handle), 0xe401, 2);
}

void test_thread_queries() {
    auto current = process();
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(current, 0x1122334455667788, 17);
    const auto normal = *handles.insert(thread);
    const auto wrong_type = *handles.insert(current);
    StartupQueries kernel(current, handles, thread);
    for (auto handle : {normal, current_thread_handle}) {
        auto input = registers();
        input[0] = UINT64_MAX; // The kernel does not dereference this SDK output pointer.
        input[1] = upper | handle;
        CHECK(kernel.dispatch(0x0c, input) == QueryDisposition::handled);
        check_reply(input, 0, upper | 17);
        input = registers();
        input[1] = upper | handle;
        CHECK(kernel.dispatch(0x25, input) == QueryDisposition::handled);
        check_reply(input, 0, 0x1122334455667788);
    }
    CHECK(thread->set_priority(19));
    CHECK(!thread->set_priority(64));
    CHECK(handles.close(normal));
    for (auto handle : {normal, wrong_type, Handle{0}, current_process_handle, Handle{0x40008000}}) {
        for (std::uint16_t svc : {0x0c, 0x25}) {
            auto input = registers();
            input[1] = upper | handle;
            CHECK(kernel.dispatch(svc, input) == QueryDisposition::handled);
            check_reply(input, 0xe401, upper | handle);
        }
    }
    auto input = registers();
    input[1] = upper | current_thread_handle;
    CHECK(kernel.dispatch(0x0c, input) == QueryDisposition::handled);
    check_reply(input, 0, upper | 19);

    // svcGetThreadCoreMask: the handle is the third SVC argument; the ideal core
    // and affinity mask are returned in x1/x2.
    input = registers();
    input[2] = current_thread_handle;
    CHECK(kernel.dispatch(0x0e, input) == QueryDisposition::handled);
    CHECK(input[0] == 0 && input[1] == 0);
    const auto allowed = current->configuration().allowed_core_mask;
    CHECK(input[2] == allowed);
    // svcSetThreadCoreMask(handle, ideal_core, mask) stores the allowed subset.
    input = registers();
    input[0] = current_thread_handle;
    input[1] = 0;
    input[2] = 1;
    CHECK(kernel.dispatch(0x0f, input) == QueryDisposition::handled);
    CHECK(input[0] == 0 && thread->affinity_mask() == (allowed & 1));
    input = registers();
    input[0] = Handle{0x5000};
    input[2] = 1;
    CHECK(kernel.dispatch(0x0f, input) == QueryDisposition::handled);
    CHECK(input[0] == 0xe401);
    input = registers();
    input[2] = Handle{0x5000};
    CHECK(kernel.dispatch(0x0e, input) == QueryDisposition::handled);
    CHECK(input[0] == 0xe401);
}

void test_current_processor() {
    auto config = configuration();
    config.allowed_core_mask = 0x6;
    auto current = ProcessRecord::create(config, accounting(), blocks()).process;
    CHECK(current != nullptr);
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(current, 1, 17);
    StartupQueries kernel(current, handles, thread);
    const auto original = registers();
    auto input = original;
    CHECK(!thread->current_processor());
    CHECK(kernel.dispatch(0x10, input) == QueryDisposition::unsupported_operation);
    CHECK(input == original);
    CHECK(!thread->bind_current_processor(0));
    CHECK(!thread->bind_current_processor(64));
    CHECK(!thread->bind_current_processor(UINT32_MAX));
    CHECK(!thread->current_processor());
    thread->set_affinity_mask(0x2);
    CHECK(thread->bind_current_processor(1));
    CHECK(!thread->bind_current_processor(2));
    CHECK(thread->current_processor() == 1);
    CHECK(kernel.dispatch(0x10, input) == QueryDisposition::handled);
    CHECK(input[0] == 1);
    for (std::size_t index = 1; index < input.size(); ++index) CHECK(input[index] == 0);

    // The process permission check still applies to an overbroad host-supplied mask.
    thread->set_affinity_mask(0x7);
    CHECK(!thread->bind_current_processor(0));
    CHECK(thread->current_processor() == 1);
    CHECK(thread->bind_current_processor(2));
    input = original;
    CHECK(kernel.dispatch(0x10, input) == QueryDisposition::handled);
    CHECK(input[0] == 2);

    // A successful guest mask change excludes the old binding. Querying a fabricated
    // fallback processor is forbidden until the runtime assigns a new admitted core.
    input = registers();
    input[0] = current_thread_handle;
    input[1] = 1;
    input[2] = 0x2;
    CHECK(kernel.dispatch(0x0f, input) == QueryDisposition::handled);
    CHECK(input[0] == 0);
    CHECK(!thread->current_processor());
    input = original;
    CHECK(kernel.dispatch(0x10, input) == QueryDisposition::unsupported_operation);
    CHECK(input == original);
    CHECK(thread->bind_current_processor(1));
    thread->set_affinity_mask(0x6);
    CHECK(thread->current_processor() == 1);

    // A real handle updates its target, while the current-thread pseudo-handle above
    // updates the caller. Changing another thread must not invalidate the caller's lane.
    auto other = std::make_shared<ThreadRecord>(current, 2, 17);
    const auto other_handle = handles.insert(other);
    CHECK(other_handle.has_value());
    CHECK(other->bind_current_processor(2));
    input = registers();
    input[0] = *other_handle;
    input[1] = 1;
    input[2] = 0x2;
    CHECK(kernel.dispatch(0x0f, input) == QueryDisposition::handled);
    CHECK(input[0] == 0);
    CHECK(other->affinity_mask() == 0x2);
    CHECK(!other->current_processor());
    CHECK(thread->affinity_mask() == 0x6);
    CHECK(thread->current_processor() == 1);

    thread->unbind_current_processor();
    CHECK(!thread->current_processor());
    input = original;
    CHECK(kernel.dispatch(0x10, input) == QueryDisposition::unsupported_operation);
    CHECK(input == original);
}

void test_per_processor_allocations() {
    auto config = configuration();
    config.allowed_core_mask = 0xf;
    auto current = ProcessRecord::create(config, accounting(), blocks()).process;
    CHECK(current != nullptr);
    HandleTable handles(8);
    std::array<std::shared_ptr<ThreadRecord>, 3> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] = std::make_shared<ThreadRecord>(current, index + 1, 17);
        threads[index]->set_affinity_mask(std::uint64_t{1} << (index + 1));
        CHECK(threads[index]->bind_current_processor(static_cast<std::uint32_t>(index + 1)));
    }
    std::array<std::atomic<std::uint32_t>, 4> cursors{};
    std::barrier start(3), loaded(3);
    struct Allocation {
        QueryDisposition disposition;
        std::uint64_t processor;
        std::uint64_t slot;
    };
    std::array<std::future<Allocation>, 3> allocations;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        allocations[index] = std::async(std::launch::async, [&, index] {
            StartupQueries kernel(current, handles, threads[index]);
            auto input = registers();
            start.arrive_and_wait();
            const auto disposition = kernel.dispatch(0x10, input);
            const auto processor = input[0];
            const auto lane = processor < cursors.size() ? processor : 0;
            // Model a per-processor cursor's separate load and store. Atomic storage
            // keeps a broken shared-lane implementation free of C++ data races; the
            // barrier makes all callers read before any caller advances its cursor.
            const auto offset = cursors[lane].load(std::memory_order_relaxed);
            loaded.arrive_and_wait();
            cursors[lane].store(offset + 1, std::memory_order_relaxed);
            return Allocation{disposition, processor, lane * 1024 + offset};
        });
    }
    std::array<std::uint64_t, 3> slots{};
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        REQUIRE_READY(allocations[index]);
        const auto allocation = allocations[index].get();
        CHECK(allocation.disposition == QueryDisposition::handled);
        CHECK(allocation.processor == index + 1);
        CHECK(allocation.slot == (index + 1) * 1024);
        slots[index] = allocation.slot;
    }
    CHECK(slots[0] != slots[1] && slots[0] != slots[2] && slots[1] != slots[2]);
    CHECK(cursors[0].load() == 0);
    for (std::size_t index = 1; index < cursors.size(); ++index)
        CHECK(cursors[index].load() == 1);
}

void test_memory_queries() {
    auto current = process();
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(current, 42, 17);
    Output output;
    StartupQueries kernel(current, handles, thread,
                          {Output::write, Output::read, &output, Output::compare_exchange,
                           Output::read_double_word});
    {
        StartupQueries no_writer(current, handles, thread);
        auto input = registers();
        input[0] = output_address;
        input[2] = stack_address;
        const auto before = input;
        CHECK(current->writable(output_address, 40));
        CHECK(no_writer.dispatch(0x06, input) == QueryDisposition::unsupported_operation);
        CHECK(input == before);
    }
    const auto query = [&](std::uint64_t destination, std::uint64_t address) {
        auto input = registers();
        input[0] = destination;
        input[1] = upper | 0x12345678; // The low word is not a kernel output pointer.
        input[2] = address;
        CHECK(kernel.dispatch(0x06, input) == QueryDisposition::handled);
        return input;
    };
    // svcWaitProcessWideKeyAtomic(mutex_word, condition_word, tag, timeout): the wait releases
    // the mutex, arms the condition word and reports the kernel timeout result when nothing
    // signals it. svcSignalProcessWideKey on a key without waiters clears that word. The
    // signal handoff needs the atomic compare-exchange; without it the SVC is reported as
    // unsupported and leaves the registers and guest words alone.
    {
        const auto mutex_address = output_address;
        const auto key_address = output_address + 4;
        output.reset();
        const auto store = [&](std::uint64_t address, std::uint32_t value) {
            for (unsigned index = 0; index != 4; ++index)
                output.bytes[address - output_address + index] =
                    static_cast<std::byte>((value >> (8 * index)) & 255);
        };
        store(mutex_address, 0x1234);
        store(key_address, 0);
        output.completed = 0;
        auto wait = registers();
        wait[0] = mutex_address;
        wait[1] = key_address;
        wait[2] = 0x5678;
        wait[3] = 1'000'000;
        CHECK(kernel.dispatch(0x1c, wait) == QueryDisposition::handled);
        CHECK(wait[0] == 0xea01);
        CHECK(output.word(0, 4) == 0);
        CHECK(output.word(4, 4) == 1);
        auto signal = registers();
        signal[0] = key_address;
        signal[1] = 1;
        output.completed = 0;
        StartupQueries no_cas(current, handles, thread,
                              {Output::write, Output::read, &output});
        const auto before = signal;
        CHECK(no_cas.dispatch(0x1d, signal) == QueryDisposition::unsupported_operation);
        CHECK(signal == before);
        CHECK(output.word(4, 4) == 1);
        CHECK(kernel.dispatch(0x1d, signal) == QueryDisposition::handled);
        CHECK(signal[0] == 0);
        CHECK(output.word(4, 4) == 0);
    }

    // A genuinely live waiter exercises the callback integration end to end: the
    // worker blocks in svcWaitProcessWideKeyAtomic, the signal's handoff runs the
    // compare-exchange thunk into this test's copy_out context, and the woken SVC
    // observes the grant.
    {
        const auto mutex_address = output_address;
        const auto key_address = output_address + 4;
        output.reset();
        const auto store = [&](std::uint64_t address, std::uint32_t value) {
            for (unsigned index = 0; index != 4; ++index)
                output.bytes[address - output_address + index] =
                    static_cast<std::byte>((value >> (8 * index)) & 255);
        };
        store(mutex_address, 0x1234);
        store(key_address, 0);
        auto wait = registers();
        wait[0] = mutex_address;
        wait[1] = key_address;
        wait[2] = 0x5678;
        wait[3] = static_cast<std::uint64_t>(-1);
        auto worker = std::async(std::launch::async, [&] {
            CHECK(kernel.dispatch(0x1c, wait) == QueryDisposition::handled);
            return wait;
        });
        REQUIRE_SETTLES(kernel.process_wide_keys()->waiters(key_address) == 1);
        auto signal = registers();
        signal[0] = key_address;
        signal[1] = 1;
        CHECK(kernel.dispatch(0x1d, signal) == QueryDisposition::handled);
        CHECK(signal[0] == 0);
        REQUIRE_READY(worker);
        const auto done = worker.get();
        CHECK(done[0] == 0);
        // The handoff exchanged the waiter's tag through this test's own callback,
        // and the condition word is cleared once no waiter is left.
        CHECK(output.cas_calls == 1);
        CHECK(output.cas_address == mutex_address);
        CHECK(output.cas_desired == 0x5678);
        CHECK(output.word(0, 4) == 0x5678);
        CHECK(output.word(4, 4) == 0);
        CHECK(kernel.process_wide_keys()->waiters(key_address) == 0);
    }

    // A mutex word the compare-exchange can no longer access fails only that
    // waiter: its SVC reports the guest error and clears scratch registers; the signal
    // itself stays successful, and the failed handoff changes no word.
    {
        const auto mutex_address = output_address;
        const auto key_address = output_address + 4;
        output.reset();
        const auto store = [&](std::uint64_t address, std::uint32_t value) {
            for (unsigned index = 0; index != 4; ++index)
                output.bytes[address - output_address + index] =
                    static_cast<std::byte>((value >> (8 * index)) & 255);
        };
        store(mutex_address, 0x1234);
        store(key_address, 0);
        auto wait = registers();
        wait[0] = mutex_address;
        wait[1] = key_address;
        wait[2] = 0x5678;
        wait[3] = static_cast<std::uint64_t>(-1);
        auto worker = std::async(std::launch::async, [&] {
            CHECK(kernel.dispatch(0x1c, wait) == QueryDisposition::handled);
            return wait;
        });
        REQUIRE_SETTLES(kernel.process_wide_keys()->waiters(key_address) == 1);
        output.force_cas_invalid = true;
        auto signal = registers();
        signal[0] = key_address;
        signal[1] = 1;
        CHECK(kernel.dispatch(0x1d, signal) == QueryDisposition::handled);
        CHECK(signal[0] == 0);
        REQUIRE_READY(worker);
        const auto done = worker.get();
        CHECK(done[0] == 0xd401);
        CHECK(done[1] == 0);
        CHECK(output.cas_calls == 1);
        // The released word is still the zero the wait left; the failed handoff
        // wrote nothing.
        CHECK(output.word(0, 4) == 0);
        CHECK(output.word(4, 4) == 0);
        CHECK(kernel.process_wide_keys()->waiters(key_address) == 0);
    }

    // svcWaitForAddress and svcSignalToAddress use the same guest words through the
    // address arbiter. A condition that does not hold is an invalid-state result, a zero
    // timeout expires immediately and the signal types update an equal word.
    {
        output.reset();
        const auto address = output_address + 8;
        const auto store = [&](std::uint32_t value) {
            for (unsigned index = 0; index != 4; ++index)
                output.bytes[address - output_address + index] =
                    static_cast<std::byte>((value >> (8 * index)) & 255);
            output.completed = 0;
        };
        const auto load = [&] { return output.word(8, 4); };
        const auto wait = [&](std::uint32_t type, std::int32_t value, std::int64_t timeout) {
            auto input = registers();
            input[0] = address;
            input[1] = type;
            input[2] = static_cast<std::uint32_t>(value);
            input[3] = static_cast<std::uint64_t>(timeout);
            CHECK(kernel.dispatch(0x34, input) == QueryDisposition::handled);
            return input;
        };
        const auto signal = [&](std::uint32_t type, std::int32_t value, std::int32_t count) {
            auto input = registers();
            input[0] = address;
            input[1] = type;
            input[2] = static_cast<std::uint32_t>(value);
            input[3] = static_cast<std::uint32_t>(count);
            CHECK(kernel.dispatch(0x35, input) == QueryDisposition::handled);
            return input;
        };
        store(5);
        CHECK(wait(0, 3, -1)[0] == 0xfa01);
        CHECK(wait(2, 4, -1)[0] == 0xfa01);
        CHECK(wait(2, 5, 1'000'000)[0] == 0xea01);
        CHECK(wait(1, 10, 0)[0] == 0xea01);
        CHECK(load() == 4);
        store(5);
        CHECK(signal(1, 5, 1)[0] == 0 && load() == 6);
        store(5);
        CHECK(signal(1, 4, 1)[0] == 0xfa01 && load() == 5);
        store(5);
        CHECK(signal(2, 5, -1)[0] == 0 && load() == 6);
        store(5);
        CHECK(signal(0, 0, 1)[0] == 0 && load() == 5);
        // Type 3 compares the 64-bit word and needs an eight-byte-aligned address.
        const auto wide = output_address + 16;
        const auto store64 = [&](std::uint64_t value) {
            for (unsigned index = 0; index != 8; ++index)
                output.bytes[wide - output_address + index] =
                    static_cast<std::byte>((value >> (8 * index)) & 255);
            output.completed = 0;
        };
        const auto wait64 = [&](std::uint64_t value, std::int64_t timeout) {
            auto input = registers();
            input[0] = wide;
            input[1] = 3;
            input[2] = value;
            input[3] = static_cast<std::uint64_t>(timeout);
            CHECK(kernel.dispatch(0x34, input) == QueryDisposition::handled);
            return input;
        };
        store64(5);
        CHECK(wait64(4, -1)[0] == 0xfa01);
        CHECK(wait64(5, 1'000'000)[0] == 0xea01);
        auto unaligned = registers();
        unaligned[0] = wide + 4;
        unaligned[1] = 3;
        unaligned[2] = 5;
        unaligned[3] = static_cast<std::uint64_t>(-1);
        CHECK(kernel.dispatch(0x34, unaligned) == QueryDisposition::handled);
        CHECK(unaligned[0] == 0xcc01);
    }

    // svcGetInfo(type 8, handle 0): the SDK asks with a zero handle whether a debugger is
    // attached, so the current process answers it.
    {
        auto info = registers();
        info[0] = output_address;
        info[1] = 8;
        info[2] = 0;
        info[3] = 0;
        output.reset();
        CHECK(kernel.dispatch(0x29, info) == QueryDisposition::handled);
        // The value travels in x1; the SDK's wrapper stores it at the caller's pointer.
        CHECK(info[1] == 0);
    }

    // svcArbitrateLock/Unlock follow the SDK's contended-mutex protocol: the caller's
    // user-mode CAS already handled the free word, so the kernel returns success without
    // writing when the word no longer names the owner with the waiters bit. A contended word
    // blocks until the owner unlocks. These hooks map one synthetic guest address to a
    // host word independently of the host address space layout.
    {
        std::uint32_t mutex_word = 0;
        const auto owner_handle = *handles.insert(thread);
        const auto real_read = [](void* context, std::uint64_t address,
                                  std::span<std::byte, 4> value) noexcept {
            if (address != output_address) return false;
            std::memcpy(value.data(), context, value.size());
            return true;
        };
        const auto real_write = [](void* context, std::uint64_t address,
                                   std::span<const std::byte, 4> value) noexcept {
            if (address != output_address) return false;
            std::memcpy(context, value.data(), value.size());
            return true;
        };
        StartupQueries locked(current, handles, thread, {real_write, real_read, &mutex_word});
        // A free word: the caller's CAS owns the write and the kernel does not touch it.
        mutex_word = 0;
        auto lock = registers();
        lock[0] = owner_handle;
        lock[1] = output_address;
        lock[2] = 0x5678;  // caller thread handle
        CHECK(locked.dispatch(0x1a, lock) == QueryDisposition::handled);
        CHECK(lock[0] == 0 && mutex_word == 0);
        // A contended word blocks until the owner hands the mutex tag to the caller.
        mutex_word = owner_handle | 0x40000000u;
        std::atomic<bool> locked_in{false};
        std::atomic<bool> done{false};
        std::thread waiter([&] {
            auto contended = registers();
            contended[0] = owner_handle;
            contended[1] = output_address;
            contended[2] = 0x5678;
            locked_in.store(true);
            CHECK(locked.dispatch(0x1a, contended) == QueryDisposition::handled);
            CHECK(contended[0] == 0);
            done.store(true);
        });
        while (!locked_in.load()) std::this_thread::yield();
        REQUIRE_SETTLES(locked.process_wide_keys()->mutex_waiters(output_address) == 1);
        auto unlock = registers();
        unlock[0] = output_address;
        CHECK(locked.dispatch(0x1b, unlock) == QueryDisposition::handled);
        CHECK(unlock[0] == 0 && mutex_word == 0x5678);
        // Bounded handshake: a lost wakeup fails the test instead of stalling the process.
        for (unsigned spin = 0; spin != 2000 && !done.load(); ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!done.load()) {
            waiter.detach();
            std::fprintf(stderr, "ArbitrateLock waiter did not complete.\n");
            CHECK(done.load());
            return;
        }
        waiter.join();
        CHECK(handles.close(owner_handle));
    }

    // svcCreateTransferMemory(out_handle, address, size, permission) returns the
    // new handle in x1 and leaves x0 as the result.
    const auto transfer = [&](std::uint64_t address, std::uint64_t size, std::uint64_t permission) {
        auto input = registers();
        input[0] = output_address;
        input[1] = address;
        input[2] = size;
        input[3] = permission;
        CHECK(kernel.dispatch(0x15, input) == QueryDisposition::handled);
        return input;
    };
    // Argument checks in kernel order: address, size, overflow, then permission
    // (none, read or read-write).
    CHECK(transfer(output_address + 8, 0, 2)[0] == 0xcc01);
    CHECK(transfer(output_address, 0, 2)[0] == 0xca01);
    CHECK(transfer(output_address, 0x1000 + 8, 0)[0] == 0xca01);
    CHECK(transfer(0xfffffffffffff000ull, 0x2000, 2)[0] == 0xd401);
    CHECK(transfer(output_address, 0x1000, 2)[0] == 0xd801);
    CHECK(transfer(output_address, 0x1000, 4)[0] == 0xd801);
    // Only a read-write data mapping without attributes may be transferred.
    CHECK(transfer(stack_address, 0x4000, 0)[0] == 0xd401);
    CHECK(transfer(code_address, 0x1000, 0)[0] == 0xd401);
    CHECK(transfer(0x7f0000000000, 0x1000, 0)[0] == 0xd401);
    CHECK(transfer(output_address + 0x1000, 0x2000, 0)[0] == 0xd401);
    // The owner's range stays locked while the object lives, so it cannot be
    // transferred again; the last reference unlocks it.
    const auto created = transfer(output_address + 0x1000, 0x1000, 0);
    CHECK(created[0] == 0 && created[1] != 0);
    CHECK(handles.lookup_as<TransferMemoryRecord>(created[1]) != nullptr);
    CHECK(handles.lookup_as<TransferMemoryRecord>(created[1])->range().address == output_address + 0x1000);
    CHECK(handles.lookup_as<TransferMemoryRecord>(created[1])->range().size == 0x1000);
    CHECK(current->query_memory(output_address + 0x1000).attributes == 1);
    CHECK(current->query_memory(output_address).attributes == 0);
    CHECK(transfer(output_address, 0x2000, 3)[0] == 0xd401);
    CHECK(handles.close(static_cast<Handle>(created[1])));
    CHECK(current->query_memory(output_address + 0x1000).attributes == 0);
    const auto whole = transfer(output_address, 0x2000, 3);
    CHECK(whole[0] == 0 && handles.close(static_cast<Handle>(whole[1])));
    CHECK(current->query_memory(output_address).attributes == 0);

    output.reset();
    check_reply(query(output_address + 1, stack_address + 23), 0, upper);
    CHECK(output.completed == 10);
    CHECK(output.word(1, 8) == stack_address);
    CHECK(output.word(9, 8) == 0x4000);
    CHECK(output.word(17, 4) == 0x0b);
    CHECK(output.word(21, 4) == 0x18);
    CHECK(output.word(25, 4) == 3);
    CHECK(output.word(29, 4) == 0x1234);
    CHECK(output.word(33, 4) == 0xabcd);
    CHECK(output.word(37, 4) == 0);
    CHECK(output.bytes[0] == std::byte{0xcd} && output.bytes[41] == std::byte{0xcd});

    for (auto address : {std::uint64_t{0}, code_address + 0x2123,
                          ProcessRecord::query_end, std::uint64_t{UINT64_MAX}}) {
        output.reset();
        check_reply(query(output_address, address), 0, upper);
        const auto expected = current->query_memory(address);
        CHECK(output.word(0, 8) == expected.range.address);
        CHECK(output.word(8, 8) == expected.range.size);
        CHECK(output.word(16, 4) == static_cast<std::uint8_t>(expected.type));
        CHECK(output.word(24, 4) == 0);
        CHECK(output.word(36, 4) == 0);
    }
    // An output outside the process address space fails before the query and keeps
    // x1 whole; an unmapped output inside it fails the copy after the query.
    for (auto destination : {std::uint64_t{0}, ProcessRecord::aslr_start - 8,
                              ProcessRecord::query_end - 32, ProcessRecord::query_end,
                              std::uint64_t{UINT64_MAX - 1}}) {
        output.reset();
        check_reply(query(destination, stack_address), 0xe601, upper | 0x12345678);
        CHECK(output.completed == 0);
    }
    output.reset();
    check_reply(query(code_address, stack_address), 0xe601, upper);
    CHECK(output.completed == 0);

    // Only the first aligned word is writable. A later fault keeps that prefix.
    auto map = blocks();
    map[2].range.size = 0x1000;
    CHECK(current->replace_memory_map(map) == ProcessStateError::none);
    output.reset();
    check_reply(query(output_address + 0xffc, stack_address), 0xe601, upper);
    CHECK(output.completed == 1);
    CHECK(output.word(0xffc, 4) == static_cast<std::uint32_t>(stack_address));
    CHECK(output.word(0x1000, 4) == 0xcdcdcdcd);

    CHECK(current->replace_memory_map(blocks()) == ProcessStateError::none);
    output.reset(3); // A backing failure after the metadata check also preserves completed words.
    check_reply(query(output_address, stack_address), 0xe601, upper);
    CHECK(output.completed == 3);
    CHECK(output.word(0, 8) == stack_address);
    CHECK(output.word(8, 4) == 0x4000);
    CHECK(output.word(12, 4) == 0xcdcdcdcd);
}

// The SVC wait timeout is the guest timeout converted once with the process's
// counter rate; a second conversion would be a different duration. The 1 ns
// waits below check the SVC wiring on each platform; the waits have no injected
// clock, so the duration they pass down is checked through svc_wait_timeout.
void test_svc_timeout_rounded_once() {
    CHECK(kernel_timeout(1) == std::chrono::nanoseconds(156));
    CHECK(kernel_timeout(kernel_timeout(1).count()) == std::chrono::nanoseconds(260));
    CHECK(kernel_timeout(1, switch2_counter) == std::chrono::nanoseconds(96));
    CHECK(kernel_timeout(kernel_timeout(1, switch2_counter).count(), switch2_counter) ==
          std::chrono::nanoseconds(160));

    CHECK(svc_wait_timeout(nullptr, 1) == 156);

    const auto run = [](armigo::GuestPlatform platform) {
        auto config = configuration();
        config.platform = platform;
        const auto built = ProcessRecord::create(config, accounting(), blocks());
        CHECK(built.ok());
        if (!built.ok()) return;
        const auto once = platform == armigo::GuestPlatform::switch2 ? 96 : 156;
        CHECK(svc_wait_timeout(built.process.get(), 1) == once);
        CHECK(svc_wait_timeout(built.process.get(), 0) == 0);
        CHECK(svc_wait_timeout(built.process.get(), -1) == -1);
        CHECK(svc_wait_timeout(built.process.get(), 1'000'000) ==
              (platform == armigo::GuestPlatform::switch2 ? 1'000'064 : 1'000'104));
        HandleTable handles(4);
        auto thread = std::make_shared<ThreadRecord>(built.process, 1, 1);
        Output output;
        output.reset();
        const auto address = output_address + 8;
        constexpr std::uint32_t word = 5;
        for (unsigned index = 0; index != 4; ++index)
            output.bytes[address - output_address + index] =
                static_cast<std::byte>((word >> (8 * index)) & 255);
        StartupQueries kernel(built.process, handles, thread,
                              {Output::write, Output::read, &output, Output::compare_exchange,
                               Output::read_double_word});
        auto wait = registers();
        wait[0] = address;
        wait[1] = 2;
        wait[2] = word;
        wait[3] = 1;
        CHECK(kernel.dispatch(0x34, wait) == QueryDisposition::handled);
        CHECK(wait[0] == 0xea01);

        auto sync = registers();
        sync[2] = 0;
        sync[3] = 1;
        CHECK(kernel.dispatch(0x18, sync) == QueryDisposition::handled);
        CHECK(sync[0] == 0xea01);

        sync = registers();
        sync[2] = 0;
        sync[3] = 0;
        CHECK(kernel.dispatch(0x18, sync) == QueryDisposition::handled);
        CHECK(sync[0] == 0xea01);
    };
    run(armigo::GuestPlatform::switch1);
    run(armigo::GuestPlatform::switch2);
}
} // namespace

int main() {
    arm_watchdog();
    test_process_state();
    test_compatible_memory_spans();
    test_get_info();
    test_thread_queries();
    test_current_processor();
    test_per_processor_allocations();
    test_memory_queries();
    test_svc_timeout_rounded_once();
    auto compact = configuration();
    compact.allocation_region = {0x100000000, std::uint64_t{5} << 30};
    compact.alias_address = compact.allocation_region.address + (256ull << 20);
    compact.base_alias_size = 512ull << 20;
    compact.stack_address = compact.allocation_region.address + (768ull << 20);
    compact.stack_size = 256ull << 20;
    compact.heap_address = compact.allocation_region.address + (1ull << 30);
    compact.heap_size = 4ull << 30;
    auto host = ProcessRecord::create(compact, accounting(), {});
    CHECK(host.ok());
    if (host.ok()) {
        HandleTable handles(2);
        auto thread = std::make_shared<ThreadRecord>(host.process, 1, 44);
        StartupQueries queries(host.process, handles, thread);
        check_reply(get_info(queries, 3), 0, compact.base_alias_size);
        check_reply(get_info(queries, 5), 0, compact.heap_size);
        check_reply(get_info(queries, 13), 0, compact.allocation_region.size);
        check_reply(get_info(queries, 15), 0, compact.stack_size);
    }
    compact.heap_size += guest_page_size;
    CHECK(ProcessRecord::create(compact, accounting(), {}).error == ProcessStateError::invalid_layout);
    // svcClearEvent clears an event through either end whether or not it was
    // signalled; any other object or no object is an invalid handle. Only the
    // writable end accepts svcSignalEvent.
    if (host.ok()) {
        HandleTable events(6);
        auto thread = std::make_shared<ThreadRecord>(host.process, 1, 44);
        StartupQueries queries(host.process, events, thread);
        auto event = std::make_shared<Event>(true);
        const auto handle = events.insert(event).value();
        const auto other = events.insert(thread).value();
        const auto clear = [&](Handle target) {
            QueryRegisters registers{};
            registers[0] = target;
            CHECK(queries.dispatch(0x12, registers) == QueryDisposition::handled);
            return registers[0];
        };
        CHECK(clear(handle) == 0 && !event->is_signaled());
        CHECK(clear(handle) == 0 && !event->is_signaled());
        CHECK(clear(0x1234) == 0xe401);
        CHECK(clear(other) == 0xe401);
        unsigned observed = 0;
        const auto writable = events.insert(std::make_shared<WritableEvent>(event, [&] { ++observed; })).value();
        const auto signal = [&](Handle target) {
            QueryRegisters registers{};
            registers[0] = target;
            CHECK(queries.dispatch(0x11, registers) == QueryDisposition::handled);
            return registers[0];
        };
        CHECK(signal(handle) == 0xe401 && !event->is_signaled());
        CHECK(signal(other) == 0xe401);
        CHECK(signal(writable) == 0 && event->is_signaled() && observed == 1);
        CHECK(signal(writable) == 0 && event->is_signaled() && observed == 2);
        CHECK(clear(writable) == 0 && !event->is_signaled());
    }
    std::printf("Horizon startup query failures: %u\n", failures);
        {
        char text[128]{};
        const auto length = armigo::horizon::describe_break(0, 0x7030103d10, 4, text, sizeof text);
        CHECK(length != 0);
        CHECK(std::strstr(text, "svcBreak") != nullptr);
        CHECK(std::strstr(text, "7030103d10") != nullptr);
        char small[8]{};
        CHECK(armigo::horizon::describe_break(0, 1, 2, small, sizeof small) < sizeof small);
        CHECK(armigo::horizon::describe_break(0, 1, 2, nullptr, 0) == 0);
    }

return failures == 0 ? 0 : 1;
}
