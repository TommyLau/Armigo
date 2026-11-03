// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/read_only_shared_memory.h>
#include <armigo/runtime/guest_memory.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace allocation_fault {
bool armed = false;
bool triggered = false;
std::size_t remaining = 0;

bool should_fail() noexcept {
    if (!armed) return false;
    if (remaining != 0) {
        --remaining;
        return false;
    }
    armed = false;
    triggered = true;
    return true;
}

void arm(std::size_t fail_after) noexcept {
    remaining = fail_after;
    triggered = false;
    armed = true;
}

void disarm() noexcept { armed = false; }
} // namespace allocation_fault

// Fault injection is confined to this test executable. One allocation fails;
// rollback allocations remain available so the original state can be restored.
void* operator new(std::size_t size) {
    if (allocation_fault::should_fail()) throw std::bad_alloc();
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {

using namespace armigo;
using namespace armigo::runtime;

constexpr std::uint64_t mib = 1024 * 1024;
constexpr std::size_t sweep_limit = 96;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

std::uint64_t native_page_size() {
    const long page = ::sysconf(_SC_PAGESIZE);
    CHECK(page > 0);
    return page > 0 ? static_cast<std::uint64_t>(page) : horizon::guest_page_size;
}

LaunchProfile profile() {
    LaunchProfile result;
    result.origin = ProfileOrigin::synthetic_test;
    result.process.allowed_core_mask = 1;
    result.process.heap_address = 0;
    result.process.alias_address = 0;
    result.process.stack_address = 0;
    result.process.heap_size = 8 * mib;
    result.process.base_alias_size = 2 * mib;
    result.process.stack_size = 2 * mib;
    result.permissions.handle_table_capacity = 64;
    return result;
}

struct Fixture {
    std::shared_ptr<horizon::HandleTable> handles =
        std::make_shared<horizon::HandleTable>(64);
    std::shared_ptr<GuestMemory> memory;

    Fixture() {
        auto made = create_guest_memory(profile(), {16 * mib}, nullptr, handles);
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
        if (auto* value = std::get_if<std::shared_ptr<GuestMemory>>(&made)) {
            memory = std::move(*value);
        }
    }
};

bool same_blocks(std::span<const horizon::MemoryBlock> left,
                 std::span<const horizon::MemoryBlock> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto& a = left[index];
        const auto& b = right[index];
        if (a.range.address != b.range.address || a.range.size != b.range.size ||
            a.type != b.type || a.attributes != b.attributes ||
            a.permissions != b.permissions || a.ipc_references != b.ipc_references ||
            a.device_references != b.device_references || a.merge_disable != b.merge_disable) return false;
    }
    return true;
}

struct Snapshot {
    std::vector<horizon::MemoryBlock> blocks;
    std::uint64_t used = 0;
    std::uint64_t total = 0;
};

Snapshot snapshot(const GuestMemory& memory) {
    const auto process = memory.process();
    return {process->memory_map(), process->used_memory(), process->total_memory()};
}

bool unchanged(const GuestMemory& memory, const Snapshot& before) {
    const auto process = memory.process();
    const auto blocks = process->memory_map();
    return same_blocks(blocks, before.blocks) && process->used_memory() == before.used &&
           process->total_memory() == before.total;
}

GuestMemoryPin pin(GuestMemory& memory, horizon::AddressRange range,
                   memory::Permission permission) {
    auto result = memory.pin(range, permission);
    CHECK(std::holds_alternative<GuestMemoryPin>(result));
    return std::holds_alternative<GuestMemoryPin>(result)
               ? std::get<GuestMemoryPin>(std::move(result)) : GuestMemoryPin{};
}

bool handled(const LaunchResult<horizon::QueryDisposition>& result) {
    const auto* disposition = std::get_if<horizon::QueryDisposition>(&result);
    return disposition && *disposition == horizon::QueryDisposition::handled;
}

bool host_failure(const LaunchResult<horizon::QueryDisposition>& result) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && (failure->code == LaunchError::resource_limit ||
                       failure->code == LaunchError::host_failure);
}

bool host_failure(const LaunchStatus& result) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && (failure->code == LaunchError::resource_limit ||
                       failure->code == LaunchError::host_failure);
}

bool host_failure(const LaunchResult<GuestMemoryPin>& result) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && (failure->code == LaunchError::resource_limit ||
                       failure->code == LaunchError::host_failure);
}

horizon::QueryRegisters seeded_registers() {
    horizon::QueryRegisters result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = 0x9876000000000000ULL + index;
    }
    return result;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint64_t free_stack_address(const GuestMemory& memory, std::uint64_t size) {
    const auto process = memory.process();
    const auto stack = process->stack_region();
    std::uint64_t cursor = stack.address;
    for (const auto& block : process->memory_map()) {
        const auto end = block.range.address + block.range.size;
        if (block.range.address < stack.address + stack.size && end > stack.address) {
            cursor = std::max(cursor, end);
        }
    }
    cursor = align_up(cursor, native_page_size());
    CHECK(cursor >= stack.address && size <= stack.address + stack.size - cursor);
    return cursor;
}

#if defined(__APPLE__)
vm_prot_t native_protection(std::uint64_t address) {
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    CHECK(result == KERN_SUCCESS && region_address <= address &&
          address - region_address < region_size);
    return result == KERN_SUCCESS ? info.protection : VM_PROT_NONE;
}
#else
using vm_prot_t = int;
vm_prot_t native_protection(std::uint64_t) { return 0; }
#endif

std::array<std::byte, 2> sample(GuestMemory& memory, horizon::AddressRange range) {
    auto view = pin(memory, range, memory::Permission::read);
    if (!view) return {};
    return {view->bytes().front(), view->bytes().back()};
}

bool sweep_bootstrap_allocate() {
    const auto page = native_page_size();
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        Fixture fixture;
        if (!fixture.memory) return false;
        const auto before = snapshot(*fixture.memory);
        allocation_fault::arm(failure);
        auto result = fixture.memory->allocate(page, horizon::MemoryType::normal,
            memory::Permission::read | memory::Permission::write);
        allocation_fault::disarm();
        if (std::holds_alternative<GuestMemoryPin>(result)) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(unchanged(*fixture.memory, before));
    }
    return false;
}

bool sweep_bootstrap_release() {
    const auto page = native_page_size();
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        Fixture fixture;
        auto allocated = fixture.memory->allocate(2 * page, horizon::MemoryType::normal,
            memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(allocated));
        if (!std::holds_alternative<GuestMemoryPin>(allocated)) return false;
        auto allocation = std::get<GuestMemoryPin>(std::move(allocated));
        const auto range = allocation->range();
        allocation->writable_bytes().front() = std::byte{0x31};
        allocation->writable_bytes().back() = std::byte{0x62};
        allocation.reset();
        horizon::QueryRegisters protect{};
        protect[0] = range.address + page;
        protect[1] = page;
        protect[2] = 1;
        CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
        const auto before = snapshot(*fixture.memory);
        const auto bytes = sample(*fixture.memory, range);
        const std::array protections{native_protection(range.address),
                                     native_protection(range.address + page)};
        allocation_fault::arm(failure);
        auto result = fixture.memory->release(range);
        allocation_fault::disarm();
        if (std::holds_alternative<std::monostate>(result)) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(unchanged(*fixture.memory, before) && sample(*fixture.memory, range) == bytes &&
              native_protection(range.address) == protections[0] &&
              native_protection(range.address + page) == protections[1]);
    }
    return false;
}

bool sweep_heap(bool shrink) {
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        Fixture fixture;
        std::uint64_t heap = fixture.memory->process()->heap_region().address;
        if (shrink) {
            horizon::QueryRegisters grow{};
            grow[1] = 2 * mib;
            CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
            heap = grow[1];
            auto bytes = pin(*fixture.memory, {heap, 2 * mib},
                             memory::Permission::read | memory::Permission::write);
            bytes->writable_bytes().front() = std::byte{0x41};
            bytes->writable_bytes().back() = std::byte{0x72};
        }
        const horizon::AddressRange range{heap, 2 * mib};
        const auto before = snapshot(*fixture.memory);
        const auto bytes = shrink ? sample(*fixture.memory, range) : std::array<std::byte, 2>{};
        const auto protection = native_protection(heap);
        auto registers = seeded_registers();
        registers[1] = shrink ? 0 : 2 * mib;
        const auto saved_registers = registers;
        allocation_fault::arm(failure);
        auto result = fixture.memory->dispatch(1, registers);
        allocation_fault::disarm();
        if (handled(result) && registers[0] == 0) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(registers == saved_registers && unchanged(*fixture.memory, before));
        CHECK(native_protection(heap) == protection);
        if (shrink) CHECK(sample(*fixture.memory, range) == bytes);
    }
    return false;
}

bool sweep_protect_split() {
    const auto page = native_page_size();
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        Fixture fixture;
        auto allocated = fixture.memory->allocate(2 * page, horizon::MemoryType::normal,
            memory::Permission::read | memory::Permission::write);
        auto allocation = std::get<GuestMemoryPin>(std::move(allocated));
        const auto range = allocation->range();
        allocation->writable_bytes()[page] = std::byte{0x53};
        allocation.reset();
        const auto before = snapshot(*fixture.memory);
        const auto bytes = sample(*fixture.memory, range);
        const auto protection = native_protection(range.address + page);
        auto registers = seeded_registers();
        registers[0] = range.address + page;
        registers[1] = page;
        registers[2] = 1;
        const auto saved_registers = registers;
        allocation_fault::arm(failure);
        auto result = fixture.memory->dispatch(2, registers);
        allocation_fault::disarm();
        if (handled(result) && registers[0] == 0) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(registers == saved_registers && unchanged(*fixture.memory, before));
        CHECK(native_protection(range.address + page) == protection &&
              sample(*fixture.memory, range) == bytes);
    }
    return false;
}

struct AliasSetup {
    Fixture fixture;
    horizon::AddressRange source;
    horizon::AddressRange destination;
};

AliasSetup alias_setup(bool mapped) {
    AliasSetup result;
    horizon::QueryRegisters grow{};
    grow[1] = 2 * mib;
    CHECK(handled(result.fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    result.source = {grow[1], native_page_size()};
    result.destination = {free_stack_address(*result.fixture.memory, result.source.size),
                          result.source.size};
    auto source = pin(*result.fixture.memory, result.source,
                      memory::Permission::read | memory::Permission::write);
    source->writable_bytes().front() = std::byte{0x64};
    source->writable_bytes().back() = std::byte{0x25};
    source.reset();
    if (mapped) {
        horizon::QueryRegisters map{};
        map[0] = result.destination.address;
        map[1] = result.source.address;
        map[2] = result.source.size;
        CHECK(handled(result.fixture.memory->dispatch(4, map)) && map[0] == 0);
    }
    return result;
}

bool sweep_stack(bool unmap) {
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        auto setup = alias_setup(unmap);
        const auto before = snapshot(*setup.fixture.memory);
        const auto visible = unmap ? setup.destination : setup.source;
        const auto bytes = sample(*setup.fixture.memory, visible);
        const auto source_protection = native_protection(setup.source.address);
        const auto destination_protection = native_protection(setup.destination.address);
        auto registers = seeded_registers();
        registers[0] = setup.destination.address;
        registers[1] = setup.source.address;
        registers[2] = setup.source.size;
        const auto saved_registers = registers;
        allocation_fault::arm(failure);
        auto result = setup.fixture.memory->dispatch(unmap ? 5 : 4, registers);
        allocation_fault::disarm();
        if (handled(result) && registers[0] == 0) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        if (!(allocation_fault::triggered && host_failure(result))) {
            const auto* launch = std::get_if<LaunchFailure>(&result);
            std::fprintf(stderr,
                         "stack %s failure point %zu returned %s (%u), triggered=%d\n",
                         unmap ? "unmap" : "map", failure,
                         launch ? launch->operation.c_str() : "handled result",
                         launch ? static_cast<unsigned>(launch->code) : 999u,
                         allocation_fault::triggered ? 1 : 0);
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(registers == saved_registers && unchanged(*setup.fixture.memory, before));
        CHECK(native_protection(setup.source.address) == source_protection &&
              native_protection(setup.destination.address) == destination_protection);
        CHECK(sample(*setup.fixture.memory, visible) == bytes);
    }
    return false;
}

struct SharedSetup {
    Fixture fixture;
    std::shared_ptr<horizon::ReadOnlySharedMemory> object;
    horizon::Handle handle = 0;
    horizon::AddressRange range;
};

SharedSetup shared_setup(bool mapped) {
    constexpr std::uint64_t size = 64 * 1024;
    SharedSetup result;
    std::vector<std::byte> payload(size);
    payload.front() = std::byte{0x75};
    payload.back() = std::byte{0x16};
    result.object = std::make_shared<horizon::ReadOnlySharedMemory>(payload);
    result.handle = result.fixture.handles->insert(result.object).value_or(0);
    result.range = {free_stack_address(*result.fixture.memory, size), size};
    if (mapped) {
        horizon::QueryRegisters map{};
        map[0] = result.handle;
        map[1] = result.range.address;
        map[2] = result.range.size;
        map[3] = 1;
        CHECK(handled(result.fixture.memory->dispatch(0x13, map)) && map[0] == 0);
    }
    return result;
}

bool sweep_shared(bool unmap) {
    for (std::size_t failure = 0; failure < sweep_limit; ++failure) {
        auto setup = shared_setup(unmap);
        if (setup.handle == 0) return false;
        const auto before = snapshot(*setup.fixture.memory);
        const auto bytes = unmap ? sample(*setup.fixture.memory, setup.range)
                                 : std::array<std::byte, 2>{setup.object->bytes().front(),
                                                            setup.object->bytes().back()};
        const auto protection = native_protection(setup.range.address);
        auto registers = seeded_registers();
        registers[0] = setup.handle;
        registers[1] = setup.range.address;
        registers[2] = setup.range.size;
        registers[3] = 1;
        const auto saved_registers = registers;
        allocation_fault::arm(failure);
        auto result = setup.fixture.memory->dispatch(unmap ? 0x14 : 0x13, registers);
        allocation_fault::disarm();
        if (handled(result) && registers[0] == 0) {
            CHECK(!allocation_fault::triggered);
            return true;
        }
        CHECK(allocation_fault::triggered && host_failure(result));
        CHECK(registers == saved_registers && unchanged(*setup.fixture.memory, before));
        CHECK(native_protection(setup.range.address) == protection);
        if (unmap) CHECK(sample(*setup.fixture.memory, setup.range) == bytes);
        CHECK(setup.object->bytes().front() == std::byte{0x75} &&
              setup.object->bytes().back() == std::byte{0x16});
    }
    return false;
}

} // namespace

int main() {
    CHECK(sweep_bootstrap_allocate());
    CHECK(sweep_bootstrap_release());
    CHECK(sweep_heap(false));
    CHECK(sweep_heap(true));
    CHECK(sweep_protect_split());
    CHECK(sweep_stack(false));
    CHECK(sweep_stack(true));
    CHECK(sweep_shared(false));
    CHECK(sweep_shared(true));
    return failures == 0 ? 0 : 1;
}
