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
#include <armigo/memory/write_tracker.h>
#include "process_memory.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
namespace native_fault {
enum class Operation { none, protect, remap };
Operation operation = Operation::none;
std::uint64_t begin = 0, end = 0;
unsigned remaining = 0, failures_left = 0, failures_seen = 0;
bool triggered = false;
std::uint64_t address_limit = 0;
unsigned limit_queries = 0;
unsigned high_alias_attempts = 0;
void (*before_failure)(void*) = nullptr;
void* before_failure_context = nullptr;
void arm(Operation requested, std::uint64_t address, std::uint64_t size, unsigned after, unsigned count = 1) {
    operation = requested; begin = address; end = address + size; remaining = after;
    failures_left = count; failures_seen = 0; triggered = false;
}
bool fail(Operation requested, std::uint64_t address) {
    if (operation != requested || address < begin || address >= end) return false;
    if (remaining) { --remaining; return false; }
    if (--failures_left == 0) operation = Operation::none;
    ++failures_seen; triggered = true;
    if (const auto callback = std::exchange(before_failure, nullptr)) callback(before_failure_context);
    return true;
}
}
namespace remote_reservation {
bool observe = false;
std::uint64_t address = 0, size = 0;
unsigned acquired = 0;
}
// Executable-local symbol overrides inject native failures without adding a
// production fault hook. RTLD_NEXT retains the actual platform implementation.
extern "C" kern_return_t task_info(task_name_t task, task_flavor_t flavor,
    task_info_t info, mach_msg_type_number_t* count) {
    static const auto real = reinterpret_cast<decltype(&task_info)>(dlsym(RTLD_NEXT, "task_info"));
    const auto result = real(task, flavor, info, count);
    if (native_fault::address_limit && result == KERN_SUCCESS && flavor == TASK_VM_INFO &&
        *count >= TASK_VM_INFO_REV2_COUNT) {
        reinterpret_cast<task_vm_info_t>(info)->max_address = native_fault::address_limit;
        ++native_fault::limit_queries;
    }
    return result;
}
extern "C" kern_return_t mach_vm_protect(vm_map_t task, mach_vm_address_t address,
    mach_vm_size_t size, boolean_t maximum, vm_prot_t protection) {
    static const auto real = reinterpret_cast<decltype(&mach_vm_protect)>(dlsym(RTLD_NEXT, "mach_vm_protect"));
    if (native_fault::fail(native_fault::Operation::protect, address)) return KERN_PROTECTION_FAILURE;
    const auto result = real(task, address, size, maximum, protection);
    if (remote_reservation::observe && result == KERN_SUCCESS && !maximum &&
        protection == VM_PROT_NONE && size == (8ull << 30)) {
        remote_reservation::address = address;
        remote_reservation::size = size;
        ++remote_reservation::acquired;
    }
    return result;
}
extern "C" kern_return_t mach_vm_remap(vm_map_t task, mach_vm_address_t* address,
    mach_vm_size_t size, mach_vm_offset_t mask, int flags, vm_map_t source_task,
    mach_vm_address_t source, boolean_t copy, vm_prot_t* current, vm_prot_t* maximum,
    vm_inherit_t inheritance) {
    static const auto real = reinterpret_cast<decltype(&mach_vm_remap)>(dlsym(RTLD_NEXT, "mach_vm_remap"));
    if (native_fault::address_limit && (flags & VM_FLAGS_ANYWHERE) && *address >= native_fault::address_limit) {
        ++native_fault::high_alias_attempts;
        return KERN_NO_SPACE;
    }
    if ((flags & VM_FLAGS_OVERWRITE) && native_fault::fail(native_fault::Operation::remap, *address))
        return KERN_NO_SPACE;
    return real(task, address, size, mask, flags, source_task, source, copy, current, maximum, inheritance);
}
#endif

namespace allocation_fault {
std::array<void*, 16> arenas{};
unsigned live_arenas = 0;
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

void* operator new(std::size_t size, std::align_val_t alignment) {
    if (allocation_fault::should_fail()) throw std::bad_alloc();
    void* pointer = nullptr;
    if (posix_memalign(&pointer, static_cast<std::size_t>(alignment), size ? size : 1)) throw std::bad_alloc();
    if (size == 65536 && static_cast<std::size_t>(alignment) == 4096) {
        for (auto& slot : allocation_fault::arenas) if (!slot) {
            slot = pointer; ++allocation_fault::live_arenas; break;
        }
    }
    return pointer;
}
void operator delete(void* pointer, std::align_val_t) noexcept {
    for (auto& slot : allocation_fault::arenas) if (pointer && slot == pointer) {
        slot = nullptr; --allocation_fault::live_arenas; break;
    }
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

namespace {

using namespace armigo;
using namespace armigo::runtime;

constexpr std::uint64_t mib = 1024 * 1024;
bool private_mode = false;
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
    if (private_mode) {
        result.process.platform = GuestPlatform::switch2;
        result.process.baseline = horizon::Baseline::horizon_23_0_1;
        result.process.private_system_resource_bytes = 65536;
    }
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
    if (private_mode && handled(result)) return true;
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
        CHECK((registers == saved_registers || (private_mode && registers[0] == 0xce01)) && unchanged(*fixture.memory, before));
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
        CHECK((registers == saved_registers || (private_mode && registers[0] == 0xce01)) && unchanged(*fixture.memory, before));
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
        CHECK((registers == saved_registers || (private_mode && registers[0] == 0xce01)) && unchanged(*setup.fixture.memory, before));
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
        CHECK((registers == saved_registers || (private_mode && registers[0] == 0xce01)) && unchanged(*setup.fixture.memory, before));
        CHECK(native_protection(setup.range.address) == protection);
        if (unmap) CHECK(sample(*setup.fixture.memory, setup.range) == bytes);
        CHECK(setup.object->bytes().front() == std::byte{0x75} &&
              setup.object->bytes().back() == std::byte{0x16});
    }
    return false;
}

void private_construction_failures() {
    auto config = profile();
    config.process.platform = GuestPlatform::switch2;
    config.process.baseline = horizon::Baseline::horizon_23_0_1;
    config.process.private_system_resource_bytes = 65536;
    bool completed = false;
    for (std::size_t failed_after = 0; failed_after < sweep_limit; ++failed_after) {
        auto handles = std::make_shared<horizon::HandleTable>(64);
        CHECK(allocation_fault::live_arenas == 0);
        {
            allocation_fault::arm(failed_after);
            auto made = create_guest_memory(config, {16 * mib}, nullptr, handles);
            allocation_fault::disarm();
            if (auto* memory = std::get_if<std::shared_ptr<GuestMemory>>(&made)) {
                CHECK(!allocation_fault::triggered && allocation_fault::live_arenas == 1);
                CHECK((*memory)->process()->used_memory() == 0);
                CHECK((*memory)->process()->total_memory() == 16 * mib - 65536);
                completed = true;
            } else CHECK(allocation_fault::triggered);
        }
        CHECK(allocation_fault::live_arenas == 0);
        if (completed) break;
    }
    CHECK(completed);
    auto handles = std::make_shared<horizon::HandleTable>(64);
    auto too_small = create_guest_memory(config, {32768}, nullptr, handles);
    CHECK(std::holds_alternative<LaunchFailure>(too_small) && allocation_fault::live_arenas == 0);
    auto retry = create_guest_memory(config, {16 * mib}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(retry));
}

void remote_cleanup_after_record_rejection() {
#if defined(__APPLE__)
    auto config = profile();
    config.process.platform = GuestPlatform::switch2;
    config.process.baseline = horizon::Baseline::horizon_23_0_1;
    // ProcessRecord validates this after native owners, including remote, exist.
    config.process.max_memory_blocks = 0;
    auto handles = std::make_shared<horizon::HandleTable>(64);
    remote_reservation::address = remote_reservation::size = 0;
    remote_reservation::acquired = 0;
    remote_reservation::observe = true;
    {
        const auto made = create_guest_memory(config, {16 * mib}, nullptr, handles);
        remote_reservation::observe = false;
        const auto* failure = std::get_if<LaunchFailure>(&made);
        CHECK(failure && failure->operation == "create_guest_memory" &&
              failure->detail == "Native reservation metadata was rejected.");
    }
    CHECK(remote_reservation::acquired == 1 && remote_reservation::address != 0 &&
          remote_reservation::size == (8ull << 30));
    if (!remote_reservation::address || remote_reservation::size != (8ull << 30)) return;
    // Reacquiring the whole interval without overwrite detects even a leaked
    // middle or final page. Always release any reservation the test acquires.
    mach_vm_address_t address = remote_reservation::address;
    const auto result = ::mach_vm_allocate(mach_task_self(), &address,
        remote_reservation::size, VM_FLAGS_FIXED);
    CHECK(result == KERN_SUCCESS && address == remote_reservation::address);
    if (result == KERN_SUCCESS)
        CHECK(::mach_vm_deallocate(mach_task_self(), address, remote_reservation::size) == KERN_SUCCESS);
#endif
}

void partial_native_failures() {
#if defined(__APPLE__)
    const auto page = native_page_size();
    Fixture fixture;
    auto made = fixture.memory->allocate(3 * page, horizon::MemoryType::normal,
        memory::Permission::read | memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(made));
    if (!std::holds_alternative<GuestMemoryPin>(made)) return;
    auto allocation = std::get<GuestMemoryPin>(std::move(made));
    const auto range = allocation->range();
    allocation->writable_bytes().front() = std::byte{0x43};
    allocation->writable_bytes().back() = std::byte{0x76};
    allocation.reset();
    const auto before = snapshot(*fixture.memory);
    auto registers = seeded_registers();
    registers[0] = range.address; registers[1] = range.size; registers[2] = 1;
    const auto saved = registers;
    native_fault::arm(native_fault::Operation::protect, range.address, range.size, 1);
    const auto failed = fixture.memory->dispatch(2, registers);
    CHECK(native_fault::triggered && host_failure(failed));
    CHECK(registers == saved && unchanged(*fixture.memory, before));
    for (unsigned i = 0; i < 3; ++i) CHECK((native_protection(range.address + i * page) & VM_PROT_WRITE) != 0);
    CHECK((sample(*fixture.memory, range) == std::array<std::byte, 2>{std::byte{0x43}, std::byte{0x76}}));
    CHECK(handled(fixture.memory->dispatch(2, registers)) && registers[0] == 0);

    // Misaligned guest ranges use copied stacks spanning multiple native pages.
    Fixture copied;
    horizon::QueryRegisters grow{}; grow[1] = 2 * mib;
    CHECK(handled(copied.memory->dispatch(1, grow)) && grow[0] == 0);
    const horizon::AddressRange source{grow[1] + 4096, 3 * page};
    const horizon::AddressRange destination{free_stack_address(*copied.memory, 4 * page) + 4096, source.size};
    auto bytes = pin(*copied.memory, source, memory::Permission::read | memory::Permission::write);
    std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x34}); bytes.reset();
    const auto unmapped = snapshot(*copied.memory);
    for (auto operation : {native_fault::Operation::remap, native_fault::Operation::protect}) {
        horizon::QueryRegisters attempt{};
        attempt[0] = destination.address; attempt[1] = source.address; attempt[2] = source.size;
        const auto saved_attempt = attempt;
        native_fault::arm(operation, destination.address - 4096, 4 * page,
            operation == native_fault::Operation::remap ? 1 : 0);
        const auto rejected = copied.memory->dispatch(4, attempt);
        CHECK(host_failure(rejected) && native_fault::triggered);
        if (const auto* error = std::get_if<LaunchFailure>(&rejected)) {
            CHECK(error->detail.find("ProcessMemoryError=") != std::string::npos);
            CHECK(error->detail.find("native_code=" + std::to_string(
                operation == native_fault::Operation::remap ? KERN_NO_SPACE : KERN_PROTECTION_FAILURE)) !=
                std::string::npos);
        }
        CHECK(attempt == saved_attempt && unchanged(*copied.memory, unmapped));
        for (unsigned i = 0; i < 4; ++i)
            CHECK(native_protection(destination.address - 4096 + i * page) == VM_PROT_NONE);
        CHECK((sample(*copied.memory, source) == std::array<std::byte, 2>{std::byte{0x34}, std::byte{0x34}}));
    }
    horizon::QueryRegisters map{};
    map[0] = destination.address; map[1] = source.address; map[2] = source.size;
    CHECK(handled(copied.memory->dispatch(4, map)) && map[0] == 0);
    bytes = pin(*copied.memory, destination, memory::Permission::read | memory::Permission::write);
    std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x79}); bytes.reset();
    const auto mapped = snapshot(*copied.memory);
    auto unmap = seeded_registers();
    unmap[0] = destination.address; unmap[1] = source.address; unmap[2] = source.size;
    const auto saved_unmap = unmap;
    native_fault::arm(native_fault::Operation::remap, destination.address - 4096, 4 * page, 1);
    CHECK(host_failure(copied.memory->dispatch(5, unmap)) && native_fault::triggered);
    CHECK(unmap == saved_unmap && unchanged(*copied.memory, mapped));
    CHECK((sample(*copied.memory, destination) == std::array<std::byte, 2>{std::byte{0x79}, std::byte{0x79}}));
    // The guest source stays inaccessible, but the retained host copy is restored.
    const auto* source_bytes = reinterpret_cast<const std::byte*>(source.address);
    CHECK(source_bytes[0] == std::byte{0x34} && source_bytes[source.size - 1] == std::byte{0x34});
    CHECK(handled(copied.memory->dispatch(5, unmap)) && unmap[0] == 0);
    CHECK((sample(*copied.memory, source) == std::array<std::byte, 2>{std::byte{0x79}, std::byte{0x79}}));
#endif
}

// Temporary rollback views must fit hosts whose VA limit is below 512 GiB.
// Native-page neighbors keep their bytes and permissions after a failed map.
// Rollback aliases are shared views, not frozen snapshots. A neighboring
// writer and a retained physical alias must keep the same bytes and identity.
void copied_stack_concurrent_neighbor_rollback() {
#if defined(__APPLE__)
    const auto page = native_page_size();
    if (page <= horizon::guest_page_size) return;
    for (const bool bounded : {false, true}) for (const bool tracked : {false, true}) {
        Fixture fixture;
        horizon::QueryRegisters grow{}; grow[1] = 2 * mib;
        CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
        const auto base = free_stack_address(*fixture.memory, 4 * page);
        const horizon::AddressRange prefix{base, 4096};
        const horizon::AddressRange source{grow[1] + page + 4096, page};
        const horizon::AddressRange destination{base + 4096, page};
        auto bytes = pin(*fixture.memory, {grow[1], 4096}, memory::Permission::read | memory::Permission::write);
        std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x27});
        bytes.reset();
        horizon::QueryRegisters neighbor{}; neighbor[0] = prefix.address; neighbor[1] = grow[1]; neighbor[2] = prefix.size;
        CHECK(handled(fixture.memory->dispatch(4, neighbor)) && neighbor[0] == 0);
        const auto before = snapshot(*fixture.memory);
        auto reserved = platform::ProcessMemoryRegion::reserve(page);
        auto* physical = std::get_if<platform::ProcessMemoryRegion>(&reserved);
        CHECK(physical != nullptr);
        if (!physical) return;
        CHECK(physical->alias(0, page, base, memory::Permission::read | memory::Permission::write).ok());
        if (tracked) {
            memory::WriteTracker::add_range(reinterpret_cast<void*>(base), page);
            CHECK(memory::WriteTracker::protect_checked(reinterpret_cast<void*>(base), page));
        }
        const auto stamp = memory::WriteTracker::now();
        struct NeighborWrite { std::uint64_t address; bool ran = false; } writer{prefix.address};
        native_fault::address_limit = bounded ? 64ull << 30 : 0;
        native_fault::arm(native_fault::Operation::protect, base, 2 * page, 0);
        native_fault::before_failure_context = &writer;
        native_fault::before_failure = [](void* context) {
            auto& write = *static_cast<NeighborWrite*>(context);
            std::thread thread([&] {
                auto* value = reinterpret_cast<volatile std::uint8_t*>(write.address);
                value[0] = 0x5a;
                value[4095] = 0x6b;
                write.ran = true;
            });
            thread.join();
        };
        auto request = seeded_registers();
        request[0] = destination.address; request[1] = source.address; request[2] = source.size;
        CHECK(host_failure(fixture.memory->dispatch(4, request)) && native_fault::triggered && writer.ran);
        native_fault::operation = native_fault::Operation::none;
        native_fault::before_failure = nullptr;
        native_fault::address_limit = 0;
        CHECK(unchanged(*fixture.memory, before));
        const auto* guest = reinterpret_cast<const volatile std::uint8_t*>(prefix.address);
        const auto* retained = reinterpret_cast<const volatile std::uint8_t*>(physical->address());
        CHECK(guest[0] == 0x5a && guest[4095] == 0x6b);
        CHECK(retained[0] == 0x5a && retained[4095] == 0x6b);
        if (tracked) {
            CHECK(memory::WriteTracker::written_since(reinterpret_cast<void*>(base), page, stamp));
            memory::WriteTracker::remove_range(reinterpret_cast<void*>(base));
        }
    }
#endif
}

void copied_stack_in_bounded_address_space() {
#if defined(__APPLE__)
    const auto page = native_page_size();
    if (page <= horizon::guest_page_size) return;
    Fixture fixture;
    horizon::QueryRegisters grow{}; grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    const auto base = free_stack_address(*fixture.memory, 4 * page);
    const horizon::AddressRange prefix{base, 4096};
    const horizon::AddressRange source{grow[1] + page + 4096, page};
    const horizon::AddressRange destination{base + 4096, page};
    auto bytes = pin(*fixture.memory, {grow[1], 4096}, memory::Permission::read | memory::Permission::write);
    std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x27});
    bytes.reset();
    bytes = pin(*fixture.memory, source, memory::Permission::read | memory::Permission::write);
    std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x65});
    bytes.reset();
    horizon::QueryRegisters neighbor{}; neighbor[0] = prefix.address; neighbor[1] = grow[1]; neighbor[2] = prefix.size;
    CHECK(handled(fixture.memory->dispatch(4, neighbor)) && neighbor[0] == 0);
    const auto before = snapshot(*fixture.memory);
    const std::array protections{native_protection(base), native_protection(base + page)};
    native_fault::high_alias_attempts = 0;
    native_fault::limit_queries = 0;
    native_fault::address_limit = (private_mode ? 64ull : 128ull) << 30;
    const auto outside = platform::ProcessMemoryRegion::reserve(page, native_fault::address_limit);
    CHECK(std::get_if<platform::ProcessMemoryError>(&outside) &&
          std::get<platform::ProcessMemoryError>(outside) == platform::ProcessMemoryError::out_of_range);
    platform::ProcessMemoryStatus unavailable;
    const auto private_alias = platform::ProcessMemoryRegion::writable_host_alias(base, page, &unavailable);
    CHECK(std::holds_alternative<platform::ProcessMemoryError>(private_alias) &&
          unavailable.error == platform::ProcessMemoryError::unsupported);
    for (const auto operation : {native_fault::Operation::remap, native_fault::Operation::protect}) {
        auto request = seeded_registers();
        request[0] = destination.address; request[1] = source.address; request[2] = source.size;
        const auto original = request;
        native_fault::arm(operation, base, 2 * page, 0);
        CHECK(host_failure(fixture.memory->dispatch(4, request)) && native_fault::triggered);
        native_fault::operation = native_fault::Operation::none;
        CHECK(request == original && unchanged(*fixture.memory, before));
        CHECK(native_protection(base) == protections[0] && native_protection(base + page) == protections[1]);
        CHECK((sample(*fixture.memory, prefix) == std::array<std::byte, 2>{std::byte{0x27}, std::byte{0x27}}));
        CHECK((sample(*fixture.memory, source) == std::array<std::byte, 2>{std::byte{0x65}, std::byte{0x65}}));
    }
    horizon::QueryRegisters map{};
    map[0] = destination.address; map[1] = source.address; map[2] = source.size;
    const auto mapped = fixture.memory->dispatch(4, map);
    CHECK(handled(mapped) && map[0] == 0);
    if (handled(mapped) && map[0] == 0) {
        bytes = pin(*fixture.memory, destination, memory::Permission::read | memory::Permission::write);
        std::fill(bytes->writable_bytes().begin(), bytes->writable_bytes().end(), std::byte{0x79});
        bytes.reset();
        horizon::QueryRegisters unmap{};
        unmap[0] = destination.address; unmap[1] = source.address; unmap[2] = source.size;
        CHECK(handled(fixture.memory->dispatch(5, unmap)) && unmap[0] == 0);
        CHECK((sample(*fixture.memory, source) == std::array<std::byte, 2>{std::byte{0x79}, std::byte{0x79}}));
    }
    CHECK((sample(*fixture.memory, prefix) == std::array<std::byte, 2>{std::byte{0x27}, std::byte{0x27}}));
    native_fault::address_limit = 0;
    CHECK(native_fault::limit_queries != 0);
    CHECK(native_fault::high_alias_attempts == 0);
#endif
}

void copied_shared_native_failures() {
#if defined(__APPLE__)
    const auto page = native_page_size();
    for (unsigned scenario = 0; scenario < 8; ++scenario) {
        Fixture fixture;
        const auto start = free_stack_address(*fixture.memory, 4 * page);
        const horizon::AddressRange range{start + 4096, 3 * page};
        const bool neighbor = scenario >= 6;
        if (neighbor) {
            const auto prefix = std::make_shared<horizon::ReadOnlySharedMemory>(std::vector<std::byte>(4096, std::byte{0x27}));
            const auto handle = fixture.handles->insert(prefix).value();
            horizon::QueryRegisters map{}; map[0] = handle; map[1] = start; map[2] = 4096; map[3] = 1;
            CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
        }
        std::vector<std::byte> payload(range.size, std::byte{0x65}); payload.back() = std::byte{0x19};
        auto object = std::make_shared<horizon::ReadOnlySharedMemory>(payload);
        const auto handle = fixture.handles->insert(object).value();
        const auto before = snapshot(*fixture.memory);
        std::array<vm_prot_t, 4> protections;
        for (unsigned i = 0; i < 4; ++i) protections[i] = native_protection(start + i * page);
        std::vector<std::byte> original_page;
        if (neighbor) {
            original_page.resize(page);
            std::memcpy(original_page.data(), reinterpret_cast<const void*>(start), page);
        }
        const bool unmap = scenario == 7;
        horizon::QueryRegisters map{}; map[0] = handle; map[1] = range.address; map[2] = range.size; map[3] = 1;
        if (!unmap) {
            const auto saved = map;
            native_fault::arm(scenario < 4 ? native_fault::Operation::remap : native_fault::Operation::protect,
                start, 4 * page, scenario < 4 ? scenario : scenario == 4 ? 0 : 1);
            CHECK(host_failure(fixture.memory->dispatch(0x13, map)) && native_fault::triggered);
            CHECK(map == saved && unchanged(*fixture.memory, before));
            for (unsigned i = 0; i < 4; ++i) CHECK(native_protection(start + i * page) == protections[i]);
            if (neighbor) CHECK(std::memcmp(original_page.data(), reinterpret_cast<const void*>(start), page) == 0);
            map = saved;
        }
        CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
        CHECK((sample(*fixture.memory, range) == std::array<std::byte, 2>{std::byte{0x65}, std::byte{0x19}}));
        if (unmap) {
            const auto mapped = snapshot(*fixture.memory);
            horizon::QueryRegisters request{}; request[0] = handle; request[1] = range.address; request[2] = range.size;
            const auto saved = request;
            native_fault::arm(native_fault::Operation::remap, start, 4 * page, 1);
            CHECK(host_failure(fixture.memory->dispatch(0x14, request)) && native_fault::triggered);
            CHECK(request == saved && unchanged(*fixture.memory, mapped));
            for (unsigned i = 0; i < 4; ++i) CHECK(native_protection(start + i * page) == VM_PROT_READ);
            CHECK((sample(*fixture.memory, range) == std::array<std::byte, 2>{std::byte{0x65}, std::byte{0x19}}));
            request = saved;
            CHECK(handled(fixture.memory->dispatch(0x14, request)) && request[0] == 0);
        }
    }
#endif
}

void failed_rollback_poisons_owner() {
#if defined(__APPLE__)
    const auto page = native_page_size();
    for (bool copied : {false, true}) {
        Fixture fixture;
        horizon::AddressRange affected;
        horizon::QueryRegisters request{};
        std::uint16_t svc = 2;
        if (!copied) {
            auto made = fixture.memory->allocate(3 * page, horizon::MemoryType::normal,
                memory::Permission::read | memory::Permission::write);
            auto pin = std::get<GuestMemoryPin>(std::move(made)); affected = pin->range(); pin.reset();
            request[0] = affected.address; request[1] = affected.size; request[2] = 1;
            native_fault::arm(native_fault::Operation::protect, affected.address, affected.size, 1, 2);
        } else {
            const auto start = free_stack_address(*fixture.memory, 4 * page);
            affected = {start + 4096, 3 * page};
            auto object = std::make_shared<horizon::ReadOnlySharedMemory>(std::vector<std::byte>(affected.size, std::byte{0x73}));
            request[0] = fixture.handles->insert(object).value();
            request[1] = affected.address; request[2] = affected.size; request[3] = 1; svc = 0x13;
            native_fault::arm(native_fault::Operation::remap, start, 4 * page, 1, 2);
        }
        const auto before = snapshot(*fixture.memory);
        CHECK(host_failure(fixture.memory->dispatch(svc, request)));
        CHECK(native_fault::failures_seen == 2 && unchanged(*fixture.memory, before));
        CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->pin(affected, memory::Permission::read)));
        std::array<std::byte, 1> byte{};
        CHECK(!fixture.memory->copy_observation(affected.address, byte));
        horizon::QueryRegisters next{};
        CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->dispatch(1, next)));
        CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->allocate(page,
            horizon::MemoryType::normal, memory::Permission::read)));
    }
#endif
}

void physical_mapping_failures() {
#if defined(__APPLE__)
    private_mode = true;
    for (bool unmap : {false, true}) {
        Fixture fixture;
        if (!fixture.memory) continue;
        const auto base = fixture.memory->process()->alias_region().address;
        const auto page = native_page_size();
        horizon::QueryRegisters request{}; request[0] = base; request[1] = 3 * page;
        if (unmap) {
            CHECK(handled(fixture.memory->dispatch(0x2c, request)) && request[0] == 0);
            auto held = fixture.memory->pin({base, 3 * page}, memory::Permission::read | memory::Permission::write);
            CHECK(std::holds_alternative<GuestMemoryPin>(held));
            if (auto* pin = std::get_if<GuestMemoryPin>(&held))
                std::fill((*pin)->writable_bytes().begin(), (*pin)->writable_bytes().end(), std::byte{0x69});
        }
        const auto before = snapshot(*fixture.memory);
        request = {}; request[0] = base; request[1] = 3 * page;
        native_fault::arm(native_fault::Operation::remap, base, 3 * page, 1);
        CHECK(host_failure(fixture.memory->dispatch(unmap ? 0x2d : 0x2c, request)));
        CHECK(native_fault::triggered && unchanged(*fixture.memory, before));
        if (unmap) {
            auto held = fixture.memory->pin({base, 3 * page}, memory::Permission::read);
            CHECK(std::holds_alternative<GuestMemoryPin>(held));
            if (auto* pin = std::get_if<GuestMemoryPin>(&held))
                CHECK(std::all_of((*pin)->bytes().begin(), (*pin)->bytes().end(),
                    [](auto b) { return b == std::byte{0x69}; }));
        }
        request = {}; request[0] = base; request[1] = 3 * page;
        CHECK(handled(fixture.memory->dispatch(unmap ? 0x2d : 0x2c, request)) && request[0] == 0);
    }
    // A failed restoration must make subsequent memory operations fail closed.
    {
        Fixture fixture;
        const auto base = fixture.memory->process()->alias_region().address;
        const auto page = native_page_size();
        horizon::QueryRegisters request{}; request[0] = base; request[1] = 3 * page;
        native_fault::arm(native_fault::Operation::remap, base, 3 * page, 1, 2);
        CHECK(host_failure(fixture.memory->dispatch(0x2c, request)));
        CHECK(native_fault::failures_seen == 2);
        request = {}; request[0] = base; request[1] = page;
        CHECK(host_failure(fixture.memory->dispatch(0x2c, request)));
    }
    for (bool unmap : {false, true}) {
        bool complete = false;
        for (unsigned fail_after = 0; fail_after < sweep_limit; ++fail_after) {
            Fixture fixture;
            const auto base = fixture.memory->process()->alias_region().address;
            horizon::QueryRegisters request{}; request[0] = base + 4096; request[1] = 4096;
            CHECK(handled(fixture.memory->dispatch(0x2c, request)) && request[0] == 0);
            const auto before = snapshot(*fixture.memory);
            request = {}; request[0] = base; request[1] = 3 * native_page_size();
            allocation_fault::arm(fail_after);
            const auto result = fixture.memory->dispatch(unmap ? 0x2d : 0x2c, request);
            allocation_fault::disarm();
            if (!allocation_fault::triggered) { complete = handled(result) && request[0] == 0; break; }
            CHECK(unchanged(*fixture.memory, before));
        }
        CHECK(complete);
    }
#endif
}

} // namespace

int main() {
    physical_mapping_failures();
    private_mode = false;
    remote_cleanup_after_record_rejection();
    private_construction_failures();
    for (bool mode : {false, true}) {
        private_mode = mode;
        partial_native_failures();
        copied_stack_in_bounded_address_space();
        copied_stack_concurrent_neighbor_rollback();
        copied_shared_native_failures();
        failed_rollback_poisons_owner();
        CHECK(sweep_bootstrap_allocate());
        CHECK(sweep_bootstrap_release());
        CHECK(sweep_heap(false));
        CHECK(sweep_heap(true));
        CHECK(sweep_protect_split());
        CHECK(sweep_stack(false));
        CHECK(sweep_stack(true));
        CHECK(sweep_shared(false));
        CHECK(sweep_shared(true));
    }
    return failures == 0 ? 0 : 1;
}
