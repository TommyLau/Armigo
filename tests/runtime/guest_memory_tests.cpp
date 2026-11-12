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
#include <armigo/horizon/shared_memory.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/memory/write_tracker.h>
#include "../../src/platform/process_memory.h"
#include "../../src/runtime/service_registry.h"
#include "../../tools/headless/pacing_clock.h"
#include <barrier>
#include <sstream>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace {

using namespace armigo;
using namespace armigo::runtime;

constexpr std::uint64_t mib = 1024 * 1024;
bool private_mode = false;
constexpr std::uint32_t result_invalid_size = 0xca01;
constexpr std::uint32_t result_invalid_memory = 0xd401;
constexpr std::uint32_t result_invalid_permission = 0xd801;
constexpr std::uint32_t result_invalid_region = 0xdc01;
constexpr std::uint32_t result_invalid_handle = 0xe401;

std::atomic_uint failures{0};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

std::uint64_t native_page_size() {
    const long value = ::sysconf(_SC_PAGESIZE);
    CHECK(value > 0);
    return value > 0 ? static_cast<std::uint64_t>(value) : horizon::guest_page_size;
}

LaunchProfile synthetic_profile() {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_address = 0;
    profile.process.alias_address = 0;
    profile.process.stack_address = 0;
    profile.process.heap_size = 8 * mib;
    profile.process.base_alias_size = 2 * mib;
    profile.process.stack_size = 2 * mib;
    profile.permissions.handle_table_capacity = 64;
    if (private_mode) {
        profile.process.platform = GuestPlatform::switch2;
        profile.process.baseline = horizon::Baseline::horizon_23_0_1;
        profile.process.private_system_resource_bytes = 65536;
    }
    return profile;
}

struct Fixture {
    std::shared_ptr<horizon::HandleTable> handles =
        std::make_shared<horizon::HandleTable>(64);
    std::shared_ptr<GuestMemory> memory;

    Fixture() {
        auto created = create_guest_memory(synthetic_profile(), {16 * mib}, nullptr, handles);
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(created));
        if (auto* value = std::get_if<std::shared_ptr<GuestMemory>>(&created)) {
            memory = std::move(*value);
        } else {
            const auto& failure = std::get<LaunchFailure>(created);
            std::fprintf(stderr, "create_guest_memory failed: %s: %s\n",
                         failure.operation.c_str(), failure.detail.c_str());
        }
    }
};

bool succeeded(const LaunchStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

template <class T>
bool failed_with(const std::variant<LaunchFailure, T>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->stage == LaunchStage::memory &&
           failure->code == code && !failure->operation.empty() &&
           !failure->detail.empty();
}

GuestMemoryPin allocate(Fixture& fixture, std::uint64_t size,
                        horizon::MemoryType type, memory::Permission permission) {
    if (!fixture.memory) return {};
    auto result = fixture.memory->allocate(size, type, permission);
    CHECK(std::holds_alternative<GuestMemoryPin>(result));
    return std::holds_alternative<GuestMemoryPin>(result)
               ? std::get<GuestMemoryPin>(std::move(result))
               : GuestMemoryPin{};
}

GuestMemoryPin pin_range(Fixture& fixture, horizon::AddressRange range,
                         memory::Permission permission) {
    if (!fixture.memory) return {};
    auto result = fixture.memory->pin(range, permission);
    CHECK(std::holds_alternative<GuestMemoryPin>(result));
    return std::holds_alternative<GuestMemoryPin>(result)
               ? std::get<GuestMemoryPin>(std::move(result))
               : GuestMemoryPin{};
}

horizon::QueryRegisters registers_with(std::uint64_t seed = 0xabc0000000000000ULL) {
    horizon::QueryRegisters registers{};
    for (std::size_t index = 0; index < registers.size(); ++index) {
        registers[index] = seed + index;
    }
    return registers;
}

bool handled(const LaunchResult<horizon::QueryDisposition>& result) {
    const auto* disposition = std::get_if<horizon::QueryDisposition>(&result);
    return disposition != nullptr &&
           *disposition == horizon::QueryDisposition::handled;
}

bool cleared_result_tail(const horizon::QueryRegisters& registers) {
    return std::all_of(registers.begin() + 2, registers.end(),
                       [](std::uint64_t value) { return value == 0; });
}

bool same_blocks(const std::vector<horizon::MemoryBlock>& left,
                 const std::vector<horizon::MemoryBlock>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto& a = left[index];
        const auto& b = right[index];
        if (a.range.address != b.range.address || a.range.size != b.range.size ||
            a.type != b.type || a.attributes != b.attributes ||
            a.permissions != b.permissions || a.ipc_references != b.ipc_references ||
            a.device_references != b.device_references || a.merge_disable != b.merge_disable) {
            return false;
        }
    }
    return true;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint64_t free_stack_address(const GuestMemory& memory, std::uint64_t size) {
    const auto process = memory.process();
    const auto stack = process->stack_region();
    std::uint64_t cursor = stack.address;
    for (const auto& block : process->memory_map()) {
        const std::uint64_t block_end = block.range.address + block.range.size;
        if (block.range.address < stack.address + stack.size &&
            block_end > stack.address) {
            cursor = std::max(cursor, block_end);
        }
    }
    cursor = align_up(cursor, native_page_size());
    CHECK(cursor >= stack.address && cursor <= stack.address + stack.size &&
          size <= stack.address + stack.size - cursor);
    return cursor;
}

#if defined(__APPLE__)
vm_prot_t native_protection(std::uint64_t address) {
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const kern_return_t result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) {
        ::mach_port_deallocate(mach_task_self(), object);
    }
    CHECK(result == KERN_SUCCESS && region_address <= address &&
          address - region_address < region_size);
    return result == KERN_SUCCESS ? info.protection : VM_PROT_NONE;
}
#endif

void test_factory_and_host_allocations() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    const auto process = fixture.memory->process();
    const auto& configuration = process->configuration();
    CHECK(process->total_memory() == 16 * mib - (private_mode ? 65536 : 0) && process->used_memory() == 0);
    CHECK(configuration.heap_address != 0 && configuration.alias_address != 0 &&
          configuration.stack_address != 0);
    CHECK(configuration.heap_address % page == 0 &&
          configuration.alias_address % page == 0 &&
          configuration.stack_address % page == 0);
    CHECK(configuration.heap_size == 8 * mib &&
          configuration.base_alias_size == 2 * mib &&
          configuration.stack_size == 2 * mib);

    const std::pair<horizon::MemoryType, memory::Permission> cases[] = {
        {horizon::MemoryType::normal, memory::Permission::read},
        {horizon::MemoryType::stack,
         memory::Permission::read | memory::Permission::write},
        {horizon::MemoryType::thread_local_storage,
         memory::Permission::read | memory::Permission::write},
    };
    for (const auto& [type, permission] : cases) {
        const auto used_before = process->used_memory();
        auto pin = allocate(fixture, page, type, permission);
        if (!pin) continue;
        const auto range = pin->range();
        CHECK(range.address != 0 && range.size == page);
        CHECK(pin->permissions() == permission && pin->bytes().size() == page);
        CHECK(std::all_of(pin->bytes().begin(), pin->bytes().end(),
                          [](std::byte value) { return value == std::byte{0}; }));
        CHECK(pin->writable_bytes().empty() ==
              !memory::has_permission(permission, memory::Permission::write));
        const auto info = process->query_memory(range.address);
        CHECK(info.range.address == range.address && info.range.size == range.size &&
              info.type == type && info.permissions == static_cast<std::uint8_t>(permission));
        // Thread-local pages are kernel pages: neither used memory nor charged.
        const auto charged = type == horizon::MemoryType::thread_local_storage ? 0 : page;
        CHECK(process->used_memory() == used_before + charged &&
              process->total_memory() == 16 * mib - (private_mode ? 65536 : 0));
        CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
        pin.reset();
        CHECK(succeeded(fixture.memory->release(range)));
        CHECK(process->used_memory() == used_before &&
              process->query_memory(range.address).type == horizon::MemoryType::free);
    }

    const auto map_before = process->memory_map();
    const auto used_before = process->used_memory();
    auto exhausted = fixture.memory->allocate(18 * mib, horizon::MemoryType::normal,
                                              memory::Permission::read);
    CHECK(failed_with(exhausted, LaunchError::resource_limit));
    CHECK(process->used_memory() == used_before &&
          same_blocks(process->memory_map(), map_before));
    // A thread-local page is still admitted when the budget is spent.
    auto spent = fixture.memory->allocate(16 * mib - (private_mode ? 65536 : 0), horizon::MemoryType::normal, memory::Permission::read);
    CHECK(std::holds_alternative<GuestMemoryPin>(spent));
    auto local = fixture.memory->allocate(page, horizon::MemoryType::thread_local_storage,
                                          memory::Permission::read | memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(local));
    CHECK(failed_with(fixture.memory->allocate(page, horizon::MemoryType::normal, memory::Permission::read),
                      LaunchError::resource_limit));

    auto npdm = synthetic_profile();
    npdm.origin = ProfileOrigin::npdm;
    auto rejected = create_guest_memory(npdm, {16 * mib}, nullptr, fixture.handles);
    CHECK(std::holds_alternative<LaunchFailure>(rejected));
}

void test_pin_validation_and_backing_lifetime() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto allocation = allocate(fixture, 2 * page, horizon::MemoryType::normal,
                               memory::Permission::read | memory::Permission::write);
    if (!allocation) return;
    const auto range = allocation->range();
    allocation->writable_bytes()[0] = std::byte{0x51};
    allocation.reset();

    auto exact = fixture.memory->pin(range, memory::Permission::read);
    CHECK(std::holds_alternative<GuestMemoryPin>(exact));
    auto writable = fixture.memory->pin(
        range, memory::Permission::read | memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(writable));
    exact = LaunchFailure{};
    writable = LaunchFailure{};

    CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->pin(
        {range.address, 0}, memory::Permission::read)));
    CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->pin(
        {range.address, 3 * page}, memory::Permission::read)));
    CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->pin(
        {std::numeric_limits<std::uint64_t>::max() - page + 1, 2 * page},
        memory::Permission::read)));

    horizon::QueryRegisters protect{};
    protect[0] = range.address + page;
    protect[1] = page;
    protect[2] = 1;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(std::holds_alternative<LaunchFailure>(fixture.memory->pin(
        range, memory::Permission::read | memory::Permission::write)));
    CHECK(std::holds_alternative<GuestMemoryPin>(fixture.memory->pin(
        range, memory::Permission::read)));
    const auto first_info = fixture.memory->process()->query_memory(range.address);
    const auto second_info = fixture.memory->process()->query_memory(range.address + page);
    CHECK(first_info.range.address == range.address && first_info.permissions == 3);
    CHECK(second_info.range.address == range.address + page && second_info.permissions == 1);

    auto retained = pin_range(fixture, {range.address, page}, memory::Permission::read);
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
    retained.reset();

    auto lifetime = pin_range(fixture, {range.address, page}, memory::Permission::read);
    fixture.memory.reset();
    CHECK(lifetime->bytes().size() == page && lifetime->bytes()[0] == std::byte{0x51});
}

void test_release_after_partial_protection() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    const auto process = fixture.memory->process();
    const auto used_before = process->used_memory();
    auto allocation = allocate(fixture, 2 * page, horizon::MemoryType::normal,
                               memory::Permission::read | memory::Permission::write);
    if (!allocation) return;
    const auto range = allocation->range();
    allocation.reset();

    auto protect = registers_with();
    protect[0] = range.address + page;
    protect[1] = page;
    protect[2] = 1;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(process->used_memory() == used_before + range.size);

    auto pin = pin_range(fixture, {range.address, page}, memory::Permission::read);
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
    pin.reset();

    const std::uint64_t destination = free_stack_address(*fixture.memory, page);
    auto map = registers_with();
    map[0] = destination;
    map[1] = range.address;
    map[2] = page;
    CHECK(handled(fixture.memory->dispatch(4, map)) && map[0] == 0);
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));

    auto unmap = registers_with();
    unmap[0] = destination;
    unmap[1] = range.address;
    unmap[2] = page;
    CHECK(handled(fixture.memory->dispatch(5, unmap)) && unmap[0] == 0);
    CHECK(succeeded(fixture.memory->release(range)));
    CHECK(process->used_memory() == used_before &&
          process->query_memory(range.address).type == horizon::MemoryType::free &&
          process->query_memory(range.address + page).type == horizon::MemoryType::free);

    allocation = allocate(fixture, 2 * page, horizon::MemoryType::normal,
                          memory::Permission::read | memory::Permission::write);
    if (!allocation) return;
    const auto inaccessible = allocation->range();
    allocation.reset();
    protect[0] = inaccessible.address + page;
    protect[1] = page;
    protect[2] = 0;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(succeeded(fixture.memory->release(inaccessible)));
    CHECK(process->used_memory() == used_before);
}

void test_heap_dispatch_and_pin_conflicts() {
    Fixture fixture;
    if (!fixture.memory) return;

    auto invalid = registers_with();
    invalid[1] = mib;
    CHECK(handled(fixture.memory->dispatch(1, invalid)));
    CHECK(invalid[0] == result_invalid_size && invalid[1] == mib &&
          cleared_result_tail(invalid));

    auto grow = registers_with();
    grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)));
    CHECK(grow[0] == 0 && cleared_result_tail(grow));
    const std::uint64_t heap = grow[1];
    CHECK(heap == fixture.memory->process()->heap_region().address);
    const auto info = fixture.memory->process()->query_memory(heap);
    CHECK(info.range.address == heap && info.range.size >= 2 * mib &&
          info.type == horizon::MemoryType::normal && info.permissions == 3);
    CHECK(fixture.memory->process()->used_memory() == 2 * mib &&
          fixture.memory->process()->total_memory() == 16 * mib - (private_mode ? 65536 : 0));

    auto pin_result = fixture.memory->pin({heap, 2 * mib}, memory::Permission::read);
    CHECK(std::holds_alternative<GuestMemoryPin>(pin_result));
    auto pin = std::holds_alternative<GuestMemoryPin>(pin_result)
                   ? std::get<GuestMemoryPin>(std::move(pin_result))
                   : GuestMemoryPin{};
    auto shrink = registers_with();
    shrink[1] = 0;
    const auto unchanged = shrink;
    auto conflict = fixture.memory->dispatch(1, shrink);
    CHECK(failed_with(conflict, LaunchError::unsupported));
    CHECK(shrink == unchanged);
    pin.reset();
    CHECK(handled(fixture.memory->dispatch(1, shrink)) && shrink[0] == 0);
    CHECK(fixture.memory->process()->used_memory() == 0 &&
          fixture.memory->process()->query_memory(heap).type == horizon::MemoryType::free);
}

void test_permission_dispatch_and_native_protection() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto allocation = allocate(fixture, page, horizon::MemoryType::normal,
                               memory::Permission::read | memory::Permission::write);
    if (!allocation) return;
    const auto range = allocation->range();
    allocation.reset();

    for (const std::uint64_t permission : {1ULL, 0ULL, 3ULL}) {
        auto registers = registers_with();
        registers[0] = range.address;
        registers[1] = range.size;
        registers[2] = permission;
        CHECK(handled(fixture.memory->dispatch(2, registers)) && registers[0] == 0 &&
              registers[1] == 0 && cleared_result_tail(registers));
        const auto info = fixture.memory->process()->query_memory(range.address);
        CHECK(info.permissions == permission);
#if defined(__APPLE__)
        const vm_prot_t protection = native_protection(range.address);
        CHECK(((protection & VM_PROT_READ) != 0) == ((permission & 1) != 0));
        CHECK(((protection & VM_PROT_WRITE) != 0) == ((permission & 2) != 0));
#endif
        const auto read_pin = fixture.memory->pin(range, memory::Permission::read);
        CHECK(std::holds_alternative<GuestMemoryPin>(read_pin) == ((permission & 1) != 0));
        const auto write_pin = fixture.memory->pin(
            range, memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(write_pin) == (permission == 3));
    }

    auto write_only = registers_with();
    write_only[0] = range.address;
    write_only[1] = range.size;
    write_only[2] = 2;
    CHECK(handled(fixture.memory->dispatch(2, write_only)) &&
          write_only[0] == result_invalid_permission);

    if (page > horizon::guest_page_size) {
        // A guest page inside a larger native page changes in the guest map;
        // the native page keeps the union of its guest pages' permissions.
        auto subpage = registers_with();
        subpage[0] = range.address;
        subpage[1] = horizon::guest_page_size;
        subpage[2] = 1;
        CHECK(handled(fixture.memory->dispatch(2, subpage)) && subpage[0] == 0);
        auto restore = registers_with();
        restore[0] = range.address;
        restore[1] = horizon::guest_page_size;
        restore[2] = 3;
        CHECK(handled(fixture.memory->dispatch(2, restore)) && restore[0] == 0);
    }

    auto held = pin_range(fixture, range,
                          memory::Permission::read | memory::Permission::write);
    auto unchanged_permission = registers_with();
    unchanged_permission[0] = range.address;
    unchanged_permission[1] = horizon::guest_page_size;
    unchanged_permission[2] = (std::uint64_t{1} << 40) | 3;
    CHECK(handled(fixture.memory->dispatch(2, unchanged_permission)) && unchanged_permission[0] == 0);
    auto conflict = registers_with();
    conflict[0] = range.address;
    conflict[1] = range.size;
    conflict[2] = 1;
    const auto unchanged = conflict;
    CHECK(failed_with(fixture.memory->dispatch(2, conflict), LaunchError::unsupported));
    CHECK(conflict == unchanged);
}

void test_stack_alias_map_and_exact_unmap() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    horizon::QueryRegisters grow{};
    grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    const horizon::AddressRange source{grow[1], page};
    auto source_pin = pin_range(
        fixture, source, memory::Permission::read | memory::Permission::write);
    if (!source_pin) return;
    source_pin->writable_bytes()[17] = std::byte{0x29};
    source_pin.reset();
    const std::uint64_t destination = free_stack_address(*fixture.memory, page);

    auto map = registers_with();
    map[0] = destination;
    map[1] = source.address;
    map[2] = page;
    CHECK(handled(fixture.memory->dispatch(4, map)) && map[0] == 0 && map[1] == 0 &&
          cleared_result_tail(map));
    const auto source_info = fixture.memory->process()->query_memory(source.address);
    const auto destination_info = fixture.memory->process()->query_memory(destination);
    CHECK(source_info.type == horizon::MemoryType::normal &&
          source_info.attributes == 1 && source_info.permissions == 0);
    CHECK(destination_info.type == horizon::MemoryType::stack &&
          destination_info.attributes == 0 && destination_info.permissions == 3);

    auto shrink_locked_heap = registers_with();
    shrink_locked_heap[1] = 0;
    CHECK(handled(fixture.memory->dispatch(1, shrink_locked_heap)) &&
          shrink_locked_heap[0] == result_invalid_memory &&
          shrink_locked_heap[1] == 0 && cleared_result_tail(shrink_locked_heap));
    CHECK(fixture.memory->process()->used_memory() == 2 * mib);

    auto alias_result = fixture.memory->pin({destination, page},
                                            memory::Permission::read |
                                                memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(alias_result));
    auto alias = std::holds_alternative<GuestMemoryPin>(alias_result)
                     ? std::get<GuestMemoryPin>(std::move(alias_result))
                     : GuestMemoryPin{};
    if (alias) {
        CHECK(alias->bytes()[17] == std::byte{0x29});
        alias->writable_bytes()[31] = std::byte{0x73};
    }

    auto pinned_unmap = registers_with();
    pinned_unmap[0] = destination;
    pinned_unmap[1] = source.address;
    pinned_unmap[2] = page;
    const auto unchanged = pinned_unmap;
    CHECK(failed_with(fixture.memory->dispatch(5, pinned_unmap), LaunchError::unsupported));
    CHECK(pinned_unmap == unchanged);
    alias.reset();

    auto wrong = registers_with();
    wrong[0] = destination;
    wrong[1] = source.address + page;
    wrong[2] = page;
    CHECK(handled(fixture.memory->dispatch(5, wrong)) &&
          wrong[0] == result_invalid_memory && wrong[1] == 0 && cleared_result_tail(wrong));
    CHECK(fixture.memory->process()->query_memory(destination).type ==
          horizon::MemoryType::stack);

    auto unmap = registers_with();
    unmap[0] = destination;
    unmap[1] = source.address;
    unmap[2] = page;
    CHECK(handled(fixture.memory->dispatch(5, unmap)) && unmap[0] == 0 && unmap[1] == 0 &&
          cleared_result_tail(unmap));
    const auto restored = fixture.memory->process()->query_memory(source.address);
    CHECK(restored.type == horizon::MemoryType::normal && restored.attributes == 0 &&
          restored.permissions == 3);
    CHECK(fixture.memory->process()->query_memory(destination).type ==
          horizon::MemoryType::free);
    auto restored_pin = pin_range(
        fixture, source, memory::Permission::read | memory::Permission::write);
    if (restored_pin) CHECK(restored_pin->bytes()[31] == std::byte{0x73});
}

horizon::QueryRegisters shared_map(horizon::Handle handle, std::uint64_t destination,
                                   std::uint64_t size, std::uint64_t permission) {
    horizon::QueryRegisters registers{};
    registers[0] = handle;
    registers[1] = destination;
    registers[2] = size;
    registers[3] = permission;
    return registers;
}

void test_read_write_shared_memory_alias_and_handles() {
    constexpr std::uint64_t size = 64 * 1024;
    Fixture fixture;
    if (!fixture.memory) return;
    auto object = std::make_shared<horizon::SharedMemory>(size);
    const auto first = fixture.handles->insert(object);
    const auto second = fixture.handles->insert(object);
    CHECK(first.has_value() && second.has_value());
    if (!first || !second) return;
    const std::uint64_t destination = free_stack_address(*fixture.memory, size);

    auto map = shared_map(*first, destination, size, 3);
    auto result = fixture.memory->dispatch(0x13, map);
    // Bytes not at native page alignment move there before the first mapping.
    CHECK(handled(result) && map[0] == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(object->data()) % native_page_size() == 0);
    auto guest = pin_range(fixture, {destination, size},
                           memory::Permission::read | memory::Permission::write);
    if (!guest) return;
    object->bytes()[9] = std::byte{0x44};
    CHECK(guest->bytes()[9] == std::byte{0x44});
    guest->writable_bytes()[23] = std::byte{0x75};
    CHECK(object->bytes()[23] == std::byte{0x75});
    guest.reset();

    auto wrong_range = shared_map(*first, destination, size / 2, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, wrong_range)) &&
          wrong_range[0] == result_invalid_size);
    auto different = std::make_shared<horizon::SharedMemory>(size);
    const auto different_handle = fixture.handles->insert(different);
    CHECK(different_handle.has_value());
    if (different_handle) {
        auto wrong_object = shared_map(*different_handle, destination, size, 0);
        CHECK(handled(fixture.memory->dispatch(0x14, wrong_object)) &&
              wrong_object[0] == result_invalid_region);
    }
    CHECK(fixture.handles->close(*first));
    auto stale = shared_map(*first, destination, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, stale)) &&
          stale[0] == result_invalid_handle);
    auto equivalent = shared_map(*second, destination, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, equivalent)) && equivalent[0] == 0);
}

void test_shared_mapping_retains_object_after_handle_close() {
    constexpr std::uint64_t size = 64 * 1024;
    Fixture fixture;
    if (!fixture.memory) return;
    auto object = std::make_shared<horizon::SharedMemory>(size);
    if (reinterpret_cast<std::uintptr_t>(object->data()) % native_page_size() != 0) return;
    std::weak_ptr<horizon::SharedMemory> weak = object;
    const auto handle = fixture.handles->insert(object);
    CHECK(handle.has_value());
    if (!handle) return;
    const std::uint64_t destination = free_stack_address(*fixture.memory, size);
    auto map = shared_map(*handle, destination, size, 3);
    CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
    auto pin = pin_range(fixture, {destination, size}, memory::Permission::read);
    if (!pin) return;
    object.reset();
    CHECK(fixture.handles->close(*handle));
    CHECK(!weak.expired());
    fixture.memory.reset();
    CHECK(!weak.expired() && pin->bytes().size() == size);
    pin.reset();
    CHECK(weak.expired());
}

// A guest page smaller than the host page lets the guest map a writable object
// at an address inside a host page. The object's bytes move to that offset and
// the host pages around the range alias them, so writes on either side stay
// visible to the other; neighbours in those host pages are refused.
void test_sub_page_shared_memory_mapping() {
    const auto page = native_page_size();
    if (page <= 4096) return;
    constexpr std::uint64_t size = 64 * 1024;
    Fixture fixture;
    if (!fixture.memory) return;
    auto object = std::make_shared<horizon::SharedMemory>(size);
    object->update([](std::span<std::byte> bytes) { bytes[5] = std::byte{0x21}; });
    const auto handle = fixture.handles->insert(object);
    CHECK(handle.has_value());
    if (!handle) return;
    const std::uint64_t base = free_stack_address(*fixture.memory, size + 2 * page);
    const std::uint64_t destination = base + page / 2;

    auto map = shared_map(*handle, destination, size, 1);
    CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(object->data()) % page == page / 2);
    auto guest = pin_range(fixture, {destination, size}, memory::Permission::read);
    if (!guest) return;
    CHECK(guest->bytes()[5] == std::byte{0x21});
    object->update([](std::span<std::byte> bytes) { bytes[size - 1] = std::byte{0x63}; });
    CHECK(guest->bytes()[size - 1] == std::byte{0x63});
    guest.reset();

    // Another object in one of those host pages would see this object's bytes.
    auto neighbour = std::make_shared<horizon::SharedMemory>(4096);
    const auto neighbour_handle = fixture.handles->insert(neighbour);
    CHECK(neighbour_handle.has_value());
    if (neighbour_handle) {
        auto next = shared_map(*neighbour_handle, destination + size, 4096, 1);
        CHECK(failed_with(fixture.memory->dispatch(0x13, next), LaunchError::unsupported));
        auto before = shared_map(*neighbour_handle, base, 4096, 1);
        CHECK(failed_with(fixture.memory->dispatch(0x13, before), LaunchError::unsupported));
    }
    // A second mapping at another offset would need the bytes in two places.
    auto other = shared_map(*handle, base + size + page, size, 1);
    CHECK(failed_with(fixture.memory->dispatch(0x13, other), LaunchError::unsupported));
    // Copies into the free part of an envelope page would write the object's
    // neighbouring bytes: a read-only shared copy and a copied stack refuse.
    std::vector<std::byte> constant(4096, std::byte{0x5a});
    const auto copied = fixture.handles->insert(std::make_shared<horizon::ReadOnlySharedMemory>(constant));
    CHECK(copied.has_value());
    if (copied) {
        auto copy = shared_map(*copied, destination + size, 4096, 1);
        CHECK(failed_with(fixture.memory->dispatch(0x13, copy), LaunchError::unsupported));
    }
    horizon::QueryRegisters grow{};
    grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    auto stack = registers_with();
    stack[0] = destination + size;
    stack[1] = grow[1];
    stack[2] = 4096;
    CHECK(failed_with(fixture.memory->dispatch(4, stack), LaunchError::unsupported));
    CHECK(object->bytes()[size - 1] == std::byte{0x63});

    auto unmap = shared_map(*handle, destination, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap)) && unmap[0] == 0);
    // A read-write mapping at the same offset reuses the moved bytes; guest
    // writes through it reach the object.
    auto writable = shared_map(*handle, destination, size, 3);
    CHECK(handled(fixture.memory->dispatch(0x13, writable)) && writable[0] == 0);
    guest = pin_range(fixture, {destination, size}, memory::Permission::read | memory::Permission::write);
    if (!guest) return;
    guest->writable_bytes()[77] = std::byte{0x19};
    CHECK(object->bytes()[77] == std::byte{0x19} && guest->bytes()[5] == std::byte{0x21});
    guest.reset();
    auto unmap_writable = shared_map(*handle, destination, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap_writable)) && unmap_writable[0] == 0);
    // Unmapped, the bytes can move back to an aligned mapping, keeping writes.
    auto aligned = shared_map(*handle, base, size, 1);
    CHECK(handled(fixture.memory->dispatch(0x13, aligned)) && aligned[0] == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(object->data()) % page == 0);
    guest = pin_range(fixture, {base, size}, memory::Permission::read);
    if (!guest) return;
    CHECK(guest->bytes()[5] == std::byte{0x21} && guest->bytes()[size - 1] == std::byte{0x63} &&
          guest->bytes()[77] == std::byte{0x19});
    guest.reset();
    auto unmap_aligned = shared_map(*handle, base, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap_aligned)) && unmap_aligned[0] == 0);
}

void test_read_only_shared_memory_mapping() {
    constexpr std::uint64_t size = 16 * 1024;
    Fixture fixture;
    if (!fixture.memory) return;
    std::vector<std::byte> payload(size);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::byte>(index & 0xff);
    }
    auto object = std::make_shared<horizon::ReadOnlySharedMemory>(payload);
    const auto handle = fixture.handles->insert(object);
    CHECK(handle.has_value());
    if (!handle) return;
    const std::uint64_t destination = free_stack_address(*fixture.memory, size);

    auto writable = shared_map(*handle, destination, size, 3);
    CHECK(handled(fixture.memory->dispatch(0x13, writable)) &&
          writable[0] == result_invalid_permission);
    auto read_only = shared_map(*handle, destination, size, 1);
    CHECK(handled(fixture.memory->dispatch(0x13, read_only)) && read_only[0] == 0);
    auto pin = pin_range(fixture, {destination, size}, memory::Permission::read);
    if (!pin) return;
    CHECK(std::equal(pin->bytes().begin(), pin->bytes().end(), payload.begin()));
    CHECK(pin->writable_bytes().empty());
    pin.reset();
    auto unmap = shared_map(*handle, destination, size, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap)) && unmap[0] == 0);
}

void test_copy_out_native_atomics() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto pin = allocate(fixture, page, horizon::MemoryType::normal,
                        memory::Permission::read | memory::Permission::write);
    if (!pin) return;
    const auto address = pin->range().address;
    auto* word = reinterpret_cast<std::uint32_t*>(pin->writable_bytes().data());
    auto* wide = reinterpret_cast<std::uint64_t*>(pin->writable_bytes().data() + 8);
    __atomic_store_n(word, 0U, __ATOMIC_RELEASE);
    __atomic_store_n(wide, 0x1122334455667788ULL, __ATOMIC_RELEASE);
    const auto copy = fixture.memory->copy_out();
    CHECK(copy.write_word != nullptr && copy.read_word != nullptr &&
          copy.compare_exchange_word != nullptr && copy.read_double_word != nullptr);

    std::array<std::byte, 4> written{std::byte{0x78}, std::byte{0x56},
                                     std::byte{0x34}, std::byte{0x12}};
    CHECK(copy.write_word(copy.context, address, written));
    CHECK(__atomic_load_n(word, __ATOMIC_ACQUIRE) == 0x12345678U);
    std::array<std::byte, 4> read{};
    CHECK(copy.read_word(copy.context, address, read) && read == written);
    CHECK(!copy.read_word(copy.context, address + 1, read));

    __atomic_store_n(word, 0U, __ATOMIC_RELEASE);
    constexpr unsigned thread_count = 4;
    constexpr unsigned increments = 2000;
    std::vector<std::thread> workers;
    for (unsigned thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([copy, address] {
            for (unsigned index = 0; index < increments; ++index) {
                std::uint32_t expected = __atomic_load_n(
                    reinterpret_cast<std::uint32_t*>(address), __ATOMIC_ACQUIRE);
                for (;;) {
                    const auto exchanged = copy.compare_exchange_word(
                        copy.context, address, expected, expected + 1);
                    if (exchanged == horizon::ProcessWideKey::ExchangeResult::exchanged) break;
                    CHECK(exchanged == horizon::ProcessWideKey::ExchangeResult::mismatch);
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    CHECK(__atomic_load_n(word, __ATOMIC_ACQUIRE) == thread_count * increments);

    std::uint64_t observed = 0;
    CHECK(copy.read_double_word(copy.context, address + 8, observed));
    CHECK(observed == 0x1122334455667788ULL);
    CHECK(!copy.read_double_word(copy.context, address + 4, observed));

    // Synchronization words may lie anywhere in the kernel's envelope, which
    // extends past the mapped 39-bit space. Every accessor refuses such a word
    // before touching host memory: a dereference there would fault the test.
    for (const auto outside : {horizon::ProcessRecord::query_end, std::uint64_t{0x8000000000},
                               std::uint64_t{0x3fffffffff8}}) {
        CHECK(!copy.read_word(copy.context, outside, read));
        CHECK(!copy.write_word(copy.context, outside, written));
        std::uint32_t expected = 0;
        CHECK(copy.compare_exchange_word(copy.context, outside, expected, 1) ==
              horizon::ProcessWideKey::ExchangeResult::invalid_address);
        CHECK(!copy.read_double_word(copy.context, outside, observed));
    }
}

void test_ipc_whole_range_and_ordered_word_copy() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto pin = allocate(fixture, page, horizon::MemoryType::normal,
                        memory::Permission::read | memory::Permission::write);
    if (!pin) return;
    const auto address = pin->range().address;
    auto buffers = fixture.memory->ipc_buffers();

    std::array<std::byte, 4> value{std::byte{1}, std::byte{2},
                                   std::byte{3}, std::byte{4}};
    CHECK(buffers.write_memory(address + 16, value));
    std::array<std::byte, 4> read{};
    CHECK(buffers.read_memory(address + 16, read) && read == value);

    const std::array<std::byte, 2> before{
        pin->bytes()[page - 2], pin->bytes()[page - 1]};
    CHECK(!buffers.write_memory(address + page - 2, value));
    CHECK(pin->bytes()[page - 2] == before[0] &&
          pin->bytes()[page - 1] == before[1]);
    read.fill(std::byte{0x66});
    CHECK(!buffers.read_memory(address + page - 2, read));
    CHECK(std::all_of(read.begin(), read.end(),
                      [](std::byte byte) { return byte == std::byte{0x66}; }));
    CHECK(!buffers.read_memory(std::numeric_limits<std::uint64_t>::max() - 1, read));

    const auto copy = fixture.memory->copy_out();
    std::array<std::byte, 4> first{std::byte{9}, std::byte{8},
                                   std::byte{7}, std::byte{6}};
    CHECK(copy.write_word(copy.context, address + page - 4, first));
    CHECK(!copy.write_word(copy.context, address + page, value));
    CHECK(std::equal(first.begin(), first.end(), pin->bytes().end() - 4));

    pin.reset();
    horizon::QueryRegisters protect{};
    protect[0] = address;
    protect[1] = page;
    protect[2] = 1;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    read.fill(std::byte{0});
    CHECK(buffers.read_memory(address + 16, read) && read == value);
    CHECK(!buffers.write_memory(address + 16, first));
    read.fill(std::byte{0});
    CHECK(buffers.read_memory(address + 16, read) && read == value);
}

void test_service_mapper_retention_and_lifetime() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto allocation = allocate(fixture, page, horizon::MemoryType::normal,
                               memory::Permission::read | memory::Permission::write);
    if (!allocation) return;
    const auto range = allocation->range();
    allocation.reset();
    auto mapper = fixture.memory->service_mapper();
    CHECK(mapper.map != nullptr && mapper.unmap != nullptr && mapper.context != nullptr &&
          mapper.lifetime != nullptr);
    CHECK(mapper.map(mapper.context, range.address, range.size));
    CHECK(mapper.map(mapper.context, range.address, range.size));
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
    mapper.unmap(mapper.context, range.address, range.size / 2);
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
    mapper.unmap(mapper.context, range.address, range.size);
    CHECK(failed_with(fixture.memory->release(range), LaunchError::unsupported));
    mapper.unmap(mapper.context, range.address, range.size);
    CHECK(succeeded(fixture.memory->release(range)));

    std::weak_ptr<GuestMemory> weak = fixture.memory;
    fixture.memory.reset();
    CHECK(weak.expired());
    mapper.lifetime.reset();
}

void test_firmware_validation_order() {
    Fixture fixture;
    if (!fixture.memory) return;
    const auto page = native_page_size();
    const auto stack = fixture.memory->process()->stack_region().address;
    const auto heap = fixture.memory->process()->heap_region().address;
    const auto end = horizon::ProcessRecord::query_end;
    const auto wrapping = std::uint64_t{0} - horizon::guest_page_size;
    const auto object = std::make_shared<horizon::ReadOnlySharedMemory>(
        std::vector<std::byte>(page));
    const auto handle = fixture.handles->insert(object);
    CHECK(handle.has_value());
    if (!handle) return;
    struct Case {
        std::uint16_t svc;
        std::array<std::uint64_t, 4> inputs;
        std::uint32_t result;
    };
    const Case cases[] = {
        {2, {1, 0, 2, 0}, 0xcc01},
        {2, {0, 0, 2, 0}, result_invalid_size},
        {2, {wrapping, 4096, 2, 0}, result_invalid_memory},
        {2, {end, page, 2, 0}, result_invalid_permission},
        {2, {end, page, 3, 0}, result_invalid_memory},
        {4, {0, 1, 0, 0}, 0xcc01},
        {5, {0, 1, 0, 0}, 0xcc01},
        {4, {0, 0, 0, 0}, result_invalid_size},
        {5, {0, 0, 0, 0}, result_invalid_size},
        {4, {wrapping, 0, 4096, 0}, result_invalid_memory},
        {5, {0, wrapping, 4096, 0}, result_invalid_memory},
        {4, {0, end, page, 0}, result_invalid_memory},
        {5, {0, end, page, 0}, result_invalid_memory},
        // Source numeric bounds precede destination placement, then source state.
        {4, {0, 0, page, 0}, result_invalid_region},
        {5, {0, 0, page, 0}, result_invalid_region},
        {4, {stack, 0, page, 0}, result_invalid_memory},
        {5, {stack, 0, page, 0}, result_invalid_memory},
        {4, {stack, stack, page, 0}, result_invalid_memory},
        {5, {stack, stack, page, 0}, result_invalid_memory},
        {0x13, {0, 1, 0, 2}, 0xcc01},
        {0x14, {0, 1, 0, 2}, 0xcc01},
        {0x13, {0, 0, 0, 2}, result_invalid_size},
        {0x14, {0, 0, 0, 2}, result_invalid_size},
        {0x13, {0, wrapping, 4096, 2}, result_invalid_memory},
        {0x14, {0, wrapping, 4096, 2}, result_invalid_memory},
        {0x13, {0, 0, page, 2}, result_invalid_permission},
        {0x13, {0, 0, page, 1}, result_invalid_handle},
        {0x14, {0, 0, page, 2}, result_invalid_handle},
        {0x13, {*handle, heap, 2 * page, 3}, result_invalid_region},
        {0x14, {*handle, heap, 2 * page, 0}, result_invalid_region},
        {0x13, {*handle, stack, 2 * page, 3}, result_invalid_size},
        {0x14, {*handle, stack, 2 * page, 0}, result_invalid_size},
        {0x13, {*handle, stack, page, 3}, result_invalid_permission},
        {0x14, {*handle, stack, page, 0}, result_invalid_memory},
    };
    const auto before = fixture.memory->process()->memory_map();
    for (const auto& test : cases) {
        auto registers = registers_with();
        std::copy(test.inputs.begin(), test.inputs.end(), registers.begin());
        CHECK(handled(fixture.memory->dispatch(test.svc, registers)));
        CHECK(registers[0] == test.result && registers[1] == 0 && cleared_result_tail(registers));
        CHECK(same_blocks(before, fixture.memory->process()->memory_map()));
    }
    for (const auto size : {mib, horizon::ProcessRecord::heap_capacity}) {
        auto registers = registers_with();
        registers[1] = size;
        CHECK(handled(fixture.memory->dispatch(1, registers)));
        CHECK(registers[0] == result_invalid_size && registers[1] == size && cleared_result_tail(registers));
    }
    // The permission argument is W3; unrelated high bits of X3 are ignored.
    auto map = shared_map(*handle, stack, page, (std::uint64_t{1} << 40) | 1);
    CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
    auto unmap = shared_map(*handle, stack, page, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap)) && unmap[0] == 0);
    unmap = shared_map(*handle, stack, page, 0);
    CHECK(handled(fixture.memory->dispatch(0x14, unmap)) && unmap[0] == result_invalid_memory);
}

void test_heap_tail_validation_and_accounting() {
    Fixture fixture;
    if (!fixture.memory) return;
    const auto page = native_page_size();
    auto grow = registers_with();
    grow[1] = 4 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    const auto heap = grow[1];
    auto protect = registers_with();
    protect[0] = heap + 4 * mib - page;
    protect[1] = page;
    protect[2] = 1;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    auto pinned = pin_range(fixture, {heap + 2 * mib, page}, memory::Permission::read);
    auto shrink = registers_with();
    shrink[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, shrink)) && shrink[0] == result_invalid_memory);
    CHECK(shrink[1] == 2 * mib && cleared_result_tail(shrink));
    CHECK(fixture.memory->process()->used_memory() == 4 * mib);
    pinned.reset();
    protect[0] = heap + 4 * mib - page;
    protect[1] = page;
    protect[2] = 3;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);

    // A locked source outside the removed tail does not prevent tail shrink.
    const auto destination = free_stack_address(*fixture.memory, page);
    auto map = registers_with();
    map[0] = destination; map[1] = heap; map[2] = page;
    CHECK(handled(fixture.memory->dispatch(4, map)) && map[0] == 0);
    shrink[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, shrink)) && shrink[0] == 0);
    CHECK(fixture.memory->process()->used_memory() == 2 * mib);
    CHECK(fixture.memory->process()->heap_region().size == 8 * mib);
    map[0] = destination; map[1] = heap; map[2] = page;
    CHECK(handled(fixture.memory->dispatch(5, map)) && map[0] == 0);
    CHECK(fixture.memory->process()->query_memory(heap).range.size == 2 * mib);
    CHECK(fixture.memory->process()->total_memory() == 16 * mib - (private_mode ? 65536 : 0));

    // Switch 1 reaches its process maximum before the resource budget. Switch 2
    // has a larger application capacity, so this explicit 8 GiB budget is the
    // constraint instead. Neither rejection commits the large virtual heap.
    auto profile = synthetic_profile();
    profile.process.heap_size = horizon::ProcessRecord::heap_capacity;
    profile.process.stack_size = 8 * mib;
    auto made = create_guest_memory(profile, {horizon::ProcessRecord::heap_capacity},
                                    nullptr, fixture.handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
    if (auto* memory = std::get_if<std::shared_ptr<GuestMemory>>(&made)) {
        auto stack = (*memory)->allocate(4 * mib, horizon::MemoryType::stack,
                                        memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(stack));
        auto request = registers_with();
        request[1] = horizon::ProcessRecord::heap_capacity - 2 * mib;
        CHECK(handled((*memory)->dispatch(1, request)) && request[0] == (private_mode ? 0x10801 : 0xd001));
        CHECK(request[1] == horizon::ProcessRecord::heap_capacity - 2 * mib);
    }
}

void test_query_boundaries_and_alias_identity() {
    Fixture fixture;
    if (!fixture.memory) return;
    const auto page = native_page_size();
    const auto process = fixture.memory->process();
    auto grow = registers_with(); grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    const auto heap = grow[1];
    const auto destination = free_stack_address(*fixture.memory, 4 * page);
    const auto used = process->used_memory();
    const auto object = std::make_shared<horizon::ReadOnlySharedMemory>(
        std::vector<std::byte>(page, std::byte{0x37}));
    const auto handle = fixture.handles->insert(object);
    CHECK(handle.has_value());
    if (!handle) return;
    for (unsigned i = 0; i < 2; ++i) {
        auto map = shared_map(*handle, destination + i * page, page, 1);
        CHECK(handled(fixture.memory->dispatch(0x13, map)) && map[0] == 0);
    }
    const auto other = fixture.handles->insert(std::make_shared<horizon::ReadOnlySharedMemory>(
        std::vector<std::byte>(page, std::byte{0x37})));
    CHECK(other.has_value());
    if (other) {
        auto wrong_object = shared_map(*other, destination, page, 0);
        CHECK(handled(fixture.memory->dispatch(0x14, wrong_object)) &&
              wrong_object[0] == result_invalid_region);
    }
    for (unsigned i = 0; i < 2; ++i) {
        const auto info = process->query_memory(destination + i * page + 7);
        CHECK(info.range.address == destination + i * page && info.range.size == page);
        CHECK(info.type == horizon::MemoryType::shared && info.permissions == 1 && info.attributes == 0);
        const auto encoded = horizon::encode_memory_info(info);
        CHECK(encoded[36] == std::byte{0} && encoded[39] == std::byte{0});
    }
    auto protect = registers_with();
    protect[0] = heap + page; protect[1] = page; protect[2] = 1;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(process->query_memory(heap).range.size == page);
    CHECK(process->query_memory(heap + page).range.size == page);
    CHECK(process->query_memory(heap + 2 * page).range.size == 2 * mib - 2 * page);
    CHECK(process->query_memory(destination).range.size == page);
    protect[0] = heap + page; protect[1] = page; protect[2] = 3;
    CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(process->query_memory(heap).range.size == 2 * mib);
    for (unsigned i = 0; i < 2; ++i) {
        auto unmap = shared_map(*handle, destination + i * page, page, 0);
        CHECK(handled(fixture.memory->dispatch(0x14, unmap)) && unmap[0] == 0);
    }
    const auto hole = process->query_memory(destination + page);
    CHECK(hole.type == horizon::MemoryType::free && hole.range.address <= destination &&
          hole.range.address + hole.range.size >= destination + 2 * page);
    CHECK(process->used_memory() == used);

    for (unsigned i = 0; i < 2; ++i) {
        auto map = registers_with();
        map[0] = destination + i * page; map[1] = heap + i * page; map[2] = page;
        CHECK(handled(fixture.memory->dispatch(4, map)) && map[0] == 0);
    }
    CHECK(process->query_memory(heap).range.size == page);
    CHECK(process->query_memory(heap + page).range.size == page);
    CHECK(process->query_memory(destination).range.size == page);
    CHECK(process->query_memory(destination + page).range.size == page);
    auto wrong = registers_with();
    wrong[0] = destination; wrong[1] = heap + page; wrong[2] = page;
    CHECK(handled(fixture.memory->dispatch(5, wrong)) && wrong[0] == result_invalid_region);
    for (unsigned i = 0; i < 2; ++i) {
        auto unmap = registers_with();
        unmap[0] = destination + i * page; unmap[1] = heap + i * page; unmap[2] = page;
        CHECK(handled(fixture.memory->dispatch(5, unmap)) && unmap[0] == 0);
        if (i == 0) CHECK(process->query_memory(heap + page).attributes == 1);
    }
    CHECK(process->query_memory(heap).range.size == 2 * mib);
    CHECK(process->used_memory() == used && process->total_memory() == 16 * mib - (private_mode ? 65536 : 0));
    const auto tail = process->query_memory(std::numeric_limits<std::uint64_t>::max());
    CHECK(tail.type == horizon::MemoryType::reserved && tail.permissions == 0 && tail.attributes == 0);
    CHECK(tail.range.address == horizon::ProcessRecord::query_end &&
          tail.range.size == std::uint64_t{0} - horizon::ProcessRecord::query_end);
}

void test_ipc_copy_uses_registered_alias() {
    Fixture fixture;
    const auto page = native_page_size();
    auto pin = allocate(fixture, page, horizon::MemoryType::normal,
                        memory::Permission::read | memory::Permission::write);
    if (!pin) return;
    const auto address = pin->range().address;
    auto made = platform::ProcessMemoryRegion::writable_host_alias(address, page);
    CHECK(std::holds_alternative<platform::ProcessMemoryRegion>(made));
    if (!std::holds_alternative<platform::ProcessMemoryRegion>(made)) return;
    auto alias = std::get<platform::ProcessMemoryRegion>(std::move(made));
    auto* host = reinterpret_cast<void*>(address);
    auto* writable = reinterpret_cast<void*>(alias.address());
    memory::WriteTracker::add_range(host, page, writable);
    CHECK(memory::WriteTracker::protect_checked(host, page));
    const auto stamp = memory::WriteTracker::now();
    const auto before = memory::WriteTracker::stats();
    const std::array<std::byte, 4> payload{std::byte{4}, std::byte{3}, std::byte{2}, std::byte{1}};
    CHECK(fixture.memory->ipc_buffers().write_memory(address + 8, payload));
    CHECK(std::equal(payload.begin(), payload.end(), pin->bytes().begin() + 8));
    CHECK(memory::WriteTracker::written_since(host, page, stamp));
    CHECK(memory::WriteTracker::stats().protect_calls == before.protect_calls);
    CHECK(memory::WriteTracker::stats().faults == before.faults);
    // Admission still checks the complete guest range before selecting an alias.
    CHECK(!fixture.memory->ipc_buffers().write_memory(address + page - 1, payload));
    static_cast<volatile std::byte*>(host)[16] = std::byte{0x66};
    CHECK(memory::WriteTracker::stats().faults == before.faults + 1);
    memory::WriteTracker::remove_range(host, writable);
}

// Expose the same mapping authority through both readers. The IPC reader
// deliberately stretches its pin lifetime across the mutation rendezvous, so
// reverting the production observer adapter to IPC fails deterministically.
class RendezvousMemory final : public GuestMemory {
public:
    explicit RendezvousMemory(std::shared_ptr<GuestMemory> memory) : memory(std::move(memory)) {}
    std::shared_ptr<horizon::ProcessRecord> process() const noexcept override { return memory->process(); }
    LaunchResult<GuestMemoryPin> allocate(std::uint64_t bytes, horizon::MemoryType type, memory::Permission access) override {
        return memory->allocate(bytes, type, access);
    }
    LaunchStatus release(horizon::AddressRange range) override { return memory->release(range); }
    LaunchResult<horizon::QueryDisposition> dispatch(std::uint16_t svc, horizon::QueryRegisters& r) override {
        return memory->dispatch(svc, r);
    }
    LaunchResult<GuestMemoryPin> pin(horizon::AddressRange range, memory::Permission access) override {
        return memory->pin(range, access);
    }
    bool copy_observation(std::uint64_t address, std::span<std::byte> out) const noexcept override {
        const auto copied = memory->copy_observation(address, out);
        ++checked_reads;
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        return copied;
    }
    horizon::GuestCopyOut copy_out() noexcept override { return memory->copy_out(); }
    horizon::GuestMemoryMapper service_mapper() noexcept override { return memory->service_mapper(); }
    horizon::GuestBuffers ipc_buffers() noexcept override {
        return {[](void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
            auto& self = *static_cast<RendezvousMemory*>(context);
            auto retained = self.memory->pin({address, out.size()}, memory::Permission::read);
            auto* pin = std::get_if<GuestMemoryPin>(&retained);
            if (pin) std::copy((*pin)->bytes().begin(), (*pin)->bytes().end(), out.begin());
            ++self.ipc_reads;
            self.rendezvous.arrive_and_wait();
            self.rendezvous.arrive_and_wait();
            return pin != nullptr;
        }, nullptr, this};
    }
    std::shared_ptr<GuestMemory> memory;
    mutable std::barrier<> rendezvous{2};
    mutable unsigned checked_reads = 0, ipc_reads = 0;
};

void test_observation_copy_does_not_retain_mappings() {
    Fixture fixture;
    if (!fixture.memory) return;
    auto allocation = allocate(fixture, native_page_size(), horizon::MemoryType::normal,
                               memory::Permission::read | memory::Permission::write);
    CHECK(allocation != nullptr);
    if (!allocation) return;
    const auto range = allocation->range();
    std::fill(allocation->writable_bytes().begin(), allocation->writable_bytes().end(), std::byte{0x5a});
    allocation.reset();
    std::array<std::byte, 9> bytes{};
    for (const std::size_t width : {1u, 2u, 4u, 8u}) {
        CHECK(fixture.memory->copy_observation(range.address, std::span(bytes).first(width)));
        CHECK(bytes[width - 1] == std::byte{0x5a});
    }
    CHECK(!fixture.memory->copy_observation(range.address, bytes));
    CHECK(!fixture.memory->copy_observation(range.address, {}));
    CHECK(!fixture.memory->copy_observation(UINT64_MAX - 1, std::span(bytes).first(4)));
    CHECK(!fixture.memory->copy_observation(range.address + range.size - 1, std::span(bytes).first(4)));

    const auto exercise = [&](std::uint64_t address, unsigned rounds, const auto& mutate) {
        auto wrapped = std::make_shared<RendezvousMemory>(fixture.memory);
        auto guest = std::get<std::shared_ptr<GuestClock>>(create_guest_clock());
        const auto sample = std::get<ClockSample>(guest->sample());
        auto base = std::get<std::shared_ptr<PresentationClock>>(create_presentation_clock(
            guest, {sample.instant, std::chrono::nanoseconds{16'666'667}}));
        auto clock = std::make_shared<tools::PacingClock>(base, std::nullopt,
            std::make_unique<std::ostringstream>(),
            std::get<std::vector<tools::PacingWatch>>(tools::parse_pacing_watches(std::to_string(address) + ":8")));
        auto observer = attach_presentation_observer(
            [](const services::GraphicsPresentation&) { return services::GraphicsPresentationResult{true, {}}; },
            clock, wrapped);
        std::thread watcher([&] {
            for (unsigned i = 0; i < rounds; ++i) {
                services::GraphicsPresentation frame;
                frame.frame_number = i + 1;
                frame.swap_interval = 1;
                CHECK(observer(frame).completed);
            }
        });
        for (unsigned i = 0; i < rounds; ++i) {
            wrapped->rendezvous.arrive_and_wait();
            mutate(i);
            wrapped->rendezvous.arrive_and_wait();
        }
        watcher.join();
        CHECK(wrapped->checked_reads == rounds && wrapped->ipc_reads == 0);
        CHECK(!clock->observation_failure());
    };
    exercise(range.address, 128, [&](unsigned i) {
        auto registers = registers_with();
        registers[0] = range.address;
        registers[1] = range.size;
        registers[2] = i % 2 ? 3 : 0;
        CHECK(handled(fixture.memory->dispatch(2, registers)) && registers[0] == 0);
    });
    exercise(range.address, 1, [&](unsigned) { CHECK(succeeded(fixture.memory->release(range))); });
    CHECK(!fixture.memory->copy_observation(range.address, std::span(bytes).first(8)));

    auto heap = registers_with();
    heap[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, heap)) && heap[0] == 0);
    const auto heap_address = heap[1];
    exercise(heap_address, 32, [&](unsigned i) {
        auto registers = registers_with();
        registers[1] = i % 2 ? 2 * mib : 0;
        CHECK(handled(fixture.memory->dispatch(1, registers)) && registers[0] == 0);
    });
}

void private_interval_authority() {
    auto profile = synthetic_profile();
    profile.process.platform = GuestPlatform::switch2;
    profile.process.baseline = horizon::Baseline::horizon_23_0_1;
    profile.process.private_system_resource_bytes = 3 * 4096;
    auto handles = std::make_shared<horizon::HandleTable>(64);
    auto created = create_guest_memory(profile, {16 * mib}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(created));
    if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(created)) return;
    auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(created));
    auto process = memory->process();
    CHECK(process->memory_intervals() != nullptr);
    CHECK(process->used_system_resource() == 4096);
    const auto page = native_page_size();
    auto made = memory->allocate(128 * page, horizon::MemoryType::normal,
        memory::Permission::read | memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(made));
    if (!std::holds_alternative<GuestMemoryPin>(made)) return;
    auto pin = std::get<GuestMemoryPin>(std::move(made));
    const auto range = pin->range();
    pin->writable_bytes().front() = std::byte{0x41}; pin.reset();
    unsigned exhausted_at = 0;
    for (unsigned index = 1; index < 128; index += 2) {
        const auto before = process->memory_map();
        const auto used = process->used_memory(), total = process->total_memory();
        horizon::QueryRegisters registers{};
        registers[0] = range.address + index * page; registers[1] = page; registers[2] = 1;
        const auto result = memory->dispatch(2, registers);
        CHECK(std::holds_alternative<horizon::QueryDisposition>(result));
        if (!std::holds_alternative<horizon::QueryDisposition>(result)) return;
        if (registers[0] == 0xce01) {
            exhausted_at = index;
            const auto after = process->memory_map();
            CHECK(after.size() == before.size());
            for (std::size_t i = 0; i < std::min(after.size(), before.size()); ++i)
                CHECK(horizon::encode_memory_info(after[i]) == horizon::encode_memory_info(before[i]));
            CHECK(process->used_memory() == used && process->total_memory() == total);
            CHECK(process->query_memory(range.address + index * page).permissions == 3);
            CHECK(process->used_system_resource() == 4096);
            break;
        }
        CHECK(registers[0] == 0 && process->query_memory(range.address + index * page).permissions == 1);
        auto rejected = memory->pin({range.address + index * page, page}, memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<LaunchFailure>(rejected));
    }
    CHECK(exhausted_at != 0); if (!exhausted_at) return;
    horizon::QueryRegisters recover{};
    recover[0] = range.address + page; recover[1] = page; recover[2] = 3;
    CHECK(std::holds_alternative<horizon::QueryDisposition>(memory->dispatch(2, recover)) && recover[0] == 0);
    horizon::QueryRegisters retry{};
    retry[0] = range.address + exhausted_at * page; retry[1] = page; retry[2] = 1;
    CHECK(std::holds_alternative<horizon::QueryDisposition>(memory->dispatch(2, retry)) && retry[0] == 0);
    CHECK(process->query_memory(range.address + exhausted_at * page).permissions == 1);
    auto retained = memory->pin({range.address, page}, memory::Permission::read);
    CHECK(std::holds_alternative<GuestMemoryPin>(retained));
    std::weak_ptr<horizon::ProcessRecord> owner = memory->process();
    // ProcessRecord and the pin keep the same pool-backed authority alive.
    memory.reset(); process.reset();
    CHECK(!owner.expired());
    if (auto* view = std::get_if<GuestMemoryPin>(&retained)) CHECK((*view)->bytes().front() == std::byte{0x41});
    retained = LaunchFailure{};
    CHECK(owner.expired());
}

} // namespace

// The Switch 2 alias window asks for 64 GiB and steps down to 2 GiB when the
// host cannot place the arena; Switch 1 keeps its profile's size.
void test_alias_window_sizes() {
    constexpr std::uint64_t gib = 1ull << 30;
    CHECK(switch2_alias_window == 64 * gib && switch2_minimum_alias_window == 2 * gib);
    const auto switch2 = alias_window_sizes(switch2_alias_window, true);
    CHECK(switch2.size() == 249 && switch2.front() == 64 * gib && switch2.back() == 2 * gib);
    CHECK(alias_window_sizes(4 * gib, true).size() == 9);
    for (std::size_t i = 1; i < switch2.size(); ++i) CHECK(switch2[i - 1] - switch2[i] == alias_window_step);
    // An unaligned request still ends at the minimum, never below it.
    const auto odd = alias_window_sizes(2 * gib + 300 * mib, true);
    CHECK(odd.size() == 3 && odd[0] == 2 * gib + 300 * mib && odd[1] == 2 * gib + 44 * mib && odd[2] == 2 * gib);
    CHECK(alias_window_sizes(2 * gib, true) == std::vector<std::uint64_t>{2 * gib});
    CHECK(alias_window_sizes(gib, true) == std::vector<std::uint64_t>{gib});
    CHECK(alias_window_sizes(64 * gib, false) == std::vector<std::uint64_t>{64 * gib});
}

int main() {
    private_interval_authority();
    test_alias_window_sizes();
    for (bool mode : {false, true}) {
        private_mode = mode;
        test_observation_copy_does_not_retain_mappings();
        test_factory_and_host_allocations();
        test_pin_validation_and_backing_lifetime();
        test_release_after_partial_protection();
        test_heap_dispatch_and_pin_conflicts();
        test_permission_dispatch_and_native_protection();
        test_stack_alias_map_and_exact_unmap();
        test_read_write_shared_memory_alias_and_handles();
        test_shared_mapping_retains_object_after_handle_close();
        test_sub_page_shared_memory_mapping();
        test_read_only_shared_memory_mapping();
        test_copy_out_native_atomics();
        test_ipc_whole_range_and_ordered_word_copy();
        test_ipc_copy_uses_registered_alias();
        test_service_mapper_retention_and_lifetime();
        test_firmware_validation_order();
        test_heap_tail_validation_and_accounting();
        test_query_boundaries_and_alias_identity();
    }
    return failures.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}
