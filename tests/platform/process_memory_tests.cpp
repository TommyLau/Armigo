// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "process_memory.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>

namespace {
using armigo::memory::Permission;
using armigo::platform::ProcessMemoryError;
using armigo::platform::ProcessMemoryRegion;
using armigo::platform::ProcessMemoryResult;
using armigo::platform::native_page_size;
using armigo::platform::protect_process_memory;

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

ProcessMemoryRegion require_region(ProcessMemoryResult result, const char* message) {
    if (const auto* error = std::get_if<ProcessMemoryError>(&result)) {
        std::fprintf(stderr, "%s: %s\n", message, armigo::platform::describe(*error));
        std::exit(1);
    }
    return std::get<ProcessMemoryRegion>(std::move(result));
}

#if defined(__APPLE__)
std::optional<vm_region_basic_info_data_64_t> mapping_info(std::uint64_t pointer) {
    mach_vm_address_t address = pointer;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    if (result != KERN_SUCCESS || address > pointer || pointer - address >= size) return {};
    return info;
}

bool has_protection(std::uint64_t address, vm_prot_t expected) {
    const auto info = mapping_info(address);
    return info && info->protection == expected;
}
#endif

bool reservation_does_not_clobber(std::uint64_t page) {
    auto first = require_region(ProcessMemoryRegion::reserve(page), "Initial reservation failed");
    const auto second = ProcessMemoryRegion::reserve(page, first.address());
    return check(std::holds_alternative<ProcessMemoryError>(second) &&
                     std::get<ProcessMemoryError>(second) == ProcessMemoryError::address_occupied,
                 "A fixed reservation replaced occupied native pages.");
}

bool commit_protect_and_geometry(std::uint64_t page) {
    auto region = require_region(ProcessMemoryRegion::reserve(page * 2),
                                 "Commit test reservation failed");
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_NONE),
               "A new reservation was accessible.")) {
        return false;
    }
#endif
    if (!check(region.commit(0, page, Permission::read | Permission::write).ok(),
               "Committing a native page failed.")) {
        return false;
    }
    auto* bytes = reinterpret_cast<std::byte*>(region.address());
    for (std::uint64_t index = 0; index < page; ++index) {
        if (!check(bytes[index] == std::byte{0}, "Committed memory was not zero-filled.")) {
            return false;
        }
    }
    bytes[0] = std::byte{0x5a};
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_READ | VM_PROT_WRITE),
               "Committed page protection is wrong.")) {
        return false;
    }
#endif
    if (!check(region.protect(0, page, Permission::read).ok(),
               "Protecting a native page failed")) {
        return false;
    }
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_READ),
               "Read-only protection was not applied.")) {
        return false;
    }
#endif
    const auto rejected = region.protect(
        0, page, Permission::write | Permission::execute);
    if (!check(rejected.error == ProcessMemoryError::write_execute_conflict,
               "Writable executable protection was not rejected.")) {
        return false;
    }
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_READ),
               "Rejected protection changed the native mapping.")) {
        return false;
    }
#endif
    const std::uint64_t subpage = page > 4096 ? 4096 : page / 2;
    return check(region.commit(0, subpage, Permission::read).error ==
                     ProcessMemoryError::unaligned_size,
                 "A native subpage commit was accepted.") &&
           check(region.protect(subpage, page, Permission::read).error ==
                     ProcessMemoryError::unaligned_address,
                 "A native subpage address was accepted.") &&
           check(std::get<ProcessMemoryError>(ProcessMemoryRegion::reserve(subpage)) ==
                     ProcessMemoryError::unaligned_size,
                 "Reservation size was implicitly rounded.");
}

bool protect_middle_page(std::uint64_t page) {
    auto region = require_region(ProcessMemoryRegion::reserve(page * 3),
                                 "Middle-page protection reservation failed");
    if (!check(region.commit(0, page * 3, Permission::read | Permission::write).ok(),
               "Three-page commit failed.")) {
        return false;
    }
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_READ | VM_PROT_WRITE) &&
                   has_protection(region.address() + page, VM_PROT_READ | VM_PROT_WRITE) &&
                   has_protection(region.address() + page * 2,
                                  VM_PROT_READ | VM_PROT_WRITE),
               "Commit remap did not retain its prepared native protection.")) {
        return false;
    }
#endif
    if (!check(region.protect(page, page, Permission::read).ok(),
               "Protecting a page inside an enclosing Mach region failed.")) {
        return false;
    }
#if defined(__APPLE__)
    return check(has_protection(region.address(), VM_PROT_READ | VM_PROT_WRITE),
                 "Middle-page protection changed the preceding page.") &&
           check(has_protection(region.address() + page, VM_PROT_READ),
                 "Middle-page protection was not applied.") &&
           check(has_protection(region.address() + page * 2,
                                VM_PROT_READ | VM_PROT_WRITE),
                 "Middle-page protection changed the following page.");
#else
    return true;
#endif
}

bool alias_coherence_and_protection(std::uint64_t page) {
    auto source = require_region(ProcessMemoryRegion::reserve(page),
                                 "Alias source reservation failed");
    auto destination = require_region(ProcessMemoryRegion::reserve(page),
                                      "Alias destination reservation failed");
    if (!check(source.commit(0, page, Permission::read | Permission::write).ok(),
               "Alias source commit failed.")) {
        return false;
    }
    auto* source_bytes = reinterpret_cast<std::byte*>(source.address());
    source_bytes[7] = std::byte{0x31};
    if (!check(destination.alias(0, page, source.address(), Permission::read).ok(),
               "Shared alias creation failed.")) {
        return false;
    }
    const auto* destination_bytes = reinterpret_cast<const std::byte*>(destination.address());
    if (!check(destination_bytes[7] == std::byte{0x31},
               "Alias did not expose the source bytes.")) {
        return false;
    }
    source_bytes[7] = std::byte{0x72};
    if (!check(destination_bytes[7] == std::byte{0x72},
               "Source writes were not coherent through the alias.")) {
        return false;
    }
#if defined(__APPLE__)
    if (!check(has_protection(source.address(), VM_PROT_READ | VM_PROT_WRITE) &&
                   has_protection(destination.address(), VM_PROT_READ),
               "Alias protection changed the source protection.")) {
        return false;
    }
#endif
    auto copy = require_region(
        ProcessMemoryRegion::alias_copy(source.address(), page,
                                        Permission::read | Permission::write),
        "Owned alias copy failed");
    auto* copy_bytes = reinterpret_cast<std::byte*>(copy.address());
    copy_bytes[8] = std::byte{0x44};
    return check(source_bytes[8] == std::byte{0x44},
                 "Owned alias copy was not shared with its source.");
}

bool writable_host_alias_lifetime(std::uint64_t page) {
    auto source = require_region(ProcessMemoryRegion::reserve(page), "Host alias source failed");
    if (!check(source.commit(0, page, Permission::read | Permission::write).ok(),
               "Host alias source commit failed.")) return false;
    auto* bytes = reinterpret_cast<std::byte*>(source.address());
    bytes[9] = std::byte{0x23};
    if (!check(source.protect(0, page, Permission::read).ok(),
               "Host alias source protection failed.")) return false;
    std::uint64_t alias_address = 0;
    {
        auto alias = require_region(ProcessMemoryRegion::writable_host_alias(source.address(), page),
                                    "Writable host alias failed");
        alias_address = alias.address();
        if (!check(alias_address >= (std::uint64_t{1} << 39),
                   "Host alias occupied the guest address space.")) return false;
        auto* writable = reinterpret_cast<std::byte*>(alias.address());
        if (!check(writable[9] == std::byte{0x23}, "Host alias lost existing contents.")) return false;
        writable[9] = std::byte{0x51};
        if (!check(bytes[9] == std::byte{0x51}, "Host alias was a private copy.")) return false;
#if defined(__APPLE__)
        if (!check(has_protection(source.address(), VM_PROT_READ) &&
                   has_protection(alias.address(), VM_PROT_READ | VM_PROT_WRITE),
                   "Host alias changed source protection.")) return false;
#endif
        auto moved = std::move(alias);
        if (!check(!alias.valid() && moved.address() == alias_address,
                   "Moving the host alias lost ownership.")) return false;
    }
#if defined(__APPLE__)
    if (!check(!mapping_info(alias_address), "Destroying the host alias leaked its mapping.")) return false;
    if (!check(has_protection(source.address(), VM_PROT_READ),
               "Destroying the host alias changed the source.")) return false;
#endif
    return true;
}

bool decommit_regrow_and_failure_rollback(std::uint64_t page) {
    auto region = require_region(ProcessMemoryRegion::reserve(page),
                                 "Decommit test reservation failed");
    if (!check(region.commit(0, page, Permission::read | Permission::write).ok(),
               "Initial decommit test commit failed.")) {
        return false;
    }
    auto* bytes = reinterpret_cast<std::byte*>(region.address());
    bytes[0] = std::byte{0xa5};
#if defined(__APPLE__)
    auto restricted_source = require_region(ProcessMemoryRegion::reserve(page),
                                            "Restricted alias source reservation failed");
    if (!check(restricted_source.commit(0, page, Permission::read).ok(),
               "Restricted alias source commit failed.") ||
        !check(::mach_vm_protect(mach_task_self(), restricted_source.address(), page,
                                 TRUE, VM_PROT_READ) == KERN_SUCCESS,
               "Could not lower the alias source maximum protection.")) {
        return false;
    }
    const auto failed_preparation = region.alias(
        0, page, restricted_source.address(), Permission::read | Permission::write);
    if (!check(failed_preparation.error == ProcessMemoryError::protection_failed,
               "An alias exceeding source maximum protection was accepted.") ||
        !check(bytes[0] == std::byte{0xa5},
               "Failed alias preparation changed destination bytes.") ||
        !check(has_protection(region.address(), VM_PROT_READ | VM_PROT_WRITE),
               "Failed alias preparation changed destination protection.")) {
        return false;
    }
#endif
    const auto failed_alias = region.alias(0, page, 0, Permission::read);
    if (!check(failed_alias.error == ProcessMemoryError::remap_failed,
               "An unmapped alias source was accepted.") ||
        !check(bytes[0] == std::byte{0xa5},
               "A failed alias changed destination bytes.")) {
        return false;
    }
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_READ | VM_PROT_WRITE),
               "A failed alias changed destination protection.")) {
        return false;
    }
#endif
    if (!check(region.decommit(0, page).ok(), "Decommit failed.")) return false;
#if defined(__APPLE__)
    if (!check(has_protection(region.address(), VM_PROT_NONE),
               "Decommitted pages remained accessible.")) {
        return false;
    }
#endif
    if (!check(region.commit(0, page, Permission::read | Permission::write).ok(),
               "Regrowing a decommitted page failed.")) {
        return false;
    }
    bytes = reinterpret_cast<std::byte*>(region.address());
    if (!check(bytes[0] == std::byte{0}, "Regrown backing retained old bytes.")) return false;
    if (!check(protect_process_memory(region.address(), page, Permission::read).ok(),
               "Borrowed mapping protection failed.")) {
        return false;
    }
#if defined(__APPLE__)
    return check(has_protection(region.address(), VM_PROT_READ),
                 "Borrowed mapping protection was not applied.");
#else
    return true;
#endif
}

bool large_aligned_reservation(std::uint64_t page) {
    constexpr std::uint64_t alignment = 0x10000;
    constexpr std::uint64_t bytes = 8ull << 30;
    auto region = require_region(ProcessMemoryRegion::reserve_low_aligned(bytes, alignment),
                                 "Bounded large reservation failed.");
    if (!region.valid()) return false;
    if (!check(region.address() >= (128ull << 20) && region.address() % alignment == 0 &&
               region.size() == bytes && region.address() + region.size() <= (1ull << 39),
               "Bounded reservation geometry is invalid.")) return false;
    const auto collision = ProcessMemoryRegion::reserve(page, region.address());
    if (!check(std::holds_alternative<ProcessMemoryError>(collision),
               "Bounded reservation did not own its native range.")) return false;
    auto neighbor = require_region(ProcessMemoryRegion::reserve_low_aligned(page, alignment),
                                   "Second bounded reservation failed.");
    if (!check(neighbor.valid() && (neighbor.address() + page <= region.address() ||
               neighbor.address() >= region.address() + bytes),
               "Bounded reservation overlaps a live owner.")) return false;
    for (const auto invalid : {std::uint64_t{0}, page / 2, page + 1})
        if (!check(std::holds_alternative<ProcessMemoryError>(
                       ProcessMemoryRegion::reserve_low_aligned(page, invalid)),
                   "Invalid reservation alignment was admitted.")) return false;
    for (const auto invalid : {std::uint64_t{0}, page + 1, (std::uint64_t{1} << 39) + page, UINT64_MAX})
        if (!check(std::holds_alternative<ProcessMemoryError>(
                       ProcessMemoryRegion::reserve_low_aligned(invalid, alignment)),
                   "Invalid reservation size was admitted.")) return false;
#if defined(__APPLE__)
    return check(has_protection(region.address(), VM_PROT_NONE) &&
                 has_protection(region.address() + bytes - page, VM_PROT_NONE),
                 "Bounded reservation must stay inaccessible without physical commitment.");
#else
    return true;
#endif
}

} // namespace

int main() {
    const auto page = native_page_size();
    if (!check(page != 0 && (page & (page - 1)) == 0,
               "Native page size is unavailable or invalid.")) {
        return 1;
    }
    if (!large_aligned_reservation(page) ||
        !reservation_does_not_clobber(page) ||
        !commit_protect_and_geometry(page) ||
        !protect_middle_page(page) ||
        !alias_coherence_and_protection(page) ||
        !writable_host_alias_lifetime(page) ||
        !decommit_regrow_and_failure_rollback(page)) {
        return 1;
    }
    std::puts("Process memory reservation, protection, alias, and rollback checks passed.");
    return 0;
}
