// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "process_memory.h"

#include <limits>
#include <utility>

#if defined(__APPLE__)
#include <mach/mach.h>
#include "mach_vm.h"
#include <mach/vm_statistics.h>
#endif

namespace armigo::platform {
namespace {

constexpr std::uint64_t query_end = std::uint64_t{1} << 39;

bool writable_executable(memory::Permission permission) noexcept {
    return memory::has_permission(
        permission, memory::Permission::write | memory::Permission::execute);
}

ProcessMemoryStatus validate_permission(memory::Permission permission) noexcept {
    if (!memory::has_known_permission_bits(permission)) {
        return {ProcessMemoryError::invalid_permission};
    }
    if (writable_executable(permission)) {
        return {ProcessMemoryError::write_execute_conflict};
    }
    return {};
}

ProcessMemoryStatus validate_native_range(std::uint64_t address,
                                          std::uint64_t size) noexcept {
    const auto page = native_page_size();
    if (page == 0) return {ProcessMemoryError::system_failure};
    if (size == 0) return {ProcessMemoryError::invalid_size};
    if ((address & (page - 1)) != 0) {
        return {ProcessMemoryError::unaligned_address};
    }
    if ((size & (page - 1)) != 0) return {ProcessMemoryError::unaligned_size};
    if (address > std::numeric_limits<std::uint64_t>::max() - size) {
        return {ProcessMemoryError::address_overflow};
    }
    return {};
}

ProcessMemoryStatus validate_supported_range(std::uint64_t address,
                                             std::uint64_t size) noexcept {
    auto status = validate_native_range(address, size);
    if (!status.ok()) return status;
    if (address >= query_end || size > query_end - address) {
        return {ProcessMemoryError::out_of_range};
    }
    return {};
}

ProcessMemoryStatus validate_subrange(std::uint64_t region_size,
                                      std::uint64_t offset,
                                      std::uint64_t size) noexcept {
    const auto page = native_page_size();
    if (page == 0) return {ProcessMemoryError::system_failure};
    if (size == 0) return {ProcessMemoryError::invalid_size};
    if ((offset & (page - 1)) != 0) return {ProcessMemoryError::unaligned_address};
    if ((size & (page - 1)) != 0) return {ProcessMemoryError::unaligned_size};
    if (offset > std::numeric_limits<std::uint64_t>::max() - size) {
        return {ProcessMemoryError::address_overflow};
    }
    if (offset > region_size || size > region_size - offset) {
        return {ProcessMemoryError::out_of_range};
    }
    return {};
}

#if defined(__APPLE__)
vm_prot_t native_protection(memory::Permission permission) noexcept {
    vm_prot_t protection = VM_PROT_NONE;
    if (memory::has_permission(permission, memory::Permission::read)) {
        protection |= VM_PROT_READ;
    }
    if (memory::has_permission(permission, memory::Permission::write)) {
        protection |= VM_PROT_WRITE;
    }
    if (memory::has_permission(permission, memory::Permission::execute)) {
        protection |= VM_PROT_EXECUTE;
    }
    return protection;
}

ProcessMemoryError allocation_error(kern_return_t result, bool fixed) noexcept {
    if (fixed && result == KERN_NO_SPACE) return ProcessMemoryError::address_occupied;
    if (result == KERN_NO_SPACE || result == KERN_RESOURCE_SHORTAGE) {
        return ProcessMemoryError::allocation_failed;
    }
    return ProcessMemoryError::system_failure;
}

ProcessMemoryStatus verify_max_protection(std::uint64_t address,
                                          std::uint64_t size,
                                          vm_prot_t requested) noexcept {
    mach_vm_address_t cursor = address;
    const mach_vm_address_t end = address + size;
    while (cursor < end) {
        mach_vm_address_t region_address = cursor;
        mach_vm_size_t region_size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const auto result = ::mach_vm_region(
            mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) {
            ::mach_port_deallocate(mach_task_self(), object);
        }
        if (result != KERN_SUCCESS) {
            return {ProcessMemoryError::protection_failed, static_cast<int>(result)};
        }
        if (region_address > cursor || region_size == 0 ||
            region_size > std::numeric_limits<mach_vm_address_t>::max() - region_address) {
            return {ProcessMemoryError::protection_failed, KERN_INVALID_ADDRESS};
        }
        const auto region_end = region_address + region_size;
        if (region_end <= cursor) {
            return {ProcessMemoryError::protection_failed, KERN_INVALID_ADDRESS};
        }
        if ((requested & ~info.max_protection) != 0) {
            return {ProcessMemoryError::protection_failed, KERN_PROTECTION_FAILURE};
        }
        cursor = region_end < end ? region_end : end;
    }
    return {};
}

ProcessMemoryStatus apply_protection(std::uint64_t address, std::uint64_t size,
                                     memory::Permission permission) noexcept {
    const auto requested = native_protection(permission);
    auto status = verify_max_protection(address, size, requested);
    if (!status.ok()) return status;
    const auto result = ::mach_vm_protect(mach_task_self(), address, size, FALSE, requested);
    if (result != KERN_SUCCESS) {
        return {ProcessMemoryError::protection_failed, static_cast<int>(result)};
    }
    return {};
}

class TemporaryMapping {
public:
    TemporaryMapping() = default;
    TemporaryMapping(const TemporaryMapping&) = delete;
    TemporaryMapping& operator=(const TemporaryMapping&) = delete;
    ~TemporaryMapping() {
        if (address_ != 0) ::mach_vm_deallocate(mach_task_self(), address_, size_);
    }

    mach_vm_address_t* address_pointer() noexcept { return &address_; }
    mach_vm_address_t address() const noexcept { return address_; }
    void set_size(mach_vm_size_t size) noexcept { size_ = size; }

private:
    mach_vm_address_t address_ = 0;
    mach_vm_size_t size_ = 0;
};

ProcessMemoryStatus publish_mapping(std::uint64_t destination,
                                    std::uint64_t size,
                                    std::uint64_t source) noexcept {
    mach_vm_address_t target = destination;
    vm_prot_t current = VM_PROT_NONE;
    vm_prot_t maximum = VM_PROT_NONE;
    const auto result = ::mach_vm_remap(
        mach_task_self(), &target, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
        mach_task_self(), source, FALSE, &current, &maximum, VM_INHERIT_NONE);
    if (result != KERN_SUCCESS) {
        return {ProcessMemoryError::remap_failed, static_cast<int>(result)};
    }
    if (target != destination) {
        if (target != 0) ::mach_vm_deallocate(mach_task_self(), target, size);
        return {ProcessMemoryError::system_failure};
    }
    return {};
}

ProcessMemoryStatus make_fresh_mapping(std::uint64_t destination,
                                       std::uint64_t size,
                                       memory::Permission permission) noexcept {
    TemporaryMapping temporary;
    temporary.set_size(size);
    auto result = ::mach_vm_allocate(
        mach_task_self(), temporary.address_pointer(), size, VM_FLAGS_ANYWHERE);
    if (result != KERN_SUCCESS) {
        return {allocation_error(result, false), static_cast<int>(result)};
    }
    auto status = apply_protection(temporary.address(), size, permission);
    if (!status.ok()) return status;
    return publish_mapping(destination, size, temporary.address());
}

ProcessMemoryStatus make_shared_mapping(std::uint64_t destination,
                                        std::uint64_t size,
                                        std::uint64_t source,
                                        memory::Permission permission) noexcept {
    TemporaryMapping temporary;
    temporary.set_size(size);
    vm_prot_t current = VM_PROT_NONE;
    vm_prot_t maximum = VM_PROT_NONE;
    const auto result = ::mach_vm_remap(
        mach_task_self(), temporary.address_pointer(), size, 0, VM_FLAGS_ANYWHERE,
        mach_task_self(), source, FALSE, &current, &maximum, VM_INHERIT_NONE);
    if (result != KERN_SUCCESS) {
        return {ProcessMemoryError::remap_failed, static_cast<int>(result)};
    }
    auto status = apply_protection(temporary.address(), size, permission);
    if (!status.ok()) return status;
    return publish_mapping(destination, size, temporary.address());
}
#endif

} // namespace

std::uint64_t native_page_size() noexcept {
#if defined(__APPLE__)
    const std::uint64_t page = vm_page_size;
    return page != 0 && (page & (page - 1)) == 0 ? page : 0;
#else
    return 0;
#endif
}

ProcessMemoryStatus protect_process_memory(
    std::uint64_t address, std::uint64_t size,
    memory::Permission permission) noexcept {
    auto status = validate_permission(permission);
    if (!status.ok()) return status;
    status = validate_native_range(address, size);
    if (!status.ok()) return status;
#if defined(__APPLE__)
    return apply_protection(address, size, permission);
#else
    return {ProcessMemoryError::unsupported};
#endif
}

ProcessMemoryResult ProcessMemoryRegion::reserve(
    std::uint64_t bytes, std::uint64_t requested_address) noexcept {
    const auto page = native_page_size();
    if (page == 0) return ProcessMemoryError::unsupported;
    if (bytes == 0) return ProcessMemoryError::invalid_size;
    if ((bytes & (page - 1)) != 0) return ProcessMemoryError::unaligned_size;
    if (requested_address != 0) {
        const auto status = validate_supported_range(requested_address, bytes);
        if (!status.ok()) return status.error;
    } else if (bytes > query_end) {
        return ProcessMemoryError::out_of_range;
    }
#if defined(__APPLE__)
    mach_vm_address_t address = requested_address;
    const bool fixed = requested_address != 0;
    const int flags = fixed ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
    auto result = ::mach_vm_allocate(mach_task_self(), &address, bytes, flags);
    if (result != KERN_SUCCESS) return allocation_error(result, fixed);
    if (address >= query_end || bytes > query_end - address) {
        ::mach_vm_deallocate(mach_task_self(), address, bytes);
        return ProcessMemoryError::unsupported;
    }
    result = ::mach_vm_protect(mach_task_self(), address, bytes, FALSE, VM_PROT_NONE);
    if (result != KERN_SUCCESS) {
        ::mach_vm_deallocate(mach_task_self(), address, bytes);
        return ProcessMemoryError::protection_failed;
    }
    return ProcessMemoryRegion(address, bytes);
#else
    (void)requested_address;
    return ProcessMemoryError::unsupported;
#endif
}

ProcessMemoryResult ProcessMemoryRegion::alias_copy(
    std::uint64_t source_address, std::uint64_t size,
    memory::Permission permission) noexcept {
    return shared_alias(source_address, size, permission, true);
}

ProcessMemoryResult ProcessMemoryRegion::writable_host_alias(
    std::uint64_t source_address, std::uint64_t size) noexcept {
    return shared_alias(source_address, size,
                        memory::Permission::read | memory::Permission::write, false);
}

ProcessMemoryResult ProcessMemoryRegion::shared_alias(
    std::uint64_t source_address, std::uint64_t size,
    memory::Permission permission, bool guest_address) noexcept {
    auto status = validate_permission(permission);
    if (!status.ok()) return status.error;
    status = validate_native_range(source_address, size);
    if (!status.ok()) return status.error;
#if defined(__APPLE__)
    vm_prot_t current = VM_PROT_NONE;
    vm_prot_t maximum = VM_PROT_NONE;
    mach_vm_address_t address = guest_address ? 0 : query_end;
    const auto result = ::mach_vm_remap(
        mach_task_self(), &address, size, 0, VM_FLAGS_ANYWHERE,
        mach_task_self(), source_address, FALSE, &current, &maximum, VM_INHERIT_NONE);
    if (result != KERN_SUCCESS) return ProcessMemoryError::remap_failed;
    if ((guest_address && (address >= query_end || size > query_end - address)) ||
        (!guest_address && address < query_end)) {
        ::mach_vm_deallocate(mach_task_self(), address, size);
        return ProcessMemoryError::unsupported;
    }
    status = apply_protection(address, size, permission);
    if (!status.ok()) {
        ::mach_vm_deallocate(mach_task_self(), address, size);
        return status.error;
    }
    return ProcessMemoryRegion(address, size);
#else
    (void)guest_address;
    return ProcessMemoryError::unsupported;
#endif
}

ProcessMemoryRegion::ProcessMemoryRegion(ProcessMemoryRegion&& other) noexcept
    : address_(std::exchange(other.address_, 0)),
      size_(std::exchange(other.size_, 0)) {}

ProcessMemoryRegion& ProcessMemoryRegion::operator=(ProcessMemoryRegion&& other) noexcept {
    if (this != &other) {
        reset();
        address_ = std::exchange(other.address_, 0);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

ProcessMemoryRegion::~ProcessMemoryRegion() { reset(); }

void ProcessMemoryRegion::reset() noexcept {
#if defined(__APPLE__)
    if (address_ != 0) ::mach_vm_deallocate(mach_task_self(), address_, size_);
#endif
    address_ = 0;
    size_ = 0;
}

ProcessMemoryStatus ProcessMemoryRegion::commit(
    std::uint64_t offset, std::uint64_t size,
    memory::Permission permission) noexcept {
    auto status = validate_permission(permission);
    if (!status.ok()) return status;
    status = validate_subrange(size_, offset, size);
    if (!status.ok()) return status;
    if (!valid()) return {ProcessMemoryError::out_of_range};
#if defined(__APPLE__)
    return make_fresh_mapping(address_ + offset, size, permission);
#else
    return {ProcessMemoryError::unsupported};
#endif
}

ProcessMemoryStatus ProcessMemoryRegion::decommit(
    std::uint64_t offset, std::uint64_t size) noexcept {
    auto status = validate_subrange(size_, offset, size);
    if (!status.ok()) return status;
    if (!valid()) return {ProcessMemoryError::out_of_range};
#if defined(__APPLE__)
    return make_fresh_mapping(address_ + offset, size, memory::Permission::none);
#else
    return {ProcessMemoryError::unsupported};
#endif
}

ProcessMemoryStatus ProcessMemoryRegion::protect(
    std::uint64_t offset, std::uint64_t size,
    memory::Permission permission) noexcept {
    auto status = validate_permission(permission);
    if (!status.ok()) return status;
    status = validate_subrange(size_, offset, size);
    if (!status.ok()) return status;
    if (!valid()) return {ProcessMemoryError::out_of_range};
    return protect_process_memory(address_ + offset, size, permission);
}

ProcessMemoryStatus ProcessMemoryRegion::alias(
    std::uint64_t offset, std::uint64_t size,
    std::uint64_t source_address, memory::Permission permission) noexcept {
    auto status = validate_permission(permission);
    if (!status.ok()) return status;
    status = validate_subrange(size_, offset, size);
    if (!status.ok()) return status;
    status = validate_native_range(source_address, size);
    if (!status.ok()) return status;
    if (!valid()) return {ProcessMemoryError::out_of_range};
#if defined(__APPLE__)
    return make_shared_mapping(address_ + offset, size, source_address, permission);
#else
    return {ProcessMemoryError::unsupported};
#endif
}

const char* describe(ProcessMemoryError error) noexcept {
    switch (error) {
    case ProcessMemoryError::none: return "none";
    case ProcessMemoryError::invalid_size: return "size is zero";
    case ProcessMemoryError::unaligned_address: return "address is not native-page aligned";
    case ProcessMemoryError::unaligned_size: return "size is not a whole native page";
    case ProcessMemoryError::address_overflow: return "address range overflows";
    case ProcessMemoryError::out_of_range: return "address range is outside the reservation";
    case ProcessMemoryError::invalid_permission: return "permission mask is invalid";
    case ProcessMemoryError::write_execute_conflict: return "writable executable mapping rejected";
    case ProcessMemoryError::address_occupied: return "fixed address is occupied";
    case ProcessMemoryError::unsupported: return "operation is unsupported on this host";
    case ProcessMemoryError::allocation_failed: return "native allocation failed";
    case ProcessMemoryError::protection_failed: return "native protection change failed";
    case ProcessMemoryError::remap_failed: return "native remap failed";
    case ProcessMemoryError::system_failure: return "unexpected native virtual-memory failure";
    }
    return "unknown process-memory error";
}

} // namespace armigo::platform
