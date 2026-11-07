// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/memory/guest_memory_model.h>

#include <cstdint>
#include <string>
#include <optional>
#include <variant>

namespace armigo::platform {

enum class ProcessMemoryError : std::uint8_t {
    none,
    invalid_size,
    unaligned_address,
    unaligned_size,
    address_overflow,
    out_of_range,
    invalid_permission,
    write_execute_conflict,
    address_occupied,
    unsupported,
    allocation_failed,
    protection_failed,
    remap_failed,
    system_failure,
};

struct ProcessMemoryStatus {
    ProcessMemoryError error = ProcessMemoryError::none;
    int native_code = 0;

    constexpr bool ok() const noexcept { return error == ProcessMemoryError::none; }
};

// Returns the native virtual-memory page size, or zero when it is unavailable.
std::uint64_t native_page_size() noexcept;

// Current native protection at an owned mapped address, for exact rollback.
// Optionally returns the remaining bytes in this native protection run.
std::optional<memory::Permission> process_memory_permissions(
    std::uint64_t address, std::uint64_t* contiguous_bytes = nullptr) noexcept;

// Changes protections on an existing native mapping without taking ownership.
// The complete range must consist of native pages and already be mapped.
ProcessMemoryStatus protect_process_memory(
    std::uint64_t address, std::uint64_t size, memory::Permission permission) noexcept;

class ProcessMemoryRegion;

// A summary of the host address space below the 39-bit guest limit: how many
// regions are mapped, the largest free gaps and the large mappings with their
// memory tags. For diagnostics when a guest reservation does not fit.
std::string describe_guest_address_space();
using ProcessMemoryResult = std::variant<ProcessMemoryError, ProcessMemoryRegion>;

// Owns one bounded native virtual-address reservation. The mapping is initially
// inaccessible. Every mutation operates at native-page granularity and is not
// thread-safe; callers serialize mutation with all access to the affected pages.
class ProcessMemoryRegion {
public:
    // Reserves exactly `bytes`, without rounding. A zero requested address asks
    // the kernel to choose an address below the supported 39-bit query end.
    // Fixed reservations never replace an existing mapping.
    static ProcessMemoryResult reserve(std::uint64_t bytes,
                                       std::uint64_t requested_address = 0) noexcept;

    // Searches actual native gaps below the 39-bit query end. Alignment must
    // be a power of two at least as large as a native page. Never overwrites
    // a mapping; concurrent occupation is retried with a bounded scan.
    static ProcessMemoryResult reserve_low_aligned(std::uint64_t bytes,
                                                   std::uint64_t alignment) noexcept;

    // Creates a separately owned, shared alias of an existing native mapping.
    // It does not copy or freeze bytes. The source owner must retain a valid
    // mapping for the duration of this call.
    static ProcessMemoryResult alias_copy(
        std::uint64_t source_address, std::uint64_t size,
        memory::Permission permission) noexcept;

    // Creates a host-private writable view of the same physical pages. Unlike
    // guest aliases, its address is above the guest query limit so it cannot
    // occupy an address that guest memory metadata reports as available.
    // This owner must be destroyed before the source backing is replaced.
    static ProcessMemoryResult writable_host_alias(
        std::uint64_t source_address, std::uint64_t size) noexcept;

    ProcessMemoryRegion(const ProcessMemoryRegion&) = delete;
    ProcessMemoryRegion& operator=(const ProcessMemoryRegion&) = delete;
    ProcessMemoryRegion(ProcessMemoryRegion&& other) noexcept;
    ProcessMemoryRegion& operator=(ProcessMemoryRegion&& other) noexcept;
    ~ProcessMemoryRegion();

    std::uint64_t address() const noexcept { return address_; }
    std::uint64_t size() const noexcept { return size_; }
    bool valid() const noexcept { return address_ != 0 && size_ != 0; }

    // Replaces the range with fresh zero-filled anonymous backing and applies
    // the requested protection only after the replacement is ready.
    ProcessMemoryStatus commit(std::uint64_t offset, std::uint64_t size,
                               memory::Permission permission) noexcept;

    // Replaces the range with fresh zero-filled inaccessible backing.
    ProcessMemoryStatus decommit(std::uint64_t offset, std::uint64_t size) noexcept;

    ProcessMemoryStatus protect(std::uint64_t offset, std::uint64_t size,
                                memory::Permission permission) noexcept;

    // Replaces the range with a shared alias. The source owner remains external
    // and must keep its mapping stable for the duration of this call.
    ProcessMemoryStatus alias(std::uint64_t offset, std::uint64_t size,
                              std::uint64_t source_address,
                              memory::Permission permission) noexcept;

private:
    static ProcessMemoryResult shared_alias(std::uint64_t source_address,
        std::uint64_t size, memory::Permission permission, bool guest_address) noexcept;
    ProcessMemoryRegion(std::uint64_t address, std::uint64_t size) noexcept
        : address_(address), size_(size) {}
    void reset() noexcept;

    std::uint64_t address_ = 0;
    std::uint64_t size_ = 0;
};

const char* describe(ProcessMemoryError error) noexcept;

} // namespace armigo::platform
