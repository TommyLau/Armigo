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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace armigo::platform {

struct ModuleRegion {
    std::size_t offset = 0;
    std::size_t size = 0;
    memory::Permission permissions = memory::Permission::none;
};

struct HostPageRegion {
    std::size_t offset = 0;
    std::size_t size = 0;
    memory::Permission permissions = memory::Permission::none;
};

// Already mapped, native-page-aligned image pieces. Executable pieces retain
// their backing object through a shared remap; all other pieces are copied on
// write. Source spans must remain mapped for the synchronous creation call.
struct EmbeddedModulePiece {
    std::size_t offset = 0;
    std::span<const std::byte> bytes;
    bool executable = false;
};

enum class HostMapError {
    none,
    invalid_image,
    invalid_region,
    invalid_permissions,
    write_execute_conflict,
    resource_limit,
    allocation_failed,
    system_failure,
    address_unavailable,
    invalid_state,
};

struct HostMapStatus {
    HostMapError error = HostMapError::none;
    int system_error = 0;
    bool ok() const noexcept { return error == HostMapError::none; }
};

struct HostMapLimits {
    std::size_t max_image_bytes = 1024ull * 1024 * 1024;
    std::size_t max_regions = 4096;
    std::uintptr_t minimum_address = 0;
    // False seals no host page executable, so a native page may hold a
    // region's code and another region's writable data together. The regions
    // keep their declared permissions (the guest's view); executable_address
    // then reports nothing.
    bool host_executes = true;
};

struct HostModuleResult;

// Owns one module mapping. Ordinary mappings start writable and nonexecutable;
// embedded mappings retain RX code backing and private writable data. Sealing
// applies the computed final host-page permissions.
// This object is not thread-safe. No thread may execute or modify the mapping
// while it is being prepared, moved, sealed, or destroyed.
class HostModule {
public:
    // Native page granularity used for mapping and sealing; empty on failure.
    static std::optional<std::size_t> native_page_size() noexcept;
    static HostModuleResult create(std::span<const std::byte> image,
                                   std::span<const ModuleRegion> regions,
                                   const HostMapLimits& limits = {});

    // Reassembles signed executable pieces without copying or writing their
    // code pages. Pieces cover the rounded image exactly, in offset order.
    // Non-code pages are private RW while linking, then seal() applies regions.
    static HostModuleResult create_embedded(std::size_t image_size,
        std::span<const ModuleRegion> regions, std::span<const EmbeddedModulePiece> pieces,
        const HostMapLimits& limits = {});

    HostModule(const HostModule&) = delete;
    HostModule& operator=(const HostModule&) = delete;
    HostModule(HostModule&& other) noexcept;
    HostModule& operator=(HostModule&& other) noexcept;
    ~HostModule();

    std::uintptr_t base_address() const noexcept;
    std::size_t image_size() const noexcept { return image_size_; }
    std::size_t mapping_size() const noexcept { return mapping_size_; }
    std::size_t host_page_size() const noexcept { return host_page_size_; }
    // False when no host page of this module is executable (HostMapLimits).
    bool host_executes() const noexcept { return host_executes_; }
    bool valid() const noexcept { return base_ != nullptr; }
    bool sealed() const noexcept { return sealed_; }

    // Borrowed writable bytes are available only before sealing. Previously
    // returned spans must be discarded before seal, move, or destruction.
    // Embedded mappings already protect executable pages RX: callers may only
    // write their non-code pages, including when applying data relocations.
    std::span<std::byte> preparation_bytes() noexcept;

    // Reports the final host-page plan, which can be more permissive than an
    // individual guest region when several guest regions share one host page.
    // A host page requiring both write and execute is rejected at creation.
    std::span<const HostPageRegion> host_regions() const noexcept { return host_regions_; }
    // Original declared permissions, before native-page unions. Runtime admission
    // must reject subpage shapes the native mapping cannot faithfully enforce.
    std::span<const ModuleRegion> guest_regions() const noexcept { return guest_regions_; }

    // Applies the final protection plan and invalidates instruction caches.
    // Failure destroys the mapping, so no partially sealed image remains usable.
    // Success is idempotent. No arbitrary guest code is invoked by this method.
    HostMapStatus seal() noexcept;

    // Returned views are limited by the declared guest-region permissions,
    // even when host-page rounding makes adjacent storage readable/executable.
    std::optional<std::span<const std::byte>> readable_bytes(
        std::size_t offset, std::size_t size) const noexcept;
    std::optional<std::uintptr_t> executable_address(std::size_t offset) const noexcept;

private:
    HostModule() = default;
    static HostModuleResult create_layout(std::size_t image_size,
        std::span<const ModuleRegion> regions, const HostMapLimits& limits);
    void reset() noexcept;
    bool permits(std::size_t offset, std::size_t size, memory::Permission permission) const noexcept;

    std::byte* base_ = nullptr;
    std::size_t image_size_ = 0;
    std::size_t mapping_size_ = 0;
    std::size_t host_page_size_ = 0;
    bool host_executes_ = true;
    bool sealed_ = false;
    bool embedded_ = false;
    std::vector<ModuleRegion> guest_regions_;
    std::vector<HostPageRegion> host_regions_;
};

struct HostModuleResult {
    HostMapStatus status;
    std::optional<HostModule> module;
    bool ok() const noexcept { return status.ok() && module.has_value(); }
};

} // namespace armigo::platform
