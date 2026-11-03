// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/host_module.h>

#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include "mach_vm.h"
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace armigo::platform {
namespace {
using memory::Permission;

bool writable_executable(Permission permission) {
    return memory::has_permission(permission, Permission::write | Permission::execute);
}

int native_protection(Permission permission) {
    int protection = PROT_NONE;
    if (memory::has_permission(permission, Permission::read)) protection |= PROT_READ;
    if (memory::has_permission(permission, Permission::write)) protection |= PROT_WRITE;
    if (memory::has_permission(permission, Permission::execute)) protection |= PROT_EXEC;
    return protection;
}

HostModuleResult failure(HostMapError error, int system_error = 0) {
    return {{error, system_error}, std::nullopt};
}
} // namespace

std::optional<std::size_t> HostModule::native_page_size() noexcept {
    const long page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0 || (page & (page - 1)) != 0) return {};
    return static_cast<std::size_t>(page);
}

HostModuleResult HostModule::create(std::span<const std::byte> image,
                                    std::span<const ModuleRegion> regions,
                                    const HostMapLimits& limits) {
    auto result = create_layout(image.size(), regions, limits);
    if (result.ok()) std::memcpy(result.module->base_, image.data(), image.size());
    return result;
}

HostModuleResult HostModule::create_layout(std::size_t image_size,
                                    std::span<const ModuleRegion> regions,
                                    const HostMapLimits& limits) {
    if (image_size == 0 || regions.empty()) {
        return failure(HostMapError::invalid_image);
    }
    if (image_size > limits.max_image_bytes || regions.size() > limits.max_regions) {
        return failure(HostMapError::resource_limit);
    }
    const auto page_size = native_page_size();
    if (!page_size) {
        return failure(HostMapError::system_failure, errno);
    }
    const auto page = *page_size;
    if (image_size > std::numeric_limits<std::size_t>::max() - (page - 1)) {
        return failure(HostMapError::resource_limit);
    }

    try {
        HostModule module;
        module.image_size_ = image_size;
        module.host_page_size_ = page;
        module.host_executes_ = limits.host_executes;
        module.mapping_size_ = (image_size + page - 1) & ~(page - 1);
        module.guest_regions_.assign(regions.begin(), regions.end());
        std::sort(module.guest_regions_.begin(), module.guest_regions_.end(),
                  [](const auto& left, const auto& right) { return left.offset < right.offset; });
        std::vector<Permission> pages(module.mapping_size_ / page, Permission::none);
        std::size_t previous_end = 0;
        for (const auto& region : module.guest_regions_) {
            if (region.size == 0 || region.offset < previous_end ||
                region.offset > image_size || region.size > image_size - region.offset) {
                return failure(HostMapError::invalid_region);
            }
            if (!memory::has_known_permission_bits(region.permissions)) {
                return failure(HostMapError::invalid_permissions);
            }
            if (writable_executable(region.permissions)) {
                return failure(HostMapError::write_execute_conflict);
            }
            previous_end = region.offset + region.size;
            const auto last_page = (previous_end - 1) / page;
            const auto host_permissions = limits.host_executes ? region.permissions
                : region.permissions & (Permission::read | Permission::write);
            for (std::size_t index = region.offset / page; index <= last_page; ++index) {
                pages[index] |= host_permissions;
                if (writable_executable(pages[index])) {
                    return failure(HostMapError::write_execute_conflict);
                }
            }
        }
        for (std::size_t first = 0; first < pages.size();) {
            std::size_t last = first + 1;
            while (last < pages.size() && pages[last] == pages[first]) ++last;
            module.host_regions_.push_back({first * page, (last - first) * page, pages[first]});
            first = last;
        }
        void* address = ::mmap(reinterpret_cast<void*>(limits.minimum_address),
                               module.mapping_size_, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANON, -1, 0);
        if (address == MAP_FAILED) {
            return failure(HostMapError::system_failure, errno);
        }
        const auto base = reinterpret_cast<std::uintptr_t>(address);
        if (base < limits.minimum_address ||
            module.mapping_size_ > std::numeric_limits<std::uintptr_t>::max() - base) {
            ::munmap(address, module.mapping_size_);
            return failure(HostMapError::address_unavailable);
        }
        module.base_ = static_cast<std::byte*>(address);
        return {{}, std::move(module)};
    } catch (const std::bad_alloc&) {
        return failure(HostMapError::allocation_failed);
    }
}

HostModuleResult HostModule::create_embedded(std::size_t image_size,
    std::span<const ModuleRegion> regions, std::span<const EmbeddedModulePiece> pieces,
    const HostMapLimits& limits) {
    auto result = create_layout(image_size, regions, limits);
    if (!result.ok()) return result;
    auto& module = *result.module;
    const auto page = module.host_page_size_;
    std::size_t cursor = 0;
    for (const auto& piece : pieces) {
        if (piece.offset != cursor || piece.bytes.empty() || piece.bytes.size() % page ||
            reinterpret_cast<std::uintptr_t>(piece.bytes.data()) % page ||
            piece.bytes.size() > module.mapping_size_ - cursor)
            return failure(HostMapError::invalid_region);
        const auto end = cursor + piece.bytes.size();
        for (const auto& region : module.host_regions_) {
            if (region.offset >= end || region.offset + region.size <= cursor) continue;
            if (memory::has_permission(region.permissions, Permission::execute) != piece.executable)
                return failure(HostMapError::invalid_permissions);
        }
        mach_vm_address_t destination = module.base_address() + cursor;
        vm_prot_t current = VM_PROT_NONE, maximum = VM_PROT_NONE;
        const auto status = ::mach_vm_remap(mach_task_self(), &destination, piece.bytes.size(), 0,
            VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(),
            reinterpret_cast<mach_vm_address_t>(piece.bytes.data()), !piece.executable,
            &current, &maximum, VM_INHERIT_COPY);
        if (status != KERN_SUCCESS) return failure(HostMapError::system_failure, status);
        const auto desired = piece.executable ? VM_PROT_READ | VM_PROT_EXECUTE : VM_PROT_READ | VM_PROT_WRITE;
        if ((maximum & desired) != desired ||
            (piece.executable && ((current & desired) != desired || (current & VM_PROT_WRITE))))
            return failure(HostMapError::invalid_permissions);
        // Restrict the maximum protection too: a remapped executable page can
        // never become writable through this mapping during subsequent linking.
        const auto protect = ::mach_vm_protect(mach_task_self(), destination, piece.bytes.size(),
                                               piece.executable, desired);
        if (protect != KERN_SUCCESS) return failure(HostMapError::system_failure, protect);
        cursor = end;
    }
    if (cursor != module.mapping_size_) return failure(HostMapError::invalid_region);
    module.embedded_ = true;
    return result;
}

HostModule::HostModule(HostModule&& other) noexcept
    : base_(std::exchange(other.base_, nullptr)),
      image_size_(std::exchange(other.image_size_, 0)),
      mapping_size_(std::exchange(other.mapping_size_, 0)),
      host_page_size_(std::exchange(other.host_page_size_, 0)),
      host_executes_(std::exchange(other.host_executes_, true)),
      sealed_(std::exchange(other.sealed_, false)),
      embedded_(std::exchange(other.embedded_, false)),
      guest_regions_(std::move(other.guest_regions_)),
      host_regions_(std::move(other.host_regions_)) {}

HostModule& HostModule::operator=(HostModule&& other) noexcept {
    if (this != &other) {
        reset();
        base_ = std::exchange(other.base_, nullptr);
        image_size_ = std::exchange(other.image_size_, 0);
        mapping_size_ = std::exchange(other.mapping_size_, 0);
        host_page_size_ = std::exchange(other.host_page_size_, 0);
        host_executes_ = std::exchange(other.host_executes_, true);
        sealed_ = std::exchange(other.sealed_, false);
        embedded_ = std::exchange(other.embedded_, false);
        guest_regions_ = std::move(other.guest_regions_);
        host_regions_ = std::move(other.host_regions_);
    }
    return *this;
}

HostModule::~HostModule() {
    reset();
}

void HostModule::reset() noexcept {
    if (base_ != nullptr) {
        ::munmap(base_, mapping_size_);
    }
    base_ = nullptr;
    image_size_ = 0;
    mapping_size_ = 0;
    host_page_size_ = 0;
    sealed_ = false;
    embedded_ = false;
    guest_regions_.clear();
    host_regions_.clear();
}

std::uintptr_t HostModule::base_address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(base_);
}

std::span<std::byte> HostModule::preparation_bytes() noexcept {
    return valid() && !sealed_ ? std::span<std::byte>{base_, image_size_}
                              : std::span<std::byte>{};
}

HostMapStatus HostModule::seal() noexcept {
    if (!valid()) return {HostMapError::invalid_state};
    if (sealed_) return {};
    for (const auto& region : host_regions_) {
        if (embedded_ && memory::has_permission(region.permissions, Permission::execute)) continue;
        if (::mprotect(base_ + region.offset, region.size,
                       native_protection(region.permissions)) != 0) {
            const int error = errno;
            reset();
            return {HostMapError::system_failure, error};
        }
    }
    for (const auto& region : host_regions_) {
        if (!embedded_ && memory::has_permission(region.permissions, Permission::execute)) {
            ::sys_icache_invalidate(base_ + region.offset, region.size);
        }
    }
    sealed_ = true;
    return {};
}

bool HostModule::permits(std::size_t offset, std::size_t size,
                         Permission permission) const noexcept {
    if (!valid() || offset > image_size_ || size > image_size_ - offset) return false;
    if (size == 0) return true;
    const auto end = offset + size;
    std::size_t cursor = offset;
    for (const auto& region : guest_regions_) {
        const auto region_end = region.offset + region.size;
        if (region_end <= cursor) continue;
        if (region.offset > cursor || !memory::has_permission(region.permissions, permission)) {
            return false;
        }
        cursor = std::min(end, region_end);
        if (cursor == end) return true;
    }
    return false;
}

std::optional<std::span<const std::byte>> HostModule::readable_bytes(
    std::size_t offset, std::size_t size) const noexcept {
    if (!permits(offset, size, Permission::read)) return std::nullopt;
    return std::span<const std::byte>{base_ + offset, size};
}

std::optional<std::uintptr_t> HostModule::executable_address(std::size_t offset) const noexcept {
    if (!sealed_ || !host_executes_ || offset % 4 != 0 || !permits(offset, 4, Permission::execute)) {
        return std::nullopt;
    }
    return base_address() + offset;
}

} // namespace armigo::platform
