// ============================================================================
// Armigo Graphics - Guest GPU Memory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/memory.h>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <utility>

namespace armigo::graphics {
namespace {

constexpr MemoryResult success() noexcept {
    return {};
}

constexpr MemoryResult failure(MemoryErrorCode code,
                               std::string_view detail) noexcept {
    return {code, detail};
}

constexpr bool is_power_of_two(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

constexpr bool adds_overflow(std::uint64_t left, std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left;
}

MemoryResult validate_mapping(const Mapping& mapping) noexcept {
    if (mapping.mapped_size == 0) {
        return failure(MemoryErrorCode::ZeroMappedSize,
                       "GPU mapping size is zero");
    }
    if (!is_power_of_two(mapping.page_size)) {
        return failure(MemoryErrorCode::InvalidPageSize,
                       "GPU mapping page size is not a power of two");
    }
    if (mapping.gpu_base % mapping.page_size != 0) {
        return failure(MemoryErrorCode::MisalignedGpuBase,
                       "GPU mapping base is not page aligned");
    }
    if (mapping.mapped_size % mapping.page_size != 0) {
        return failure(MemoryErrorCode::MisalignedMappedSize,
                       "GPU mapping size is not page aligned");
    }
    if (mapping.buffer_offset % mapping.page_size != 0) {
        return failure(MemoryErrorCode::MisalignedBufferOffset,
                       "GPU mapping buffer offset is not page aligned");
    }
    if (adds_overflow(mapping.gpu_base, mapping.mapped_size)) {
        return failure(MemoryErrorCode::GpuRangeOverflow,
                       "GPU mapping range overflows the address space");
    }
    if (mapping.buffer_offset > mapping.backing_size) {
        return failure(MemoryErrorCode::BackingRangeExceeded,
                       "GPU mapping buffer offset exceeds its guest backing");
    }
    if (adds_overflow(mapping.guest_address, mapping.backing_size)) {
        return failure(MemoryErrorCode::GuestRangeOverflow,
                       "GPU mapping guest backing overflows the address space");
    }
    return success();
}

}  // namespace

std::uint64_t allocate_address_space_generation() noexcept {
    static std::atomic<std::uint64_t> next_generation{0};
    return next_generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

GpuAddressSpace::GpuAddressSpace(std::vector<Mapping> mappings,
                                 MemoryCallbacks callbacks, std::uint64_t generation)
    : mappings_(std::move(mappings)), callbacks_(callbacks),
      generation_(generation ? generation : allocate_address_space_generation()) {}

AddressSpaceResult make_address_space(std::span<const Mapping> mappings,
                                      MemoryCallbacks callbacks, std::uint64_t generation) {
    std::vector<Mapping> owned(mappings.begin(), mappings.end());
    for (const Mapping& mapping : owned) {
        if (const MemoryResult result = validate_mapping(mapping); !result) {
            return result;
        }
    }

    std::sort(owned.begin(), owned.end(), [](const Mapping& left,
                                              const Mapping& right) {
        return left.gpu_base < right.gpu_base;
    });
    for (std::size_t index = 1; index < owned.size(); ++index) {
        const Mapping& previous = owned[index - 1];
        if (owned[index].gpu_base < previous.gpu_base + previous.mapped_size) {
            return failure(MemoryErrorCode::OverlappingMappings,
                           "GPU address-space mappings overlap");
        }
    }

    return GpuAddressSpace(std::move(owned), callbacks, generation);
}

std::variant<std::uint64_t, MemoryResult> GpuAddressSpace::translate(
    std::uint64_t gpu_address, std::size_t size) const noexcept {
    if (size == 0) {
        return gpu_address;
    }
    const auto byte_count = static_cast<std::uint64_t>(size);
    if (adds_overflow(gpu_address, byte_count)) {
        return failure(MemoryErrorCode::GpuRangeOverflow,
                       "GPU access range overflows the address space");
    }

    const auto after = std::upper_bound(
        mappings_.begin(), mappings_.end(), gpu_address,
        [](std::uint64_t address, const Mapping& mapping) {
            return address < mapping.gpu_base;
        });
    if (after == mappings_.begin()) {
        return failure(MemoryErrorCode::UnmappedAddress,
                       "GPU access starts outside a mapping");
    }
    const Mapping& mapping = *std::prev(after);
    const auto mapping_offset = gpu_address - mapping.gpu_base;
    if (mapping_offset >= mapping.mapped_size) {
        return failure(MemoryErrorCode::UnmappedAddress,
                       "GPU access starts outside a mapping");
    }
    if (byte_count > mapping.mapped_size - mapping_offset) {
        return failure(MemoryErrorCode::AccessCrossesMapping,
                       "GPU access crosses a mapping boundary");
    }

    if (mapping_offset > mapping.backing_size - mapping.buffer_offset ||
        byte_count > mapping.backing_size - mapping.buffer_offset -
                         mapping_offset) {
        return failure(MemoryErrorCode::BackingRangeExceeded,
                       "GPU access exceeds the mapped guest backing");
    }

    return mapping.guest_address + mapping.buffer_offset + mapping_offset;
}

MemoryResult GpuAddressSpace::read(
    std::uint64_t gpu_address,
    std::span<std::byte> destination) const noexcept {
    if (destination.empty()) {
        return success();
    }
    const auto translated = translate(gpu_address, destination.size());
    if (const auto* error = std::get_if<MemoryResult>(&translated)) {
        return *error;
    }
    if (callbacks_.read == nullptr) {
        return failure(MemoryErrorCode::ReaderUnavailable,
                       "GPU memory reader is unavailable");
    }
    if (!callbacks_.read(callbacks_.context,
                         std::get<std::uint64_t>(translated), destination)) {
        return failure(MemoryErrorCode::ReadFailed,
                       "GPU memory read callback failed");
    }
    return success();
}

MemoryResult GpuAddressSpace::write(
    std::uint64_t gpu_address,
    std::span<const std::byte> source) const noexcept {
    if (source.empty()) {
        return success();
    }
    const auto translated = translate(gpu_address, source.size());
    if (const auto* error = std::get_if<MemoryResult>(&translated)) {
        return *error;
    }
    if (callbacks_.write == nullptr) {
        return failure(MemoryErrorCode::WriterUnavailable,
                       "GPU memory writer is unavailable");
    }
    if (!callbacks_.write(callbacks_.context,
                          std::get<std::uint64_t>(translated), source)) {
        return failure(MemoryErrorCode::WriteFailed,
                       "GPU memory write callback failed");
    }
    return success();
}

}  // namespace armigo::graphics
