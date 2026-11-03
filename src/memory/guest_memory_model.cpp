// ============================================================================
// Armigo Memory - Guest Memory Management
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Guest virtual-memory metadata model implementation.
//
// Regions are kept in a vector sorted by start address. Because regions never
// overlap, start and last addresses are both strictly increasing, which lets
// every lookup use binary search. All address arithmetic is performed on
// inclusive bounds with explicit overflow checks; exclusive ends such as
// `start + size` are never computed because they can wrap at the top of the
// 64-bit range.

#include "armigo/memory/guest_memory_model.h"

#include <algorithm>
#include <array>
#include <limits>

namespace armigo::memory {

namespace {

constexpr GuestAddress kMaxAddress = std::numeric_limits<GuestAddress>::max();

}  // namespace

const char* to_string(MemoryError error) noexcept {
    switch (error) {
        case MemoryError::success:
            return "success";
        case MemoryError::invalid_page_size:
            return "invalid page size (must be a nonzero power of two)";
        case MemoryError::invalid_address_space:
            return "invalid address space geometry";
        case MemoryError::invalid_permission:
            return "permission mask contains unknown bits";
        case MemoryError::write_execute_rejected:
            return "permission mask may not combine write and execute";
        case MemoryError::zero_size:
            return "range size is zero";
        case MemoryError::unaligned_address:
            return "start address is not guest-page aligned";
        case MemoryError::unaligned_size:
            return "range size is not a whole number of guest pages";
        case MemoryError::address_overflow:
            return "address range overflows the 64-bit address space";
        case MemoryError::out_of_range:
            return "address range lies outside the configured address space";
        case MemoryError::overlapping_map:
            return "address range overlaps an existing mapping";
        case MemoryError::not_mapped:
            return "address range is not completely mapped";
    }
    return "unknown memory error";
}

std::optional<GuestMemoryModel> GuestMemoryModel::create(GuestAddress start,
                                                         std::uint64_t size,
                                                         std::uint64_t page_size) {
    if (page_size == 0 || (page_size & (page_size - 1)) != 0) {
        return std::nullopt;
    }
    if (size == 0) {
        return std::nullopt;
    }
    if (start % page_size != 0 || size % page_size != 0) {
        return std::nullopt;
    }
    // start + size - 1 must be representable.
    if (size - 1 > kMaxAddress - start) {
        return std::nullopt;
    }
    return GuestMemoryModel(start, size, start + (size - 1), page_size);
}

std::optional<MemoryError> GuestMemoryModel::validate_permissions(
    Permission permissions) noexcept {
    if (!has_known_permission_bits(permissions)) {
        return MemoryError::invalid_permission;
    }
    if (has_permission(permissions, Permission::write) &&
        has_permission(permissions, Permission::execute)) {
        return MemoryError::write_execute_rejected;
    }
    return std::nullopt;
}

std::optional<MemoryError> GuestMemoryModel::validate_range(GuestAddress address,
                                                            std::uint64_t size,
                                                            GuestAddress& range_last) const noexcept {
    if (size == 0) {
        return MemoryError::zero_size;
    }
    if (address % page_size_ != 0) {
        return MemoryError::unaligned_address;
    }
    if (size % page_size_ != 0) {
        return MemoryError::unaligned_size;
    }
    // The inclusive end must be representable; an exclusive end above the
    // highest guest address is allowed without computing that value.
    if (size - 1 > kMaxAddress - address) {
        return MemoryError::address_overflow;
    }
    range_last = address + (size - 1);
    if (address < start_ || range_last > last_) {
        return MemoryError::out_of_range;
    }
    return std::nullopt;
}

MemoryResult GuestMemoryModel::map(GuestAddress address, std::uint64_t size,
                                   Permission permissions) {
    if (!valid()) {
        return {MemoryError::invalid_address_space, address};
    }
    if (auto permission_error = validate_permissions(permissions)) {
        return {*permission_error, address};
    }
    GuestAddress range_last = 0;
    if (auto range_error = validate_range(address, size, range_last)) {
        return {*range_error,
                *range_error == MemoryError::zero_size || *range_error == MemoryError::unaligned_size
                    ? size
                    : address};
    }

    // First region whose start is >= address.
    auto it = std::lower_bound(
        regions_.begin(), regions_.end(), address,
        [](const Region& region, GuestAddress value) { return region.start < value; });

    // A following region overlapping [address, range_last]?
    if (it != regions_.end() && it->start <= range_last) {
        return {MemoryError::overlapping_map, it->start};
    }
    // A preceding region reaching at or past address?
    if (it != regions_.begin()) {
        const Region& previous = *std::prev(it);
        if (previous.last >= address) {
            return {MemoryError::overlapping_map, previous.start};
        }
    }

    const std::size_t index = static_cast<std::size_t>(it - regions_.begin());
    regions_.insert(regions_.begin() + static_cast<std::ptrdiff_t>(index),
                    Region{address, range_last, permissions});
    coalesce_from(index);
    return {};
}

MemoryResult GuestMemoryModel::unmap(GuestAddress address, std::uint64_t size) {
    if (!valid()) {
        return {MemoryError::invalid_address_space, address};
    }
    GuestAddress range_last = 0;
    if (auto range_error = validate_range(address, size, range_last)) {
        return {*range_error,
                *range_error == MemoryError::zero_size || *range_error == MemoryError::unaligned_size
                    ? size
                    : address};
    }
    std::size_t first = 0;
    std::size_t last_index = 0;
    if (auto hole = find_full_coverage(address, range_last, first, last_index)) {
        return {MemoryError::not_mapped, *hole};
    }

    // Split fragments kept outside the unmapped range. Computed before any
    // mutation so the operation stays transactional; every fragment address
    // is derived from already-validated bounds and cannot wrap.
    std::optional<Region> head;
    std::optional<Region> tail;
    if (regions_[first].start < address) {
        head = Region{regions_[first].start, address - 1, regions_[first].permissions};
    }
    if (regions_[last_index].last > range_last) {
        tail = Region{range_last + 1, regions_[last_index].last, regions_[last_index].permissions};
    }

    const std::size_t fragments = static_cast<std::size_t>(head.has_value()) +
                                  static_cast<std::size_t>(tail.has_value());
    // Reserve before erasing. Region moves cannot throw, so an allocation
    // failure leaves the original mapping set intact.
    regions_.reserve(regions_.size() - (last_index - first + 1) + fragments);
    regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(first),
                   regions_.begin() + static_cast<std::ptrdiff_t>(last_index + 1));
    std::size_t insert_at = first;
    if (head) {
        regions_.insert(regions_.begin() + static_cast<std::ptrdiff_t>(insert_at), *head);
        ++insert_at;
    }
    if (tail) {
        regions_.insert(regions_.begin() + static_cast<std::ptrdiff_t>(insert_at), *tail);
        ++insert_at;
    }
    // Unmapping carves a hole and cannot create mergeable neighbors; the
    // coalescing pass is defensive and must remain a no-op here.
    if (insert_at > first) {
        coalesce_from(first);
    }
    return {};
}

MemoryResult GuestMemoryModel::protect(GuestAddress address, std::uint64_t size,
                                       Permission permissions) {
    if (!valid()) {
        return {MemoryError::invalid_address_space, address};
    }
    if (auto permission_error = validate_permissions(permissions)) {
        return {*permission_error, address};
    }
    GuestAddress range_last = 0;
    if (auto range_error = validate_range(address, size, range_last)) {
        return {*range_error,
                *range_error == MemoryError::zero_size || *range_error == MemoryError::unaligned_size
                    ? size
                    : address};
    }
    std::size_t first = 0;
    std::size_t last_index = 0;
    if (auto hole = find_full_coverage(address, range_last, first, last_index)) {
        return {MemoryError::not_mapped, *hole};
    }

    // Full coverage was established above, so the changed range is one
    // contiguous span regardless of its previous permission boundaries.
    std::array<Region, 3> replacement{};
    std::size_t replacement_count = 0;
    const bool has_head = regions_[first].start < address;
    if (has_head) {
        replacement[replacement_count++] =
            Region{regions_[first].start, address - 1, regions_[first].permissions};
    }
    replacement[replacement_count++] = Region{address, range_last, permissions};
    if (regions_[last_index].last > range_last) {
        replacement[replacement_count++] =
            Region{range_last + 1, regions_[last_index].last, regions_[last_index].permissions};
    }

    // Allocate all necessary capacity before the first mutation.
    regions_.reserve(regions_.size() - (last_index - first + 1) + replacement_count);
    regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(first),
                   regions_.begin() + static_cast<std::ptrdiff_t>(last_index + 1));
    regions_.insert(regions_.begin() + static_cast<std::ptrdiff_t>(first),
                    replacement.begin(),
                    replacement.begin() + static_cast<std::ptrdiff_t>(replacement_count));
    coalesce_from(first + static_cast<std::size_t>(has_head));
    return {};
}

std::optional<MappingInfo> GuestMemoryModel::query(GuestAddress address) const noexcept {
    if (!valid()) {
        return std::nullopt;
    }
    // First region whose start is greater than address.
    auto it = std::upper_bound(
        regions_.begin(), regions_.end(), address,
        [](GuestAddress value, const Region& region) { return value < region.start; });
    if (it == regions_.begin()) {
        return std::nullopt;
    }
    --it;
    if (address > it->last) {
        return std::nullopt;
    }
    return MappingInfo{it->start, it->last - it->start + 1, it->permissions};
}

std::vector<MappingInfo> GuestMemoryModel::mappings() const {
    std::vector<MappingInfo> snapshot;
    snapshot.reserve(regions_.size());
    for (const Region& region : regions_) {
        snapshot.push_back(
            MappingInfo{region.start, region.last - region.start + 1, region.permissions});
    }
    return snapshot;
}

std::optional<GuestAddress> GuestMemoryModel::find_full_coverage(
    GuestAddress address, GuestAddress range_last, std::size_t& first,
    std::size_t& last_index) const noexcept {
    // First region whose last address is >= address (regions are sorted and
    // disjoint, so last addresses are strictly increasing too).
    auto it = std::lower_bound(
        regions_.begin(), regions_.end(), address,
        [](const Region& region, GuestAddress value) { return region.last < value; });
    if (it == regions_.end() || it->start > address) {
        return address;
    }
    std::size_t i = static_cast<std::size_t>(it - regions_.begin());
    // Walk forward until a region reaches range_last, requiring contiguous
    // coverage. regions_[i].last < range_last <= kMaxAddress, so last + 1
    // cannot wrap inside the loop.
    while (regions_[i].last < range_last) {
        if (i + 1 >= regions_.size() || regions_[i + 1].start != regions_[i].last + 1) {
            return regions_[i].last + 1;
        }
        ++i;
    }
    first = static_cast<std::size_t>(it - regions_.begin());
    last_index = i;
    return std::nullopt;
}

void GuestMemoryModel::coalesce_from(std::size_t index) {
    // Merge left while the region and its predecessor touch with equal
    // permissions.
    while (index > 0) {
        Region& previous = regions_[index - 1];
        Region& current = regions_[index];
        // previous.last < current.start always holds for sorted disjoint
        // regions, so previous.last + 1 cannot wrap.
        if (previous.permissions != current.permissions || previous.last + 1 != current.start) {
            break;
        }
        previous.last = current.last;
        regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(index));
        --index;
    }
    // Merge right symmetrically.
    while (index + 1 < regions_.size()) {
        Region& current = regions_[index];
        Region& next = regions_[index + 1];
        if (current.permissions != next.permissions || current.last + 1 != next.start) {
            break;
        }
        current.last = next.last;
        regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(index + 1));
    }
}

}  // namespace armigo::memory
