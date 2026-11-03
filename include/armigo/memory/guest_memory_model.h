// ============================================================================
// Armigo Memory - Guest Memory Management
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Guest virtual-memory metadata model.
//
// This component tracks which guest virtual address ranges are mapped and
// which permissions they carry. It is pure metadata: it never allocates,
// maps, or touches host memory, never dereferences guest pointers, and never
// executes guest code. The guest page size is a caller-chosen power of two
// and is completely independent of the host page size.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace armigo::memory {

/// Fixed-width guest virtual address.
using GuestAddress = std::uint64_t;

/// Guest access permissions. Values are composable bit masks.
///
/// `none` describes a mapped region with no access rights (for example a
/// guard reservation); it is distinct from "not mapped" in every query API.
enum class Permission : std::uint8_t {
    none = 0,
    read = 1 << 0,
    write = 1 << 1,
    execute = 1 << 2,
};

/// Composes two permission masks.
constexpr Permission operator|(Permission lhs, Permission rhs) noexcept {
    return static_cast<Permission>(static_cast<std::uint8_t>(lhs) |
                                   static_cast<std::uint8_t>(rhs));
}

/// Intersects two permission masks.
constexpr Permission operator&(Permission lhs, Permission rhs) noexcept {
    return static_cast<Permission>(static_cast<std::uint8_t>(lhs) &
                                   static_cast<std::uint8_t>(rhs));
}

/// In-place composition.
constexpr Permission& operator|=(Permission& lhs, Permission rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

/// Tests whether `value` contains every bit of `flag`.
constexpr bool has_permission(Permission value, Permission flag) noexcept {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flag)) ==
           static_cast<std::uint8_t>(flag);
}

/// Tests whether `value` contains only known permission bits. Note that this
/// accepts write+execute; callers that must forbid it use
/// the model's mutating operations instead, which classify the failure precisely.
constexpr bool has_known_permission_bits(Permission value) noexcept {
    const auto raw = static_cast<std::uint8_t>(value);
    constexpr std::uint8_t known = static_cast<std::uint8_t>(Permission::read) |
                                   static_cast<std::uint8_t>(Permission::write) |
                                   static_cast<std::uint8_t>(Permission::execute);
    return (raw & ~known) == 0;
}

/// Structured failure reported by model operations.
enum class MemoryError : std::uint8_t {
    success = 0,
    /// Page size is zero or not a power of two (model construction failure).
    invalid_page_size,
    /// Address space geometry is unusable: zero size, overflow, or not a
    /// whole number of guest pages. Also reported when an operation is
    /// attempted on a default-constructed (invalid) model.
    invalid_address_space,
    /// Permission mask contains bits outside `Permission`.
    invalid_permission,
    /// Permission mask sets both write and execute.
    write_execute_rejected,
    /// Requested range size is zero.
    zero_size,
    /// Start address is not guest-page aligned.
    unaligned_address,
    /// Size is not a whole number of guest pages.
    unaligned_size,
    /// address + size would exceed the 64-bit address range.
    address_overflow,
    /// Range lies partly or fully outside the configured address space.
    out_of_range,
    /// `map` would overlap an existing mapped region.
    overlapping_map,
    /// `unmap` or `protect` range is not completely covered by mappings.
    not_mapped,
};

/// English diagnostic text for a `MemoryError` value. Never returns null.
const char* to_string(MemoryError error) noexcept;

/// Result of a mutating model operation.
///
/// `detail` carries an operation-dependent value:
///   - `invalid_permission`, `write_execute_rejected`: the request address;
///   - `zero_size`, `unaligned_size`: the offending size;
///   - `unaligned_address`, `address_overflow`, `out_of_range`: the request
///     address;
///   - `overlapping_map`: the start of the conflicting existing region;
///   - `not_mapped`: the first uncovered guest address.
/// It is zero when not meaningful.
struct MemoryResult {
    MemoryError error = MemoryError::success;
    GuestAddress detail = 0;

    constexpr bool ok() const noexcept { return error == MemoryError::success; }
};

/// Snapshot view of one mapped region.
struct MappingInfo {
    GuestAddress start = 0;
    /// Region size in bytes; always a positive multiple of the guest page size.
    std::uint64_t size = 0;
    Permission permissions = Permission::none;

    /// Inclusive last guest address covered by the region.
    /// Use this instead of computing `start + size`, which can wrap when the
    /// region ends at the top of the 64-bit address range.
    constexpr GuestAddress last_address() const noexcept { return start + (size - 1); }
};

/// Bounded metadata model of a finite guest virtual address space.
///
/// The model keeps a sorted, non-overlapping set of mapped regions with
/// explicit permissions. All mutating operations validate their entire
/// request before changing any state, so a failed operation is guaranteed to
/// leave the model exactly as it was.
///
/// Canonical form: adjacent regions whose permissions are equal are always
/// merged, so two neighboring regions touch only when their permissions
/// differ.
class GuestMemoryModel {
public:
    /// Validates the geometry and constructs a model.
    ///
    /// `size` is in bytes. Requirements:
    ///   - `page_size` is a power of two (any power, including 1, and any
    ///     value independent of the host page size) and not zero;
    ///   - `size` is not zero and `start + size - 1` does not overflow;
    ///   - `start` and `size` are multiples of `page_size`.
    ///
    /// Returns `std::nullopt` when a requirement fails; nothing is allocated
    /// in that case.
    static std::optional<GuestMemoryModel> create(GuestAddress start,
                                                  std::uint64_t size,
                                                  std::uint64_t page_size);

    /// Constructs an invalid model. Every mutating operation fails with
    /// `invalid_address_space`; queries report nothing mapped.
    GuestMemoryModel() noexcept = default;

    GuestMemoryModel(const GuestMemoryModel&) = default;
    GuestMemoryModel& operator=(const GuestMemoryModel&) = default;
    GuestMemoryModel(GuestMemoryModel&&) noexcept = default;
    GuestMemoryModel& operator=(GuestMemoryModel&&) noexcept = default;

    /// Maps `[address, address + size)` with `permissions`.
    ///
    /// Requires: nonzero, page-aligned, page-multiple size within the address
    /// space, no overlap with any existing mapping, and a valid permission
    /// mask (known bits only, and not simultaneously writable and executable).
    /// Adjacent regions with equal permissions are merged on success.
    MemoryResult map(GuestAddress address, std::uint64_t size, Permission permissions);

    /// Removes all mappings in `[address, address + size)`.
    ///
    /// Requires the whole range to be mapped; edges of partially covered
    /// regions are split and kept. Transactional: a not-fully-mapped range is
    /// rejected without modifying anything.
    MemoryResult unmap(GuestAddress address, std::uint64_t size);

    /// Sets `permissions` on every mapping in `[address, address + size)`.
    ///
    /// Requires the whole range to be mapped and the mask to be valid.
    /// Regions are split at the range edges as needed and re-merged under the
    /// canonical form rule. Transactional on failure.
    MemoryResult protect(GuestAddress address, std::uint64_t size, Permission permissions);

    /// Looks up the region containing `address`.
    ///
    /// Returns `std::nullopt` only when the address is unmapped. A mapped
    /// region without access rights is reported as a present `MappingInfo`
    /// whose `permissions` equals `Permission::none`; the two states are
    /// always distinguishable through this API.
    std::optional<MappingInfo> query(GuestAddress address) const noexcept;

    /// Ascending snapshot of all mapped regions. Regions never overlap and
    /// never touch with equal permissions.
    std::vector<MappingInfo> mappings() const;

    /// Number of regions in the canonical mapping set.
    std::size_t mapping_count() const noexcept { return regions_.size(); }

    /// Whether the model was constructed with a valid geometry.
    bool valid() const noexcept { return page_size_ != 0; }

    /// Configured guest page size in bytes (a power of two), or 0 when the
    /// model is invalid.
    std::uint64_t page_size() const noexcept { return page_size_; }

    /// First guest address of the address space.
    GuestAddress address_space_start() const noexcept { return start_; }

    /// Address space extent in bytes.
    std::uint64_t address_space_size() const noexcept { return size_; }

    /// Inclusive last guest address of the address space.
    GuestAddress address_space_last() const noexcept { return last_; }

private:
    struct Region {
        GuestAddress start;
        GuestAddress last;  // inclusive; never wraps
        Permission permissions;
    };

    GuestMemoryModel(GuestAddress start, GuestAddress size, GuestAddress last,
                     std::uint64_t page_size) noexcept
        : start_(start), size_(size), last_(last), page_size_(page_size) {}

    /// Validates address/size shape common to map, unmap, and protect.
    /// Returns an error, or stores the inclusive range end in `range_last`.
    std::optional<MemoryError> validate_range(GuestAddress address, std::uint64_t size,
                                              GuestAddress& range_last) const noexcept;

    /// Validates a permission mask for map/protect.
    static std::optional<MemoryError> validate_permissions(Permission permissions) noexcept;

    /// Finds the region span fully covering `[address, range_last]`.
    /// Returns the first uncovered guest address, or stores the covering
    /// region index range in `first`/`last_index`.
    std::optional<GuestAddress> find_full_coverage(GuestAddress address,
                                                   GuestAddress range_last,
                                                   std::size_t& first,
                                                   std::size_t& last_index) const noexcept;

    /// Merges the region at `index` with equal-permission neighbors while
    /// possible, restoring the canonical form after a mutation.
    void coalesce_from(std::size_t index);

    std::vector<Region> regions_;
    GuestAddress start_ = 0;
    std::uint64_t size_ = 0;
    GuestAddress last_ = 0;  // inclusive; valid only when page_size_ != 0
    std::uint64_t page_size_ = 0;
};

}  // namespace armigo::memory
