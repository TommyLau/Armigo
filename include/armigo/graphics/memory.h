// ============================================================================
// Armigo Graphics - Guest GPU Memory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::graphics {

using Reader = bool (*)(void*, std::uint64_t, std::span<std::byte>) noexcept;
using Writer = bool (*)(void*, std::uint64_t,
                        std::span<const std::byte>) noexcept;

struct MemoryCallbacks {
    Reader read = nullptr;
    Writer write = nullptr;
    void* context = nullptr;
};

// One immutable GPU-VA mapping. The guest address names the start of the
// backing allocation; buffer_offset selects the mapped portion of it.
struct Mapping {
    std::uint64_t gpu_base = 0;
    std::uint64_t mapped_size = 0;
    std::uint64_t buffer_offset = 0;
    std::uint64_t guest_address = 0;
    std::uint64_t backing_size = 0;
    std::uint64_t page_size = 0;
    // Optional allocation pin. Persistent renderers require it; CPU callbacks
    // still remain borrowed and may only run within the synchronous call.
    std::shared_ptr<void> owner{};
};

enum class MemoryErrorCode : std::uint8_t {
    None,
    ZeroMappedSize,
    InvalidPageSize,
    MisalignedGpuBase,
    MisalignedMappedSize,
    MisalignedBufferOffset,
    GpuRangeOverflow,
    BackingRangeExceeded,
    GuestRangeOverflow,
    OverlappingMappings,
    UnmappedAddress,
    AccessCrossesMapping,
    ReaderUnavailable,
    WriterUnavailable,
    ReadFailed,
    WriteFailed,
};

struct MemoryResult {
    MemoryErrorCode code = MemoryErrorCode::None;
    std::string_view detail{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == MemoryErrorCode::None;
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }
};

class GpuAddressSpace {
public:
    [[nodiscard]] MemoryResult read(std::uint64_t gpu_address,
                                    std::span<std::byte> destination) const noexcept;
    [[nodiscard]] MemoryResult write(
        std::uint64_t gpu_address,
        std::span<const std::byte> source) const noexcept;

    // Mapping values are copied during construction. The callback context and
    // its guest backing are borrowed and must outlive all calls on this initial,
    // synchronous view. A later asynchronous owner must pin them explicitly.
    [[nodiscard]] std::span<const Mapping> mappings() const noexcept {
        return mappings_;
    }

    // Borrowed raw CPU-backing capability for a renderer that independently
    // retains and bounds-checks the allocation. Never retain beyond this view's
    // synchronous callback scope or interpret GPU addresses as CPU addresses.
    [[nodiscard]] MemoryCallbacks backing_callbacks() const noexcept {
        return callbacks_;
    }

    // Unique per constructed view and shared by its copies. Views are
    // immutable, so an equal generation guarantees identical mappings.
    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_;
    }

private:
    friend std::variant<GpuAddressSpace, MemoryResult> make_address_space(
        std::span<const Mapping>, MemoryCallbacks, std::uint64_t);

    GpuAddressSpace(std::vector<Mapping> mappings, MemoryCallbacks callbacks,
                    std::uint64_t generation);

    [[nodiscard]] std::variant<std::uint64_t, MemoryResult> translate(
        std::uint64_t gpu_address, std::size_t size) const noexcept;

    std::vector<Mapping> mappings_;
    MemoryCallbacks callbacks_{};
    std::uint64_t generation_ = 0;
};

using AddressSpaceResult = std::variant<GpuAddressSpace, MemoryResult>;

// A fresh view generation. A caller that rebuilds views of mappings it has
// not changed may pass one generation to all of them, so consumers keyed by
// generation keep their derived state across views.
[[nodiscard]] std::uint64_t allocate_address_space_generation() noexcept;

// Copies and sorts mappings into an immutable synchronous view. Generation
// zero allocates a fresh one; a nonzero generation must belong to views of
// identical mappings.
[[nodiscard]] AddressSpaceResult make_address_space(
    std::span<const Mapping> mappings, MemoryCallbacks callbacks, std::uint64_t generation = 0);

}  // namespace armigo::graphics
