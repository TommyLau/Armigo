// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace armigo::horizon {

// Private kernel metadata storage. Allocations keep their pool alive; freed
// objects return to their slab, whose backing pages remain owned by the pool.
class SystemResourcePool final : public std::enable_shared_from_this<SystemResourcePool> {
public:
    static constexpr std::size_t page_size = 4096;
    enum class Kind : std::uint8_t { memory_block, block_info, page_table };
    struct Layout {
        std::size_t total_bytes = 0;
        std::size_t reference_bytes = 0;
        std::size_t bitmap_bytes = 0;
        std::size_t usable_pages = 0;
        unsigned bitmap_levels = 0;
    };
    using Entropy = std::uint64_t (*)(void*) noexcept;

    class Allocation {
    public:
        Allocation(Allocation&& other) noexcept;
        Allocation& operator=(Allocation&& other) noexcept;
        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;
        ~Allocation();
        std::span<std::byte> bytes() const noexcept;
        void reset() noexcept;
    private:
        friend class SystemResourcePool;
        friend class MemoryIntervalMap;
        Allocation(std::shared_ptr<SystemResourcePool> pool, Kind kind, std::byte* data)
            : pool_(std::move(pool)), kind_(kind), data_(data) {}
        std::shared_ptr<SystemResourcePool> pool_;
        Kind kind_{};
        std::byte* data_ = nullptr;
    };

    static std::optional<Layout> layout(std::size_t bytes) noexcept;
    // Entropy and its context must remain valid while any allocation exists.
    // The callback runs under the pool lock and must not reenter the pool.
    static std::shared_ptr<SystemResourcePool> create(std::size_t bytes,
        Entropy entropy, void* context, std::string& error);
    ~SystemResourcePool();
    std::optional<Allocation> allocate(Kind kind);
    Layout geometry() const noexcept {
        auto result = layout_;
        result.bitmap_levels = static_cast<unsigned>(bitmap_.size());
        return result;
    }
    std::size_t used_bytes() const noexcept;
    std::size_t live_objects(Kind kind) const noexcept;

private:
    friend class MemoryIntervalMap;
    explicit SystemResourcePool(Layout layout, Entropy entropy, void* context);
    static std::size_t object_size(Kind kind) noexcept;
    std::byte* allocate_page();
    void release(Kind kind, std::byte* data) noexcept;
    unsigned choose_bit(std::uint64_t bits);
    void set_free(std::size_t index);
    void clear_free(std::size_t index);
    Layout layout_;
    Entropy entropy_;
    void* entropy_context_;
    std::uint64_t random_bits_ = 0;
    unsigned bits_left_ = 0;
    std::byte* arena_ = nullptr;
    // Leaf level first; each parent bit describes a nonempty child word.
    std::vector<std::span<std::uint64_t>> bitmap_;
    std::array<std::byte*, 3> heads_{};
    std::array<std::size_t, 3> live_{};
    std::size_t used_pages_ = 0;
    mutable std::mutex mutex_;
};

} // namespace armigo::horizon
