// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/system_resource_pool.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace armigo::horizon {
namespace {
constexpr std::size_t round_page(std::size_t bytes) {
    return (bytes + SystemResourcePool::page_size - 1) & ~(SystemResourcePool::page_size - 1);
}
unsigned bitmap_depth(std::size_t pages) {
    unsigned depth = 0;
    do { pages /= 64; ++depth; } while (pages);
    return depth;
}
std::size_t bitmap_words(std::size_t pages) {
    const auto depth = bitmap_depth(pages);
    std::size_t result = 0;
    for (unsigned level = 0; level < depth; ++level) {
        pages = (pages + 63) / 64;
        result += pages;
    }
    return result;
}
std::size_t slot(SystemResourcePool::Kind kind) { return static_cast<std::size_t>(kind); }
} // namespace

std::optional<SystemResourcePool::Layout> SystemResourcePool::layout(std::size_t bytes) noexcept {
    if (!bytes || bytes % page_size) return {};
    const auto references = round_page((bytes / page_size) * sizeof(std::uint16_t));
    if (references >= bytes) return {};
    const auto remaining = bytes - references;
    const auto words = bitmap_words(remaining / page_size);
    if (words > (std::numeric_limits<std::size_t>::max() - page_size) / sizeof(std::uint64_t)) return {};
    const auto bitmap = round_page(words * sizeof(std::uint64_t));
    if (bitmap >= remaining) return {};
    const auto pages = (remaining - bitmap) / page_size;
    return Layout{bytes, references, bitmap, pages, bitmap_depth(pages)};
}

std::size_t SystemResourcePool::object_size(Kind kind) noexcept {
    switch (kind) {
    case Kind::memory_block: return 64;
    case Kind::block_info: return 16;
    case Kind::page_table: return page_size;
    }
    return 0;
}

SystemResourcePool::SystemResourcePool(Layout layout, Entropy entropy, void* context)
    : layout_(layout), entropy_(entropy), entropy_context_(context) {
    // Reserve descriptors before allocating the arena, so a constructor failure
    // cannot strand the backing allocation.
    const auto depth = bitmap_depth(layout.usable_pages);
    bitmap_.reserve(depth);
    arena_ = static_cast<std::byte*>(::operator new(layout.total_bytes, std::align_val_t(page_size)));
    std::memset(arena_, 0, layout.total_bytes);
    auto storage = reinterpret_cast<std::uint64_t*>(arena_ + layout.total_bytes - layout.bitmap_bytes);
    auto pages = layout.usable_pages;
    for (unsigned level = 0; level < depth; ++level) {
        pages = (pages + 63) / 64;
        bitmap_.emplace_back(storage, pages);
        storage += pages;
    }
    for (std::size_t index = 0; index < layout.usable_pages; ++index) set_free(index);
}

std::shared_ptr<SystemResourcePool> SystemResourcePool::create(std::size_t bytes,
    Entropy entropy, void* context, std::string& error) {
    const auto geometry = layout(bytes);
    if (!geometry || !entropy) {
        error = "Private system resources require page-aligned capacity, metadata overhead and an entropy source.";
        return {};
    }
    try {
        return std::shared_ptr<SystemResourcePool>(new SystemResourcePool(*geometry, entropy, context));
    } catch (const std::bad_alloc&) {
        error = "Private system-resource backing could not be allocated.";
        return {};
    }
}

SystemResourcePool::~SystemResourcePool() {
    ::operator delete(arena_, std::align_val_t(page_size));
}

void SystemResourcePool::set_free(std::size_t index) {
    for (auto level : bitmap_) {
        const auto word = index / 64;
        const auto previous = level[word];
        level[word] |= std::uint64_t{1} << (index % 64);
        if (previous) break;
        index = word;
    }
}

void SystemResourcePool::clear_free(std::size_t index) {
    for (auto level : bitmap_) {
        const auto word = index / 64;
        level[word] &= ~(std::uint64_t{1} << (index % 64));
        if (level[word]) break;
        index = word;
    }
}

unsigned SystemResourcePool::choose_bit(std::uint64_t bits) {
    unsigned selected = 0;
    for (unsigned half = 32; half; half /= 2) {
        const auto high = bits >> half;
        const auto low = bits & ((std::uint64_t{1} << half) - 1);
        bool choose_high = high && !low;
        if (high && low) {
            if (!bits_left_) { random_bits_ = entropy_(entropy_context_); bits_left_ = 64; }
            choose_high = (random_bits_ & 1) != 0;
            random_bits_ >>= 1; --bits_left_;
        }
        if (choose_high) { bits = high; selected += half; }
        else bits = low;
    }
    return selected;
}

std::byte* SystemResourcePool::allocate_page() {
    if (!bitmap_.back()[0]) return nullptr;
    std::size_t index = 0;
    for (auto it = bitmap_.rbegin(); it != bitmap_.rend(); ++it)
        index = index * 64 + choose_bit((*it)[index]);
    clear_free(index);
    ++used_pages_;
    return arena_ + layout_.reference_bytes + index * page_size;
}

std::optional<SystemResourcePool::Allocation> SystemResourcePool::allocate(Kind kind) {
    const auto size = object_size(kind);
    if (!size) return {};
    std::lock_guard lock(mutex_);
    auto& head = heads_[slot(kind)];
    auto* data = head;
    if (data) {
        std::memcpy(&head, data, sizeof(head));
    } else {
        data = allocate_page();
        if (!data) return {};
        for (std::size_t offset = size; offset < page_size; offset += size) {
            auto* entry = data + offset;
            std::memcpy(entry, &head, sizeof(head));
            head = entry;
        }
    }
    std::memset(data, 0, size);
    ++live_[slot(kind)];
    return Allocation(shared_from_this(), kind, data);
}

void SystemResourcePool::release(Kind kind, std::byte* data) noexcept {
    std::lock_guard lock(mutex_);
    auto& head = heads_[slot(kind)];
    std::memcpy(data, &head, sizeof(head));
    head = data;
    --live_[slot(kind)];
}

std::size_t SystemResourcePool::used_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    return used_pages_ * page_size;
}
std::size_t SystemResourcePool::live_objects(Kind kind) const noexcept {
    if (!object_size(kind)) return 0;
    std::lock_guard lock(mutex_);
    return live_[slot(kind)];
}

SystemResourcePool::Allocation::Allocation(Allocation&& other) noexcept
    : pool_(std::move(other.pool_)), kind_(other.kind_), data_(std::exchange(other.data_, nullptr)) {}
SystemResourcePool::Allocation& SystemResourcePool::Allocation::operator=(Allocation&& other) noexcept {
    if (this != &other) {
        reset(); pool_ = std::move(other.pool_); kind_ = other.kind_;
        data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
}
SystemResourcePool::Allocation::~Allocation() { reset(); }
void SystemResourcePool::Allocation::reset() noexcept {
    if (data_) { pool_->release(kind_, data_); data_ = nullptr; }
    pool_.reset();
}
std::span<std::byte> SystemResourcePool::Allocation::bytes() const noexcept {
    return data_ ? std::span<std::byte>(data_, object_size(kind_)) : std::span<std::byte>{};
}
} // namespace armigo::horizon
