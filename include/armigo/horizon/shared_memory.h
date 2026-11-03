// ============================================================================
// Armigo Horizon - Shared Memory Object
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/handle_table.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>

namespace armigo::horizon {

// A read-write block of memory two processes can map, such as the applet
// resource shared memory the human interface device service hands the
// application.
//
// A guest mapping aliases the object's region rather than copying it, because
// the service writes controller state into the same bytes the application
// reads; a copy at map time would freeze the state. The size is chosen by the
// creating service and must contain complete 4096-byte pages.
//
// A host page larger than the guest page can only alias bytes at the same
// offset within a page, so before a guest maps the object at an address that is
// not host-page aligned, its bytes move to a new region at that phase
// (rephase). The guest memory map moves the bytes only while no guest mapping
// of the object exists. A writer that runs while the object can be mapped
// changes the bytes through update(), which a rephase waits for, so no write is
// lost or torn by the move and no writer holds the previous region, which the
// move frees. A pointer from data() or bytes() is valid until the next move.
class SharedMemory final : public KernelObject {
public:
    // Allocates a zero-filled region. Throws std::invalid_argument for an empty or
    // non-page-multiple size and std::bad_alloc when the region cannot be allocated.
    explicit SharedMemory(std::size_t size);
    ~SharedMemory() override;

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    SharedMemory(SharedMemory&&) = delete;
    SharedMemory& operator=(SharedMemory&&) = delete;

    std::size_t size() const noexcept { return size_; }
    // The object's first byte. A host mapping aliases this memory.
    std::byte* data() noexcept { return current_.load(std::memory_order_acquire); }
    const std::byte* data() const noexcept { return current_.load(std::memory_order_acquire); }
    std::span<std::byte> bytes() noexcept { return {data(), size_}; }
    std::span<const std::byte> bytes() const noexcept { return {data(), size_}; }
    // Runs `write` on the bytes while no rephase can move them.
    template <class Write>
    void update(Write&& write) {
        const std::lock_guard lock(mutex_);
        write(bytes());
    }
    // Whether the bytes start `phase` bytes into a region aligned to `page`
    // that spans whole pages past the last byte, so those pages can be
    // aliased whole.
    bool spans(std::size_t phase, std::size_t page);
    // Copies the bytes into a new region aligned to `page` (a power of two at
    // least 4096) with the first byte `phase` bytes into it; the region spans
    // whole pages from its start past the last byte. False when the region
    // cannot be allocated or the arguments are invalid; the bytes then stay.
    bool rephase(std::size_t phase, std::size_t page);

private:
    struct Release {
        void operator()(std::byte* memory) const noexcept;
    };

    std::mutex mutex_;
    std::unique_ptr<std::byte, Release> region_;
    std::atomic<std::byte*> current_{nullptr};
    // The current region's length and the offset of the first byte in it.
    std::size_t length_ = 0;
    std::size_t phase_ = 0;
    std::size_t size_ = 0;
};

}  // namespace armigo::horizon
