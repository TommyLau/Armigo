// ============================================================================
// Armigo Horizon - Shared Memory Object
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/shared_memory.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace armigo::horizon {
namespace {
constexpr std::size_t page_size = 4096;
}

void SharedMemory::Release::operator()(std::byte* memory) const noexcept {
    std::free(memory);
}

SharedMemory::SharedMemory(std::size_t size) : size_(size) {
    if (size == 0 || size % page_size != 0)
        throw std::invalid_argument("Shared memory must contain complete 4096-byte pages");
    // aligned_alloc requires a size that is a multiple of the alignment, which the check
    // above already guarantees.
    std::unique_ptr<std::byte, Release> memory(static_cast<std::byte*>(std::aligned_alloc(page_size, size)));
    if (!memory) throw std::bad_alloc();
    std::memset(memory.get(), 0, size);
    current_.store(memory.get(), std::memory_order_release);
    region_ = std::move(memory);
    length_ = size;
}

bool SharedMemory::spans(std::size_t phase, std::size_t page) {
    if (page == 0 || (page & (page - 1)) != 0 || phase >= page) return false;
    const std::lock_guard lock(mutex_);
    const auto base = reinterpret_cast<std::uintptr_t>(current_.load(std::memory_order_relaxed)) - phase_;
    return phase == phase_ && base % page == 0 && length_ >= ((phase + size_ + page - 1) & ~(page - 1));
}

bool SharedMemory::rephase(std::size_t phase, std::size_t page) {
    if (page < page_size || (page & (page - 1)) != 0 || phase >= page || phase % page_size != 0) return false;
    const std::lock_guard lock(mutex_);
    const std::size_t length = (phase + size_ + page - 1) & ~(page - 1);
    std::unique_ptr<std::byte, Release> memory(static_cast<std::byte*>(std::aligned_alloc(page, length)));
    if (!memory) return false;
    std::memset(memory.get(), 0, length);
    std::memcpy(memory.get() + phase, current_.load(std::memory_order_acquire), size_);
    current_.store(memory.get() + phase, std::memory_order_release);
    region_ = std::move(memory);  // frees the previous region
    length_ = length;
    phase_ = phase;
    return true;
}

SharedMemory::~SharedMemory() = default;

}  // namespace armigo::horizon
