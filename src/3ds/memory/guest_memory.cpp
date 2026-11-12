// ============================================================================
// Armigo Memory - Guest Memory Management
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/3ds/guest_memory.h"

namespace armigo::n3ds {
namespace {
constexpr std::uint32_t kMisalignedAddress = 0xe0e01bf1;
constexpr std::uint32_t kMisalignedSize = 0xe0e01bf2;
constexpr std::uint32_t kInvalidCombination = 0xe0e01bee;
constexpr std::uint32_t kInvalidAddress = 0xe0e01bf5;
constexpr std::uint32_t kInvalidAddressState = 0xe0a01bf5;
constexpr std::uint32_t kOutOfHeapMemory = 0xc860180a;
constexpr std::uint32_t kNotImplemented = 0xe0e01bf4;
constexpr std::uint32_t kHeapBegin = 0x08000000;
constexpr std::uint32_t kHeapEnd = 0x10000000;
constexpr std::uint32_t kPageMask = 4095;
} // namespace

bool GuestMemory::Initialize(std::uint64_t application_budget, MapGuestMemory mapper,
                             void* context) noexcept {
    if (!application_budget || (application_budget & kPageMask) ||
        application_budget > 0xfffff000ULL || !mapper) {
        return false;
    }
    budget_ = application_budget;
    committed_ = 0;
    count_ = 0;
    commits_ = 0;
    linear_base_ = 0;
    linear_size_ = 0;
    linear_used_ = 0;
    host_error_ = 0;
    mapper_ = mapper;
    context_ = context;
    return true;
}

bool GuestMemory::ConfigureLinearHeap(std::uint32_t address, std::uint32_t size) noexcept {
    if (!mapper_ || linear_size_ || commits_ || address < kHeapEnd ||
        (address & kPageMask) || (size & kPageMask) || size < budget_ ||
        !Available(address, size)) {
        return false;
    }
    linear_base_ = address;
    linear_size_ = size;
    linear_used_ = 0;
    return true;
}

bool GuestMemory::Available(std::uint32_t address, std::uint32_t size) const noexcept {
    const auto end = std::uint64_t{address} + size;
    if (!size || end > (1ULL << 32)) {
        return false;
    }
    for (std::size_t i = 0; i < count_; ++i) {
        const auto& range = ranges_[i];
        if (address < std::uint64_t{range.address} + range.size && end > range.address) {
            return false;
        }
    }
    return true;
}

bool GuestMemory::RegisterExisting(GuestMemoryRange range, bool charge_application) noexcept {
    if (!mapper_ || count_ == ranges_.size() || (range.address & kPageMask) ||
        (range.size & kPageMask) || (range.permissions & ~7U) ||
        !Available(range.address, range.size) ||
        (linear_size_ && range.address < std::uint64_t{linear_base_} + linear_size_ &&
         std::uint64_t{range.address} + range.size > linear_base_) ||
        (charge_application && range.size > budget_ - committed_)) {
        return false;
    }
    ranges_[count_++] = range;
    if (charge_application) {
        committed_ += range.size;
    }
    return true;
}

std::uint32_t GuestMemory::Commit(std::uint32_t operation, std::uint32_t address,
                                 std::uint32_t alias_address, std::uint32_t size,
                                 std::uint32_t permissions,
                                 std::uint32_t& output_address) noexcept {
    host_error_ = 0;
    if ((address & kPageMask) || (alias_address & kPageMask)) {
        return kMisalignedAddress;
    }
    if (size & kPageMask) {
        return kMisalignedSize;
    }
    if (permissions & ~3U) {
        return kInvalidCombination;
    }
    if ((operation != 3 && operation != 0x10003) || !size || !mapper_ || count_ == ranges_.size()) {
        return kNotImplemented;
    }
    const bool linear = operation == 0x10003;
    if (linear) {
        if (!linear_size_ || address != 0) {
            return kNotImplemented;
        }
        if (size > linear_size_ - linear_used_) {
            return kOutOfHeapMemory;
        }
        address = linear_base_ + linear_used_;
    } else if (address < kHeapBegin || address >= kHeapEnd || size > kHeapEnd - address) {
        return kInvalidAddress;
    }
    if (!Available(address, size)) {
        return kInvalidAddressState;
    }
    if (size > budget_ - committed_) {
        return kOutOfHeapMemory;
    }
    const auto mapped = mapper_(context_, address, size, permissions);
    if (mapped.error || mapped.address != address) {
        host_error_ = mapped.error ? mapped.error : -1;
        return kOutOfHeapMemory;
    }
    ranges_[count_++] = {address, size, permissions};
    committed_ += size;
    ++commits_;
    if (linear) {
        linear_used_ += size;
    }
    output_address = address;
    return 0;
}

} // namespace armigo::n3ds
