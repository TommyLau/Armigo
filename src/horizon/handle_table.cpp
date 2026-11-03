// ============================================================================
// Armigo Horizon - Kernel Object Handle Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Kernel-object handle table implementation. The concurrency and generation
// contracts live in include/armigo/horizon/handle_table.h.

#include <armigo/horizon/handle_table.h>

#include <stdexcept>
#include <utility>

namespace armigo::horizon {

HandleTable::HandleTable(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0 || capacity > max_capacity) {
        throw std::invalid_argument(
            "handle table capacity must be within 1..32768 slots");
    }

    // Reserve every slot and link them into the free list now so ordinary
    // operations never allocate.
    entries_.resize(capacity_);
    for (std::size_t index = 0; index < capacity_; ++index) {
        entries_[index].next_free =
            (index + 1 < capacity_) ? static_cast<std::uint32_t>(index + 1)
                                    : no_index;
    }
    free_head_ = 0;
}

std::optional<Handle> HandleTable::insert(std::shared_ptr<KernelObject> object) {
    if (!object) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (free_head_ == no_index) {
        return std::nullopt;  // Table exhausted.
    }

    const std::uint32_t index = free_head_;
    Entry& entry = entries_[index];
    free_head_ = entry.next_free;

    // Advance the single per-table serial 1..32767 and wrap back to 1. The
    // serial is shared by all slots, so a reused slot normally changes its
    // generation on every insertion.
    const std::uint16_t serial = next_serial_;
    next_serial_ = serial == max_serial ? min_serial : static_cast<std::uint16_t>(serial + 1);

    entry.serial = serial;
    entry.object = std::move(object);
    entry.next_free = no_index;
    ++count_;

    return (static_cast<Handle>(serial) << index_bits) | static_cast<Handle>(index);
}

std::shared_ptr<KernelObject> HandleTable::lookup(Handle handle) const {
    std::shared_ptr<KernelObject> pinned;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const Entry* entry = find_valid_entry(handle);
        if (entry != nullptr) {
            pinned = entry->object;  // Take the pin while the table is locked.
        }
    }
    return pinned;  // The pin is released by the caller, not under the lock.
}

bool HandleTable::close(Handle handle) {
    std::shared_ptr<KernelObject> released;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const Entry* found = find_valid_entry(handle);
        if (found == nullptr) {
            return false;  // Malformed, stale, or already closed.
        }

        // Only the handle's own generation is removed; find_valid_entry has
        // already matched the serial.
        Entry& entry = entries_[handle & index_mask];
        released = std::move(entry.object);
        entry.serial = 0;
        entry.next_free = free_head_;
        free_head_ = static_cast<std::uint32_t>(handle & index_mask);
        --count_;
    }

    // Drop the table's reference outside the lock. If this is the last
    // reference, the object destructor runs here and may reenter this table.
    released.reset();
    return true;
}

std::size_t HandleTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

std::size_t HandleTable::capacity() const {
    return capacity_;  // Immutable after construction.
}

const HandleTable::Entry* HandleTable::find_valid_entry(Handle handle) const {
    // Bits 30..31 must be zero. This also rejects pseudo-handles, which name
    // the invoking context and never a table slot.
    if ((handle & ~(index_mask | (serial_mask << index_bits))) != 0) {
        return nullptr;
    }
    const std::uint32_t index = handle & index_mask;
    const std::uint32_t serial = (handle >> index_bits) & serial_mask;
    if (serial == 0) {
        return nullptr;  // Zero serial is never an allocated generation.
    }
    if (index >= capacity_) {
        return nullptr;
    }

    const Entry& entry = entries_[index];
    if (!entry.object || entry.serial != serial) {
        return nullptr;  // Free slot or a stale generation.
    }
    return &entry;
}

}  // namespace armigo::horizon
