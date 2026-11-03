// ============================================================================
// Armigo Horizon - Kernel Object Handle Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Typed kernel-object handle storage for the Horizon compatibility layer.
//
// A handle value packs a slot index, a nonzero allocation serial, and two
// reserved zero bits:
//
//   bits  0..14  slot index within this table
//   bits 15..29  allocation serial, always nonzero for a valid handle
//   bits 30..31  reserved, always zero
//
// The serial is a single per-table value advanced by every successful insert
// (1, 2, ..., 32767, then back to 1), not a per-slot counter. A slot reused
// after a close therefore normally carries a different serial than the handle
// that was closed, and the stale handle fails lookup and close. Because the
// serial space is finite, a full cycle of 32767 later allocations can make a
// stale handle value valid again. This table never promises that a stale
// handle stays invalid forever; callers that need longer guarantees must not
// keep handles across that many allocations.
//
// Pseudo-handles (such as current process and current thread) name the
// invoking context instead of a table slot and are the caller's
// responsibility. They are rejected here like any other value with reserved
// bits set.
//
// Concurrency contract: all member functions are thread-safe against each
// other while the table object stays alive. The caller must keep the table
// alive for the duration of every concurrent operation; concurrent
// destruction is not supported. lookup pins the object with a shared_ptr
// while the table lock is held, so a returned object outlives the table.
// close removes only the entry whose serial matches the handle and releases
// the last table reference outside the table lock, so a KernelObject
// destructor may safely reenter a live table.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace armigo::horizon {

// A Horizon user-mode handle value as passed through the guest ABI.
using Handle = std::uint32_t;

// Polymorphic base class of every object stored in a handle table. Derived
// classes carry their own Horizon object type; lookup_as checks that type
// with a dynamic cast. The constructor is protected so only derived object
// types can be created.
class KernelObject {
public:
    virtual ~KernelObject() = default;

protected:
    KernelObject() = default;
};

// A fixed-capacity table mapping Handle values to shared KernelObject
// ownership, guarded by an internal mutex.
//
// All storage and free-slot bookkeeping is allocated by the constructor.
// insert, lookup, lookup_as, close, size, and capacity do not allocate.
// The constructor throws std::invalid_argument when capacity is outside
// 1..max_capacity and std::bad_alloc when storage cannot be allocated. It
// does not emulate Horizon result codes; callers translate failures.
//
// max_capacity is the limit of this handle encoding (15 index bits), not a
// claim about the handle-table size any guest process is permitted to use.
// A process's real table size comes from its process configuration.
class HandleTable {
public:
    static constexpr unsigned index_bits = 15;
    static constexpr std::size_t max_capacity = std::size_t{1} << index_bits;

    HandleTable() = delete;
    explicit HandleTable(std::size_t capacity);
    HandleTable(const HandleTable&) = delete;
    HandleTable& operator=(const HandleTable&) = delete;
    HandleTable(HandleTable&&) = delete;
    HandleTable& operator=(HandleTable&&) = delete;

    // Takes ownership of object and returns its new handle. Fails (returns
    // std::nullopt) for a null object or when the table is full. On failure
    // the table is unchanged. Normal by-value shared_ptr ownership applies:
    // passing the caller's last reference with std::move consumes it even
    // when insertion fails.
    std::optional<Handle> insert(std::shared_ptr<KernelObject> object);

    // Returns the object for handle, or null for a zero handle, a handle
    // with reserved bits set (including pseudo-handles), a zero serial, an
    // out-of-range index, a stale serial, or a wrong generation. The
    // returned shared_ptr pins the object beyond this call.
    std::shared_ptr<KernelObject> lookup(Handle handle) const;

    // lookup with a dynamic cast to T. Returns null when the handle is
    // invalid or its object is not dynamically of type T.
    template <class T>
    std::shared_ptr<T> lookup_as(Handle handle) const {
        return std::dynamic_pointer_cast<T>(lookup(handle));
    }

    // Releases the entry for handle. Returns false and changes nothing when
    // handle is invalid or its serial does not name the entry's current
    // generation. On true, the object's last table reference is dropped
    // after the table lock is released.
    bool close(Handle handle);

    // Number of live entries.
    std::size_t size() const;

    // Number of slots, fixed for the table's lifetime.
    std::size_t capacity() const;

private:
    static constexpr unsigned serial_bits = 15;
    static constexpr std::uint32_t index_mask = (std::uint32_t{1} << index_bits) - 1;
    static constexpr std::uint32_t serial_mask = (std::uint32_t{1} << serial_bits) - 1;
    static constexpr std::uint16_t min_serial = 1;
    static constexpr std::uint16_t max_serial =
        static_cast<std::uint16_t>(serial_mask);
    // Free-list terminator; never a valid slot index.
    static constexpr std::uint32_t no_index = std::uint32_t{1} << (index_bits + 1);

    struct Entry {
        std::shared_ptr<KernelObject> object;
        std::uint16_t serial = 0;
        std::uint32_t next_free = no_index;
    };

    // The caller must hold mutex_. Returns the entry when handle names its
    // current generation, otherwise null.
    const Entry* find_valid_entry(Handle handle) const;

    // Slot allocation order is an internal policy: the constructor chains
    // slots in ascending index order and close pushes freed slots to the
    // front. No observable contract depends on which free slot insert uses.
    std::vector<Entry> entries_;
    const std::size_t capacity_;
    std::size_t count_ = 0;
    std::uint32_t free_head_ = no_index;
    std::uint16_t next_serial_ = min_serial;
    mutable std::mutex mutex_;
};

}  // namespace armigo::horizon
