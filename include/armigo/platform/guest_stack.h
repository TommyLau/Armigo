// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Owned, guarded native stack for running prepared guest code.
//
// A GuardedStack is one Mach virtual-memory allocation laid out as
//
//     [ lower guard | usable interior | upper guard ]
//
// Both guards are inaccessible: their current and maximum protections are
// VM_PROT_NONE, so no later protection change can open them. The interior is
// zero-filled and readable/writable, with its maximum protection limited to
// read/write so it can never become executable. Every size is a whole number
// of native pages; the usable top is therefore 16-byte aligned and is the
// initial stack pointer an AAPCS64 entry may use.
//
// Placement is either anywhere (the kernel chooses) or fixed at a
// caller-chosen usable base, for callers that must reproduce a supplied guest
// address layout. Fixed placement never replaces an existing mapping: it uses
// a fixed allocation without VM_FLAGS_OVERWRITE, so any overlap with an
// existing mapping, including one that only touches a guard, fails and leaves
// that mapping untouched. A layout also outside the task's address range can
// report address_unavailable before the kernel reports its overlap.
//
// The object is neither copyable nor movable; create() returns it through a
// unique_ptr, so the addresses it reports, and its exclusive-use flag, stay
// valid for its whole lifetime. Destroying a stack that is in use aborts the
// process, because a guest frame may still be live on it.
//
// This type knows nothing about Horizon, guest threads, or the guest address
// space; it only owns host memory and reports its layout.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace armigo::platform {

enum class GuardedStackError : std::uint8_t {
    none,
    invalid_size,           // usable size is zero
    invalid_limits,         // limits exceed the hard ceilings below
    resource_limit,         // usable or guard size above the configured limits
    address_overflow,       // the layout does not fit in the address space
    misaligned_fixed_base,  // fixed usable base is not native-page aligned
    address_occupied,       // fixed placement overlaps an existing mapping
    address_unavailable,    // fixed placement outside the task's address range
    allocation_failed,      // the kernel could not provide the memory
    protection_failed,      // guard or interior protection could not be applied
    system_failure,         // unexpected host failure (see system_error)
};

const char* describe(GuardedStackError error) noexcept;

struct GuardedStackStatus {
    GuardedStackError error = GuardedStackError::none;
    // The kern_return_t of the failing Mach call, or zero.
    int system_error = 0;
    bool ok() const noexcept { return error == GuardedStackError::none; }
};

// Ceilings that configured limits may not exceed.
inline constexpr std::size_t kGuardedStackHardMaxUsableBytes = std::size_t{1} << 32;  // 4 GiB
inline constexpr std::size_t kGuardedStackHardMaxGuardBytes = std::size_t{1} << 28;   // 256 MiB

struct GuardedStackLimits {
    std::size_t max_usable_bytes = std::size_t{256} << 20;  // after page rounding
    std::size_t max_guard_bytes = std::size_t{16} << 20;    // per guard, after rounding
};

struct GuardedStackOptions {
    // Requested usable bytes; rounded up to whole native pages.
    std::size_t usable_size = 0;
    // Bytes in each guard; rounded up to whole native pages. Zero selects one
    // native page. A guard is never smaller than one page.
    std::size_t guard_size = 0;
    // When set, the usable interior must start exactly here. The address must
    // be native-page aligned; the lower guard occupies the pages just below.
    std::optional<std::uintptr_t> fixed_usable_base;
    GuardedStackLimits limits{};
};

// Host addresses of one stack. All ranges are half-open and contiguous:
// lower_guard_base + guard_size == usable_base, usable_base + usable_size ==
// aligned_top == upper_guard_base.
struct GuardedStackLayout {
    std::size_t page_size = 0;
    std::uintptr_t mapping_base = 0;
    std::size_t mapping_size = 0;
    std::uintptr_t lower_guard_base = 0;
    std::size_t guard_size = 0;  // size of each guard
    std::uintptr_t usable_base = 0;
    std::size_t usable_size = 0;
    std::size_t requested_usable_size = 0;
    // One past the last usable byte; 16-byte aligned (in fact page aligned).
    std::uintptr_t aligned_top = 0;
    std::uintptr_t upper_guard_base = 0;
    bool fixed = false;
};

class GuardedStack;

struct GuardedStackResult {
    GuardedStackStatus status;
    std::unique_ptr<GuardedStack> stack;  // null on failure
    bool ok() const noexcept { return status.ok() && stack != nullptr; }
};

// Exclusive-use token for one stack. Obtained from GuardedStack::try_acquire;
// `acquired()` is false when another holder had the stack. The token releases
// the stack on destruction and must not outlive it. It is neither copyable
// nor movable, so a release can never happen twice or from a moved-from
// object. Holding the token is what "in use" means for GuardedStack: an
// executor holds it for the whole time any frame may be live on the stack.
class GuardedStackUse {
public:
    ~GuardedStackUse();
    GuardedStackUse(const GuardedStackUse&) = delete;
    GuardedStackUse& operator=(const GuardedStackUse&) = delete;
    GuardedStackUse(GuardedStackUse&&) = delete;
    GuardedStackUse& operator=(GuardedStackUse&&) = delete;

    bool acquired() const noexcept { return stack_ != nullptr; }
    GuardedStack* stack() const noexcept { return stack_; }

private:
    friend class GuardedStack;
    explicit GuardedStackUse(GuardedStack* stack) noexcept : stack_(stack) {}
    GuardedStack* stack_;
};

class GuardedStack {
public:
    static GuardedStackResult create(const GuardedStackOptions& options) noexcept;

    // Releases the allocation. Aborts the process if the stack is in use.
    ~GuardedStack();

    GuardedStack(const GuardedStack&) = delete;
    GuardedStack& operator=(const GuardedStack&) = delete;
    GuardedStack(GuardedStack&&) = delete;
    GuardedStack& operator=(GuardedStack&&) = delete;

    const GuardedStackLayout& layout() const noexcept { return layout_; }
    std::uintptr_t usable_base() const noexcept { return layout_.usable_base; }
    std::size_t usable_size() const noexcept { return layout_.usable_size; }
    std::uintptr_t aligned_top() const noexcept { return layout_.aligned_top; }

    // True when [address, address + size) lies inside the usable interior.
    bool contains(std::uintptr_t address, std::size_t size = 1) const noexcept;

    // Claims exclusive use. Never blocks: a stack already in use (on this or
    // another thread) yields an unacquired token.
    GuardedStackUse try_acquire() noexcept;
    bool in_use() const noexcept { return in_use_.load(std::memory_order_acquire); }

private:
    friend class GuardedStackUse;
    explicit GuardedStack(const GuardedStackLayout& layout) noexcept : layout_(layout) {}
    void release() noexcept;

    GuardedStackLayout layout_;
    std::atomic<bool> in_use_{false};
};

} // namespace armigo::platform
