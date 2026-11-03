// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/platform/guest_stack.h>

#include <mach/mach.h>
#include "mach_vm.h"
#include <mach/vm_statistics.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ARMIGO_GUARDED_STACK_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(ARMIGO_GUARDED_STACK_ASAN)
#define ARMIGO_GUARDED_STACK_ASAN 1
#endif
#if defined(ARMIGO_GUARDED_STACK_ASAN)
#include <sanitizer/asan_interface.h>
#endif

namespace armigo::platform {
namespace {

GuardedStackResult failure(GuardedStackError error, kern_return_t system_error = KERN_SUCCESS) {
    return {{error, static_cast<int>(system_error)}, nullptr};
}

// Rounds up to a whole number of pages; nullopt on overflow.
std::optional<std::size_t> round_to_pages(std::size_t bytes, std::size_t page) noexcept {
    if (bytes > std::numeric_limits<std::size_t>::max() - (page - 1)) return std::nullopt;
    return (bytes + page - 1) & ~(page - 1);
}

// Shadow left by instrumented frames that ran on the stack must not leak into
// whatever memory the kernel later reuses at these addresses.
void unpoison(std::uintptr_t base, std::size_t size) noexcept {
#if defined(ARMIGO_GUARDED_STACK_ASAN)
    __asan_unpoison_memory_region(reinterpret_cast<void*>(base), size);
#else
    (void)base;
    (void)size;
#endif
}

kern_return_t protect(mach_vm_address_t base, mach_vm_size_t size, vm_prot_t protection) noexcept {
    // Lower the maximum first so the range can never be reopened, then make
    // the current protection explicit as well.
    kern_return_t result = mach_vm_protect(mach_task_self(), base, size, TRUE, protection);
    if (result == KERN_SUCCESS) {
        result = mach_vm_protect(mach_task_self(), base, size, FALSE, protection);
    }
    return result;
}

} // namespace

const char* describe(GuardedStackError error) noexcept {
    switch (error) {
    case GuardedStackError::none: return "none";
    case GuardedStackError::invalid_size: return "usable size is zero";
    case GuardedStackError::invalid_limits: return "limits exceed the hard ceilings";
    case GuardedStackError::resource_limit: return "size above the configured limit";
    case GuardedStackError::address_overflow: return "layout does not fit in the address space";
    case GuardedStackError::misaligned_fixed_base: return "fixed usable base is not page aligned";
    case GuardedStackError::address_occupied: return "fixed placement overlaps an existing mapping";
    case GuardedStackError::address_unavailable: return "fixed placement outside the address range";
    case GuardedStackError::allocation_failed: return "kernel allocation failed";
    case GuardedStackError::protection_failed: return "protection change failed";
    case GuardedStackError::system_failure: return "unexpected host failure";
    }
    return "unknown guarded stack error";
}

GuardedStackResult GuardedStack::create(const GuardedStackOptions& options) noexcept {
    const auto& limits = options.limits;
    if (limits.max_usable_bytes > kGuardedStackHardMaxUsableBytes ||
        limits.max_guard_bytes > kGuardedStackHardMaxGuardBytes) {
        return failure(GuardedStackError::invalid_limits);
    }
    if (options.usable_size == 0) return failure(GuardedStackError::invalid_size);

    const std::size_t page = vm_page_size;
    if (page == 0 || (page & (page - 1)) != 0) return failure(GuardedStackError::system_failure);

    const auto usable = round_to_pages(options.usable_size, page);
    if (!usable || *usable > limits.max_usable_bytes) {
        return failure(GuardedStackError::resource_limit);
    }
    const auto guard = round_to_pages(options.guard_size == 0 ? page : options.guard_size, page);
    if (!guard || *guard > limits.max_guard_bytes) {
        return failure(GuardedStackError::resource_limit);
    }
    // Both sizes are bounded by the hard ceilings, so this cannot overflow.
    const std::size_t total = *usable + 2 * *guard;

    mach_vm_address_t address = 0;
    int flags = VM_MAKE_TAG(VM_MEMORY_STACK);
    const bool fixed = options.fixed_usable_base.has_value();
    if (fixed) {
        const std::uintptr_t base = *options.fixed_usable_base;
        if ((base & (page - 1)) != 0) return failure(GuardedStackError::misaligned_fixed_base);
        if (base < *guard ||
            base - *guard > std::numeric_limits<std::uintptr_t>::max() - total) {
            return failure(GuardedStackError::address_overflow);
        }
        address = base - *guard;
        // No VM_FLAGS_OVERWRITE: an overlap fails instead of replacing.
        flags |= VM_FLAGS_FIXED;
    } else {
        flags |= VM_FLAGS_ANYWHERE;
    }

    kern_return_t result = mach_vm_allocate(mach_task_self(), &address, total, flags);
    if (result != KERN_SUCCESS) {
        if (fixed && result == KERN_NO_SPACE) return failure(GuardedStackError::address_occupied, result);
        if (fixed && result == KERN_INVALID_ADDRESS) {
            return failure(GuardedStackError::address_unavailable, result);
        }
        if (result == KERN_NO_SPACE || result == KERN_RESOURCE_SHORTAGE) {
            return failure(GuardedStackError::allocation_failed, result);
        }
        return failure(GuardedStackError::system_failure, result);
    }

    GuardedStackLayout layout;
    layout.page_size = page;
    layout.mapping_base = static_cast<std::uintptr_t>(address);
    layout.mapping_size = total;
    layout.lower_guard_base = layout.mapping_base;
    layout.guard_size = *guard;
    layout.usable_base = layout.mapping_base + *guard;
    layout.usable_size = *usable;
    layout.requested_usable_size = options.usable_size;
    layout.aligned_top = layout.usable_base + *usable;
    layout.upper_guard_base = layout.aligned_top;
    layout.fixed = fixed;

    if (fixed && layout.usable_base != *options.fixed_usable_base) {
        mach_vm_deallocate(mach_task_self(), address, total);
        return failure(GuardedStackError::system_failure);
    }

    result = protect(layout.lower_guard_base, *guard, VM_PROT_NONE);
    if (result == KERN_SUCCESS) result = protect(layout.upper_guard_base, *guard, VM_PROT_NONE);
    if (result == KERN_SUCCESS) {
        result = protect(layout.usable_base, *usable, VM_PROT_READ | VM_PROT_WRITE);
    }
    if (result != KERN_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), address, total);
        return failure(GuardedStackError::protection_failed, result);
    }

    auto* stack = new (std::nothrow) GuardedStack(layout);
    if (stack == nullptr) {
        mach_vm_deallocate(mach_task_self(), address, total);
        return failure(GuardedStackError::allocation_failed);
    }
    return {{}, std::unique_ptr<GuardedStack>(stack)};
}

GuardedStack::~GuardedStack() {
    if (in_use_.load(std::memory_order_acquire)) {
        armigo::diag::error("%s", "armigo: guarded stack destroyed while in use\n");
        std::abort();
    }
    unpoison(layout_.usable_base, layout_.usable_size);
    mach_vm_deallocate(mach_task_self(), layout_.mapping_base, layout_.mapping_size);
}

bool GuardedStack::contains(std::uintptr_t address, std::size_t size) const noexcept {
    return address >= layout_.usable_base && address <= layout_.aligned_top &&
           size <= layout_.aligned_top - address && (size != 0 || address < layout_.aligned_top);
}

GuardedStackUse GuardedStack::try_acquire() noexcept {
    bool expected = false;
    if (in_use_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        return GuardedStackUse(this);
    }
    return GuardedStackUse(nullptr);
}

void GuardedStack::release() noexcept {
    unpoison(layout_.usable_base, layout_.usable_size);
    in_use_.store(false, std::memory_order_release);
}

GuardedStackUse::~GuardedStackUse() {
    if (stack_ != nullptr) stack_->release();
}

} // namespace armigo::platform
