// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/thread_slot.h>

#include <mach/mach.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace armigo::platform {
namespace {

// Innermost active binding of the current thread, across all keys.
constinit thread_local ThreadSlotBinding* innermost_binding = nullptr;

// Distinct, process-private value written during a probe so that a match
// cannot be explained by a stale or coincidental slot value.
constinit unsigned char probe_token = 0;

[[noreturn]] void contract_violation(const char* message) noexcept {
    static constexpr char prefix[] = "armigo: thread slot contract violation: ";
    (void)::write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    (void)::write(STDERR_FILENO, message, std::strlen(message));
    (void)::write(STDERR_FILENO, "\n", 1);
    std::abort();
}

// libpthread calls this only for a thread that exits while a value is still
// published, which means a binding was never destroyed on that thread.
void abandoned_binding(void*) noexcept {
    contract_violation("a thread exited with an active binding");
}

ThreadSlotStatus failure(ThreadSlotError error, int system_error = 0) noexcept {
    return {error, system_error};
}

#if defined(__APPLE__) && defined(__aarch64__)

std::uintptr_t tsd_base() noexcept {
    std::uintptr_t value = 0;
    // A volatile read: the probe must observe the register, not a value the
    // compiler cached from an earlier read.
    __asm__ volatile("mrs %0, TPIDRRO_EL0" : "=r"(value));
    return value;
}

// Reads the candidate slot of the calling thread without dereferencing the
// address directly, so an unexpected layout yields an error instead of a fault.
ThreadSlotStatus read_candidate(std::uint32_t offset, std::uintptr_t& value) noexcept {
    const auto base = tsd_base();
    if (base == 0 || base % alignof(void*) != 0 ||
        base > std::numeric_limits<std::uintptr_t>::max() - offset) {
        return failure(ThreadSlotError::layout_mismatch);
    }
    std::uintptr_t word = 0;
    vm_size_t read = 0;
    const auto status = ::vm_read_overwrite(mach_task_self(), base + offset, sizeof(word),
        reinterpret_cast<vm_address_t>(&word), &read);
    if (status != KERN_SUCCESS) return failure(ThreadSlotError::probe_failed, status);
    if (read != sizeof(word)) return failure(ThreadSlotError::probe_failed);
    value = word;
    return {};
}

ThreadSlotStatus publish(pthread_key_t key, const void* value) noexcept {
    const int result = ::pthread_setspecific(key, value);
    if (result == ENOMEM) return failure(ThreadSlotError::resource_failure, result);
    if (result != 0) return failure(ThreadSlotError::system_failure, result);
    return {};
}

ThreadSlotStatus expect_slot(pthread_key_t key, std::uint32_t offset, const void* value) noexcept {
    if (auto status = publish(key, value); !status.ok()) return status;
    std::uintptr_t observed = 0;
    if (auto status = read_candidate(offset, observed); !status.ok()) return status;
    if (observed != reinterpret_cast<std::uintptr_t>(value) ||
        ::pthread_getspecific(key) != value) {
        return failure(ThreadSlotError::layout_mismatch);
    }
    return {};
}

// Publishes final_value after proving on this thread that the candidate slot
// tracks pthread_setspecific for two distinct values. On failure the slot is
// restored to restore_value.
ThreadSlotStatus probe_and_publish(pthread_key_t key, std::uint32_t offset,
                                   const void* final_value, const void* restore_value) noexcept {
    auto status = expect_slot(key, offset, &probe_token);
    if (status.ok()) status = expect_slot(key, offset, final_value);
    if (!status.ok()) (void)::pthread_setspecific(key, restore_value);
    return status;
}

#else

ThreadSlotStatus probe_and_publish(pthread_key_t, std::uint32_t, const void*, const void*) noexcept {
    return failure(ThreadSlotError::unsupported_host);
}

#endif

} // namespace

ThreadSlotKey::ThreadSlotKey() noexcept {
#if !(defined(__APPLE__) && defined(__aarch64__))
    status_ = failure(ThreadSlotError::unsupported_host);
    return;
#else
    const int result = ::pthread_key_create(&key_, abandoned_binding);
    if (result == EAGAIN) {
        status_ = failure(ThreadSlotError::key_exhausted, result);
        return;
    }
    if (result == ENOMEM) {
        status_ = failure(ThreadSlotError::resource_failure, result);
        return;
    }
    if (result != 0) {
        status_ = failure(ThreadSlotError::system_failure, result);
        return;
    }

    // Candidate from the libpthread convention tsd[key]; it is trusted only
    // after the probe below and again on every binding thread.
    const auto key_value = static_cast<std::uintmax_t>(key_);
    if (key_value > NativeThreadSlotAccess::max_tsd_byte_offset / sizeof(void*)) {
        (void)::pthread_key_delete(key_);
        status_ = failure(ThreadSlotError::layout_mismatch);
        return;
    }
    access_.tsd_byte_offset = static_cast<std::uint32_t>(key_value * sizeof(void*));

    status_ = probe_and_publish(key_, access_.tsd_byte_offset, nullptr, nullptr);
    if (!status_.ok()) {
        (void)::pthread_key_delete(key_);
        access_ = {};
        return;
    }
    valid_ = true;
#endif
}

ThreadSlotKey::~ThreadSlotKey() {
    if (!valid_) return;
    if (active_bindings_.load(std::memory_order_acquire) != 0) {
        contract_violation("a key was destroyed while bindings were active");
    }
    (void)::pthread_key_delete(key_);
}

std::optional<NativeThreadSlotAccess> ThreadSlotKey::native_access() const noexcept {
    if (!valid_) return {};
    return access_;
}

ThreadSlotBinding::ThreadSlotBinding(ThreadSlotKey& key, void* pointer) noexcept {
    if (!key.valid_) {
        status_ = failure(ThreadSlotError::invalid_state);
        return;
    }
    if (pointer == nullptr) {
        status_ = failure(ThreadSlotError::invalid_argument);
        return;
    }

    void* const previous = ::pthread_getspecific(key.key_);
    // Only an active binding on this thread may have published a value, and
    // every active binding is on this thread's stack of bindings.
    bool expected = previous == nullptr;
    for (auto* binding = innermost_binding; binding != nullptr; binding = binding->outer_) {
        if (binding->key_ == &key) {
            expected = binding->pointer_ == previous;
            break;
        }
    }
    if (!expected) {
        status_ = failure(ThreadSlotError::invalid_state);
        return;
    }

    status_ = probe_and_publish(key.key_, key.access_.tsd_byte_offset, pointer, previous);
    if (!status_.ok()) return;

    key_ = &key;
    pointer_ = pointer;
    previous_ = previous;
    outer_ = innermost_binding;
    owner_ = ::pthread_self();
    key.active_bindings_.fetch_add(1, std::memory_order_acq_rel);
    innermost_binding = this;
    active_.store(true, std::memory_order_release);
}

ThreadSlotBinding::~ThreadSlotBinding() {
    if (!active_.load(std::memory_order_acquire)) return;
    if (!::pthread_equal(owner_, ::pthread_self())) {
        contract_violation("a binding was destroyed on a thread that does not own it");
    }
    if (innermost_binding != this) {
        contract_violation("bindings were not destroyed in reverse order");
    }
    if (::pthread_getspecific(key_->key_) != pointer_) {
        contract_violation("a bound slot was changed outside its binding");
    }
    if (::pthread_setspecific(key_->key_, previous_) != 0) {
        contract_violation("a bound slot could not be restored");
    }
    innermost_binding = outer_;
    key_->active_bindings_.fetch_sub(1, std::memory_order_acq_rel);
    active_.store(false, std::memory_order_release);
}

std::optional<NativeThreadSlotAccess> ThreadSlotBinding::native_access() const noexcept {
    if (!active_.load(std::memory_order_acquire) || !::pthread_equal(owner_, ::pthread_self()))
        return {};
    return key_->access_;
}

} // namespace armigo::platform
