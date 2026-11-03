// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <pthread.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

// A host thread slot lets generated arm64 code reach one caller-owned pointer
// per host thread with two instructions and no call:
//
//     mrs  xT, TPIDRRO_EL0
//     ldr  xP, [xT, #tsd_byte_offset]
//
// Ownership and binding use only ordinary pthread APIs. The native descriptor
// is reported only after the candidate slot has been read back through a
// fault-safe Mach memory read and matched against pthread-written values on
// the thread concerned. Generated code only loads the slot, and only while
// that thread holds an active binding; an unbound thread is not diagnosed.

namespace armigo::platform {

enum class ThreadSlotError {
    none,
    unsupported_host,
    key_exhausted,
    resource_failure,
    invalid_argument,
    invalid_state,
    layout_mismatch,
    probe_failed,
    system_failure,
};

struct ThreadSlotStatus {
    ThreadSlotError error = ThreadSlotError::none;
    // errno for pthread failures or kern_return_t for a failed Mach read.
    int system_error = 0;
    bool ok() const noexcept { return error == ThreadSlotError::none; }
};

// Byte displacement of a slot from the thread-specific-data base read from
// TPIDRRO_EL0, used unmasked. Always a multiple of eight and at most
// max_tsd_byte_offset, so one unsigned-offset LDR (immediate) encodes it.
// Generated code may only load the slot. It must never store to it, never
// write TPIDR_EL0 or TPIDRRO_EL0, and never retain the loaded pointer
// beyond the binding that published it.
struct NativeThreadSlotAccess {
    static constexpr std::uint32_t max_tsd_byte_offset = 4095 * 8;
    std::uint32_t tsd_byte_offset = 0;
};

class ThreadSlotBinding;

// Owns one private pthread key. Creation verifies the candidate slot on the
// creating thread; every thread that executes generated code still requires
// its own successful ThreadSlotBinding.
//
// The key is neither copyable nor movable, so bindings may refer to it. It
// may be bound concurrently from any number of threads. Its destruction must
// happen after the construction and destruction of every binding of it have
// completed, by the caller's own synchronization; the key cannot detect a
// binding constructor or destructor that overlaps its destructor. Destroying
// it while a completed binding is still active aborts the process instead of
// deleting a key number that libpthread could hand out again.
class ThreadSlotKey {
public:
    ThreadSlotKey() noexcept;
    ThreadSlotKey(const ThreadSlotKey&) = delete;
    ThreadSlotKey& operator=(const ThreadSlotKey&) = delete;
    ~ThreadSlotKey();

    ThreadSlotStatus status() const noexcept { return status_; }
    bool valid() const noexcept { return valid_; }

    // Present only for a valid key. Code may be translated with this
    // descriptor before a binding exists, but must not run on a thread until
    // that thread holds an active binding.
    std::optional<NativeThreadSlotAccess> native_access() const noexcept;

    std::size_t active_bindings() const noexcept {
        return active_bindings_.load(std::memory_order_acquire);
    }

private:
    friend class ThreadSlotBinding;

    pthread_key_t key_{};
    NativeThreadSlotAccess access_{};
    ThreadSlotStatus status_{};
    bool valid_ = false;
    std::atomic<std::size_t> active_bindings_{0};
};

// Publishes one caller-owned, non-null pointer in the key's slot for the
// constructing thread only, and restores the previous slot value on
// destruction.
//
// Checked binding rules:
// - The binding is neither copyable nor movable and belongs to the thread
//   that constructed it. Destroying it on another thread aborts.
// - Active bindings on one thread form a strict stack across all keys; they
//   must be destroyed in reverse order of construction or the process aborts.
//   Nesting on the same key is allowed and restores the outer pointer.
// - A thread must destroy its bindings before it exits. A thread that exits
//   with a published pointer aborts the process from the key destructor.
//
// The caller must keep the pointed-to storage valid until binding destruction.
// The slot holds an opaque pointer and cannot check its storage lifetime.
//
// A failed binding leaves the slot unchanged, reports a status, and is inert.
// Construction and destruction are not async-signal-safe; a signal handler
// must not run code that depends on the slot while either is in progress.
//
// status(), active() and native_access() may be called from any thread, but
// only while the binding object is alive: the caller must order every such
// call after construction completes and before destruction begins, as for
// any C++ object. The internal atomic flag only keeps such calls free of data
// races; it does not make a call that overlaps or follows destruction valid.
//
// The slot gives generated code no diagnostic for a thread without a binding.
// Before entering generated code on a thread, the runtime must confirm that
// the thread holds an active binding, for example with native_access() on
// that thread.
class ThreadSlotBinding {
public:
    ThreadSlotBinding(ThreadSlotKey& key, void* pointer) noexcept;
    ThreadSlotBinding(const ThreadSlotBinding&) = delete;
    ThreadSlotBinding& operator=(const ThreadSlotBinding&) = delete;
    ~ThreadSlotBinding();

    ThreadSlotStatus status() const noexcept { return status_; }
    bool active() const noexcept { return active_.load(std::memory_order_acquire); }

    // Present only when called on the owning thread while this binding is
    // active; empty on every other thread.
    std::optional<NativeThreadSlotAccess> native_access() const noexcept;

private:
    ThreadSlotKey* key_ = nullptr;
    void* pointer_ = nullptr;
    void* previous_ = nullptr;
    ThreadSlotBinding* outer_ = nullptr;
    pthread_t owner_{};
    ThreadSlotStatus status_{};
    std::atomic<bool> active_{false};
};

} // namespace armigo::platform
