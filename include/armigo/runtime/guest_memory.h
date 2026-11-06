// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/startup_queries.h>
#include <armigo/memory/guest_memory_model.h>
#include <armigo/runtime/launch_result.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/module_set.h>
#include <cstddef>
#include <memory>
#include <span>

namespace armigo::runtime {

// Lane C validates and pins the WHOLE range and required permissions atomically
// with mapping changes. A pin retains backing, callback context and its access
// rights against heap shrink, protect, alias changes and unmap until last release.
// It is not an arena keepalive or a snapshot permission check. Address-space
// metadata, shared-memory records and stack-alias plans never enforce native
// access by themselves.
// Unsupported guest/native subpage shapes fail transactionally; no fake mappings.
class PinnedGuestMemory {
public:
    virtual ~PinnedGuestMemory() = default;
    virtual horizon::AddressRange range() const noexcept = 0;
    virtual memory::Permission permissions() const noexcept = 0;
    // Contiguous native view, exactly range().size bytes. This is a live alias,
    // not a copy. Read permission is mandatory for this bounded pin interface.
    virtual std::span<const std::byte> bytes() const noexcept = 0;
    // Empty unless write was explicitly requested and admitted.
    virtual std::span<std::byte> writable_bytes() noexcept = 0;
};
using GuestMemoryPin = std::shared_ptr<PinnedGuestMemory>;
class GuestMemory {
public:
    virtual ~GuestMemory() = default;
    // The authoritative metadata/accounting object created with actual backing.
    virtual std::shared_ptr<horizon::ProcessRecord> process() const noexcept = 0;
    // Host bootstrap allocation, restricted to supported stack/TLS/normal uses.
    // Zero-filled, accounted, with a retained full-range pin. Guest SVC callers
    // must instead go through dispatch to preserve firmware validation/results.
    virtual LaunchResult<GuestMemoryPin> allocate(std::uint64_t bytes,
        horizon::MemoryType type, memory::Permission access) = 0;
    // Exact allocation retirement; fails without changing anything while pinned.
    virtual LaunchStatus release(horizon::AddressRange allocation) = 0;
    // Lane C's single memory-SVC transaction boundary. Unknown operations return
    // unsupported_operation unchanged; supported guest errors are encoded in
    // registers under their documented Horizon ABI. Host inability is Failure,
    // never a made-up guest Result. Includes heap/protect/alias/shared mappings.
    virtual LaunchResult<horizon::QueryDisposition> dispatch(
        std::uint16_t supervisor_call, horizon::QueryRegisters& registers) = 0;
    // Nonzero, overflow-free range. A success always contains a nonnull pin.
    // Mapping locks may not be held across the pin lifetime; conflicting
    // mutations fail or defer under their evidenced operation contract.
    virtual LaunchResult<GuestMemoryPin> pin(horizon::AddressRange range,
                                             memory::Permission access) = 0;
    // Bounded diagnostic copy (1..8 bytes), checked under the mapping lock.
    // It publishes no retained pin: concurrent mapping changes wait for the
    // copy instead of failing because instrumentation retained their range.
    // This protects mappings, not a multi-field snapshot of guest writes.
    virtual bool copy_observation(std::uint64_t address, std::span<std::byte> out) const noexcept {
        static_cast<void>(address);
        static_cast<void>(out);
        return false;
    }
    // Existing adapters borrow this object; the process owns it through all
    // callbacks. Each invocation pins its complete access. Preserve partial
    // ordered word-copy semantics and native aligned atomics from GuestCopyOut;
    // a host mutex around non-atomic memcpy does not implement guest atomics.
    virtual horizon::GuestCopyOut copy_out() noexcept = 0;
    virtual horizon::GuestBuffers ipc_buffers() noexcept = 0;
    virtual horizon::GuestMemoryMapper service_mapper() noexcept = 0;
};

struct ProcessMemoryOptions {
    // Explicit host resource budget, constrained by the admitted process profile.
    // Zero is invalid. Allocation geometry is chosen within profile.process and
    // published in process(); guessed private harness placements are forbidden.
    std::uint64_t resource_limit_bytes = 0;
};
// Registers every sealed module mapping/accounting entry without taking a second
// allocation owner. Retains module owners and the handle table for memory SVCs;
// creates ProcessRecord only after native reservation/metadata agree. Rollback
// leaves no published mappings or changed handles on failure.
[[nodiscard]] LaunchResult<std::shared_ptr<GuestMemory>> create_guest_memory(
    const LaunchProfile& profile, ProcessMemoryOptions options,
    std::shared_ptr<ModuleSet> modules, std::shared_ptr<horizon::HandleTable> handles);

// Pins do not serialize data access: callers separately coordinate guest writes,
// native atomics and GPU fences. Pin destruction is thread-safe, never reenters
// services and does not require an owning runtime that would form a cycle.

} // namespace armigo::runtime
