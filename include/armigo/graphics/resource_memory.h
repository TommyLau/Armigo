// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/render_submission.h>
#include <armigo/graphics/memory.h>
#include <span>
#include <utility>
#include <vector>

namespace armigo::graphics {

// All fields are nonzero. allocation is scoped to owner; generation changes
// on free/reallocation even at the same native or GPU address. Never wrap/reuse.
struct BackingId {
    std::uint64_t owner = 0, allocation = 0, generation = 0;
    bool operator==(const BackingId&) const = default;
};
struct BackingRange {
    BackingId backing;
    std::uint64_t offset = 0, size = 0;
    bool operator==(const BackingRange&) const = default;
};
struct MappingId {
    std::uint64_t address_space = 0, mapping = 0, generation = 0;
    bool operator==(const MappingId&) const = default;
};
struct RetainedRange {
    MappingId mapping;
    BackingRange bytes;
    // Pins the WHOLE range and callback owner, not only the containing arena.
    RetainedResource lease;
};
struct BackingRegistration {
    BackingId id;
    std::uint64_t size = 0;
    RetainedResource owner;
};
struct MappingRegistration {
    MappingId id;
    BackingId backing;
    Mapping mapping;
};

// Versions belong to backing-relative intervals, not GPU-VA cache entries.
// cpu_write increases only after an admitted CPU write; gpu_write increases
// in command order when GPU ownership is reserved, before a consumer acquires.
struct ResourceVersion {
    std::uint64_t cpu_write = 0, gpu_write = 0;
    bool operator==(const ResourceVersion&) const = default;
};
struct DirtyRange {
    BackingRange bytes;
    ResourceVersion version;
    bool cpu_dirty = false, gpu_dirty = false;
    CompletionToken last_writer;
};
enum class CoherenceMode : std::uint8_t { SubmissionBoundary, TrackedNativeAccess };
enum class ResolveReason : std::uint8_t {
    CpuRead, CpuEngineRead, Capture, IncompatibleAlias, SubmissionBoundary
};

struct GpuAccessRequest {
    RetainedRange range;
    ResourceAccess access = ResourceAccess::Read;
};

// Pins/excludes overlapping access for ALL ranges atomically. Destruction before
// commit cancels without changing versions. Commit validates/retains native
// storage first, then publishes all read/write intents atomically. Read intents
// remain live until the token is GPU-quiescent; writes invalidate overlaps in
// command order and retain the authoritative allocation even on failed feedback.
class GpuAccessReservation : public ResourceLease {
public:
    virtual std::vector<CompletionToken> dependencies() const = 0;
    virtual RendererStatus commit(std::vector<RetainedResource> storage) = 0;
};

// Retain this lease through completion or failed-authority handoff, and honor
// its dependencies before submission.
// Eager acceptance has published interval versions and storage authority;
// batch acceptance records them in the journal until an observation or flush.
struct GpuAccessAcceptance {
    std::vector<CompletionToken> dependencies;
    RetainedResource lease;
};

// Retains the resolved CPU-readable bytes and excludes overlapping GPU writes
// through the actual native load/copy, not merely through the resolve call.
// Destruction releases the exclusion; read-only accesses never dirty versions.
class CpuReadTransaction {
public:
    virtual ~CpuReadTransaction() = default;
};

// prepare resolves/preserves GPU-newer bytes and excludes aliases before a
// native CPU store. commit records exactly the written interval and releases
// exclusion. Destroying an unused transaction cancels it; after ANY native
// store commit is mandatory before release. Tracker faults marshal this work
// to the coordinator, never hold driver locks across the native instruction.
class CpuWriteTransaction {
public:
    virtual ~CpuWriteTransaction() = default;
    virtual void commit() noexcept = 0;
};

// L2 owns the registry; L3 proves native tracking before TrackedNativeAccess.
// Serialized coordinator calls, except native tracker notifications marshalled
// to that coordinator. No Metal or driver-lock work in a native fault handler.
// Missing pins, unmapped/crossing/overflowing ranges, stale generations and
// unsupported reinterpretations return owned CommandResult errors. No fallback
// to gpu_address == host pointer. Aliases resolve to the same interval versions.
class ResourceMemory {
public:
    virtual ~ResourceMemory() = default;
    virtual RendererStatus register_backing(BackingRegistration backing) = 0;
    virtual RendererStatus register_mapping(MappingRegistration mapping) = 0;
    // Remove lookup identity atomically; defer/fail conflicting pinned changes.
    // Old leases must never become views of newly allocated contents.
    virtual RendererStatus unmap(MappingId mapping) = 0;
    virtual RendererStatus retire_backing(BackingId backing) = 0;
    virtual RendererResult<RetainedRange> retain(std::uint64_t gpu_address,
        std::uint64_t size, ResourceAccess access) = 0;
    // Borrow only during the synchronous call. The implementation owns the
    // callbacks and pins; copying GpuAddressSpace alone does not retain them.
    virtual const GpuAddressSpace& address_space() const noexcept = 0;
    virtual CoherenceMode coherence_mode() const noexcept = 0;
    virtual RendererResult<std::vector<DirtyRange>> inspect(BackingRange range) = 0;
    // Whether GPU work wrote or will write any byte of the range (a GPU-dirty
    // or pending-writer interval); equivalent to scanning inspect().
    virtual RendererResult<bool> gpu_written(BackingRange range) {
        auto state = inspect(range);
        if (auto* failure = std::get_if<RendererError>(&state)) return *failure;
        for (const auto& interval : std::get<std::vector<DirtyRange>>(state))
            if (interval.gpu_dirty) return true;
        return false;
    }
    // Wait for successful GPU feedback (not guest Complete: coherence may be
    // finishing that same token), then merge only requested dirty
    // bytes/aspects. Failure keeps dirty state and authoritative storage alive.
    // No unconditional draw-time readback. Boundary mode is the conservative
    // exception until native loads/stores through EVERY alias are tracked.
    virtual RendererStatus resolve(BackingRange range, ResolveReason reason) = 0;
    // Native tracking must keep this transaction through the resumed load;
    // resolve alone is not permission to race a later GPU writer.
    virtual RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange range, ResolveReason reason) = 0;
    // Preserve bytes outside a partial write in the native tracking granule.
    // Failure returns no transaction and permits no native store.
    virtual RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(
        BackingRange range) = 0;
    // Include GPU readers, not only writers, so native stores cannot race fetch.
    // All ranges and the valid unique writer token are checked before reserving
    // anything. Dependencies returned by the reservation must be honored.
    virtual RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) = 0;

    // Fused acceptance for serialized callers with all native storage prepared.
    // Failure publishes no access; success requires no later registry commit.
    // Prepare caller-owned tracking capacity before entering this method.
    virtual RendererResult<GpuAccessAcceptance> accept_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer,
        std::vector<RetainedResource> storage) {
        try {
            auto reserved = reserve_accesses(requests, consumer);
            if (const auto* failure = std::get_if<RendererError>(&reserved)) return *failure;
            auto reservation = std::get<std::shared_ptr<GpuAccessReservation>>(std::move(reserved));
            GpuAccessAcceptance accepted{reservation->dependencies(), reservation};
            auto status = reservation->commit(std::move(storage));
            if (const auto* failure = std::get_if<RendererError>(&status)) return *failure;
            return accepted;
        } catch (...) {
            return RendererError({CommandStatus::Stopped,
                                  "GPU access acceptance failed with an exception"});
        }
    }

    // Batch acceptance defers interval publication until observation or flush.
    // The batch identity must live until flush/cancellation. Dependencies and
    // storage are retained immediately; existing eager acceptance is unchanged.
    // Packets of a batch execute in queue order; returned dependencies cover
    // accesses outside that batch. Use a distinct identity for each batch.
    virtual RendererResult<GpuAccessAcceptance> accept_batch_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer,
        std::span<const RetainedResource> storage, const RetainedResource& batch) {
        static_cast<void>(batch);
        return accept_accesses(requests, consumer, {storage.begin(), storage.end()});
    }
    virtual RendererStatus flush_access_journal() { return renderer_success(); }
    // Cancels only unpublished entries. Already observed authority survives
    // failure, just as it does for eager acceptance.
    virtual void discard_access_journal(const RetainedResource& batch) noexcept {
        static_cast<void>(batch);
    }

};

// Cache/queue domain checks use the exact retained registry instance plus a
// nonzero device identity. GPU address-space identity alone is insufficient.
struct ResidencyDomain {
    std::uint64_t device = 0;
    const ResourceMemory* memory = nullptr;
    bool operator==(const ResidencyDomain&) const = default;
};

} // namespace armigo::graphics
