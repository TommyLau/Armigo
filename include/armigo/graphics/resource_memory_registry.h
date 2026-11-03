// ============================================================================
// Armigo Graphics - Backing and Coherence Registry
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/resource_memory.h>

namespace armigo::graphics {

// Owned adapter for raw backing access. Registration owners pin the complete
// allocation; this object owns callback contexts. Calls must not reenter the
// registry. A failed write may have copied a prefix; the registry retains dirty
// authority and permits an idempotent retry. No native fault-handler calls.
class ResourceMemoryBackend : public ResourceLease {
public:
    virtual RendererStatus read_cpu(BackingRange range,
                                   std::span<std::byte> destination) = 0;
    virtual RendererStatus write_cpu(BackingRange range,
                                    std::span<const std::byte> source) = 0;
    // Wait only for GPU feedback, never finish()/guest-visible completion.
    // The registry calls this only for Submitted tokens, never Accepted work.
    // May return an error/timeout; the registry independently checks feedback.
    virtual RendererStatus wait_gpu(CompletionToken token) = 0;
};

// One immutable authority descriptor per committed write request. The storage
// must retain the native allocation, cover the entire request and return exactly
// the requested backing bytes (including layout/aspect conversion). It must not
// expose an unrelated cached version. A read failure must not write guest memory.
class ResourceMemoryStorage : public ResourceLease {
public:
    virtual BackingRange coverage() const noexcept = 0;
    virtual RendererStatus read_gpu(BackingRange range,
                                   std::span<std::byte> destination) const = 0;
    // Whether guest_boundary() writes this authority back to guest memory.
    // GPU-resident authorities (rendered textures, whose native texture owns
    // the content) decline and are resolved only on demand.
    virtual bool guest_boundary_writeback() const noexcept { return true; }
    // The authority is guest memory itself (a zero-copy view of the backing,
    // through host-addressable memory pools): GPU results already sit in guest
    // memory, so resolving only waits for the writer and copies nothing.
    virtual bool in_place() const noexcept { return false; }
};

struct ResourceStorageRange {
    DirtyRange dirty;
    std::shared_ptr<const ResourceMemoryStorage> storage;
};

// Retained prefix of registry authority. Publication is coordinator-only and
// idempotent after success; a failed write keeps every captured authority for
// retry. This object owns the registry state and backing allocation pins.
class GuestBoundary : public ResourceLease {
public:
    virtual RendererStatus publish() = 0;
};

// L2 implementation seam, separate from the frozen ResourceMemory interface.
class ResourceMemoryRegistry : public ResourceMemory {
public:
    // Coordinator-only address-space mutation. Unlike unmap(), passive cache
    // leases may survive but become stale. Pending CPU/GPU reservations reject
    // the change; submitted accesses must become quiescent before publication.
    // Resolve affected live backings through their owned CPU allocation first.
    // Explicitly discarded backings are no longer guest-owned: remove all their
    // mappings and dirty authority without writing to potentially reused bytes.
    // Failure preserves lookup identity and any unresolved dirty authority.
    virtual RendererStatus invalidate_mappings(
        std::span<const MappingId> mappings,
        std::span<const BackingId> discarded_backings = {}) = 0;
    // Exact live-identity query for bounded cache lifecycle checks. Calls are
    // serialized with registration and retirement on the coordinator.
    virtual bool has_backing(BackingId backing) const noexcept = 0;
    // Validate lease provenance, current mapping/backing and permitted access.
    // This does not reserve access or reject overlapping pending reservations.
    virtual RendererStatus validate_retained(const RetainedRange& range,
                                            ResourceAccess access) = 0;
    // Copy CPU bytes for residency through a current retained mapping. The
    // clipped range must be inside the lease, including for a GPU-write lease
    // whose initial contents need uploading. Resolves GPU authority and holds
    // a CPU-read transaction through the copy; stale leases cannot read bytes.
    virtual RendererStatus read_retained(const RetainedRange& retained,
                                        BackingRange bytes,
                                        std::span<std::byte> destination) = 0;
    // Snapshot exact clipped intervals and retain their current GPU authority.
    // Storage is null when GPU bytes are not newer. This does not wait, resolve,
    // or reserve access; consumers must still honor the writer dependencies.
    virtual RendererResult<std::vector<ResourceStorageRange>> inspect_storage(
        BackingRange range) = 0;
    // Call while native guest access is stopped, before resuming it and before
    // acquiring the next submission's resources. Waits for all GPU accesses,
    // resolves every GPU-dirty interval, then invalidates CPU-backed contents.
    // On failure do not resume native execution. Cache keys must include this
    // epoch as well as interval versions; cpu_write counts only actual stores.
    virtual RendererStatus submission_boundary() = 0;
    // Boundary between submissions that the guest cannot observe: it waits and
    // invalidates CPU-backed contents like submission_boundary(), but leaves
    // GPU-dirty intervals unresolved and GPU-authoritative. Under the guest
    // API's synchronization contract the CPU cannot legitimately touch memory
    // the GPU owns until a guest-visible synchronization point, where the
    // caller must run submission_boundary() first.
    virtual RendererStatus internal_boundary() { return submission_boundary(); }
    // Guest-visible boundary that keeps GPU-resident authorities: it acts like
    // submission_boundary() but leaves intervals whose storage declines
    // guest_boundary_writeback() unresolved and GPU-authoritative. Registry CPU
    // access, explicit resolves and mapping changes still write them back. A
    // native CPU store observed (with CPU write tracking) on such an interval
    // after it was kept fails a later guest boundary with a structured error,
    // because the stored bytes and the GPU result can no longer be merged.
    virtual RendererStatus guest_boundary() { return submission_boundary(); }
    // Capture after committing the preceding GPU batch. Later registry accesses
    // may be accepted, but the coordinator must withhold their native commits
    // and any other mutation of captured storage or guest bytes until publish()
    // succeeds. Publish boundaries in capture order, before mapping changes or
    // CPU stores. Publication waits only for captured accesses and never clears
    // a newer authority. preserve_resident applies guest_boundary() policy;
    // false writes back every captured dirty interval.
    virtual RendererResult<std::shared_ptr<GuestBoundary>> capture_guest_boundary(
        bool preserve_resident) {
        static_cast<void>(preserve_resident);
        return RendererError({CommandStatus::Unsupported,
                              "Captured guest boundaries are unsupported"});
    }
    // Reserve a later host store without waiting, resolving authority or
    // changing guest bytes. The guard pins the backing and excludes overlapping
    // CPU reads and GPU admissions; disjoint work and epoch advances remain
    // valid. Capture the preceding guest boundary before reserving this guard.
    // Publish that prefix and perform the owned host store before releasing the
    // guard, with later native commits still held. Destruction is coordinator-only.
    virtual RendererResult<std::shared_ptr<ResourceLease>> defer_host_store(BackingRange range) {
        static_cast<void>(range);
        return RendererError({CommandStatus::Unsupported,
                              "Deferred host stores are unsupported"});
    }
    virtual bool deferred_host_store(BackingRange range) const noexcept {
        static_cast<void>(range);
        return false;
    }
    virtual bool has_deferred_host_store() const noexcept { return false; }
    // Drops every retained access whose submission reached quiescence, with
    // the storage it kept alive, so ownership-sensitive callers see only live
    // holders. Never waits or resolves.
    virtual void release_settled_accesses() {}
    // Host address and size of a backing whose bytes are directly
    // CPU-addressable (native guest execution maps guest memory into the
    // host), for zero-copy native views; nullopt otherwise.
    virtual std::optional<std::pair<std::uint64_t, std::uint64_t>> backing_host_range(
        BackingId) const noexcept { return std::nullopt; }
    // Copies a range whose bytes are CPU-authoritative and host-addressable,
    // re-arming write tracking first as a read transaction would. Returns
    // false, copying nothing, when the range needs a read transaction
    // (GPU-written intervals, a pending writer, no host view).
    virtual RendererResult<bool> read_cpu_backed(BackingRange, std::span<std::byte>) { return false; }
    // Registers several mappings in order as register_mapping() would,
    // stopping at the first failure; registries may validate and publish the
    // whole batch with one address-space rebuild.
    virtual RendererStatus register_mappings(std::span<const MappingRegistration> registrations) {
        for (const auto& registration : registrations) {
            auto status = register_mapping(registration);
            if (!std::holds_alternative<std::monostate>(status)) return status;
        }
        return std::monostate{};
    }
    // Advance the CPU-uncertainty epoch without waiting for, rejecting or
    // resolving any GPU work, so a coordinator callback that enqueues new work
    // observes guest CPU bytes as of that entry while earlier submissions keep
    // the immutable copies they captured. Valid while GPU reservations are
    // pending or their work is unsubmitted; GPU-dirty intervals stay
    // GPU-authoritative, and the caller still must not touch bytes the GPU
    // owns until a guest-visible boundary. The default falls back to the full
    // submission boundary for registries without a tracked epoch.
    virtual RendererStatus advance_cpu_epoch() { return submission_boundary(); }
    virtual std::uint64_t cpu_cache_epoch() const noexcept = 0;
    // Optional CPU write tracking. Take a stamp BEFORE reading bytes a cache
    // will keep; cpu_written_since() is false only when no CPU store can have
    // reached any byte of the range after that stamp. Untracked ranges, stamp
    // zero and an unknown history answer true. Without a tracker the registry
    // keeps these defaults, so callers fall back to epoch revalidation.
    virtual std::uint64_t cpu_write_stamp() const noexcept { return 0; }
    // Announces a store the caller is about to make through the guest's CPU
    // mapping (a front-end constant-buffer update): its pages become writable
    // and publish a fresh write generation, so the store takes no fault.
    virtual void prepare_host_store(BackingRange range) noexcept { static_cast<void>(range); }
    // Retains a range of a known backing through any mapping that holds it
    // whole; nullopt when none does.
    virtual std::optional<RetainedRange> retain_backing(BackingRange range, ResourceAccess access) {
        static_cast<void>(range);
        static_cast<void>(access);
        return std::nullopt;
    }
    virtual bool cpu_written_since(BackingRange range, std::uint64_t stamp) const noexcept
    {
        static_cast<void>(range);
        static_cast<void>(stamp);
        return true;
    }
};

// Optional coordinator-only work accounting for deterministic scale tests.
// Counts actual index comparisons and interval visits, never elapsed time.
struct ResourceMemoryWorkCounters {
    std::uint64_t journal_publications = 0;
    std::uint64_t journal_entries_published = 0;
    std::uint64_t mapping_comparisons = 0;
    std::uint64_t backing_comparisons = 0;
    std::uint64_t interval_comparisons = 0;
    std::uint64_t interval_visits = 0;
    std::uint64_t boundary_backings = 0;
    std::uint64_t boundary_epoch_edges = 0;
};

struct ResourceMemoryOptions {
    std::uint64_t address_space = 0;
    std::shared_ptr<ResourceMemoryBackend> backend;
    CoherenceMode mode = CoherenceMode::SubmissionBoundary;
    std::uint64_t cpu_tracking_granule = 4096;
    std::shared_ptr<ResourceMemoryWorkCounters> work_counters{};
    // Track native guest CPU stores through the process write tracker while
    // backings are registered: registered mappings become protected host
    // ranges, cpu_write_stamp()/cpu_written_since() answer from page write
    // generations, and boundaries re-arm protection. Without it the registry
    // keeps the conservative contract defaults.
    bool cpu_write_tracking = false;
    // Mapping guest addresses are host pointers to the backing bytes (native
    // guest execution); backing_host_range() reports them only then.
    bool host_addressable_backings = false;
};

// TrackedNativeAccess is rejected until L3 exists. All registry calls, borrowed
// address-space access and transaction commit/destruction belong to a serialized
// coordinator. GPU feedback/token polling may run concurrently. Transactions and
// reservations retain registry state. Every accepted submission must also own
// the registry through GPU quiescence and coordinator finish (for example via
// CompletionActions, captured before constructing SubmissionCompletion). A token
// alone does not own the registry. Failed retirement must transfer the registry
// or its authoritative storage to recovery ownership independent of the retiring
// completion before releasing that submission owner.
// The borrowed address-space view must not
// survive registry destruction or a mapping mutation. Pending overlaps fail
// explicitly rather than blocking the coordinator on its own transactions.
// Committed Accepted intents also return an owned Stopped error from operations
// that need to wait: submit the work, then retry. Commit and submission remain
// separate; neither CPU access nor a submission boundary submits work implicitly.
// CPU write preservation uses the registered absolute guest backing base and
// clips native granules to the backing bounds. A granule larger than one byte
// requires a mapping to establish that base before preparing a CPU write.
//
// commit(storage) takes exactly one nonnull lease per request, in request order.
// Write/ReadWrite leases must implement ResourceMemoryStorage. Read leases pin
// already prepared contents. Dependencies must be honored before GPU execution;
// the caller prepares uploaded/converted contents before commit. Overlapping
// writes within one reservation are rejected as ambiguous storage authority.
[[nodiscard]] RendererResult<std::shared_ptr<ResourceMemoryRegistry>>
make_resource_memory(ResourceMemoryOptions options);

} // namespace armigo::graphics
