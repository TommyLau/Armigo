// ============================================================================
// Armigo Graphics - Backing and Coherence Registry
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// Backing-relative interval splitting, GPU-newer upload exclusion, and
// pre-access preservation use retained leases. The optional CPU write stamp
// tracks registered host mappings: CPU reads re-arm page protection and CPU
// writes publish generations. Writable aliases preserve protection when their
// host mapping can be shared.

#include <armigo/graphics/resource_memory_registry.h>

#include <armigo/memory/write_tracker.h>
#include "content_hash.h"
#include "../platform/process_memory.h"

#include <algorithm>
#include <limits>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unistd.h>

namespace armigo::graphics {
namespace {

RendererError error(const char* message) {
    return RendererError({CommandStatus::Stopped, message});
}
bool ok(const RendererStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}
bool valid(BackingId id) { return id.owner && id.allocation && id.generation; }
bool valid(MappingId id) { return id.address_space && id.mapping && id.generation; }
bool valid(ResourceAccess access) {
    return access == ResourceAccess::Read || access == ResourceAccess::Write ||
           access == ResourceAccess::ReadWrite;
}
bool writes(ResourceAccess access) { return access != ResourceAccess::Read; }
bool contains(BackingRange outer, BackingRange inner) {
    return outer.backing == inner.backing && inner.offset >= outer.offset &&
           inner.offset - outer.offset <= outer.size &&
           inner.size <= outer.size - (inner.offset - outer.offset);
}
bool overlaps(BackingRange a, BackingRange b) {
    return a.backing == b.backing && a.offset < b.offset + b.size &&
           b.offset < a.offset + a.size;
}
bool quiescent(const CompletionSnapshot& snapshot) {
    return snapshot.phase != SubmissionPhase::Accepted &&
           snapshot.phase != SubmissionPhase::Submitted;
}
using BackingKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
BackingKey key(BackingId id) { return {id.owner, id.allocation, id.generation}; }
using GenerationKey = std::pair<std::uint64_t, std::uint64_t>;

template <class Key>
struct CountedLess {
    std::uint64_t* comparisons = nullptr;
    bool operator()(const Key& left, const Key& right) const noexcept {
        if (comparisons) ++*comparisons;
        return left < right;
    }
};

struct Segment {
    DirtyRange dirty;
    std::shared_ptr<const ResourceMemoryStorage> storage;
    // Cached immutable token identity keeps coalescing allocation-free.
    SubmissionId writer_id{};
    // A committed GPU access clears CPU dirtiness only in this epoch.
    std::uint64_t cpu_epoch = 0;
};
using Segments = std::map<std::uint64_t, Segment, CountedLess<std::uint64_t>>;
using DirtyOffsets = std::set<std::uint64_t, CountedLess<std::uint64_t>>;
struct Backing {
    BackingRegistration registration;
    std::optional<std::uint64_t> guest_base;
    // While a mapping publishes the backing's CPU identity, its host pages
    // are tracked by the process write tracker. A failed protection makes
    // the stamp contract conservatively answer true for this backing.
    bool cpu_tracked = false;
    bool cpu_tracking_reliable = true;
    // This shared mapping is published only through the tracker registration.
    // Retirement drains alias stores before releasing the native mapping.
    std::optional<platform::ProcessMemoryRegion> writable_alias;
    void* alias_start = nullptr;
    ~Backing() { untrack(); }
    void untrack() noexcept {
        if (cpu_tracked && guest_base)
            memory::WriteTracker::remove_range(reinterpret_cast<const void*>(*guest_base), alias_start);
        cpu_tracked = false;
        cpu_tracking_reliable = true;
        writable_alias.reset();
        alias_start = nullptr;
    }
    explicit Backing(ResourceMemoryWorkCounters* counters = nullptr)
        : segments(CountedLess<std::uint64_t>{counters ? &counters->interval_comparisons : nullptr}),
          dirty_offsets(CountedLess<std::uint64_t>{counters ? &counters->interval_comparisons : nullptr}),
          epoch_edges(CountedLess<std::uint64_t>{counters ? &counters->interval_comparisons : nullptr}) {}
    Segments segments;
    DirtyOffsets dirty_offsets;
    // Only committed GPU accesses create CPU-clean edges that a later epoch
    // can make redundant. Remember those edges, not every historical interval.
    DirtyOffsets epoch_edges;
};
struct MappingEntry {
    MappingRegistration registration;
    std::shared_ptr<Backing> backing;
};
// Exact keys preserve byte ownership; overlapping keys replay in last-use order.
struct JournalKey {
    BackingRange bytes;
    ResourceAccess access;
    const ResourceLease* storage;
    bool operator==(const JournalKey&) const = default;
};
struct JournalHash {
    std::size_t operator()(const JournalKey& key) const noexcept {
        std::size_t hash = 0;
        for (const auto value : {key.bytes.backing.owner, key.bytes.backing.allocation,
             key.bytes.backing.generation, key.bytes.offset, key.bytes.size,
             std::uint64_t(key.access), std::uint64_t(reinterpret_cast<std::uintptr_t>(key.storage))})
            hash ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};
struct JournalEntry {
    GpuAccessRequest request;
    RetainedResource storage;
    CompletionToken token;
    std::uint64_t order = 0;
    std::vector<CompletionToken> dependencies;
};
struct AccessJournal {
    std::weak_ptr<const ResourceLease> batch;
    // Backings with write entries in this batch (a superset after rollbacks).
    std::set<BackingKey> written;
    CompletionToken latest;
    std::uint64_t order = 0;
    std::vector<JournalEntry> entries;
    std::unordered_map<JournalKey, std::size_t, JournalHash> index;
};
struct RegistryIdentity {};
struct State;
struct RangeLease final : ResourceLease {
    std::shared_ptr<const RegistryIdentity> registry;
    std::shared_ptr<MappingEntry> mapping;
    std::shared_ptr<ResourceMemoryBackend> backend;
    BackingRange bytes;
    ResourceAccess access;
};
struct Exclusion {
    std::uint64_t id;
    BackingRange range;
    bool write;
    std::optional<BackingRange> prepared_write{};
    bool deferred_store = false;
};
struct Intent {
    RetainedRange range;
    ResourceAccess access;
    CompletionToken token;
    RetainedResource storage;
    std::uint64_t serial = 0;
};
// One backing's in-flight accesses in acceptance order. Settled entries leave
// from the front without moving the rest until half the storage is free.
struct IntentList {
    std::vector<Intent> items;
    std::size_t head = 0;
    // The queue of the first accepted intent and whether any other queue's
    // intents followed: single-queue lists answer dependencies newest first.
    std::uint64_t queue = 0;
    bool mixed_queues = false;
    auto begin() const { return items.begin() + static_cast<std::ptrdiff_t>(head); }
    auto end() const { return items.end(); }
    bool empty() const { return head == items.size(); }
    std::size_t size() const { return items.size() - head; }
    const Intent& front() const { return items[head]; }
    void pop_front() {
        items[head++] = Intent{};
        if (head >= 64 && 2 * head >= items.size()) compact();
    }
    void compact() {
        items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
    }
};
struct Pending {
    std::vector<GpuAccessRequest> requests;
    CompletionToken token;
};

struct State final : ResourceLease {
    const std::shared_ptr<const RegistryIdentity> identity = std::make_shared<RegistryIdentity>();
    explicit State(ResourceMemoryOptions value)
        : options(std::move(value)),
          backings(CountedLess<BackingKey>{options.work_counters ? &options.work_counters->backing_comparisons : nullptr}),
          gpu_mappings(CountedLess<std::uint64_t>{options.work_counters ? &options.work_counters->mapping_comparisons : nullptr}),
          cpu_backings(CountedLess<std::uint64_t>{options.work_counters ? &options.work_counters->backing_comparisons : nullptr}),
          dirty_backings(CountedLess<BackingKey>{options.work_counters ? &options.work_counters->backing_comparisons : nullptr}),
          epoch_backings(CountedLess<BackingKey>{options.work_counters ? &options.work_counters->backing_comparisons : nullptr}),
          mappings(CountedLess<std::uint64_t>{options.work_counters ? &options.work_counters->mapping_comparisons : nullptr}) {}
    ~State() {
        for (auto& [unused, backing] : backings) backing->untrack();
    }
    std::optional<AccessJournal> journal;
    RendererStatus flush_journal();
    ResourceMemoryOptions options;
    std::map<BackingKey, std::shared_ptr<Backing>, CountedLess<BackingKey>> backings;
    // Non-owning indices preserve the pin counts used by unmap and retirement.
    std::map<std::uint64_t, MappingEntry*, CountedLess<std::uint64_t>> gpu_mappings;
    std::map<std::uint64_t, Backing*, CountedLess<std::uint64_t>> cpu_backings;
    std::map<BackingKey, Backing*, CountedLess<BackingKey>> dirty_backings;
    std::map<BackingKey, Backing*, CountedLess<BackingKey>> epoch_backings;
    std::map<std::uint64_t, std::shared_ptr<MappingEntry>, CountedLess<std::uint64_t>> mappings;
    std::map<GenerationKey, std::uint64_t> backing_generations;
    std::map<std::uint64_t, std::uint64_t> mapping_generations;
    std::map<std::uint64_t, std::uint64_t> queue_sequences;
    std::vector<Exclusion> exclusions;
    // Exclusions that are deferred host stores: checks for one return at
    // once while there are none (they run on every stamp check).
    std::size_t deferred_stores = 0;
    std::map<std::uint64_t, Pending> pending;
    // In-flight GPU accesses grouped by backing, so overlap queries visit only
    // the queried backing. `intent_order` lists every accepted intent by
    // serial across backings; entries a full sweep removed are skipped.
    // Keyed by backing and whether the intents write: reads conflict only
    // with writes, so a read scans just its backing's write list.
    using IntentKey = std::pair<BackingKey, bool>;
    std::map<IntentKey, IntentList> intents;
    std::vector<std::pair<std::uint64_t, IntentKey>> intent_order;
    std::size_t intent_order_head = 0, intent_count = 0;
    std::uint64_t intent_serial = 0;
    // Guest bytes under GPU-resident authority that guest boundaries left in
    // place: a write stamp and a content hash of the (stale) guest bytes. A
    // page-granular CPU write report is confirmed by rehashing; registry
    // writes into a range drop its record.
    struct ResidentCheck {
        std::uint64_t stamp = 0, hash = 0;
    };
    std::map<std::tuple<BackingKey, std::uint64_t, std::uint64_t>, ResidentCheck> resident_checks;
    void forget_resident(BackingRange range) {
        std::erase_if(resident_checks, [&](const auto& entry) {
            const auto& [backing, offset, size] = entry.first;
            return backing == key(range.backing) && offset < range.offset + range.size &&
                   range.offset < offset + size;
        });
    }
    // Intent count left by the last full prune sweep.
    std::size_t swept_intents = 0;
    std::optional<RendererError> boundary_error;
    std::uint64_t operation = 0, cpu_version = 0, gpu_version = 0, epoch = 1;

    // Recent backing lookups without the map or a shared_ptr copy; any
    // insertion or erasure of a backing bumps the version and empties it.
    std::uint64_t backings_version = 1;
    mutable std::array<std::pair<BackingId, Backing*>, 16> recent_backings{};
    mutable std::uint64_t recent_version = 0;
    const Backing* find_raw(BackingRange range) const {
        if (recent_version != backings_version) {
            recent_backings = {};
            recent_version = backings_version;
        }
        auto& slot = recent_backings[(range.backing.allocation ^ range.backing.owner) & 15];
        Backing* backing = slot.first == range.backing ? slot.second : nullptr;
        if (!backing) {
            const auto it = backings.find(key(range.backing));
            if (it == backings.end()) return nullptr;
            backing = it->second.get();
            slot = {range.backing, backing};
        }
        if (range.size == 0 || range.offset > backing->registration.size ||
            range.size > backing->registration.size - range.offset) return nullptr;
        return backing;
    }
    std::shared_ptr<Backing> find(BackingRange range) const {
        const auto it = backings.find(key(range.backing));
        if (it == backings.end() || range.size == 0 ||
            range.offset > it->second->registration.size ||
            range.size > it->second->registration.size - range.offset) return {};
        return it->second;
    }
    // Host pages of a tracked backing; valid only while cpu_tracked holds.
    static const void* tracked_host(const Backing& backing,
                                    std::uint64_t offset) noexcept {
        return reinterpret_cast<const void*>(*backing.guest_base + offset);
    }
    // Best-effort re-arm of write protection for a CPU read. Caches keep the
    // bytes copied through a read transaction, so every store after the copy
    // must be observable; a failed re-arm disqualifies the backing's stamps.
    void rearm_cpu_read(Backing& backing, BackingRange range) noexcept {
        if (!options.cpu_write_tracking || !backing.cpu_tracked || !backing.guest_base)
            return;
        if (!memory::WriteTracker::protect_checked(tracked_host(backing, range.offset),
                                                   range.size))
            backing.cpu_tracking_reliable = false;
    }
    // Registry-issued CPU writes must not fault on protected pages and must
    // publish a fresh write generation before the bytes move.
    void prepare_registry_write(const Backing& backing, BackingRange range) const noexcept {
        if (!options.cpu_write_tracking || !backing.cpu_tracked || !backing.guest_base)
            return;
        memory::WriteTracker::prepare_host_write(tracked_host(backing, range.offset),
                                                 range.size);
    }
    // Internal copies can use the alias; external write transactions still
    // prepare the original address because their caller chooses the writer.
    RendererStatus write_cpu(const Backing& backing, BackingRange range,
                             std::span<const std::byte> source) {
        if (options.host_addressable_backings && backing.cpu_tracked && backing.guest_base &&
            memory::WriteTracker::try_alias_store(tracked_host(backing, range.offset), source))
            return renderer_success();
        prepare_registry_write(backing, range);
        return options.backend->write_cpu(range, source);
    }
    // Per-interval CPU dirtiness is derived lazily from this epoch.
    RendererStatus advance_epoch() {
        if (epoch == std::numeric_limits<std::uint64_t>::max())
            return error("CPU cache epoch is exhausted");
        ++epoch;
        // Only these touched edges can become equivalent solely because
        // the epoch advanced. Compact them without scanning clean backings
        // or letting historical read-only submissions accumulate splits.
        for (const auto& [unused, backing] : epoch_backings) {
            for (const auto edge : backing->epoch_edges) {
                if (options.work_counters) ++options.work_counters->boundary_epoch_edges;
                coalesce(*backing, {backing->registration.id, edge, 0});
            }
            backing->epoch_edges.clear();
        }
        epoch_backings.clear();
        return renderer_success();
    }
    bool mapping_retained(const Backing* backing) const noexcept {
        for (const auto& [unused, entry] : mappings)
            if (entry->backing.get() == backing) return true;
        return false;
    }
    // Removes tracking with the registry's own authority. Backings registered
    // with the registry are host ranges; they are tracked only while the
    // registry holds their CPU identity.
    void untrack_backing(Backing& backing) noexcept {
        backing.untrack();
    }
    // Drops intents whose submissions reached quiescence. Hot paths trim only
    // the settled prefix: one queue completes in order, so the scan stops at
    // the first pending intent instead of locking every token per call. A
    // long-lived pending intent can hold completed ones (and their storage)
    // behind it, so the list also gets a full sweep whenever it doubles past
    // its size after the previous sweep. Boundaries always sweep.
    void prune(bool full = false) {
        if (!full) {
            while (intent_order_head < intent_order.size()) {
                const auto& [serial, backing] = intent_order[intent_order_head];
                const auto list = intents.find(backing);
                if (list != intents.end() && !list->second.empty() && list->second.front().serial == serial) {
                    const auto snapshot = list->second.front().token.snapshot();
                    if (snapshot.error && !boundary_error) boundary_error = snapshot.error;
                    if (!quiescent(snapshot)) break;
                    list->second.pop_front();
                    --intent_count;
                    if (list->second.empty()) intents.erase(list);
                }
                ++intent_order_head;
            }
            if (intent_order_head >= 256 && 2 * intent_order_head >= intent_order.size()) {
                intent_order.erase(intent_order.begin(),
                                   intent_order.begin() + static_cast<std::ptrdiff_t>(intent_order_head));
                intent_order_head = 0;
            }
            if (intent_count <= std::max<std::size_t>(256, 2 * swept_intents)) return;
        }
        intent_count = 0;
        for (auto list = intents.begin(); list != intents.end();) {
            list->second.compact();
            std::erase_if(list->second.items, [&](const Intent& intent) {
                const auto snapshot = intent.token.snapshot();
                if (snapshot.error && !boundary_error) boundary_error = snapshot.error;
                return quiescent(snapshot);
            });
            intent_count += list->second.size();
            list = list->second.empty() ? intents.erase(list) : std::next(list);
        }
        intent_order.clear();
        intent_order_head = 0;
        for (const auto& [backing, list] : intents)
            for (const auto& intent : list) intent_order.emplace_back(intent.serial, backing);
        std::sort(intent_order.begin(), intent_order.end());
        swept_intents = intent_count;
    }
    const IntentList* intents_of(BackingId backing, bool write) const {
        const auto found = intents.find({key(backing), write});
        return found == intents.end() ? nullptr : &found->second;
    }
    bool excluded(BackingRange range, bool write, std::uint64_t except = 0) const {
        return std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& e) {
            return e.id != except && (write || e.write) && overlaps(range, e.range);
        });
    }
    bool deferred_store(BackingRange range) const noexcept {
        if (!deferred_stores) return false;
        return std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& exclusion) {
            return exclusion.deferred_store && overlaps(range, exclusion.range);
        });
    }
    void visit_interval() const noexcept {
        if (options.work_counters) ++options.work_counters->interval_visits;
    }
    // GPU-dirty intervals stay GPU-authoritative across internal boundaries:
    // the epoch alone never marks them CPU-modified.
    bool cpu_dirty(const Segment& segment) const noexcept {
        return segment.dirty.cpu_dirty || (segment.cpu_epoch != epoch && !segment.dirty.gpu_dirty);
    }
    bool equivalent(const Segment& left, const Segment& right) const noexcept {
        return left.dirty.version == right.dirty.version &&
            cpu_dirty(left) == cpu_dirty(right) &&
            left.dirty.gpu_dirty == right.dirty.gpu_dirty &&
            left.writer_id == right.writer_id && left.storage == right.storage;
    }
    static auto first_segment(Backing& backing, std::uint64_t offset) {
        auto it = backing.segments.upper_bound(offset);
        return it == backing.segments.begin() ? it : std::prev(it);
    }
    bool protected_boundary(BackingId backing, std::uint64_t boundary) const noexcept {
        return std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& exclusion) {
            if (!exclusion.prepared_write) return false;
            const auto& range = *exclusion.prepared_write;
            return range.backing == backing &&
                (boundary == range.offset || boundary == range.offset + range.size);
        });
    }
    void coalesce(Backing& backing, BackingRange range) noexcept {
        auto previous = first_segment(backing, range.offset);
        if (previous == backing.segments.end()) return;
        if (previous != backing.segments.begin()) --previous;
        auto next = std::next(previous);
        const auto end = range.offset + range.size;
        while (next != backing.segments.end() && next->first <= end) {
            visit_interval();
            auto& left = previous->second;
            const auto& right = next->second;
            if (!protected_boundary(backing.registration.id, next->first) &&
                equivalent(left, right)) {
                left.dirty.cpu_dirty = cpu_dirty(left);
                left.cpu_epoch = epoch;
                left.dirty.bytes.size += right.dirty.bytes.size;
                if (right.dirty.gpu_dirty) backing.dirty_offsets.erase(next->first);
                next = backing.segments.erase(next);
            } else {
                previous = next++;
            }
        }
    }
    struct CoalesceOnExit {
        State& state;
        Backing& backing;
        BackingRange range;
        ~CoalesceOnExit() {
            state.coalesce(backing, {range.backing, range.offset, 0});
            state.coalesce(backing, {range.backing, range.offset + range.size, 0});
        }
    };
    void release(std::uint64_t id) noexcept {
        for (auto it = exclusions.begin(); it != exclusions.end();) {
            if (it->id != id) { ++it; continue; }
            const auto range = it->prepared_write.value_or(it->range);
            const bool changed = it->prepared_write.has_value() || pending.contains(id);
            auto backing = changed ? find(range) : nullptr;
            if (it->deferred_store) --deferred_stores;
            it = exclusions.erase(it);
            if (backing) coalesce(*backing, range);
        }
        pending.erase(id);
    }
    RendererStatus wait(CompletionToken token) {
        auto snapshot = token.snapshot();
        if (snapshot.phase == SubmissionPhase::Accepted)
            return error("GPU access is not submitted; submit before retrying the wait");
        if (!quiescent(snapshot)) {
            auto result = options.backend->wait_gpu(token);
            if (!ok(result)) return result;
            snapshot = token.snapshot();
        }
        if (snapshot.error) return *snapshot.error;
        if (!quiescent(snapshot)) return error("GPU feedback is not available");
        return renderer_success();
    }
    RendererStatus wait_accesses(BackingRange range, bool write) {
        auto flushed = flush_journal();
        if (!ok(flushed)) return flushed;
        // Newest first: on one queue completing the latest conflicting
        // intent completes every earlier one. Reads wait for writers only.
        for (const bool writers : {true, false}) {
            if (!writers && !write) break;
            const auto* list = intents_of(range.backing, writers);
            if (!list) continue;
            for (auto it = list->items.rbegin();
                 it != list->items.rend() - std::ptrdiff_t(list->head); ++it) {
                if (!overlaps(range, it->range.bytes)) continue;
                auto result = wait(it->token);
                if (!ok(result)) return result;
                if (!list->mixed_queues) break;
            }
        }
        prune();
        return renderer_success();
    }
    static void split(Backing& backing, std::uint64_t point) {
        auto it = first_segment(backing, point);
        if (it == backing.segments.end()) return;
        auto& range = it->second.dirty.bytes;
        if (point <= range.offset || point - range.offset >= range.size) return;
        auto right = it->second;
        right.dirty.bytes.size -= point - range.offset;
        right.dirty.bytes.offset = point;
        // Allocate both index entries before changing the left interval.
        const auto inserted = backing.segments.emplace(point, std::move(right)).first;
        try {
            if (inserted->second.dirty.gpu_dirty) backing.dirty_offsets.insert(point);
        } catch (...) {
            backing.segments.erase(inserted);
            throw;
        }
        range.size = point - range.offset;
    }
    static void split(Backing& backing, BackingRange range) {
        split(backing, range.offset);
        split(backing, range.offset + range.size);
    }
    RendererStatus resolve(BackingRange range) {
        auto backing = find(range);
        if (!backing) return error("resolve range is invalid or retired");
        if (excluded(range, false)) return error("resolve overlaps a pending writer");
        auto ready = wait_accesses(range, false);
        if (!ok(ready)) return ready;
        auto dirty = backing->dirty_offsets.lower_bound(range.offset);
        if (dirty != backing->dirty_offsets.begin()) {
            const auto previous = std::prev(dirty);
            const auto& bytes = backing->segments.at(*previous).dirty.bytes;
            if (bytes.offset + bytes.size > range.offset) dirty = previous;
        }
        if (dirty == backing->dirty_offsets.end() || *dirty >= range.offset + range.size)
            return renderer_success();
        if (excluded(range, true)) return error("writeback overlaps a pending reader");
        ready = wait_accesses(range, true);
        if (!ok(ready)) return ready;
        CoalesceOnExit cleanup{*this, *backing, range};
        split(*backing, range);
        struct Copy { std::uint64_t offset; std::vector<std::byte> bytes; };
        std::vector<Copy> copies;
        std::vector<std::uint64_t> in_place;
        for (auto it = backing->dirty_offsets.lower_bound(range.offset);
             it != backing->dirty_offsets.end() && *it < range.offset + range.size; ++it) {
            visit_interval();
            const auto& segment = backing->segments.at(*it);
            ready = wait(segment.dirty.last_writer);
            if (!ok(ready)) return ready;
            if (!segment.storage) return error("GPU-dirty interval has no authority");
            // A zero-copy authority wrote guest memory directly.
            if (segment.storage->in_place()) {
                in_place.push_back(*it);
                continue;
            }
            Copy copy{*it, std::vector<std::byte>(segment.dirty.bytes.size)};
            ready = segment.storage->read_gpu(segment.dirty.bytes, copy.bytes);
            if (!ok(ready)) return ready;
            copies.push_back(std::move(copy));
        }
        // Keep every dirty bit and authority until the entire requested writeback
        // succeeds, including when the raw writer has copied only a prefix.
        // Only copied bytes are a registry CPU write; in-place results are not.
        if (!copies.empty()) {
            if (!resident_checks.empty()) forget_resident(range);
        }
        for (const auto& copy : copies) {
            ready = write_cpu(*backing, backing->segments.at(copy.offset).dirty.bytes, copy.bytes);
            if (!ok(ready)) return ready;
        }
        for (const auto& copy : copies) {
            auto& segment = backing->segments.at(copy.offset);
            segment.dirty.gpu_dirty = false;
            segment.storage.reset();
            backing->dirty_offsets.erase(copy.offset);
        }
        std::vector<BackingRange> settled;
        for (const auto offset : in_place) {
            auto& segment = backing->segments.at(offset);
            settled.push_back(segment.dirty.bytes);
            segment.dirty.gpu_dirty = false;
            segment.storage.reset();
            backing->dirty_offsets.erase(offset);
        }
        if (backing->dirty_offsets.empty()) dirty_backings.erase(key(range.backing));
        for (const auto& copy : copies)
            coalesce(*backing, {range.backing, copy.offset, copy.bytes.size()});
        for (const auto& bytes : settled) coalesce(*backing, bytes);
        return renderer_success();
    }

};

// Public operations contain backend and allocation exceptions. Transaction
// commit only touches pre-split intervals and never allocates.
template<class F> auto guarded(F&& call) -> decltype(call()) {
    try { return call(); }
    catch (...) { return error("resource memory operation failed with an exception"); }
}

class CapturedGuestBoundary final : public GuestBoundary {
public:
    struct Authority {
        std::shared_ptr<Backing> backing;
        Segment segment;
    };
    CapturedGuestBoundary(std::shared_ptr<State> state, std::vector<Authority> authorities,
                          std::vector<CompletionToken> accesses, bool preserve_resident)
        : state_(std::move(state)), authorities_(std::move(authorities)),
          accesses_(std::move(accesses)), preserve_resident_(preserve_resident) {}

    RendererStatus publish() override {
        return guarded([&]() -> RendererStatus {
            if (published_) return renderer_success();
            // Do not prune or flush the registry here: either could observe a
            // later access which the coordinator deliberately has not run.
            for (const auto& token : accesses_) {
                auto ready = state_->wait(token);
                if (!ok(ready)) return ready;
            }
            if (state_->epoch == std::numeric_limits<std::uint64_t>::max())
                return error("CPU cache epoch is exhausted");
            struct Copy { const Authority* authority; std::vector<std::byte> bytes; };
            std::vector<Copy> copies;
            std::vector<const Authority*> settled;
            for (const auto& authority : authorities_) {
                const auto& segment = authority.segment;
                const auto range = segment.dirty.bytes;
                // Pinning must not turn an explicitly discarded allocation
                // into permission to write reused guest memory.
                if (state_->find_raw(range) != authority.backing.get())
                    return error("captured guest boundary backing was retired");
                auto ready = state_->wait(segment.dirty.last_writer);
                if (!ok(ready)) return ready;
                if (preserve_resident_ && !segment.storage->guest_boundary_writeback()) {
                    const auto stamp = state_->options.cpu_write_tracking
                        ? memory::WriteTracker::now() : 0;
                    auto& check = state_->resident_checks[
                        {key(range.backing), range.offset, range.size}];
                    const auto& backing = *authority.backing;
                    const bool unchanged = check.stamp && state_->options.cpu_write_tracking &&
                        backing.cpu_tracked && backing.cpu_tracking_reliable && backing.guest_base &&
                        !memory::WriteTracker::written_since(
                            State::tracked_host(backing, range.offset), range.size, check.stamp);
                    if (!unchanged) {
                        std::vector<std::byte> guest(range.size);
                        ready = state_->options.backend->read_cpu(range, guest);
                        if (!ok(ready)) return ready;
                        const auto hash = content_hash64(guest);
                        if (check.stamp && hash != check.hash)
                            return error("native CPU store reached GPU-resident texture memory after a guest boundary");
                        check = {stamp ? stamp : 1, hash};
                    }
                    continue;
                }
                settled.push_back(&authority);
                if (!segment.storage->in_place()) {
                    Copy copy{&authority, std::vector<std::byte>(range.size)};
                    ready = segment.storage->read_gpu(range, copy.bytes);
                    if (!ok(ready)) return ready;
                    copies.push_back(std::move(copy));
                }
            }
            // Allocate interval edges before touching guest bytes. Failed
            // readback or a partially successful CPU write retains authority.
            for (const auto* authority : settled)
                State::split(*authority->backing, authority->segment.dirty.bytes);
            for (const auto& copy : copies) {
                const auto range = copy.authority->segment.dirty.bytes;
                state_->forget_resident(range);
                auto ready = state_->write_cpu(*copy.authority->backing, range, copy.bytes);
                if (!ok(ready)) return ready;
            }
            for (const auto* authority : settled) {
                auto& backing = *authority->backing;
                const auto& captured = authority->segment;
                const auto range = captured.dirty.bytes;
                for (auto it = backing.segments.lower_bound(range.offset);
                     it != backing.segments.end() && it->first < range.offset + range.size; ++it) {
                    auto& current = it->second;
                    if (!current.dirty.gpu_dirty || current.dirty.version != captured.dirty.version ||
                        current.writer_id != captured.writer_id || current.storage != captured.storage)
                        continue;
                    current.dirty.gpu_dirty = false;
                    current.storage.reset();
                    backing.dirty_offsets.erase(it->first);
                }
                if (backing.dirty_offsets.empty()) state_->dirty_backings.erase(key(range.backing));
                state_->coalesce(backing, range);
            }
            auto result = state_->advance_epoch();
            if (!ok(result)) return result;
            published_ = true;
            authorities_.clear();
            accesses_.clear();
            return renderer_success();
        });
    }
private:
    std::shared_ptr<State> state_;
    std::vector<Authority> authorities_;
    std::vector<CompletionToken> accesses_;
    bool preserve_resident_ = false;
    bool published_ = false;
};

class DeferredHostStore final : public ResourceLease {
public:
    DeferredHostStore(std::shared_ptr<State> state, std::uint64_t id,
                      std::shared_ptr<Backing> backing)
        : state_(std::move(state)), id_(id), backing_(std::move(backing)) {}
    ~DeferredHostStore() override { state_->release(id_); }
private:
    std::shared_ptr<State> state_;
    std::uint64_t id_;
    std::shared_ptr<Backing> backing_;
};

class ReadTransaction final : public CpuReadTransaction {
public:
    ReadTransaction(std::shared_ptr<State> state, std::uint64_t id,
                    std::shared_ptr<Backing> backing)
        : state_(std::move(state)), id_(id), backing_(std::move(backing)) {}
    ~ReadTransaction() override { state_->release(id_); }
private:
    std::shared_ptr<State> state_;
    std::uint64_t id_;
    std::shared_ptr<Backing> backing_;
};
class WriteTransaction final : public CpuWriteTransaction {
public:
    WriteTransaction(std::shared_ptr<State> state, std::uint64_t id,
                     std::shared_ptr<Backing> backing, BackingRange range,
                     std::uint64_t version)
        : state_(std::move(state)), id_(id), backing_(std::move(backing)),
          range_(range), version_(version) {}
    ~WriteTransaction() override { state_->release(id_); }
    void commit() noexcept override {
        if (committed_) return;
        for (auto it = backing_->segments.lower_bound(range_.offset);
             it != backing_->segments.end() && it->first < range_.offset + range_.size; ++it) {
            state_->visit_interval();
            auto& segment = it->second;
            segment.dirty.version.cpu_write = version_;
            segment.dirty.cpu_dirty = true;
            segment.cpu_epoch = state_->epoch;
            segment.dirty.gpu_dirty = false;
            segment.dirty.last_writer = {};
            segment.writer_id = {};
            segment.storage.reset();
        }
        committed_ = true;
        state_->release(id_);
    }
private:
    std::shared_ptr<State> state_;
    std::uint64_t id_;
    std::shared_ptr<Backing> backing_;
    BackingRange range_;
    std::uint64_t version_;
    bool committed_ = false;
};

// Both acceptance routes publish the same staged intervals and reader intents.
// All fallible allocation precedes publication; the caller owns cancellation.
RendererStatus publish_accesses(State& state, std::span<const GpuAccessRequest> requests,
    CompletionToken consumer, std::span<const CompletionToken> dependencies,
    std::span<const RetainedResource> storage, std::uint64_t reservation_id = 0,
    std::span<const CompletionToken> entry_tokens = {}) {
    const auto snapshot = consumer.snapshot();
    if (snapshot.phase != SubmissionPhase::Accepted || snapshot.error)
        return error("reservation token is no longer awaiting submission");
    if (snapshot.id.sequence <= state.queue_sequences[snapshot.id.queue])
        return error("reservation commit is out of queue order");
    for (const auto& [other_id, other] : state.pending) {
        const auto other_submission = other.token.id();
        if (other_id != reservation_id && other_submission.queue == snapshot.id.queue &&
            other_submission.sequence < snapshot.id.sequence)
            return error("earlier queue reservation must commit or cancel first");
    }
    if (storage.size() != requests.size() ||
        (!entry_tokens.empty() && entry_tokens.size() != requests.size()))
        return error("storage or token count does not match access requests");
    for (const auto& dependency : dependencies) {
        if (auto failure = dependency.snapshot().error) return *failure;
    }
    std::vector<std::shared_ptr<const ResourceMemoryStorage>> authorities(storage.size());
    for (std::size_t i = 0; i < storage.size(); ++i) {
        if (!storage[i]) return error("access storage is not retained");
        if (writes(requests[i].access)) {
            authorities[i] = std::dynamic_pointer_cast<const ResourceMemoryStorage>(storage[i]);
            if (!authorities[i] || !state.find(authorities[i]->coverage()) ||
                !contains(authorities[i]->coverage(), requests[i].range.bytes))
                return error("write storage does not cover the requested backing interval");
        }
    }
    if (state.gpu_version == std::numeric_limits<std::uint64_t>::max())
        return error("GPU resource version is exhausted");
    // Repeated accesses (a draw like the previous one) change no interval
    // shape: a read inside one clean interval of this epoch changes nothing,
    // a write over exactly one interval that is already GPU-dirty updates it
    // in place. Those publish without staging copies; anything else stages.
    const auto grow = [](auto& items, std::size_t extra) {
        if (items.capacity() - items.size() < extra)
            items.reserve(std::max(items.size() + extra, 2 * items.capacity()));
    };
    {
        std::vector<Segment*> targets(requests.size(), nullptr);
        bool repeated = true;
        for (std::size_t i = 0; i < requests.size() && repeated; ++i) {
            const auto& request = requests[i];
            const auto backing = state.find(request.range.bytes);
            if (!backing) return error("reserved backing was retired");
            // A GPU read changes no interval: CPU dirtiness and epochs only
            // steer coalescing and inspection, which no cache consumes for
            // reads. It publishes its intent alone.
            if (!writes(request.access)) continue;
            const auto k = key(request.range.bytes.backing);
            const auto segment_it = State::first_segment(*backing, request.range.bytes.offset);
            if (segment_it == backing->segments.end() || !state.epoch_backings.contains(k)) {
                repeated = false;
                break;
            }
            auto& segment = segment_it->second;
            const auto& bytes = segment.dirty.bytes;
            const auto end = request.range.bytes.offset + request.range.bytes.size;
            if (writes(request.access)) {
                repeated = bytes.offset == request.range.bytes.offset && bytes.size == request.range.bytes.size &&
                    segment.dirty.gpu_dirty && backing->dirty_offsets.contains(bytes.offset) &&
                    state.dirty_backings.contains(k) && backing->epoch_edges.contains(bytes.offset) &&
                    backing->epoch_edges.contains(end);
            } else {
                // Inside one interval that needs no change, or exactly one
                // interval whose epoch fields update in place.
                const bool unchanged = bytes.offset <= request.range.bytes.offset &&
                    end <= bytes.offset + bytes.size && !segment.dirty.cpu_dirty &&
                    segment.cpu_epoch == state.epoch;
                const bool exact = bytes.offset == request.range.bytes.offset && bytes.size == request.range.bytes.size;
                repeated = (unchanged || exact) &&
                    backing->epoch_edges.contains(request.range.bytes.offset) &&
                    backing->epoch_edges.contains(end);
            }
            // Two requests of one packet on one interval, one of them a write,
            // keep request order through the staging path.
            if (repeated) {
                for (std::size_t j = 0; j < i && repeated; ++j)
                    if (targets[j] == &segment && (writes(request.access) || writes(requests[j].access)))
                        repeated = false;
            }
            if (!repeated) {
                static std::array<std::uint64_t, 2> reasons{};
                ++reasons[writes(request.access) ? 1 : 0];
                if ([] { static const bool on = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr; return on; }() && (reasons[0] + reasons[1]) % 50000 == 0)
                    armigo::diag::opt_in( "publish_miss read=%llu write=%llu\n",
                                 (unsigned long long)reasons[0], (unsigned long long)reasons[1]);
            }
            targets[i] = &segment;
        }
        {
            static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
            static std::uint64_t fast = 0, slow = 0;
            if (trace && (repeated ? ++fast : ++slow) && (fast + slow) % 50000 == 0)
                armigo::diag::opt_in( "publish fast=%llu staged=%llu\n", (unsigned long long)fast,
                             (unsigned long long)slow);
        }
        if (repeated) {
            std::vector<IntentList*> lists;
            lists.reserve(requests.size());
            for (const auto& request : requests)
                lists.push_back(&state.intents[{key(request.range.bytes.backing), writes(request.access)}]);
            for (std::size_t i = 0; i < requests.size(); ++i)
                grow(lists[i]->items, static_cast<std::size_t>(std::count(lists.begin(), lists.end(), lists[i])));
            grow(state.intent_order, requests.size());
            for (std::size_t i = 0; i < requests.size(); ++i) {
                if (!targets[i]) continue;
                auto& segment = *targets[i];
                segment.dirty.cpu_dirty = false;
                segment.cpu_epoch = state.epoch;
                segment.dirty.version.gpu_write = state.gpu_version + 1;
                segment.dirty.last_writer = entry_tokens.empty() ? consumer : entry_tokens[i];
                segment.writer_id = segment.dirty.last_writer.id();
                segment.storage = authorities[i];
            }
            std::size_t appended = 0;
            for (std::size_t i = 0; i < requests.size(); ++i) {
                // The same access through the same storage as the newest
                // intent of its list takes the newer token: on one queue it
                // completes after the one it replaces.
                if (!lists[i]->empty() && !lists[i]->mixed_queues) {
                    auto& newest = lists[i]->items.back();
                    if (newest.range.bytes == requests[i].range.bytes && newest.access == requests[i].access &&
                        newest.storage == storage[i] && newest.token.id().queue == consumer.id().queue) {
                        newest.token = entry_tokens.empty() ? consumer : entry_tokens[i];
                        newest.range = requests[i].range;
                        continue;
                    }
                }
                ++appended;
                Intent intent{requests[i].range, requests[i].access, entry_tokens.empty() ? consumer : entry_tokens[i], storage[i]};
                intent.serial = ++state.intent_serial;
                state.intent_order.emplace_back(intent.serial,
                    State::IntentKey{key(intent.range.bytes.backing), writes(intent.access)});
                const auto queue = consumer.id().queue;
                if (lists[i]->empty() && !lists[i]->mixed_queues) lists[i]->queue = queue;
                else if (lists[i]->queue != queue) lists[i]->mixed_queues = true;
                lists[i]->items.push_back(std::move(intent));
            }
            state.intent_count += appended;
            ++state.gpu_version;
            state.queue_sequences[snapshot.id.queue] = snapshot.id.sequence;
            return renderer_success();
        }
    }
    // Stage only original intervals touched by this reservation. Their
    // complete extents preserve untouched prefixes/suffixes, and gathering
    // every request first keeps overlapping reads/writes in request order.
    // Publication transfers preallocated nodes and cannot allocate.
    struct Replacement {
        explicit Replacement(ResourceMemoryWorkCounters* counters) : contents(counters) {}
        Backing contents;
        std::vector<std::uint64_t> originals;
    };
    std::map<BackingKey, Replacement> replacements;
    decltype(state.dirty_backings) new_dirty_backings(state.dirty_backings.key_comp());
    decltype(state.epoch_backings) new_epoch_backings(state.epoch_backings.key_comp());
    std::vector<Intent> new_intents;
    new_intents.reserve(storage.size());
    for (const auto& request : requests) {
        auto backing = state.find(request.range.bytes);
        if (!backing) return error("reserved backing was retired");
        // GPU reads publish only their intent (see the in-place path).
        if (!writes(request.access)) continue;
        auto& replacement = replacements.try_emplace(
            key(request.range.bytes.backing), state.options.work_counters.get()).first->second;
        for (auto segment = State::first_segment(*backing, request.range.bytes.offset);
             segment != backing->segments.end() &&
                 segment->first < request.range.bytes.offset + request.range.bytes.size; ++segment) {
            state.visit_interval();
            const auto [unused, inserted] = replacement.contents.segments.insert(*segment);
            if (!inserted) continue;
            replacement.originals.push_back(segment->first);
            if (segment->second.dirty.gpu_dirty)
                replacement.contents.dirty_offsets.insert(segment->first);
        }
    }
    for (std::size_t i = 0; i < storage.size(); ++i) {
        const auto& request = requests[i];
        if (!writes(request.access)) {
            new_intents.push_back({request.range, request.access, entry_tokens.empty() ? consumer : entry_tokens[i], storage[i]});
            continue;
        }
        auto& temporary = replacements.at(key(request.range.bytes.backing)).contents;
        State::split(temporary, request.range.bytes);
        temporary.epoch_edges.insert(request.range.bytes.offset);
        temporary.epoch_edges.insert(request.range.bytes.offset + request.range.bytes.size);
        if (!state.epoch_backings.contains(key(request.range.bytes.backing)))
            new_epoch_backings.emplace(key(request.range.bytes.backing),
                                       state.backings.at(key(request.range.bytes.backing)).get());
        for (auto segment_it = temporary.segments.lower_bound(request.range.bytes.offset);
             segment_it != temporary.segments.end() &&
                 segment_it->first < request.range.bytes.offset + request.range.bytes.size; ++segment_it) {
            state.visit_interval();
            auto& segment = segment_it->second;
            segment.dirty.cpu_dirty = false;
            segment.cpu_epoch = state.epoch;
            if (writes(request.access)) {
                segment.dirty.version.gpu_write = state.gpu_version + 1;
                segment.dirty.gpu_dirty = true;
                segment.dirty.last_writer = entry_tokens.empty() ? consumer : entry_tokens[i];
                segment.writer_id = segment.dirty.last_writer.id();
                segment.storage = authorities[i];
                temporary.dirty_offsets.insert(segment_it->first);
            }
        }
        if (!temporary.dirty_offsets.empty() &&
            !state.dirty_backings.contains(key(request.range.bytes.backing)))
            new_dirty_backings.emplace(key(request.range.bytes.backing),
                                       state.backings.at(key(request.range.bytes.backing)).get());
        new_intents.push_back({request.range, request.access, entry_tokens.empty() ? consumer : entry_tokens[i], storage[i]});
    }
    // Create and reserve before publication so appending the intents
    // cannot fail. Capacity grows geometrically: an exact reservation
    // would move every in-flight intent on each commit.
    std::vector<IntentList*> lists;
    lists.reserve(new_intents.size());
    for (const auto& intent : new_intents)
        lists.push_back(&state.intents[{key(intent.range.bytes.backing), writes(intent.access)}]);
    for (std::size_t i = 0; i < new_intents.size(); ++i)
        grow(lists[i]->items, static_cast<std::size_t>(std::count(lists.begin(), lists.end(), lists[i])));
    grow(state.intent_order, new_intents.size());
    for (auto& [id, replacement] : replacements) {
        auto& backing = *state.backings.at(id);
        for (const auto offset : replacement.originals) {
            backing.segments.erase(offset);
            backing.dirty_offsets.erase(offset);
        }
        backing.segments.merge(replacement.contents.segments);
        backing.dirty_offsets.merge(replacement.contents.dirty_offsets);
        backing.epoch_edges.merge(replacement.contents.epoch_edges);
    }
    state.dirty_backings.merge(new_dirty_backings);
    state.epoch_backings.merge(new_epoch_backings);
    for (std::size_t i = 0; i < new_intents.size(); ++i) {
        auto& intent = new_intents[i];
        intent.serial = ++state.intent_serial;
        state.intent_order.emplace_back(intent.serial,
            State::IntentKey{key(intent.range.bytes.backing), writes(intent.access)});
        const auto queue = intent.token.id().queue;
        if (lists[i]->empty() && !lists[i]->mixed_queues) lists[i]->queue = queue;
        else if (lists[i]->queue != queue) lists[i]->mixed_queues = true;
        lists[i]->items.push_back(std::move(intent));
    }
    state.intent_count += new_intents.size();
    ++state.gpu_version;
    state.queue_sequences[snapshot.id.queue] = snapshot.id.sequence;
    return renderer_success();
}

RendererStatus State::flush_journal() {
    if (!journal || journal->entries.empty()) return renderer_success();
    if (journal->batch.expired())
        return error("access journal batch expired before flush or cancellation");
    for (const auto& entry : journal->entries) {
        if (const auto failure = entry.token.snapshot().error) {
            journal.reset();
            return *failure;
        }
    }
    std::vector<std::size_t> order(journal->entries.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return journal->entries[a].order < journal->entries[b].order;
    });
    std::vector<GpuAccessRequest> requests;
    std::vector<RetainedResource> storage;
    std::vector<CompletionToken> tokens, dependencies;
    for (const auto i : order) {
        const auto& entry = journal->entries[i];
        requests.push_back(entry.request);
        storage.push_back(entry.storage);
        tokens.push_back(entry.token);
        dependencies.insert(dependencies.end(), entry.dependencies.begin(), entry.dependencies.end());
    }
    auto result = publish_accesses(*this, requests, journal->latest, dependencies, storage, 0, tokens);
    if (!ok(result)) return result;
    if (options.work_counters) {
        ++options.work_counters->journal_publications;
        options.work_counters->journal_entries_published += requests.size();
    }
    journal.reset();
    for (const auto& request : requests) coalesce(*find(request.range.bytes), request.range.bytes);
    return renderer_success();
}

class Reservation final : public GpuAccessReservation {
public:
    Reservation(std::shared_ptr<State> state, std::uint64_t id,
                std::vector<CompletionToken> dependencies)
        : state_(std::move(state)), id_(id), dependencies_(std::move(dependencies)) {}
    ~Reservation() override { state_->release(id_); }
    std::vector<CompletionToken> dependencies() const override { return dependencies_; }
    RendererStatus commit(std::vector<RetainedResource> storage) override {
        return guarded([&]() -> RendererStatus {
            auto it = state_->pending.find(id_);
            if (it == state_->pending.end()) return error("reservation is no longer pending");
            const auto& pending = it->second;
            auto status = publish_accesses(*state_, pending.requests, pending.token,
                                           dependencies_, storage, id_);
            if (!ok(status)) return status;
            state_->release(id_);
            return renderer_success();
        });
    }
private:
    std::shared_ptr<State> state_;
    std::uint64_t id_;
    std::vector<CompletionToken> dependencies_;
};

class Registry final : public ResourceMemoryRegistry {
public:
    explicit Registry(ResourceMemoryOptions options)
        : state_(std::make_shared<State>(std::move(options))),
          address_space_(std::get<GpuAddressSpace>(make_address_space({}, {read, write, this}))) {}
    RendererStatus register_backing(BackingRegistration registration) override {
        return guarded([&]() -> RendererStatus {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            if (!valid(registration.id) || !registration.size || !registration.owner)
                return error("backing registration lacks identity, bytes or owner");
            GenerationKey id{registration.id.owner, registration.id.allocation};
            if (registration.id.generation <= state_->backing_generations[id])
                return error("backing generation is stale or reused");
            const auto existing = state_->backings.lower_bound(
                {registration.id.owner, registration.id.allocation, 0});
            if (existing != state_->backings.end() &&
                existing->second->registration.id.owner == registration.id.owner &&
                existing->second->registration.id.allocation == registration.id.allocation)
                return error("previous backing allocation is still registered");
            auto backing = std::make_shared<Backing>(state_->options.work_counters.get());
            backing->registration = registration;
            backing->segments.emplace(0, Segment{{{registration.id, 0, registration.size}, {}, true, false, {}}, {}});
            state_->backings.emplace(key(registration.id), backing);
            ++state_->backings_version;
            state_->backing_generations[id] = registration.id.generation;
            return renderer_success();
        });
    }
    RendererStatus register_mapping(MappingRegistration registration) override {
        return register_mappings({&registration, 1});
    }
    // Publishes each registration in order, stopping at the first failure,
    // and rebuilds the GPU address-space view once for all of them.
    RendererStatus register_mappings(std::span<const MappingRegistration> registrations) override {
        return guarded([&]() -> RendererStatus {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            bool published = false;
            const auto rebuild = [&]() -> RendererStatus {
                if (!published) return renderer_success();
                auto view = make_address_space(mapping_values(), {read, write, this});
                if (const auto* failure = std::get_if<MemoryResult>(&view))
                    return RendererError({CommandStatus::Stopped, failure->detail});
                address_space_ = std::get<GpuAddressSpace>(std::move(view));
                return renderer_success();
            };
            for (const auto& registration : registrations) {
                auto status = register_one(registration);
                if (!ok(status)) {
                    auto rebuilt = rebuild();
                    return ok(rebuilt) ? status : rebuilt;
                }
                published = true;
            }
            return rebuild();
        });
    }
    // Validates one registration against the published mappings (including
    // earlier ones of the same batch) and publishes it without a view rebuild.
    RendererStatus register_one(const MappingRegistration& registration) {
        if (!valid(registration.id) || registration.id.address_space != state_->options.address_space)
            return error("mapping address-space identity is invalid");
        if (state_->mappings.contains(registration.id.mapping) ||
            registration.id.generation <= state_->mapping_generations[registration.id.mapping])
            return error("mapping identity is live or its generation was reused");
        auto found = state_->backings.find(key(registration.backing));
        if (found == state_->backings.end()) return error("mapping backing is not registered");
        auto backing = found->second;
        const auto& mapping = registration.mapping;
        if (mapping.backing_size != backing->registration.size ||
            (backing->guest_base && *backing->guest_base != mapping.guest_address))
            return error("mapping disagrees with backing size or CPU identity");
        if (mapping.backing_size > std::numeric_limits<std::uint64_t>::max() - mapping.guest_address)
            return error("mapping guest backing overflows");
        auto next = state_->cpu_backings.lower_bound(mapping.guest_address);
        if (next != state_->cpu_backings.end() && next->second != backing.get() &&
            next->first < mapping.guest_address + mapping.backing_size)
            return error("CPU backing aliases must use the same backing identity");
        if (next != state_->cpu_backings.begin()) {
            const auto previous = std::prev(next);
            if (previous->second != backing.get() &&
                mapping.guest_address - previous->first < previous->second->registration.size)
                return error("CPU backing aliases must use the same backing identity");
        }
        // The single-mapping view applies the address-space validation; the
        // ordered GPU index then rejects overlap with published neighbours.
        const auto single = make_address_space({&mapping, 1}, {});
        if (const auto* failure = std::get_if<MemoryResult>(&single))
            return RendererError({CommandStatus::Stopped, failure->detail});
        const auto after = state_->gpu_mappings.lower_bound(mapping.gpu_base);
        if ((after != state_->gpu_mappings.end() &&
             after->first < mapping.gpu_base + mapping.mapped_size) ||
            (after != state_->gpu_mappings.begin() &&
             [&] {
                 const auto& prior = std::prev(after)->second->registration.mapping;
                 return prior.gpu_base + prior.mapped_size > mapping.gpu_base;
             }()))
            return RendererError({CommandStatus::Stopped, "GPU address-space mappings overlap"});
        auto entry = std::make_shared<MappingEntry>();
        entry->registration = registration;
        entry->backing = backing;
        // Index allocation must finish before publishing any mapping identity.
        const auto gpu_index = state_->gpu_mappings.emplace(mapping.gpu_base, entry.get()).first;
        bool cpu_inserted = false;
        try {
            cpu_inserted = state_->cpu_backings.emplace(mapping.guest_address, backing.get()).second;
            state_->mappings.emplace(registration.id.mapping, std::move(entry));
        } catch (...) {
            state_->gpu_mappings.erase(gpu_index);
            if (cpu_inserted) state_->cpu_backings.erase(mapping.guest_address);
            throw;
        }
        backing->guest_base = mapping.guest_address;
        state_->mapping_generations[registration.id.mapping] = registration.id.generation;
        if (state_->options.cpu_write_tracking && !backing->cpu_tracked) {
            // Track while registered; overlapping peers share canonical
            // page records, and registration invalidates older stamps.
            // Round only the native alias, not the registered byte ownership.
            // Shared remapping preserves physical identity for Metal no-copy
            // buffers of the original mapping. Tracking uses that guest VA.
            const auto page = platform::native_page_size();
            void* alias = nullptr;
            if (page && state_->options.host_addressable_backings) {
                const auto prefix = mapping.guest_address % page;
                const auto bytes = backing->registration.size;
                const auto maximum = std::numeric_limits<std::uint64_t>::max();
                if (bytes <= maximum - prefix - (page - 1)) {
                    const auto rounded = (bytes + prefix + page - 1) & ~(page - 1);
                    auto shared = platform::ProcessMemoryRegion::writable_host_alias(
                        mapping.guest_address - prefix, rounded);
                    if (auto* region = std::get_if<platform::ProcessMemoryRegion>(&shared)) {
                        backing->writable_alias.emplace(std::move(*region));
                        alias = reinterpret_cast<void*>(backing->writable_alias->address() + prefix);
                    }
                }
            }
            backing->alias_start = alias;
            backing->cpu_tracked = true;
            memory::WriteTracker::add_range(
                reinterpret_cast<const void*>(mapping.guest_address),
                backing->registration.size, alias);
            state_->rearm_cpu_read(*backing,
                                   BackingRange{backing->registration.id, 0,
                                                backing->registration.size});
        }
        return renderer_success();
    }
    RendererStatus unmap(MappingId id) override {
        return guarded([&]() -> RendererStatus {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            state_->prune(true);
            auto it = state_->mappings.find(id.mapping);
            if (it == state_->mappings.end() || it->second->registration.id != id)
                return error("mapping identity is stale or absent");
            if (it->second.use_count() != 1) return error("mapping is pinned by a retained access");
            auto mappings = mapping_values(id.mapping);
            auto view = make_address_space(mappings, {read, write, this});
            const auto backing = it->second->backing;
            state_->gpu_mappings.erase(it->second->registration.mapping.gpu_base);
            state_->mappings.erase(it);
            if (!state_->mapping_retained(backing.get()))
                state_->untrack_backing(*backing);
            address_space_ = std::get<GpuAddressSpace>(std::move(view));
            return renderer_success();
        });
    }
    RendererStatus invalidate_mappings(std::span<const MappingId> removed,
        std::span<const BackingId> discarded) override {
        return guarded([&]() -> RendererStatus {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            std::vector<std::shared_ptr<Backing>> affected;
            auto add = [&](const std::shared_ptr<Backing>& backing) {
                if (std::find(affected.begin(), affected.end(), backing) == affected.end())
                    affected.push_back(backing);
            };
            const auto drops = [&](BackingId id) {
                return std::find(discarded.begin(), discarded.end(), id) != discarded.end();
            };
            const auto removes = [&](MappingId id) {
                return std::find(removed.begin(), removed.end(), id) != removed.end();
            };
            for (const auto id : removed) {
                const auto it = state_->mappings.find(id.mapping);
                if (it == state_->mappings.end() || it->second->registration.id != id)
                    return error("mapping invalidation identity is stale or absent");
                add(it->second->backing);
            }
            for (const auto id : discarded) {
                const auto it = state_->backings.find(key(id));
                if (it == state_->backings.end())
                    return error("discarded backing identity is stale or absent");
                add(it->second);
            }
            // Prepare all allocating lookup changes before resolving or retiring
            // anything. Passive leases keep their old objects but cannot acquire
            // new access once the mapping identity is removed from this table.
            std::vector<Mapping> remaining;
            for (const auto& [unused, entry] : state_->mappings)
                if (!removes(entry->registration.id) && !drops(entry->registration.backing))
                    remaining.push_back(entry->registration.mapping);
            auto view = make_address_space(remaining, {read, write, this});
            if (const auto* failed = std::get_if<MemoryResult>(&view))
                return RendererError({CommandStatus::Stopped, failed->detail});
            for (const auto& backing : affected) {
                const BackingRange range{backing->registration.id, 0, backing->registration.size};
                if (state_->excluded(range, true))
                    return error("mapping invalidation overlaps a pending access");
            }
            for (const auto& backing : affected) {
                const BackingRange range{backing->registration.id, 0, backing->registration.size};
                auto status = state_->wait_accesses(range, true);
                if (!ok(status)) return status;
                if (!drops(range.backing)) {
                    status = state_->resolve(range);
                    if (!ok(status)) return status;
                }
            }
            std::erase_if(state_->mappings, [&](const auto& item) {
                const bool remove = removes(item.second->registration.id) || drops(item.second->registration.backing);
                if (remove) state_->gpu_mappings.erase(item.second->registration.mapping.gpu_base);
                return remove;
            });
            for (const auto& backing : affected) {
                if (!drops(backing->registration.id)) continue;
                // Break authority -> lease -> backing retention even when an
                // external stale lease still pins this retired allocation.
                state_->untrack_backing(*backing);
                backing->segments.clear();
                backing->dirty_offsets.clear();
                backing->epoch_edges.clear();
                state_->dirty_backings.erase(key(backing->registration.id));
                state_->epoch_backings.erase(key(backing->registration.id));
                if (backing->guest_base) state_->cpu_backings.erase(*backing->guest_base);
                state_->backings.erase(key(backing->registration.id));
                ++state_->backings_version;
            }
            for (const auto& backing : affected)
                if (!drops(backing->registration.id) && !state_->mapping_retained(backing.get()))
                    state_->untrack_backing(*backing);
            address_space_ = std::get<GpuAddressSpace>(std::move(view));
            return renderer_success();
        });
    }
    RendererStatus retire_backing(BackingId id) override {
        return guarded([&]() -> RendererStatus {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            state_->prune(true);
            auto it = state_->backings.find(key(id));
            if (it == state_->backings.end()) return error("backing identity is stale or absent");
            if (it->second.use_count() != 1) return error("backing is mapped or pinned");
            if (!it->second->dirty_offsets.empty()) return error("backing retains GPU-dirty authority");
            state_->untrack_backing(*it->second);
            if (it->second->guest_base) state_->cpu_backings.erase(*it->second->guest_base);
            state_->epoch_backings.erase(key(id));
            state_->backings.erase(it);
            ++state_->backings_version;
            return renderer_success();
        });
    }
    RendererResult<RetainedRange> retain(std::uint64_t address, std::uint64_t size,
                                        ResourceAccess access) override {
        return guarded([&]() -> RendererResult<RetainedRange> {
            if (!size || !valid(access) || size > std::numeric_limits<std::uint64_t>::max() - address)
                return error("retained GPU range or access is invalid");
            auto found = state_->gpu_mappings.upper_bound(address);
            if (found != state_->gpu_mappings.begin()) {
                const auto* entry = std::prev(found)->second;
                const auto& mapping = entry->registration.mapping;
                if (address - mapping.gpu_base >= mapping.mapped_size)
                    return error("retained GPU address is unmapped");
                const auto offset = address - mapping.gpu_base;
                if (size > mapping.mapped_size - offset ||
                    offset > mapping.backing_size - mapping.buffer_offset ||
                    size > mapping.backing_size - mapping.buffer_offset - offset)
                {
                    static const bool describe = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr;
                    if (!describe)
                        return RendererError({CommandStatus::Stopped,
                                              "retained access crosses a mapping or backing boundary"});
                    std::ostringstream detail;
                    detail << "retained access crosses a mapping or backing boundary"
                           << " address=0x" << std::hex << address << " size=0x" << size;
                    // Name at most four overlapped mappings: a guest limit far
                    // past its buffer would otherwise list thousands per call.
                    unsigned listed = 0;
                    for (auto overlap_it = std::prev(found);
                         overlap_it != state_->gpu_mappings.end() && overlap_it->first < address + size &&
                         listed++ < 4;
                         ++overlap_it) {
                        const auto* overlap = overlap_it->second;
                        const auto id = overlap->registration.id.mapping;
                        const auto& m = overlap->registration.mapping;
                        const auto backing = overlap->registration.backing;
                        detail << " [mapping=" << std::dec << id << ':'
                               << overlap->registration.id.generation
                               << " gpu=0x" << std::hex << m.gpu_base
                               << " size=0x" << m.mapped_size
                               << " backing=" << std::dec << backing.owner << ':'
                               << backing.allocation << ':' << backing.generation
                               << " offset=0x" << std::hex << m.buffer_offset
                               << " backing_size=0x" << m.backing_size << ']';
                    }
                    return RendererError({CommandStatus::Stopped, detail.str()});
                }
                BackingRange range{entry->registration.backing, mapping.buffer_offset + offset, size};
                auto lease = std::make_shared<RangeLease>();
                lease->registry = state_->identity;
                lease->mapping = state_->mappings.at(entry->registration.id.mapping);
                lease->backend = state_->options.backend;
                lease->bytes = range;
                lease->access = access;
                return RetainedRange{entry->registration.id, range, std::move(lease)};
            }
            return error("retained GPU address is unmapped");
        });
    }
    const GpuAddressSpace& address_space() const noexcept override { return address_space_; }
    CoherenceMode coherence_mode() const noexcept override { return CoherenceMode::SubmissionBoundary; }
    std::optional<std::pair<std::uint64_t, std::uint64_t>> backing_host_range(
        BackingId backing) const noexcept override {
        if (!state_->options.host_addressable_backings) return std::nullopt;
        const auto found = state_->backings.find(key(backing));
        if (found == state_->backings.end() || !found->second->guest_base) return std::nullopt;
        return std::pair{*found->second->guest_base, found->second->registration.size};
    }
    bool has_backing(BackingId backing) const noexcept override {
        return state_->backings.contains(
            BackingKey{backing.owner, backing.allocation, backing.generation});
    }
    std::uint64_t cpu_cache_epoch() const noexcept override { return state_->epoch; }
    // Optional CPU write tracking. The tracker answers from page write
    // generations; without it (or for an untracked backing) the conservative
    // contract defaults from ResourceMemoryRegistry apply.
    std::optional<RetainedRange> retain_backing(BackingRange range, ResourceAccess access) override {
        // Try the mapping that last served this backing first.
        const auto backing_key = key(range.backing);
        if (const auto cached = backing_mapping_.find(backing_key); cached != backing_mapping_.end()) {
            if (const auto entry = state_->gpu_mappings.find(cached->second);
                entry != state_->gpu_mappings.end() && entry->second->registration.backing == range.backing) {
                const auto& mapping = entry->second->registration.mapping;
                if (range.offset >= mapping.buffer_offset) {
                    const auto offset = range.offset - mapping.buffer_offset;
                    if (offset < mapping.mapped_size && range.size <= mapping.mapped_size - offset) {
                        auto retained = retain(mapping.gpu_base + offset, range.size, access);
                        if (auto* value = std::get_if<RetainedRange>(&retained); value && value->bytes == range)
                            return std::move(*value);
                    }
                }
            }
        }
        for (const auto& [gpu_base, entry] : state_->gpu_mappings) {
            if (!(entry->registration.backing == range.backing)) continue;
            const auto& mapping = entry->registration.mapping;
            if (range.offset < mapping.buffer_offset) continue;
            const auto offset = range.offset - mapping.buffer_offset;
            if (offset >= mapping.mapped_size || range.size > mapping.mapped_size - offset) continue;
            auto retained = retain(mapping.gpu_base + offset, range.size, access);
            if (auto* value = std::get_if<RetainedRange>(&retained); value && value->bytes == range) {
                backing_mapping_[backing_key] = gpu_base;
                return std::move(*value);
            }
        }
        return std::nullopt;
    }
    std::map<BackingKey, std::uint64_t> backing_mapping_;
    void prepare_host_store(BackingRange range) noexcept override {
        const auto backing = state_->find(range);
        if (backing) state_->prepare_registry_write(*backing, range);
    }
    std::uint64_t cpu_write_stamp() const noexcept override {
        return state_->options.cpu_write_tracking ? memory::WriteTracker::now() : 0;
    }
    bool cpu_written_since(BackingRange range, std::uint64_t stamp) const noexcept override {
        if (deferred_host_store(range)) return true;
        if (!state_->options.cpu_write_tracking || !stamp) return true;
        const auto* backing = state_->find_raw(range);
        if (!backing || !backing->cpu_tracked || !backing->cpu_tracking_reliable ||
            !backing->guest_base)
            return true;
        return memory::WriteTracker::written_since(
            State::tracked_host(*backing, range.offset), range.size, stamp);
    }
    RendererResult<std::shared_ptr<ResourceLease>> defer_host_store(BackingRange range) override {
        return guarded([&]() -> RendererResult<std::shared_ptr<ResourceLease>> {
            const auto backing = state_->find(range);
            if (!backing) return error("deferred host store range is invalid or retired");
            if (state_->excluded(range, true))
                return error("deferred host store overlaps a pending access");
            if (state_->operation == std::numeric_limits<std::uint64_t>::max())
                return error("access identity is exhausted");
            state_->exclusions.reserve(state_->exclusions.size() + 1);
            const auto id = ++state_->operation;
            auto guard = std::make_shared<DeferredHostStore>(state_, id, backing);
            // Shader-package identities query the process tracker directly.
            // Invalidate even when registry CPU tracking is disabled, and even
            // when the native pages were already writable before reservation.
            if ((state_->options.host_addressable_backings || backing->cpu_tracked) &&
                backing->guest_base) {
                const auto native_page = sysconf(_SC_PAGESIZE);
                if (native_page <= 0) return error("deferred host store has no valid native page size");
                const auto page = static_cast<std::uint64_t>(native_page);
                // A whole-span tracked() query can fail across independently
                // registered pages while each page still has valid shader
                // certificates. Tracker registrations cover rounded host pages.
                for (std::uint64_t offset = 0; offset < range.size;) {
                    const auto address = *backing->guest_base + range.offset + offset;
                    const auto size = std::min(range.size - offset, page - address % page);
                    const auto* host = reinterpret_cast<const void*>(address);
                    if (memory::WriteTracker::tracked(host, size)) {
                        const auto stamp = memory::WriteTracker::now();
                        // Preserve reservation-time invalidation for shader
                        // certificates queried outside this registry. The
                        // eventual store publishes another fresh generation.
                        memory::WriteTracker::note_store(host, size);
                        if (!memory::WriteTracker::written_since(host, size, stamp))
                            return error("deferred host store could not invalidate native write tracking");
                    }
                    offset += size;
                }
            }
            state_->exclusions.push_back({id, range, true, {}, true});
            ++state_->deferred_stores;
            return std::shared_ptr<ResourceLease>(std::move(guard));
        });
    }
    bool deferred_host_store(BackingRange range) const noexcept override {
        return state_->deferred_stores && state_->find_raw(range) && state_->deferred_store(range);
    }
    bool has_deferred_host_store() const noexcept override {
        return state_->deferred_stores != 0;
    }
    RendererResult<std::vector<DirtyRange>> inspect(BackingRange range) override {
        return guarded([&]() -> RendererResult<std::vector<DirtyRange>> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            const auto backing = state_->find(range);
            if (!backing) return error("inspection range is invalid or retired");
            std::vector<DirtyRange> result;
            // Storage readback may inspect authority while resolve owns split
            // edges. Merge the snapshot, never mutate those live intervals.
            const Segment* previous = nullptr;
            for (auto it = State::first_segment(*backing, range.offset);
                 it != backing->segments.end() && it->first < range.offset + range.size; ++it) {
                state_->visit_interval();
                const auto& segment = it->second;
                auto dirty = segment.dirty;
                dirty.cpu_dirty = state_->cpu_dirty(segment);
                const auto end = std::min(range.offset + range.size, dirty.bytes.offset + dirty.bytes.size);
                dirty.bytes.offset = std::max(range.offset, dirty.bytes.offset);
                dirty.bytes.size = end - dirty.bytes.offset;
                if (previous && state_->equivalent(*previous, segment) &&
                    !state_->protected_boundary(range.backing, it->first))
                    result.back().bytes.size += dirty.bytes.size;
                else
                    result.push_back(dirty);
                previous = &segment;
            }
            return result;
        });
    }
    // Like inspect(), but publishes the batch journal only when it holds a
    // write to this backing, and stops at the first GPU-dirty interval.
    RendererResult<bool> gpu_written(BackingRange range) override {
        return guarded([&]() -> RendererResult<bool> {
            if (state_->journal && state_->journal->written.contains(key(range.backing))) {
                auto flushed = state_->flush_journal();
                if (!ok(flushed)) return std::get<RendererError>(flushed);
            }
            auto* backing = const_cast<Backing*>(state_->find_raw(range));
            if (!backing) return error("inspection range is invalid or retired");
            for (auto it = State::first_segment(*backing, range.offset);
                 it != backing->segments.end() && it->first < range.offset + range.size; ++it)
                if (it->second.dirty.gpu_dirty) return true;
            return false;
        });
    }
    RendererResult<bool> read_cpu_backed(BackingRange range, std::span<std::byte> output) override {
        return guarded([&]() -> RendererResult<bool> {
            if (output.size() != range.size || !state_->options.host_addressable_backings) return false;
            if (state_->journal && state_->journal->written.contains(key(range.backing))) {
                auto flushed = state_->flush_journal();
                if (!ok(flushed)) return std::get<RendererError>(flushed);
            }
            auto* backing = const_cast<Backing*>(state_->find_raw(range));
            if (!backing || !backing->guest_base || state_->excluded(range, false)) return false;
            for (auto it = State::first_segment(*backing, range.offset);
                 it != backing->segments.end() && it->first < range.offset + range.size; ++it)
                if (it->second.dirty.gpu_dirty) return false;
            // Stores after the copy must stay observable: re-arm, then copy.
            state_->rearm_cpu_read(*backing, range);
            std::memcpy(output.data(), reinterpret_cast<const void*>(*backing->guest_base + range.offset),
                        output.size());
            return true;
        });
    }
    RendererStatus resolve(BackingRange range, ResolveReason) override {
        return guarded([&] { return state_->resolve(range); });
    }
    RendererResult<std::vector<ResourceStorageRange>> inspect_storage(BackingRange range) override {
        return guarded([&]() -> RendererResult<std::vector<ResourceStorageRange>> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            const auto backing = state_->find(range);
            if (!backing) return error("inspection range is invalid or retired");
            std::vector<ResourceStorageRange> result;
            // Storage readback may inspect authority while resolve owns split
            // edges. Merge the snapshot, never mutate those live intervals.
            const Segment* previous = nullptr;
            for (auto it = State::first_segment(*backing, range.offset);
                 it != backing->segments.end() && it->first < range.offset + range.size; ++it) {
                state_->visit_interval();
                const auto& segment = it->second;
                auto dirty = segment.dirty;
                dirty.cpu_dirty = state_->cpu_dirty(segment);
                const auto end = std::min(range.offset + range.size, dirty.bytes.offset + dirty.bytes.size);
                dirty.bytes.offset = std::max(range.offset, dirty.bytes.offset);
                dirty.bytes.size = end - dirty.bytes.offset;
                if (previous && state_->equivalent(*previous, segment) &&
                    !state_->protected_boundary(range.backing, it->first))
                    result.back().dirty.bytes.size += dirty.bytes.size;
                else
                    result.push_back({dirty, dirty.gpu_dirty ? segment.storage : nullptr});
                previous = &segment;
            }
            return result;
        });
    }
    RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange range, ResolveReason) override {
        return guarded([&]() -> RendererResult<std::unique_ptr<CpuReadTransaction>> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            auto backing = state_->find(range);
            if (!backing) return error("CPU read range is invalid or retired");
            if (state_->excluded(range, false)) return error("CPU read overlaps a pending writer");
            if (state_->operation == std::numeric_limits<std::uint64_t>::max())
                return error("access identity is exhausted");
            auto status = state_->resolve(range);
            if (!ok(status)) return std::get<RendererError>(status);
            // The transaction's consumer keeps the copied bytes; stores after
            // the copy must be observable, so re-arm protection now.
            state_->rearm_cpu_read(*backing, range);
            const auto id = ++state_->operation;
            state_->exclusions.reserve(state_->exclusions.size() + 1);
            auto transaction = std::make_unique<ReadTransaction>(state_, id, backing);
            state_->exclusions.push_back({id, range, false});
            return std::unique_ptr<CpuReadTransaction>(std::move(transaction));
        });
    }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(BackingRange range) override {
        return prepare_cpu_write_impl(range, true);
    }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write_impl(
        BackingRange range, bool external_store) {
        return guarded([&]() -> RendererResult<std::unique_ptr<CpuWriteTransaction>> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            auto backing = state_->find(range);
            if (!backing) return error("CPU write range is invalid or retired");
            auto granule = state_->options.cpu_tracking_granule;
            if (granule > 1 && !backing->guest_base)
                return error("CPU write granule requires a registered guest backing base");
            const auto base = backing->guest_base.value_or(0);
            auto preserved = range;
            preserved.offset -= std::min(range.offset, (base + range.offset) % granule);
            const auto end = range.offset + range.size;
            const auto remainder = (base + end) % granule;
            const auto padding = remainder ? granule - remainder : 0;
            const auto available = backing->registration.size - end;
            preserved.size = end + std::min(padding, available) - preserved.offset;
            if (state_->excluded(preserved, true)) return error("CPU write granule overlaps a pending access");
            if (state_->operation == std::numeric_limits<std::uint64_t>::max() ||
                state_->cpu_version == std::numeric_limits<std::uint64_t>::max())
                return error("CPU access identity or version is exhausted");
            auto status = state_->wait_accesses(preserved, true);
            if (!ok(status)) return std::get<RendererError>(status);
            // External transactions expose the original VA to their caller.
            // Internal writes select the alias only when the bytes move.
            if (external_store) state_->prepare_registry_write(*backing, range);
            status = state_->resolve(preserved);
            if (!ok(status)) return std::get<RendererError>(status);
            State::CoalesceOnExit cleanup{*state_, *backing, range};
            State::split(*backing, range);
            const auto id = ++state_->operation;
            state_->exclusions.reserve(state_->exclusions.size() + 1);
            auto transaction = std::make_unique<WriteTransaction>(state_, id, backing, range,
                                                                 ++state_->cpu_version);
            state_->exclusions.push_back({id, preserved, true, range});
            return std::unique_ptr<CpuWriteTransaction>(std::move(transaction));
        });
    }
    RendererStatus validate_retained(const RetainedRange& range, ResourceAccess access) override {
        return guarded([&]() -> RendererStatus {
            const auto lease = std::dynamic_pointer_cast<const RangeLease>(range.lease);
            if (!valid(access) || !lease || lease->registry != state_->identity ||
                lease->mapping->registration.id != range.mapping || lease->bytes != range.bytes ||
                (lease->access != ResourceAccess::ReadWrite && lease->access != access))
                return error("GPU request has a missing, foreign or mismatched range lease");
            const auto mapping = state_->mappings.find(range.mapping.mapping);
            if (mapping == state_->mappings.end() || mapping->second != lease->mapping)
                return error("GPU request mapping is stale or absent");
            const auto backing = state_->find(range.bytes);
            if (!backing || backing != lease->mapping->backing)
                return error("GPU request backing is retired");
            return renderer_success();
        });
    }
    RendererStatus read_retained(const RetainedRange& retained, BackingRange bytes,
                                std::span<std::byte> destination) override {
        return guarded([&]() -> RendererStatus {
            const auto lease = std::dynamic_pointer_cast<const RangeLease>(retained.lease);
            if (!lease) return error("CPU residency read requires a retained mapping");
            auto status = validate_retained(retained, lease->access);
            if (!ok(status)) return status;
            if (!contains(retained.bytes, bytes) || !bytes.size || destination.size() != bytes.size)
                return error("CPU residency read exceeds the retained range");
            auto prepared = prepare_cpu_read(bytes, ResolveReason::CpuRead);
            if (const auto* failure = std::get_if<RendererError>(&prepared)) return *failure;
            return state_->options.backend->read_cpu(bytes, destination);
        });
    }
    RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) override {
        return guarded([&]() -> RendererResult<std::shared_ptr<GpuAccessReservation>> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            auto checked = access_dependencies(requests, consumer);
            if (const auto* failure = std::get_if<RendererError>(&checked)) return *failure;
            auto dependencies = std::get<std::vector<CompletionToken>>(std::move(checked));
            Pending pending{{requests.begin(), requests.end()}, consumer};
            const auto id = ++state_->operation;
            state_->exclusions.reserve(state_->exclusions.size() + requests.size());
            auto reservation = std::make_shared<Reservation>(state_, id, std::move(dependencies));
            state_->pending.emplace(id, std::move(pending));
            for (const auto& request : requests)
                state_->exclusions.push_back({id, request.range.bytes, writes(request.access)});
            return std::shared_ptr<GpuAccessReservation>(std::move(reservation));
        });
    }
    RendererResult<GpuAccessAcceptance> accept_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer,
        std::vector<RetainedResource> storage) override {
        return guarded([&]() -> RendererResult<GpuAccessAcceptance> {
            auto flushed = state_->flush_journal();
            if (!ok(flushed)) return std::get<RendererError>(flushed);
            auto checked = access_dependencies(requests, consumer);
            if (const auto* failure = std::get_if<RendererError>(&checked)) return *failure;
            GpuAccessAcceptance accepted{
                std::get<std::vector<CompletionToken>>(std::move(checked)), state_};
            auto status = publish_accesses(*state_, requests, consumer,
                                           accepted.dependencies, storage);
            if (const auto* failure = std::get_if<RendererError>(&status)) return *failure;
            // Match reservation release without creating pending exclusions.
            for (const auto& request : requests)
                state_->coalesce(*state_->find(request.range.bytes), request.range.bytes);
            return accepted;
        });
    }
    RendererStatus flush_access_journal() override {
        return guarded([&] { return state_->flush_journal(); });
    }
    void discard_access_journal(const RetainedResource& batch) noexcept override {
        if (state_->journal && state_->journal->batch.lock() == batch) state_->journal.reset();
    }
    RendererResult<GpuAccessAcceptance> accept_batch_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer,
        std::span<const RetainedResource> storage, const RetainedResource& batch) override {
        return guarded([&]() -> RendererResult<GpuAccessAcceptance> {
            const auto snapshot = consumer.snapshot();
            if (!batch || requests.empty() || storage.size() != requests.size() ||
                !consumer.valid() || snapshot.phase != SubmissionPhase::Accepted || snapshot.error ||
                !snapshot.id.queue || !snapshot.id.sequence)
                return error("batch access requires storage and a fresh accepted token");
            if (state_->journal && (state_->journal->batch.lock() != batch ||
                state_->journal->latest.id().queue != snapshot.id.queue)) {
                auto flushed = state_->flush_journal();
                if (!ok(flushed)) return std::get<RendererError>(flushed);
            }
            if (snapshot.id.sequence <= state_->queue_sequences[snapshot.id.queue] ||
                (state_->journal && snapshot.id.sequence <= state_->journal->latest.id().sequence))
                return error("batch access token is out of queue order");
            for (const auto& [unused, pending] : state_->pending)
                if (pending.token.id().queue == snapshot.id.queue &&
                    pending.token.id().sequence <= snapshot.id.sequence)
                    return error("earlier queue reservation must commit or cancel first");

            // New keys are inserted in place and rolled back if the packet is
            // refused (copying the journal per new key was quadratic in the
            // batch). The steady-state path uses only hash lookups and cached
            // dependencies outside this batch. Entries in this batch execute in
            // queue order, so their mutual hazards need no submission
            // dependency. Cross-queue access first publishes the journal and
            // queries the exact registry intervals.
            const bool created = !state_->journal;
            if (created) {
                state_->journal.emplace();
                state_->journal->batch = batch;
            }
            AccessJournal* journal = &*state_->journal;
            struct Rollback {
                State& state;
                AccessJournal& journal;
                std::size_t entries;
                std::vector<JournalKey> inserted;
                bool created, committed = false;
                ~Rollback() {
                    if (committed) return;
                    for (const auto& key : inserted) journal.index.erase(key);
                    journal.entries.erase(journal.entries.begin() + std::ptrdiff_t(entries), journal.entries.end());
                    if (created) state.journal.reset();
                }
            } rollback{*state_, *journal, journal->entries.size(), {}, created};
            std::vector<std::size_t> indices;
            indices.reserve(requests.size());
            std::vector<CompletionToken> dependencies;
            const auto depend = [&](const CompletionToken& token) {
                if (!token.valid()) return;
                for (auto& previous : dependencies) {
                    if (previous.id().queue != token.id().queue) continue;
                    if (previous.id().sequence < token.id().sequence) previous = token;
                    return;
                }
                dependencies.push_back(token);
            };
            for (std::size_t i = 0; i < requests.size(); ++i) {
                const auto& request = requests[i];
                const auto* lease = dynamic_cast<const RangeLease*>(request.range.lease.get());
                if (!valid(request.access) || !lease || lease->registry != state_->identity ||
                    lease->mapping->registration.id != request.range.mapping || lease->bytes != request.range.bytes ||
                    (lease->access != ResourceAccess::ReadWrite && lease->access != request.access))
                    return error("batch request has a foreign or mismatched range lease");
                // Cached journal keys must still honor CPU exclusions created
                // after the key's first admission.
                if (state_->excluded(request.range.bytes, writes(request.access)))
                    return error("batch GPU access overlaps a pending access");
                for (std::size_t j = 0; j < i; ++j)
                    if (writes(request.access) && writes(requests[j].access) &&
                        overlaps(request.range.bytes, requests[j].range.bytes))
                        return error("one packet contains overlapping write authorities");
                const JournalKey key{request.range.bytes, request.access, storage[i].get()};
                auto found = journal->index.find(key);
                if (found == journal->index.end()) {
                    if (!storage[i]) return error("batch access storage is not retained");
                    if (writes(request.access)) {
                        const auto authority = std::dynamic_pointer_cast<const ResourceMemoryStorage>(storage[i]);
                        if (!authority || !state_->find(authority->coverage()) ||
                            !contains(authority->coverage(), request.range.bytes))
                            return error("batch write storage does not cover the requested interval");
                    }
                    auto checked = access_dependencies(requests.subspan(i, 1), consumer);
                    if (const auto* failure = std::get_if<RendererError>(&checked)) return *failure;
                    JournalEntry entry{request, storage[i], {}, 0,
                        std::get<std::vector<CompletionToken>>(std::move(checked))};
                    const auto index = journal->entries.size();
                    if (writes(request.access)) {
                        const auto& id = request.range.bytes.backing;
                        journal->written.insert(BackingKey{id.owner, id.allocation, id.generation});
                    }
                    journal->entries.push_back(std::move(entry));
                    found = journal->index.emplace(key, index).first;
                    rollback.inserted.push_back(key);
                }
                indices.push_back(found->second);
                const auto& entry = journal->entries[found->second];
                for (const auto& dependency : entry.dependencies) depend(dependency);
            }
            for (const auto& dependency : dependencies)
                if (const auto failure = dependency.snapshot().error) return *failure;
            if (requests.size() > std::numeric_limits<std::uint64_t>::max() - journal->order)
                return error("batch access order is exhausted");
            GpuAccessAcceptance accepted{std::move(dependencies), state_};
            for (std::size_t i = 0; i < indices.size(); ++i) {
                auto& entry = journal->entries[indices[i]];
                entry.token = consumer;
                entry.request = requests[i];
                entry.order = ++journal->order;
            }
            journal->latest = consumer;
            rollback.committed = true;
            return accepted;
        });
    }
    RendererStatus submission_boundary() override { return boundary(true); }
    RendererResult<std::shared_ptr<GuestBoundary>> capture_guest_boundary(
        bool preserve_resident) override {
        return guarded([&]() -> RendererResult<std::shared_ptr<GuestBoundary>> {
            auto ready = state_->flush_journal();
            if (!ok(ready)) return std::get<RendererError>(ready);
            if (!state_->exclusions.empty()) return error("guest boundary capture has pending accesses");
            if (state_->boundary_error) return *state_->boundary_error;
            std::vector<CapturedGuestBoundary::Authority> authorities;
            std::vector<CompletionToken> accesses;
            std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
            const auto remember = [&](const CompletionToken& token) {
                const auto id = token.id();
                if (seen.emplace(id.queue, id.sequence).second) accesses.push_back(token);
            };
            for (const auto& [unused, list] : state_->intents)
                for (const auto& intent : list) remember(intent.token);
            for (const auto& [unused, backing] : state_->dirty_backings) {
                for (const auto offset : backing->dirty_offsets) {
                    const auto& segment = backing->segments.at(offset);
                    if (!segment.storage) return error("GPU-dirty interval has no authority");
                    authorities.push_back({state_->find(segment.dirty.bytes), segment});
                    remember(segment.dirty.last_writer);
                }
            }
            return std::shared_ptr<GuestBoundary>(std::make_shared<CapturedGuestBoundary>(
                state_, std::move(authorities), std::move(accesses), preserve_resident));
        });
    }
    RendererStatus internal_boundary() override { return boundary(false); }
    void release_settled_accesses() override {
        static_cast<void>(guarded([&]() -> RendererStatus {
            state_->prune(true);
            return renderer_success();
        }));
    }
    RendererStatus guest_boundary() override {
        return guarded([&]() -> RendererStatus {
            auto ready = quiesce();
            if (!ok(ready)) return ready;
            const auto stamp = cpu_write_stamp();
            std::vector<BackingRange> writeback;
            for (const auto& [unused, backing] : state_->dirty_backings) {
                for (const auto offset : backing->dirty_offsets) {
                    auto& segment = backing->segments.at(offset);
                    if (!segment.storage) return error("GPU-dirty interval has no authority");
                    if (segment.storage->guest_boundary_writeback()) {
                        writeback.push_back(segment.dirty.bytes);
                        continue;
                    }
                    const auto& bytes = segment.dirty.bytes;
                    auto& check = state_->resident_checks[{key(bytes.backing), bytes.offset, bytes.size}];
                    if (check.stamp && !cpu_written_since(bytes, check.stamp)) continue;
                    std::vector<std::byte> guest(bytes.size);
                    auto read = state_->options.backend->read_cpu(bytes, guest);
                    if (!ok(read)) return read;
                    const auto hash = content_hash64(guest);
                    if (check.stamp && hash != check.hash)
                        return error("native CPU store reached GPU-resident texture memory after a guest boundary");
                    check = {stamp ? stamp : 1, hash};
                }
            }
            for (const auto& range : writeback) {
                if (state_->options.work_counters) ++state_->options.work_counters->boundary_backings;
                auto status = state_->resolve(range);
                if (!ok(status)) return status;
            }
            return advance_epoch();
        });
    }
    RendererStatus advance_cpu_epoch() override {
        return guarded([&]() -> RendererStatus {
            // A prepared CPU write is exclusive; an epoch advance inside one
            // would republish its granules mid-transaction.
            if (std::any_of(state_->exclusions.begin(), state_->exclusions.end(),
                            [](const Exclusion& exclusion) {
                                return exclusion.prepared_write.has_value();
                            }))
                return error("CPU epoch advance overlaps a pending CPU write");
            // Pending reservations keep the storage they captured: the epoch
            // advance waits for nothing, so uncommitted and in-flight GPU work
            // is neither rejected nor observed. GPU-dirty intervals stay
            // GPU-authoritative; clean CPU-backed copies become stale.
            state_->prune();
            if (state_->boundary_error) return *state_->boundary_error;
            // With page write tracking every native CPU store publishes a
            // write generation that caches check by stamp, so the per-entry
            // epoch only defeated in-place publication; dirty pages alone are
            // authoritative. Guest-visible boundaries still
            // advance the epoch.
            if (state_->options.cpu_write_tracking) return renderer_success();
            return advance_epoch();
        });
    }
    // Waits for every GPU access once no CPU or GPU reservation is pending.
    RendererStatus quiesce() {
        auto flushed = state_->flush_journal();
        if (!ok(flushed)) return flushed;
        if (!state_->exclusions.empty()) return error("submission boundary has pending accesses");
        state_->prune(true);
        if (state_->boundary_error) return *state_->boundary_error;
        for (const auto& [backing, list] : state_->intents) {
            for (const auto& intent : list) {
                auto status = state_->wait(intent.token);
                if (!ok(status)) return status;
            }
        }
        state_->prune(true);
        return renderer_success();
    }
    RendererStatus boundary(bool resolve_gpu_dirty) {
        return guarded([&]() -> RendererStatus {
            auto ready = quiesce();
            if (!ok(ready)) return ready;
            while (resolve_gpu_dirty && !state_->dirty_backings.empty()) {
                const auto* backing = state_->dirty_backings.begin()->second;
                if (state_->options.work_counters) ++state_->options.work_counters->boundary_backings;
                auto status = state_->resolve({backing->registration.id, 0, backing->registration.size});
                if (!ok(status)) return status;
            }
            return advance_epoch();
        });
    }
private:
    RendererResult<std::vector<CompletionToken>> access_dependencies(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) {
        state_->prune();
        const auto snapshot = consumer.snapshot();
        if (requests.empty() || !consumer.valid() || snapshot.phase != SubmissionPhase::Accepted ||
            snapshot.error || !snapshot.id.queue || !snapshot.id.sequence ||
            snapshot.id.sequence <= state_->queue_sequences[snapshot.id.queue])
            return error("GPU reservation requires a fresh accepted submission token");
        for (const auto& [unused, pending] : state_->pending)
            if (pending.token.id() == snapshot.id) return error("submission already has a reservation");
        if (state_->operation == std::numeric_limits<std::uint64_t>::max())
            return error("GPU access identity is exhausted");
        // One dependency per queue: a queue completes in sequence order,
        // so its latest conflicting submission orders after all earlier
        // ones (the order prune() relies on).
        std::vector<CompletionToken> dependencies;
        auto depend = [&](const CompletionToken& token) {
            const auto id = token.id();
            for (auto& dependency : dependencies) {
                const auto current = dependency.id();
                if (current.queue != id.queue) continue;
                if (id.sequence > current.sequence) dependency = token;
                return;
            }
            dependencies.push_back(token);
        };
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            const auto validation = validate_retained(request.range, request.access);
            if (!ok(validation)) return std::get<RendererError>(validation);
            auto backing = state_->find(request.range.bytes);
            if (state_->excluded(request.range.bytes, writes(request.access)))
                return error("GPU reservation overlaps a pending access");
            for (std::size_t j = 0; j < i; ++j)
                if (writes(request.access) && writes(requests[j].access) &&
                    overlaps(request.range.bytes, requests[j].range.bytes))
                    return error("one reservation contains overlapping write authorities");
            for (auto segment_it = State::first_segment(*backing, request.range.bytes.offset);
                 segment_it != backing->segments.end() &&
                     segment_it->first < request.range.bytes.offset + request.range.bytes.size; ++segment_it) {
                state_->visit_interval();
                const auto& segment = segment_it->second;
                if (!segment.dirty.gpu_dirty) continue;
                auto writer = segment.dirty.last_writer.snapshot();
                if (writer.error) return *writer.error;
                if (!quiescent(writer)) depend(segment.dirty.last_writer);
            }
            // Newest first: on one queue the latest conflicting intent orders
            // after every earlier one, so the scan stops there instead of
            // visiting each draw of the batch. Reads conflict with writers only.
            for (const bool writers : {true, false}) {
                if (!writers && !writes(request.access)) break;
                const auto* list = state_->intents_of(request.range.bytes.backing, writers);
                if (!list) continue;
                for (auto it = list->items.rbegin();
                     it != list->items.rend() - std::ptrdiff_t(list->head); ++it) {
                    if (!overlaps(request.range.bytes, it->range.bytes)) continue;
                    depend(it->token);
                    if (!list->mixed_queues) break;
                }
            }
        }
        return dependencies;
    }

    // Native guest stores are untracked. Every successful epoch advance
    // invalidates CPU copies, even when no registry operation intervened.
    RendererStatus advance_epoch() { return state_->advance_epoch(); }
    std::vector<Mapping> mapping_values(std::optional<std::uint64_t> omit = {}) const {
        std::vector<Mapping> result;
        for (const auto& [id, entry] : state_->mappings)
            if (!omit || id != *omit) result.push_back(entry->registration.mapping);
        return result;
    }
    std::optional<BackingRange> cpu_range(std::uint64_t address, std::size_t size) const {
        auto found = state_->cpu_backings.upper_bound(address);
        if (found != state_->cpu_backings.begin()) {
            const auto [base, backing] = *std::prev(found);
            const auto offset = address - base;
            if (offset < backing->registration.size && size <= backing->registration.size - offset)
                return BackingRange{backing->registration.id, offset, size};
        }
        return {};
    }
    static bool read(void* context, std::uint64_t address, std::span<std::byte> destination) noexcept {
        auto& registry = *static_cast<Registry*>(context);
        try {
            const auto range = registry.cpu_range(address, destination.size());
            if (!range) return false;
            auto transaction = registry.prepare_cpu_read(*range, ResolveReason::CpuEngineRead);
            if (std::holds_alternative<RendererError>(transaction)) return false;
            return ok(registry.state_->options.backend->read_cpu(*range, destination));
        } catch (...) { return false; }
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> source) noexcept {
        auto& registry = *static_cast<Registry*>(context);
        try {
            const auto range = registry.cpu_range(address, source.size());
            if (!range) return false;
            auto prepared = registry.prepare_cpu_write_impl(*range, false);
            auto* transaction = std::get_if<std::unique_ptr<CpuWriteTransaction>>(&prepared);
            if (!transaction) return false;
            RendererStatus result;
            try { result = registry.state_->write_cpu(*registry.state_->find(*range), *range, source); }
            catch (...) { (*transaction)->commit(); return false; }
            (*transaction)->commit();
            return ok(result);
        } catch (...) { return false; }
    }
    std::shared_ptr<State> state_;
    GpuAddressSpace address_space_;
};
} // namespace

RendererResult<std::shared_ptr<ResourceMemoryRegistry>> make_resource_memory(ResourceMemoryOptions options) {
    return guarded([&]() -> RendererResult<std::shared_ptr<ResourceMemoryRegistry>> {
        if (!options.address_space || !options.backend || !options.cpu_tracking_granule ||
            (options.cpu_tracking_granule & (options.cpu_tracking_granule - 1)))
            return error("resource memory requires owned callbacks and valid address-space/granule identities");
        if (options.mode != CoherenceMode::SubmissionBoundary)
            return error("tracked native coherence requires the L3 native-access implementation");
        if (options.cpu_write_tracking) memory::WriteTracker::install();
        return std::shared_ptr<ResourceMemoryRegistry>(std::make_shared<Registry>(std::move(options)));
    });
}
} // namespace armigo::graphics
