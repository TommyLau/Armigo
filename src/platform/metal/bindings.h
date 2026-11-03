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

#include <armigo/graphics/buffer_cache.h>
#include <armigo/graphics/draw_plan.h>
#include <armigo/graphics/texture_cache.h>
#include "pipeline_state.h"

#include <algorithm>
#include <span>
#include <tuple>
#include <map>
#include <unordered_map>
#include <vector>

namespace armigo::platform::metal {

class DeviceContext;

struct BufferBinding {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::uint32_t index = 0;
    graphics::ResidentBuffer buffer;
    graphics::BufferUse use = graphics::BufferUse::Uniform;
    graphics::ResourceAccess access = graphics::ResourceAccess::Read;
};
struct TextureBinding {
    shader_input::Stage stage = shader_input::Stage::Fragment;
    std::uint32_t texture_index = 0, sampler_index = 0;
    graphics::ResidentTexture texture;
    graphics::RetainedResource sampler;
    // Present only for an independently reflected argument-buffer texture.
    std::optional<std::uint32_t> argument_byte_offset = std::nullopt;
};
struct SamplerBinding {
    shader_input::Stage stage = shader_input::Stage::Fragment;
    std::uint32_t index = 0;
    graphics::RetainedResource sampler;
    std::optional<std::uint32_t> argument_byte_offset = std::nullopt;
};
struct ArgumentBufferBinding {
    shader_input::Stage stage = shader_input::Stage::Fragment;
    CompiledArgumentBufferLayout layout;
};
struct Bindings {
    AdmittedGraphicsStages stages;
    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    std::vector<SamplerBinding> samplers;
    std::vector<ArgumentBufferBinding> argument_buffers;
    // Host-only coverage images retained with the immutable sampling records.
    std::vector<graphics::RetainedResource> custom_border_textures;
    // Native vertex stream indices match PipelineKey's buffer layout exactly.
    std::vector<BufferBinding> vertex_buffers;
    std::optional<graphics::ResidentBuffer> index_buffer;
    std::vector<graphics::CompletionToken> dependencies;
    // Explicit shader/vertex/index read/write intents, including storage writes.
    // Host-generated immutable upload slices have no guest backing interval.
    std::vector<graphics::GpuAccessRequest> accesses;
};

// A native allocation admitted without alias conversion or partial subresources.
// Weak ownership preserves identity without pinning storage in the epoch log.
struct ResidentTextureIdentity {
    std::weak_ptr<const void> allocation;
    graphics::TextureAspect aspect = graphics::TextureAspect::Color;
    bool operator==(const ResidentTextureIdentity& other) const {
        return !allocation.owner_before(other.allocation) && !other.allocation.owner_before(allocation) &&
               aspect == other.aspect;
    }
};
[[nodiscard]] ResidentTextureIdentity resident_texture_identity(
    const graphics::ResidentTexture& texture);

// Scope in which a planner may reuse per-stage bindings of earlier draws: the
// register groups changed since the previous draw (graphics::threed_dirty),
// a renderer epoch that changes whenever GPU work other than consecutive draws
// with these attachments intervenes (zero disables reuse), and the GPU ranges this
// draw renders to, which reused sampled textures must not overlap.
// GPU writes per renderer epoch: each epoch's attachments plus the targets of
// the operation that began it, or `all` when they are unknown. A binding set
// built in an earlier epoch stays valid while nothing written since reaches
// the guest ranges it read, except full-storage attachment writes to the same
// resident allocation. Those preserve the view but require fresh writer tokens.
struct EpochLog {
    // Latest epoch that wrote each range, keyed by range start; a range
    // written again keeps its start and widens to the larger end.
    struct Write {
        std::uint64_t end = 0, epoch = 0;
        ResidentTextureIdentity resident;
        // A later same-storage write must not hide an intervening alias write.
        std::uint64_t incompatible_epoch = 0;
    };
    struct ResidentRead {
        std::uint64_t base = 0, size = 0;
        graphics::BackingRange bytes;
        ResidentTextureIdentity resident;
    };
    std::map<std::uint64_t, Write> gpu;
    // Writes known by backing range (compute dispatch storage).
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>, Write> backing;
    std::uint64_t gpu_span = 0, backing_span = 0;
    // Latest epoch of any write touching each 1 MiB granule (GPU addresses;
    // backing ranges by a hash of backing and granule, where a collision only
    // costs an exact scan): reads whose granules took no write since an epoch
    // skip the scans above, which reach back a whole render target.
    static constexpr unsigned granule_shift = 20;
    std::unordered_map<std::uint64_t, std::uint64_t> gpu_granules, backing_granules;
    static std::uint64_t backing_granule(const graphics::BackingId& backing, std::uint64_t offset) noexcept {
        std::uint64_t hash = backing.owner * 0x9e3779b97f4a7c15ull;
        hash ^= backing.allocation + 0x632be59bd9b4e019ull + (hash << 6) + (hash >> 2);
        hash ^= backing.generation + 0x85ebca77c2b2ae63ull + (hash << 6) + (hash >> 2);
        return hash ^ ((offset >> granule_shift) * 0xc2b2ae3d27d4eb4full);
    }
    static void mark(std::unordered_map<std::uint64_t, std::uint64_t>& granules, std::uint64_t key,
                     std::uint64_t epoch) {
        auto& latest = granules[key];
        latest = std::max(latest, epoch);
    }
    static bool touched_since(const std::unordered_map<std::uint64_t, std::uint64_t>& granules,
                              std::uint64_t key, std::uint64_t epoch) noexcept {
        const auto found = granules.find(key);
        return found != granules.end() && found->second >= epoch;
    }
    // Newest epoch whose writes are unknown: no older set can be proven clean.
    std::uint64_t last_unknown = 0;
    static constexpr std::size_t limit = 4096;

    // GPU ranges whose mapping was added, removed or replaced, with the epoch
    // of the change: a set that read them may name other memory now. Few
    // entries; the oldest are forgotten into last_unknown.
    struct Remap {
        std::uint64_t base = 0, end = 0, epoch = 0;
    };
    std::vector<Remap> remaps;
    std::uint64_t latest_remap = 0;
    static constexpr std::size_t remap_limit = 64;

    void unknown(std::uint64_t epoch) {
        last_unknown = epoch;
        gpu.clear();
        backing.clear();
        gpu_granules.clear();
        backing_granules.clear();
        remaps.clear();
        gpu_span = backing_span = 0;
    }
    void remap(std::uint64_t base, std::uint64_t size, std::uint64_t epoch) {
        if (!size) return;
        if (remaps.size() >= remap_limit) {
            last_unknown = std::max(last_unknown, remaps.front().epoch);
            remaps.erase(remaps.begin());
        }
        remaps.push_back({base, base + size, epoch});
        latest_remap = std::max(latest_remap, epoch);
    }
    // Times a full log forgot its older half (tracing).
    std::uint64_t overflows = 0;
    // A full log forgets the writes of its older half: sets built at or before
    // the newest forgotten epoch can no longer be proven clean, newer ones
    // stay valid (forgetting everything invalidated every set at once).
    template <class Map, class Start>
    void forget_older(Map& map, std::uint64_t& span, Start start) {
        ++overflows;
        std::vector<std::uint64_t> epochs;
        epochs.reserve(map.size());
        for (const auto& entry : map) epochs.push_back(entry.second.epoch);
        const auto middle = epochs.begin() + std::ptrdiff_t(epochs.size() / 2);
        std::nth_element(epochs.begin(), middle, epochs.end());
        const auto cutoff = *middle;
        last_unknown = std::max(last_unknown, cutoff);
        span = 0;
        for (auto it = map.begin(); it != map.end();) {
            if (it->second.epoch <= cutoff) {
                it = map.erase(it);
                continue;
            }
            span = std::max(span, it->second.end - start(it->first));
            ++it;
        }
    }
    void write(std::uint64_t base, std::uint64_t size, std::uint64_t epoch,
               ResidentTextureIdentity resident = {}) {
        if (!size) return;
        if (gpu.size() >= limit && !gpu.contains(base))
            forget_older(gpu, gpu_span, [](std::uint64_t start) { return start; });
        auto& entry = gpu[base];
        if (entry.resident != resident || entry.end != base + size)
            entry.incompatible_epoch = std::max(entry.incompatible_epoch, entry.epoch);
        entry.resident = std::move(resident);
        entry.end = std::max(entry.end, base + size);
        entry.epoch = epoch;
        gpu_span = std::max(gpu_span, entry.end - base);
        for (auto granule = base >> granule_shift; granule <= (base + size - 1) >> granule_shift; ++granule)
            mark(gpu_granules, granule, epoch);
    }
    void write(const graphics::BackingRange& range, std::uint64_t epoch) {
        if (!range.size) return;
        if (backing.size() >= limit &&
            !backing.contains({range.backing.owner, range.backing.allocation, range.backing.generation, range.offset}))
            forget_older(backing, backing_span, [](const auto& key) { return std::get<3>(key); });
        auto& entry = backing[{range.backing.owner, range.backing.allocation, range.backing.generation, range.offset}];
        entry.end = std::max(entry.end, range.offset + range.size);
        entry.epoch = epoch;
        backing_span = std::max(backing_span, range.size);
        for (auto offset = range.offset >> granule_shift << granule_shift; offset < range.offset + range.size;
             offset += std::uint64_t{1} << granule_shift)
            mark(backing_granules, backing_granule(range.backing, offset), epoch);
    }
    // True when writes in or after `epoch` miss reads or preserve their resident storage.
    bool clean_since(std::uint64_t epoch,
                     std::span<const std::pair<std::uint64_t, std::uint64_t>> reads,
                     std::span<const graphics::RetainedRange> backing_reads = {},
                     std::span<const ResidentRead> resident_reads = {},
                     std::vector<graphics::BackingRange>* resident_writes = nullptr) const {
        if (last_unknown >= epoch) return false;
        if (latest_remap >= epoch)
            for (const auto& [read, length] : reads)
                for (const auto& change : remaps)
                    if (change.epoch >= epoch && length && change.base < read + length && read < change.end)
                        return false;
        for (const auto& [read, length] : reads) {
            if (!length) continue;
            bool touched = false;
            for (auto granule = read >> granule_shift; !touched && granule <= (read + length - 1) >> granule_shift;
                 ++granule)
                touched = touched_since(gpu_granules, granule, epoch);
            if (!touched) continue;
            const auto first = read > gpu_span ? read - gpu_span : 0;
            for (auto it = gpu.lower_bound(first); it != gpu.end() && it->first < read + length; ++it)
                if (it->second.end > read && it->second.epoch >= epoch) {
                    const auto& write = it->second;
                    if (write.resident.allocation.expired() || write.incompatible_epoch >= epoch ||
                        it->first != read || write.end != read + length) return false;
                    const auto found = std::ranges::find_if(resident_reads, [&](const ResidentRead& sampled) {
                        return sampled.base == read && sampled.size == length && sampled.resident == write.resident;
                    });
                    if (found == resident_reads.end()) return false;
                    if (resident_writes && std::ranges::find(*resident_writes, found->bytes) == resident_writes->end())
                        resident_writes->push_back(found->bytes);
                }
        }
        for (const auto& range : backing_reads) {
            const auto& bytes = range.bytes;
            if (!bytes.size) continue;
            bool touched = false;
            for (auto offset = bytes.offset >> granule_shift << granule_shift;
                 !touched && offset < bytes.offset + bytes.size; offset += std::uint64_t{1} << granule_shift)
                touched = touched_since(backing_granules, backing_granule(bytes.backing, offset), epoch);
            if (!touched) continue;
            const auto first = bytes.offset > backing_span ? bytes.offset - backing_span : 0;
            const auto owner = std::tuple{bytes.backing.owner, bytes.backing.allocation, bytes.backing.generation};
            for (auto it = backing.lower_bound({bytes.backing.owner, bytes.backing.allocation,
                                                bytes.backing.generation, first});
                 it != backing.end(); ++it) {
                const auto& [o, a, g, offset] = it->first;
                if (std::tuple{o, a, g} != owner || offset >= bytes.offset + bytes.size) break;
                if (it->second.end > bytes.offset && it->second.epoch >= epoch) return false;
            }
        }
        return true;
    }
    bool textures_clean_since(std::uint64_t epoch, std::span<const ResidentRead> reads,
                              std::vector<graphics::BackingRange>* resident_writes = nullptr) const {
        for (const auto& read : reads) {
            const std::pair range{read.base, read.size};
            if (!clean_since(epoch, std::span{&range, 1}, {}, std::span{&read, 1}, resident_writes)) return false;
        }
        return true;
    }
};

struct BindingReuse {
    std::uint64_t dirty = graphics::threed_dirty::all;
    std::uint64_t epoch = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> attachments;
    const EpochLog* log = nullptr;
};

// L10: preserve CompiledGraphicsStage's live texture facts/cbuf selector checks,
// class-typed vertex inputs, zero-filled bounded windows, disabled cbuf slots,
// programmed-size support buffer and binding-index collision rejection.
// Use existing descriptor decode, not cache membership as admission evidence.
// No guest callback/span escapes; upload immutable slices via L5, retain L4
// views/samplers and both stages. Missing package entries, stale facts, invalid
// layouts, absent owners or unsupported shapes return owned structured errors.
// A failed prepare may allocate cache entries, but encodes no GPU/guest effects.
class BindingPlanner {
public:
    virtual ~BindingPlanner() = default;
    virtual graphics::RendererResult<Bindings> prepare(
        const graphics::RenderStop& stop, const AdmittedGraphicsStages& stages,
        const graphics::DrawPlan& draw, graphics::ResourceMemory& memory,
        graphics::BufferCache& buffers, graphics::TextureCache& textures,
        const BindingReuse& reuse = {}) = 0;
    // Releases bindings kept for reuse, so their storage stops counting as
    // in use (a failed preparation retries against unpinned allocations).
    virtual void drop_reuse() {}
    // For a draw repeating the last prepared draw's state: appends the
    // buffer bindings whose contents changed (uniform windows a CPU store
    // reached, uploaded again) and returns true, or returns false when
    // anything else the last bindings read changed.
    // dirty: register groups changed since the last draw; only index buffer,
    // vertex stream and constant-buffer groups are admitted. New vertex and
    // index bindings are returned for those groups' changes.
    struct RepeatBindings {
        std::vector<BufferBinding> changed;
        std::vector<BufferBinding> vertex_buffers;
        std::optional<graphics::ResidentBuffer> index_buffer;
    };
    // Engine stores (front-end constant-buffer and inline-upload data, guest
    // effect payloads) the backend made inside a WriteTracker::NotifiedStores
    // scope, by GPU range; with reported stores a repeat skips the stamp checks
    // while the tracker's unnoted clock holds still.
    virtual void note_engine_store(std::uint64_t gpu_address, std::uint64_t size) {
        static_cast<void>(gpu_address);
        static_cast<void>(size);
    }
    virtual void set_engine_stores_reported(bool reported) { static_cast<void>(reported); }
    virtual bool refresh_repeat(const graphics::RenderStop& stop, const graphics::DrawPlan& draw,
                                std::uint64_t dirty, graphics::ResourceMemory& memory,
                                graphics::BufferCache& buffers, RepeatBindings& out) {
        static_cast<void>(stop);
        static_cast<void>(draw);
        static_cast<void>(dirty);
        static_cast<void>(memory);
        static_cast<void>(buffers);
        static_cast<void>(out);
        return false;
    }
    // Compute launches bind through the same constant-buffer window, storage
    // descriptor and texture paths as draw stages, with the queue meta data's
    // constant buffers in place of a 3D stage's. Every binding carries
    // shader_input::Stage::Compute; Bindings::stages stays empty.
    virtual graphics::RendererResult<Bindings> prepare_compute(
        const graphics::RenderStop&, const CompiledGraphicsStage&, graphics::ResourceMemory&,
        graphics::BufferCache&, graphics::TextureCache&) {
        return graphics::RendererError{{graphics::CommandStatus::Unsupported,
                                        "binding planner has no compute path"}};
    }
};

// Rebuilds the descriptor-baked vertex portion of key from the same checked
// guest attribute plan consumed by BindingPlanner. Existing vertex entries are
// replaced only after the complete layout has been validated.
[[nodiscard]] graphics::RendererStatus populate_vertex_layout(
    PipelineKey& key, const graphics::DrawPlan& draw);

// Sampler objects share the device-owned pipeline/state cache. The returned
// planner borrows no service: the cache remains retained for its lifetime.
// Supply its native device domain to admit typed null sampled resources.
[[nodiscard]] graphics::RendererResult<std::shared_ptr<BindingPlanner>>
create_binding_planner(std::shared_ptr<PipelineStateCache> states,
                       std::shared_ptr<DeviceContext> device = {});

} // namespace armigo::platform::metal
