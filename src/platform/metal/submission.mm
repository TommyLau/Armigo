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
// Persistent argument-table slots retain address shadows, arena allocations,
// resource residency, and submission allocators across compatible commands.

#include "submission.h"

#include "attachments.h"
#include "buffer_cache.h"
#include "clear_encoder.h"
#include "device_context.h"
#include "native_resources.h"
#include "pass_access.h"
#include "texture_residency.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <system_error>
#include <thread>
#include <unordered_set>

namespace armigo::platform::metal {
namespace {

using namespace graphics;

RendererError failure(std::string_view detail,
                      CommandStatus status = CommandStatus::Unsupported) {
    return RendererError({status, detail});
}

bool succeeded(const RendererStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

constexpr std::string_view commit_hold_detail = "submission commit hold blocks retirement";

bool writes(ResourceAccess access) {
    return access != ResourceAccess::Read;
}

bool overlaps(const BackingRange& left, const BackingRange& right) {
    if (left.backing != right.backing) return false;
    return left.offset < right.offset + right.size &&
           right.offset < left.offset + left.size;
}

// Metal snapshots table entries at each encoded draw or dispatch. The table
// itself can therefore persist. Indirect argument slices stay immutable, and
// the packet retains referenced resources until GPU completion.
struct StageTable {
    struct BufferSlot {
        MTLGPUAddress address = 0;
        NSUInteger stride = 0;
        bool attribute = false;
    };
    id<MTL4ArgumentTable> table = nil;
    bool vertex = false;
    std::array<BufferSlot, 31> buffers{};
    std::array<MTLResourceID, 128> textures{};
    std::array<MTLResourceID, 16> samplers{};
    std::uint32_t buffers_used = 0, buffers_seen = 0;
    std::array<std::uint64_t, 2> textures_used{}, textures_seen{};
    std::uint32_t samplers_used = 0, samplers_seen = 0;

    void begin() {
        buffers_seen = samplers_seen = 0;
        textures_seen = {};
    }
    void address(MTLGPUAddress value, std::uint32_t index,
                 bool attribute = false, NSUInteger stride = 0) {
        const auto bit = std::uint32_t{1} << index;
        buffers_seen |= bit;
        auto& old = buffers[index];
        if (old.address != value || old.attribute != attribute || old.stride != stride) {
            if (vertex && (attribute || old.attribute))
                [table setAddress:value attributeStride:stride atIndex:index];
            else
                [table setAddress:value atIndex:index];
            old = {value, stride, attribute};
        }
        buffers_used |= bit;
    }
    void texture(MTLResourceID value, std::uint32_t index) {
        const auto bit = std::uint64_t{1} << (index % 64);
        textures_seen[index / 64] |= bit;
        if (textures[index]._impl != value._impl) {
            [table setTexture:value atIndex:index];
            textures[index] = value;
        }
        textures_used[index / 64] |= bit;
    }
    void sampler(MTLResourceID value, std::uint32_t index) {
        const auto bit = std::uint32_t{1} << index;
        samplers_seen |= bit;
        if (samplers[index]._impl != value._impl) {
            [table setSamplerState:value atIndex:index];
            samplers[index] = value;
        }
        samplers_used |= bit;
    }
    void finish() {
        // Removed slots must retain the old fresh-table behavior. Track actual
        // writes even during rejected preparation; the next begin/finish pair
        // clears its partial bindings before another draw can consume them.
        auto removed_buffers = buffers_used & ~buffers_seen;
        const auto keep_buffers = buffers_seen;
        while (removed_buffers) {
            const auto index = std::countr_zero(removed_buffers);
            address(0, index);
            removed_buffers &= removed_buffers - 1;
        }
        buffers_used = keep_buffers;
        for (std::size_t word = 0; word < textures_used.size(); ++word) {
            auto removed = textures_used[word] & ~textures_seen[word];
            const auto keep = textures_seen[word];
            while (removed) {
                texture(MTLResourceID{}, std::uint32_t(word * 64 + std::countr_zero(removed)));
                removed &= removed - 1;
            }
            textures_used[word] = keep;
        }
        auto removed_samplers = samplers_used & ~samplers_seen;
        const auto keep_samplers = samplers_seen;
        while (removed_samplers) {
            sampler(MTLResourceID{}, std::countr_zero(removed_samplers));
            removed_samplers &= removed_samplers - 1;
        }
        samplers_used = keep_samplers;
    }
};

struct ArgumentTables {
    StageTable vertex, fragment, compute;
    std::atomic<std::uint64_t> created{0};
};

struct NativePacket {
    struct FeedbackCopy {
        id<MTLTexture> source = nil, destination = nil;
        std::uint32_t first_level = 0, first_layer = 0;
        std::uint32_t levels = 0, layers = 0;
    };
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLComputePipelineState> compute_pipeline = nil;
    id<MTL4ArgumentTable> compute_table = nil;
    id<MTLDepthStencilState> depth_stencil = nil;
    id<MTL4ArgumentTable> vertex_table = nil;
    id<MTL4ArgumentTable> fragment_table = nil;
    id<MTLBuffer> draw_support = nil;
    NativeRenderTargets targets;
    ClearBindings clear;
    std::vector<FeedbackCopy> feedback_copies;
    // Sub-resource copies into sampled storage, encoded before the work.
    std::vector<std::shared_ptr<const SubresourceCopy>> subresource_copies;
    std::uint64_t feedback_bytes = 0, argument_buffer_bytes = 0;
    NSMutableArray<id>* feedback_objects = [NSMutableArray array];
    id<MTLResidencySet> feedback_residency = nil;
};

struct ArgumentPage {
    id<MTLBuffer> buffer = nil;
    std::byte* cpu = nullptr;
    MTLGPUAddress gpu = 0;
    std::uint64_t length = 0, used = 0;
};

struct CommandBundle {
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commands = nil;
};

struct ArgumentSlice {
    id<MTLBuffer> buffer = nil;
    MTLGPUAddress address = 0;
};

struct StageArguments {
    CompiledArgumentBufferLayout layout{};
    std::vector<std::uint64_t> members;
    ArgumentSlice slice;
};

struct NativePool {
    std::mutex mutex;
    std::vector<ArgumentPage> pages;
    std::vector<CommandBundle> commands;
    std::atomic<std::uint64_t> pages_created{0}, pages_reused{0};
    std::atomic<std::uint64_t> slices_created{0}, slices_reused{0};
    std::atomic<std::uint64_t> commands_created{0}, commands_reused{0};

    RendererResult<ArgumentPage> acquire_page(id<MTLDevice> device, std::uint64_t minimum) {
        std::unique_lock lock(mutex);
        const auto found = std::ranges::find_if(pages, [minimum](const auto& page) {
            return page.length >= minimum;
        });
        if (found != pages.end()) {
            auto page = std::move(*found);
            pages.erase(found);
            lock.unlock();
            page.used = 0;
            pages_reused.fetch_add(1, std::memory_order_relaxed);
            return page;
        }
        lock.unlock();
        const auto length = std::max<std::uint64_t>(4 * 1024 * 1024, minimum);
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!buffer || !buffer.contents)
            return failure("argument arena allocation failed", CommandStatus::MemoryFault);
        pages_created.fetch_add(1, std::memory_order_relaxed);
        return ArgumentPage{buffer, static_cast<std::byte*>(buffer.contents),
                            buffer.gpuAddress, buffer.length, 0};
    }

    RendererResult<CommandBundle> acquire_commands(id<MTLDevice> device) {
        std::unique_lock lock(mutex);
        if (!commands.empty()) {
            auto bundle = std::move(commands.back());
            commands.pop_back();
            lock.unlock();
            // Bundles reach this pool only after whole-batch retirement, not
            // when a segment ends or when its command buffer is submitted.
            [bundle.allocator reset];
            commands_reused.fetch_add(1, std::memory_order_relaxed);
            return bundle;
        }
        lock.unlock();
        CommandBundle bundle{[device newCommandAllocator], [device newCommandBuffer]};
        if (!bundle.allocator || !bundle.commands)
            return failure("submission command allocation failed", CommandStatus::MemoryFault);
        commands_created.fetch_add(1, std::memory_order_relaxed);
        return bundle;
    }
};

struct BatchLease final : ResourceLease {
    NSMutableArray<id>* objects = [NSMutableArray array];
    std::vector<ArgumentPage> argument_pages;
    std::vector<CommandBundle> command_bundles;
};

// The coordinator calls this only after every packet in the batch has finished.
// Recycling is optional: allocation failure in the free lists drops storage
// that is already quiescent without changing publication or failure semantics.
void recycle_native(NativePool& pool, BatchLease& lease) noexcept {
    std::lock_guard lock(pool.mutex);
    try {
        for (auto& page : lease.argument_pages) pool.pages.push_back(std::move(page));
        for (auto& bundle : lease.command_bundles) pool.commands.push_back(std::move(bundle));
    } catch (const std::bad_alloc&) {
    }
    lease.argument_pages.clear();
    lease.command_bundles.clear();
}

struct PacketLease final : ResourceLease {
    explicit PacketLease(PreparedRender value, std::shared_ptr<BatchLease> batch_owner)
        : prepared(std::move(value)), batch(std::move(batch_owner)) {}

    PreparedRender prepared;
    NativePacket native;
    GpuAccessAcceptance acceptance;
    std::shared_ptr<BatchLease> batch;
    // Uploads bound by repeats of this draw, retained with it until the
    // batch completes. Written by the caller, never read by the worker.
    std::vector<RetainedResource> repeat_resources;
};

// A repeat of the last draw: new segments and the buffer slots whose
// contents changed; everything else stays bound.
struct RepeatJob {
    std::vector<DrawSegment> segments;
    std::vector<BufferBinding> changed;
    std::vector<BufferBinding> vertex_buffers;
    std::optional<ResidentBuffer> index_buffer;
    std::shared_ptr<const RenderState> state;
    id<MTLRenderPipelineState> pipeline = nil;
    // The swapped pipeline's vertex buffer layouts (strides), if any.
    std::shared_ptr<const ResidentPipeline> layout;
};

std::optional<MTLIndexType> index_type(IndexFormat format);
RendererResult<MTL4RenderPassDescriptor*> pass_descriptor(
    const PacketLease& packet);

struct Packet {
    Packet(std::shared_ptr<PacketLease> retained, SubmissionCompletion state)
        : lease(std::move(retained)), completion(std::move(state)) {}

    std::shared_ptr<PacketLease> lease;
    SubmissionCompletion completion;
    std::vector<CompletionToken> dependencies;
    bool finish_attempted = false;
    bool prefix_safe_completion = false;
};

struct Batch {
    explicit Batch(NativePool& pool) : pool(pool) {}
    NativePool& pool;
    // The caller fills these before publishing the commit job. Only the worker
    // writes commit_done; retirement acquires it before using native storage.
    std::vector<SubmissionCompletion> completions;
    std::atomic<bool> commit_done{false};
    bool without_gpu_work = false;
    std::array<StageArguments, 3> arguments;
    std::shared_ptr<BatchLease> lease = std::make_shared<BatchLease>();
    std::vector<Packet> packets;
    std::vector<id<MTL4CommandBuffer>> buffers;
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commands = nil;
    id<MTL4RenderCommandEncoder> render = nil;
    // State last set on `render`, so draws set only what changed. Reset
    // whenever a new render encoder opens.
    struct EncodedState {
        id<MTLRenderPipelineState> pipeline = nil;
        id<MTLDepthStencilState> depth_stencil = nil;
        id<MTL4ArgumentTable> vertex_table = nil, fragment_table = nil;
        int cull = -1, winding = -1, fill = -1, clip = -1;
        std::array<float, 4> blend{-1.0f, -1.0f, -1.0f, -1.0f};
        std::uint32_t front_reference = ~0u, back_reference = ~0u;
        std::vector<MTLViewport> viewports;
        std::vector<MTLScissorRect> scissors;
        std::array<float, 2> depth_bias{-1.0f, -1.0f};
        bool valid = false;
    } encoded;
    id<MTLResidencySet> residency = nil;
    std::unordered_set<const void*> resident_allocations;
    std::unordered_set<const SubresourceCopy*> encoded_copies;
    std::optional<RenderPassKey> pass;
    NativeRenderTargets pass_targets;
    PassAccessSummary pass_accesses;
    std::uint64_t render_passes = 0;
    std::uint64_t draws = 0;
};

// Residency already retains its allocations. Cache native allocation identity
// only for this batch, so repeated bindings avoid native set bookkeeping without
// retaining any additional historical guest storage after batch retirement.
void add_resident(Batch& batch, id<MTLAllocation> allocation) {
    try {
        if (!batch.resident_allocations.insert((__bridge const void*)allocation).second) return;
    } catch (const std::bad_alloc&) {
        // Deduplication is optional; fall back to the existing idempotent API
        // instead of rejecting an otherwise valid packet on cache growth failure.
    }
    [batch.residency addAllocation:allocation];
}

struct SubmittedBatch {
    std::shared_ptr<Batch> batch;
    std::vector<Packet> packets;
    // Only successfully retired batches may return this storage to the pool.
    std::shared_ptr<BatchLease> recyclable;
    std::size_t next = 0;
    bool without_gpu_work = false;
};

// ARMIGO_TRACE_ENGINE_FLUSH: share of wall time the GPU spent executing
// committed work, from commit feedback (feedback handlers run in order).
void note_gpu_time(CFTimeInterval start, CFTimeInterval end) noexcept {
    static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
    if (!trace || end <= start) return;
    static CFTimeInterval first = 0, busy = 0, last_end = 0;
    static std::uint64_t commits = 0;
    if (!first) first = start;
    // Overlapping intervals count once.
    busy += end - std::max(start, last_end > start ? std::min(last_end, end) : start);
    last_end = std::max(last_end, end);
    if (++commits % 2000) return;
    // The window since the previous line, so boot does not dilute it.
    static CFTimeInterval window_start = 0, window_busy = 0;
    const auto wall = end - (window_start ? window_start : first);
    armigo::diag::opt_in( "gpu_busy commits=%llu window_ms=%.0f busy=%.0f%%\n", (unsigned long long)commits,
                 wall * 1000.0, wall > 0 ? 100.0 * (busy - window_busy) / wall : 0.0);
    window_start = end;
    window_busy = busy;
}

struct SharedState {
    std::shared_ptr<DeviceContext> device;
    std::shared_ptr<ResourceMemory> memory;
    std::mutex feedback_mutex;
    std::condition_variable feedback_ready;
    std::uint64_t feedback_generation = 0;
    // Bumped with every feedback generation and encoding failure: poll() has
    // nothing new to retire or report while it stands still.
    std::atomic<std::uint64_t> progress{0};
    std::shared_ptr<SubmissionTestControl> test_control;
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> command_buffers{0};
    std::atomic<std::uint64_t> render_passes{0};
    std::atomic<std::uint64_t> draws{0};
    std::optional<RendererError> ordered_failure;
};

class OrderedActions final : public CompletionActions {
public:
    OrderedActions(std::shared_ptr<SharedState> state,
                   std::shared_ptr<CompletionActions> actions)
        : state_(std::move(state)), actions_(std::move(actions)) {}

    RendererStatus make_coherent() override {
        if (state_->ordered_failure) return *state_->ordered_failure;
        auto status = actions_->make_coherent();
        latch(status);
        return status;
    }
    RendererStatus prepare_publication() override {
        if (state_->ordered_failure) return *state_->ordered_failure;
        auto status = actions_->prepare_publication();
        latch(status);
        return status;
    }
    void publish() noexcept override { actions_->publish(); }
    RendererStatus preserve_failed_authority() override {
        return actions_->preserve_failed_authority();
    }

private:
    void latch(const RendererStatus& status) {
        if (!state_->ordered_failure) {
            if (const auto* error = std::get_if<RendererError>(&status))
                state_->ordered_failure = *error;
        }
    }
    std::shared_ptr<SharedState> state_;
    std::shared_ptr<CompletionActions> actions_;
};

bool contains_attachment(const PreparedRender& prepared, const BackingRange& bytes) {
    const auto matches = [&](const auto& texture) {
        return texture && overlaps(texture->key.storage.bytes, bytes);
    };
    for (const auto& color : prepared.attachments.colors)
        if (matches(color)) return true;
    return matches(prepared.attachments.depth) || matches(prepared.attachments.stencil);
}

// The shader resource bindings of a draw or a compute dispatch.
const Bindings* packet_bindings(const PreparedRender& prepared) {
    if (const auto* draw = std::get_if<PreparedDraw>(&prepared.operation)) return &draw->bindings;
    if (const auto* dispatch = std::get_if<PreparedDispatch>(&prepared.operation))
        return &dispatch->bindings;
    return nullptr;
}

std::uint64_t argument_bytes_limit(const PreparedRender& prepared) {
    if (const auto* draw = std::get_if<PreparedDraw>(&prepared.operation))
        return draw->feedback_bytes_limit;
    if (const auto* dispatch = std::get_if<PreparedDispatch>(&prepared.operation))
        return dispatch->argument_bytes_limit;
    return 0;
}

bool contains_buffer_binding(const PreparedRender& prepared, const BackingRange& bytes) {
    const auto* bindings = packet_bindings(prepared);
    if (!bindings) return false;
    const auto buffer_matches = [&](const BufferBinding& binding) {
        return overlaps(binding.buffer.key.bytes, bytes);
    };
    if (std::ranges::any_of(bindings->buffers, buffer_matches) ||
        std::ranges::any_of(bindings->vertex_buffers, buffer_matches)) return true;
    return bindings->index_buffer &&
           overlaps(bindings->index_buffer->key.bytes, bytes);
}

RendererStatus validate_packet(const PreparedRender& prepared,
                               const SharedState& state) {
    const auto& request = prepared.request;
    if (request.id.queue == 0 || request.id.sequence == 0 || !request.snapshot ||
        !request.completion || request.memory.get() != state.memory.get())
        return failure("submission packet has invalid identity, ownership or memory domain");
    if (prepared.accesses.size() != prepared.access_storage.size())
        return failure("submission access and storage counts differ");
    if (const auto* clear = std::get_if<PreparedClear>(&prepared.operation);
        clear && clear->clear.no_op) {
        if (clear->clear.color_mask || clear->clear.clear_depth || clear->clear.clear_stencil ||
            !prepared.accesses.empty() || prepared.attachments.depth || prepared.attachments.stencil ||
            std::ranges::any_of(prepared.attachments.colors, [](const auto& color) { return color.has_value(); }))
            return failure("empty clear must not retain attachment work");
    }
    for (std::size_t index = 0; index < prepared.accesses.size(); ++index) {
        const auto& access = prepared.accesses[index];
        if (!access.range.lease || !access.range.bytes.size ||
            !prepared.access_storage[index])
            return failure("submission access lacks a retained range or prepared storage");
        if (contains_attachment(prepared, access.range.bytes) &&
            contains_buffer_binding(prepared, access.range.bytes))
            return failure("a render pass cannot access buffers through its attachment storage");
    }
    if (const auto* bindings = packet_bindings(prepared)) {
        for (const auto& binding : bindings->buffers) {
            if (binding.use != BufferUse::Storage) continue;
            if ((binding.buffer.offset & 3u) || !binding.buffer.size)
                return failure("storage buffer has an invalid native extent or alignment");
            if (!binding.buffer.key.bytes.backing.owner) {
                if (binding.buffer.size != 4 || binding.access != ResourceAccess::Read)
                    return failure("empty storage binding requires a guarded read-only placeholder");
                continue;
            }
            bool tracked = false;
            for (std::size_t i = 0; i != prepared.accesses.size(); ++i) {
                if (prepared.access_storage[i] == binding.buffer.storage &&
                    prepared.accesses[i].access == binding.access &&
                    overlaps(prepared.accesses[i].range.bytes, binding.buffer.key.bytes)) {
                    tracked = true;
                    break;
                }
            }
            if (!tracked)
                return failure("storage binding is missing its retained GPU access intent");
        }
        for (const auto& access : bindings->accesses) {
            if (!writes(access.access)) continue;
            if (contains_attachment(prepared, access.range.bytes))
                return failure("writable shader storage aliases a render attachment");
            for (const auto& other : bindings->accesses)
                if (&other != &access && overlaps(access.range.bytes, other.range.bytes))
                    return failure("writable shader storage aliases another shader resource");
        }
    }
    return renderer_success();
}

std::optional<NativeAttachment> native_attachment(
    const std::optional<ResidentTexture>& resident) {
    if (!resident) return NativeAttachment{};
    const auto texture = std::dynamic_pointer_cast<const TextureResource>(resident->storage);
    if (!texture || !texture->texture() || !texture->allocation()) return std::nullopt;
    const auto allocation = texture->allocation();
    const auto slices = allocation.arrayLength *
        ((allocation.textureType == MTLTextureTypeCube ||
          allocation.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    if (resident->key.first_level >= allocation.mipmapLevelCount ||
        resident->key.first_layer >= slices) return std::nullopt;
    // Sampling may use an aspect-only view. Render attachments must share the
    // packed allocation, with the selected subresource expressed explicitly.
    return NativeAttachment{texture->allocation(), resident->key.first_level,
                            resident->key.first_layer};
}

RendererStatus validate_attachments(PacketLease& packet) {
    auto& native = packet.native.targets;
    for (std::size_t slot = 0; slot < packet.prepared.attachments.colors.size(); ++slot) {
        auto attachment = native_attachment(packet.prepared.attachments.colors[slot]);
        if (!attachment) return failure("color attachment lacks native Metal storage");
        native.colors[slot] = *attachment;
    }
    auto depth = native_attachment(packet.prepared.attachments.depth);
    auto stencil = native_attachment(packet.prepared.attachments.stencil);
    if (!depth || !stencil)
        return failure("depth or stencil attachment lacks native Metal storage");
    native.depth = *depth;
    native.stencil = *stencil;
    return renderer_success();
}

struct TableSizes {
    NSUInteger buffers = 0, textures = 0, samplers = 0;
};

bool stage_supported(shader_input::Stage stage) {
    return stage == shader_input::Stage::Vertex ||
           stage == shader_input::Stage::Fragment;
}

TableSizes table_sizes(const Bindings& bindings, shader_input::Stage stage) {
    TableSizes sizes;
    if (stage == shader_input::Stage::Vertex && bindings.stages.stages.vertex &&
        bindings.stages.stages.vertex->draw_support_buffer_index)
        sizes.buffers = NSUInteger(*bindings.stages.stages.vertex->draw_support_buffer_index) + 1;
    const auto include_buffer = [&](const BufferBinding& binding) {
        if (binding.stage == stage) sizes.buffers = std::max<NSUInteger>(
            sizes.buffers, NSUInteger(binding.index) + 1);
    };
    for (const auto& binding : bindings.buffers) include_buffer(binding);
    for (const auto& binding : bindings.vertex_buffers) include_buffer(binding);
    for (const auto& binding : bindings.argument_buffers)
        if (binding.stage == stage) sizes.buffers = std::max<NSUInteger>(
            sizes.buffers, NSUInteger(binding.layout.buffer_index) + 1);
    for (const auto& binding : bindings.samplers)
        if (binding.stage == stage && !binding.argument_byte_offset)
            sizes.samplers = std::max<NSUInteger>(sizes.samplers, NSUInteger(binding.index) + 1);
    for (const auto& binding : bindings.textures) {
        if (binding.stage != stage) continue;
        if (!binding.argument_byte_offset)
            sizes.textures = std::max<NSUInteger>(sizes.textures,
                                                  NSUInteger(binding.texture_index) + 1);
        if (binding.sampler_index != shader::graphics_msl::TextureBinding::no_sampler)
            sizes.samplers = std::max<NSUInteger>(sizes.samplers,
                                                  NSUInteger(binding.sampler_index) + 1);
    }
    return sizes;
}

RendererResult<id<MTL4ArgumentTable>> make_table(
    id<MTLDevice> device, TableSizes sizes, bool vertex, StageTable& state,
    ArgumentTables& tables) {
    if (sizes.buffers > 31 || sizes.textures > 128 || sizes.samplers > 16)
        return failure("binding index exceeds the Metal argument-table limits");
    state.begin();
    if (state.table || (!sizes.buffers && !sizes.textures && !sizes.samplers))
        return state.table;
    auto* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 31;
    descriptor.maxTextureBindCount = 128;
    descriptor.maxSamplerStateBindCount = 16;
    descriptor.initializeBindings = YES;
    descriptor.supportAttributeStrides = vertex;
    NSError* error = nil;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:descriptor error:&error];
    if (!table) return failure("Metal argument-table creation failed",
                               CommandStatus::MemoryFault);
    state.table = table;
    state.vertex = vertex;
    tables.created.fetch_add(1, std::memory_order_relaxed);
    return table;
}

RendererResult<ArgumentSlice> allocate_arguments(Batch& batch, id<MTLDevice> device,
    const CompiledArgumentBufferLayout& layout, std::span<const std::uint64_t> members) {
    const auto alignment = std::max<std::uint64_t>(256, layout.alignment);
    const auto padding = [alignment](const ArgumentPage& page) {
        return (alignment - ((page.gpu + page.used) % alignment)) % alignment;
    };
    auto& pages = batch.lease->argument_pages;
    const auto fits = [&](const ArgumentPage& page) {
        const auto pad = padding(page);
        return page.used <= page.length && pad <= page.length - page.used &&
            layout.byte_extent <= page.length - page.used - pad;
    };
    if (pages.empty() || !fits(pages.back())) {
        auto acquired = batch.pool.acquire_page(device,
            std::uint64_t(layout.byte_extent) + alignment - 1);
        if (const auto* error = std::get_if<RendererError>(&acquired)) return *error;
        try {
            pages.push_back(std::get<ArgumentPage>(std::move(acquired)));
        } catch (const std::bad_alloc&) {
            return failure("argument arena tracking allocation failed", CommandStatus::MemoryFault);
        }
        // A page is retained by the batch before it joins residency. Failed
        // preparation may leave unused space, but never rewinds earlier slices.
        add_resident(batch, pages.back().buffer);
    }
    auto& page = pages.back();
    const auto offset = page.used + padding(page);
    std::memcpy(page.cpu + offset, members.data(), layout.byte_extent);
    page.used = offset + layout.byte_extent;
    batch.pool.slices_created.fetch_add(1, std::memory_order_relaxed);
    return ArgumentSlice{page.buffer, page.gpu + offset};
}

bool same_layout(const CompiledArgumentBufferLayout& left,
                 const CompiledArgumentBufferLayout& right) {
    return left.abi_version == right.abi_version && left.buffer_index == right.buffer_index &&
        left.byte_extent == right.byte_extent && left.alignment == right.alignment;
}

// A sampled copy is a packet-local read resource, never a guest writeback
// authority. The original attachment remains registered for the new GPU version.
RendererResult<id<MTLTexture>> feedback_texture(
    PacketLease& packet, const TextureBinding& binding,
    const TextureResource& resource) {
    const auto& sampled = binding.texture.key;
    bool copy = false;
    const auto check = [&](const std::optional<ResidentTexture>& attachment,
                           const char* kind, std::size_t slot) {
        if (!attachment || !overlaps(sampled.storage.bytes, attachment->key.storage.bytes)) return;
        const auto attached = std::dynamic_pointer_cast<const TextureResource>(attachment->storage);
        const auto& target = attachment->key;
        // Only shared native storage proves that these coordinates name the
        // same subresources. Different guest aliases may use different layouts.
        const bool same_allocation = attached && resource.allocation() == attached->allocation();
        const bool disjoint = same_allocation &&
            (sampled.first_level + sampled.level_count <= target.first_level ||
             target.first_level + target.level_count <= sampled.first_level ||
             sampled.first_layer + sampled.layer_count <= target.first_layer ||
             target.first_layer + target.layer_count <= sampled.first_layer);
        copy |= !disjoint;
        static const bool counting = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        if (counting && !disjoint) {
            // Feedback snapshots by attachment kind and storage sharing.
            static thread_local std::array<std::uint64_t, 4> kinds{};
            ++kinds[(kind[0] == 'c' ? 0 : 2) + (same_allocation ? 0 : 1)];
            if ((kinds[0] + kinds[1] + kinds[2] + kinds[3]) % 5000 == 0)
                armigo::diag::opt_in( "feedback color_same=%llu color_alias=%llu depth_same=%llu depth_alias=%llu "
                    "format=%u target_format=%u\n", (unsigned long long)kinds[0], (unsigned long long)kinds[1],
                    (unsigned long long)kinds[2], (unsigned long long)kinds[3],
                    unsigned(sampled.storage.descriptor.format), unsigned(target.storage.descriptor.format));
        }
        const auto* trace = std::getenv("ARMIGO_GPU_SUBMISSION_TRACE");
        if (trace && trace[0] == '1') {
            armigo::diag::opt_in(
                "[metal-feedback] sequence=%llu stage=%u texture=%u attachment=%s%zu mode=%s same_allocation=%u\n",
                static_cast<unsigned long long>(packet.prepared.request.id.sequence),
                unsigned(binding.stage), binding.texture_index, kind, slot,
                disjoint ? "subresource-view" : "snapshot", unsigned(same_allocation));
            const auto describe = [](const char* name, const TextureViewKey& key) {
                const auto& b = key.storage.bytes;
                const auto& d = key.storage.descriptor;
                armigo::diag::opt_in(
                    "[metal-feedback] %s backing=%llu:%llu:%llu offset=0x%llx size=0x%llx address=0x%llx "
                    "format=%u type=%u layout=%u size=%ux%ux%u levels=%u base_level=%u layers=%u "
                    "view_level=%u+%u view_layer=%u+%u aspect=%u gob=%u,%u row=%llu stride=%llu\n",
                    name, static_cast<unsigned long long>(b.backing.owner),
                    static_cast<unsigned long long>(b.backing.allocation),
                    static_cast<unsigned long long>(b.backing.generation),
                    static_cast<unsigned long long>(b.offset), static_cast<unsigned long long>(b.size),
                    static_cast<unsigned long long>(d.address), unsigned(d.format), unsigned(d.type),
                    unsigned(d.layout), d.width, d.height, d.depth, d.levels, d.base_level, d.layers,
                    key.first_level, key.level_count, key.first_layer, key.layer_count, unsigned(key.aspect),
                    d.block_height_log2, d.block_depth_log2,
                    static_cast<unsigned long long>(d.row_bytes), static_cast<unsigned long long>(d.layer_stride));
            };
            describe("sample", sampled);
            describe("target", target);
        }
    };
    for (std::size_t slot = 0; slot != packet.prepared.attachments.colors.size(); ++slot)
        check(packet.prepared.attachments.colors[slot], "color", slot);
    check(packet.prepared.attachments.depth, "depth", 0);
    check(packet.prepared.attachments.stencil, "stencil", 0);
    // Bind a sampled texture that shares storage with an attachment as is: a
    // mip chain that renders one
    // level while sampling the others reads the stored levels, and an alias
    // keeps its own contents as elsewhere. Only sampling the very texels the
    // pass writes differs, which is undefined on Maxwell too. The snapshot
    // copy that ended the pass for every such draw remains behind
    // ARMIGO_FEEDBACK_SNAPSHOTS=1 for comparison runs.
    static const bool snapshots = [] {
        const char* value = std::getenv("ARMIGO_FEEDBACK_SNAPSHOTS");
        return value && value[0] == '1';
    }();
    if (!copy || !snapshots) return resource.texture();

    const auto source = resource.allocation();
    const auto view = resource.texture();
    const auto slices = source.arrayLength *
        ((source.textureType == MTLTextureTypeCube || source.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    if (source.sampleCount != 1 || sampled.first_level >= source.mipmapLevelCount ||
        sampled.level_count > source.mipmapLevelCount - sampled.first_level ||
        sampled.first_layer >= slices || sampled.layer_count > slices - sampled.first_layer)
        return failure("attachment feedback has unsupported samples or subresources");
    auto* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = view.textureType;
    descriptor.pixelFormat = source.pixelFormat;
    descriptor.width = view.width;
    descriptor.height = view.height;
    descriptor.depth = view.depth;
    descriptor.mipmapLevelCount = sampled.level_count;
    descriptor.arrayLength = view.arrayLength;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> snapshot = [source.device newTextureWithDescriptor:descriptor];
    if (!snapshot) return failure("attachment feedback snapshot allocation failed", CommandStatus::MemoryFault);
    const auto limit = argument_bytes_limit(packet.prepared);
    if (packet.native.argument_buffer_bytes > limit ||
        packet.native.feedback_bytes > limit - packet.native.argument_buffer_bytes ||
        snapshot.allocatedSize > limit - packet.native.argument_buffer_bytes - packet.native.feedback_bytes)
        return failure("attachment feedback snapshots exceed the staging budget", CommandStatus::BudgetExceeded);
    id<MTLTexture> sampled_snapshot = [snapshot newTextureViewWithPixelFormat:view.pixelFormat
        textureType:view.textureType levels:NSMakeRange(0, sampled.level_count)
        slices:NSMakeRange(0, sampled.layer_count) swizzle:view.swizzle];
    if (!sampled_snapshot) return failure("attachment feedback snapshot view is unsupported");
    packet.native.feedback_bytes += snapshot.allocatedSize;
    packet.native.feedback_copies.push_back({source, snapshot, sampled.first_level,
        sampled.first_layer, sampled.level_count, sampled.layer_count});
    // Preparation can still fail, including a later binding or reservation.
    // Keep temporary allocations out of shared batch ownership until accepted.
    if (!packet.native.feedback_residency) {
        auto* residency = [MTLResidencySetDescriptor new];
        residency.initialCapacity = 4;
        packet.native.feedback_residency = [source.device newResidencySetWithDescriptor:residency error:nil];
        if (!packet.native.feedback_residency)
            return failure("attachment feedback residency allocation failed", CommandStatus::MemoryFault);
    }
    [packet.native.feedback_objects addObject:sampled_snapshot];
    [packet.native.feedback_residency addAllocation:snapshot];
    return sampled_snapshot;
}

RendererStatus encode_draw_call(Batch& batch, PacketLease& packet, id<MTLDevice> device,
                                std::span<const DrawSegment> segments,
                                const ResidentBuffer* index_override = nullptr,
                                const RenderState* state_override = nullptr,
                                id<MTLRenderPipelineState> pipeline_override = nil);

// A repeat rebinds only its changed buffer slots (the tables still hold the
// source draw's bindings) and draws its own segments in the source's state.
RendererStatus encode_repeat(Batch& batch, PacketLease& source, const RepeatJob& repeat,
                             id<MTLDevice> device, ArgumentTables& tables) {
    for (const auto& binding : repeat.changed) {
        id<MTLBuffer> buffer = resident_native_buffer(binding.buffer.storage);
        if (!buffer || binding.buffer.offset > buffer.length ||
            binding.buffer.size > buffer.length - binding.buffer.offset)
            return failure("repeat buffer binding lacks a valid native Metal slice");
        add_resident(batch, buffer);
        auto& table = binding.stage == shader_input::Stage::Vertex ? tables.vertex : tables.fragment;
        table.address(buffer.gpuAddress + binding.buffer.offset, binding.index);
    }
    const auto& draw = std::get<PreparedDraw>(source.prepared.operation);
    const auto& layouts = repeat.layout ? repeat.layout->key.buffers : draw.pipeline.key.buffers;
    for (const auto& binding : repeat.vertex_buffers) {
        id<MTLBuffer> buffer = resident_native_buffer(binding.buffer.storage);
        const auto layout = std::ranges::find_if(layouts,
            [&](const VertexBufferLayoutKey& key) { return key.buffer_index == binding.index; });
        if (!buffer || layout == layouts.end())
            return failure("repeat vertex binding has no native slice or pipeline layout");
        add_resident(batch, buffer);
        tables.vertex.address(buffer.gpuAddress + binding.buffer.offset, binding.index, true, layout->stride);
    }
    if (repeat.index_buffer) {
        id<MTLBuffer> buffer = resident_native_buffer(repeat.index_buffer->storage);
        if (!buffer) return failure("repeat index binding has no native slice");
        add_resident(batch, buffer);
    }
    return encode_draw_call(batch, source, device, repeat.segments,
                            repeat.index_buffer ? &*repeat.index_buffer : nullptr, repeat.state.get(),
                            repeat.pipeline);
}

RendererStatus bind_table(PacketLease& packet, Batch& batch,
                          shader_input::Stage stage,
                          StageTable& table) {
    if (!table.table) return renderer_success();
    const auto& bindings = *packet_bindings(packet.prepared);
    const auto* draw = std::get_if<PreparedDraw>(&packet.prepared.operation);
    id<MTLDevice> device = packet.native.pipeline ? packet.native.pipeline.device
                                                  : packet.native.compute_pipeline.device;
    const auto bind_buffer = [&](const BufferBinding& binding,
                                 bool attribute) -> RendererStatus {
        if (binding.stage != stage) return renderer_success();
        id<MTLBuffer> buffer = resident_native_buffer(binding.buffer.storage);
        if (!buffer || binding.buffer.offset > buffer.length ||
            binding.buffer.size > buffer.length - binding.buffer.offset)
            return failure("buffer binding lacks a valid native Metal slice");
        add_resident(batch, buffer);
        const auto address = buffer.gpuAddress + binding.buffer.offset;
        if (attribute) {
            if (!draw) return failure("vertex binding outside a draw packet");
            const auto found = std::ranges::find_if(draw->pipeline.key.buffers,
                [&](const VertexBufferLayoutKey& layout) {
                    return layout.buffer_index == binding.index;
                });
            if (found == draw->pipeline.key.buffers.end())
                return failure("vertex binding has no canonical pipeline layout");
            table.address(address, binding.index, true, found->stride);
        } else {
            table.address(address, binding.index);
        }
        return renderer_success();
    };
    for (const auto& binding : bindings.buffers) {
        auto status = bind_buffer(binding, false);
        if (!succeeded(status)) return status;
    }
    for (const auto& binding : bindings.vertex_buffers) {
        auto status = bind_buffer(binding, true);
        if (!succeeded(status)) return status;
    }
    std::optional<CompiledArgumentBufferLayout> argument_layout;
    std::vector<std::uint64_t> argument_members;
    for (const auto& binding : bindings.argument_buffers) {
        if (binding.stage != stage) continue;
        const auto& layout = binding.layout;
        if (argument_layout || layout.abi_version != 1 || !layout.byte_extent ||
            layout.buffer_index >= 31 || layout.alignment < 8 ||
            (layout.alignment & (layout.alignment - 1)) || (layout.byte_extent & 7u))
            return failure("argument-buffer layout is invalid or duplicated");
        const auto bytes = std::uint64_t(layout.byte_extent) + layout.alignment - 1;
        const auto limit = argument_bytes_limit(packet.prepared);
        if (packet.native.argument_buffer_bytes > limit ||
            packet.native.feedback_bytes > limit - packet.native.argument_buffer_bytes ||
            bytes > limit - packet.native.argument_buffer_bytes - packet.native.feedback_bytes)
            return failure("argument buffers exceed the staging budget", CommandStatus::BudgetExceeded);
        // Charge the original logical per-packet extent even when an immutable
        // slice is reused. Physical arena capacity is shared across the batch.
        packet.native.argument_buffer_bytes += bytes;
        argument_layout = layout;
        try {
            argument_members.resize(layout.byte_extent / sizeof(std::uint64_t), 0);
        } catch (const std::bad_alloc&) {
            return failure("argument member staging allocation failed", CommandStatus::MemoryFault);
        }
    }
    const auto write_member = [&](std::uint32_t offset, MTLResourceID resource) -> RendererStatus {
        static_assert(sizeof(MTLResourceID) == sizeof(std::uint64_t));
        if (!argument_layout || (offset & 7u) || offset > argument_layout->byte_extent ||
            argument_layout->byte_extent - offset < 8)
            return failure("argument-buffer member exceeds its immutable slice");
        argument_members[offset / 8] = resource._impl;
        return renderer_success();
    };
    for (const auto& resource : bindings.custom_border_textures) {
        const auto texture = std::dynamic_pointer_cast<const TextureResource>(resource);
        if (!texture || !texture->texture() || !texture->allocation())
            return failure("custom border coverage lacks native texture ownership");
        add_resident(batch, texture->allocation());
    }
    for (const auto& binding : bindings.samplers) {
        if (binding.stage != stage) continue;
        const auto sampler = std::dynamic_pointer_cast<const MetalSamplerResource>(binding.sampler);
        if (!sampler || !sampler->sampler()) return failure("sampler binding lacks native ownership");
        if (binding.argument_byte_offset) {
            const auto status = write_member(*binding.argument_byte_offset, sampler->sampler().gpuResourceID);
            if (!succeeded(status)) return status;
        } else table.sampler(sampler->sampler().gpuResourceID, binding.index);
    }
    for (const auto& binding : bindings.textures) {
        if (binding.stage != stage) continue;
        const auto texture = std::dynamic_pointer_cast<const TextureResource>(
            binding.texture.storage);
        const auto sampler = std::dynamic_pointer_cast<const MetalSamplerResource>(
            binding.sampler);
        const bool needs_sampler =
            binding.sampler_index != shader::graphics_msl::TextureBinding::no_sampler;
        if (!texture || !texture->texture() || !texture->allocation() ||
            (needs_sampler && (!sampler || !sampler->sampler())))
            return failure("texture binding lacks native texture or sampler ownership");
        add_resident(batch, texture->allocation());
        auto sampled = feedback_texture(packet, binding, *texture);
        if (const auto* error = std::get_if<RendererError>(&sampled)) return *error;
        const auto resource_id = std::get<id<MTLTexture>>(sampled).gpuResourceID;
        if (binding.argument_byte_offset) {
            const auto status = write_member(*binding.argument_byte_offset, resource_id);
            if (!succeeded(status)) return status;
        } else table.texture(resource_id, binding.texture_index);
        if (needs_sampler)
            table.sampler(sampler->sampler().gpuResourceID, binding.sampler_index);
    }
    if (argument_layout) {
        const auto slot = stage == shader_input::Stage::Vertex ? 0u :
            stage == shader_input::Stage::Fragment ? 1u : 2u;
        auto& cached = batch.arguments[slot];
        // Resolve every resource, transfer and feedback snapshot before this
        // comparison. Only values and an arena slice are cached, never guest
        // storage owners; the current packet independently retains its inputs.
        if (cached.slice.buffer && same_layout(cached.layout, *argument_layout) &&
            cached.members == argument_members) {
            batch.pool.slices_reused.fetch_add(1, std::memory_order_relaxed);
        } else {
            auto allocated = allocate_arguments(batch, device, *argument_layout, argument_members);
            if (const auto* error = std::get_if<RendererError>(&allocated)) return *error;
            cached.layout = *argument_layout;
            cached.members = std::move(argument_members);
            cached.slice = std::get<ArgumentSlice>(std::move(allocated));
        }
        table.address(cached.slice.address, argument_layout->buffer_index);
    }
    return renderer_success();
}

RendererStatus validate_draw(PacketLease& packet) {
    auto& draw = std::get<PreparedDraw>(packet.prepared.operation);
    const auto pipeline = std::dynamic_pointer_cast<const MetalRenderPipelineResource>(
        draw.pipeline.pipeline);
    if (!pipeline || !pipeline->pipeline())
        return failure("draw pipeline lease has no native Metal pipeline");
    packet.native.pipeline = pipeline->pipeline();
    if (draw.depth_stencil) {
        const auto depth = std::dynamic_pointer_cast<const MetalDepthStencilResource>(
            draw.depth_stencil);
        if (!depth || !depth->depth_stencil())
            return failure("depth-stencil lease has no native Metal state");
        packet.native.depth_stencil = depth->depth_stencil();
    }
    for (const auto& binding : draw.bindings.buffers)
        if (!stage_supported(binding.stage))
            return failure("buffer binding uses an unsupported graphics stage");
    for (const auto& binding : draw.bindings.vertex_buffers)
        if (binding.stage != shader_input::Stage::Vertex)
            return failure("vertex buffer binding is not a vertex-stage resource");
    for (const auto& binding : draw.bindings.textures)
        if (!stage_supported(binding.stage))
            return failure("texture binding uses an unsupported graphics stage");

    for (const auto& binding : draw.bindings.samplers)
        if (!stage_supported(binding.stage))
            return failure("sampler binding uses an unsupported graphics stage");
    for (const auto& binding : draw.bindings.argument_buffers)
        if (!stage_supported(binding.stage))
            return failure("argument buffer uses an unsupported graphics stage");

    const auto& stages = draw.bindings.stages.stages;
    if (stages.fragment && stages.fragment->draw_support_buffer_index)
        return failure("Draw-parameter support requires a vertex stage");
    if ((stages.vertex && stages.vertex->guest_memory) || (stages.fragment && stages.fragment->guest_memory))
        return failure("Guest-memory window tables are supported by the NVN driver only");
    const auto draw_support_index = stages.vertex
        ? stages.vertex->draw_support_buffer_index : std::optional<std::uint8_t>{};
    if (draw_support_index) {
        const auto index = *draw_support_index;
        const auto collides = [index](const BufferBinding& binding) {
            return binding.stage == shader_input::Stage::Vertex && binding.index == index;
        };
        if (index >= 31 || std::ranges::any_of(draw.bindings.buffers, collides) ||
            std::ranges::any_of(draw.bindings.vertex_buffers, collides) ||
            std::ranges::any_of(draw.bindings.argument_buffers, [index](const auto& binding) {
                return binding.stage == shader_input::Stage::Vertex &&
                       binding.layout.buffer_index == index;
            }))
            return failure("Draw support buffer overlaps another vertex binding or exceeds Metal limits");
    }
    for (const auto& segment : draw.draw.segments) {
        if (segment.instance_index > segment.base_instance ||
            segment.base_instance - segment.instance_index != draw.draw.source.base_instance ||
            segment.instance_index != draw.draw.source.instance_index)
            return failure("Draw segment does not preserve the guest first instance and continuation index");
    }

    const auto& state = draw.state;
    if (state.rasterizer.cull == CullMode::FrontAndBack ||
        state.rasterizer.front_fill != state.rasterizer.back_fill ||
        state.rasterizer.front_fill == FillMode::Point)
        return failure("draw contains rasterizer state unsupported by Metal encoding");
    for (const auto& viewport : state.viewports)
        if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) ||
            !std::isfinite(viewport.width) || !std::isfinite(viewport.height) ||
            !std::isfinite(viewport.near_depth) ||
            !std::isfinite(viewport.far_depth) || viewport.width < 0 ||
            viewport.height < 0)
            return failure("draw viewport is not finite and nonnegative");
    const auto& area = packet.prepared.targets.area;
    if (area.x < 0 || area.y < 0 ||
        area.width > std::numeric_limits<std::uint64_t>::max() -
            std::uint64_t(area.x) ||
        area.height > std::numeric_limits<std::uint64_t>::max() -
            std::uint64_t(area.y))
        return failure("render area cannot be represented by Metal");
    const auto area_right = std::uint64_t(area.x) + area.width;
    const auto area_bottom = std::uint64_t(area.y) + area.height;
    for (auto& scissor : draw.state.scissors) {
        // Planner intersections can retain an outside origin for empty
        // coverage. Clip both endpoints in wide arithmetic before converting
        // to Metal's unsigned coordinates; never turn an empty rect nonempty.
        const auto clip_axis = [](std::int64_t& origin, std::uint64_t& size,
                                  std::int64_t minimum, std::uint64_t maximum) {
            const auto first = std::clamp<__int128>(origin, minimum, maximum);
            const auto last = std::clamp<__int128>(
                __int128(origin) + size, minimum, maximum);
            origin = static_cast<std::int64_t>(first);
            size = static_cast<std::uint64_t>(last - first);
        };
        clip_axis(scissor.x, scissor.width, area.x, area_right);
        clip_axis(scissor.y, scissor.height, area.y, area_bottom);
    }
    if (draw.draw.topology != PrimitiveTopology::Points &&
        draw.draw.topology != PrimitiveTopology::Lines &&
        draw.draw.topology != PrimitiveTopology::LineStrip &&
        draw.draw.topology != PrimitiveTopology::Triangles &&
        draw.draw.topology != PrimitiveTopology::TriangleStrip)
        return failure("draw topology was not normalized before Metal submission");
    if (draw.bindings.index_buffer && !resident_native_buffer(draw.bindings.index_buffer->storage))
        return failure("index binding lacks native Metal storage");
    if (draw.draw.indices) {
        if (!draw.bindings.index_buffer)
            return failure("indexed draw has no resident index buffer");
        const auto type = index_type(draw.draw.indices->encoded_format);
        if (!type) return failure("indexed draw uses an unsupported index type");
        const std::uint64_t element_size = *type == MTLIndexTypeUInt16 ? 2 : 4;
        for (const auto& segment : draw.draw.segments) {
            if (segment.first > std::numeric_limits<std::uint64_t>::max() / element_size ||
                std::uint64_t(segment.first) * element_size >
                    draw.bindings.index_buffer->size ||
                std::uint64_t(segment.count) * element_size >
                    draw.bindings.index_buffer->size -
                    std::uint64_t(segment.first) * element_size)
                return failure("indexed draw segment exceeds its resident buffer");
        }
    } else if (draw.bindings.index_buffer) {
        return failure("non-indexed draw unexpectedly retains an index buffer");
    }
    return renderer_success();
}

RendererStatus prepare_draw(PacketLease& packet, Batch& batch,
                            id<MTLDevice> device, ArgumentTables& tables) {
    const auto& draw = std::get<PreparedDraw>(packet.prepared.operation);
    const auto& stages = draw.bindings.stages.stages;
    const auto draw_support_index = stages.vertex
        ? stages.vertex->draw_support_buffer_index : std::optional<std::uint8_t>{};
    auto vertex = make_table(device,
        table_sizes(draw.bindings, shader_input::Stage::Vertex), true, tables.vertex, tables);
    if (const auto* error = std::get_if<RendererError>(&vertex)) return *error;
    auto fragment = make_table(device,
        table_sizes(draw.bindings, shader_input::Stage::Fragment), false, tables.fragment, tables);
    if (const auto* error = std::get_if<RendererError>(&fragment)) return *error;
    packet.native.vertex_table = std::get<id<MTL4ArgumentTable>>(vertex);
    packet.native.fragment_table = std::get<id<MTL4ArgumentTable>>(fragment);
    auto status = bind_table(packet, batch, shader_input::Stage::Vertex,
                             tables.vertex);
    if (!succeeded(status)) return status;
    status = bind_table(packet, batch, shader_input::Stage::Fragment,
                        tables.fragment);
    if (!succeeded(status)) return status;
    if (draw_support_index) {
        // Packet-owned storage stays immutable across subsequent continuation
        // draws and argument-table updates until this packet has completed.
        const std::uint32_t first_instance = draw.draw.source.base_instance;
        packet.native.draw_support = [device newBufferWithBytes:&first_instance
            length:sizeof(first_instance) options:MTLResourceStorageModeShared];
        if (!packet.native.draw_support)
            return failure("Draw support allocation failed", CommandStatus::MemoryFault);
        tables.vertex.address(packet.native.draw_support.gpuAddress, *draw_support_index);
    }
    tables.vertex.finish();
    tables.fragment.finish();
    if (draw.bindings.index_buffer) {
        id<MTLBuffer> index = resident_native_buffer(draw.bindings.index_buffer->storage);
        if (!index) return failure("index binding lacks native Metal storage");
        add_resident(batch, index);
    }
    return renderer_success();
}

// A compute launch binds its stage resources through the draw path's argument
// encoding and dispatches the queue-meta-data grid with the reflected block.
RendererStatus validate_dispatch(PacketLease& packet, id<MTLDevice> device) {
    auto& dispatch = std::get<PreparedDispatch>(packet.prepared.operation);
    const auto pipeline = std::dynamic_pointer_cast<const MetalComputePipelineResource>(
        dispatch.pipeline.pipeline);
    if (!pipeline || !pipeline->pipeline())
        return failure("compute pipeline lease has no native Metal pipeline");
    packet.native.compute_pipeline = pipeline->pipeline();
    const auto& bindings = dispatch.bindings;
    const auto compute = [](const auto& binding) {
        return binding.stage == shader_input::Stage::Compute;
    };
    if (!std::ranges::all_of(bindings.buffers, compute) ||
        !std::ranges::all_of(bindings.textures, compute) ||
        !std::ranges::all_of(bindings.samplers, compute) ||
        !std::ranges::all_of(bindings.argument_buffers, compute) ||
        !bindings.vertex_buffers.empty() || bindings.index_buffer)
        return failure("compute dispatch carries a non-compute binding");
    for (const auto extent : dispatch.grid)
        if (!extent) return failure("compute dispatch grid dimensions must be nonzero");
    const auto limits = device.maxThreadsPerThreadgroup;
    const auto& block = dispatch.block;
    if (!block[0] || !block[1] || !block[2] || block[0] > limits.width ||
        block[1] > limits.height || block[2] > limits.depth ||
        std::uint64_t(block[0]) * block[1] * block[2] > packet.native.compute_pipeline.maxTotalThreadsPerThreadgroup)
        return failure("compute dispatch block exceeds the device or pipeline threadgroup limits");
    return renderer_success();
}

RendererStatus prepare_dispatch(PacketLease& packet, Batch& batch, id<MTLDevice> device,
                                ArgumentTables& tables) {
    const auto& bindings = std::get<PreparedDispatch>(packet.prepared.operation).bindings;
    auto table = make_table(device, table_sizes(bindings, shader_input::Stage::Compute),
                            false, tables.compute, tables);
    if (const auto* error = std::get_if<RendererError>(&table)) return *error;
    packet.native.compute_table = std::get<id<MTL4ArgumentTable>>(table);
    auto status = bind_table(packet, batch, shader_input::Stage::Compute, tables.compute);
    if (succeeded(status)) tables.compute.finish();
    return status;
}

// An accepted packet's feedback allocations
// join the batch's residency set, which is committed before the batch is
// submitted. Metal caps the residency sets one command buffer may use at 32,
// and a deferred batch holds many packets. The packet lease keeps them alive.
void adopt_residency(Batch& batch, id<MTLResidencySet> set) {
    if (!set) return;
    for (id<MTLAllocation> allocation in set.allAllocations) add_resident(batch, allocation);
}

RendererStatus create_residency(Batch& batch, id<MTLDevice> device) {
    if (batch.residency) return renderer_success();
    auto* descriptor = [MTLResidencySetDescriptor new];
    descriptor.initialCapacity = 32;
    NSError* error = nil;
    batch.residency = [device newResidencySetWithDescriptor:descriptor error:&error];
    if (!batch.residency)
        return failure("submission residency-set creation failed",
                       CommandStatus::MemoryFault);
    [batch.lease->objects addObject:batch.residency];
    return renderer_success();
}

// Admission checks immutable native leases and layout limits without changing
// shared tables, residency, command buffers or the resource registry.
// Per-packet admission keeps the checks whose failure the encoder could not
// report (argument-table limits and layouts) and snapshots pending texture
// transfers; native ownership of each resource is established where the
// planner creates it and is not re-verified per draw.
RendererStatus validate_bindings(PacketLease& packet) {
    const auto& prepared = packet.prepared;
    const auto* bindings = packet_bindings(prepared);
    if (!bindings) return renderer_success();
    std::uint64_t argument_bytes = 0;
    for (const auto stage : {shader_input::Stage::Vertex, shader_input::Stage::Fragment,
                             shader_input::Stage::Compute}) {
        const auto sizes = table_sizes(*bindings, stage);
        if (sizes.buffers > 31 || sizes.textures > 128 || sizes.samplers > 16)
            return failure("binding index exceeds the Metal argument-table limits");
        const CompiledArgumentBufferLayout* layout = nullptr;
        for (const auto& binding : bindings->argument_buffers) {
            if (binding.stage != stage) continue;
            const auto& value = binding.layout;
            if (layout || value.abi_version != 1 || !value.byte_extent ||
                value.buffer_index >= 31 || value.alignment < 8 ||
                (value.alignment & (value.alignment - 1)) || (value.byte_extent & 7u))
                return failure("argument-buffer layout is invalid or duplicated");
            const auto bytes = std::uint64_t(value.byte_extent) + value.alignment - 1;
            const auto limit = argument_bytes_limit(prepared);
            if (argument_bytes > limit || bytes > limit - argument_bytes)
                return failure("argument buffers exceed the staging budget", CommandStatus::BudgetExceeded);
            argument_bytes += bytes;
            layout = &value;
        }
    }
    for (const auto& binding : bindings->textures) {
        if (!binding.texture.storage ||
            (binding.sampler_index != shader::graphics_msl::TextureBinding::no_sampler && !binding.sampler))
            return failure("texture binding lacks native texture or sampler ownership");
        for (const auto& transfer : binding.texture.transfers) {
            const auto copy = std::static_pointer_cast<const SubresourceCopy>(transfer);
            // The cache-visible flag belongs to the coordinator. Snapshot
            // pending plans here; workers only consult their batch set.
            if (copy && !copy->encoded && std::ranges::find(packet.native.subresource_copies, copy) ==
                                              packet.native.subresource_copies.end())
                packet.native.subresource_copies.push_back(copy);
        }
    }
    for (const auto& list : {std::span<const BufferBinding>(bindings->buffers),
                             std::span<const BufferBinding>(bindings->vertex_buffers)})
        for (const auto& binding : list)
            if (!binding.buffer.storage) return failure("buffer binding lacks a native Metal slice");
    return renderer_success();
}

RendererStatus validate_native(PacketLease& packet, id<MTLDevice> device) {
    if (const auto* clear = std::get_if<PreparedClear>(&packet.prepared.operation);
        clear && clear->clear.no_op) return renderer_success();
    auto status = validate_bindings(packet);
    if (!succeeded(status)) return status;
    if (std::holds_alternative<PreparedDispatch>(packet.prepared.operation))
        return validate_dispatch(packet, device);
    status = validate_attachments(packet);
    if (!succeeded(status)) return status;
    if (std::holds_alternative<PreparedDraw>(packet.prepared.operation)) {
        // The pass descriptor is built (and checked) when a pass opens.
        return validate_draw(packet);
    }

    const auto& clear = std::get<PreparedClear>(packet.prepared.operation).clear;
    if (clear.color_mask) {
        for (std::size_t slot = 0; slot < packet.prepared.targets.colors.size(); ++slot) {
            const auto& target = packet.prepared.targets.colors[slot];
            if (target && target->physical_target == clear.target.physical_target) {
                packet.native.clear.color = packet.native.targets.colors[slot].texture;
                break;
            }
        }
        if (!packet.native.clear.color)
            return failure("clear target has no matching resident color attachment");
    }
    if (clear.clear_depth) packet.native.clear.depth = packet.native.targets.depth.texture;
    if (clear.clear_stencil) packet.native.clear.stencil = packet.native.targets.stencil.texture;
    if ((clear.clear_depth && !packet.native.clear.depth) ||
        (clear.clear_stencil && !packet.native.clear.stencil))
        return failure("clear target has no matching depth-stencil attachment");
    packet.native.clear.level = clear.target.level;
    packet.native.clear.slice = clear.target.layer;
    if (clear.region.x < 0 || clear.region.y < 0 ||
        std::uint64_t(clear.region.x) > clear.target.texture.width ||
        std::uint64_t(clear.region.y) > clear.target.texture.height ||
        clear.region.width > clear.target.texture.width -
            std::uint64_t(clear.region.x) ||
        clear.region.height > clear.target.texture.height -
            std::uint64_t(clear.region.y))
        return failure("clear region exceeds the admitted attachment");
    status = validate_clear(clear, packet.native.clear);
    if (!succeeded(status)) return status;
    auto descriptor = pass_descriptor(packet);
    if (const auto* error = std::get_if<RendererError>(&descriptor)) return *error;
    return renderer_success();
}

RendererStatus prepare_native(PacketLease& packet, Batch& batch,
                              id<MTLDevice> device, ArgumentTables& tables) {
    if (const auto* clear = std::get_if<PreparedClear>(&packet.prepared.operation);
        clear && clear->clear.no_op) return renderer_success();
    auto status = create_residency(batch, device);
    if (!succeeded(status)) return status;
    for (const auto& copy : packet.native.subresource_copies) {
        if (batch.encoded_copies.contains(copy.get())) continue;
        add_resident(batch, copy->source);
        add_resident(batch, copy->destination);
        if (copy->staging) add_resident(batch, copy->staging);
    }
    if (std::holds_alternative<PreparedDispatch>(packet.prepared.operation))
        return prepare_dispatch(packet, batch, device, tables);
    for (const auto& color : packet.native.targets.colors)
        if (color.texture) add_resident(batch, color.texture);
    if (packet.native.targets.depth.texture) add_resident(batch, packet.native.targets.depth.texture);
    if (packet.native.targets.stencil.texture) add_resident(batch, packet.native.targets.stencil.texture);
    if (std::holds_alternative<PreparedDraw>(packet.prepared.operation))
        return prepare_draw(packet, batch, device, tables);
    return renderer_success();
}

MTLPrimitiveType primitive_type(PrimitiveTopology topology) {
    switch (topology) {
        case PrimitiveTopology::Points: return MTLPrimitiveTypePoint;
        case PrimitiveTopology::Lines: return MTLPrimitiveTypeLine;
        case PrimitiveTopology::LineStrip: return MTLPrimitiveTypeLineStrip;
        case PrimitiveTopology::Triangles: return MTLPrimitiveTypeTriangle;
        case PrimitiveTopology::TriangleStrip: return MTLPrimitiveTypeTriangleStrip;
        default: return MTLPrimitiveTypeTriangle;
    }
}

std::optional<MTLIndexType> index_type(IndexFormat format) {
    if (format == IndexFormat::U16) return MTLIndexTypeUInt16;
    if (format == IndexFormat::U32) return MTLIndexTypeUInt32;
    return std::nullopt;
}

// Diagnostic counts of passes ended by: command buffer end, sub-resource
// copies, attachment feedback copies, compute dispatch.
thread_local std::array<std::uint64_t, 4> close_sites{};

RendererStatus close_render(Batch& batch) {
    if (!batch.render) return renderer_success();
    [batch.render barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll
                     visibilityOptions:MTL4VisibilityOptionDevice];
    [batch.render endEncoding];
    batch.render = nil;
    batch.pass.reset();
    batch.pass_targets = {};
    batch.pass_accesses.clear();
    return renderer_success();
}

RendererStatus close_commands(Batch& batch) {
    if (batch.render) ++close_sites[0];
    close_render(batch);
    if (!batch.commands) return renderer_success();
    [batch.commands endCommandBuffer];
    batch.buffers.push_back(batch.commands);
    batch.commands = nil;
    batch.allocator = nil;
    return renderer_success();
}

RendererStatus ensure_commands(Batch& batch, id<MTLDevice> device) {
    if (batch.commands) return renderer_success();
    auto acquired = batch.pool.acquire_commands(device);
    if (const auto* error = std::get_if<RendererError>(&acquired)) return *error;
    batch.lease->command_bundles.push_back(std::get<CommandBundle>(std::move(acquired)));
    const auto& bundle = batch.lease->command_bundles.back();
    batch.allocator = bundle.allocator;
    batch.commands = bundle.commands;
    [batch.commands beginCommandBufferWithAllocator:batch.allocator];
    [batch.commands useResidencySet:batch.residency];
    return renderer_success();
}

bool pass_hazard(const Batch& batch, const PreparedRender& prepared) {
    for (const auto& incoming : prepared.accesses) {
        const bool incoming_attachment = contains_attachment(prepared, incoming.range.bytes);
        if (batch.pass_accesses.conflicts(incoming.range.bytes, incoming.access,
                                         incoming_attachment)) return true;
    }
    return false;
}

RendererResult<MTL4RenderPassDescriptor*> pass_descriptor(const PacketLease& packet) {
    const auto* clear = std::get_if<PreparedClear>(&packet.prepared.operation);
    if (!clear || !clear->clear.fold_into_load)
        return make_render_pass(packet.prepared.targets, packet.native.targets, nullptr);
    auto targets = packet.prepared.targets;
    RenderPassClearValues values;
    if (clear->clear.color_mask) {
        auto converted = metal_clear_color(clear->clear.color_bits,
                                           clear->clear.target.texture.format);
        if (const auto* error = std::get_if<CommandResult>(&converted))
            return failure(error->detail);
        for (std::size_t slot = 0; slot < packet.prepared.targets.colors.size(); ++slot) {
            const auto& target = packet.prepared.targets.colors[slot];
            if (target && target->physical_target == clear->clear.target.physical_target) {
                values.color[slot] = std::get<std::array<double, 4>>(converted);
                targets.colors[slot]->load = AttachmentLoad::Clear;
            }
        }
    }
    if (clear->clear.clear_depth && targets.depth)
        targets.depth->load = AttachmentLoad::Clear;
    if (clear->clear.clear_stencil && targets.stencil)
        targets.stencil->load = AttachmentLoad::Clear;
    values.depth = clear->clear.depth_value;
    values.stencil = clear->clear.stencil_value;
    return make_render_pass(targets, packet.native.targets, &values);
}

bool same_native_attachment(const NativeAttachment& left,
                            const NativeAttachment& right) {
    return left.texture == right.texture && left.level == right.level &&
           left.slice == right.slice;
}

bool same_native_targets(const NativeRenderTargets& left,
                         const NativeRenderTargets& right) {
    for (std::size_t slot = 0; slot < left.colors.size(); ++slot)
        if (!same_native_attachment(left.colors[slot], right.colors[slot]))
            return false;
    return same_native_attachment(left.depth, right.depth) &&
           same_native_attachment(left.stencil, right.stencil);
}

RendererStatus open_pass(Batch& batch, PacketLease& packet, id<MTLDevice> device) {
    const auto* clear = std::get_if<PreparedClear>(&packet.prepared.operation);
    // An attachment-less draw has an ordinary key (including its extent and
    // sample count). Only an absent encoder/key means that no pass is open.
    const bool compatible = batch.render && (!clear || !clear->clear.fold_into_load) &&
                            batch.pass == packet.prepared.attachments.key &&
                            same_native_targets(batch.pass_targets,
                                                packet.native.targets) &&
                            !pass_hazard(batch, packet.prepared);
    if (compatible) return renderer_success();
    static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
    if (trace) {
        // Why a pass opens: none open, folded clear, key, targets, hazard.
        static thread_local std::array<std::uint64_t, 5> reasons{};
        const std::size_t reason = !batch.render ? 0 : (clear && clear->clear.fold_into_load) ? 1
            : !(batch.pass == packet.prepared.attachments.key) ? 2
            : !same_native_targets(batch.pass_targets, packet.native.targets) ? 3 : 4;
        ++reasons[reason];
        if ((reasons[0] + reasons[1] + reasons[2] + reasons[3] + reasons[4]) % 5000 == 0)
            armigo::diag::opt_in( "pass_close commands=%llu subresource=%llu feedback=%llu dispatch=%llu\n",
                (unsigned long long)close_sites[0], (unsigned long long)close_sites[1],
                (unsigned long long)close_sites[2], (unsigned long long)close_sites[3]),
            armigo::diag::opt_in( "pass_open none=%llu clear=%llu key=%llu targets=%llu hazard=%llu\n",
                (unsigned long long)reasons[0], (unsigned long long)reasons[1], (unsigned long long)reasons[2],
                (unsigned long long)reasons[3], (unsigned long long)reasons[4]);
    }
    close_render(batch);
    auto status = ensure_commands(batch, device);
    if (!succeeded(status)) return status;
    auto descriptor = pass_descriptor(packet);
    if (const auto* error = std::get_if<RendererError>(&descriptor)) return *error;
    batch.encoded = {};
    batch.render = [batch.commands renderCommandEncoderWithDescriptor:
        std::get<MTL4RenderPassDescriptor*>(descriptor)];
    if (!batch.render)
        return failure("Metal render encoder creation failed", CommandStatus::MemoryFault);
    // The acquire barrier also orders the first encoder against an earlier
    // command buffer committed on this queue. It is harmless for the first
    // submission and avoids treating queue order as a visibility guarantee.
    [batch.render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                      visibilityOptions:MTL4VisibilityOptionDevice];
    batch.pass = packet.prepared.attachments.key;
    batch.pass_targets = packet.native.targets;
    batch.pass_accesses.clear();
    ++batch.render_passes;
    return renderer_success();
}

// Copies rendered sub-resources into the storage this packet samples, after
// their producers and before its work. Cache-visible flags are published by
// the coordinator when handing the batch to the ordered commit worker.
RendererStatus encode_subresource_copies(Batch& batch, PacketLease& packet, id<MTLDevice> device) {
    if (std::ranges::all_of(packet.native.subresource_copies, [&](const auto& copy) {
            return batch.encoded_copies.contains(copy.get());
        }))
        return renderer_success();
    if (batch.render) ++close_sites[1];
    close_render(batch);
    auto status = ensure_commands(batch, device);
    if (!succeeded(status)) return status;
    auto encoder = [batch.commands computeCommandEncoder];
    if (!encoder) return failure("sub-resource copy encoder allocation failed", CommandStatus::MemoryFault);
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                   visibilityOptions:MTL4VisibilityOptionDevice];
    for (const auto& copy : packet.native.subresource_copies) {
        if (!batch.encoded_copies.insert(copy.get()).second) continue;
        if (copy->staging) {
            const auto image = copy->bytes_per_row * copy->height;
            [encoder copyFromTexture:copy->source sourceSlice:0 sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(copy->width, copy->height, 1)
                            toBuffer:copy->staging destinationOffset:0
                destinationBytesPerRow:copy->bytes_per_row destinationBytesPerImage:image];
            [encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit
                             visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder copyFromBuffer:copy->staging sourceOffset:0 sourceBytesPerRow:copy->bytes_per_row
                  sourceBytesPerImage:image sourceSize:MTLSizeMake(copy->width, copy->height, 1)
                            toTexture:copy->destination destinationSlice:copy->layer
                     destinationLevel:copy->level destinationOrigin:MTLOriginMake(0, 0, 0)];
            continue;
        }
        [encoder copyFromTexture:copy->source sourceSlice:0 sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(copy->width, copy->height, 1)
                       toTexture:copy->destination destinationSlice:copy->layer
                destinationLevel:copy->level destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [encoder barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll
                visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder endEncoding];
    return renderer_success();
}

RendererStatus encode_draw_call(Batch& batch, PacketLease& packet, id<MTLDevice> device,
                                std::span<const DrawSegment> segments,
                                const ResidentBuffer* index_override,
                                const RenderState* state_override,
                                id<MTLRenderPipelineState> pipeline_override);

RendererStatus encode_draw(Batch& batch, PacketLease& packet,
                           id<MTLDevice> device) {
    // Only accepted packets contribute this allocation to shared residency.
    // Rejection during preparation releases the packet-owned support buffer.
    if (packet.native.draw_support)
        add_resident(batch, packet.native.draw_support);
    if (auto copied = encode_subresource_copies(batch, packet, device); !succeeded(copied)) return copied;
    // End the preceding pass before taking the snapshot. Device-visible
    // release/acquire barriers order earlier attachment writes, the blit reads
    // and writes, and this draw's texture reads across encoder boundaries.
    if (!packet.native.feedback_copies.empty()) {
        if (batch.render) ++close_sites[2];
        close_render(batch);
        auto status = ensure_commands(batch, device);
        if (!succeeded(status)) return status;
        adopt_residency(batch, packet.native.feedback_residency);
        auto transfer = [batch.commands computeCommandEncoder];
        if (!transfer) return failure("attachment feedback encoder allocation failed", CommandStatus::MemoryFault);
        [transfer barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                        visibilityOptions:MTL4VisibilityOptionDevice];
        for (const auto& copy : packet.native.feedback_copies)
            [transfer copyFromTexture:copy.source sourceSlice:copy.first_layer sourceLevel:copy.first_level
                            toTexture:copy.destination destinationSlice:0 destinationLevel:0
                           sliceCount:copy.layers levelCount:copy.levels];
        [transfer barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [transfer endEncoding];
    }
    return encode_draw_call(batch, packet, device,
                            std::get<PreparedDraw>(packet.prepared.operation).draw.segments);
}

// The draw itself: pass, changed encoder state, argument tables and the
// draw calls for `segments` (the packet's own, or a repeat's).
RendererStatus encode_draw_call(Batch& batch, PacketLease& packet, id<MTLDevice> device,
                                std::span<const DrawSegment> segments,
                                const ResidentBuffer* index_override,
                                const RenderState* state_override,
                                id<MTLRenderPipelineState> pipeline_override) {
    auto status = open_pass(batch, packet, device);
    if (!succeeded(status)) return status;
    auto& prepared = std::get<PreparedDraw>(packet.prepared.operation);
    auto& encoded = batch.encoded;
    const id<MTLRenderPipelineState> pipeline = pipeline_override ? pipeline_override : packet.native.pipeline;
    if (encoded.pipeline != pipeline) {
        [batch.render setRenderPipelineState:pipeline];
        encoded.pipeline = pipeline;
    }
    if (encoded.depth_stencil != packet.native.depth_stencil) {
        [batch.render setDepthStencilState:packet.native.depth_stencil];
        encoded.depth_stencil = packet.native.depth_stencil;
    }
    // Argument tables are retained and updated in place between draws, so
    // they rebind every draw; only the table object identity is cached here.
    if (packet.native.vertex_table)
        [batch.render setArgumentTable:packet.native.vertex_table
                              atStages:MTLRenderStageVertex];
    if (packet.native.fragment_table)
        [batch.render setArgumentTable:packet.native.fragment_table
                              atStages:MTLRenderStageFragment];

    const auto& state = state_override ? *state_override : prepared.state;
    if (state.rasterizer.cull == CullMode::FrontAndBack ||
        state.rasterizer.front_fill != state.rasterizer.back_fill ||
        state.rasterizer.front_fill == FillMode::Point)
        return failure("draw contains rasterizer state unsupported by Metal encoding");
    MTLCullMode cull = MTLCullModeNone;
    if (state.rasterizer.cull == CullMode::Front) cull = MTLCullModeFront;
    if (state.rasterizer.cull == CullMode::Back) cull = MTLCullModeBack;
    if (encoded.cull != int(cull)) {
        [batch.render setCullMode:cull];
        encoded.cull = int(cull);
    }
    const auto winding = state.rasterizer.front_face == FrontFace::Clockwise
        ? MTLWindingClockwise : MTLWindingCounterClockwise;
    if (encoded.winding != int(winding)) {
        [batch.render setFrontFacingWinding:winding];
        encoded.winding = int(winding);
    }
    const auto fill = state.rasterizer.front_fill == FillMode::Line
        ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
    if (encoded.fill != int(fill)) {
        [batch.render setTriangleFillMode:fill];
        encoded.fill = int(fill);
    }
    const auto clip = state.rasterizer.depth_clip == DepthClipMode::Clamp
        ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
    if (encoded.clip != int(clip)) {
        [batch.render setDepthClipMode:clip];
        encoded.clip = int(clip);
    }
    const std::array<float, 4> blend{state.blend_constant[0], state.blend_constant[1],
                                     state.blend_constant[2], state.blend_constant[3]};
    if (encoded.blend != blend) {
        [batch.render setBlendColorRed:blend[0] green:blend[1] blue:blend[2] alpha:blend[3]];
        encoded.blend = blend;
    }
    if (encoded.front_reference != state.front_stencil_reference ||
        encoded.back_reference != state.back_stencil_reference) {
        [batch.render setStencilFrontReferenceValue:state.front_stencil_reference
                                 backReferenceValue:state.back_stencil_reference];
        encoded.front_reference = state.front_stencil_reference;
        encoded.back_reference = state.back_stencil_reference;
    }
    thread_local std::vector<MTLViewport> viewports;
    viewports.clear();
    for (const auto& viewport : state.viewports)
        viewports.push_back({viewport.x, viewport.y, viewport.width, viewport.height,
                             viewport.near_depth, viewport.far_depth});
    const auto same_viewports = viewports.size() == encoded.viewports.size() &&
        std::memcmp(viewports.data(), encoded.viewports.data(), viewports.size() * sizeof(MTLViewport)) == 0;
    if (!viewports.empty() && !same_viewports) {
        [batch.render setViewports:viewports.data() count:viewports.size()];
        encoded.viewports = viewports;
    }
    thread_local std::vector<MTLScissorRect> scissors;
    scissors.clear();
    for (const auto& scissor : state.scissors)
        scissors.push_back({NSUInteger(scissor.x), NSUInteger(scissor.y),
                            NSUInteger(scissor.width), NSUInteger(scissor.height)});
    const auto same_scissors = scissors.size() == encoded.scissors.size() &&
        std::memcmp(scissors.data(), encoded.scissors.data(), scissors.size() * sizeof(MTLScissorRect)) == 0;
    if (!scissors.empty() && !same_scissors) {
        [batch.render setScissorRects:scissors.data() count:scissors.size()];
        encoded.scissors = scissors;
    }
    // Polygon-offset enables select polygon rasterization modes, not native
    // point/line topologies. Reset every draw, including reused encoders.
    const bool biased = packet.prepared.targets.depth &&
        prepared.draw.primitive_class == PrimitiveClass::Triangle &&
        state.rasterizer.front_fill == FillMode::Fill && state.depth_bias.fill;
    const bool native_slope = biased &&
        (!prepared.shader_depth_bias || state.depth_bias.clamp == 0.0f);
    [batch.render setDepthBias:0.0f
                    slopeScale:native_slope ? state.depth_bias.factor : 0.0f
                         clamp:native_slope ? state.depth_bias.clamp : 0.0f];

    const auto primitive = primitive_type(prepared.draw.topology);
    if (prepared.draw.topology != PrimitiveTopology::Points &&
        prepared.draw.topology != PrimitiveTopology::Lines &&
        prepared.draw.topology != PrimitiveTopology::LineStrip &&
        prepared.draw.topology != PrimitiveTopology::Triangles &&
        prepared.draw.topology != PrimitiveTopology::TriangleStrip)
        return failure("draw topology was not normalized before Metal submission");
    if (prepared.draw.indices) {
        const auto* index_binding = index_override ? index_override
            : prepared.bindings.index_buffer ? &*prepared.bindings.index_buffer : nullptr;
        if (!index_binding)
            return failure("indexed draw has no resident index buffer");
        const auto type = index_type(prepared.draw.indices->encoded_format);
        if (!type) return failure("indexed draw uses an unsupported index type");
        id<MTLBuffer> index = resident_native_buffer(index_binding->storage);
        const auto element_size = *type == MTLIndexTypeUInt16 ? 2u : 4u;
        for (const auto& segment : segments) {
            const std::uint64_t byte_offset =
                std::uint64_t(segment.first) * element_size;
            if (byte_offset > index_binding->size)
                return failure("indexed draw segment exceeds its resident buffer");
            [batch.render drawIndexedPrimitives:primitive
                                     indexCount:segment.count
                                      indexType:*type
                                    indexBuffer:index.gpuAddress + index_binding->offset + byte_offset
                              indexBufferLength:index_binding->size - byte_offset
                                  instanceCount:segment.instance_count
                                     baseVertex:segment.base_vertex
                                   baseInstance:segment.base_instance];
        }
    } else {
        for (const auto& segment : segments)
            [batch.render drawPrimitives:primitive vertexStart:segment.first
                             vertexCount:segment.count
                           instanceCount:segment.instance_count
                            baseInstance:segment.base_instance];
    }
    for (const auto& access : packet.prepared.accesses)
        batch.pass_accesses.record(access.range.bytes, access.access,
            contains_attachment(packet.prepared, access.range.bytes));
    ++batch.draws;
    return renderer_success();
}

RendererStatus encode_clear_packet(Batch& batch, PacketLease& packet,
                                   id<MTLDevice> device) {
    const auto& clear = std::get<PreparedClear>(packet.prepared.operation).clear;
    if (clear.no_op) return renderer_success();
    if (clear.fold_into_load) {
        auto status = open_pass(batch, packet, device);
        if (!succeeded(status)) return status;
        for (const auto& access : packet.prepared.accesses)
            batch.pass_accesses.record(access.range.bytes, access.access, true);
        return renderer_success();
    }
    // Because clears execute inside the command stream, the clear pass joins
    // the batch's open command buffer; only the render pass ends.
    if (batch.render) ++close_sites[0];
    close_render(batch);
    if (auto status = ensure_commands(batch, device); !succeeded(status)) return status;
    auto bindings = packet.native.clear;
    bindings.into = batch.commands;
    bindings.into_residency = batch.residency;
    auto encoded = encode_clear(device, clear, bindings);
    if (const auto* error = std::get_if<RendererError>(&encoded)) return *error;
    auto clear_commands = std::get<EncodedClear>(encoded);
    if (clear_commands.retained)
        [batch.lease->objects addObjectsFromArray:clear_commands.retained];
    ++batch.render_passes;
    return renderer_success();
}

// Compute dispatch ends the open render pass and runs in its own
// compute encoder on the same command buffer. Device-visible barriers order
// it after earlier passes and before later ones, like attachment feedback.
RendererStatus encode_dispatch(Batch& batch, PacketLease& packet, id<MTLDevice> device) {
    if (auto copied = encode_subresource_copies(batch, packet, device); !succeeded(copied)) return copied;
    if (batch.render) ++close_sites[3];
    close_render(batch);
    auto status = ensure_commands(batch, device);
    if (!succeeded(status)) return status;
    const auto& dispatch = std::get<PreparedDispatch>(packet.prepared.operation);
    auto encoder = [batch.commands computeCommandEncoder];
    if (!encoder) return failure("compute encoder allocation failed", CommandStatus::MemoryFault);
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setComputePipelineState:packet.native.compute_pipeline];
    if (packet.native.compute_table) [encoder setArgumentTable:packet.native.compute_table];
    [encoder dispatchThreadgroups:MTLSizeMake(dispatch.grid[0], dispatch.grid[1], dispatch.grid[2])
            threadsPerThreadgroup:MTLSizeMake(dispatch.block[0], dispatch.block[1], dispatch.block[2])];
    [encoder barrierAfterStages:MTLStageDispatch beforeQueueStages:MTLStageAll
              visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder endEncoding];
    return renderer_success();
}

RendererStatus encode_packet(Batch& batch, PacketLease& packet,
                             id<MTLDevice> device) {
    if (std::holds_alternative<PreparedDraw>(packet.prepared.operation))
        return encode_draw(batch, packet, device);
    if (std::holds_alternative<PreparedDispatch>(packet.prepared.operation))
        return encode_dispatch(batch, packet, device);
    return encode_clear_packet(batch, packet, device);
}

std::vector<CompletionToken> dependencies(const PacketLease& packet) {
    std::vector<CompletionToken> result = packet.prepared.request.dependencies;
    if (const auto* bindings = packet_bindings(packet.prepared))
        result.insert(result.end(), bindings->dependencies.begin(),
                      bindings->dependencies.end());
    const auto append_texture = [&](const auto& texture) {
        if (texture)
            result.insert(result.end(), texture->dependencies.begin(),
                          texture->dependencies.end());
    };
    for (const auto& color : packet.prepared.attachments.colors) append_texture(color);
    append_texture(packet.prepared.attachments.depth);
    append_texture(packet.prepared.attachments.stencil);
    return result;
}

class Queue final : public SubmissionQueue {
    struct EncodeJob {
        std::shared_ptr<Batch> batch;
        std::shared_ptr<PacketLease> packet;
        std::shared_ptr<RepeatJob> repeat;
        std::uint64_t hold_generation = 0;
    };
    struct DeferredCommit {
        std::shared_ptr<Batch> batch;
        std::uint64_t generation = 0;
    };
public:
    Queue(std::shared_ptr<SharedState> shared, std::uint32_t commands,
          std::uint32_t inflight)
        : shared_(std::move(shared)), max_commands_(commands),
          max_inflight_(inflight) {
        inflight_.reserve(inflight);
        deferred_commits_.reserve(inflight);
        retired_batches_.resize(inflight);
        encode_jobs_.resize((std::size_t(commands) + 1) * inflight);
        encoder_ = std::thread([this] { encode_loop(); });
    }

    ~Queue() override {
        if (commit_held())
            static_cast<void>(abort_commit_hold({CommandStatus::Stopped,
                "submission queue destroyed with a held commit"}));
        join_encodes();
        {
            std::lock_guard lock(encode_mutex_);
            stopping_ = true;
        }
        encode_ready_.notify_one();
        encoder_.join();
        @autoreleasepool {
            if (current_) fail_current({CommandStatus::Stopped,
                                       "submission queue destroyed before flush"});
        }
    }

    ResidencyDomain domain() const noexcept override {
        return {shared_->device->identity().device, shared_->memory.get()};
    }

    RendererResult<SubmissionReceipt> append(PreparedRender prepared) override {
        @autoreleasepool {
            if (auto error = encoding_failure()) {
                fail_current(error->command(), false);
                return *error;
            }
            if (inflight_.size() >= max_inflight_)
                return failure("submission in-flight limit requires GPU retirement",
                               CommandStatus::Stopped);
            if (shared_->ordered_failure) return *shared_->ordered_failure;
            if (current_ && current_->packets.size() >= max_commands_)
                return failure("submission command limit requires a flush",
                               CommandStatus::Stopped);
            auto validated = validate_packet(prepared, *shared_);
            if (const auto* error = std::get_if<RendererError>(&validated)) return *error;
            if (queue_identity_ && prepared.request.id.queue != queue_identity_)
                return failure("submission queue identity changed");
            if (prepared.request.id.sequence <= last_sequence_)
                return failure("submission sequence is not strictly increasing");
            for (const auto& dependency : prepared.request.dependencies) {
                const auto snapshot = dependency.snapshot();
                if (!dependency.valid() || snapshot.error) {
                    if (snapshot.error) return *snapshot.error;
                    return failure("submission dependency is invalid");
                }
                if (snapshot.id == prepared.request.id)
                    return failure("submission cannot depend on itself");
                if (snapshot.id.queue == prepared.request.id.queue &&
                    snapshot.id.sequence >= prepared.request.id.sequence)
                    return failure("submission dependency points forward in queue order");
            }
            try {
                if (!current_) current_ = std::make_shared<Batch>(pool_);
                prepared.request.completion = std::make_shared<OrderedActions>(
                    shared_, std::move(prepared.request.completion));
                auto lease = std::make_shared<PacketLease>(std::move(prepared),
                                                           current_->lease);
                auto validated_native = validate_native(*lease, shared_->device->device());
                if (const auto* error = std::get_if<RendererError>(&validated_native)) return *error;
                SubmissionCompletion completion(lease->prepared.request.id,
                    std::vector<RetainedResource>{lease},
                    lease->prepared.request.completion);
                const auto token = completion.token();
                Packet packet(lease, completion);
                packet.prefix_safe_completion = lease->prepared.prefix_safe_completion;
                packet.dependencies = dependencies(*lease);
                if (current_->packets.size() == current_->packets.capacity())
                    current_->packets.reserve(std::max(current_->packets.size() + 1,
                                                      2 * current_->packets.capacity()));

                // Record only after all packet tracking capacity is ready.
                // Acceptance owns its dependency vector separately, so retaining
                // it after journaling cannot allocate or lose accepted work.
                const auto* clear = std::get_if<PreparedClear>(&lease->prepared.operation);
                if ((!clear || !clear->clear.no_op) && !lease->prepared.accesses.empty()) {
                    auto accepted = shared_->memory->accept_batch_accesses(
                        lease->prepared.accesses, token, lease->prepared.access_storage, current_->lease);
                    if (const auto* error = std::get_if<RendererError>(&accepted)) {
                        reject_unaccepted(completion, error->command());
                        return *error;
                    }
                    lease->acceptance = std::get<GpuAccessAcceptance>(std::move(accepted));
                }
                current_->packets.push_back(std::move(packet));
                enqueue_encode(lease);
                if (std::holds_alternative<PreparedDraw>(lease->prepared.operation)) last_draw_ = lease;
                else last_draw_.reset();
                queue_identity_ = lease->prepared.request.id.queue;
                last_sequence_ = lease->prepared.request.id.sequence;
                shared_->accepted.fetch_add(1, std::memory_order_relaxed);
                return SubmissionReceipt{token};
            } catch (const std::bad_alloc&) {
                if (current_ && !current_->packets.empty())
                    fail_current({CommandStatus::MemoryFault,
                                  "submission allocation failed"});
                return failure("submission allocation failed", CommandStatus::MemoryFault);
            }
        }
    }

    // A draw whose state matches the last appended draw
    // costs only its draw parameters. It joins that draw's packet (same
    // completion, accesses and registry intents) and the worker encodes the
    // changed buffer slots plus the draw calls into the open pass.
    bool append_repeat(RepeatDraw repeat) override {
        if (!last_draw_ || shared_->ordered_failure) return false;
        {
            // A full ring waits in enqueue_job like packet admission: falling
            // back to a full preparation would wait for a slot as well.
            std::lock_guard lock(encode_mutex_);
            if (encode_failure_) return false;
        }
        try {
            for (const auto& binding : repeat.changed)
                last_draw_->repeat_resources.push_back(binding.buffer.storage);
            for (const auto& binding : repeat.vertex_buffers)
                last_draw_->repeat_resources.push_back(binding.buffer.storage);
            if (repeat.index_buffer) last_draw_->repeat_resources.push_back(repeat.index_buffer->storage);
            id<MTLRenderPipelineState> pipeline = nil;
            if (repeat.pipeline) {
                const auto native = std::dynamic_pointer_cast<const MetalRenderPipelineResource>(
                    repeat.pipeline->pipeline);
                if (!native || !native->pipeline()) return false;
                pipeline = native->pipeline();
                last_draw_->repeat_resources.push_back(repeat.pipeline->pipeline);
            }
            enqueue_encode(last_draw_, std::make_shared<RepeatJob>(
                RepeatJob{std::move(repeat.segments), std::move(repeat.changed),
                          std::move(repeat.vertex_buffers), std::move(repeat.index_buffer),
                          std::move(repeat.state), pipeline, repeat.pipeline}));
            return true;
        } catch (const std::bad_alloc&) {
            return false;
        }
    }

    // Progress value of the last complete poll (caller thread).
    std::uint64_t polled_progress_ = ~std::uint64_t{0};

    RendererStatus flush(SubmissionBoundary) override {
        polled_progress_ = ~std::uint64_t{0};
        @autoreleasepool {
            last_draw_.reset();
            if (auto error = encoding_failure()) {
                fail_current(error->command(), false);
                return *error;
            }
            if (!current_ || current_->packets.empty()) return renderer_success();
            if (inflight_.size() >= max_inflight_)
                return failure("submission in-flight limit requires GPU retirement",
                               CommandStatus::Stopped);
            try {
                bool terminal_dependency = false;
                auto ready = dependencies_ready(*current_, terminal_dependency);
                if (const auto* error = std::get_if<RendererError>(&ready)) {
                    if (terminal_dependency) return fail_current_status(error->command());
                    return ready;
                }
                // Allocate all tracking before publishing registry access or
                // Submitted tokens. The worker never touches the registry or
                // caller-owned packet/retirement lists.
                current_->completions.reserve(current_->packets.size());
                for (const auto& packet : current_->packets) {
                    current_->completions.push_back(packet.completion);
                    submitted_ids_.insert(packet.lease->prepared.request.id);
                }
                current_->without_gpu_work = std::ranges::all_of(current_->packets,
                    [](const auto& packet) {
                        const auto* clear = std::get_if<PreparedClear>(&packet.lease->prepared.operation);
                        return clear && clear->clear.no_op;
                    });
                auto published = shared_->memory->flush_access_journal();
                if (!succeeded(published))
                    return fail_current_status(std::get<RendererError>(published).command());
                for (auto& packet : current_->packets) {
                    auto status = packet.completion.submitted();
                    if (!succeeded(status)) return status;
                    // Later batches may omit these copies because their commit
                    // jobs cannot pass this batch. A failure poisons later work.
                    // Keep the cache-visible flag exclusively on the caller.
                    for (const auto& copy : packet.lease->native.subresource_copies)
                        copy->encoded = true;
                }
                SubmittedBatch submitted;
                submitted.batch = current_;
                submitted.recyclable = current_->lease;
                submitted.without_gpu_work = current_->without_gpu_work;
                submitted.packets = std::move(current_->packets);
                inflight_.push_back(std::move(submitted));
                enqueue_job(EncodeJob{current_, {}, {}, current_hold_generation()});
                current_.reset();
                return renderer_success();
            } catch (const std::bad_alloc&) {
                return fail_current_status({CommandStatus::MemoryFault,
                    "submission commit preparation allocation failed"});
            }
        }
    }

    RendererStatus poll() override {
        // Nothing completes, fails or commits without a progress step, and a
        // flush (which may unblock retirement) forces the next walk.
        const auto progress = shared_->progress.load(std::memory_order_acquire);
        if (progress == polled_progress_) return renderer_success();
        // Both an open batch and a pending commit may still be encoding.
        // Retirement checks commit_done without blocking the submitting thread.
        const auto encode_error = encoding_failure();
        if (encode_error) fail_current(encode_error->command(), false);
        auto status = retire();
        if (!succeeded(status)) return status;
        if (encode_error) return *encode_error;
        // Failures stay reported on every poll; only a clean walk is skipped.
        polled_progress_ = progress;
        return renderer_success();
    }

private:
    RendererStatus retire() {
        // A boundary completion action may inspect the complete registry. Do
        // not finish older feedback while this queue still has Accepted work.
        while (!inflight_.empty()) {
            auto& submitted = inflight_.front();
            if (submitted.batch && !submitted.batch->commit_done.load(std::memory_order_acquire))
                return renderer_success();
            while (submitted.next < submitted.packets.size()) {
                auto& packet = submitted.packets[submitted.next];
                if (current_ && !current_->packets.empty() &&
                    (!packet.prefix_safe_completion ||
                     packet.completion.token().snapshot().error)) return renderer_success();
                if (submitted.without_gpu_work &&
                    packet.completion.token().snapshot().phase == SubmissionPhase::Submitted) {
                    // Flush checked external dependencies; reaching this point
                    // also proves earlier packets in this queue have retired.
                    // Preserve a predecessor's failure before exposing feedback.
                    auto status = packet.completion.feedback(shared_->ordered_failure
                        ? shared_->ordered_failure->command() : CommandResult{});
                    if (!succeeded(status)) return status;
                }
                const auto snapshot = packet.completion.token().snapshot();
                if (snapshot.phase == SubmissionPhase::Submitted ||
                    snapshot.phase == SubmissionPhase::Accepted ||
                    snapshot.phase == SubmissionPhase::Publishing)
                    return renderer_success();
                if (snapshot.phase == SubmissionPhase::Complete) {
                    packet.lease->prepared.request.completion.reset();
                    packet.lease->acceptance.lease.reset();
                    submitted_ids_.erase(packet.lease->prepared.request.id);
                    ++submitted.next;
                    continue;
                }
                RendererStatus outcome = snapshot.error ? RendererStatus(*snapshot.error)
                                                        : renderer_success();
                if (snapshot.error && !shared_->ordered_failure)
                    shared_->ordered_failure = *snapshot.error;
                if (!packet.finish_attempted) {
                    outcome = packet.completion.finish();
                    packet.finish_attempted = true;
                }
                const auto finished = packet.completion.token().snapshot();
                if (finished.phase == SubmissionPhase::Failed) {
                    // finish() can create a new failure when coherence or
                    // publication preparation throws or returns an error. It
                    // must poison ordering before any later packet is polled.
                    if (finished.error && !shared_->ordered_failure)
                        shared_->ordered_failure = *finished.error;
                    auto released = packet.completion.release_failed();
                    if (!succeeded(released)) return released;
                    packet.lease->prepared.request.completion.reset();
                    packet.lease->acceptance.lease.reset();
                    submitted_ids_.erase(packet.lease->prepared.request.id);
                    ++submitted.next;
                    auto error = finished.error ? RendererStatus(*finished.error) : outcome;
                    return error;
                }
                if (!succeeded(outcome)) return outcome;
                packet.lease->prepared.request.completion.reset();
                packet.lease->acceptance.lease.reset();
                submitted_ids_.erase(packet.lease->prepared.request.id);
                ++submitted.next;
            }
            if (submitted.recyclable && !shared_->ordered_failure)
                recycle_native(pool_, *submitted.recyclable);
            // Completion actions and registry authority transitions stay here.
            // Only fully retired ownership goes to the encoder for destruction.
            // A full cleanup ring keeps the in-flight slot occupied; poll never
            // waits for a busy encoder and admission keeps its existing bound.
            if (!defer_destruction(submitted)) return renderer_success();
            inflight_.erase(inflight_.begin());
        }
        return renderer_success();
    }

public:
    RendererStatus set_commit_hold(bool held) override {
        polled_progress_ = ~std::uint64_t{0};
        bool wake = false;
        {
            std::lock_guard lock(encode_mutex_);
            if (held && encode_failure_) return *encode_failure_;
            if (held && !commit_held_) {
                if (hold_generation_ == std::numeric_limits<std::uint64_t>::max())
                    return failure("submission commit hold generation is exhausted",
                                   CommandStatus::BudgetExceeded);
                ++hold_generation_;
            } else if (!held && commit_held_) {
                released_generation_ = hold_generation_;
            }
            commit_held_ = held;
            wake = !held || (encode_sleeping_ && encode_queued_);
        }
        if (wake) encode_ready_.notify_one();
        return renderer_success();
    }

    RendererStatus abort_commit_hold(CommandResult error) override {
        polled_progress_ = ~std::uint64_t{0};
        {
            std::lock_guard lock(encode_mutex_);
            if (!encode_failure_) encode_failure_ = RendererError(error);
            released_generation_ = hold_generation_;
            commit_held_ = false;
            aborted_hold_ = true;
        }
        encode_ready_.notify_one();
        encode_space_.notify_all();
        return renderer_success();
    }

    bool commit_held() const noexcept override {
        std::lock_guard lock(encode_mutex_);
        return commit_held_;
    }

    bool can_accept_while_held() const noexcept override {
        std::lock_guard lock(encode_mutex_);
        if (!commit_held_) return true;
        return !encode_failure_ && inflight_.size() < max_inflight_ &&
            (!current_ || current_->packets.size() < max_commands_);
    }

    RendererStatus retire_below(std::size_t limit, std::uint64_t steady_deadline_ns) override {
        if (commit_held() && (current_ || inflight_.size() >= limit))
            return failure(commit_hold_detail, CommandStatus::Stopped);
        auto status = flush(SubmissionBoundary::ResourceBudget);
        if (!succeeded(status)) return status;
        for (;;) {
            std::unique_lock lock(shared_->feedback_mutex);
            const auto observed = shared_->feedback_generation;
            lock.unlock();
            status = retire();
            if (!succeeded(status)) return status;
            if (inflight_.size() < limit) return renderer_success();
            const auto deadline = std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(steady_deadline_ns));
            lock.lock();
            if (shared_->feedback_generation != observed) continue;
            if (std::chrono::steady_clock::now() >= deadline ||
                !shared_->feedback_ready.wait_until(lock, deadline, [&] {
                    return shared_->feedback_generation != observed;
                }))
                return failure("submission retirement deadline expired",
                               CommandStatus::Stopped);
        }
    }

    RendererStatus drain(std::uint64_t steady_deadline_ns) override {
        polled_progress_ = ~std::uint64_t{0};
        if (commit_held() && (current_ || !inflight_.empty()))
            return failure(commit_hold_detail, CommandStatus::Stopped);
        auto status = flush(SubmissionBoundary::Shutdown);
        // Admission remains poisoned, but shutdown must still retire earlier
        // committed work and hand failed authority back on the coordinator.
        if (!succeeded(status) && !encoding_failure()) return status;
        std::optional<RendererError> first_error;
        for (;;) {
            std::unique_lock lock(shared_->feedback_mutex);
            const auto observed = shared_->feedback_generation;
            lock.unlock();
            const auto count_before = inflight_.size();
            const auto next_before = count_before ? inflight_.front().next : 0;
            status = retire();
            if (const auto* error = std::get_if<RendererError>(&status)) {
                if (!aborted_hold_) return *error;
                if (!first_error) first_error = *error;
                if (inflight_.size() == count_before &&
                    (!count_before || inflight_.front().next == next_before))
                    return *error;
                continue;
            }
            if (inflight_.empty()) {
                // A successful drain includes deferred ownership release. Honor
                // the same deadline while the encoder destroys retired batches.
                std::unique_lock cleanup_lock(encode_mutex_);
                const auto deadline = std::chrono::steady_clock::time_point(
                    std::chrono::nanoseconds(steady_deadline_ns));
                if (!encode_idle_.wait_until(cleanup_lock, deadline,
                        [this] { return retirement_pending_ == 0; }))
                    return failure("submission cleanup deadline expired", CommandStatus::Stopped);
                return first_error ? RendererStatus(*first_error) : renderer_success();
            }
            const auto deadline = std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(steady_deadline_ns));
            lock.lock();
            if (shared_->feedback_generation != observed) continue;
            if (std::chrono::steady_clock::now() >= deadline ||
                !shared_->feedback_ready.wait_until(lock, deadline, [&] {
                    return shared_->feedback_generation != observed;
                }))
                return failure("submission drain deadline expired",
                               CommandStatus::Stopped);
        }
    }

private:
    std::uint64_t current_hold_generation() const noexcept {
        std::lock_guard lock(encode_mutex_);
        return commit_held_ ? hold_generation_ : 0;
    }

    // Jobs retain their own batch; a null packet is its ordered commit marker.
    // Reserve a ring slot for that marker so flush never waits for encoding.
    // Repeats can fill the ring; only packet admission applies backpressure.
    void enqueue_job(EncodeJob job) {
        std::unique_lock lock(encode_mutex_);
        if (job.packet) {
            if (encode_queued_ >= encode_jobs_.size() - 1) encode_ready_.notify_one();
            encode_space_.wait(lock, [this] { return encode_queued_ < encode_jobs_.size() - 1; });
        }
        const bool commit = !job.packet;
        encode_jobs_[encode_tail_] = std::move(job);
        encode_tail_ = (encode_tail_ + 1) % encode_jobs_.size();
        ++encode_queued_;
        ++encode_pending_;
        const bool wake = encode_sleeping_ &&
            (commit || commit_held_ ||
             encode_queued_ >= std::max<std::size_t>(1, max_commands_ / 4));
        lock.unlock();
        if (wake) encode_ready_.notify_one();
    }

    bool defer_destruction(SubmittedBatch& submitted) {
        std::unique_lock lock(encode_mutex_);
        if (retired_queued_ == retired_batches_.size()) return false;
        retired_batches_[retired_tail_].emplace(std::move(submitted));
        retired_tail_ = (retired_tail_ + 1) % retired_batches_.size();
        ++retired_queued_;
        ++retirement_pending_;
        ++encode_pending_;
        const bool wake = encode_sleeping_;
        lock.unlock();
        if (wake) encode_ready_.notify_one();
        return true;
    }

    void enqueue_encode(const std::shared_ptr<PacketLease>& packet,
                        std::shared_ptr<RepeatJob> repeat = {}) {
        enqueue_job(EncodeJob{current_, packet, std::move(repeat)});
    }

    void notify_feedback() {
        {
            std::lock_guard lock(shared_->feedback_mutex);
            ++shared_->feedback_generation;
        }
        shared_->progress.fetch_add(1, std::memory_order_release);
        shared_->feedback_ready.notify_all();
    }

    RendererStatus commit_batch(const std::shared_ptr<Batch>& batch) {
        auto closed = close_commands(*batch);
        if (!succeeded(closed)) return closed;
        if (batch->buffers.empty()) {
            if (!batch->without_gpu_work)
                return failure("submission encoded no Metal commands");
            // Retirement synthesizes feedback only after predecessors retire.
            return renderer_success();
        }
        if (shared_->test_control) {
            auto status = shared_->test_control->before_commit(batch->completions.front().token().id());
            if (!succeeded(status)) return status;
        }
        [batch->residency commit];
        auto* options = [MTL4CommitOptions new];
        if (!options)
            return failure("Metal commit-options allocation failed", CommandStatus::MemoryFault);
        auto state = shared_;
        auto retained = batch->lease;
        auto completions = batch->completions;
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            note_gpu_time(feedback.GPUStartTime, feedback.GPUEndTime);
            std::string detail;
            CommandResult result;
            if (feedback.error) {
                const char* message = feedback.error.localizedDescription.UTF8String;
                detail = message ? message : "Metal queue reported an unknown GPU failure";
                result = {CommandStatus::MemoryFault, detail};
            }
            if (state->test_control) {
                try {
                    result = state->test_control->after_gpu_quiescence(result);
                } catch (...) {
                    result = {CommandStatus::Stopped,
                              "submission test feedback control threw an exception"};
                }
            }
            for (const auto& completion : completions) {
                auto feedback_completion = completion;
                static_cast<void>(feedback_completion.feedback(result));
            }
            static_cast<void>(retained->objects.count);
            {
                std::lock_guard lock(state->feedback_mutex);
                ++state->feedback_generation;
                state->progress.fetch_add(1, std::memory_order_release);
            }
            state->feedback_ready.notify_all();
        }];
        [shared_->device->queue() commit:batch->buffers.data() count:batch->buffers.size()
                                  options:options];
        shared_->command_buffers.fetch_add(batch->buffers.size(), std::memory_order_relaxed);
        shared_->render_passes.fetch_add(batch->render_passes, std::memory_order_relaxed);
        shared_->draws.fetch_add(batch->draws, std::memory_order_relaxed);
        return renderer_success();
    }

    void join_encodes() const noexcept {
        std::unique_lock lock(encode_mutex_);
        if (encode_sleeping_ && encode_queued_) encode_ready_.notify_one();
        ++encode_joiners_;
        encode_idle_.wait(lock, [this] { return encode_pending_ == 0; });
        --encode_joiners_;
    }

    std::optional<RendererError> encoding_failure() const {
        std::lock_guard lock(encode_mutex_);
        return encode_failure_;
    }

    void encode_loop() {
        for (;;) {
            std::unique_lock lock(encode_mutex_);
            while (!stopping_ && retired_queued_ == 0 && encode_queued_ == 0 &&
                   (deferred_commits_.empty() ||
                    deferred_commits_.front().generation > released_generation_)) {
                encode_sleeping_ = true;
                encode_ready_.wait(lock);
                encode_sleeping_ = false;
            }
            if (stopping_) return;
            // Cleanup is independent of commit holds and encoding failures.
            // Prioritize it so retirement can release admission pressure even
            // while later command commits are held behind an ordered effect.
            if (retired_queued_) {
                auto retired = std::move(retired_batches_[retired_head_]);
                retired_batches_[retired_head_].reset();
                retired_head_ = (retired_head_ + 1) % retired_batches_.size();
                --retired_queued_;
                lock.unlock();
                @autoreleasepool {
                    retired.reset();
                }
                lock.lock();
                --retirement_pending_;
                --encode_pending_;
                lock.unlock();
                encode_idle_.notify_all();
                notify_feedback();
                continue;
            }
            EncodeJob job;
            if (!deferred_commits_.empty() &&
                deferred_commits_.front().generation <= released_generation_) {
                job.batch = std::move(deferred_commits_.front().batch);
                deferred_commits_.erase(deferred_commits_.begin());
            } else {
                job = std::move(encode_jobs_[encode_head_]);
                encode_head_ = (encode_head_ + 1) % encode_jobs_.size();
                --encode_queued_;
            }
            const bool defer_commit = !job.packet &&
                job.hold_generation > released_generation_;
            auto packet = std::move(job.packet);
            auto prior_error = encode_failure_;
            lock.unlock();
            encode_space_.notify_one();
            @autoreleasepool {
                RendererStatus status = prior_error ? RendererStatus(*prior_error) : renderer_success();
                if (!prior_error) {
                    try {
                        if (packet && shared_->test_control)
                            status = shared_->test_control->before_encode(packet->prepared.request.id);
                        if (!packet) {
                            status = defer_commit ? close_commands(*job.batch)
                                                  : commit_batch(job.batch);
                        } else if (job.repeat) {
                            if (succeeded(status))
                                status = encode_repeat(*job.batch, *packet, *job.repeat,
                                                       shared_->device->device(), tables_);
                        } else {
                            if (succeeded(status))
                                status = prepare_native(*packet, *job.batch, shared_->device->device(), tables_);
                            if (succeeded(status))
                                status = encode_packet(*job.batch, *packet, shared_->device->device());
                        }
                    } catch (const std::bad_alloc&) {
                        status = failure("submission encoding allocation failed", CommandStatus::MemoryFault);
                    } catch (...) {
                        status = failure("submission encoding threw an exception", CommandStatus::Stopped);
                    }
                }
                if (defer_commit && succeeded(status)) {
                    lock.lock();
                    deferred_commits_.push_back(
                        DeferredCommit{std::move(job.batch), job.hold_generation});
                    lock.unlock();
                    continue;
                }
                if (!packet) {
                    if (const auto* error = std::get_if<RendererError>(&status)) {
                        if (job.batch->render) close_render(*job.batch);
                        for (auto& completion : job.batch->completions)
                            static_cast<void>(completion.feedback(error->command()));
                    }
                    job.batch->commit_done.store(true, std::memory_order_release);
                    notify_feedback();
                }
                // Release worker ownership before announcing quiescence.
                packet.reset();
                job.repeat.reset();
                job.batch.reset();
                lock.lock();
                if (auto* error = std::get_if<RendererError>(&status); error && !encode_failure_)
                    encode_failure_ = std::move(*error);
                if (std::holds_alternative<RendererError>(status))
                    shared_->progress.fetch_add(1, std::memory_order_release);
                --encode_pending_;
            }
            const bool idle = encode_pending_ == 0 && encode_joiners_ != 0;
            lock.unlock();
            if (idle) encode_idle_.notify_all();
        }
    }

    struct SubmissionIdLess {
        bool operator()(const SubmissionId& left, const SubmissionId& right) const {
            return std::tie(left.queue, left.sequence) <
                   std::tie(right.queue, right.sequence);
        }
    };

    void reject_unaccepted(SubmissionCompletion& completion, CommandResult error) {
        static_cast<void>(completion.reject(error));
    }

    RendererStatus fail_current_status(CommandResult error) {
        fail_current(error);
        return RendererError(error);
    }

    void fail_current(CommandResult error, bool poison_publication = true) {
        last_draw_.reset();
        if (commit_held()) static_cast<void>(abort_commit_hold(error));
        if (!current_) return;
        join_encodes();
        shared_->memory->discard_access_journal(current_->lease);
        // An encoder failure poisons admission immediately via encode_failure_,
        // but earlier committed batches may still publish. Retirement advances
        // ordered_failure when it reaches this rejected batch.
        if (poison_publication && !shared_->ordered_failure)
            shared_->ordered_failure = RendererError(error);
        if (current_->render) [current_->render endEncoding];
        for (auto& packet : current_->packets)
            static_cast<void>(packet.completion.reject(error));
        SubmittedBatch rejected;
        rejected.packets = std::move(current_->packets);
        if (!rejected.packets.empty()) inflight_.push_back(std::move(rejected));
        current_.reset();
    }

    RendererStatus dependencies_ready(const Batch& batch, bool& terminal) const {
        terminal = false;
        bool pending = false;
        std::set<SubmissionId, SubmissionIdLess> earlier;
        for (const auto& packet : batch.packets) {
            for (const auto list : {std::span<const CompletionToken>(packet.dependencies),
                                    std::span<const CompletionToken>(packet.lease->acceptance.dependencies)}) {
                for (const auto& dependency : list) {
                    const auto snapshot = dependency.snapshot();
                    if (snapshot.error) {
                        terminal = true;
                        return *snapshot.error;
                    }
                    if (snapshot.phase == SubmissionPhase::Complete ||
                        snapshot.phase == SubmissionPhase::GpuComplete ||
                        snapshot.phase == SubmissionPhase::Publishing)
                        continue;
                    if (earlier.contains(snapshot.id) ||
                        submitted_ids_.contains(snapshot.id))
                        continue;
                    pending = true;
                }
            }
            earlier.insert(packet.lease->prepared.request.id);
        }
        if (pending)
            return failure("submission dependency has not reached a GPU-visible order point",
                           CommandStatus::Stopped);
        return renderer_success();
    }

    std::shared_ptr<SharedState> shared_;
    ArgumentTables tables_;
    NativePool pool_;
    std::uint32_t max_commands_ = 0, max_inflight_ = 0;
    std::uint64_t queue_identity_ = 0, last_sequence_ = 0;
    std::shared_ptr<Batch> current_;
    std::vector<SubmittedBatch> inflight_;
    std::set<SubmissionId, SubmissionIdLess> submitted_ids_;
    mutable std::mutex encode_mutex_;
    mutable std::condition_variable encode_ready_;
    mutable std::condition_variable encode_idle_;
    std::condition_variable encode_space_;
    std::vector<EncodeJob> encode_jobs_;
    std::vector<DeferredCommit> deferred_commits_;
    // At most max_inflight_ waiting batches plus one active destruction. Full
    // rings retain retired batches in inflight_ until the next poll.
    std::vector<std::optional<SubmittedBatch>> retired_batches_;
    std::size_t retired_head_ = 0, retired_tail_ = 0, retired_queued_ = 0;
    std::size_t retirement_pending_ = 0;
    // The last appended packet when it is a draw still open in the current
    // batch: the only packet a repeat may extend.
    std::shared_ptr<PacketLease> last_draw_;
    std::size_t encode_head_ = 0, encode_tail_ = 0, encode_queued_ = 0;
    bool encode_sleeping_ = false;
    mutable std::size_t encode_joiners_ = 0;
    std::size_t encode_pending_ = 0;
    std::optional<RendererError> encode_failure_;
    bool stopping_ = false;
    bool commit_held_ = false;
    bool aborted_hold_ = false;
    std::uint64_t hold_generation_ = 0, released_generation_ = 0;
    std::thread encoder_;

public:
    SubmissionStatistics statistics(bool wait_for_encoding) const noexcept {
        if (wait_for_encoding && !commit_held()) join_encodes();
        return {
            shared_->accepted.load(std::memory_order_relaxed),
            shared_->command_buffers.load(std::memory_order_relaxed),
            shared_->render_passes.load(std::memory_order_relaxed),
            shared_->draws.load(std::memory_order_relaxed),
            tables_.created.load(std::memory_order_relaxed),
            pool_.pages_created.load(std::memory_order_relaxed),
            pool_.pages_reused.load(std::memory_order_relaxed),
            pool_.slices_created.load(std::memory_order_relaxed),
            pool_.slices_reused.load(std::memory_order_relaxed),
            pool_.commands_created.load(std::memory_order_relaxed),
            pool_.commands_reused.load(std::memory_order_relaxed),
        };
    }
};

} // namespace

RendererResult<std::shared_ptr<SubmissionQueue>> create_submission_queue(
    std::shared_ptr<DeviceContext> device, std::shared_ptr<ResourceMemory> memory,
    std::uint32_t max_commands_per_submission,
    std::uint32_t max_inflight_submissions) {
    return create_submission_queue_for_testing(std::move(device), std::move(memory),
        max_commands_per_submission, max_inflight_submissions, nullptr);
}

RendererResult<std::shared_ptr<SubmissionQueue>> create_submission_queue_for_testing(
    std::shared_ptr<DeviceContext> device, std::shared_ptr<ResourceMemory> memory,
    std::uint32_t max_commands_per_submission,
    std::uint32_t max_inflight_submissions,
    std::shared_ptr<SubmissionTestControl> control) {
    if (!device || !device->device() || !device->queue() ||
        !device->identity().device || !memory || !max_commands_per_submission ||
        !max_inflight_submissions ||
        ![device->device() supportsFamily:MTLGPUFamilyMetal4])
        return failure("submission queue requires a Metal 4 domain and nonzero limits");
    try {
        auto state = std::make_shared<SharedState>();
        state->device = std::move(device);
        state->memory = std::move(memory);
        state->test_control = std::move(control);
        return std::shared_ptr<SubmissionQueue>(std::make_shared<Queue>(
            std::move(state), max_commands_per_submission,
            max_inflight_submissions));
    } catch (const std::bad_alloc&) {
        return failure("submission queue allocation failed", CommandStatus::MemoryFault);
    } catch (const std::system_error&) {
        return failure("submission encoder thread creation failed", CommandStatus::Stopped);
    }
}

SubmissionStatistics submission_statistics(const SubmissionQueue& queue, bool wait_for_encoding) noexcept {
    const auto* native = dynamic_cast<const Queue*>(&queue);
    return native ? native->statistics(wait_for_encoding) : SubmissionStatistics{};
}

bool commit_hold_blocked(const RendererStatus& status) noexcept {
    const auto* error = std::get_if<RendererError>(&status);
    return error && error->command().status == CommandStatus::Stopped &&
        error->command().detail == commit_hold_detail;
}

RendererStatus commit_hold_wait() {
    return failure(commit_hold_detail, CommandStatus::Stopped);
}

} // namespace armigo::platform::metal
