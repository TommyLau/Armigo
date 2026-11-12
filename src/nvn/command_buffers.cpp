// ============================================================================
// Armigo NVN - Command Buffer Recording
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/execution/guest_preempt.h>

#include "commands.h"
#include "entry_table.h"
#include "flat_map.h"
#include "programs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace armigo::nvn {

namespace {

constexpr std::size_t command_arena_size = 1 << 20;
constexpr std::size_t command_reserve = 24;

struct CommandArena {
    explicit CommandArena(std::size_t capacity)
        : bytes(std::make_unique_for_overwrite<std::byte[]>(capacity)), capacity(capacity) {}
    std::unique_ptr<std::byte[]> bytes;
    std::size_t capacity, used = 0;
};

struct CommandBlock {
    std::shared_ptr<CommandArena> arena;
    std::byte* bytes;
    std::size_t capacity, used = 0;
};

// Lists share allocation arenas, but retain their own ranges. An arena is
// released only after every list and submission referencing it has retired.
CommandBlock allocate_block(std::size_t size) {
    static std::mutex mutex;
    static std::shared_ptr<CommandArena> current;
    size = (size + 15) & ~std::size_t(15);
    std::lock_guard lock(mutex);
    auto arena = current;
    if (size > command_arena_size / 4) {
        arena = std::make_shared<CommandArena>(size);
    } else if (!arena || size > arena->capacity - arena->used) {
        arena = current = std::make_shared<CommandArena>(command_arena_size);
    }
    auto* bytes = arena->bytes.get() + arena->used;
    arena->used += size;
    return {std::move(arena), bytes, size};
}

void write_header(std::byte* storage, Op op, std::size_t size) {
    const RecordHeader header{static_cast<std::uint32_t>(op), static_cast<std::uint32_t>(size)};
    std::memcpy(storage, &header, sizeof(header));
}

} // namespace

// The record stream is written once and read in place. Guest command-memory
// accounting remains separate because host records exceed native method words.
struct CommandList {
    NVNdevice* device{};
    // Sampler registrations last stamped into this list; 0 at the start and
    // after a call, so the next draw stamps again.
    std::uint64_t stamped_registrations = 0;
    std::vector<CommandBlock> blocks;
    std::byte* cursor = nullptr;
    std::byte* end_cursor = nullptr;
    std::uintptr_t native_start = 0;
    // The game command memory the recording took, in the native positions the
    // game reads back: one range per segment it used. range_start opens the
    // range of the current segment while recording.
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> native_ranges;
    std::uintptr_t range_start = 0;

    // A recording that comes back to memory it already took (the game added
    // the same segment again) keeps one range there: the ranges of one list
    // never overlap, so the registry's index stays a set of disjoint ranges.
    void close_native_range(std::uintptr_t end) {
        if (!range_start) return;
        auto start = range_start;
        auto stop = end;
        range_start = 0;
        std::erase_if(native_ranges, [&](const auto& range) {
            if (std::max(range.second, range.first + 1) <= start || std::max(stop, start + 1) <= range.first)
                return false;
            start = std::min(start, range.first);
            stop = std::max(stop, range.second);
            return true;
        });
        native_ranges.emplace_back(start, stop);
    }

    std::byte* allocate(std::size_t size) {
        if (!cursor || size + command_reserve > std::size_t(end_cursor - cursor)) {
            auto capacity = blocks.empty() ? 512 : std::min<std::size_t>(blocks.back().capacity * 2, 64 << 10);
            capacity = std::max(capacity, size + command_reserve + 16);
            auto next = allocate_block(capacity);
            if (cursor) {
                write_header(cursor, Op::jump, 16);
                std::memcpy(cursor + sizeof(RecordHeader), &next.bytes, sizeof(next.bytes));
                blocks.back().used = std::size_t(cursor + 16 - blocks.back().bytes);
            }
            cursor = next.bytes;
            end_cursor = cursor + next.capacity;
            blocks.push_back(std::move(next));
        }
        auto* result = cursor;
        cursor += size;
        return result;
    }

    void finish() {
        write_header(allocate(sizeof(RecordHeader)), Op::end, sizeof(RecordHeader));
        blocks.back().used = std::size_t(cursor - blocks.back().bytes);
    }

    struct Iterator {
        const CommandList* list;
        std::size_t index = 0, offset = 0;
        std::span<const std::byte> operator*() const {
            const auto& block = list->blocks[index];
            const auto remaining = std::span<const std::byte>(block.bytes, block.used).subspan(offset);
            if (remaining.size() < sizeof(RecordHeader)) return remaining;
            RecordHeader header;
            std::memcpy(&header, remaining.data(), sizeof(header));
            if (header.size < sizeof(header) || header.size > remaining.size()) return remaining;
            return remaining.first(header.size);
        }
        Iterator& operator++() {
            offset += operator*().size();
            if (offset == list->blocks[index].used) { ++index; offset = 0; }
            return *this;
        }
        bool operator!=(const Iterator& other) const { return index != other.index || offset != other.offset; }
    };
    Iterator begin() const { return {this}; }
    Iterator end() const { return {this, blocks.size(), 0}; }

    bool valid_jump(const std::byte* record, const std::byte* target) const {
        for (std::size_t i = 0; i + 1 < blocks.size(); ++i)
            if (record == blocks[i].bytes + blocks[i].used - 16)
                return target == blocks[i + 1].bytes;
        return false;
    }
    // Only records needing submit-thread work are indexed, never ordinary draws.
    std::vector<const std::byte*> submit_records;
    std::vector<std::shared_ptr<const ResolvedProgram>> program_snapshots;
    std::vector<std::shared_ptr<const CommandList>> copied_lists;
};

struct CommandSubmission {
    std::uint64_t sampler_registration_limit = 0;
    struct List {
        std::shared_ptr<const CommandList> commands;
        unsigned height = 0;
        bool complete = false;
    };
    std::vector<NVNcommandHandle> handles;
    // Keyed by handle, and by record address for the submit-time snapshots.
    FlatMap64<List> lists;
    struct DeferredBinding { std::int32_t stage, first, count, reserved; };
    static constexpr std::size_t deferred_header_words = sizeof(DeferredBinding) / sizeof(std::uint64_t);
    FlatMap64<std::size_t> deferred;
    std::vector<std::uint64_t> deferred_words;
    FlatMap64<DescriptorPoolState> descriptor_pools;
    FlatMap64<MemoryPoolState> scratch_pools;
    std::vector<SyncState*> syncs;
    // A list of the submission records an event wait: only then does replay
    // look for waits to park the submission's queue at.
    bool event_waits = false;
};

namespace {

constexpr std::uint32_t command_buffer_tag = object_tag('C', 'M', 'D', 'B');
constexpr std::size_t record_alignment = 8;
constexpr std::size_t control_accounting_size = sizeof(NVNcommandHandle);
constexpr unsigned max_call_depth = 64;

struct CommandCall {
    NVNcommandHandle handle;
};

std::mutex registry_mutex;
FlatMap64<std::shared_ptr<const CommandList>> command_handles;
std::atomic<NVNcommandHandle> next_handle{1};

// The game command memory each registered list occupies, by range start. A
// recording that reuses memory a registered list occupies replaces that list:
// natively its commands have been overwritten, so its handle no longer names
// them. Without this, lists whose start was not recorded at again stayed
// registered for the whole run, and so did the host blocks they hold.
struct OccupiedRange {
    std::uintptr_t end;
    NVNcommandHandle handle;
};
std::map<std::uintptr_t, OccupiedRange> occupied_ranges;

// An empty range still claims its start: a recording there replaces it.
std::uintptr_t claimed_end(std::uintptr_t start, std::uintptr_t end) noexcept {
    return std::max(end, start + 1);
}

// registry_mutex held.
void unregister_handle(NVNcommandHandle handle) {
    const auto* found = command_handles.find(handle);
    if (!found) return;
    for (const auto& [start, end] : (*found)->native_ranges) {
        static_cast<void>(end);
        if (const auto range = occupied_ranges.find(start);
            range != occupied_ranges.end() && range->second.handle == handle)
            occupied_ranges.erase(range);
    }
    command_handles.erase(handle);
}

// registry_mutex held. Unregisters every list whose memory `list` reuses, then
// registers `list` under `handle`.
void register_handle(NVNcommandHandle handle, std::shared_ptr<const CommandList> list) {
    std::vector<NVNcommandHandle> replaced;
    for (const auto& [start, end] : list->native_ranges) {
        const auto limit = claimed_end(start, end);
        auto range = occupied_ranges.upper_bound(start);
        if (range != occupied_ranges.begin()) {
            const auto previous = std::prev(range);
            if (claimed_end(previous->first, previous->second.end) > start) range = previous;
        }
        for (; range != occupied_ranges.end() && range->first < limit; ++range)
            replaced.push_back(range->second.handle);
    }
    for (const auto old : replaced) unregister_handle(old);
    unregister_handle(handle);
    for (const auto& [start, end] : list->native_ranges)
        occupied_ranges.insert_or_assign(start, OccupiedRange{end, handle});
    command_handles.insert_or_assign(handle, std::move(list));
}

template <typename State, typename Object> State* as(Object* object) noexcept {
    return reinterpret_cast<State*>(object);
}

template <typename State, typename Object> const State* as(const Object* object) noexcept {
    return reinterpret_cast<const State*>(object);
}

std::size_t aligned_size(std::size_t size) noexcept { return (size + record_alignment - 1) & ~(record_alignment - 1); }

void diagnostic(const char* operation, const char* message) noexcept {
    armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: %s: %s\n", operation, message);
}

CommandBufferState* command_state(NVNcommandBuffer* command_buffer, const char* operation) noexcept {
    if (!command_buffer) {
        diagnostic(operation, "null command buffer");
        return nullptr;
    }
    auto* state = as<CommandBufferState>(command_buffer);
    if (state->tag != command_buffer_tag) {
        diagnostic(operation, "uninitialized command buffer");
        return nullptr;
    }
    return state;
}

const CommandBufferState* command_state(const NVNcommandBuffer* command_buffer, const char* operation) noexcept {
    return command_state(const_cast<NVNcommandBuffer*>(command_buffer), operation);
}

CommandList* recording_list(CommandBufferState* state, const char* operation) noexcept {
    if (!state || !state->recording || !state->list) {
        diagnostic(operation, "command buffer is not recording");
        return nullptr;
    }
    return state->list;
}

void fail_recording(NVNcommandBuffer* command_buffer) noexcept {
    if (command_buffer)
        as<CommandBufferState>(command_buffer)->recording_failed = 1;
}

// Every recorded command advances the game's command-memory position by one
// native word, the least a native command takes, so the game's cursor never
// runs further than natively. When the segment is used up the game's memory
// callback is asked for more; without it the position simply stops advancing.
constexpr std::size_t native_command_size = 4;
void account_native_command(NVNcommandBuffer* command_buffer, CommandBufferState& state) noexcept {
    if (!state.donated_cursor) return;
    if (std::size_t(state.donated_end - state.donated_cursor) < native_command_size &&
        state.memory_callback && !state.command_callback_disabled && !state.callback_exhausted) {
        // The game's callback runs guest code on top of live NVN frames: an
        // asynchronous stop must not unwind through them.
        const execution::GuestPreemptHold hold;
        state.memory_callback(command_buffer, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,
                              native_command_size, state.callback_data);
        if (std::size_t(state.donated_end - state.donated_cursor) < native_command_size)
            state.callback_exhausted = 1;
    }
    if (std::size_t(state.donated_end - state.donated_cursor) >= native_command_size)
        state.donated_cursor += native_command_size;
}

bool request_control_memory(NVNcommandBuffer* command_buffer, CommandBufferState& state) noexcept {
    auto consume = [&]() {
        if (state.control_base && state.control_cursor <= state.control_end &&
            std::size_t(state.control_end - state.control_cursor) >= control_accounting_size) {
            state.control_cursor += control_accounting_size;
            state.control_exhausted = 0;
            return true;
        }
        return false;
    };
    if (consume())
        return true;

    if (!state.memory_callback || state.control_callback_disabled || state.control_exhausted) {
        if (!state.control_exhausted)
            diagnostic("command recording", "out of donated control memory");
        state.control_exhausted = 1;
        fail_recording(command_buffer);
        return false;
    }

    {
        // Same as above: guest code on top of live host frames is never stopped.
        const execution::GuestPreemptHold hold;
        state.memory_callback(command_buffer, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY,
                              control_accounting_size, state.callback_data);
    }
    if (consume())
        return true;
    diagnostic("command recording", "control-memory callback did not provide enough memory");
    state.control_exhausted = 1;
    fail_recording(command_buffer);
    return false;
}

bool samples_textures(Op op) {
    switch (op) {
    case Op::draw: case Op::draw_indexed: case Op::draw_indirect: case Op::draw_indexed_indirect:
    case Op::multi_draw_indirect: case Op::multi_draw_indexed_indirect: case Op::draw_texture:
    case Op::transform_feedback: case Op::dispatch: case Op::dispatch_indirect:
    case Op::draw_mesh_tasks: case Op::draw_mesh_tasks_indirect:
        return true;
    default:
        return false;
    }
}

void* append_record(NVNcommandBuffer* command_buffer, Op op, std::size_t payload_size, const char* operation) {
    auto* state = command_state(command_buffer, operation);
    auto* list = recording_list(state, operation);
    if (!list || state->recording_failed)
        return nullptr;
    // Keep legacy registration stamps in the recording format. Captured
    // replay selects descriptor contents at its immutable submission boundary.
    if (samples_textures(op)) {
        const auto registrations = sampler_registrations.load(std::memory_order_acquire);
        if (registrations != list->stamped_registrations) {
            auto* stamp = static_cast<SamplerRegistrations*>(append_record(
                command_buffer, Op::sampler_registrations, sizeof(SamplerRegistrations), operation));
            if (!stamp)
                return nullptr;
            stamp->value = registrations;
            list->stamped_registrations = registrations;
        }
    }
    if (payload_size > std::numeric_limits<std::uint32_t>::max() - sizeof(RecordHeader)) {
        diagnostic(operation, "record is too large");
        return nullptr;
    }
    const auto total = aligned_size(sizeof(RecordHeader) + payload_size);
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        diagnostic(operation, "record size overflow");
        return nullptr;
    }
    auto* storage = list->allocate(total);
    // A stamp is the replay's own record: it takes none of the game's command
    // memory.
    if (op != Op::sampler_registrations)
        account_native_command(command_buffer, *state);
    const RecordHeader header{static_cast<std::uint32_t>(op), static_cast<std::uint32_t>(total)};
    std::memcpy(storage, &header, sizeof(header));
    if (op == Op::call)
        list->stamped_registrations = 0;
    if (op == Op::call || op == Op::bind_textures_deferred || op == Op::fence_sync || op == Op::wait_event ||
        op == Op::set_texture_pool || op == Op::set_sampler_pool || op == Op::shader_scratch_memory)
        list->submit_records.push_back(storage);
    return storage + sizeof(header);
}

template <typename Payload> Payload* append(NVNcommandBuffer* command_buffer, Op op, const char* operation) {
    return static_cast<Payload*>(append_record(command_buffer, op, sizeof(Payload), operation));
}

void append_unsupported(NVNcommandBuffer* command_buffer, Entry entry, const char* operation) {
    diagnostic(operation, "operation is recorded as unsupported");
    if (auto* payload = append<Unsupported>(command_buffer, Op::unsupported, operation))
        payload->entry = static_cast<std::uint32_t>(entry);
}

const TextureState* texture_snapshot(NVNcommandBuffer* command_buffer, const NVNtexture* texture,
                                     const char* operation) {
    if (!texture)
        return nullptr;
    const auto* state = as<TextureState>(texture);
    if (!state->snapshot) {
        diagnostic(operation, "texture has no immutable snapshot");
        fail_recording(command_buffer);
        return nullptr;
    }
    return static_cast<const TextureState*>(state->snapshot);
}

const ProgramState* program_snapshot(NVNcommandBuffer* command_buffer, const NVNprogram* program,
                                     const char* operation) noexcept {
    if (!program)
        return nullptr;
    const auto* state = as<ProgramState>(program);
    if (!state->snapshot) {
        diagnostic(operation, "program has no immutable snapshot");
        fail_recording(command_buffer);
        return nullptr;
    }
    // The list owns one executable reference per serial, independent of the
    // mutable guest object. Repeated binds only reuse the immutable pointer.
    auto* list = as<CommandBufferState>(command_buffer)->list;
    const auto* snapshot = static_cast<const ProgramState*>(state->snapshot);
    for (const auto& retained : list->program_snapshots)
        if (&retained->state == snapshot) return snapshot;
    auto retained = retain_program_snapshot(*state);
    if (!retained) {
        diagnostic(operation, "program snapshot is no longer available");
        fail_recording(command_buffer);
        return nullptr;
    }
    list->program_snapshots.push_back(std::move(retained));
    return snapshot;
}

void copy_view(TextureViewState& destination, std::uint8_t& present, const NVNtextureView* view) noexcept {
    present = view ? 1 : 0;
    destination = view ? *as<TextureViewState>(view) : TextureViewState{};
}

bool valid_range(int first, int count, int limit) noexcept {
    return first >= 0 && count >= 0 && first <= limit && count <= limit - first;
}

bool valid_deferred_texture_range(int first, int count) noexcept {
    if (first < 0) {
        char text[224];
        std::snprintf(text, sizeof text,
            "unsupported deferred first=%d: original driver uses 8-bit truncation, "
            "offset = 0x20 + 8*(first & 0xff) = %#x",
            first, 0x20u + 8u * (static_cast<unsigned>(first) & 0xffu));
        diagnostic("nvnCommandBufferBindTexturesDeferred", text);
        return false;
    }
    return valid_range(first, count, 128);
}

bool valid_binding_stage(std::int32_t stage) noexcept {
    return static_cast<unsigned>(stage) < logical_binding_stage_count;
}

template <typename Handle>
void bind_handles(NVNcommandBuffer* command_buffer, Op op, NVNshaderStage stage, int first, int count,
                  const Handle* handles, int limit, Entry entry, const char* operation, int slot_base = 0) {
    const auto api_stage = stage;
    // The graphics combined-handle table begins at byte 0x20. Index -1
    // updates its 0x18 prefix, rather than any nonnegative texture slot.
    const bool combined = entry == Entry::nvnCommandBufferBindTexture || entry == Entry::nvnCommandBufferBindTextures;
    if (combined && first == -1 && valid_binding_stage(stage) && stage != NVN_SHADER_STAGE_COMPUTE &&
        count > 0 && handles && valid_range(0, count - 1, limit)) {
        auto* prefix = append<BindTexturePrefix>(command_buffer, Op::bind_texture_prefix, operation);
        if (!prefix) return;
        prefix->stage = stage;
        prefix->bytes = entry == Entry::nvnCommandBufferBindTexture ? 4 : 8;
        std::memcpy(&prefix->handle, handles, 8);
        prefix->sampler_registrations = sampler_registrations.load(std::memory_order_acquire);
        if (const char* trace = std::getenv("ARMIGO_NVN_STAGE_DIAGNOSTICS"); trace && *trace == '1') {
            static std::array<std::atomic_bool, logical_binding_stage_count * 2> reported{};
            const auto key = std::size_t(stage) * 2 + (prefix->bytes == 8);
            if (!reported[key].exchange(true, std::memory_order_relaxed)) {
                char text[160];
                std::snprintf(text, sizeof text,
                    "graphics texture prefix write stage=%d index=-1 offset=0x18 bytes=%u handle=%#llx",
                    static_cast<int>(stage), prefix->bytes, static_cast<unsigned long long>(prefix->handle));
                diagnostic(operation, text);
            }
        }
        if (auto* list = as<CommandBufferState>(command_buffer)->list)
            list->stamped_registrations = prefix->sampler_registrations;
        ++handles;
        --count;
        first = 0;
    }
    if (!valid_binding_stage(stage) || !valid_range(first, count, limit) ||
        (count && !handles)) {
        char text[176];
        std::snprintf(text, sizeof text,
            "invalid handle binding api_stage=%d replay_stage=%d first=%d count=%d limit=%d source=%s",
            static_cast<int>(api_stage), static_cast<int>(stage), first, count, limit,
            handles ? "present" : "null");
        diagnostic(operation, text);
        append_unsupported(command_buffer, entry, operation);
        return;
    }
    for (int consumed = 0; consumed < count;) {
        const int batch = std::min(32, count - consumed);
        auto* payload = static_cast<BindHandles*>(append_record(command_buffer, op,
            offsetof(BindHandles, handles) + std::size_t(batch) * sizeof(std::uint64_t), operation));
        if (!payload)
            return;
        payload->stage = stage;
        payload->first = slot_base + first + consumed;
        payload->count = batch;
        payload->sampler_registrations = sampler_registrations.load(std::memory_order_acquire);
        if (op != Op::bind_images)
            if (auto* list = command_state(command_buffer, operation)->list)
                list->stamped_registrations = payload->sampler_registrations;
        for (int i = 0; i < batch; ++i)
            std::memcpy(&payload->handles[i], &handles[consumed + i], sizeof(std::uint64_t));
        consumed += batch;
    }
}

void bind_buffer_ranges(NVNcommandBuffer* command_buffer, Op op, NVNshaderStage stage, int first, int count,
                        const NVNbufferRange* ranges, Entry entry, const char* operation) {
    if (!valid_binding_stage(stage) || !valid_range(first, count, 16) ||
        (count && !ranges)) {
        diagnostic(operation, "invalid stage, range, or source array");
        fail_recording(command_buffer);
        return;
    }
    for (int i = 0; i < count; ++i) {
        auto* payload = append<BindBuffer>(command_buffer, op, operation);
        if (!payload)
            return;
        *payload = {stage, first + i, ranges[i].address, ranges[i].size};
    }
}

std::shared_ptr<const CommandList> find_handle(NVNcommandHandle handle) noexcept {
    std::lock_guard lock(registry_mutex);
    const auto* found = command_handles.find(handle);
    return found ? *found : nullptr;
}

constexpr std::size_t payload_minimum(Op op) noexcept {
    switch (op) {
    case Op::signal_event:
        return sizeof(SignalEvent);
    case Op::report_counter:
        return sizeof(ReportCounter);
    case Op::copy_buffer:
        return sizeof(CopyBuffer);
    case Op::update_uniforms:
        return offsetof(UpdateUniforms, data);
    case Op::set_render_targets:
        return sizeof(SetRenderTargets);
    case Op::clear_color:
        return sizeof(ClearColor);
    case Op::clear_depth_stencil:
        return sizeof(ClearDepthStencil);
    case Op::viewports:
        return sizeof(Viewports);
    case Op::scissor:
        return sizeof(Scissor);
    case Op::depth_range:
        return sizeof(DepthRange);
    case Op::bind_program:
        return sizeof(BindProgram);
    case Op::bind_state:
        return sizeof(BindState);
    case Op::bind_vertex_attributes:
        return offsetof(BindVertexAttributes, states);
    case Op::bind_vertex_streams:
        return offsetof(BindVertexStreams, states);
    case Op::bind_vertex_buffers:
        return offsetof(BindVertexBuffers, ranges);
    case Op::bind_uniform_buffer:
    case Op::bind_storage_buffer:
        return sizeof(BindBuffer);
    case Op::bind_textures:
    case Op::bind_samplers:
    case Op::bind_images:
        return offsetof(BindHandles, handles);
    case Op::stencil:
        return sizeof(Stencil);
    case Op::polygon_offset:
        return sizeof(PolygonOffset);
    case Op::draw:
        return sizeof(Draw);
    case Op::draw_indexed:
        return sizeof(DrawIndexed);
    case Op::dispatch:
        return sizeof(Dispatch);
    case Op::barrier:
    case Op::geometry_processing_mode:
        return sizeof(ScalarInt);
    case Op::set_texture_pool:
    case Op::set_sampler_pool:
        return sizeof(SetDescriptorPool);
    case Op::copy_buffer_to_texture:
    case Op::copy_texture_to_buffer:
        return sizeof(CopyBufferToTexture);
    case Op::copy_stride:
    case Op::copy_image_stride:
        return sizeof(CopyStride);
    case Op::blend_color:
        return sizeof(BlendColor);
    case Op::copy_texture_to_texture:
        return sizeof(CopyTextureToTexture);
    case Op::reset_counter:
        return sizeof(ResetCounter);
    case Op::clear_texture:
        return sizeof(ClearTexture);
    case Op::depth_clamp:
        return sizeof(DepthClamp);
    case Op::line_width:
        return sizeof(LineWidth);
    case Op::point_size:
        return sizeof(PointSize);
    case Op::bind_textures_deferred:
        return sizeof(BindTexturesDeferred);
    case Op::unsupported:
        return sizeof(Unsupported);
    case Op::wait_event:
        return sizeof(WaitEvent);
    case Op::wait_sync:
        return sizeof(WaitSync);
    case Op::fence_sync:
        return sizeof(FenceSync);
    case Op::clear_buffer:
        return sizeof(ClearBuffer);
    case Op::report_value:
        return sizeof(ReportValue);
    case Op::render_enable:
        return sizeof(RenderEnable);
    case Op::render_enable_conditional:
        return sizeof(RenderEnableConditional);
    case Op::discard_color:
        return sizeof(DiscardColor);
    case Op::discard_depth_stencil:
    case Op::resolve_depth_buffer:
        return 0;
    case Op::downsample:
        return sizeof(Downsample);
    case Op::draw_indirect:
        return sizeof(DrawIndirect);
    case Op::draw_indexed_indirect:
        return sizeof(DrawIndexedIndirect);
    case Op::multi_draw_indirect:
        return sizeof(MultiDrawIndirect);
    case Op::multi_draw_indexed_indirect:
        return sizeof(MultiDrawIndexedIndirect);
    case Op::viewport_swizzles:
        return sizeof(ViewportSwizzles);
    case Op::scissors:
        return sizeof(Scissors);
    case Op::depth_bounds:
        return sizeof(DepthBounds);
    case Op::depth_ranges:
        return sizeof(DepthRanges);
    case Op::bind_transform_feedback_buffer:
        return sizeof(BindTransformFeedbackBuffer);
    case Op::transform_feedback:
        return sizeof(TransformFeedback);
    case Op::patch_size:
        return sizeof(PatchSize);
    case Op::tessellation_levels:
        return sizeof(TessellationLevels);
    case Op::primitive_restart:
        return sizeof(PrimitiveRestart);
    case Op::alpha_ref:
    case Op::conservative_raster_dilate:
        return sizeof(ScalarFloat);
    case Op::sample_mask:
    case Op::rasterizer_discard:
        return sizeof(ScalarInt);
    case Op::conservative_raster:
        return sizeof(ConservativeRaster);
    case Op::subpixel_precision_bias:
        return sizeof(SubpixelPrecisionBias);
    case Op::shader_scratch_memory:
        return sizeof(ShaderScratchMemory);
    case Op::zcull_data:
        return sizeof(ZCullData);
    case Op::draw_texture:
        return sizeof(DrawTexture);
    case Op::program_subroutines:
        return offsetof(ProgramSubroutines, values);
    case Op::coverage_modulation_table:
        return sizeof(CoverageModulationTable);
    case Op::debug_marker:
        return offsetof(DebugMarker, text);
    case Op::clear_color_integer:
        return sizeof(ClearColorInteger);
    case Op::dispatch_indirect:
        return sizeof(DispatchIndirect);
    case Op::bind_texture_prefix:
        return sizeof(BindTexturePrefix);
    case Op::sampler_registrations:
        return sizeof(SamplerRegistrations);
    case Op::push_constants:
        return sizeof(PushConstants);
    case Op::draw_mesh_tasks:
        return sizeof(DrawMeshTasks);
    case Op::draw_mesh_tasks_indirect:
        return sizeof(DrawMeshTasksIndirect);
    case Op::call:
        return sizeof(CommandCall);
    default:
        return 0;
    }
}

// Looked up by op: a switch over every op compiles to an indirect branch that
// the record stream's mixed ops mispredict, once per record.
constexpr auto minimum_payload_sizes = [] {
    std::array<std::uint32_t, std::size_t(Op::count)> sizes{};
    for (std::size_t op = 0; op != sizes.size(); ++op)
        sizes[op] = static_cast<std::uint32_t>(payload_minimum(static_cast<Op>(op)));
    return sizes;
}();

std::size_t minimum_payload_size(Op op) noexcept {
    return std::size_t(op) < minimum_payload_sizes.size() ? minimum_payload_sizes[std::size_t(op)] : 0;
}

// The ops whose payloads need more than the minimum size.
constexpr auto payload_needs_check = [] {
    std::array<bool, std::size_t(Op::count)> checked{};
    for (const auto op : {Op::set_render_targets, Op::viewports, Op::bind_vertex_attributes,
                          Op::bind_vertex_streams, Op::bind_vertex_buffers, Op::bind_textures,
                          Op::bind_samplers, Op::bind_images, Op::update_uniforms,
                          Op::viewport_swizzles, Op::scissors, Op::depth_ranges,
                          Op::program_subroutines, Op::debug_marker, Op::clear_texture,
                          Op::unsupported, Op::bind_texture_prefix})
        checked[std::size_t(op)] = true;
    return checked;
}();

// Reads one field of a record's payload in place: the validator needs a few
// header integers of structures that run to hundreds of bytes, and copying a
// whole payload to read two of them is the walk's largest cost.
template <typename Type>
Type payload_field(const std::byte* bytes, std::size_t offset) noexcept {
    Type value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

bool valid_payload(Op op, const std::byte* bytes, std::size_t size) noexcept {
    if (op == Op::end || op == Op::jump || op == Op::begin || op == Op::count)
        return false;
    if (size < minimum_payload_size(op))
        return false;
    if (std::size_t(op) >= payload_needs_check.size() || !payload_needs_check[std::size_t(op)])
        return true;
    switch (op) {
    case Op::set_render_targets: {
        const auto count = payload_field<std::int32_t>(bytes, offsetof(SetRenderTargets, count));
        return count >= 0 && count <= 8;
    }
    case Op::viewports:
        return valid_range(payload_field<std::int32_t>(bytes, offsetof(Viewports, first)),
                           payload_field<std::int32_t>(bytes, offsetof(Viewports, count)), 16);
    case Op::bind_vertex_attributes: {
        const auto first = payload_field<std::int32_t>(bytes, offsetof(BindVertexAttributes, first));
        const auto count = payload_field<std::int32_t>(bytes, offsetof(BindVertexAttributes, count));
        return valid_range(first, count, 16) &&
               std::size_t(count) <= (size - offsetof(BindVertexAttributes, states)) /
                                         sizeof(BindVertexAttributes::states[0]);
    }
    case Op::bind_vertex_streams: {
        const auto first = payload_field<std::int32_t>(bytes, offsetof(BindVertexStreams, first));
        const auto count = payload_field<std::int32_t>(bytes, offsetof(BindVertexStreams, count));
        return valid_range(first, count, 16) &&
               std::size_t(count) <= (size - offsetof(BindVertexStreams, states)) /
                                         sizeof(BindVertexStreams::states[0]);
    }
    case Op::bind_vertex_buffers: {
        const auto first = payload_field<std::int32_t>(bytes, offsetof(BindVertexBuffers, first));
        const auto count = payload_field<std::int32_t>(bytes, offsetof(BindVertexBuffers, count));
        return valid_range(first, count, 16) &&
               std::size_t(count) <= (size - offsetof(BindVertexBuffers, ranges)) /
                                         sizeof(BindVertexBuffers::ranges[0]);
    }
    case Op::bind_texture_prefix: {
        const auto stage = payload_field<std::int32_t>(bytes, offsetof(BindTexturePrefix, stage));
        const auto width = payload_field<std::uint32_t>(bytes, offsetof(BindTexturePrefix, bytes));
        return valid_binding_stage(stage) && stage != NVN_SHADER_STAGE_COMPUTE && (width == 4 || width == 8);
    }
    case Op::bind_textures:
    case Op::bind_samplers:
    case Op::bind_images: {
        const auto stage = payload_field<std::int32_t>(bytes, offsetof(BindHandles, stage));
        const auto first = payload_field<std::int32_t>(bytes, offsetof(BindHandles, first));
        const auto count = payload_field<std::int32_t>(bytes, offsetof(BindHandles, count));
        const int limit = op == Op::bind_images ? 32 : op == Op::bind_textures ? 256 : 128;
        return valid_binding_stage(stage) &&
               count <= 32 && valid_range(first, count, limit) &&
               std::size_t(count) <= (size - offsetof(BindHandles, handles)) /
                                         sizeof(BindHandles::handles[0]);
    }
    case Op::update_uniforms:
        return payload_field<std::uint64_t>(bytes, offsetof(UpdateUniforms, size)) <=
               size - offsetof(UpdateUniforms, data);
    case Op::viewport_swizzles:
        return valid_range(payload_field<std::int32_t>(bytes, offsetof(ViewportSwizzles, first)),
                           payload_field<std::int32_t>(bytes, offsetof(ViewportSwizzles, count)), 16);
    case Op::scissors:
        return valid_range(payload_field<std::int32_t>(bytes, offsetof(Scissors, first)),
                           payload_field<std::int32_t>(bytes, offsetof(Scissors, count)), 16);
    case Op::depth_ranges:
        return valid_range(payload_field<std::int32_t>(bytes, offsetof(DepthRanges, first)),
                           payload_field<std::int32_t>(bytes, offsetof(DepthRanges, count)), 16);
    case Op::program_subroutines: {
        const auto signed_count = payload_field<std::int32_t>(bytes, offsetof(ProgramSubroutines, count));
        if (signed_count < 0)
            return false;
        const auto count = std::size_t(signed_count);
        return count <= (size - offsetof(ProgramSubroutines, values)) / sizeof(ProgramSubroutines::values[0]);
    }
    case Op::debug_marker:
        return std::memchr(bytes + offsetof(DebugMarker, text), '\0', size - offsetof(DebugMarker, text)) != nullptr;
    case Op::clear_texture:
        return payload_field<std::uint8_t>(bytes, offsetof(ClearTexture, is_integer)) <= 2;
    case Op::unsupported:
        return payload_field<std::uint32_t>(bytes, offsetof(Unsupported, entry)) <
               static_cast<std::uint32_t>(Entry::count);
    default:
        return true;
    }
}

CommandWalkStatus walk_list(const std::shared_ptr<const CommandList>& list, CommandVisitor visitor, void* context,
                            std::array<NVNcommandHandle, max_call_depth + 1>& stack, unsigned depth,
                            const CommandSubmission* submission = nullptr) noexcept {
    if (depth > max_call_depth)
        return CommandWalkStatus::call_depth_exceeded;
    for (const auto record : *list) {
        if (record.size() < sizeof(RecordHeader))
            return CommandWalkStatus::malformed_record;
        // The guest wrote these records on another core: ask for the lines the
        // next records will read before the walk reaches them.
        __builtin_prefetch(record.data() + 256);
        __builtin_prefetch(record.data() + 384);
        RecordHeader header{};
        std::memcpy(&header, record.data(), sizeof(header));
        if (header.size < sizeof(RecordHeader) || (header.size & (record_alignment - 1)) != 0 ||
            header.size != record.size() || header.op >= static_cast<std::uint32_t>(Op::count))
            return CommandWalkStatus::malformed_record;

        const auto op = static_cast<Op>(header.op);
        const auto payload_size = std::size_t(header.size) - sizeof(RecordHeader);
        const auto* payload = record.data() + sizeof(RecordHeader);
        if (op == Op::begin) {
            if (record.data() != list->blocks.front().bytes || header.size != sizeof(RecordHeader))
                return CommandWalkStatus::malformed_record;
            continue;
        }
        if (op == Op::end)
            return header.size == sizeof(RecordHeader) ? CommandWalkStatus::complete : CommandWalkStatus::malformed_record;
        if (op == Op::jump) {
            const std::byte* target = nullptr;
            if (payload_size != sizeof(target)) return CommandWalkStatus::malformed_record;
            std::memcpy(&target, payload, sizeof(target));
            if (!list->valid_jump(record.data(), target)) return CommandWalkStatus::malformed_record;
            continue;
        }
        if (!valid_payload(op, payload, payload_size))
            return CommandWalkStatus::malformed_record;

        if (op == Op::call) {
            CommandCall call{};
            std::memcpy(&call, payload, sizeof(call));
            if (std::find(stack.begin(), stack.begin() + depth + 1, call.handle) != stack.begin() + depth + 1)
                return CommandWalkStatus::call_cycle;
            if (depth >= max_call_depth)
                return CommandWalkStatus::call_depth_exceeded;
            const auto* retained = submission ? submission->lists.find(call.handle) : nullptr;
            const auto called = submission ? (retained ? retained->commands : nullptr)
                                           : find_handle(call.handle);
            if (!called)
                return CommandWalkStatus::invalid_handle;
            stack[depth + 1] = call.handle;
            const auto status = walk_list(called, visitor, context, stack, depth + 1, submission);
            if (status != CommandWalkStatus::complete)
                return status;
        } else if (op == Op::bind_textures_deferred) {
            if (submission) {
                const auto* found = submission->deferred.find(reinterpret_cast<std::uint64_t>(record.data()));
                if (!found) return CommandWalkStatus::invalid_deferred_binding;
                CommandSubmission::DeferredBinding binding{};
                const auto* words = submission->deferred_words.data() + *found;
                std::memcpy(&binding, words, sizeof(binding));
                for (int consumed = 0; consumed < binding.count;) {
                    BindHandles captured{};
                    captured.stage = binding.stage;
                    captured.first = binding.first + consumed;
                    captured.count = std::min(32, binding.count - consumed);
                    std::memcpy(captured.handles, words + CommandSubmission::deferred_header_words + consumed,
                                std::size_t(captured.count) * sizeof(std::uint64_t));
                    if (!visitor({Op::bind_textures, &captured, sizeof(captured)}, context))
                        return CommandWalkStatus::stopped;
                    consumed += captured.count;
                }
                continue;
            }
            BindTexturesDeferred deferred{};
            std::memcpy(&deferred, payload, sizeof(deferred));
            if (!valid_binding_stage(static_cast<NVNshaderStage>(deferred.stage)) ||
                !valid_deferred_texture_range(deferred.first, deferred.count) || (deferred.count && !deferred.handles))
                return CommandWalkStatus::invalid_deferred_binding;
            for (int consumed = 0; consumed < deferred.count;) {
                BindHandles captured{};
                captured.stage = deferred.stage;
                captured.first = deferred.first + consumed;
                captured.count = std::min(32, deferred.count - consumed);
                std::memcpy(captured.handles, deferred.handles + consumed,
                            std::size_t(captured.count) * sizeof(captured.handles[0]));
                const CommandRecordView view{Op::bind_textures, &captured, sizeof(captured)};
                if (!visitor(view, context))
                    return CommandWalkStatus::stopped;
                consumed += captured.count;
            }
        } else {
            // These descriptions were mutable at recording time. Preserve their
            // submit-time value without copying the command stream.
            if (submission && (op == Op::set_texture_pool || op == Op::set_sampler_pool)) {
                const auto* found = submission->descriptor_pools.find(reinterpret_cast<std::uint64_t>(record.data()));
                const SetDescriptorPool value{found};
                if (!visitor({op, &value, sizeof(value)}, context)) return CommandWalkStatus::stopped;
                continue;
            }
            if (submission && op == Op::shader_scratch_memory) {
                ShaderScratchMemory value{};
                std::memcpy(&value, payload, sizeof(value));
                value.pool = submission->scratch_pools.find(reinterpret_cast<std::uint64_t>(record.data()));
                if (!visitor({op, &value, sizeof(value)}, context)) return CommandWalkStatus::stopped;
                continue;
            }
            const CommandRecordView view{op, payload, payload_size};
            if (!visitor(view, context))
                return CommandWalkStatus::stopped;
        }
    }
    return CommandWalkStatus::complete;
}

// Visit each list once. The cached subtree height also checks a shared subtree
// reached later through a deeper call path; a node still in progress is a cycle.
CommandWalkStatus prepare_list(CommandSubmission& submission, NVNcommandHandle handle, unsigned depth) {
    if (depth > max_call_depth) return CommandWalkStatus::call_depth_exceeded;
    if (const auto* existing = submission.lists.find(handle)) {
        if (!existing->complete) return CommandWalkStatus::call_cycle;
        return depth + existing->height > max_call_depth ? CommandWalkStatus::call_depth_exceeded :
            CommandWalkStatus::complete;
    }
    auto list = find_handle(handle);
    if (!list) return CommandWalkStatus::invalid_handle;
    submission.lists.emplace(handle, CommandSubmission::List{list});
    unsigned height = 0;
    for (const auto* record : list->submit_records) {
        RecordHeader header{};
        std::memcpy(&header, record, sizeof(header));
        const auto op = static_cast<Op>(header.op);
        const auto* bytes = record + sizeof(header);
        if (header.size < sizeof(header) || !valid_payload(op, bytes, header.size - sizeof(header)))
            return CommandWalkStatus::malformed_record;
        switch (op) {
        case Op::call: {
            CommandCall value{};
            std::memcpy(&value, bytes, sizeof(value));
            const auto status = prepare_list(submission, value.handle, depth + 1);
            if (status != CommandWalkStatus::complete) return status;
            height = std::max(height, 1 + submission.lists.find(value.handle)->height);
            break;
        }
        case Op::bind_textures_deferred: {
            BindTexturesDeferred value{};
            std::memcpy(&value, bytes, sizeof(value));
            if (!valid_binding_stage(static_cast<NVNshaderStage>(value.stage)) ||
                !valid_deferred_texture_range(value.first, value.count) || (value.count && !value.handles))
                return CommandWalkStatus::invalid_deferred_binding;
            const auto offset = submission.deferred_words.size();
            submission.deferred.emplace(reinterpret_cast<std::uint64_t>(record), offset);
            submission.deferred_words.resize(offset + CommandSubmission::deferred_header_words + value.count);
            const CommandSubmission::DeferredBinding binding{value.stage, value.first, value.count, 0};
            auto* words = submission.deferred_words.data() + offset;
            std::memcpy(words, &binding, sizeof(binding));
            if (value.count)
                std::memcpy(words + CommandSubmission::deferred_header_words, value.handles,
                            std::size_t(value.count) * sizeof(std::uint64_t));
            break;
        }
        case Op::fence_sync: {
            FenceSync value{};
            std::memcpy(&value, bytes, sizeof(value));
            if (value.sync) submission.syncs.push_back(value.sync);
            break;
        }
        case Op::set_texture_pool:
        case Op::set_sampler_pool: {
            SetDescriptorPool value{};
            std::memcpy(&value, bytes, sizeof(value));
            if (value.pool) submission.descriptor_pools.emplace(reinterpret_cast<std::uint64_t>(record), *value.pool);
            break;
        }
        case Op::shader_scratch_memory: {
            ShaderScratchMemory value{};
            std::memcpy(&value, bytes, sizeof(value));
            if (value.pool) submission.scratch_pools.emplace(reinterpret_cast<std::uint64_t>(record), *value.pool);
            break;
        }
        case Op::wait_event:
            submission.event_waits = true;
            break;
        default: return CommandWalkStatus::malformed_record;
        }
    }
    auto& retained = *submission.lists.find(handle);
    retained.height = height;
    retained.complete = true;
    return CommandWalkStatus::complete;
}

NVNboolean command_initialize(NVNcommandBuffer* command_buffer, NVNdevice* device) {
    if (!command_buffer || !device) {
        diagnostic("nvnCommandBufferInitialize", "null object or device");
        return 0;
    }
    std::memset(command_buffer, 0, sizeof(*command_buffer));
    auto* state = as<CommandBufferState>(command_buffer);
    state->tag = command_buffer_tag;
    state->device = device;
    return 1;
}

void command_finalize(NVNcommandBuffer* command_buffer) {
    auto* state = command_state(command_buffer, "nvnCommandBufferFinalize");
    if (!state)
        return;
    delete state->list;
    state->list = nullptr;
    std::memset(state, 0, sizeof(*state));
}

void command_set_debug_label(NVNcommandBuffer*, const char*) {
    // Labels do not affect command execution and are consumed only by
    // diagnostics.
}

void command_set_memory_callback(NVNcommandBuffer* command_buffer, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC callback) {
    if (auto* state = command_state(command_buffer, "nvnCommandBufferSetMemoryCallback"))
        state->memory_callback = callback;
}

void command_set_memory_callback_data(NVNcommandBuffer* command_buffer, void* data) {
    if (auto* state = command_state(command_buffer, "nvnCommandBufferSetMemoryCallbackData"))
        state->callback_data = data;
}

void command_add_command_memory(NVNcommandBuffer* command_buffer, const NVNmemoryPool* pool, ptrdiff_t offset,
                                std::size_t size) {
    auto* state = command_state(command_buffer, "nvnCommandBufferAddCommandMemory");
    if (!state)
        return;
    // A recording that moves to new memory (the memory callback adds it) took
    // the old segment up to its cursor and continues at the new base.
    if (state->recording && state->list) state->list->close_native_range(state->donated_cursor);
    const auto* memory = as<MemoryPoolState>(pool);
    if (!pool || !memory->memory) {
        // Nothing to account against: records need no game memory.
        state->donated_base = state->donated_end = state->donated_cursor = 0;
        state->callback_exhausted = 0;
        return;
    }
    // The range starts inside the pool and is no larger than it. Its end is
    // not bounded by the pool: the title's driver takes such a range (a game's
    // ring of 512-byte chunks from offset 4 runs 4 bytes past a 1 MiB pool),
    // and nothing is written there; the range only accounts for the game's
    // command-memory position.
    const auto base = reinterpret_cast<std::uintptr_t>(memory->memory) + std::uintptr_t(offset);
    if (offset < 0 || std::uint64_t(offset) >= memory->size || size > memory->size ||
        size > std::numeric_limits<std::uintptr_t>::max() - base) {
        diagnostic("nvnCommandBufferAddCommandMemory", "invalid pool range");
        static std::atomic<int> detailed{0};
        if (detailed.fetch_add(1, std::memory_order_relaxed) < 3)
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: command memory pool=%p memory=%p pool_size=%llu offset=%lld size=%zu\n",
                         static_cast<const void*>(pool), pool ? static_cast<const void*>(memory->memory) : nullptr,
                         pool ? static_cast<unsigned long long>(memory->size) : 0ull,
                         static_cast<long long>(offset), size);
        state->donated_base = state->donated_end = state->donated_cursor = 0;
        return;
    }
    state->donated_base = base;
    state->donated_cursor = base;
    state->donated_end = base + size;
    state->segment_base = memory->memory + offset;
    state->callback_exhausted = 0;
    if (state->recording && state->list) state->list->range_start = state->donated_cursor;
}

void command_add_control_memory(NVNcommandBuffer* command_buffer, void* memory, std::size_t size) {
    if (!command_state(command_buffer, "nvnCommandBufferAddControlMemory"))
        return;
    if (!memory && size) {
        diagnostic("nvnCommandBufferAddControlMemory", "null non-empty control range");
        return;
    }
    auto* state = as<CommandBufferState>(command_buffer);
    state->control_base = static_cast<std::uint8_t*>(memory);
    state->control_cursor = state->control_base;
    state->control_end = state->control_base ? state->control_base + size : nullptr;
    state->control_exhausted = 0;
}

std::size_t command_memory_size(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetCommandMemorySize");
    return state && state->donated_base ? std::size_t(state->donated_end - state->donated_base) : 0;
}

std::size_t command_memory_used(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetCommandMemoryUsed");
    return state && state->donated_base ? std::size_t(state->donated_cursor - state->donated_base) : 0;
}

std::size_t command_memory_free(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetCommandMemoryFree");
    return state && state->donated_base ? std::size_t(state->donated_end - state->donated_cursor) : 0;
}

std::size_t control_memory_size(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetControlMemorySize");
    return state && state->control_base ? std::size_t(state->control_end - state->control_base) : 0;
}

std::size_t control_memory_used(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetControlMemoryUsed");
    return state && state->control_base ? std::size_t(state->control_cursor - state->control_base) : 0;
}

std::size_t control_memory_free(const NVNcommandBuffer* command_buffer) {
    const auto* state = command_state(command_buffer, "nvnCommandBufferGetControlMemoryFree");
    return state && state->control_base ? std::size_t(state->control_end - state->control_cursor) : 0;
}

void command_begin_recording(NVNcommandBuffer* command_buffer) {
    auto* state = command_state(command_buffer, "nvnCommandBufferBeginRecording");
    if (!state)
        return;
    if (state->recording) {
        diagnostic("nvnCommandBufferBeginRecording", "recording is already active");
        return;
    }
    delete state->list;
    state->list = new CommandList;
    state->list->device = state->device;
    state->list->native_start = state->donated_cursor;
    state->list->range_start = state->donated_cursor;
    write_header(state->list->allocate(sizeof(RecordHeader)), Op::begin, sizeof(RecordHeader));
    state->recording = 1;
    state->recording_failed = 0;
}

NVNcommandHandle command_end_recording(NVNcommandBuffer* command_buffer) {
    auto* state = command_state(command_buffer, "nvnCommandBufferEndRecording");
    if (!state || !state->recording || !state->list) {
        diagnostic("nvnCommandBufferEndRecording", "recording is not active");
        return 0;
    }
    auto* recorded = state->list;
    state->list = nullptr;
    state->recording = 0;
    if (state->recording_failed) {
        delete recorded;
        diagnostic("nvnCommandBufferEndRecording", "recording failed");
        return 0;
    }
    recorded->finish();
    recorded->close_native_range(state->donated_cursor);
    std::shared_ptr<const CommandList> list(recorded);
    // The handle is the native start position: a later recording at the same
    // position replaces the list, and handles the game cached run the new
    // commands, as they would natively. Recordings without game command memory
    // take a counter handle.
    NVNcommandHandle handle = static_cast<NVNcommandHandle>(recorded->native_start);
    if (!handle) {
        handle = next_handle.fetch_add(1, std::memory_order_relaxed);
        if (!handle) handle = next_handle.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard lock(registry_mutex);
    register_handle(handle, std::move(list));
    return handle;
}

void command_call_commands(NVNcommandBuffer* command_buffer, int count, const NVNcommandHandle* handles) {
    if (count < 0 || (count && !handles)) {
        diagnostic("nvnCommandBufferCallCommands", "invalid handle array");
        return;
    }
    for (int i = 0; i < count; ++i) {
        if (auto* call = append<CommandCall>(command_buffer, Op::call, "nvnCommandBufferCallCommands")) {
            call->handle = handles[i];
            if (auto* state = command_state(command_buffer, "nvnCommandBufferCallCommands"))
                request_control_memory(command_buffer, *state);
        }
    }
}

void command_copy_commands(NVNcommandBuffer* command_buffer, int count, const NVNcommandHandle* handles) {
    if (count < 0 || (count && !handles)) {
        diagnostic("nvnCommandBufferCopyCommands", "invalid handle array");
        return;
    }
    for (int index = 0; index < count; ++index) {
        const auto source = find_handle(handles[index]);
        if (!source) {
            diagnostic("nvnCommandBufferCopyCommands", "invalid source handle");
            fail_recording(command_buffer);
            return;
        }
        if (auto* state = command_state(command_buffer, "nvnCommandBufferCopyCommands")) {
            if (auto* destination = recording_list(state, "nvnCommandBufferCopyCommands"))
                destination->copied_lists.push_back(source);
        }
        for (const auto record : *source) {
            RecordHeader header{};
            if (record.size() < sizeof(header)) {
                diagnostic("nvnCommandBufferCopyCommands", "malformed source record");
                fail_recording(command_buffer);
                return;
            }
            std::memcpy(&header, record.data(), sizeof(header));
            if (header.size < sizeof(header) || (header.size & 7u) || header.size != record.size() ||
                header.op >= static_cast<std::uint32_t>(Op::count)) {
                diagnostic("nvnCommandBufferCopyCommands", "malformed source record");
                fail_recording(command_buffer);
                return;
            }
            const auto op = static_cast<Op>(header.op);
            if (op == Op::begin) continue;
            if (op == Op::end) break;
            if (op == Op::jump) {
                const std::byte* target = nullptr;
                if (header.size == sizeof(header) + sizeof(target))
                    std::memcpy(&target, record.data() + sizeof(header), sizeof(target));
                if (!source->valid_jump(record.data(), target)) {
                    diagnostic("nvnCommandBufferCopyCommands", "invalid command segment jump");
                    fail_recording(command_buffer);
                    return;
                }
                continue;
            }
            const auto payload_size = std::size_t(header.size) - sizeof(header);
            void* destination =
                append_record(command_buffer, static_cast<Op>(header.op), payload_size, "nvnCommandBufferCopyCommands");
            if (!destination)
                break;
            std::memcpy(destination, record.data() + sizeof(header), payload_size);
            if (static_cast<Op>(header.op) == Op::call) {
                if (auto* state = command_state(command_buffer, "nvnCommandBufferCopyCommands"))
                    request_control_memory(command_buffer, *state);
            }
        }
        // Copied draws keep the source's stamps: the next draw of this buffer
        // stamps its own count again.
        if (auto* state = command_state(command_buffer, "nvnCommandBufferCopyCommands"))
            if (auto* destination = recording_list(state, "nvnCommandBufferCopyCommands"))
                destination->stamped_registrations = 0;
    }
}

void device_finalize_command_handle(NVNdevice* device, NVNcommandHandle handle) {
    std::lock_guard lock(registry_mutex);
    const auto* found = command_handles.find(handle);
    if (!found) {
        diagnostic("nvnDeviceFinalizeCommandHandle", "invalid command handle");
        return;
    }
    if ((*found)->device != device) {
        diagnostic("nvnDeviceFinalizeCommandHandle", "handle belongs to another device");
        return;
    }
    unregister_handle(handle);
}

void bind_state(NVNcommandBuffer* command_buffer, int kind, const void* state, std::size_t size,
                const char* operation) {
    if (!state || size > sizeof(BindState::bytes)) {
        diagnostic(operation, "invalid state object");
        return;
    }
    auto* payload = append<BindState>(command_buffer, Op::bind_state, operation);
    if (!payload)
        return;
    *payload = {};
    payload->kind = kind;
    std::memcpy(payload->bytes, state, size);
}

void command_bind_blend_state(NVNcommandBuffer* c, const NVNblendState* s) {
    bind_state(c, 0, s, sizeof(BlendStateBits), "nvnCommandBufferBindBlendState");
}
void command_bind_channel_mask_state(NVNcommandBuffer* c, const NVNchannelMaskState* s) {
    bind_state(c, 2, s, sizeof(ChannelMaskStateBits), "nvnCommandBufferBindChannelMaskState");
}
void command_bind_color_state(NVNcommandBuffer* c, const NVNcolorState* s) {
    bind_state(c, 1, s, sizeof(ColorStateBits), "nvnCommandBufferBindColorState");
}
void command_bind_multisample_state(NVNcommandBuffer* c, const NVNmultisampleState* s) {
    bind_state(c, 3, s, sizeof(MultisampleStateBits), "nvnCommandBufferBindMultisampleState");
}
void command_bind_polygon_state(NVNcommandBuffer* c, const NVNpolygonState* s) {
    bind_state(c, 4, s, sizeof(PolygonStateBits), "nvnCommandBufferBindPolygonState");
}
void command_bind_depth_stencil_state(NVNcommandBuffer* c, const NVNdepthStencilState* s) {
    bind_state(c, 5, s, sizeof(DepthStencilStateBits), "nvnCommandBufferBindDepthStencilState");
}

void command_bind_vertex_attributes(NVNcommandBuffer* c, int count, const NVNvertexAttribState* states) {
    if (count < 0 || count > 16 || (count && !states)) {
        diagnostic("nvnCommandBufferBindVertexAttribState", "invalid state array");
        return;
    }
    auto* payload =
        static_cast<BindVertexAttributes*>(append_record(c, Op::bind_vertex_attributes,
            offsetof(BindVertexAttributes, states) + std::size_t(count) * sizeof(BindVertexAttributes::states[0]),
            "nvnCommandBufferBindVertexAttribState"));
    if (!payload)
        return;
    payload->first = 0;
    payload->count = count;
    if (count)
        std::memcpy(payload->states, states, std::size_t(count) * sizeof(payload->states[0]));
}

void command_bind_vertex_streams(NVNcommandBuffer* c, int count, const NVNvertexStreamState* states) {
    if (count < 0 || count > 16 || (count && !states)) {
        diagnostic("nvnCommandBufferBindVertexStreamState", "invalid state array");
        return;
    }
    auto* payload = static_cast<BindVertexStreams*>(append_record(c, Op::bind_vertex_streams,
            offsetof(BindVertexStreams, states) + std::size_t(count) * sizeof(BindVertexStreams::states[0]),
            "nvnCommandBufferBindVertexStreamState"));
    if (!payload)
        return;
    payload->first = 0;
    payload->count = count;
    if (count)
        std::memcpy(payload->states, states, std::size_t(count) * sizeof(payload->states[0]));
}

void command_bind_program(NVNcommandBuffer* c, const NVNprogram* program, int stages) {
    auto* payload = append<BindProgram>(c, Op::bind_program, "nvnCommandBufferBindProgram");
    if (payload)
        *payload = {program_snapshot(c, program, "nvnCommandBufferBindProgram"), stages};
}

void command_bind_vertex_buffers(NVNcommandBuffer* c, int first, int count, const NVNbufferRange* ranges) {
    if (!valid_range(first, count, 16) || (count && !ranges)) {
        diagnostic("nvnCommandBufferBindVertexBuffers", "invalid range");
        return;
    }
    auto* payload = static_cast<BindVertexBuffers*>(append_record(c, Op::bind_vertex_buffers,
            offsetof(BindVertexBuffers, ranges) + std::size_t(count) * sizeof(BindVertexBuffers::ranges[0]),
            "nvnCommandBufferBindVertexBuffers"));
    if (!payload)
        return;
    payload->first = first;
    payload->count = count;
    if (count)
        std::memcpy(payload->ranges, ranges, std::size_t(count) * sizeof(*ranges));
}

void command_bind_vertex_buffer(NVNcommandBuffer* c, int index, NVNbufferAddress address, std::size_t size) {
    const NVNbufferRange range{address, size};
    command_bind_vertex_buffers(c, index, 1, &range);
}

void command_bind_uniform_buffer(NVNcommandBuffer* c, NVNshaderStage stage, int index, NVNbufferAddress address,
                                 std::size_t size) {
    const NVNbufferRange range{address, size};
    bind_buffer_ranges(c, Op::bind_uniform_buffer, stage, index, 1, &range,
                       Entry::nvnCommandBufferBindUniformBuffer, "nvnCommandBufferBindUniformBuffer");
}
void command_bind_uniform_buffers(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                                  const NVNbufferRange* ranges) {
    bind_buffer_ranges(c, Op::bind_uniform_buffer, stage, first, count, ranges,
                       Entry::nvnCommandBufferBindUniformBuffers, "nvnCommandBufferBindUniformBuffers");
}
void command_bind_storage_buffer(NVNcommandBuffer* c, NVNshaderStage stage, int index, NVNbufferAddress address,
                                 std::size_t size) {
    const NVNbufferRange range{address, size};
    bind_buffer_ranges(c, Op::bind_storage_buffer, stage, index, 1, &range,
                       Entry::nvnCommandBufferBindStorageBuffer, "nvnCommandBufferBindStorageBuffer");
}
void command_bind_storage_buffers(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                                  const NVNbufferRange* ranges) {
    bind_buffer_ranges(c, Op::bind_storage_buffer, stage, first, count, ranges,
                       Entry::nvnCommandBufferBindStorageBuffers, "nvnCommandBufferBindStorageBuffers");
}

void command_bind_transform_feedback_buffer(NVNcommandBuffer* c, int index, NVNbufferAddress address,
                                            std::size_t size) {
    auto* payload = append<BindTransformFeedbackBuffer>(c, Op::bind_transform_feedback_buffer,
                                                        "nvnCommandBufferBindTransformFeedbackBuffer");
    if (payload)
        *payload = {index, address, size};
}
void command_bind_transform_feedback_buffers(NVNcommandBuffer* c, int first, int count, const NVNbufferRange* ranges) {
    if (first < 0 || count < 0 || (count && !ranges)) {
        diagnostic("nvnCommandBufferBindTransformFeedbackBuffers", "invalid range");
        return;
    }
    for (int i = 0; i < count; ++i)
        command_bind_transform_feedback_buffer(c, first + i, ranges[i].address, ranges[i].size);
}

void command_bind_texture(NVNcommandBuffer* c, NVNshaderStage stage, int index, NVNtextureHandle handle) {
    bind_handles(c, Op::bind_textures, stage, index, 1, &handle, 128, Entry::nvnCommandBufferBindTexture,
                 "nvnCommandBufferBindTexture");
}
void command_bind_textures(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                           const NVNtextureHandle* handles) {
    bind_handles(c, Op::bind_textures, stage, first, count, handles, 128, Entry::nvnCommandBufferBindTextures,
                 "nvnCommandBufferBindTextures");
}
void command_bind_image(NVNcommandBuffer* c, NVNshaderStage stage, int index, NVNimageHandle handle) {
    bind_handles(c, Op::bind_images, stage, index, 1, &handle, 32, Entry::nvnCommandBufferBindImage,
                 "nvnCommandBufferBindImage");
}
void command_bind_images(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                         const NVNimageHandle* handles) {
    bind_handles(c, Op::bind_images, stage, first, count, handles, 32, Entry::nvnCommandBufferBindImages,
                 "nvnCommandBufferBindImages");
}

void command_bind_separate_texture(NVNcommandBuffer* c, NVNshaderStage stage, int index,
                                   NVNseparateTextureHandle handle) {
    bind_handles(c, Op::bind_textures, stage, index, 1, &handle, 128, Entry::nvnCommandBufferBindSeparateTexture,
                 "nvnCommandBufferBindSeparateTexture", 128);
}
void command_bind_separate_sampler(NVNcommandBuffer* c, NVNshaderStage stage, int index,
                                   NVNseparateSamplerHandle handle) {
    bind_handles(c, Op::bind_samplers, stage, index, 1, &handle, 128, Entry::nvnCommandBufferBindSeparateSampler,
                 "nvnCommandBufferBindSeparateSampler");
}
void command_bind_separate_textures(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                                    const NVNseparateTextureHandle* handles) {
    bind_handles(c, Op::bind_textures, stage, first, count, handles, 128, Entry::nvnCommandBufferBindSeparateTextures,
                 "nvnCommandBufferBindSeparateTextures", 128);
}
void command_bind_separate_samplers(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                                    const NVNseparateSamplerHandle* handles) {
    bind_handles(c, Op::bind_samplers, stage, first, count, handles, 128, Entry::nvnCommandBufferBindSeparateSamplers,
                 "nvnCommandBufferBindSeparateSamplers");
}

void command_bind_textures_deferred(NVNcommandBuffer* c, NVNshaderStage stage, int first, int count,
                                    const NVNtextureHandle* handles) {
    auto* payload = append<BindTexturesDeferred>(c, Op::bind_textures_deferred, "nvnCommandBufferBindTexturesDeferred");
    if (payload)
        *payload = {stage, first, count, 0, handles};
}

void command_set_patch_size(NVNcommandBuffer* c, int size) {
    if (auto* p = append<PatchSize>(c, Op::patch_size, "nvnCommandBufferSetPatchSize"))
        p->size = size;
}
void command_set_inner_tessellation(NVNcommandBuffer* c, const float* values) {
    if (!values) {
        diagnostic("nvnCommandBufferSetInnerTessellationLevels", "null level array");
        return;
    }
    auto* p = append<TessellationLevels>(c, Op::tessellation_levels, "nvnCommandBufferSetInnerTessellationLevels");
    if (p) {
        p->outer = 0;
        std::copy_n(values, 2, p->values);
    }
}
void command_set_outer_tessellation(NVNcommandBuffer* c, const float* values) {
    if (!values) {
        diagnostic("nvnCommandBufferSetOuterTessellationLevels", "null level array");
        return;
    }
    auto* p = append<TessellationLevels>(c, Op::tessellation_levels, "nvnCommandBufferSetOuterTessellationLevels");
    if (p) {
        p->outer = 1;
        std::copy_n(values, 4, p->values);
    }
}
void command_set_primitive_restart(NVNcommandBuffer* c, std::uint32_t enable, int index) {
    if (auto* p = append<PrimitiveRestart>(c, Op::primitive_restart, "nvnCommandBufferSetPrimitiveRestart"))
        *p = {static_cast<std::uint32_t>(index), guest_bool(enable)};
}

void command_transform_feedback(NVNcommandBuffer* c, int action, int argument, NVNbufferAddress address,
                                const char* operation) {
    if (auto* p = append<TransformFeedback>(c, Op::transform_feedback, operation))
        *p = {action, argument, address};
}
void command_begin_transform_feedback(NVNcommandBuffer* c, NVNbufferAddress a) {
    command_transform_feedback(c, 0, 0, a, "nvnCommandBufferBeginTransformFeedback");
}
void command_end_transform_feedback(NVNcommandBuffer* c, NVNbufferAddress a) {
    command_transform_feedback(c, 1, 0, a, "nvnCommandBufferEndTransformFeedback");
}
void command_pause_transform_feedback(NVNcommandBuffer* c, NVNbufferAddress a) {
    command_transform_feedback(c, 2, 0, a, "nvnCommandBufferPauseTransformFeedback");
}
void command_resume_transform_feedback(NVNcommandBuffer* c, NVNbufferAddress a) {
    command_transform_feedback(c, 3, 0, a, "nvnCommandBufferResumeTransformFeedback");
}
void command_draw_transform_feedback(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNbufferAddress a) {
    command_transform_feedback(c, 4, primitive, a, "nvnCommandBufferDrawTransformFeedback");
}

void command_draw_arrays(NVNcommandBuffer* c, NVNdrawPrimitive primitive, int first, int count) {
    if (auto* p = append<Draw>(c, Op::draw, "nvnCommandBufferDrawArrays"))
        *p = {primitive, first, count, 1, 0};
}
void command_draw_arrays_instanced(NVNcommandBuffer* c, NVNdrawPrimitive primitive, int first, int count,
                                   int base_instance, int instances) {
    if (auto* p = append<Draw>(c, Op::draw, "nvnCommandBufferDrawArraysInstanced"))
        *p = {primitive, first, count, instances, base_instance};
}
void command_draw_elements_instanced(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNindexType type, int count,
                                     NVNbufferAddress indices, int base_vertex, int base_instance, int instances) {
    if (auto* p = append<DrawIndexed>(c, Op::draw_indexed, "nvnCommandBufferDrawElementsInstanced"))
        *p = {primitive, type, count, base_vertex, base_instance, instances, indices};
}
void command_draw_elements(NVNcommandBuffer* c, NVNdrawPrimitive p, NVNindexType t, int n, NVNbufferAddress a) {
    command_draw_elements_instanced(c, p, t, n, a, 0, 0, 1);
}
void command_draw_elements_base_vertex(NVNcommandBuffer* c, NVNdrawPrimitive p, NVNindexType t, int n,
                                       NVNbufferAddress a, int base) {
    command_draw_elements_instanced(c, p, t, n, a, base, 0, 1);
}

void command_draw_arrays_indirect(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNbufferAddress indirect) {
    if (auto* p = append<DrawIndirect>(c, Op::draw_indirect, "nvnCommandBufferDrawArraysIndirect"))
        *p = {primitive, indirect};
}
void command_draw_elements_indirect(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNindexType type,
                                    NVNbufferAddress indices, NVNbufferAddress indirect) {
    if (auto* p = append<DrawIndexedIndirect>(c, Op::draw_indexed_indirect, "nvnCommandBufferDrawElementsIndirect"))
        *p = {primitive, type, indices, indirect};
}
void command_multi_draw_arrays_indirect(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNbufferAddress indirect,
                                        NVNbufferAddress parameter, int max_count, ptrdiff_t stride) {
    if (auto* p = append<MultiDrawIndirect>(c, Op::multi_draw_indirect, "nvnCommandBufferMultiDrawArraysIndirectCount"))
        *p = {primitive, max_count, indirect, parameter, stride};
}
void command_multi_draw_elements_indirect(NVNcommandBuffer* c, NVNdrawPrimitive primitive, NVNindexType type,
                                          NVNbufferAddress indices, NVNbufferAddress indirect,
                                          NVNbufferAddress parameter, int max_count, ptrdiff_t stride) {
    if (auto* p = append<MultiDrawIndexedIndirect>(c, Op::multi_draw_indexed_indirect,
                                                   "nvnCommandBufferMultiDrawElementsIndirectCount"))
        *p = {primitive, type, max_count, indices, indirect, parameter, stride};
}

void command_clear_color(NVNcommandBuffer* c, int index, const float* rgba, int mask) {
    if (!rgba) {
        diagnostic("nvnCommandBufferClearColor", "null color");
        return;
    }
    auto* p = append<ClearColor>(c, Op::clear_color, "nvnCommandBufferClearColor");
    if (p) {
        p->index = index;
        p->mask = mask;
        std::copy_n(rgba, 4, p->rgba);
    }
}
template <typename Value>
void command_clear_color_integer(NVNcommandBuffer* c, int index, const Value* rgba, int mask, bool is_unsigned,
                                 const char* operation) {
    if (!rgba) {
        diagnostic(operation, "null color");
        return;
    }
    auto* p = append<ClearColorInteger>(c, Op::clear_color_integer, operation);
    if (p) {
        p->index = index;
        p->mask = mask;
        p->is_unsigned = is_unsigned;
        std::memcpy(p->rgba, rgba, sizeof(p->rgba));
    }
}
void command_clear_colori(NVNcommandBuffer* c, int i, const int* rgba, int mask) {
    command_clear_color_integer(c, i, rgba, mask, false, "nvnCommandBufferClearColori");
}
void command_clear_colorui(NVNcommandBuffer* c, int i, const std::uint32_t* rgba, int mask) {
    command_clear_color_integer(c, i, rgba, mask, true, "nvnCommandBufferClearColorui");
}
void command_clear_depth_stencil(NVNcommandBuffer* c, float depth, std::uint32_t depth_mask, int stencil,
                                 int stencil_mask) {
    if (auto* p = append<ClearDepthStencil>(c, Op::clear_depth_stencil, "nvnCommandBufferClearDepthStencil"))
        *p = {depth, guest_bool(depth_mask), stencil, stencil_mask};
}

void command_dispatch(NVNcommandBuffer* c, int x, int y, int z) {
    // The game calls this entry directly, so the return address is its call site.
    if (auto* p = append<Dispatch>(c, Op::dispatch, "nvnCommandBufferDispatchCompute"))
        *p = {x, y, z, reinterpret_cast<std::uint64_t>(__builtin_return_address(0))};
}
void command_dispatch_indirect(NVNcommandBuffer* c, NVNbufferAddress address) {
    if (auto* p = append<DispatchIndirect>(c, Op::dispatch_indirect, "nvnCommandBufferDispatchComputeIndirect"))
        p->address = address;
}

void command_set_viewports(NVNcommandBuffer* c, int first, int count, const float* values) {
    if (!valid_range(first, count, 16) || (count && !values)) {
        diagnostic("nvnCommandBufferSetViewports", "invalid viewport array");
        return;
    }
    auto* p = append<Viewports>(c, Op::viewports, "nvnCommandBufferSetViewports");
    if (!p)
        return;
    p->first = first;
    p->count = count;
    for (int i = 0; i < count; ++i)
        std::copy_n(values + i * 4, 4, p->values[i]);
}
void command_set_viewport(NVNcommandBuffer* c, int x, int y, int width, int height) {
    const float values[]{float(x), float(y), float(width), float(height)};
    command_set_viewports(c, 0, 1, values);
}
void command_set_viewport_swizzles(NVNcommandBuffer* c, int first, int count, const NVNviewportSwizzle* values) {
    if (!valid_range(first, count, 16) || (count && !values)) {
        diagnostic("nvnCommandBufferSetViewportSwizzles", "invalid swizzle array");
        return;
    }
    auto* p = append<ViewportSwizzles>(c, Op::viewport_swizzles, "nvnCommandBufferSetViewportSwizzles");
    if (!p)
        return;
    p->first = first;
    p->count = count;
    for (int i = 0; i < count; ++i)
        for (int component = 0; component < 4; ++component)
            p->values[i][component] = values[i * 4 + component];
}
void command_set_scissor(NVNcommandBuffer* c, int x, int y, int width, int height) {
    if (auto* p = append<Scissor>(c, Op::scissor, "nvnCommandBufferSetScissor"))
        *p = {x, y, width, height};
}
void command_set_scissors(NVNcommandBuffer* c, int first, int count, const int* values) {
    if (!valid_range(first, count, 16) || (count && !values)) {
        diagnostic("nvnCommandBufferSetScissors", "invalid scissor array");
        return;
    }
    auto* p = append<Scissors>(c, Op::scissors, "nvnCommandBufferSetScissors");
    if (!p)
        return;
    p->first = first;
    p->count = count;
    if (count)
        std::memcpy(p->values, values, std::size_t(count) * sizeof(p->values[0]));
}
void command_set_depth_range(NVNcommandBuffer* c, float near_value, float far_value) {
    if (auto* p = append<DepthRange>(c, Op::depth_range, "nvnCommandBufferSetDepthRange"))
        *p = {near_value, far_value};
}
void command_set_depth_bounds(NVNcommandBuffer* c, std::uint32_t enable, float near_value, float far_value) {
    if (auto* p = append<DepthBounds>(c, Op::depth_bounds, "nvnCommandBufferSetDepthBounds"))
        *p = {guest_bool(enable), near_value, far_value};
}
void command_set_depth_ranges(NVNcommandBuffer* c, int first, int count, const float* values) {
    if (!valid_range(first, count, 16) || (count && !values)) {
        diagnostic("nvnCommandBufferSetDepthRanges", "invalid depth-range array");
        return;
    }
    auto* p = append<DepthRanges>(c, Op::depth_ranges, "nvnCommandBufferSetDepthRanges");
    if (!p)
        return;
    p->first = first;
    p->count = count;
    if (count)
        std::memcpy(p->values, values, std::size_t(count) * sizeof(p->values[0]));
}

void command_set_tiled_cache_action(NVNcommandBuffer*, NVNtiledCacheAction action) {
    if (action < NVN_TILED_CACHE_ACTION_ENABLE || action > NVN_TILED_CACHE_ACTION_DISABLE_RENDER_TARGET_BINNING)
        diagnostic("nvnCommandBufferSetTiledCacheAction", "invalid tiled-cache action");
    // Tiled-cache controls are hardware scheduling hints with no replay-visible
    // state.
}
void command_set_tiled_cache_tile_size(NVNcommandBuffer*, int width, int height) {
    if (width <= 0 || height <= 0)
        diagnostic("nvnCommandBufferSetTiledCacheTileSize", "invalid tile size");
    // Tile size is a hardware scheduling hint with no replay-visible state.
}

void command_set_stencil(NVNcommandBuffer* c, NVNface face, int value, int which, const char* operation) {
    auto* p = append<Stencil>(c, Op::stencil, operation);
    if (!p)
        return;
    *p = {face, which == 0 ? value : 0, which == 1 ? value : 0, which == 2 ? value : 0, which};
}
void command_set_stencil_value_mask(NVNcommandBuffer* c, NVNface f, int v) {
    command_set_stencil(c, f, v, 0, "nvnCommandBufferSetStencilValueMask");
}
void command_set_stencil_mask(NVNcommandBuffer* c, NVNface f, int v) {
    command_set_stencil(c, f, v, 1, "nvnCommandBufferSetStencilMask");
}
void command_set_stencil_ref(NVNcommandBuffer* c, NVNface f, int v) {
    command_set_stencil(c, f, v, 2, "nvnCommandBufferSetStencilRef");
}
void command_set_blend_color(NVNcommandBuffer* c, const float* values) {
    if (!values) {
        diagnostic("nvnCommandBufferSetBlendColor", "null color");
        return;
    }
    if (auto* p = append<BlendColor>(c, Op::blend_color, "nvnCommandBufferSetBlendColor"))
        std::copy_n(values, 4, p->rgba);
}
void command_set_point_size(NVNcommandBuffer* c, float value) {
    if (auto* p = append<PointSize>(c, Op::point_size, "nvnCommandBufferSetPointSize"))
        p->size = value;
}
void command_set_line_width(NVNcommandBuffer* c, float value) {
    if (auto* p = append<LineWidth>(c, Op::line_width, "nvnCommandBufferSetLineWidth"))
        p->width = value;
}
void command_set_polygon_offset(NVNcommandBuffer* c, float factor, float units, float clamp) {
    if (auto* p = append<PolygonOffset>(c, Op::polygon_offset, "nvnCommandBufferSetPolygonOffsetClamp"))
        *p = {factor, units, clamp};
}
void command_set_alpha_ref(NVNcommandBuffer* c, float value) {
    if (auto* p = append<ScalarFloat>(c, Op::alpha_ref, "nvnCommandBufferSetAlphaRef"))
        p->value = value;
}
void command_set_sample_mask(NVNcommandBuffer* c, int value) {
    if (auto* p = append<ScalarInt>(c, Op::sample_mask, "nvnCommandBufferSetSampleMask"))
        p->value = value;
}
void command_set_rasterizer_discard(NVNcommandBuffer* c, std::uint32_t value) {
    if (auto* p = append<ScalarInt>(c, Op::rasterizer_discard, "nvnCommandBufferSetRasterizerDiscard"))
        p->value = guest_bool(value);
}
void command_set_depth_clamp(NVNcommandBuffer* c, std::uint32_t value) {
    if (auto* p = append<DepthClamp>(c, Op::depth_clamp, "nvnCommandBufferSetDepthClamp"))
        p->enable = guest_bool(value);
}
void command_set_conservative_raster(NVNcommandBuffer* c, std::uint32_t value) {
    if (auto* p = append<ConservativeRaster>(c, Op::conservative_raster, "nvnCommandBufferSetConservativeRasterEnable"))
        p->enable = guest_bool(value);
}
void command_set_conservative_dilate(NVNcommandBuffer* c, float value) {
    if (auto* p = append<ScalarFloat>(c, Op::conservative_raster_dilate, "nvnCommandBufferSetConservativeRasterDilate"))
        p->value = value;
}
void command_set_subpixel_bias(NVNcommandBuffer* c, int x, int y) {
    if (auto* p =
            append<SubpixelPrecisionBias>(c, Op::subpixel_precision_bias, "nvnCommandBufferSetSubpixelPrecisionBias"))
        *p = {x, y};
}

void command_copy_buffer_to_texture(NVNcommandBuffer* c, NVNbufferAddress source, const NVNtexture* texture,
                                    const NVNtextureView* view, const NVNcopyRegion* region, int flags) {
    if (!region) {
        diagnostic("nvnCommandBufferCopyBufferToTexture", "null region");
        return;
    }
    auto* p = append<CopyBufferToTexture>(c, Op::copy_buffer_to_texture, "nvnCommandBufferCopyBufferToTexture");
    if (!p)
        return;
    p->source = source;
    p->texture = texture_snapshot(c, texture, "nvnCommandBufferCopyBufferToTexture");
    copy_view(p->view, p->has_view, view);
    p->region = *region;
    p->flags = flags;
}
void command_copy_texture_to_buffer(NVNcommandBuffer* c, const NVNtexture* texture, const NVNtextureView* view,
                                    const NVNcopyRegion* region, NVNbufferAddress destination, int flags) {
    if (!region) {
        diagnostic("nvnCommandBufferCopyTextureToBuffer", "null region");
        return;
    }
    auto* p = append<CopyBufferToTexture>(c, Op::copy_texture_to_buffer, "nvnCommandBufferCopyTextureToBuffer");
    if (!p)
        return;
    p->source = destination;
    p->texture = texture_snapshot(c, texture, "nvnCommandBufferCopyTextureToBuffer");
    copy_view(p->view, p->has_view, view);
    p->region = *region;
    p->flags = flags;
}
void command_copy_texture_to_texture(NVNcommandBuffer* c, const NVNtexture* source, const NVNtextureView* source_view,
                                     const NVNcopyRegion* source_region, const NVNtexture* destination,
                                     const NVNtextureView* destination_view, const NVNcopyRegion* destination_region,
                                     int flags) {
    if (!source_region || !destination_region) {
        diagnostic("nvnCommandBufferCopyTextureToTexture", "null region");
        return;
    }
    auto* p = append<CopyTextureToTexture>(c, Op::copy_texture_to_texture, "nvnCommandBufferCopyTextureToTexture");
    if (!p)
        return;
    p->source = texture_snapshot(c, source, "nvnCommandBufferCopyTextureToTexture");
    p->destination = texture_snapshot(c, destination, "nvnCommandBufferCopyTextureToTexture");
    copy_view(p->source_view, p->has_source_view, source_view);
    copy_view(p->destination_view, p->has_destination_view, destination_view);
    p->source_region = *source_region;
    p->destination_region = *destination_region;
    p->flags = flags;
}
void command_copy_buffer(NVNcommandBuffer* c, NVNbufferAddress source, NVNbufferAddress destination, std::size_t size,
                         int flags) {
    if (auto* p = append<CopyBuffer>(c, Op::copy_buffer, "nvnCommandBufferCopyBufferToBuffer"))
        *p = {source, destination, size, flags};
}
void command_clear_buffer(NVNcommandBuffer* c, NVNbufferAddress address, std::size_t size, std::uint32_t value) {
    if (auto* p = append<ClearBuffer>(c, Op::clear_buffer, "nvnCommandBufferClearBuffer"))
        *p = {address, size, value};
}

template <typename Value>
void command_clear_texture_value(NVNcommandBuffer* c, const NVNtexture* texture, const NVNtextureView* view,
                                 const NVNcopyRegion* region, const Value* color, int mask, std::uint8_t kind,
                                 const char* operation) {
    if (!region || !color) {
        diagnostic(operation, "null region or color");
        return;
    }
    auto* p = append<ClearTexture>(c, Op::clear_texture, operation);
    if (!p)
        return;
    p->texture = texture_snapshot(c, texture, operation);
    copy_view(p->view, p->has_view, view);
    p->is_integer = kind;
    p->region = *region;
    p->mask = mask;
    std::memcpy(p->color.u, color, sizeof(p->color));
}
void command_clear_texture(NVNcommandBuffer* c, const NVNtexture* t, const NVNtextureView* v, const NVNcopyRegion* r,
                           const float* color, int mask) {
    command_clear_texture_value(c, t, v, r, color, mask, 0, "nvnCommandBufferClearTexture");
}
void command_clear_texturei(NVNcommandBuffer* c, const NVNtexture* t, const NVNtextureView* v, const NVNcopyRegion* r,
                            const int* color, int mask) {
    command_clear_texture_value(c, t, v, r, color, mask, 2, "nvnCommandBufferClearTexturei");
}
void command_clear_textureui(NVNcommandBuffer* c, const NVNtexture* t, const NVNtextureView* v, const NVNcopyRegion* r,
                             const std::uint32_t* color, int mask) {
    command_clear_texture_value(c, t, v, r, color, mask, 1, "nvnCommandBufferClearTextureui");
}

void command_update_uniforms(NVNcommandBuffer* c, NVNbufferAddress buffer, std::size_t buffer_size, ptrdiff_t offset,
                             std::size_t size, const void* data) {
    if (offset < 0 || std::size_t(offset) > buffer_size || size > buffer_size - std::size_t(offset) ||
        size > UINT32_MAX - offsetof(UpdateUniforms, data) - sizeof(RecordHeader) - record_alignment ||
        (size && !data) || buffer > std::numeric_limits<std::uint64_t>::max() - std::size_t(offset)) {
        append_unsupported(c, Entry::nvnCommandBufferUpdateUniformBuffer, "nvnCommandBufferUpdateUniformBuffer");
        return;
    }
    void* raw = append_record(c, Op::update_uniforms, offsetof(UpdateUniforms, data) + size,
                              "nvnCommandBufferUpdateUniformBuffer");
    if (!raw)
        return;
    auto* p = static_cast<UpdateUniforms*>(raw);
    p->destination = buffer + std::size_t(offset);
    p->size = size;
    if (size)
        std::memcpy(p->data, data, size);
}
void command_report_counter(NVNcommandBuffer* c, NVNcounterType type, NVNbufferAddress address) {
    if (auto* p = append<ReportCounter>(c, Op::report_counter, "nvnCommandBufferReportCounter"))
        *p = {type, address};
}
void command_reset_counter(NVNcommandBuffer* c, NVNcounterType type) {
    if (auto* p = append<ResetCounter>(c, Op::reset_counter, "nvnCommandBufferResetCounter"))
        p->type = type;
}
void command_report_value(NVNcommandBuffer* c, std::uint32_t value, NVNbufferAddress address) {
    if (auto* p = append<ReportValue>(c, Op::report_value, "nvnCommandBufferReportValue"))
        *p = {value, address};
}
void command_set_render_enable(NVNcommandBuffer* c, std::uint32_t enable) {
    if (auto* p = append<RenderEnable>(c, Op::render_enable, "nvnCommandBufferSetRenderEnable"))
        p->enable = guest_bool(enable);
}
void command_set_render_enable_conditional(NVNcommandBuffer* c, NVNconditionalRenderMode mode,
                                           NVNbufferAddress address) {
    if (auto* p = append<RenderEnableConditional>(c, Op::render_enable_conditional,
                                                  "nvnCommandBufferSetRenderEnableConditional"))
        *p = {mode, address};
}

void command_set_render_targets(NVNcommandBuffer* c, int count, const NVNtexture* const* colors,
                                const NVNtextureView* const* views, const NVNtexture* depth,
                                const NVNtextureView* depth_view) {
    if (count < 0 || count > 8 || (count && !colors)) {
        diagnostic("nvnCommandBufferSetRenderTargets", "invalid color target array");
        return;
    }
    auto* p = append<SetRenderTargets>(c, Op::set_render_targets, "nvnCommandBufferSetRenderTargets");
    if (!p)
        return;
    *p = {};
    p->count = count;
    for (int i = 0; i < count; ++i) {
        p->color[i] = texture_snapshot(c, colors[i], "nvnCommandBufferSetRenderTargets");
        copy_view(p->color_views[i], p->has_color_view[i], views ? views[i] : nullptr);
    }
    p->depth = texture_snapshot(c, depth, "nvnCommandBufferSetRenderTargets");
    copy_view(p->depth_view, p->has_depth_view, depth_view);
}
void command_discard_color(NVNcommandBuffer* c, int index) {
    if (auto* p = append<DiscardColor>(c, Op::discard_color, "nvnCommandBufferDiscardColor"))
        p->index = index;
}
void command_discard_depth_stencil(NVNcommandBuffer* c) {
    append_record(c, Op::discard_depth_stencil, 0, "nvnCommandBufferDiscardDepthStencil");
}

void command_downsample(NVNcommandBuffer* c, const NVNtexture* source, const NVNtextureView* source_view,
                        const NVNtexture* destination, const NVNtextureView* destination_view, bool tiled,
                        const char* operation) {
    auto* p = append<Downsample>(c, Op::downsample, operation);
    if (!p)
        return;
    p->source = texture_snapshot(c, source, operation);
    p->destination = texture_snapshot(c, destination, operation);
    copy_view(p->source_view, p->has_source_view, source_view);
    copy_view(p->destination_view, p->has_destination_view, destination_view);
    p->tiled = tiled;
}
void command_downsample_plain(NVNcommandBuffer* c, const NVNtexture* s, const NVNtexture* d) {
    command_downsample(c, s, nullptr, d, nullptr, false, "nvnCommandBufferDownsample");
}
void command_tiled_downsample(NVNcommandBuffer* c, const NVNtexture* s, const NVNtexture* d) {
    command_downsample(c, s, nullptr, d, nullptr, true, "nvnCommandBufferTiledDownsample");
}
void command_downsample_view(NVNcommandBuffer* c, const NVNtexture* s, const NVNtextureView* sv, const NVNtexture* d,
                             const NVNtextureView* dv) {
    command_downsample(c, s, sv, d, dv, false, "nvnCommandBufferDownsampleTextureView");
}
void command_tiled_downsample_view(NVNcommandBuffer* c, const NVNtexture* s, const NVNtextureView* sv,
                                   const NVNtexture* d, const NVNtextureView* dv) {
    command_downsample(c, s, sv, d, dv, true, "nvnCommandBufferTiledDownsampleTextureView");
}

void command_barrier(NVNcommandBuffer* c, int bits) {
    if (auto* p = append<ScalarInt>(c, Op::barrier, "nvnCommandBufferBarrier"))
        p->value = bits;
}
void command_wait_sync(NVNcommandBuffer* c, const NVNsync* sync) {
    if (auto* p = append<WaitSync>(c, Op::wait_sync, "nvnCommandBufferWaitSync"))
        p->sync = as<SyncState>(sync);
}
void command_fence_sync(NVNcommandBuffer* c, NVNsync* sync, NVNsyncCondition condition, int flags) {
    if (auto* p = append<FenceSync>(c, Op::fence_sync, "nvnCommandBufferFenceSync"))
        *p = {as<SyncState>(sync), condition, flags};
}
void command_set_texture_pool(NVNcommandBuffer* c, const NVNtexturePool* pool) {
    if (auto* p = append<SetDescriptorPool>(c, Op::set_texture_pool, "nvnCommandBufferSetTexturePool"))
        p->pool = as<DescriptorPoolState>(pool);
}
void command_set_sampler_pool(NVNcommandBuffer* c, const NVNsamplerPool* pool) {
    if (auto* p = append<SetDescriptorPool>(c, Op::set_sampler_pool, "nvnCommandBufferSetSamplerPool"))
        p->pool = as<DescriptorPoolState>(pool);
}
void command_set_shader_scratch(NVNcommandBuffer* c, const NVNmemoryPool* pool, ptrdiff_t offset, std::size_t size) {
    if (auto* p = append<ShaderScratchMemory>(c, Op::shader_scratch_memory, "nvnCommandBufferSetShaderScratchMemory"))
        *p = {as<MemoryPoolState>(pool), offset, size};
}
void command_zcull(NVNcommandBuffer* c, NVNbufferAddress address, std::size_t size, bool restore,
                   const char* operation) {
    if (auto* p = append<ZCullData>(c, Op::zcull_data, operation))
        *p = {address, size, restore};
}
void command_save_zcull(NVNcommandBuffer* c, NVNbufferAddress a, std::size_t n) {
    command_zcull(c, a, n, false, "nvnCommandBufferSaveZCullData");
}
void command_restore_zcull(NVNcommandBuffer* c, NVNbufferAddress a, std::size_t n) {
    command_zcull(c, a, n, true, "nvnCommandBufferRestoreZCullData");
}

void command_set_copy_stride(NVNcommandBuffer* c, ptrdiff_t stride, bool image, const char* operation) {
    if (auto* state = command_state(c, operation))
        (image ? state->copy_image_stride : state->copy_row_stride) = stride;
    if (auto* p = append<CopyStride>(c, image ? Op::copy_image_stride : Op::copy_stride, operation))
        p->stride = stride;
}
void command_set_copy_row_stride(NVNcommandBuffer* c, ptrdiff_t stride) {
    command_set_copy_stride(c, stride, false, "nvnCommandBufferSetCopyRowStride");
}
void command_set_copy_image_stride(NVNcommandBuffer* c, ptrdiff_t stride) {
    command_set_copy_stride(c, stride, true, "nvnCommandBufferSetCopyImageStride");
}
ptrdiff_t command_get_copy_row_stride(const NVNcommandBuffer* c) {
    const auto* state = command_state(c, "nvnCommandBufferGetCopyRowStride");
    return state ? state->copy_row_stride : 0;
}
ptrdiff_t command_get_copy_image_stride(const NVNcommandBuffer* c) {
    const auto* state = command_state(c, "nvnCommandBufferGetCopyImageStride");
    return state ? state->copy_image_stride : 0;
}

void command_draw_texture(NVNcommandBuffer* c, NVNtextureHandle handle, const NVNdrawTextureRegion* destination,
                          const NVNdrawTextureRegion* source) {
    (void)handle;
    (void)destination;
    (void)source;
    append_unsupported(c, Entry::nvnCommandBufferDrawTexture, "nvnCommandBufferDrawTexture");
}
void command_set_program_subroutines(NVNcommandBuffer* c, NVNprogram* program, NVNshaderStage stage, const int first,
                                     const int count, const int* values) {
    if (first < 0 || count < 0 || (count && !values)) {
        diagnostic("nvnCommandBufferSetProgramSubroutines", "invalid subroutine array");
        return;
    }
    void* raw = append_record(c, Op::program_subroutines,
                              offsetof(ProgramSubroutines, values) + std::size_t(count) * sizeof(int),
                              "nvnCommandBufferSetProgramSubroutines");
    if (!raw)
        return;
    auto* p = static_cast<ProgramSubroutines*>(raw);
    p->program = program_snapshot(c, program, "nvnCommandBufferSetProgramSubroutines");
    p->stage = stage;
    p->first = first;
    p->count = count;
    if (count)
        std::memcpy(p->values, values, std::size_t(count) * sizeof(int));
}
void command_bind_coverage_modulation_table(NVNcommandBuffer* c, const float* values) {
    if (!values) {
        diagnostic("nvnCommandBufferBindCoverageModulationTable", "null table");
        return;
    }
    if (auto* p = append<CoverageModulationTable>(c, Op::coverage_modulation_table,
                                                  "nvnCommandBufferBindCoverageModulationTable"))
        std::copy_n(values, 16, p->values);
}
void command_resolve_depth_buffer(NVNcommandBuffer* c) {
    append_record(c, Op::resolve_depth_buffer, 0, "nvnCommandBufferResolveDepthBuffer");
}

void command_debug_marker(NVNcommandBuffer* c, std::uint32_t domain, const char* text, std::uint8_t action,
                          const char* operation) {
    const std::size_t length = text ? std::strlen(text) : 0;
    void* raw = append_record(c, Op::debug_marker, offsetof(DebugMarker, text) + length + 1, operation);
    if (!raw)
        return;
    auto* p = static_cast<DebugMarker*>(raw);
    p->domain_id = domain;
    p->action = action;
    if (text)
        std::memcpy(p->text, text, length);
    p->text[length] = '\0';
}
void command_push_debug_static(NVNcommandBuffer* c, std::uint32_t d, const char* s) {
    command_debug_marker(c, d, s, 0, "nvnCommandBufferPushDebugGroupStatic");
}
void command_push_debug_dynamic(NVNcommandBuffer* c, std::uint32_t d, const char* s) {
    command_debug_marker(c, d, s, 0, "nvnCommandBufferPushDebugGroupDynamic");
}
void command_push_debug(NVNcommandBuffer* c, std::uint32_t d, const char* s) {
    command_debug_marker(c, d, s, 0, "nvnCommandBufferPushDebugGroup");
}
void command_pop_debug(NVNcommandBuffer* c) { command_debug_marker(c, 0, nullptr, 1, "nvnCommandBufferPopDebugGroup"); }
void command_pop_debug_id(NVNcommandBuffer* c, std::uint32_t d) {
    command_debug_marker(c, d, nullptr, 1, "nvnCommandBufferPopDebugGroupId");
}
void command_insert_debug_static(NVNcommandBuffer* c, std::uint32_t d, const char* s) {
    command_debug_marker(c, d, s, 2, "nvnCommandBufferInsertDebugMarkerStatic");
}
void command_insert_debug_dynamic(NVNcommandBuffer* c, std::uint32_t d, const char* s) {
    command_debug_marker(c, d, s, 2, "nvnCommandBufferInsertDebugMarkerDynamic");
}
void command_insert_debug(NVNcommandBuffer* c, const char* s) {
    command_debug_marker(c, 0, s, 2, "nvnCommandBufferInsertDebugMarker");
}

PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC
command_get_memory_callback(const NVNcommandBuffer* c) {
    const auto* state = command_state(c, "nvnCommandBufferGetMemoryCallback");
    return state ? state->memory_callback : nullptr;
}
void* command_get_memory_callback_data(const NVNcommandBuffer* c) {
    const auto* state = command_state(c, "nvnCommandBufferGetMemoryCallbackData");
    return state ? state->callback_data : nullptr;
}
NVNboolean command_is_recording(const NVNcommandBuffer* c) {
    const auto* state = command_state(c, "nvnCommandBufferIsRecording");
    return state && state->recording ? 1 : 0;
}

void command_wait_event(NVNcommandBuffer* c, const NVNevent* event, NVNeventWaitMode mode, std::uint32_t value) {
    const auto* state = as<EventState>(event);
    if (!state) {
        diagnostic("nvnCommandBufferWaitEvent", "null event");
        return;
    }
    if (auto* p = append<WaitEvent>(c, Op::wait_event, "nvnCommandBufferWaitEvent"))
        *p = {state->value, mode, value};
}
void command_signal_event(NVNcommandBuffer* c, const NVNevent* event, NVNeventSignalMode mode,
                          NVNeventSignalLocation location, int flags, std::uint32_t value) {
    const auto* state = as<EventState>(event);
    if (!state) {
        diagnostic("nvnCommandBufferSignalEvent", "null event");
        return;
    }
    if (auto* p = append<SignalEvent>(c, Op::signal_event, "nvnCommandBufferSignalEvent"))
        *p = {state->value, mode, location, flags, value};
}

} // namespace

void command_refuse(NVNcommandBuffer* c, const char* operation, const char* detail) noexcept {
    diagnostic(operation, detail);
    fail_recording(c);
}

void command_geometry_processing_mode(NVNcommandBuffer* c, int mode) noexcept {
    if (auto* p = append<ScalarInt>(c, Op::geometry_processing_mode, "nvnCommandBufferSetGeometryProcessingMode"))
        p->value = mode;
}

void command_draw_mesh_tasks(NVNcommandBuffer* c, int first, int count) noexcept {
    command_draw_mesh_tasks_at(c, first, count, 0);
}

void command_draw_mesh_tasks_at(NVNcommandBuffer* c, int first, int count, std::uint64_t api_return) noexcept {
    if (first < 0 || count < 0) {
        diagnostic("nvnCommandBufferDrawMeshTasks", "negative first task or count");
        fail_recording(c);
        return;
    }
    if (auto* p = append<DrawMeshTasks>(c, Op::draw_mesh_tasks, "nvnCommandBufferDrawMeshTasks"))
        *p = {first, count, api_return};
}

void command_draw_mesh_tasks_indirect(NVNcommandBuffer* c, NVNbufferAddress address) noexcept {
    if (auto* p = append<DrawMeshTasksIndirect>(c, Op::draw_mesh_tasks_indirect,
                                                "nvnCommandBufferDrawMeshTasksIndirect"))
        p->address = address;
}

void command_push_constants(NVNcommandBuffer* c, NVNshaderStage stage, std::uint32_t bank, std::uint32_t offset,
                            std::uint32_t size, const void* data, const char* operation) noexcept {
    if (!valid_binding_stage(stage) || offset % 4 || size % 4 ||
        offset > push_constant_bytes || size > push_constant_bytes - offset || (size && !data)) {
        diagnostic(operation, "invalid stage, or range outside the push-constant block");
        fail_recording(c);
        return;
    }
    auto* p = append<PushConstants>(c, Op::push_constants, operation);
    if (!p)
        return;
    p->stage = stage;
    p->bank = bank;
    p->offset = offset;
    p->size = size;
    std::memset(p->data, 0, sizeof p->data);
    if (size)
        std::memcpy(p->data, data, size);
}

CommandWalkStatus walk_commands(NVNcommandHandle handle, CommandVisitor visitor, void* context) noexcept {
    if (!visitor)
        return CommandWalkStatus::stopped;
    const auto list = find_handle(handle);
    if (!list)
        return CommandWalkStatus::invalid_handle;
    std::array<NVNcommandHandle, max_call_depth + 1> stack{};
    stack[0] = handle;
    return walk_list(list, visitor, context, stack, 0);
}

std::shared_ptr<const CommandSubmission> prepare_commands(
    std::span<const NVNcommandHandle> handles, CommandWalkStatus& status) {
    auto submission = std::make_shared<CommandSubmission>();
    submission->handles.assign(handles.begin(), handles.end());
    submission->sampler_registration_limit = sampler_registrations.load(std::memory_order_acquire);
    // Frames repeat their shape: start from the previous submission's sizes so
    // the tables do not grow while the lists are pinned.
    static std::atomic<std::size_t> list_hint{0}, deferred_hint{0};
    submission->lists.reserve(list_hint.load(std::memory_order_relaxed));
    submission->deferred.reserve(deferred_hint.load(std::memory_order_relaxed));
    status = CommandWalkStatus::complete;
    for (const auto handle : handles) {
        status = prepare_list(*submission, handle, 0);
        if (status != CommandWalkStatus::complete) return nullptr;
    }
    list_hint.store(submission->lists.size(), std::memory_order_relaxed);
    deferred_hint.store(submission->deferred.size(), std::memory_order_relaxed);
    return submission;
}

std::uint64_t submission_sampler_registrations(const CommandSubmission& submission) noexcept {
    return submission.sampler_registration_limit;
}

CommandWalkStatus walk_commands(const CommandSubmission& submission, CommandVisitor visitor, void* context) noexcept {
    if (!visitor) return CommandWalkStatus::stopped;
    std::array<NVNcommandHandle, max_call_depth + 1> stack{};
    for (const auto handle : submission.handles) {
        stack[0] = handle;
        const auto* list = submission.lists.find(handle);
        if (!list) return CommandWalkStatus::invalid_handle;
        const auto status = walk_list(list->commands, visitor, context, stack, 0, &submission);
        if (status != CommandWalkStatus::complete) return status;
    }
    return CommandWalkStatus::complete;
}

void fence_commands(const CommandSubmission& submission, std::uint64_t fence) noexcept {
    for (auto* sync : submission.syncs) {
        sync->fence = fence;
        sync->fenced = 1;
    }
}

bool submission_has_event_waits(const CommandSubmission& submission) noexcept { return submission.event_waits; }

std::string_view command_walk_status_name(CommandWalkStatus status) noexcept {
    switch (status) {
    case CommandWalkStatus::complete:
        return "complete";
    case CommandWalkStatus::stopped:
        return "stopped";
    case CommandWalkStatus::invalid_handle:
        return "invalid handle";
    case CommandWalkStatus::malformed_record:
        return "malformed record";
    case CommandWalkStatus::call_cycle:
        return "command call cycle";
    case CommandWalkStatus::call_depth_exceeded:
        return "command call depth exceeded";
    case CommandWalkStatus::invalid_deferred_binding:
        return "invalid deferred binding";
    }
    return "unknown command walk status";
}

void register_command_entries() noexcept {
    register_state_entries();
    register_program_entries();
#define ARMIGO_INSTALL(name, function) install<Entry::name>(&function)
    ARMIGO_INSTALL(nvnDeviceFinalizeCommandHandle, device_finalize_command_handle);
    ARMIGO_INSTALL(nvnCommandBufferInitialize, command_initialize);
    ARMIGO_INSTALL(nvnCommandBufferFinalize, command_finalize);
    ARMIGO_INSTALL(nvnCommandBufferSetDebugLabel, command_set_debug_label);
    ARMIGO_INSTALL(nvnCommandBufferSetMemoryCallback, command_set_memory_callback);
    ARMIGO_INSTALL(nvnCommandBufferSetMemoryCallbackData, command_set_memory_callback_data);
    ARMIGO_INSTALL(nvnCommandBufferAddCommandMemory, command_add_command_memory);
    ARMIGO_INSTALL(nvnCommandBufferAddControlMemory, command_add_control_memory);
    ARMIGO_INSTALL(nvnCommandBufferGetCommandMemorySize, command_memory_size);
    ARMIGO_INSTALL(nvnCommandBufferGetCommandMemoryUsed, command_memory_used);
    ARMIGO_INSTALL(nvnCommandBufferGetCommandMemoryFree, command_memory_free);
    ARMIGO_INSTALL(nvnCommandBufferGetControlMemorySize, control_memory_size);
    ARMIGO_INSTALL(nvnCommandBufferGetControlMemoryUsed, control_memory_used);
    ARMIGO_INSTALL(nvnCommandBufferGetControlMemoryFree, control_memory_free);
    ARMIGO_INSTALL(nvnCommandBufferBeginRecording, command_begin_recording);
    ARMIGO_INSTALL(nvnCommandBufferEndRecording, command_end_recording);
    ARMIGO_INSTALL(nvnCommandBufferCallCommands, command_call_commands);
    ARMIGO_INSTALL(nvnCommandBufferCopyCommands, command_copy_commands);
    ARMIGO_INSTALL(nvnCommandBufferBindBlendState, command_bind_blend_state);
    ARMIGO_INSTALL(nvnCommandBufferBindChannelMaskState, command_bind_channel_mask_state);
    ARMIGO_INSTALL(nvnCommandBufferBindColorState, command_bind_color_state);
    ARMIGO_INSTALL(nvnCommandBufferBindMultisampleState, command_bind_multisample_state);
    ARMIGO_INSTALL(nvnCommandBufferBindPolygonState, command_bind_polygon_state);
    ARMIGO_INSTALL(nvnCommandBufferBindDepthStencilState, command_bind_depth_stencil_state);
    ARMIGO_INSTALL(nvnCommandBufferBindVertexAttribState, command_bind_vertex_attributes);
    ARMIGO_INSTALL(nvnCommandBufferBindVertexStreamState, command_bind_vertex_streams);
    ARMIGO_INSTALL(nvnCommandBufferBindProgram, command_bind_program);
    ARMIGO_INSTALL(nvnCommandBufferBindVertexBuffer, command_bind_vertex_buffer);
    ARMIGO_INSTALL(nvnCommandBufferBindVertexBuffers, command_bind_vertex_buffers);
    ARMIGO_INSTALL(nvnCommandBufferBindUniformBuffer, command_bind_uniform_buffer);
    ARMIGO_INSTALL(nvnCommandBufferBindUniformBuffers, command_bind_uniform_buffers);
    ARMIGO_INSTALL(nvnCommandBufferBindTransformFeedbackBuffer, command_bind_transform_feedback_buffer);
    ARMIGO_INSTALL(nvnCommandBufferBindTransformFeedbackBuffers, command_bind_transform_feedback_buffers);
    ARMIGO_INSTALL(nvnCommandBufferBindStorageBuffer, command_bind_storage_buffer);
    ARMIGO_INSTALL(nvnCommandBufferBindStorageBuffers, command_bind_storage_buffers);
    ARMIGO_INSTALL(nvnCommandBufferBindTexture, command_bind_texture);
    ARMIGO_INSTALL(nvnCommandBufferBindTextures, command_bind_textures);
    ARMIGO_INSTALL(nvnCommandBufferBindImage, command_bind_image);
    ARMIGO_INSTALL(nvnCommandBufferBindImages, command_bind_images);
    ARMIGO_INSTALL(nvnCommandBufferSetPatchSize, command_set_patch_size);
    ARMIGO_INSTALL(nvnCommandBufferSetInnerTessellationLevels, command_set_inner_tessellation);
    ARMIGO_INSTALL(nvnCommandBufferSetOuterTessellationLevels, command_set_outer_tessellation);
    ARMIGO_INSTALL(nvnCommandBufferSetPrimitiveRestart, command_set_primitive_restart);
    ARMIGO_INSTALL(nvnCommandBufferBeginTransformFeedback, command_begin_transform_feedback);
    ARMIGO_INSTALL(nvnCommandBufferEndTransformFeedback, command_end_transform_feedback);
    ARMIGO_INSTALL(nvnCommandBufferPauseTransformFeedback, command_pause_transform_feedback);
    ARMIGO_INSTALL(nvnCommandBufferResumeTransformFeedback, command_resume_transform_feedback);
    ARMIGO_INSTALL(nvnCommandBufferDrawTransformFeedback, command_draw_transform_feedback);
    ARMIGO_INSTALL(nvnCommandBufferDrawArrays, command_draw_arrays);
    ARMIGO_INSTALL(nvnCommandBufferDrawElements, command_draw_elements);
    ARMIGO_INSTALL(nvnCommandBufferDrawElementsBaseVertex, command_draw_elements_base_vertex);
    ARMIGO_INSTALL(nvnCommandBufferDrawArraysInstanced, command_draw_arrays_instanced);
    ARMIGO_INSTALL(nvnCommandBufferDrawElementsInstanced, command_draw_elements_instanced);
    ARMIGO_INSTALL(nvnCommandBufferDrawArraysIndirect, command_draw_arrays_indirect);
    ARMIGO_INSTALL(nvnCommandBufferDrawElementsIndirect, command_draw_elements_indirect);
    ARMIGO_INSTALL(nvnCommandBufferMultiDrawArraysIndirectCount, command_multi_draw_arrays_indirect);
    ARMIGO_INSTALL(nvnCommandBufferMultiDrawElementsIndirectCount, command_multi_draw_elements_indirect);
    ARMIGO_INSTALL(nvnCommandBufferClearColor, command_clear_color);
    ARMIGO_INSTALL(nvnCommandBufferClearColori, command_clear_colori);
    ARMIGO_INSTALL(nvnCommandBufferClearColorui, command_clear_colorui);
    ARMIGO_INSTALL(nvnCommandBufferClearDepthStencil, command_clear_depth_stencil);
    ARMIGO_INSTALL(nvnCommandBufferDispatchCompute, command_dispatch);
    ARMIGO_INSTALL(nvnCommandBufferDispatchComputeIndirect, command_dispatch_indirect);
    ARMIGO_INSTALL(nvnCommandBufferSetViewport, command_set_viewport);
    ARMIGO_INSTALL(nvnCommandBufferSetViewports, command_set_viewports);
    ARMIGO_INSTALL(nvnCommandBufferSetViewportSwizzles, command_set_viewport_swizzles);
    ARMIGO_INSTALL(nvnCommandBufferSetScissor, command_set_scissor);
    ARMIGO_INSTALL(nvnCommandBufferSetScissors, command_set_scissors);
    ARMIGO_INSTALL(nvnCommandBufferSetDepthRange, command_set_depth_range);
    ARMIGO_INSTALL(nvnCommandBufferSetDepthBounds, command_set_depth_bounds);
    ARMIGO_INSTALL(nvnCommandBufferSetDepthRanges, command_set_depth_ranges);
    ARMIGO_INSTALL(nvnCommandBufferSetTiledCacheAction, command_set_tiled_cache_action);
    ARMIGO_INSTALL(nvnCommandBufferSetTiledCacheTileSize, command_set_tiled_cache_tile_size);
    ARMIGO_INSTALL(nvnCommandBufferBindSeparateTexture, command_bind_separate_texture);
    ARMIGO_INSTALL(nvnCommandBufferBindSeparateSampler, command_bind_separate_sampler);
    ARMIGO_INSTALL(nvnCommandBufferBindSeparateTextures, command_bind_separate_textures);
    ARMIGO_INSTALL(nvnCommandBufferBindSeparateSamplers, command_bind_separate_samplers);
    ARMIGO_INSTALL(nvnCommandBufferSetStencilValueMask, command_set_stencil_value_mask);
    ARMIGO_INSTALL(nvnCommandBufferSetStencilMask, command_set_stencil_mask);
    ARMIGO_INSTALL(nvnCommandBufferSetStencilRef, command_set_stencil_ref);
    ARMIGO_INSTALL(nvnCommandBufferSetBlendColor, command_set_blend_color);
    ARMIGO_INSTALL(nvnCommandBufferSetPointSize, command_set_point_size);
    ARMIGO_INSTALL(nvnCommandBufferSetLineWidth, command_set_line_width);
    ARMIGO_INSTALL(nvnCommandBufferSetPolygonOffsetClamp, command_set_polygon_offset);
    ARMIGO_INSTALL(nvnCommandBufferSetAlphaRef, command_set_alpha_ref);
    ARMIGO_INSTALL(nvnCommandBufferSetSampleMask, command_set_sample_mask);
    ARMIGO_INSTALL(nvnCommandBufferSetRasterizerDiscard, command_set_rasterizer_discard);
    ARMIGO_INSTALL(nvnCommandBufferSetDepthClamp, command_set_depth_clamp);
    ARMIGO_INSTALL(nvnCommandBufferSetConservativeRasterEnable, command_set_conservative_raster);
    ARMIGO_INSTALL(nvnCommandBufferSetConservativeRasterDilate, command_set_conservative_dilate);
    ARMIGO_INSTALL(nvnCommandBufferSetSubpixelPrecisionBias, command_set_subpixel_bias);
    ARMIGO_INSTALL(nvnCommandBufferCopyBufferToTexture, command_copy_buffer_to_texture);
    ARMIGO_INSTALL(nvnCommandBufferCopyTextureToBuffer, command_copy_texture_to_buffer);
    ARMIGO_INSTALL(nvnCommandBufferCopyTextureToTexture, command_copy_texture_to_texture);
    ARMIGO_INSTALL(nvnCommandBufferCopyBufferToBuffer, command_copy_buffer);
    ARMIGO_INSTALL(nvnCommandBufferClearBuffer, command_clear_buffer);
    ARMIGO_INSTALL(nvnCommandBufferClearTexture, command_clear_texture);
    ARMIGO_INSTALL(nvnCommandBufferClearTexturei, command_clear_texturei);
    ARMIGO_INSTALL(nvnCommandBufferClearTextureui, command_clear_textureui);
    ARMIGO_INSTALL(nvnCommandBufferUpdateUniformBuffer, command_update_uniforms);
    ARMIGO_INSTALL(nvnCommandBufferReportCounter, command_report_counter);
    ARMIGO_INSTALL(nvnCommandBufferResetCounter, command_reset_counter);
    ARMIGO_INSTALL(nvnCommandBufferReportValue, command_report_value);
    ARMIGO_INSTALL(nvnCommandBufferSetRenderEnable, command_set_render_enable);
    ARMIGO_INSTALL(nvnCommandBufferSetRenderEnableConditional, command_set_render_enable_conditional);
    ARMIGO_INSTALL(nvnCommandBufferSetRenderTargets, command_set_render_targets);
    ARMIGO_INSTALL(nvnCommandBufferDiscardColor, command_discard_color);
    ARMIGO_INSTALL(nvnCommandBufferDiscardDepthStencil, command_discard_depth_stencil);
    ARMIGO_INSTALL(nvnCommandBufferDownsample, command_downsample_plain);
    ARMIGO_INSTALL(nvnCommandBufferTiledDownsample, command_tiled_downsample);
    ARMIGO_INSTALL(nvnCommandBufferDownsampleTextureView, command_downsample_view);
    ARMIGO_INSTALL(nvnCommandBufferTiledDownsampleTextureView, command_tiled_downsample_view);
    ARMIGO_INSTALL(nvnCommandBufferBarrier, command_barrier);
    ARMIGO_INSTALL(nvnCommandBufferWaitSync, command_wait_sync);
    ARMIGO_INSTALL(nvnCommandBufferFenceSync, command_fence_sync);
    ARMIGO_INSTALL(nvnCommandBufferSetTexturePool, command_set_texture_pool);
    ARMIGO_INSTALL(nvnCommandBufferSetSamplerPool, command_set_sampler_pool);
    ARMIGO_INSTALL(nvnCommandBufferSetShaderScratchMemory, command_set_shader_scratch);
    ARMIGO_INSTALL(nvnCommandBufferSaveZCullData, command_save_zcull);
    ARMIGO_INSTALL(nvnCommandBufferRestoreZCullData, command_restore_zcull);
    ARMIGO_INSTALL(nvnCommandBufferSetCopyRowStride, command_set_copy_row_stride);
    ARMIGO_INSTALL(nvnCommandBufferSetCopyImageStride, command_set_copy_image_stride);
    ARMIGO_INSTALL(nvnCommandBufferGetCopyRowStride, command_get_copy_row_stride);
    ARMIGO_INSTALL(nvnCommandBufferGetCopyImageStride, command_get_copy_image_stride);
    ARMIGO_INSTALL(nvnCommandBufferDrawTexture, command_draw_texture);
    ARMIGO_INSTALL(nvnCommandBufferSetProgramSubroutines, command_set_program_subroutines);
    ARMIGO_INSTALL(nvnCommandBufferBindCoverageModulationTable, command_bind_coverage_modulation_table);
    ARMIGO_INSTALL(nvnCommandBufferResolveDepthBuffer, command_resolve_depth_buffer);
    ARMIGO_INSTALL(nvnCommandBufferPushDebugGroupStatic, command_push_debug_static);
    ARMIGO_INSTALL(nvnCommandBufferPushDebugGroupDynamic, command_push_debug_dynamic);
    ARMIGO_INSTALL(nvnCommandBufferPushDebugGroup, command_push_debug);
    ARMIGO_INSTALL(nvnCommandBufferPopDebugGroup, command_pop_debug);
    ARMIGO_INSTALL(nvnCommandBufferPopDebugGroupId, command_pop_debug_id);
    ARMIGO_INSTALL(nvnCommandBufferInsertDebugMarkerStatic, command_insert_debug_static);
    ARMIGO_INSTALL(nvnCommandBufferInsertDebugMarkerDynamic, command_insert_debug_dynamic);
    ARMIGO_INSTALL(nvnCommandBufferInsertDebugMarker, command_insert_debug);
    ARMIGO_INSTALL(nvnCommandBufferGetMemoryCallback, command_get_memory_callback);
    ARMIGO_INSTALL(nvnCommandBufferGetMemoryCallbackData, command_get_memory_callback_data);
    ARMIGO_INSTALL(nvnCommandBufferIsRecording, command_is_recording);
    ARMIGO_INSTALL(nvnCommandBufferWaitEvent, command_wait_event);
    ARMIGO_INSTALL(nvnCommandBufferSignalEvent, command_signal_event);
    ARMIGO_INSTALL(nvnCommandBufferBindTexturesDeferred, command_bind_textures_deferred);
#undef ARMIGO_INSTALL
}

} // namespace armigo::nvn
