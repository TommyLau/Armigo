// ============================================================================
// Armigo Tests - NVN Command Recording
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "commands.h"
#include "entry_table.h"
#include "programs.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                                               \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (false)

template <Entry E> typename EntrySignature<E>::type load() {
    return reinterpret_cast<typename EntrySignature<E>::type>(armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

struct ObservedRecord {
    Op op{};
    std::vector<std::byte> payload;
};

// Collects the guest's commands; the replay's sampler stamps are checked
// separately.
bool collect(const CommandRecordView& view, void* context) {
    if (view.op == Op::sampler_registrations)
        return true;
    auto& records = *static_cast<std::vector<ObservedRecord>*>(context);
    ObservedRecord record{view.op, std::vector<std::byte>(view.payload_size)};
    if (view.payload_size)
        std::memcpy(record.payload.data(), view.payload, view.payload_size);
    records.push_back(std::move(record));
    return true;
}

template <typename Payload> const Payload& payload(const ObservedRecord& record) {
    return *reinterpret_cast<const Payload*>(record.payload.data());
}

alignas(8) std::array<std::byte, 8192> callback_command_memory{};
std::array<std::byte, 32> callback_control_memory{};
NVNmemoryPool callback_pool{};
unsigned command_callbacks = 0;
unsigned control_callbacks = 0;

struct MemoryCallbackContext {
    NVNmemoryPool* command_pool{};
    std::size_t command_size{};
    void* control_memory{};
    std::size_t control_size{};
    unsigned command_callbacks{};
    unsigned control_callbacks{};
};

void memory_callback(NVNcommandBuffer* command_buffer, NVNcommandBufferMemoryEvent event, std::size_t minimum_size,
                     void*) {
    if (event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        ++command_callbacks;
        CHECK(minimum_size == 4);  // one native command word
        load<Entry::nvnCommandBufferAddCommandMemory>()(command_buffer, &callback_pool, 0,
                                                        callback_command_memory.size());
    } else {
        ++control_callbacks;
        CHECK(minimum_size == sizeof(NVNcommandHandle));
        load<Entry::nvnCommandBufferAddControlMemory>()(command_buffer, callback_control_memory.data(),
                                                        callback_control_memory.size());
    }
}

void configurable_memory_callback(NVNcommandBuffer* command_buffer, NVNcommandBufferMemoryEvent event,
                                  std::size_t minimum_size, void* data) {
    auto& context = *static_cast<MemoryCallbackContext*>(data);
    if (event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        ++context.command_callbacks;
        CHECK(minimum_size == 4);  // one native command word
        if (context.command_pool)
            load<Entry::nvnCommandBufferAddCommandMemory>()(command_buffer, context.command_pool, 0,
                                                            context.command_size);
    } else {
        ++context.control_callbacks;
        CHECK(minimum_size == sizeof(NVNcommandHandle));
        if (context.control_memory)
            load<Entry::nvnCommandBufferAddControlMemory>()(command_buffer, context.control_memory,
                                                            context.control_size);
    }
}

void initialize_pool(NVNmemoryPool& pool, void* memory, std::size_t size) {
    auto& state = *reinterpret_cast<MemoryPoolState*>(&pool);
    state.memory = static_cast<std::uint8_t*>(memory);
    state.size = size;
}

NVNcommandHandle record_draw(NVNcommandBuffer& command_buffer, int first) {
    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferDrawArrays>()(&command_buffer, NVN_DRAW_PRIMITIVE_TRIANGLES, first, 3);
    return load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
}

// A draw recorded after sampler registrations moved is preceded by a stamp of
// them; a later draw with no registration in between is not. (Stamps take none
// of the game's command memory: test_command_memory_accounting.)
bool collect_stamps(const CommandRecordView& view, void* context) {
    auto& stamps = *static_cast<std::vector<std::pair<Op, std::uint64_t>>*>(context);
    std::uint64_t value = 0;
    if (view.op == Op::sampler_registrations)
        std::memcpy(&value, view.payload, sizeof value);
    if (view.op == Op::sampler_registrations || view.op == Op::draw || view.op == Op::call)
        stamps.emplace_back(view.op, value);
    return true;
}

void test_sampler_stamps(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device) == 1);
    sampler_registrations.fetch_add(1);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto first = sampler_registrations.fetch_add(1) + 1;
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    std::vector<std::pair<Op, std::uint64_t>> seen;
    CHECK(walk_commands(handle, collect_stamps, &seen) == CommandWalkStatus::complete);
    // The first draw stamps whatever count was current, the second the new one.
    CHECK(seen.size() == 5);
    if (seen.size() == 5) {
        CHECK(seen[0].first == Op::sampler_registrations && seen[1].first == Op::draw);
        CHECK(seen[2].first == Op::sampler_registrations && seen[2].second == first);
        CHECK(seen[3].first == Op::draw && seen[4].first == Op::draw);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

// Records live in host blocks, so recording never fails for lack of game
// command memory; the game's segment only advances one native word per
// command and asks the memory callback when it is used up.
void test_command_memory_accounting(NVNdevice& device) {
    NVNcommandBuffer no_memory{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&no_memory, &device) == 1);
    const auto unaccounted = record_draw(no_memory, 3);
    CHECK(unaccounted != 0);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(unaccounted, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && records[0].op == Op::draw && payload<Draw>(records[0]).first == 3);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, unaccounted);
    load<Entry::nvnCommandBufferFinalize>()(&no_memory);

    // A one-byte segment cannot hold a native word: the callback is asked
    // once, adds nothing usable, and recording still completes.
    NVNcommandBuffer undersized_callback{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&undersized_callback, &device) == 1);
    std::array<std::byte, 1> undersized_memory{};
    NVNmemoryPool undersized_pool{};
    initialize_pool(undersized_pool, undersized_memory.data(), undersized_memory.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&undersized_callback, &undersized_pool, 0, undersized_memory.size());
    MemoryCallbackContext undersized_context{&undersized_pool, undersized_memory.size()};
    load<Entry::nvnCommandBufferSetMemoryCallback>()(&undersized_callback, configurable_memory_callback);
    load<Entry::nvnCommandBufferSetMemoryCallbackData>()(&undersized_callback, &undersized_context);
    load<Entry::nvnCommandBufferBeginRecording>()(&undersized_callback);
    load<Entry::nvnCommandBufferDrawArrays>()(&undersized_callback, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferDrawArrays>()(&undersized_callback, NVN_DRAW_PRIMITIVE_TRIANGLES, 1, 3);
    CHECK(load<Entry::nvnCommandBufferEndRecording>()(&undersized_callback) != 0);
    CHECK(undersized_context.command_callbacks == 1);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryUsed>()(&undersized_callback) == 0);
    CHECK(undersized_memory[0] == std::byte{});
    load<Entry::nvnCommandBufferFinalize>()(&undersized_callback);

    // The handle is the recording's native start: recording again at the same
    // position replaces the list behind the handle the game already holds.
    NVNcommandBuffer reused{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&reused, &device) == 1);
    alignas(8) std::array<std::byte, 64> reused_memory{};
    NVNmemoryPool reused_pool{};
    initialize_pool(reused_pool, reused_memory.data(), reused_memory.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&reused, &reused_pool, 0, reused_memory.size());
    const auto first = record_draw(reused, 5);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&reused, &reused_pool, 0, reused_memory.size());
    const auto second = record_draw(reused, 6);
    CHECK(first == second && first == reinterpret_cast<NVNcommandHandle>(reused_memory.data()));
    records.clear();
    CHECK(walk_commands(first, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && payload<Draw>(records[0]).first == 6);
    load<Entry::nvnCommandBufferFinalize>()(&reused);
}

// A command-memory range that starts inside its pool may run past the pool's
// end, as a game's ring of chunks does; nothing is written there, so the range
// only accounts for the game's position. A range that starts at or past the
// end, or is larger than the pool, is refused.
void test_command_memory_past_pool_end(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device) == 1);
    alignas(8) std::array<std::byte, 64> memory{};
    NVNmemoryPool pool{};
    initialize_pool(pool, memory.data(), 60);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 52, 12);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemorySize>()(&commands) == 12);
    CHECK(record_draw(commands, 3) == reinterpret_cast<NVNcommandHandle>(memory.data() + 52));
    // Recording past the logical end: the position stops at it (no callback),
    // and the guard bytes past the pool stay untouched.
    for (int first = 4; first != 9; ++first) record_draw(commands, first);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryUsed>()(&commands) == 12);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryFree>()(&commands) == 0);
    CHECK(memory[60] == std::byte{} && memory[63] == std::byte{});
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 60, 4);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemorySize>()(&commands) == 0);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0, 64);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemorySize>()(&commands) == 0);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

// A recording that reuses command memory a registered list occupies replaces
// that list: its handle stops resolving, as natively its commands have been
// overwritten, and the host blocks behind it can be freed. Lists the new
// recording does not overlap stay registered.
void test_reused_memory_releases_lists(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device) == 1);
    alignas(8) std::array<std::byte, 64> memory{};
    NVNmemoryPool pool{};
    initialize_pool(pool, memory.data(), memory.size());
    const auto base = reinterpret_cast<NVNcommandHandle>(memory.data());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0, memory.size());
    // A takes three native words [0, 12), B one word [12, 16).
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    for (int first = 0; first != 3; ++first)
        load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, first, 3);
    const auto a = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    const auto b = record_draw(commands, 7);
    CHECK(a == base && b == base + 12);
    std::vector<ObservedRecord> adjacent;
    CHECK(walk_commands(a, collect, &adjacent) == CommandWalkStatus::complete && adjacent.size() == 3);
    // C starts inside A's memory: A is replaced, B is untouched.
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 4, memory.size() - 4);
    const auto c = record_draw(commands, 9);
    CHECK(c == base + 4);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(a, collect, &records) == CommandWalkStatus::invalid_handle);
    records.clear();
    CHECK(walk_commands(b, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && payload<Draw>(records[0]).first == 7);
    records.clear();
    CHECK(walk_commands(c, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && payload<Draw>(records[0]).first == 9);

    // D spans two segments (the second added mid-recording); a recording that
    // reuses only its second segment still replaces D.
    alignas(8) std::array<std::byte, 64> second{};
    NVNmemoryPool second_pool{};
    initialize_pool(second_pool, second.data(), second.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 32, 8);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 20, 3);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &second_pool, 0, second.size());
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 21, 3);
    const auto d = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(d == base + 32);
    records.clear();
    CHECK(walk_commands(d, collect, &records) == CommandWalkStatus::complete && records.size() == 2);
    const auto second_base = reinterpret_cast<NVNcommandHandle>(second.data());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &second_pool, 0, second.size());
    const auto e = record_draw(commands, 22);
    CHECK(e == second_base);
    CHECK(walk_commands(d, collect, &records) == CommandWalkStatus::invalid_handle);
    // Finalizing a handle releases its memory claim too: a later recording
    // there registers without disturbing anything else.
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, e);
    CHECK(walk_commands(e, collect, &records) == CommandWalkStatus::invalid_handle);
    records.clear();
    CHECK(walk_commands(b, collect, &records) == CommandWalkStatus::complete && records.size() == 1);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, b);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, c);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

void test_control_memory_failure(NVNdevice& device, NVNcommandHandle called) {
    NVNcommandBuffer command_buffer{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&command_buffer, &device) == 1);
    alignas(8) std::array<std::byte, 128> command_memory{};
    NVNmemoryPool pool{};
    initialize_pool(pool, command_memory.data(), command_memory.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&command_buffer, &pool, 0, command_memory.size());
    MemoryCallbackContext context{};
    load<Entry::nvnCommandBufferSetMemoryCallback>()(&command_buffer, configurable_memory_callback);
    load<Entry::nvnCommandBufferSetMemoryCallbackData>()(&command_buffer, &context);
    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferCallCommands>()(&command_buffer, 1, &called);
    load<Entry::nvnCommandBufferCallCommands>()(&command_buffer, 1, &called);
    CHECK(load<Entry::nvnCommandBufferEndRecording>()(&command_buffer) == 0);
    CHECK(context.control_callbacks == 1);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryUsed>()(&command_buffer) == 4);  // the failed recording stops at one call

    std::array<std::byte, sizeof(NVNcommandHandle)> control_memory{};
    load<Entry::nvnCommandBufferAddControlMemory>()(&command_buffer, control_memory.data(), control_memory.size());
    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferCallCommands>()(&command_buffer, 1, &called);
    const auto recovered = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
    CHECK(recovered != 0);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(recovered, collect, &records) == CommandWalkStatus::complete);
    CHECK(!records.empty());
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, recovered);
    load<Entry::nvnCommandBufferFinalize>()(&command_buffer);
}

void test_call_depth_limit(NVNdevice& device) {
    constexpr std::size_t chain_length = 66;
    NVNcommandBuffer command_buffer{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&command_buffer, &device) == 1);
    alignas(8) std::array<std::byte, 2048> command_memory{};
    std::array<std::byte, 1024> control_memory{};
    NVNmemoryPool pool{};
    initialize_pool(pool, command_memory.data(), command_memory.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&command_buffer, &pool, 0, command_memory.size());
    load<Entry::nvnCommandBufferAddControlMemory>()(&command_buffer, control_memory.data(), control_memory.size());

    std::array<NVNcommandHandle, chain_length> handles{};
    handles[0] = record_draw(command_buffer, 0);
    CHECK(handles[0] != 0);
    for (std::size_t i = 1; i < handles.size(); ++i) {
        load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
        load<Entry::nvnCommandBufferCallCommands>()(&command_buffer, 1, &handles[i - 1]);
        handles[i] = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
        CHECK(handles[i] != 0);
    }

    std::vector<ObservedRecord> records;
    CHECK(walk_commands(handles[handles.size() - 2], collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && records[0].op == Op::draw);
    records.clear();
    CHECK(walk_commands(handles.back(), collect, &records) == CommandWalkStatus::call_depth_exceeded);
    CHECK(records.empty());
    CommandWalkStatus status;
    // The first root memoizes a subtree that the second root reaches deeper.
    const std::array<NVNcommandHandle, 2> valid{handles[0], handles[64]};
    CHECK(prepare_commands(valid, status) && status == CommandWalkStatus::complete);
    const std::array<NVNcommandHandle, 2> too_deep{handles[64], handles[65]};
    CHECK(!prepare_commands(too_deep, status) && status == CommandWalkStatus::call_depth_exceeded);
    for (const auto handle : handles)
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&command_buffer);
}

void test_donated_segments_and_copy(NVNdevice& device) {
    NVNcommandBuffer commands{};
    NVNmemoryPool first_pool{}, second_pool{};
    alignas(8) std::array<std::byte, 32> first{};
    alignas(8) std::array<std::byte, 128> second{};
    initialize_pool(first_pool, first.data(), first.size());
    initialize_pool(second_pool, second.data(), second.size());
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &first_pool, 0, first.size());
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 2, 3);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &second_pool, 0, second.size());
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 5, 3);
    const auto original = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(original != 0);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(original, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 2 && payload<Draw>(records[0]).first == 2 && payload<Draw>(records[1]).first == 5);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferCopyCommands>()(&commands, 1, &original);
    const auto copied = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(copied != 0);
    // Records live in host blocks: clearing the game's segment changes neither list.
    first.fill(std::byte{0});
    records.clear();
    CHECK(walk_commands(original, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 2);
    records.clear();
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 2 && payload<Draw>(records[0]).first == 2 && payload<Draw>(records[1]).first == 5);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, original);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, copied);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

// Records cross arena boundaries without changing order or ownership. Programs
// remain executable after replacement/finalization, including copied commands.
void test_record_blocks_and_program_lifetime(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &device));
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 0, nullptr));
    const auto first = *reinterpret_cast<const ProgramState*>(&program);
    std::weak_ptr<const ResolvedProgram> first_owner = retain_program(first);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    constexpr int draws = 4096;
    for (int i = 0; i < draws; ++i) {
        load<Entry::nvnCommandBufferBindProgram>()(&commands, &program, NVN_SHADER_STAGE_VERTEX_BIT);
        load<Entry::nvnCommandBufferDrawElementsBaseVertex>()(
            &commands, NVN_DRAW_PRIMITIVE_TRIANGLES, NVN_INDEX_TYPE_UNSIGNED_SHORT, 3, 0x1000, i);
    }
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 0, nullptr));
    const auto second = *reinterpret_cast<const ProgramState*>(&program);
    std::weak_ptr<const ResolvedProgram> second_owner = retain_program(second);
    load<Entry::nvnCommandBufferBindProgram>()(&commands, &program, NVN_SHADER_STAGE_FRAGMENT_BIT);
    load<Entry::nvnProgramFinalize>()(&program);
    CHECK(!first_owner.expired() && !second_owner.expired());
    // A flexible record larger than the ordinary maximum block size.
    std::vector<std::byte> data(80 * 1024, std::byte{0x5a});
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&commands, 0x2000, data.size(), 0, data.size(), data.data());
    const auto original = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(original);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferCopyCommands>()(&commands, 1, &original);
    const auto copied = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(copied);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, original);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 2 * draws + 2);
    if (records.size() == 2 * draws + 2) {
        for (int i = 0; i < draws; ++i) {
            CHECK(payload<BindProgram>(records[2 * i]).program == first.snapshot);
            CHECK(payload<DrawIndexed>(records[2 * i + 1]).base_vertex == i);
        }
        CHECK(payload<BindProgram>(records[2 * draws]).program == second.snapshot);
        const auto& update = payload<UpdateUniforms>(records.back());
        CHECK(update.size == data.size());
        CHECK(std::memcmp(update.data, data.data(), data.size()) == 0);
    }
    CHECK(!first_owner.expired() && !second_owner.expired());
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, copied);
    CHECK(first_owner.expired() && second_owner.expired());
}

// Submission owns list storage and callees, but only deferred arrays are copied.
// Pointer identity verifies that replay visits the original ordinary payload.
void test_submission_lifetime(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
    std::array<std::byte, 64> control{};
    load<Entry::nvnCommandBufferAddControlMemory>()(&commands, control.data(), control.size());
    std::array<NVNtextureHandle, 40> handles{};
    handles.fill(17);
    NVNsync sync{};
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &device));
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 0, nullptr));
    std::weak_ptr<const ResolvedProgram> executable = retain_program(*reinterpret_cast<ProgramState*>(&program));
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferBindProgram>()(&commands, &program, NVN_SHADER_STAGE_VERTEX_BIT);
    load<Entry::nvnCommandBufferDrawArrays>()(&commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 11, 3);
    load<Entry::nvnCommandBufferBindTexturesDeferred>()(&commands, NVN_SHADER_STAGE_FRAGMENT, 2,
                                                       handles.size(), handles.data());
    load<Entry::nvnCommandBufferFenceSync>()(&commands, &sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    const auto leaf = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    const void* draw_pointer = nullptr;
    CHECK(walk_commands(leaf, [](const CommandRecordView& view, void* context) {
        if (view.op == Op::draw) *static_cast<const void**>(context) = view.payload;
        return true;
    }, &draw_pointer) == CommandWalkStatus::complete);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    const std::array<NVNcommandHandle, 2> repeated{leaf, leaf};
    load<Entry::nvnCommandBufferCallCommands>()(&commands, repeated.size(), repeated.data());
    const auto root = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CommandWalkStatus status;
    auto first = prepare_commands(std::span(&root, 1), status);
    CHECK(first && status == CommandWalkStatus::complete);
    handles.fill(29);
    auto second = prepare_commands(std::span(&root, 1), status);
    CHECK(second && status == CommandWalkStatus::complete);
    handles.fill(99);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, leaf);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, root);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
    load<Entry::nvnProgramFinalize>()(&program);
    CHECK(!executable.expired());
    if (!first || !second) return;
    fence_commands(*first, 83);
    CHECK(reinterpret_cast<SyncState*>(&sync)->fence == 83 && reinterpret_cast<SyncState*>(&sync)->fenced);
    const void* submitted_draw = nullptr;
    CHECK(walk_commands(*first, [](const CommandRecordView& view, void* context) {
        if (view.op == Op::draw) *static_cast<const void**>(context) = view.payload;
        return true;
    }, &submitted_draw) == CommandWalkStatus::complete);
    CHECK(draw_pointer == submitted_draw);
    for (const auto* submission : {first.get(), second.get()}) {
        std::vector<ObservedRecord> records;
        CHECK(walk_commands(*submission, collect, &records) == CommandWalkStatus::complete);
        CHECK(records.size() == 10);
        if (records.size() == 10) {
            for (int base : {0, 5}) {
                CHECK(payload<Draw>(records[base + 1]).first == 11);
                const auto expected = submission == first.get() ? 17u : 29u;
                CHECK(payload<BindHandles>(records[base + 2]).handles[0] == expected);
                CHECK(payload<BindHandles>(records[base + 3]).handles[7] == expected);
            }
        }
    }
    first.reset();
    CHECK(!executable.expired());
    second.reset();
    CHECK(executable.expired());
    const auto invalid = prepare_commands(std::span(&root, 1), status);
    CHECK(!invalid && status == CommandWalkStatus::invalid_handle);
}

void test_compact_bindings(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
    std::array<NVNvertexAttribState, 16> attributes{};
    std::array<NVNvertexStreamState, 16> streams{};
    std::array<NVNbufferRange, 16> ranges{};
    std::array<NVNtextureHandle, 32> textures{};
    for (std::size_t i = 0; i < ranges.size(); ++i) ranges[i] = {0x1000 + i * 256, i + 1};
    for (std::size_t i = 0; i < textures.size(); ++i) textures[i] = 100 + i;
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&commands, 0, nullptr);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&commands, 3, attributes.data());
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&commands, 16, attributes.data());
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&commands, 1, streams.data());
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&commands, 16, streams.data());
    load<Entry::nvnCommandBufferBindVertexBuffers>()(&commands, 15, 1, ranges.data());
    load<Entry::nvnCommandBufferBindVertexBuffers>()(&commands, 0, 16, ranges.data());
    load<Entry::nvnCommandBufferBindTextures>()(&commands, NVN_SHADER_STAGE_FRAGMENT, 127, 1, textures.data());
    load<Entry::nvnCommandBufferBindTextures>()(&commands, NVN_SHADER_STAGE_FRAGMENT, 96, 32, textures.data());
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(handle, collect, &records) == CommandWalkStatus::complete);
    // Texture bindings also carry the sampler registrations made before them.
    const std::array<std::size_t, 9> sizes{8, 24, 72, 16, 136, 24, 264, 32, 280};
    CHECK(records.size() == sizes.size());
    if (records.size() == sizes.size()) {
        for (std::size_t i = 0; i < sizes.size(); ++i) CHECK(records[i].payload.size() == sizes[i]);
        CHECK(payload<BindVertexBuffers>(records[5]).ranges[0].address == ranges[0].address);
        CHECK(payload<BindVertexBuffers>(records[6]).ranges[15].size == ranges[15].size);
        CHECK(payload<BindHandles>(records[8]).handles[31] == textures[31]);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

void test_submission_validation(NVNdevice& device) {
    NVNcommandBuffer commands{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
    alignas(8) std::array<std::byte, 64> memory{}, control{};
    NVNmemoryPool pool{};
    initialize_pool(pool, memory.data(), memory.size());
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0, memory.size());
    load<Entry::nvnCommandBufferAddControlMemory>()(&commands, control.data(), control.size());
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    const auto self = reinterpret_cast<NVNcommandHandle>(memory.data());
    load<Entry::nvnCommandBufferCallCommands>()(&commands, 1, &self);
    const auto cyclic = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(cyclic == self);
    CommandWalkStatus status;
    CHECK(!prepare_commands(std::span(&cyclic, 1), status) && status == CommandWalkStatus::call_cycle);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, cyclic);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferBindTexturesDeferred>()(&commands, NVN_SHADER_STAGE_FRAGMENT, 128, 0,
                                                       reinterpret_cast<const NVNtextureHandle*>(1));
    auto handle = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    auto submission = prepare_commands(std::span(&handle, 1), status);
    CHECK(submission && status == CommandWalkStatus::complete);
    std::vector<ObservedRecord> records;
    if (submission) CHECK(walk_commands(*submission, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.empty());
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
    load<Entry::nvnCommandBufferBeginRecording>()(&commands);
    load<Entry::nvnCommandBufferBindTexturesDeferred>()(&commands, NVN_SHADER_STAGE_FRAGMENT, 128, 1, nullptr);
    handle = load<Entry::nvnCommandBufferEndRecording>()(&commands);
    CHECK(!prepare_commands(std::span(&handle, 1), status) && status == CommandWalkStatus::invalid_deferred_binding);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&commands);
}

} // namespace

int main() {
    select_api_version({55, 15});
    register_command_entries();

    NVNdevice device{};
    test_record_blocks_and_program_lifetime(device);
    test_submission_lifetime(device);
    test_compact_bindings(device);
    test_submission_validation(device);
    NVNcommandBuffer command_buffer{};
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&command_buffer, &device) == 1);
    CHECK(load<Entry::nvnCommandBufferIsRecording>()(&command_buffer) == 0);

    auto& pool = *reinterpret_cast<MemoryPoolState*>(&callback_pool);
    pool.memory = reinterpret_cast<std::uint8_t*>(callback_command_memory.data());
    pool.size = callback_command_memory.size();
    alignas(8) std::array<std::byte, 16> first_command_memory{};
    NVNmemoryPool first_pool{};
    auto& first_pool_state = *reinterpret_cast<MemoryPoolState*>(&first_pool);
    first_pool_state.memory = reinterpret_cast<std::uint8_t*>(first_command_memory.data());
    first_pool_state.size = first_command_memory.size();

    int callback_cookie = 0;
    load<Entry::nvnCommandBufferSetMemoryCallback>()(&command_buffer, memory_callback);
    load<Entry::nvnCommandBufferSetMemoryCallbackData>()(&command_buffer, &callback_cookie);
    CHECK(load<Entry::nvnCommandBufferGetMemoryCallback>()(&command_buffer) == memory_callback);
    CHECK(load<Entry::nvnCommandBufferGetMemoryCallbackData>()(&command_buffer) == &callback_cookie);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&command_buffer, &first_pool, 0, first_command_memory.size());
    load<Entry::nvnCommandBufferAddControlMemory>()(&command_buffer, nullptr, 0);

    TextureState mutable_texture{};
    TextureState immutable_texture{};
    immutable_texture.serial = 42;
    mutable_texture.snapshot = &immutable_texture;
    NVNtexture texture{};
    std::memcpy(&texture, &mutable_texture, sizeof(mutable_texture));

    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &device) == 1);
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 0, nullptr) == 1);
    const auto* immutable_program =
        static_cast<const ProgramState*>(reinterpret_cast<const ProgramState*>(&program)->snapshot);
    CHECK(immutable_program != nullptr);

    std::array<NVNtextureHandle, 40> deferred_handles{};
    for (std::size_t i = 0; i < deferred_handles.size(); ++i)
        deferred_handles[i] = 100 + i;

    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    CHECK(load<Entry::nvnCommandBufferIsRecording>()(&command_buffer) == 1);
    load<Entry::nvnCommandBufferSetDepthClamp>()(&command_buffer, 0x100u);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&command_buffer, 0.25f, 0x101u, 7, 0xff);
    load<Entry::nvnCommandBufferBindProgram>()(&command_buffer, &program, NVN_SHADER_STAGE_VERTEX_BIT);
    const NVNtexture* colors[]{&texture};
    load<Entry::nvnCommandBufferSetRenderTargets>()(&command_buffer, 1, colors, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindTexturesDeferred>()(&command_buffer, NVN_SHADER_STAGE_FRAGMENT, 3,
                                                        int(deferred_handles.size()), deferred_handles.data());
    load<Entry::nvnCommandBufferDrawArraysIndirect>()(&command_buffer, NVN_DRAW_PRIMITIVE_TRIANGLES, 0x12340000);
    load<Entry::nvnCommandBufferDispatchComputeIndirect>()(&command_buffer, 0x98760000);
    std::array<std::byte, 4> uniform_data{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&command_buffer, 0x1000, 16, 4, uniform_data.size(),
                                                       uniform_data.data());
    const auto recorded = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
    load<Entry::nvnProgramFinalize>()(&program);
    CHECK(recorded != 0);
    CHECK(load<Entry::nvnCommandBufferIsRecording>()(&command_buffer) == 0);
    CHECK(command_callbacks == 1);
    // Eight commands, one native word each: four fill the 16-byte first
    // segment, the other four go to the segment the callback added. The game's
    // memory itself holds no records.
    const std::size_t expected_used = 4 * 4;
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryUsed>()(&command_buffer) == expected_used);
    CHECK(load<Entry::nvnCommandBufferGetCommandMemoryFree>()(&command_buffer) ==
        callback_command_memory.size() - expected_used);
    CHECK(std::all_of(first_command_memory.begin(), first_command_memory.end(),
                      [](std::byte value) { return value == std::byte{}; }));

    deferred_handles[0] = 0xfeed;
    deferred_handles[39] = 0xbeef;
    std::vector<ObservedRecord> records;
    CHECK(walk_commands(recorded, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 9);
    CHECK(records[0].op == Op::depth_clamp && payload<DepthClamp>(records[0]).enable == 0);
    CHECK(records[1].op == Op::clear_depth_stencil && payload<ClearDepthStencil>(records[1]).depth_mask == 1);
    CHECK(records[2].op == Op::bind_program && payload<BindProgram>(records[2]).program == immutable_program);
    CHECK(payload<BindProgram>(records[2]).program->serial != 0);
    CHECK(records[3].op == Op::set_render_targets);
    CHECK(payload<SetRenderTargets>(records[3]).color[0] == &immutable_texture);
    CHECK(payload<SetRenderTargets>(records[3]).color[0]->serial == immutable_texture.serial);
    CHECK(records[4].op == Op::bind_textures && payload<BindHandles>(records[4]).count == 32);
    CHECK(payload<BindHandles>(records[4]).handles[0] == 0xfeed);
    CHECK(records[5].op == Op::bind_textures && payload<BindHandles>(records[5]).count == 8);
    CHECK(payload<BindHandles>(records[5]).handles[7] == 0xbeef);
    CHECK(records[6].op == Op::draw_indirect);
    CHECK(records[7].op == Op::dispatch_indirect);
    CHECK(records[8].op == Op::update_uniforms);

    // Uniform data is copied at record time, not retained by pointer.
    records.clear();
    CHECK(walk_commands(recorded, collect, &records) == CommandWalkStatus::complete);
    const auto& uniform = payload<UpdateUniforms>(records.back());
    CHECK(uniform.destination == 0x1004 && uniform.size == uniform_data.size());
    CHECK(std::memcmp(uniform.data, uniform_data.data(), uniform_data.size()) == 0);

    test_command_memory_accounting(device);
    test_reused_memory_releases_lists(device);
    test_command_memory_past_pool_end(device);
    test_sampler_stamps(device);
    test_control_memory_failure(device, recorded);
    test_call_depth_limit(device);
    test_donated_segments_and_copy(device);

    const auto child = record_draw(command_buffer, 4);
    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferCallCommands>()(&command_buffer, 1, &child);
    const auto caller = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
    CHECK(control_callbacks == 1);
    CHECK(load<Entry::nvnCommandBufferGetControlMemorySize>()(&command_buffer) == callback_control_memory.size());
    CHECK(load<Entry::nvnCommandBufferGetControlMemoryUsed>()(&command_buffer) == sizeof(NVNcommandHandle));
    records.clear();
    CHECK(walk_commands(caller, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && records[0].op == Op::draw && payload<Draw>(records[0]).first == 4);

    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferCopyCommands>()(&command_buffer, 1, &child);
    const auto copied = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, child);
    CHECK(walk_commands(caller, collect, &records) == CommandWalkStatus::invalid_handle);
    records.clear();
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 1 && payload<Draw>(records[0]).first == 4);

    load<Entry::nvnCommandBufferBeginRecording>()(&command_buffer);
    load<Entry::nvnCommandBufferSetPrimitiveRestart>()(&command_buffer, 0x100u, -1);
    load<Entry::nvnCommandBufferSetPrimitiveRestart>()(&command_buffer, 0x101u, 0x1234);
    load<Entry::nvnCommandBufferSetCopyRowStride>()(&command_buffer, 96);
    load<Entry::nvnCommandBufferSetCopyImageStride>()(&command_buffer, 4096);
    NVNdrawTextureRegion draw_destination{}, draw_source{};
    load<Entry::nvnCommandBufferDrawTexture>()(&command_buffer, 9, &draw_destination, &draw_source);
    const auto strides = load<Entry::nvnCommandBufferEndRecording>()(&command_buffer);
    CHECK(load<Entry::nvnCommandBufferGetCopyRowStride>()(&command_buffer) == 96);
    CHECK(load<Entry::nvnCommandBufferGetCopyImageStride>()(&command_buffer) == 4096);
    records.clear();
    CHECK(walk_commands(strides, collect, &records) == CommandWalkStatus::complete);
    CHECK(records.size() == 5 && records[4].op == Op::unsupported);
    CHECK(records[0].op == Op::primitive_restart && payload<PrimitiveRestart>(records[0]).enable == 0);
    CHECK(payload<PrimitiveRestart>(records[0]).index == UINT32_MAX);
    CHECK(payload<PrimitiveRestart>(records[1]).enable == 1 && payload<PrimitiveRestart>(records[1]).index == 0x1234);
    CHECK(payload<Unsupported>(records[4]).entry == static_cast<std::uint32_t>(Entry::nvnCommandBufferDrawTexture));

    load<Entry::nvnCommandBufferFinalize>()(&command_buffer);
    records.clear();
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::complete);
    records.clear();
    CHECK(walk_commands(recorded, collect, &records) == CommandWalkStatus::complete);
    CHECK(payload<BindProgram>(records[2]).program->serial != 0);
    NVNdevice other_device{};
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&other_device, copied);
    records.clear();
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::complete);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, copied);
    CHECK(walk_commands(copied, collect, &records) == CommandWalkStatus::invalid_handle);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, caller);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, recorded);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, strides);

    CHECK(command_walk_status_name(CommandWalkStatus::invalid_handle) == "invalid handle");
    std::printf("NVN command recording failures: %u\n", failures);
    return failures ? 1 : 0;
}
