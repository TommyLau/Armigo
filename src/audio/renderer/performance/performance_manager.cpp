// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/behavior/behavior_info.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

void PerformanceManager::create_impl(const size_t version) {
    switch (version) {
    case 1:
        impl = std::make_unique<
            PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                                   PerformanceEntryVersion1, PerformanceDetailVersion1>>();
        break;
    case 2:
        impl = std::make_unique<
            PerformanceManagerImpl<PerformanceVersion::Version2, PerformanceFrameHeaderVersion2,
                                   PerformanceEntryVersion2, PerformanceDetailVersion2>>();
        break;
    default:
        AUDIO_LOG_WARNING("Invalid PerformanceMetricsDataFormat {}, creating version 1",
                    static_cast<u32>(version));
        impl = std::make_unique<
            PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                                   PerformanceEntryVersion1, PerformanceDetailVersion1>>();
        break;
    }
}

void PerformanceManager::initialize(std::span<u8> workbuffer, const u64 workbuffer_size,
                                    const AudioRendererParameterInternal& params,
                                    const BehaviorInfo& behavior,
                                    const MemoryPoolInfo& memory_pool) {
    create_impl(behavior.get_performance_metrics_data_format());
    impl->initialize(workbuffer, workbuffer_size, params, behavior, memory_pool);
}

bool PerformanceManager::is_initialized() const {
    if (impl) {
        return impl->is_initialized();
    }
    return false;
}

u32 PerformanceManager::copy_histories(u8* out_buffer, u64 out_size) {
    if (impl) {
        return impl->copy_histories(out_buffer, out_size);
    }
    return 0;
}

bool PerformanceManager::get_next_entry(PerformanceEntryAddresses& addresses, u32** unk,
                                      const PerformanceSysDetailType sys_detail_type,
                                      const s32 node_id) {
    if (impl) {
        return impl->get_next_entry(addresses, unk, sys_detail_type, node_id);
    }
    return false;
}

bool PerformanceManager::get_next_entry(PerformanceEntryAddresses& addresses,
                                      const PerformanceEntryType entry_type, const s32 node_id) {
    if (impl) {
        return impl->get_next_entry(addresses, entry_type, node_id);
    }
    return false;
}

bool PerformanceManager::get_next_entry(PerformanceEntryAddresses& addresses,
                                      const PerformanceDetailType detail_type,
                                      const PerformanceEntryType entry_type, const s32 node_id) {
    if (impl) {
        return impl->get_next_entry(addresses, detail_type, entry_type, node_id);
    }
    return false;
}

void PerformanceManager::tap_frame(const bool dsp_behind, const u32 voices_dropped,
                                  const u64 rendering_start_tick) {
    if (impl) {
        impl->tap_frame(dsp_behind, voices_dropped, rendering_start_tick);
    }
}

bool PerformanceManager::is_detail_target(const u32 target_node_id) const {
    if (impl) {
        return impl->is_detail_target(target_node_id);
    }
    return false;
}

void PerformanceManager::set_detail_target(const u32 target_node_id) {
    if (impl) {
        impl->set_detail_target(target_node_id);
    }
}

template <>
void PerformanceManagerImpl<
    PerformanceVersion::Version1, PerformanceFrameHeaderVersion1, PerformanceEntryVersion1,
    PerformanceDetailVersion1>::initialize(std::span<u8> workbuffer_, const u64 workbuffer_size,
                                           const AudioRendererParameterInternal& params,
                                           const BehaviorInfo& behavior,
                                           const MemoryPoolInfo& memory_pool) {
    workbuffer = workbuffer_;
    entries_per_frame = params.voices + params.effects + params.sinks + params.sub_mixes + 1;
    max_detail_count = MaxDetailEntries;
    frame_size = get_required_buffer_size_for_performance_metrics_per_frame(behavior, params);
    const auto frame_count{static_cast<u32>(workbuffer_size / frame_size)};
    max_frames = frame_count - 1;
    translated_buffer = memory_pool.translate(CpuAddr(workbuffer.data()), workbuffer_size);

    // The first frame is the "current" frame we're writing to.
    auto buffer_offset{workbuffer.data()};
    frame_header = reinterpret_cast<PerformanceFrameHeaderVersion1*>(buffer_offset);
    buffer_offset += sizeof(PerformanceFrameHeaderVersion1);
    entry_buffer = {reinterpret_cast<PerformanceEntryVersion1*>(buffer_offset), entries_per_frame};
    buffer_offset += entries_per_frame * sizeof(PerformanceEntryVersion1);
    detail_buffer = {reinterpret_cast<PerformanceDetailVersion1*>(buffer_offset), max_detail_count};

    // After the current, is a ringbuffer of history frames, the current frame will be copied here
    // before a new frame is written.
    frame_history = std::span<u8>(workbuffer.data() + frame_size, workbuffer_size - frame_size);

    // If there's room for any history frames.
    if (frame_count >= 2) {
        buffer_offset = frame_history.data();
        frame_history_header = reinterpret_cast<PerformanceFrameHeaderVersion1*>(buffer_offset);
        buffer_offset += sizeof(PerformanceFrameHeaderVersion1);
        frame_history_entries = {reinterpret_cast<PerformanceEntryVersion1*>(buffer_offset),
                                 entries_per_frame};
        buffer_offset += entries_per_frame * sizeof(PerformanceEntryVersion1);
        frame_history_details = {reinterpret_cast<PerformanceDetailVersion1*>(buffer_offset),
                                 max_detail_count};
    } else {
        frame_history_header = {};
        frame_history_entries = {};
        frame_history_details = {};
    }

    target_node_id = 0;
    version = PerformanceVersion(behavior.get_performance_metrics_data_format());
    entry_count = 0;
    detail_count = 0;
    frame_header->entry_count = 0;
    frame_header->detail_count = 0;
    output_frame_index = 0;
    last_output_frame_index = 0;
    initialized = true;
}

template <>
bool PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                            PerformanceEntryVersion1, PerformanceDetailVersion1>::is_initialized()
    const {
    return initialized;
}

template <>
u32 PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                           PerformanceEntryVersion1,
                           PerformanceDetailVersion1>::copy_histories(u8* out_buffer, u64 out_size) {
    if (out_buffer == nullptr || out_size == 0 || !initialized) {
        return 0;
    }

    // Are there any new frames waiting to be output?
    if (last_output_frame_index == output_frame_index) {
        return 0;
    }

    PerformanceFrameHeaderVersion1* out_header{nullptr};
    u32 out_history_size{0};

    while (last_output_frame_index != output_frame_index) {
        PerformanceFrameHeaderVersion1* history_header{nullptr};
        std::span<PerformanceEntryVersion1> history_entries{};
        std::span<PerformanceDetailVersion1> history_details{};

        if (max_frames > 0) {
            auto frame_offset{&frame_history[last_output_frame_index * frame_size]};
            history_header = reinterpret_cast<PerformanceFrameHeaderVersion1*>(frame_offset);
            frame_offset += sizeof(PerformanceFrameHeaderVersion1);
            history_entries = {reinterpret_cast<PerformanceEntryVersion1*>(frame_offset),
                               history_header->entry_count};
            frame_offset += entries_per_frame * sizeof(PerformanceFrameHeaderVersion1);
            history_details = {reinterpret_cast<PerformanceDetailVersion1*>(frame_offset),
                               history_header->detail_count};
        } else {
            // Original code does not break here, but will crash when trying to dereference the
            // header in the next if, so let's just skip this frame and continue...
            // Hopefully this will not happen.
            AUDIO_LOG_WARNING("max_frames should not be 0! Skipping frame to avoid a crash");
            last_output_frame_index++;
            continue;
        }

        if (out_size < history_header->entry_count * sizeof(PerformanceEntryVersion1) +
                           history_header->detail_count * sizeof(PerformanceDetailVersion1) +
                           2 * sizeof(PerformanceFrameHeaderVersion1)) {
            break;
        }

        u32 out_offset{sizeof(PerformanceFrameHeaderVersion1)};
        auto out_entries{std::span<PerformanceEntryVersion1>(
            reinterpret_cast<PerformanceEntryVersion1*>(out_buffer + out_offset),
            history_header->entry_count)};
        u32 out_entry_count{0};
        u32 total_processing_time{0};
        for (auto& history_entry : history_entries) {
            if (history_entry.processed_time > 0 || history_entry.start_time > 0) {
                out_entries[out_entry_count++] = history_entry;
                total_processing_time += history_entry.processed_time;
            }
        }

        out_offset += static_cast<u32>(out_entry_count * sizeof(PerformanceEntryVersion1));
        auto out_details{std::span<PerformanceDetailVersion1>(
            reinterpret_cast<PerformanceDetailVersion1*>(out_buffer + out_offset),
            history_header->detail_count)};
        u32 out_detail_count{0};
        for (auto& history_detail : history_details) {
            if (history_detail.processed_time > 0 || history_detail.start_time > 0) {
                out_details[out_detail_count++] = history_detail;
            }
        }

        out_offset += static_cast<u32>(out_detail_count * sizeof(PerformanceDetailVersion1));
        out_header = reinterpret_cast<PerformanceFrameHeaderVersion1*>(out_buffer);
        out_header->magic = make_magic('P', 'E', 'R', 'F');
        out_header->entry_count = out_entry_count;
        out_header->detail_count = out_detail_count;
        out_header->next_offset = out_offset;
        out_header->total_processing_time = total_processing_time;
        out_header->frame_index = history_header->frame_index;

        out_history_size += out_offset;

        out_buffer += out_offset;
        out_size -= out_offset;
        last_output_frame_index = (last_output_frame_index + 1) % max_frames;
    }

    // We're out of frames to output, so if there's enough left in the output buffer for another
    // header, and we output at least 1 frame, set the next header to null.
    if (out_size > sizeof(PerformanceFrameHeaderVersion1) && out_header != nullptr) {
        std::memset(out_buffer, 0, sizeof(PerformanceFrameHeaderVersion1));
    }

    return out_history_size;
}

template <>
bool PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                            PerformanceEntryVersion1, PerformanceDetailVersion1>::
    get_next_entry([[maybe_unused]] PerformanceEntryAddresses& addresses, [[maybe_unused]] u32** unk,
                 [[maybe_unused]] PerformanceSysDetailType sys_detail_type,
                 [[maybe_unused]] s32 node_id) {
    return false;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version1, PerformanceFrameHeaderVersion1, PerformanceEntryVersion1,
    PerformanceDetailVersion1>::get_next_entry(PerformanceEntryAddresses& addresses,
                                             const PerformanceEntryType entry_type,
                                             const s32 node_id) {
    if (!initialized) {
        return false;
    }

    addresses.translated_address = translated_buffer;
    addresses.header_entry_count_offset = CpuAddr(frame_header) - CpuAddr(workbuffer.data()) +
                                          offsetof(PerformanceFrameHeaderVersion1, entry_count);

    auto entry{&entry_buffer[entry_count++]};
    addresses.entry_start_time_offset = CpuAddr(entry) - CpuAddr(workbuffer.data()) +
                                        offsetof(PerformanceEntryVersion1, start_time);
    addresses.entry_processed_time_offset = CpuAddr(entry) - CpuAddr(workbuffer.data()) +
                                            offsetof(PerformanceEntryVersion1, processed_time);

    std::memset(entry, 0, sizeof(PerformanceEntryVersion1));
    entry->node_id = node_id;
    entry->entry_type = entry_type;
    return true;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version1, PerformanceFrameHeaderVersion1, PerformanceEntryVersion1,
    PerformanceDetailVersion1>::get_next_entry(PerformanceEntryAddresses& addresses,
                                             const PerformanceDetailType detail_type,
                                             const PerformanceEntryType entry_type,
                                             const s32 node_id) {
    if (!initialized || detail_count > MaxDetailEntries) {
        return false;
    }

    auto detail{&detail_buffer[detail_count++]};

    addresses.translated_address = translated_buffer;
    addresses.header_entry_count_offset = CpuAddr(frame_header) - CpuAddr(workbuffer.data()) +
                                          offsetof(PerformanceFrameHeaderVersion1, detail_count);
    addresses.entry_start_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                        offsetof(PerformanceDetailVersion1, start_time);
    addresses.entry_processed_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                            offsetof(PerformanceDetailVersion1, processed_time);

    std::memset(detail, 0, sizeof(PerformanceDetailVersion1));
    detail->node_id = node_id;
    detail->entry_type = entry_type;
    detail->detail_type = detail_type;
    return true;
}

template <>
void PerformanceManagerImpl<
    PerformanceVersion::Version1, PerformanceFrameHeaderVersion1, PerformanceEntryVersion1,
    PerformanceDetailVersion1>::tap_frame([[maybe_unused]] bool dsp_behind,
                                         [[maybe_unused]] u32 voices_dropped,
                                         [[maybe_unused]] u64 rendering_start_tick) {
    if (!initialized) {
        return;
    }

    if (max_frames > 0) {
        if (!frame_history.empty() && !workbuffer.empty()) {
            auto history_frame = reinterpret_cast<PerformanceFrameHeaderVersion1*>(
                &frame_history[output_frame_index * frame_size]);
            std::memcpy(history_frame, workbuffer.data(), frame_size);
            history_frame->frame_index = history_frame_index++;
        }
        output_frame_index = (output_frame_index + 1) % max_frames;
    }

    entry_count = 0;
    detail_count = 0;
    frame_header->entry_count = 0;
    frame_header->detail_count = 0;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version1, PerformanceFrameHeaderVersion1, PerformanceEntryVersion1,
    PerformanceDetailVersion1>::is_detail_target(const u32 target_node_id_) const {
    return target_node_id == target_node_id_;
}

template <>
void PerformanceManagerImpl<PerformanceVersion::Version1, PerformanceFrameHeaderVersion1,
                            PerformanceEntryVersion1,
                            PerformanceDetailVersion1>::set_detail_target(const u32 target_node_id_) {
    target_node_id = target_node_id_;
}

template <>
void PerformanceManagerImpl<
    PerformanceVersion::Version2, PerformanceFrameHeaderVersion2, PerformanceEntryVersion2,
    PerformanceDetailVersion2>::initialize(std::span<u8> workbuffer_, const u64 workbuffer_size,
                                           const AudioRendererParameterInternal& params,
                                           const BehaviorInfo& behavior,
                                           const MemoryPoolInfo& memory_pool) {
    workbuffer = workbuffer_;
    entries_per_frame = params.voices + params.effects + params.sinks + params.sub_mixes + 1;
    max_detail_count = MaxDetailEntries;
    frame_size = get_required_buffer_size_for_performance_metrics_per_frame(behavior, params);
    const auto frame_count{static_cast<u32>(workbuffer_size / frame_size)};
    max_frames = frame_count - 1;
    translated_buffer = memory_pool.translate(CpuAddr(workbuffer.data()), workbuffer_size);

    // The first frame is the "current" frame we're writing to.
    auto buffer_offset{workbuffer.data()};
    frame_header = reinterpret_cast<PerformanceFrameHeaderVersion2*>(buffer_offset);
    buffer_offset += sizeof(PerformanceFrameHeaderVersion2);
    entry_buffer = {reinterpret_cast<PerformanceEntryVersion2*>(buffer_offset), entries_per_frame};
    buffer_offset += entries_per_frame * sizeof(PerformanceEntryVersion2);
    detail_buffer = {reinterpret_cast<PerformanceDetailVersion2*>(buffer_offset), max_detail_count};

    // After the current, is a ringbuffer of history frames, the current frame will be copied here
    // before a new frame is written.
    frame_history = std::span<u8>(workbuffer.data() + frame_size, workbuffer_size - frame_size);

    // If there's room for any history frames.
    if (frame_count >= 2) {
        buffer_offset = frame_history.data();
        frame_history_header = reinterpret_cast<PerformanceFrameHeaderVersion2*>(buffer_offset);
        buffer_offset += sizeof(PerformanceFrameHeaderVersion2);
        frame_history_entries = {reinterpret_cast<PerformanceEntryVersion2*>(buffer_offset),
                                 entries_per_frame};
        buffer_offset += entries_per_frame * sizeof(PerformanceEntryVersion2);
        frame_history_details = {reinterpret_cast<PerformanceDetailVersion2*>(buffer_offset),
                                 max_detail_count};
    } else {
        frame_history_header = {};
        frame_history_entries = {};
        frame_history_details = {};
    }

    target_node_id = 0;
    version = PerformanceVersion(behavior.get_performance_metrics_data_format());
    entry_count = 0;
    detail_count = 0;
    frame_header->entry_count = 0;
    frame_header->detail_count = 0;
    output_frame_index = 0;
    last_output_frame_index = 0;
    initialized = true;
}

template <>
bool PerformanceManagerImpl<PerformanceVersion::Version2, PerformanceFrameHeaderVersion2,
                            PerformanceEntryVersion2, PerformanceDetailVersion2>::is_initialized()
    const {
    return initialized;
}

template <>
u32 PerformanceManagerImpl<PerformanceVersion::Version2, PerformanceFrameHeaderVersion2,
                           PerformanceEntryVersion2,
                           PerformanceDetailVersion2>::copy_histories(u8* out_buffer, u64 out_size) {
    if (out_buffer == nullptr || out_size == 0 || !initialized) {
        return 0;
    }

    // Are there any new frames waiting to be output?
    if (last_output_frame_index == output_frame_index) {
        return 0;
    }

    PerformanceFrameHeaderVersion2* out_header{nullptr};
    u32 out_history_size{0};

    while (last_output_frame_index != output_frame_index) {
        PerformanceFrameHeaderVersion2* history_header{nullptr};
        std::span<PerformanceEntryVersion2> history_entries{};
        std::span<PerformanceDetailVersion2> history_details{};

        if (max_frames > 0) {
            auto frame_offset{&frame_history[last_output_frame_index * frame_size]};
            history_header = reinterpret_cast<PerformanceFrameHeaderVersion2*>(frame_offset);
            frame_offset += sizeof(PerformanceFrameHeaderVersion2);
            history_entries = {reinterpret_cast<PerformanceEntryVersion2*>(frame_offset),
                               history_header->entry_count};
            frame_offset += entries_per_frame * sizeof(PerformanceFrameHeaderVersion2);
            history_details = {reinterpret_cast<PerformanceDetailVersion2*>(frame_offset),
                               history_header->detail_count};
        } else {
            // Original code does not break here, but will crash when trying to dereference the
            // header in the next if, so let's just skip this frame and continue...
            // Hopefully this will not happen.
            AUDIO_LOG_WARNING("max_frames should not be 0! Skipping frame to avoid a crash");
            last_output_frame_index++;
            continue;
        }

        if (out_size < history_header->entry_count * sizeof(PerformanceEntryVersion2) +
                           history_header->detail_count * sizeof(PerformanceDetailVersion2) +
                           2 * sizeof(PerformanceFrameHeaderVersion2)) {
            break;
        }

        u32 out_offset{sizeof(PerformanceFrameHeaderVersion2)};
        auto out_entries{std::span<PerformanceEntryVersion2>(
            reinterpret_cast<PerformanceEntryVersion2*>(out_buffer + out_offset),
            history_header->entry_count)};
        u32 out_entry_count{0};
        u32 total_processing_time{0};
        for (auto& history_entry : history_entries) {
            if (history_entry.processed_time > 0 || history_entry.start_time > 0) {
                out_entries[out_entry_count++] = history_entry;
                total_processing_time += history_entry.processed_time;
            }
        }

        out_offset += static_cast<u32>(out_entry_count * sizeof(PerformanceEntryVersion2));
        auto out_details{std::span<PerformanceDetailVersion2>(
            reinterpret_cast<PerformanceDetailVersion2*>(out_buffer + out_offset),
            history_header->detail_count)};
        u32 out_detail_count{0};
        for (auto& history_detail : history_details) {
            if (history_detail.processed_time > 0 || history_detail.start_time > 0) {
                out_details[out_detail_count++] = history_detail;
            }
        }

        out_offset += static_cast<u32>(out_detail_count * sizeof(PerformanceDetailVersion2));
        out_header = reinterpret_cast<PerformanceFrameHeaderVersion2*>(out_buffer);
        out_header->magic = make_magic('P', 'E', 'R', 'F');
        out_header->entry_count = out_entry_count;
        out_header->detail_count = out_detail_count;
        out_header->next_offset = out_offset;
        out_header->total_processing_time = total_processing_time;
        out_header->voices_dropped = history_header->voices_dropped;
        out_header->start_time = history_header->start_time;
        out_header->frame_index = history_header->frame_index;
        out_header->render_time_exceeded = history_header->render_time_exceeded;

        out_history_size += out_offset;

        out_buffer += out_offset;
        out_size -= out_offset;
        last_output_frame_index = (last_output_frame_index + 1) % max_frames;
    }

    // We're out of frames to output, so if there's enough left in the output buffer for another
    // header, and we output at least 1 frame, set the next header to null.
    if (out_size > sizeof(PerformanceFrameHeaderVersion2) && out_header != nullptr) {
        std::memset(out_buffer, 0, sizeof(PerformanceFrameHeaderVersion2));
    }

    return out_history_size;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version2, PerformanceFrameHeaderVersion2, PerformanceEntryVersion2,
    PerformanceDetailVersion2>::get_next_entry(PerformanceEntryAddresses& addresses, u32** unk,
                                             const PerformanceSysDetailType sys_detail_type,
                                             const s32 node_id) {
    if (!initialized || detail_count > MaxDetailEntries) {
        return false;
    }

    auto detail{&detail_buffer[detail_count++]};

    addresses.translated_address = translated_buffer;
    addresses.header_entry_count_offset = CpuAddr(frame_header) - CpuAddr(workbuffer.data()) +
                                          offsetof(PerformanceFrameHeaderVersion2, detail_count);
    addresses.entry_start_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                        offsetof(PerformanceDetailVersion2, start_time);
    addresses.entry_processed_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                            offsetof(PerformanceDetailVersion2, processed_time);

    std::memset(detail, 0, sizeof(PerformanceDetailVersion2));
    detail->node_id = node_id;
    detail->detail_type = static_cast<PerformanceDetailType>(sys_detail_type);

    if (unk) {
        *unk = &detail->unk_10;
    }
    return true;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version2, PerformanceFrameHeaderVersion2, PerformanceEntryVersion2,
    PerformanceDetailVersion2>::get_next_entry(PerformanceEntryAddresses& addresses,
                                             const PerformanceEntryType entry_type,
                                             const s32 node_id) {
    if (!initialized) {
        return false;
    }

    auto entry{&entry_buffer[entry_count++]};

    addresses.translated_address = translated_buffer;
    addresses.header_entry_count_offset = CpuAddr(frame_header) - CpuAddr(workbuffer.data()) +
                                          offsetof(PerformanceFrameHeaderVersion2, entry_count);
    addresses.entry_start_time_offset = CpuAddr(entry) - CpuAddr(workbuffer.data()) +
                                        offsetof(PerformanceEntryVersion2, start_time);
    addresses.entry_processed_time_offset = CpuAddr(entry) - CpuAddr(workbuffer.data()) +
                                            offsetof(PerformanceEntryVersion2, processed_time);

    std::memset(entry, 0, sizeof(PerformanceEntryVersion2));
    entry->node_id = node_id;
    entry->entry_type = entry_type;
    return true;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version2, PerformanceFrameHeaderVersion2, PerformanceEntryVersion2,
    PerformanceDetailVersion2>::get_next_entry(PerformanceEntryAddresses& addresses,
                                             const PerformanceDetailType detail_type,
                                             const PerformanceEntryType entry_type,
                                             const s32 node_id) {
    if (!initialized || detail_count > MaxDetailEntries) {
        return false;
    }

    auto detail{&detail_buffer[detail_count++]};

    addresses.translated_address = translated_buffer;
    addresses.header_entry_count_offset = CpuAddr(frame_header) - CpuAddr(workbuffer.data()) +
                                          offsetof(PerformanceFrameHeaderVersion2, detail_count);
    addresses.entry_start_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                        offsetof(PerformanceDetailVersion2, start_time);
    addresses.entry_processed_time_offset = CpuAddr(detail) - CpuAddr(workbuffer.data()) +
                                            offsetof(PerformanceDetailVersion2, processed_time);

    std::memset(detail, 0, sizeof(PerformanceDetailVersion2));
    detail->node_id = node_id;
    detail->entry_type = entry_type;
    detail->detail_type = detail_type;
    return true;
}

template <>
void PerformanceManagerImpl<PerformanceVersion::Version2, PerformanceFrameHeaderVersion2,
                            PerformanceEntryVersion2,
                            PerformanceDetailVersion2>::tap_frame(const bool dsp_behind,
                                                                 const u32 voices_dropped,
                                                                 const u64 rendering_start_tick) {
    if (!initialized) {
        return;
    }

    if (max_frames > 0) {
        if (!frame_history.empty() && !workbuffer.empty()) {
            auto history_frame{reinterpret_cast<PerformanceFrameHeaderVersion2*>(
                &frame_history[output_frame_index * frame_size])};
            std::memcpy(history_frame, workbuffer.data(), frame_size);
            history_frame->render_time_exceeded = dsp_behind;
            history_frame->voices_dropped = voices_dropped;
            history_frame->start_time = rendering_start_tick;
            history_frame->frame_index = history_frame_index++;
        }
        output_frame_index = (output_frame_index + 1) % max_frames;
    }

    entry_count = 0;
    detail_count = 0;
    frame_header->entry_count = 0;
    frame_header->detail_count = 0;
}

template <>
bool PerformanceManagerImpl<
    PerformanceVersion::Version2, PerformanceFrameHeaderVersion2, PerformanceEntryVersion2,
    PerformanceDetailVersion2>::is_detail_target(const u32 target_node_id_) const {
    return target_node_id == target_node_id_;
}

template <>
void PerformanceManagerImpl<PerformanceVersion::Version2, PerformanceFrameHeaderVersion2,
                            PerformanceEntryVersion2,
                            PerformanceDetailVersion2>::set_detail_target(const u32 target_node_id_) {
    target_node_id = target_node_id_;
}

} // namespace armigo::audio::renderer
