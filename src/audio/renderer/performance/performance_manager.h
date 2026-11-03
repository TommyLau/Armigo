// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <chrono>
#include <memory>
#include <span>

#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/performance/performance_detail.h"
#include "renderer/performance/performance_entry.h"
#include "renderer/performance/performance_entry_addresses.h"
#include "renderer/performance/performance_frame_header.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class BehaviorInfo;
class MemoryPoolInfo;

enum class PerformanceVersion {
    Version1,
    Version2,
};

enum class PerformanceSysDetailType {
    PcmInt16 = 15,
    PcmFloat = 16,
    Adpcm = 17,
    LightLimiter = 37,
};

enum class PerformanceState {
    Invalid,
    Start,
    Stop,
};

// Manages performance information.
//
// The performance buffer is split into frames, each comprised of:
//     Frame header - Information about the number of entries/details and some others
//     Entries      - Created when starting to generate types of commands, such as voice
// commands, mix commands, sink commands etc. Details      - Created for specific commands
// within each group. Up to MaxDetailEntries per frame.
//
// A current frame is written to by the AudioRenderer, and before it processes the next command
// list, the current frame is copied to a ringbuffer of history frames. These frames are then
// output back to the game if it supplies a performance buffer to RequestUpdate.
//
// Two versions currently exist, version 2 adds a few extra fields to the header, and a new
// SysDetail type which is seemingly unused.
class PerformanceManager {
public:
    static constexpr size_t MaxDetailEntries = 100;

    struct InParameter {
        /* 0x00 */ s32 target_node_id;
        /* 0x04 */ char unk04[0xC];
    };
    static_assert(sizeof(InParameter) == 0x10,
                  "PerformanceManager::InParameter has the wrong size!");

    struct OutStatus {
        /* 0x00 */ s32 history_size;
        /* 0x04 */ char unk04[0xC];
    };
    static_assert(sizeof(OutStatus) == 0x10, "PerformanceManager::OutStatus has the wrong size!");

    // Calculate the required size for the performance workbuffer.
    static u64 get_required_buffer_size_for_performance_metrics_per_frame(
        const BehaviorInfo& behavior, const AudioRendererParameterInternal& params) {
        u64 entry_count{params.voices + params.effects + params.sub_mixes + params.sinks + 1};
        switch (behavior.get_performance_metrics_data_format()) {
        case 1:
            return sizeof(PerformanceFrameHeaderVersion1) +
                   PerformanceManager::MaxDetailEntries * sizeof(PerformanceDetailVersion1) +
                   entry_count * sizeof(PerformanceEntryVersion1);
        case 2:
            return sizeof(PerformanceFrameHeaderVersion2) +
                   PerformanceManager::MaxDetailEntries * sizeof(PerformanceDetailVersion2) +
                   entry_count * sizeof(PerformanceEntryVersion2);
        }

        AUDIO_LOG_WARNING("Invalid PerformanceMetrics version, assuming version 1");
        return sizeof(PerformanceFrameHeaderVersion1) +
               PerformanceManager::MaxDetailEntries * sizeof(PerformanceDetailVersion1) +
               entry_count * sizeof(PerformanceEntryVersion1);
    }

    virtual ~PerformanceManager() = default;

    // Initialize the performance manager.
    virtual void initialize(std::span<u8> workbuffer, u64 workbuffer_size,
                            const AudioRendererParameterInternal& params,
                            const BehaviorInfo& behavior, const MemoryPoolInfo& memory_pool);

    // Check if the manager is initialized.
    virtual bool is_initialized() const;

    // Copy the waiting performance frames to the output buffer.
    virtual u32 copy_histories(u8* out_buffer, u64 out_size);

    // Setup a new sys detail in the current frame, filling in addresses with offsets to the
    // current workbuffer, to be written by the AudioRenderer. Note: This version is
    // unused/incomplete.
    //
    // the AudioRenderer with Performance commands to be written.
    virtual bool get_next_entry(PerformanceEntryAddresses& addresses, u32** unk,
                              PerformanceSysDetailType sys_detail_type, s32 node_id);

    // Setup a new entry in the current frame, filling in addresses with offsets to the current
    // workbuffer, to be written by the AudioRenderer.
    //
    // the AudioRenderer with Performance commands to be written.
    virtual bool get_next_entry(PerformanceEntryAddresses& addresses, PerformanceEntryType entry_type,
                              s32 node_id);

    // Setup a new detail in the current frame, filling in addresses with offsets to the current
    // workbuffer, to be written by the AudioRenderer.
    virtual bool get_next_entry(PerformanceEntryAddresses& addresses,
                              PerformanceDetailType detail_type, PerformanceEntryType entry_type,
                              s32 node_id);

    // Save the current frame to the ring buffer.
    virtual void tap_frame(bool dsp_behind, u32 voices_dropped, u64 rendering_start_tick);

    // Check if the node id is a detail type.
    virtual bool is_detail_target(u32 target_node_id) const;

    // Set the given node to be a detail type.
    virtual void set_detail_target(u32 target_node_id);

private:
    // Create the performance manager.
    void create_impl(size_t version);

    std::unique_ptr<PerformanceManager>
        // Impl for the performance manager, may be version 1 or 2.
        impl;
};

template <PerformanceVersion Version, typename FrameHeaderVersion, typename EntryVersion,
          typename DetailVersion>
class PerformanceManagerImpl : public PerformanceManager {
public:
    void initialize(std::span<u8> workbuffer, u64 workbuffer_size,
                    const AudioRendererParameterInternal& params, const BehaviorInfo& behavior,
                    const MemoryPoolInfo& memory_pool) override;
    bool is_initialized() const override;
    u32 copy_histories(u8* out_buffer, u64 out_size) override;
    bool get_next_entry(PerformanceEntryAddresses& addresses, u32** unk,
                      PerformanceSysDetailType sys_detail_type, s32 node_id) override;
    bool get_next_entry(PerformanceEntryAddresses& addresses, PerformanceEntryType entry_type,
                      s32 node_id) override;
    bool get_next_entry(PerformanceEntryAddresses& addresses, PerformanceDetailType detail_type,
                      PerformanceEntryType entry_type, s32 node_id) override;
    void tap_frame(bool dsp_behind, u32 voices_dropped, u64 rendering_start_tick) override;
    bool is_detail_target(u32 target_node_id) const override;
    void set_detail_target(u32 target_node_id) override;

private:
    // Workbuffer used to store the current performance frame
    std::span<u8> workbuffer{};
    // DSP address of the workbuffer, used by the AudioRenderer
    CpuAddr translated_buffer{};
    // Current frame index
    u32 history_frame_index{};
    // Current frame header
    FrameHeaderVersion* frame_header{};
    // Current frame entry buffer
    std::span<EntryVersion> entry_buffer{};
    // Current frame detail buffer
    std::span<DetailVersion> detail_buffer{};
    // Current frame entry count
    u32 entry_count{};
    // Current frame detail count
    u32 detail_count{};
    // Ringbuffer of previous frames
    std::span<u8> frame_history{};
    // Current history frame header
    FrameHeaderVersion* frame_history_header{};
    // Current history entry buffer
    std::span<EntryVersion> frame_history_entries{};
    // Current history detail buffer
    std::span<DetailVersion> frame_history_details{};
    // Current history ringbuffer write index
    u32 output_frame_index{};
    // Last history frame index that was written back to the game
    u32 last_output_frame_index{};
    // Maximum number of history frames in the ringbuffer
    u32 max_frames{};
    // Number of entries per frame
    u32 entries_per_frame{};
    // Maximum number of details per frame
    u32 max_detail_count{};
    // Frame size in bytes
    u64 frame_size{};
    // Is the performance manager initialized?
    bool initialized{};
    // Target node id
    u32 target_node_id{};
    // Performance version in use
    PerformanceVersion version{};
};

} // namespace armigo::audio::renderer
