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

#include <span>

#include "renderer/splitter/splitter_destinations_data.h"
#include "renderer/splitter/splitter_info.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
struct AudioRendererParameterInternal;
class WorkbufferAllocator;

class BehaviorInfo;

// The splitter allows much more control over how sound is mixed together.
// Previously, one mix can only connect to one other, and you may need
// more mixes (and duplicate processing) to achieve the same result.
// With the splitter, many-to-one and one-to-many mixing is possible.
// This was added in revision 2.
// Had a bug with incorrect numbers of destinations, fixed in revision 5.
class SplitterContext {
    struct InParameterHeader {
        /* 0x00 */ u32 magic; // 'SNDH'
        /* 0x04 */ s32 info_count;
        /* 0x08 */ s32 destination_count;
        /* 0x0C */ char unk0C[0x14];
    };
    static_assert(sizeof(InParameterHeader) == 0x20,
                  "SplitterContext::InParameterHeader has the wrong size!");

public:
    // Get a destination mix from the given splitter and destination index.
    SplitterDestinationData* get_destination_data(s32 splitter_id, s32 destination_id);

    // Get a splitter from the given index.
    SplitterInfo& get_info(s32 index);

    // Get the total number of splitter destinations.
    u32 get_data_count() const;

    // Get the total number of splitters.
    u32 get_info_count() const;

    // Get a specific global destination.
    SplitterDestinationData& get_data(u32 index);

    // Check if the splitter is in use.
    bool using_splitter() const;

    // Mark all splitters as having new connections.
    void clear_all_new_connection_flag();

    // Initialize the context.
    bool initialize(const BehaviorInfo& behavior, const AudioRendererParameterInternal& params,
                    WorkbufferAllocator& allocator);

    // Update the context.
    bool update(const u8* input, u64 size);

    // Update the splitters.
    u32 update_info(const u8* input, u64 size, u32 offset, u32 splitter_count);

    // Update the splitters.
    u32 update_data(const u8* input, u64 size, u32 offset, u32 destination_count);

    // Update the state of all destinations in all splitters.
    void update_internal_state();

    // Replace the given splitter's destinations with new ones.
    void recompose_destination(SplitterInfo& out_info, const SplitterInfo::InParameter* info_header);

    // Old calculation for destinations, this is the thing the splitter bug fixes.
    // Left for compatibility, and now min'd with the actual count to not bug.
    u32 get_dest_count_per_info_for_compat() const;

    // Calculate the size of the required workbuffer for splitters and destinations.
    static u64 calc_work_buffer_size(const BehaviorInfo& behavior,
                                  const AudioRendererParameterInternal& params);

private:
    // Setup the context.
    void setup(std::span<SplitterInfo> splitter_infos, u32 splitter_info_count,
               SplitterDestinationData* splitter_destinations, u32 destination_count,
               bool splitter_bug_fixed, const BehaviorInfo& behavior);

    // Workbuffer for splitters
    std::span<SplitterInfo> splitter_infos{};
    // Number of splitters in buffer
    s32 info_count{};
    // Workbuffer for destinations
    SplitterDestinationData* splitter_destinations{};
    // Number of destinations in buffer
    s32 destinations_count{};
    // Is the splitter bug fixed?
    bool splitter_bug_fixed{};
    // Is explicit previous mix volume reset supported?
    bool splitter_prev_volume_reset_supported{};
    // Is biquad filter parameter for splitter (REV12) supported?
    bool splitter_biquad_param_supported{};
    // Is float coefficient/biquad filter v2b parameter supported?
    bool splitter_float_coeff_supported{};
};

} // namespace armigo::audio::renderer
