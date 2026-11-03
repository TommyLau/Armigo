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
#include <vector>

#include "renderer/common/common.h"
#include "renderer/common/types.h"

namespace armigo::audio {
class PcmOutput;
}

namespace armigo::audio::renderer {

class GuestMemory;
struct CommandListHeader;

// The DSP side of rendering: walks a generated command list and runs each
// enabled command against the list's mix buffers.
class CommandListProcessor {
public:
    // Initialize the processor for the command list at `buffer`.
    void initialize(GuestMemory& memory, CpuAddr buffer, u64 size, PcmOutput* output);

    // Set the maximum processing time for this command list.
    void set_process_time_max(u64 time);

    // Get the remaining command count for this list.
    u32 get_remaining_command_count() const;

    // Get the host output the device sink writes to.
    PcmOutput* get_output() const;

    // Process the command list; returns the time taken in microseconds.
    u64 process(u32 session_id);

    // The mix buffer samples [offset, offset + count). Commands take their
    // buffer indices from guest parameters; a range outside the mix buffers
    // maps to zeroed scratch storage instead of host memory past them.
    std::span<s32> mix_buffer(u64 offset, u64 count) const;

    // Guest memory reachable through attached pools
    GuestMemory* memory{};
    // Host output for the processed samples
    PcmOutput* output{};
    // Header info for this command list
    CommandListHeader* header{};
    // The command buffer
    u8* commands{};
    // The command buffer size
    u64 commands_buffer_size{};
    // The maximum processing time allotted
    u64 max_process_time{};
    // The number of commands in the buffer
    u32 command_count{};
    // The target sample count for output
    u32 sample_count{};
    // The target sample rate for output
    u32 target_sample_rate{};
    // The mixing buffers used by the commands
    std::span<s32> mix_buffers{};
    // The number of mix buffers
    u32 buffer_count{};
    // The number of processed commands so far
    u32 processed_command_count{};
    // The processing start time of this list
    u64 start_time{};
    // The current processing time for this list
    u64 current_processing_time{};
    // The end processing time for this list
    u64 end_time{};
    // Stand-in for out-of-range mix buffer ranges
    mutable std::vector<s32> scratch{};
};

}  // namespace armigo::audio::renderer
