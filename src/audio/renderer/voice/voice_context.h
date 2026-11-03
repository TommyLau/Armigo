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

#include "renderer/voice/voice_channel_resource.h"
#include "renderer/voice/voice_info.h"
#include "renderer/voice/voice_state.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Contains all voices, with utility functions for managing them.
class VoiceContext {
public:
    // Get the AudioRenderer state for a given index
    VoiceState& get_dsp_shared_state(u32 index);

    // Get the channel resource for a given index
    VoiceChannelResource& get_channel_resource(u32 index);

    // Initialize the voice context.
    void initialize(std::span<VoiceInfo*> sorted_voice_infos, std::span<VoiceInfo> voice_infos,
                    std::span<VoiceChannelResource> voice_channel_resources,
                    std::span<VoiceState> cpu_states, std::span<VoiceState> dsp_states,
                    u32 voice_count);

    // Get a sorted voice with the given index.
    VoiceInfo* get_sorted_info(u32 index);

    // Get a voice with the given index.
    VoiceInfo& get_info(u32 index);

    // Get a host voice state with the given index.
    VoiceState& get_state(u32 index);

    // Get the maximum number of voices.
    // Not all voices in the buffers may be in use, see GetActiveCount.
    u32 get_count() const;

    // Get the number of active voices.
    // Can be less than or equal to the maximum number of voices.
    u32 get_active_count() const;

    // Set the number of active voices.
    // Can be less than or equal to the maximum number of voices.
    void set_active_count(u32 active_count);

    // Sort all voices. Results are available via GetSortedInfo.
    // Voices are sorted descendingly, according to priority, and then sort order.
    void sort_info();

    // Update all voice states, copying AudioRenderer-side states to host-side states.
    void update_state_by_dsp_shared();

private:
    // Sorted voices
    std::span<VoiceInfo*> sorted_voice_info{};
    // Voices
    std::span<VoiceInfo> voices{};
    // Channel resources
    std::span<VoiceChannelResource> channel_resources{};
    // Host-side voice states
    std::span<VoiceState> cpu_states{};
    // AudioRenderer-side voice states
    std::span<VoiceState> dsp_states{};
    // Maximum number of voices
    u32 voice_count{};
    // Number of active voices
    u32 active_count{};
};

} // namespace armigo::audio::renderer
