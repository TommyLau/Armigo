// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <ranges>

#include "renderer/voice/voice_context.h"
#include <ranges>

namespace armigo::audio::renderer {

VoiceState& VoiceContext::get_dsp_shared_state(const u32 index) {
    if (index >= dsp_states.size()) {
        AUDIO_LOG_ERROR("Invalid voice dsp state index {:04X}", index);
    }
    return dsp_states[index];
}

VoiceChannelResource& VoiceContext::get_channel_resource(const u32 index) {
    if (index >= channel_resources.size()) {
        AUDIO_LOG_ERROR("Invalid voice channel resource index {:04X}", index);
    }
    return channel_resources[index];
}

void VoiceContext::initialize(std::span<VoiceInfo*> sorted_voice_infos_,
                              std::span<VoiceInfo> voice_infos_,
                              std::span<VoiceChannelResource> voice_channel_resources_,
                              std::span<VoiceState> cpu_states_, std::span<VoiceState> dsp_states_,
                              const u32 voice_count_) {
    sorted_voice_info = sorted_voice_infos_;
    voices = voice_infos_;
    channel_resources = voice_channel_resources_;
    cpu_states = cpu_states_;
    dsp_states = dsp_states_;
    voice_count = voice_count_;
    active_count = 0;
}

VoiceInfo* VoiceContext::get_sorted_info(const u32 index) {
    if (index >= sorted_voice_info.size()) {
        AUDIO_LOG_ERROR("Invalid voice sorted info index {:04X}", index);
    }
    return sorted_voice_info[index];
}

VoiceInfo& VoiceContext::get_info(const u32 index) {
    if (index >= voices.size()) {
        AUDIO_LOG_ERROR("Invalid voice info index {:04X}", index);
    }
    return voices[index];
}

VoiceState& VoiceContext::get_state(const u32 index) {
    if (index >= cpu_states.size()) {
        AUDIO_LOG_ERROR("Invalid voice cpu state index {:04X}", index);
    }
    return cpu_states[index];
}

u32 VoiceContext::get_count() const {
    return voice_count;
}

u32 VoiceContext::get_active_count() const {
    return active_count;
}

void VoiceContext::set_active_count(const u32 active_count_) {
    active_count = active_count_;
}

void VoiceContext::sort_info() {
    for (u32 i = 0; i < voice_count; i++) {
        sorted_voice_info[i] = &voices[i];
    }

    std::ranges::sort(sorted_voice_info, [](const VoiceInfo* a, const VoiceInfo* b) {
        return a->priority != b->priority ? a->priority > b->priority
                                          : a->sort_order > b->sort_order;
    });
}

void VoiceContext::update_state_by_dsp_shared() {
    std::memcpy(cpu_states.data(), dsp_states.data(), voice_count * sizeof(VoiceState));
}

} // namespace armigo::audio::renderer
