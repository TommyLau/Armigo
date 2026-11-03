// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/memory/pool_mapper.h"
#include "renderer/voice/voice_context.h"
#include "renderer/voice/voice_info.h"
#include "renderer/voice/voice_state.h"

namespace armigo::audio::renderer {

VoiceInfo::VoiceInfo() {
    initialize();
}

void VoiceInfo::initialize() {
    in_use = false;
    is_new = false;
    id = 0;
    node_id = 0;
    current_play_state = ServerPlayState::Stopped;
    src_quality = SrcQuality::Medium;
    priority = LowestVoicePriority;
    sample_format = SampleFormat::Invalid;
    sample_rate = 0;
    channel_count = 0;
    wave_buffer_count = 0;
    wave_buffer_index = 0;
    pitch = 0.0f;
    volume = 0.0f;
    prev_volume = 0.0f;
    mix_id = UnusedMixId;
    splitter_id = UnusedSplitterId;
    biquads = {};
    biquad_initialized = {};
    voice_dropped = false;
    data_unmapped = false;
    buffer_unmapped = false;
    flush_buffer_count = 0;

    data_address.setup(0, 0);
    for (auto& wavebuffer : wavebuffers) {
        wavebuffer.initialize();
    }
}

bool VoiceInfo::should_update_parameters(const InParameter& params) const {
    return data_address.get_cpu_addr() != params.src_data_address ||
           data_address.get_size() != params.src_data_size || data_unmapped;
}

void VoiceInfo::update_parameters(BehaviorInfo::ErrorInfo& error_info, const InParameter& params,
                                 const PoolMapper& pool_mapper, const BehaviorInfo& behavior) {
    in_use = params.in_use;
    id = params.id;
    node_id = params.node_id;
    update_play_state(params.play_state);
    update_src_quality(params.src_quality);
    priority = params.priority;
    sort_order = params.sort_order;
    sample_rate = params.sample_rate;
    sample_format = params.sample_format;
    channel_count = static_cast<s8>(params.channel_count);
    pitch = params.pitch;
    volume = params.volume;
    biquads = params.biquads;
    wave_buffer_count = params.wave_buffer_count;
    wave_buffer_index = params.wave_buffer_index;

    if (behavior.is_flush_voice_wave_buffers_supported()) {
        flush_buffer_count += params.flush_buffer_count;
    }

    mix_id = params.mix_id;

    if (behavior.is_splitter_supported()) {
        splitter_id = params.splitter_id;
    } else {
        splitter_id = UnusedSplitterId;
    }

    channel_resource_ids = params.channel_resource_ids;

    flags &= u16(~0b11);
    if (behavior.is_voice_played_sample_count_reset_at_loop_point_supported()) {
        flags |= u16(params.flags.IsVoicePlayedSampleCountResetAtLoopPointSupported);
    }

    if (behavior.is_voice_pitch_and_src_skipped_supported()) {
        flags |= u16(u16(params.flags.IsVoicePitchAndSrcSkippedSupported) << 1);
    }

    if (params.clear_voice_drop) {
        voice_dropped = false;
    }

    if (should_update_parameters(params)) {
        data_unmapped = !pool_mapper.try_attach_buffer(error_info, data_address,
                                                     params.src_data_address, params.src_data_size);
    } else {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }
}

void VoiceInfo::update_play_state(const PlayState state) {
    last_play_state = current_play_state;

    switch (state) {
    case PlayState::Started:
        current_play_state = ServerPlayState::Started;
        break;
    case PlayState::Stopped:
        if (current_play_state != ServerPlayState::Stopped) {
            current_play_state = ServerPlayState::RequestStop;
        }
        break;
    case PlayState::Paused:
        current_play_state = ServerPlayState::Paused;
        break;
    default:
        AUDIO_LOG_ERROR("Invalid input play state {}", static_cast<u32>(state));
        break;
    }
}

void VoiceInfo::update_src_quality(const SrcQuality quality) {
    switch (quality) {
    case SrcQuality::Medium:
        src_quality = quality;
        break;
    case SrcQuality::High:
        src_quality = quality;
        break;
    case SrcQuality::Low:
        src_quality = quality;
        break;
    default:
        AUDIO_LOG_ERROR("Invalid input src quality {}", static_cast<u32>(quality));
        break;
    }
}

void VoiceInfo::update_wave_buffers(std::span<std::array<BehaviorInfo::ErrorInfo, 2>> error_infos,
                                  [[maybe_unused]] u32 error_count, const InParameter& params,
                                  std::span<VoiceState*> voice_states,
                                  const PoolMapper& pool_mapper, const BehaviorInfo& behavior) {
    if (params.is_new) {
        for (size_t i = 0; i < wavebuffers.size(); i++) {
            wavebuffers[i].initialize();
        }

        for (s8 channel = 0; channel < static_cast<s8>(params.channel_count); channel++) {
            voice_states[channel]->wave_buffer_valid.fill(false);
        }
    }

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        update_wave_buffer(error_infos[i], wavebuffers[i], params.wave_buffer_internal[i],
                         params.sample_format, voice_states[0]->wave_buffer_valid[i], pool_mapper,
                         behavior);
    }
}

void VoiceInfo::update_wave_buffer(std::span<BehaviorInfo::ErrorInfo> error_info,
                                 WaveBuffer& wave_buffer,
                                 const WaveBufferInternal& wave_buffer_internal,
                                 const SampleFormat sample_format_, const bool valid,
                                 const PoolMapper& pool_mapper, const BehaviorInfo& behavior) {
    if (!valid && wave_buffer.sent_to_DSP && wave_buffer.buffer_address.get_cpu_addr() != 0) {
        pool_mapper.force_unmap_pointer(wave_buffer.buffer_address);
        wave_buffer.buffer_address.setup(0, 0);
    }

    if (!should_update_wave_buffer(wave_buffer_internal)) {
        return;
    }

    switch (sample_format_) {
    case SampleFormat::PcmInt16: {
        constexpr auto byte_size{get_sample_format_byte_size(SampleFormat::PcmInt16)};
        if (wave_buffer_internal.start_offset * byte_size > wave_buffer_internal.size ||
            wave_buffer_internal.end_offset * byte_size > wave_buffer_internal.size) {
            AUDIO_LOG_ERROR("Invalid PCM16 start/end wavebuffer sizes!");
            error_info[0].error_code = ResultInvalidUpdateInfo;
            error_info[0].address = wave_buffer_internal.address;
            return;
        }
    } break;

    case SampleFormat::PcmFloat: {
        constexpr auto byte_size{get_sample_format_byte_size(SampleFormat::PcmFloat)};
        if (wave_buffer_internal.start_offset * byte_size > wave_buffer_internal.size ||
            wave_buffer_internal.end_offset * byte_size > wave_buffer_internal.size) {
            AUDIO_LOG_ERROR("Invalid PCMFloat start/end wavebuffer sizes!");
            error_info[0].error_code = ResultInvalidUpdateInfo;
            error_info[0].address = wave_buffer_internal.address;
            return;
        }
    } break;

    case SampleFormat::Adpcm: {
        const auto start_frame{wave_buffer_internal.start_offset / 14};
        auto start_extra{wave_buffer_internal.start_offset % 14 == 0
                             ? 0
                             : (wave_buffer_internal.start_offset % 14) / 2 + 1 +
                                   ((wave_buffer_internal.start_offset % 14) % 2)};
        const auto start{start_frame * 8 + start_extra};

        const auto end_frame{wave_buffer_internal.end_offset / 14};
        const auto end_extra{wave_buffer_internal.end_offset % 14 == 0
                                 ? 0
                                 : (wave_buffer_internal.end_offset % 14) / 2 + 1 +
                                       ((wave_buffer_internal.end_offset % 14) % 2)};
        const auto end{end_frame * 8 + end_extra};

        if (start > static_cast<s64>(wave_buffer_internal.size) ||
            end > static_cast<s64>(wave_buffer_internal.size)) {
            AUDIO_LOG_ERROR("Invalid ADPCM start/end wavebuffer sizes!");
            error_info[0].error_code = ResultInvalidUpdateInfo;
            error_info[0].address = wave_buffer_internal.address;
            return;
        }
    } break;

    default:
        break;
    }

    if (wave_buffer_internal.start_offset < 0 || wave_buffer_internal.end_offset < 0) {
        AUDIO_LOG_ERROR("Invalid input start/end wavebuffer sizes!");
        error_info[0].error_code = ResultInvalidUpdateInfo;
        error_info[0].address = wave_buffer_internal.address;
        return;
    }

    wave_buffer.start_offset = wave_buffer_internal.start_offset;
    wave_buffer.end_offset = wave_buffer_internal.end_offset;
    wave_buffer.loop = wave_buffer_internal.loop;
    wave_buffer.stream_ended = wave_buffer_internal.stream_ended;
    wave_buffer.sent_to_DSP = false;
    wave_buffer.loop_start_offset = wave_buffer_internal.loop_start;
    wave_buffer.loop_end_offset = wave_buffer_internal.loop_end;
    wave_buffer.loop_count = wave_buffer_internal.loop_count;

    buffer_unmapped =
        !pool_mapper.try_attach_buffer(error_info[0], wave_buffer.buffer_address,
                                     wave_buffer_internal.address, wave_buffer_internal.size);

    if (sample_format_ == SampleFormat::Adpcm && behavior.is_adpcm_loop_context_bug_fixed() &&
        wave_buffer_internal.context_address != 0) {
        buffer_unmapped = !pool_mapper.try_attach_buffer(error_info[1], wave_buffer.context_address,
                                                       wave_buffer_internal.context_address,
                                                       wave_buffer_internal.context_size) ||
                          data_unmapped;
    } else {
        wave_buffer.context_address.setup(0, 0);
    }
}

bool VoiceInfo::should_update_wave_buffer(const WaveBufferInternal& wave_buffer_internal) const {
    return !wave_buffer_internal.sent_to_DSP || buffer_unmapped;
}

void VoiceInfo::write_out_status(OutStatus& out_status, const InParameter& params,
                               std::span<VoiceState*> voice_states) {
    if (params.is_new) {
        is_new = true;
    }

    if (params.is_new || is_new) {
        out_status.played_sample_count = 0;
        out_status.wave_buffers_consumed = 0;
        out_status.voice_dropped = false;
    } else {
        out_status.played_sample_count = voice_states[0]->played_sample_count;
        out_status.wave_buffers_consumed = voice_states[0]->wave_buffers_consumed;
        out_status.voice_dropped = voice_dropped;
    }
}

bool VoiceInfo::should_skip() const {
    return !in_use || wave_buffer_count == 0 || data_unmapped || buffer_unmapped || voice_dropped;
}

bool VoiceInfo::has_any_connection() const {
    return mix_id != UnusedMixId || splitter_id != UnusedSplitterId;
}

void VoiceInfo::flush_wave_buffers(const u32 flush_count, std::span<VoiceState*> voice_states,
                                 const s8 channel_count_) {
    auto wave_index{wave_buffer_index};

    for (size_t i = 0; i < flush_count; i++) {
        wavebuffers[wave_index].sent_to_DSP = true;

        for (s8 j = 0; j < channel_count_; j++) {
            auto voice_state{voice_states[j]};
            if (voice_state->wave_buffer_index == wave_index) {
                voice_state->wave_buffer_index =
                    (voice_state->wave_buffer_index + 1) % MaxWaveBuffers;
                voice_state->wave_buffers_consumed++;
            }
            voice_state->wave_buffer_valid[wave_index] = false;
        }

        wave_index = (wave_index + 1) % MaxWaveBuffers;
    }
}

bool VoiceInfo::update_parameters_for_command_generation(std::span<VoiceState*> voice_states) {
    if (flush_buffer_count > 0) {
        flush_wave_buffers(flush_buffer_count, voice_states, channel_count);
        flush_buffer_count = 0;
    }

    switch (current_play_state) {
    case ServerPlayState::Started:
        for (u32 i = 0; i < MaxWaveBuffers; i++) {
            if (!wavebuffers[i].sent_to_DSP) {
                for (s8 channel = 0; channel < channel_count; channel++) {
                    voice_states[channel]->wave_buffer_valid[i] = true;
                }
                wavebuffers[i].sent_to_DSP = true;
            }
        }

        was_playing = false;

        for (u32 i = 0; i < MaxWaveBuffers; i++) {
            if (voice_states[0]->wave_buffer_valid[i]) {
                return true;
            }
        }
        break;

    case ServerPlayState::Stopped:
    case ServerPlayState::Paused:
        for (auto& wavebuffer : wavebuffers) {
            if (!wavebuffer.sent_to_DSP) {
                wavebuffer.buffer_address.get_reference(true);
                wavebuffer.context_address.get_reference(true);
            }
        }

        if (sample_format == SampleFormat::Adpcm && data_address.get_cpu_addr() != 0) {
            data_address.get_reference(true);
        }

        was_playing = last_play_state == ServerPlayState::Started;
        break;

    case ServerPlayState::RequestStop:
        for (u32 i = 0; i < MaxWaveBuffers; i++) {
            wavebuffers[i].sent_to_DSP = true;

            for (s8 channel = 0; channel < channel_count; channel++) {
                if (voice_states[channel]->wave_buffer_valid[i]) {
                    voice_states[channel]->wave_buffer_index =
                        (voice_states[channel]->wave_buffer_index + 1) % MaxWaveBuffers;
                    voice_states[channel]->wave_buffers_consumed++;
                }
                voice_states[channel]->wave_buffer_valid[i] = false;
            }
        }

        for (s8 channel = 0; channel < channel_count; channel++) {
            voice_states[channel]->offset = 0;
            voice_states[channel]->played_sample_count = 0;
            voice_states[channel]->adpcm_context = {};
            voice_states[channel]->sample_history.fill(0);
            voice_states[channel]->fraction = 0;
        }

        current_play_state = ServerPlayState::Stopped;
        was_playing = last_play_state == ServerPlayState::Started;
        break;
    }

    return was_playing;
}

bool VoiceInfo::update_for_command_generation(VoiceContext& voice_context) {
    std::array<VoiceState*, MaxChannels> voice_states{};

    if (is_new) {
        reset_resources(voice_context);
        prev_volume = volume;
        is_new = false;
    }

    for (s8 channel = 0; channel < channel_count; channel++) {
        voice_states[channel] = &voice_context.get_dsp_shared_state(channel_resource_ids[channel]);
    }

    return update_parameters_for_command_generation(voice_states);
}

void VoiceInfo::reset_resources(VoiceContext& voice_context) const {
    for (s8 channel = 0; channel < channel_count; channel++) {
        auto& state{voice_context.get_dsp_shared_state(channel_resource_ids[channel])};
        state = {};

        auto& channel_resource{voice_context.get_channel_resource(channel_resource_ids[channel])};
        channel_resource.prev_mix_volumes = channel_resource.mix_volumes;
    }
}

} // namespace armigo::audio::renderer
