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

#include "renderer/command/commands.h"
#include "renderer/effect/light_limiter.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
struct UpsamplerInfo;
struct VoiceState;
class EffectInfoBase;
class CommandProcessingTimeEstimator;
class MixInfo;
class MemoryPoolInfo;
class SinkInfoBase;
class VoiceInfo;

// Utility functions to generate and add commands into the current command list.
class CommandBuffer {
public:
    // Generate a PCM s16 version 1 command, adding it to the command list.
    void generate_pcm_int16_version1_command(s32 node_id, const MemoryPoolInfo& memory_pool,
                                         VoiceInfo& voice_info, const VoiceState& voice_state,
                                         s16 buffer_count, s8 channel);

    // Generate a PCM s16 version 2 command, adding it to the command list.
    void generate_pcm_int16_version2_command(s32 node_id, VoiceInfo& voice_info,
                                         const VoiceState& voice_state, s16 buffer_count,
                                         s8 channel);

    // Generate a PCM f32 version 1 command, adding it to the command list.
    void generate_pcm_float_version1_command(s32 node_id, const MemoryPoolInfo& memory_pool,
                                         VoiceInfo& voice_info, const VoiceState& voice_state,
                                         s16 buffer_count, s8 channel);

    // Generate a PCM f32 version 2 command, adding it to the command list.
    void generate_pcm_float_version2_command(s32 node_id, VoiceInfo& voice_info,
                                         const VoiceState& voice_state, s16 buffer_count,
                                         s8 channel);

    // Generate an ADPCM version 1 command, adding it to the command list.
    void generate_adpcm_version1_command(s32 node_id, const MemoryPoolInfo& memory_pool,
                                      VoiceInfo& voice_info, const VoiceState& voice_state,
                                      s16 buffer_count, s8 channel);

    // Generate an ADPCM version 2 command, adding it to the command list.
    void generate_adpcm_version2_command(s32 node_id, VoiceInfo& voice_info,
                                      const VoiceState& voice_state, s16 buffer_count, s8 channel);

    // Generate a volume command, adding it to the command list.
    void generate_volume_command(s32 node_id, s16 buffer_offset, s16 input_index, f32 volume,
                               u8 precision);

    // Generate a volume ramp command, adding it to the command list.
    void generate_volume_ramp_command(s32 node_id, VoiceInfo& voice_info, s16 buffer_count,
                                   u8 precision);

    // Generate a biquad filter command from a voice, adding it to the command list.
    void generate_biquad_filter_command(s32 node_id, VoiceInfo& voice_info,
                                     const VoiceState& voice_state, s16 buffer_count, s8 channel,
                                     u32 biquad_index, bool use_float_processing);

    // Generate a biquad filter effect command, adding it to the command list.
    void generate_biquad_filter_command(s32 node_id, EffectInfoBase& effect_info, s16 buffer_offset,
                                     s8 channel, bool needs_init, bool use_float_processing);

    // Generate a mix command, adding it to the command list.
    void generate_mix_command(s32 node_id, s16 input_index, s16 output_index, s16 buffer_offset,
                            f32 volume, u8 precision);

    // Generate a mix ramp command, adding it to the command list.
    void generate_mix_ramp_command(s32 node_id, s16 buffer_count, s16 input_index, s16 output_index,
                                f32 volume, f32 prev_volume, CpuAddr prev_samples, u8 precision);

    // Generate a mix ramp grouped command, adding it to the command list.
    void generate_mix_ramp_grouped_command(s32 node_id, s16 buffer_count, s16 input_index,
                                       s16 output_index, std::span<const f32> volumes,
                                       std::span<const f32> prev_volumes, CpuAddr prev_samples,
                                       u8 precision);

    // Generate a depop prepare command, adding it to the command list.
    void generate_depop_prepare_command(s32 node_id, const VoiceState& voice_state,
                                     std::span<const s32> buffer, s16 buffer_count,
                                     s16 buffer_offset, bool was_playing);

    // Generate a depop command, adding it to the command list.
    void generate_depop_for_mix_buffers_command(s32 node_id, const MixInfo& mix_info,
                                           std::span<const s32> depop_buffer);

    // Generate a delay command, adding it to the command list.
    void generate_delay_command(s32 node_id, EffectInfoBase& effect_info, s16 buffer_offset);

    // Generate an upsample command, adding it to the command list.
    void generate_upsample_command(s32 node_id, s16 buffer_offset, UpsamplerInfo& upsampler_info,
                                 u32 input_count, std::span<const s8> inputs, s16 buffer_count,
                                 u32 sample_count, u32 sample_rate);

    // Generate a downmix 6 -> 2 command, adding it to the command list.
    void generate_down_mix6ch_to2ch_command(s32 node_id, std::span<const s8> inputs, s16 buffer_offset,
                                        std::span<const f32> downmix_coeff);

    // Generate an aux buffer command, adding it to the command list.
    void generate_aux_command(s32 node_id, EffectInfoBase& effect_info, s16 input_index,
                            s16 output_index, s16 buffer_offset, u32 update_count, u32 count_max,
                            u32 write_offset);

    // Generate a device sink command, adding it to the command list.
    void generate_device_sink_command(s32 node_id, s16 buffer_offset, SinkInfoBase& sink_info,
                                   u32 session_id, std::span<s32> samples_buffer);

    // Generate a circular buffer sink command, adding it to the command list.
    void generate_circular_buffer_sink_command(s32 node_id, SinkInfoBase& sink_info, s16 buffer_offset);

    // Generate a reverb command, adding it to the command list.
    void generate_reverb_command(s32 node_id, EffectInfoBase& effect_info, s16 buffer_offset,
                               bool long_size_pre_delay_supported);

    // Generate an I3DL2 reverb command, adding it to the command list.
    void generate_i3dl2_reverb_command(s32 node_id, EffectInfoBase& effect_info, s16 buffer_offset);

    // Generate a performance command, adding it to the command list.
    void generate_performance_command(s32 node_id, PerformanceState state,
                                    const PerformanceEntryAddresses& entry_addresses);

    // Generate a clear mix command, adding it to the command list.
    void generate_clear_mix_command(s32 node_id);

    // Generate a copy mix command, adding it to the command list.
    void generate_copy_mix_buffer_command(s32 node_id, EffectInfoBase& effect_info, s16 buffer_offset,
                                      s8 channel);

    // Generate a light limiter version 1 command, adding it to the command list.
    void generate_light_limiter_command(s32 node_id, s16 buffer_offset,
                                     const LightLimiterInfo::ParameterVersion1& parameter,
                                     const LightLimiterInfo::State& state, bool enabled,
                                     CpuAddr workbuffer);

    // Generate a light limiter version 2 command, adding it to the command list.
    void generate_light_limiter_command(s32 node_id, s16 buffer_offset,
                                     const LightLimiterInfo::ParameterVersion2& parameter,
                                     const LightLimiterInfo::StatisticsInternal& statistics,
                                     const LightLimiterInfo::State& state, bool enabled,
                                     CpuAddr workbuffer);

    // Generate a multitap biquad filter command, adding it to the command list.
    void generate_multitap_biquad_filter_command(s32 node_id, VoiceInfo& voice_info,
                                             const VoiceState& voice_state, s16 buffer_count,
                                             s8 channel);

    // Generate a capture command, adding it to the command list.
    void generate_capture_command(s32 node_id, EffectInfoBase& effect_info, s16 input_index,
                                s16 output_index, s16 buffer_offset, u32 update_count,
                                u32 count_max, u32 write_offset);

    // Generate a compressor command, adding it to the command list.
    void generate_compressor_command(s16 buffer_offset, EffectInfoBase& effect_info, s32 node_id);

    // Command list buffer generated commands will be added to
    std::span<u8> command_list{};
    // Input sample count, unused
    u32 sample_count{};
    // Input sample rate, unused
    u32 sample_rate{};
    // Current size of the command buffer
    u64 size{};
    // Current number of commands added
    u32 count{};
    // Current estimated processing time for all commands
    u32 estimated_process_time{};
    // Used for mapping buffers for the AudioRenderer
    MemoryPoolInfo* memory_pool{};
    // Used for estimating command process times
    CommandProcessingTimeEstimator* time_estimator{};
    // Used to check which rendering features are currently enabled
    BehaviorInfo* behavior{};
    // Set when a command did not fit and was left out
    bool overflowed{};

private:
    template <typename T, CommandId Id>
    T& generate_start(const s32 node_id);
    template <typename T>
    void generate_end(T& cmd);
};

} // namespace armigo::audio::renderer
