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
#include "renderer/command/command_buffer.h"
#include "renderer/command/command_list_header.h"
#include "renderer/command/command_processing_time_estimator.h"
#include "renderer/effect/biquad_filter.h"
#include "renderer/effect/delay.h"
#include "renderer/effect/reverb.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/mix/mix_info.h"
#include "renderer/sink/circular_buffer_sink_info.h"
#include "renderer/sink/device_sink_info.h"
#include "renderer/sink/sink_info_base.h"
#include "renderer/voice/voice_info.h"
#include "renderer/voice/voice_state.h"

namespace armigo::audio::renderer {

namespace {
constexpr f32 BiquadParameterFixedScaleQ14 = 16384.0f; // 1 << 14

[[nodiscard]] inline s16 to_q14_clamped(f32 v) {
    const f32 scaled = std::clamp(v * BiquadParameterFixedScaleQ14, -32768.0f, 32767.0f);
    return static_cast<s16>(scaled);
}
} // namespace

template <typename T, CommandId Id>
T& CommandBuffer::generate_start(const s32 node_id) {
    // A command that would not fit is built in scratch storage and left out
    // of the list (see generate_end), so the list never overruns its buffer.
    T* storage{reinterpret_cast<T*>(&command_list[size])};
    if (size + sizeof(T) >= command_list.size_bytes()) {
        if (!overflowed) {
            AUDIO_LOG_ERROR("Attempting to write commands beyond the end of allocated command buffer memory!");
        }
        overflowed = true;
        alignas(64) thread_local std::array<u8, 0x800> scratch;
        static_assert(sizeof(T) <= sizeof(scratch));
        storage = reinterpret_cast<T*>(scratch.data());
    }

    auto& cmd{*std::construct_at<T>(storage)};

    cmd.magic = CommandMagic;
    cmd.enabled = true;
    cmd.type = Id;
    cmd.size = sizeof(T);
    cmd.node_id = node_id;

    return cmd;
}

template <typename T>
void CommandBuffer::generate_end(T& cmd) {
    if (reinterpret_cast<u8*>(&cmd) != &command_list[size]) {
        return;
    }
    cmd.estimated_process_time = time_estimator->estimate(cmd);
    estimated_process_time += cmd.estimated_process_time;
    size += sizeof(T);
    count++;
}

void CommandBuffer::generate_pcm_int16_version1_command(const s32 node_id,
                                                    const MemoryPoolInfo& memory_pool_,
                                                    VoiceInfo& voice_info,
                                                    const VoiceState& voice_state,
                                                    const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<PcmInt16DataSourceVersion1Command, CommandId::DataSourcePcmInt16Version1>(
            node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;
    cmd.channel_index = channel;
    cmd.channel_count = voice_info.channel_count;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool_.translate(CpuAddr(&voice_state), sizeof(VoiceState));

    generate_end<PcmInt16DataSourceVersion1Command>(cmd);
}

void CommandBuffer::generate_pcm_int16_version2_command(const s32 node_id, VoiceInfo& voice_info,
                                                    const VoiceState& voice_state,
                                                    const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<PcmInt16DataSourceVersion2Command, CommandId::DataSourcePcmInt16Version2>(
            node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;
    cmd.channel_index = channel;
    cmd.channel_count = voice_info.channel_count;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool->translate(CpuAddr(&voice_state), sizeof(VoiceState));

    generate_end<PcmInt16DataSourceVersion2Command>(cmd);
}

void CommandBuffer::generate_pcm_float_version1_command(const s32 node_id,
                                                    const MemoryPoolInfo& memory_pool_,
                                                    VoiceInfo& voice_info,
                                                    const VoiceState& voice_state,
                                                    const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<PcmFloatDataSourceVersion1Command, CommandId::DataSourcePcmFloatVersion1>(
            node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;
    cmd.channel_index = channel;
    cmd.channel_count = voice_info.channel_count;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool_.translate(CpuAddr(&voice_state), sizeof(VoiceState));

    generate_end<PcmFloatDataSourceVersion1Command>(cmd);
}

void CommandBuffer::generate_pcm_float_version2_command(const s32 node_id, VoiceInfo& voice_info,
                                                    const VoiceState& voice_state,
                                                    const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<PcmFloatDataSourceVersion2Command, CommandId::DataSourcePcmFloatVersion2>(
            node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;
    cmd.channel_index = channel;
    cmd.channel_count = voice_info.channel_count;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool->translate(CpuAddr(&voice_state), sizeof(VoiceState));

    generate_end<PcmFloatDataSourceVersion2Command>(cmd);
}

void CommandBuffer::generate_adpcm_version1_command(const s32 node_id,
                                                 const MemoryPoolInfo& memory_pool_,
                                                 VoiceInfo& voice_info,
                                                 const VoiceState& voice_state,
                                                 const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<AdpcmDataSourceVersion1Command, CommandId::DataSourceAdpcmVersion1>(node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool_.translate(CpuAddr(&voice_state), sizeof(VoiceState));
    cmd.data_address = voice_info.data_address.get_reference(true);
    cmd.data_size = voice_info.data_address.get_size();

    generate_end<AdpcmDataSourceVersion1Command>(cmd);
}

void CommandBuffer::generate_adpcm_version2_command(const s32 node_id, VoiceInfo& voice_info,
                                                 const VoiceState& voice_state,
                                                 const s16 buffer_count, const s8 channel) {
    auto& cmd{
        generate_start<AdpcmDataSourceVersion2Command, CommandId::DataSourceAdpcmVersion2>(node_id)};

    cmd.src_quality = voice_info.src_quality;
    cmd.output_index = buffer_count + channel;
    cmd.flags = voice_info.flags & 3;
    cmd.sample_rate = voice_info.sample_rate;
    cmd.pitch = voice_info.pitch;
    cmd.channel_index = channel;
    cmd.channel_count = voice_info.channel_count;

    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        voice_info.wavebuffers[i].copy(cmd.wave_buffers[i]);
    }

    cmd.voice_state = memory_pool->translate(CpuAddr(&voice_state), sizeof(VoiceState));
    cmd.data_address = voice_info.data_address.get_reference(true);
    cmd.data_size = voice_info.data_address.get_size();

    generate_end<AdpcmDataSourceVersion2Command>(cmd);
}

void CommandBuffer::generate_volume_command(const s32 node_id, const s16 buffer_offset,
                                          const s16 input_index, const f32 volume,
                                          const u8 precision) {
    auto& cmd{generate_start<VolumeCommand, CommandId::Volume>(node_id)};

    cmd.precision = precision;
    cmd.input_index = buffer_offset + input_index;
    cmd.output_index = buffer_offset + input_index;
    cmd.volume = volume;

    generate_end<VolumeCommand>(cmd);
}

void CommandBuffer::generate_volume_ramp_command(const s32 node_id, VoiceInfo& voice_info,
                                              const s16 buffer_count, const u8 precision) {
    auto& cmd{generate_start<VolumeRampCommand, CommandId::VolumeRamp>(node_id)};

    cmd.input_index = buffer_count;
    cmd.output_index = buffer_count;
    cmd.prev_volume = voice_info.prev_volume;
    cmd.volume = voice_info.volume;
    cmd.precision = precision;

    generate_end<VolumeRampCommand>(cmd);
}

void CommandBuffer::generate_biquad_filter_command(const s32 node_id, VoiceInfo& voice_info,
                                                const VoiceState& voice_state,
                                                const s16 buffer_count, const s8 channel,
                                                const u32 biquad_index,
                                                const bool use_float_processing) {
    auto& cmd{generate_start<BiquadFilterCommand, CommandId::BiquadFilter>(node_id)};

    cmd.input = buffer_count + channel;
    cmd.output = buffer_count + channel;

    cmd.biquad = voice_info.biquads[biquad_index];

    if (voice_info.use_float_biquads) {
        cmd.biquad_float = voice_info.biquads_float[biquad_index];
        cmd.use_float_coefficients = true;
    } else {
        cmd.use_float_coefficients = false;
    }

    cmd.state = memory_pool->translate(CpuAddr(voice_state.biquad_states[biquad_index].data()),
                                       MaxBiquadFilters * sizeof(VoiceState::BiquadFilterState));

    cmd.needs_init = !voice_info.biquad_initialized[biquad_index];
    cmd.use_float_processing = use_float_processing;

    generate_end<BiquadFilterCommand>(cmd);
}

void CommandBuffer::generate_biquad_filter_command(const s32 node_id, EffectInfoBase& effect_info,
                                                const s16 buffer_offset, const s8 channel,
                                                const bool needs_init,
                                                const bool use_float_processing) {
    auto& cmd{generate_start<BiquadFilterCommand, CommandId::BiquadFilter>(node_id)};

    const auto state{reinterpret_cast<VoiceState::BiquadFilterState*>(
        effect_info.get_state_buffer() + channel * sizeof(VoiceState::BiquadFilterState))};

    if (behavior->is_effect_info_version2_supported()) {
        const auto& parameter{
            *reinterpret_cast<BiquadFilterInfo::ParameterVersion2*>(effect_info.get_parameter())};

        cmd.input = buffer_offset + parameter.inputs[channel];
        cmd.output = buffer_offset + parameter.outputs[channel];

        // Convert float coefficients to Q2.14 fixed-point as expected by the legacy DSP path.
        cmd.biquad.b[0] = to_q14_clamped(parameter.b[0]);
        cmd.biquad.b[1] = to_q14_clamped(parameter.b[1]);
        cmd.biquad.b[2] = to_q14_clamped(parameter.b[2]);
        cmd.biquad.a[0] = to_q14_clamped(parameter.a[0]);
        cmd.biquad.a[1] = to_q14_clamped(parameter.a[1]);
    } else {
        const auto& parameter{
            *reinterpret_cast<BiquadFilterInfo::ParameterVersion1*>(effect_info.get_parameter())};

        cmd.input = buffer_offset + parameter.inputs[channel];
        cmd.output = buffer_offset + parameter.outputs[channel];

        cmd.biquad.b = parameter.b;
        cmd.biquad.a = parameter.a;
    }

    // Effects always use the fixed-point coefficient path on the DSP.
    cmd.use_float_coefficients = false;

    cmd.state = memory_pool->translate(CpuAddr(state),
                                       MaxBiquadFilters * sizeof(VoiceState::BiquadFilterState));

    cmd.needs_init = needs_init;
    cmd.use_float_processing = use_float_processing;

    generate_end<BiquadFilterCommand>(cmd);
}

void CommandBuffer::generate_mix_command(const s32 node_id, const s16 input_index,
                                       const s16 output_index, [[maybe_unused]] const s16 buffer_offset,
                                       const f32 volume, const u8 precision) {
    auto& cmd{generate_start<MixCommand, CommandId::Mix>(node_id)};

    cmd.input_index = input_index;
    cmd.output_index = output_index;
    cmd.volume = volume;
    cmd.precision = precision;

    generate_end<MixCommand>(cmd);
}

void CommandBuffer::generate_mix_ramp_command(const s32 node_id,
                                           [[maybe_unused]] const s16 buffer_count,
                                           const s16 input_index, const s16 output_index,
                                           const f32 volume, const f32 prev_volume,
                                           const CpuAddr prev_samples, const u8 precision) {
    if (volume == 0.0f && prev_volume == 0.0f) {
        return;
    }

    auto& cmd{generate_start<MixRampCommand, CommandId::MixRamp>(node_id)};

    cmd.input_index = input_index;
    cmd.output_index = output_index;
    cmd.prev_volume = prev_volume;
    cmd.volume = volume;
    cmd.previous_sample = prev_samples;
    cmd.precision = precision;

    generate_end<MixRampCommand>(cmd);
}

void CommandBuffer::generate_mix_ramp_grouped_command(const s32 node_id, const s16 buffer_count,
                                                  const s16 input_index, s16 output_index,
                                                  std::span<const f32> volumes,
                                                  std::span<const f32> prev_volumes,
                                                  const CpuAddr prev_samples, const u8 precision) {
    auto& cmd{generate_start<MixRampGroupedCommand, CommandId::MixRampGrouped>(node_id)};

    cmd.buffer_count = buffer_count;

    for (s32 i = 0; i < buffer_count; i++) {
        cmd.inputs[i] = input_index;
        cmd.outputs[i] = output_index++;
        cmd.prev_volumes[i] = prev_volumes[i];
        cmd.volumes[i] = volumes[i];
    }

    cmd.previous_samples = prev_samples;
    cmd.precision = precision;

    generate_end<MixRampGroupedCommand>(cmd);
}

void CommandBuffer::generate_depop_prepare_command(const s32 node_id, const VoiceState& voice_state,
                                                std::span<const s32> buffer, const s16 buffer_count,
                                                s16 buffer_offset, const bool was_playing) {
    auto& cmd{generate_start<DepopPrepareCommand, CommandId::DepopPrepare>(node_id)};

    cmd.enabled = was_playing;

    for (u32 i = 0; i < MaxMixBuffers; i++) {
        cmd.inputs[i] = buffer_offset++;
    }

    cmd.previous_samples = memory_pool->translate(CpuAddr(voice_state.previous_samples.data()),
                                                  MaxMixBuffers * sizeof(s32));
    cmd.buffer_count = buffer_count;
    cmd.depop_buffer = memory_pool->translate(CpuAddr(buffer.data()), buffer.size_bytes());

    generate_end<DepopPrepareCommand>(cmd);
}

void CommandBuffer::generate_depop_for_mix_buffers_command(const s32 node_id, const MixInfo& mix_info,
                                                      std::span<const s32> depop_buffer) {
    auto& cmd{generate_start<DepopForMixBuffersCommand, CommandId::DepopForMixBuffers>(node_id)};

    cmd.input = mix_info.buffer_offset;
    cmd.count = mix_info.buffer_count;
    cmd.decay = mix_info.sample_rate == TargetSampleRate ? 0.96218872f : 0.94369507f;
    cmd.depop_buffer =
        memory_pool->translate(CpuAddr(depop_buffer.data()), mix_info.buffer_count * sizeof(s32));

    generate_end<DepopForMixBuffersCommand>(cmd);
}

void CommandBuffer::generate_delay_command(const s32 node_id, EffectInfoBase& effect_info,
                                         const s16 buffer_offset) {
    auto& cmd{generate_start<DelayCommand, CommandId::Delay>(node_id)};

    const auto& parameter{
        *reinterpret_cast<DelayInfo::ParameterVersion1*>(effect_info.get_parameter())};
    const auto state{effect_info.get_state_buffer()};

    if (is_channel_count_valid(parameter.channel_count)) {
        const auto state_buffer{memory_pool->translate(CpuAddr(state), sizeof(DelayInfo::State))};
        if (state_buffer) {
            for (s16 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }

            if (!behavior->is_delay_channel_mapping_changed() && parameter.channel_count == 6) {
                use_old_channel_mapping(cmd.inputs, cmd.outputs);
            }

            cmd.parameter = parameter;
            cmd.effect_enabled = effect_info.is_enabled();
            cmd.state = state_buffer;
            cmd.workbuffer = effect_info.get_workbuffer(-1);
        }
    }

    generate_end<DelayCommand>(cmd);
}

void CommandBuffer::generate_upsample_command(const s32 node_id, const s16 buffer_offset,
                                            UpsamplerInfo& upsampler_info, const u32 input_count,
                                            std::span<const s8> inputs, const s16 buffer_count,
                                            const u32 sample_count_, const u32 sample_rate_) {
    auto& cmd{generate_start<UpsampleCommand, CommandId::Upsample>(node_id)};

    cmd.samples_buffer = memory_pool->translate(upsampler_info.samples_pos,
                                                upsampler_info.sample_count * sizeof(s32));
    cmd.inputs = memory_pool->translate(CpuAddr(upsampler_info.inputs.data()), MaxChannels);
    cmd.buffer_count = buffer_count;
    cmd.unk_20 = 0;
    cmd.source_sample_count = sample_count_;
    cmd.source_sample_rate = sample_rate_;

    upsampler_info.input_count = input_count;
    for (u32 i = 0; i < input_count; i++) {
        upsampler_info.inputs[i] = buffer_offset + inputs[i];
    }

    cmd.upsampler_info = memory_pool->translate(CpuAddr(&upsampler_info), sizeof(UpsamplerInfo));

    generate_end<UpsampleCommand>(cmd);
}

void CommandBuffer::generate_down_mix6ch_to2ch_command(const s32 node_id, std::span<const s8> inputs,
                                                   const s16 buffer_offset,
                                                   std::span<const f32> downmix_coeff) {
    auto& cmd{generate_start<DownMix6chTo2chCommand, CommandId::DownMix6chTo2ch>(node_id)};

    for (u32 i = 0; i < MaxChannels; i++) {
        cmd.inputs[i] = buffer_offset + inputs[i];
        cmd.outputs[i] = buffer_offset + inputs[i];
    }

    for (u32 i = 0; i < 4; i++) {
        cmd.down_mix_coeff[i] = downmix_coeff[i];
    }

    generate_end<DownMix6chTo2chCommand>(cmd);
}

void CommandBuffer::generate_aux_command(const s32 node_id, EffectInfoBase& effect_info,
                                       const s16 input_index, const s16 output_index,
                                       const s16 buffer_offset, const u32 update_count,
                                       const u32 count_max, const u32 write_offset) {
    auto& cmd{generate_start<AuxCommand, CommandId::Aux>(node_id)};

    if (effect_info.get_send_buffer() != 0 && effect_info.get_return_buffer() != 0) {
        cmd.input = buffer_offset + input_index;
        cmd.output = buffer_offset + output_index;
        cmd.send_buffer_info = effect_info.get_send_buffer_info();
        cmd.send_buffer = effect_info.get_send_buffer();
        cmd.return_buffer_info = effect_info.get_return_buffer_info();
        cmd.return_buffer = effect_info.get_return_buffer();
        cmd.count_max = count_max;
        cmd.write_offset = write_offset;
        cmd.update_count = update_count;
        cmd.effect_enabled = effect_info.is_enabled();
    }

    generate_end<AuxCommand>(cmd);
}

void CommandBuffer::generate_device_sink_command(const s32 node_id, const s16 buffer_offset,
                                              SinkInfoBase& sink_info, const u32 session_id,
                                              std::span<s32> samples_buffer) {
    auto& cmd{generate_start<DeviceSinkCommand, CommandId::DeviceSink>(node_id)};
    const auto& parameter{
        *reinterpret_cast<DeviceSinkInfo::DeviceInParameter*>(sink_info.get_parameter())};
    auto state{*reinterpret_cast<DeviceSinkInfo::DeviceState*>(sink_info.get_state())};

    cmd.session_id = session_id;

    cmd.input_count = parameter.input_count;
    s16 max_input{0};
    for (u32 i = 0; i < parameter.input_count; i++) {
        cmd.inputs[i] = buffer_offset + parameter.inputs[i];
        max_input = (std::max)(max_input, cmd.inputs[i]);
    }

    if (state.upsampler_info != nullptr) {
        const auto size_{state.upsampler_info->sample_count * parameter.input_count};
        const auto size_bytes{size_ * sizeof(s32)};
        const auto addr{memory_pool->translate(state.upsampler_info->samples_pos, size_bytes)};
        cmd.sample_buffer = {reinterpret_cast<s32*>(addr),
                             (max_input + 1) * state.upsampler_info->sample_count};
    } else {
        cmd.sample_buffer = samples_buffer;
    }

    generate_end<DeviceSinkCommand>(cmd);
}

void CommandBuffer::generate_circular_buffer_sink_command(const s32 node_id, SinkInfoBase& sink_info,
                                                      const s16 buffer_offset) {
    auto& cmd{generate_start<CircularBufferSinkCommand, CommandId::CircularBufferSink>(node_id)};
    const auto& parameter{*reinterpret_cast<CircularBufferSinkInfo::CircularBufferInParameter*>(
        sink_info.get_parameter())};
    auto state{
        *reinterpret_cast<CircularBufferSinkInfo::CircularBufferState*>(sink_info.get_state())};

    cmd.input_count = parameter.input_count;
    for (u32 i = 0; i < parameter.input_count; i++) {
        cmd.inputs[i] = buffer_offset + parameter.inputs[i];
    }

    cmd.address = state.address_info.get_reference(true);
    cmd.size = parameter.size;
    cmd.pos = state.current_pos;

    generate_end<CircularBufferSinkCommand>(cmd);
}

void CommandBuffer::generate_reverb_command(const s32 node_id, EffectInfoBase& effect_info,
                                          const s16 buffer_offset,
                                          const bool long_size_pre_delay_supported) {
    auto& cmd{generate_start<ReverbCommand, CommandId::Reverb>(node_id)};

    const auto& parameter{
        *reinterpret_cast<ReverbInfo::ParameterVersion2*>(effect_info.get_parameter())};
    const auto state{effect_info.get_state_buffer()};

    if (is_channel_count_valid(parameter.channel_count)) {
        const auto state_buffer{memory_pool->translate(CpuAddr(state), sizeof(ReverbInfo::State))};
        if (state_buffer) {
            for (s16 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }

            if (!behavior->is_reverb_channel_mapping_changed() && parameter.channel_count == 6) {
                use_old_channel_mapping(cmd.inputs, cmd.outputs);
            }

            cmd.parameter = parameter;
            cmd.effect_enabled = effect_info.is_enabled();
            cmd.state = state_buffer;
            cmd.workbuffer = effect_info.get_workbuffer(-1);
            cmd.long_size_pre_delay_supported = long_size_pre_delay_supported;
        }
    }

    generate_end<ReverbCommand>(cmd);
}

void CommandBuffer::generate_i3dl2_reverb_command(const s32 node_id, EffectInfoBase& effect_info,
                                               const s16 buffer_offset) {
    auto& cmd{generate_start<I3dl2ReverbCommand, CommandId::I3dl2Reverb>(node_id)};

    const auto& parameter{
        *reinterpret_cast<I3dl2ReverbInfo::ParameterVersion1*>(effect_info.get_parameter())};
    const auto state{effect_info.get_state_buffer()};

    if (is_channel_count_valid(parameter.channel_count)) {
        const auto state_buffer{
            memory_pool->translate(CpuAddr(state), sizeof(I3dl2ReverbInfo::State))};
        if (state_buffer) {
            for (s16 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }

            if (!behavior->is_i3dl2_reverb_channel_mapping_changed() && parameter.channel_count == 6) {
                use_old_channel_mapping(cmd.inputs, cmd.outputs);
            }

            cmd.parameter = parameter;
            cmd.effect_enabled = effect_info.is_enabled();
            cmd.state = state_buffer;
            cmd.workbuffer = effect_info.get_workbuffer(-1);
        }
    }

    generate_end<I3dl2ReverbCommand>(cmd);
}

void CommandBuffer::generate_performance_command(const s32 node_id, const PerformanceState state,
                                               const PerformanceEntryAddresses& entry_addresses) {
    auto& cmd{generate_start<PerformanceCommand, CommandId::Performance>(node_id)};

    cmd.state = state;
    cmd.entry_address = entry_addresses;

    generate_end<PerformanceCommand>(cmd);
}

void CommandBuffer::generate_clear_mix_command(const s32 node_id) {
    auto& cmd{generate_start<ClearMixBufferCommand, CommandId::ClearMixBuffer>(node_id)};
    generate_end<ClearMixBufferCommand>(cmd);
}

void CommandBuffer::generate_copy_mix_buffer_command(const s32 node_id, EffectInfoBase& effect_info,
                                                 const s16 buffer_offset, const s8 channel) {
    auto& cmd{generate_start<CopyMixBufferCommand, CommandId::CopyMixBuffer>(node_id)};

    if (behavior->is_effect_info_version2_supported()) {
        const auto& parameter_v2{
            *reinterpret_cast<BiquadFilterInfo::ParameterVersion2*>(effect_info.get_parameter())};
        cmd.input_index = buffer_offset + parameter_v2.inputs[channel];
        cmd.output_index = buffer_offset + parameter_v2.outputs[channel];
        generate_end<CopyMixBufferCommand>(cmd);
        return;
    }

    const auto& parameter{
        *reinterpret_cast<BiquadFilterInfo::ParameterVersion1*>(effect_info.get_parameter())};
    cmd.input_index = buffer_offset + parameter.inputs[channel];
    cmd.output_index = buffer_offset + parameter.outputs[channel];

    generate_end<CopyMixBufferCommand>(cmd);
}

void CommandBuffer::generate_light_limiter_command(
    const s32 node_id, const s16 buffer_offset,
    const LightLimiterInfo::ParameterVersion1& parameter, const LightLimiterInfo::State& state,
    const bool enabled, const CpuAddr workbuffer) {
    auto& cmd{generate_start<LightLimiterVersion1Command, CommandId::LightLimiterVersion1>(node_id)};

    if (is_channel_count_valid(parameter.channel_count)) {
        const auto state_buffer{
            memory_pool->translate(CpuAddr(&state), sizeof(LightLimiterInfo::State))};
        if (state_buffer) {
            for (s8 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }

            std::memcpy(&cmd.parameter, &parameter, sizeof(LightLimiterInfo::ParameterVersion1));
            cmd.effect_enabled = enabled;
            cmd.state = state_buffer;
            cmd.workbuffer = workbuffer;
        }
    }

    generate_end<LightLimiterVersion1Command>(cmd);
}

void CommandBuffer::generate_light_limiter_command(
    const s32 node_id, const s16 buffer_offset,
    const LightLimiterInfo::ParameterVersion2& parameter,
    const LightLimiterInfo::StatisticsInternal& statistics, const LightLimiterInfo::State& state,
    const bool enabled, const CpuAddr workbuffer) {
    auto& cmd{generate_start<LightLimiterVersion2Command, CommandId::LightLimiterVersion2>(node_id)};
    if (is_channel_count_valid(parameter.channel_count)) {
        const auto state_buffer{
            memory_pool->translate(CpuAddr(&state), sizeof(LightLimiterInfo::State))};
        if (state_buffer) {
            for (s8 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }

            cmd.parameter = parameter;
            cmd.effect_enabled = enabled;
            cmd.state = state_buffer;
            if (cmd.parameter.statistics_enabled) {
                cmd.result_state = memory_pool->translate(
                    CpuAddr(&statistics), sizeof(LightLimiterInfo::StatisticsInternal));
            } else {
                cmd.result_state = 0;
            }
            cmd.workbuffer = workbuffer;
        }
    }

    generate_end<LightLimiterVersion2Command>(cmd);
}

void CommandBuffer::generate_multitap_biquad_filter_command(const s32 node_id, VoiceInfo& voice_info,
                                                        const VoiceState& voice_state,
                                                        const s16 buffer_count, const s8 channel) {
    auto& cmd{generate_start<MultiTapBiquadFilterCommand, CommandId::MultiTapBiquadFilter>(node_id)};

    cmd.input = buffer_count + channel;
    cmd.output = buffer_count + channel;
    cmd.biquads = voice_info.biquads;

    if (voice_info.use_float_biquads) {
        cmd.biquads_float = voice_info.biquads_float;
        cmd.use_float_coefficients = true;
    } else {
        cmd.use_float_coefficients = false;
    }

    cmd.states[0] =
        memory_pool->translate(CpuAddr(voice_state.biquad_states[0].data()),
                               MaxBiquadFilters * sizeof(VoiceState::BiquadFilterState));
    cmd.states[1] =
        memory_pool->translate(CpuAddr(voice_state.biquad_states[1].data()),
                               MaxBiquadFilters * sizeof(VoiceState::BiquadFilterState));

    cmd.needs_init[0] = !voice_info.biquad_initialized[0];
    cmd.needs_init[1] = !voice_info.biquad_initialized[1];
    cmd.filter_tap_count = MaxBiquadFilters;

    generate_end<MultiTapBiquadFilterCommand>(cmd);
}

void CommandBuffer::generate_capture_command(const s32 node_id, EffectInfoBase& effect_info,
                                           const s16 input_index, const s16 output_index,
                                           const s16 buffer_offset, const u32 update_count,
                                           const u32 count_max, const u32 write_offset) {
    auto& cmd{generate_start<CaptureCommand, CommandId::Capture>(node_id)};

    if (effect_info.get_send_buffer()) {
        cmd.input = buffer_offset + input_index;
        cmd.output = buffer_offset + output_index;
        cmd.send_buffer_info = effect_info.get_send_buffer_info();
        cmd.send_buffer = effect_info.get_send_buffer();
        cmd.count_max = count_max;
        cmd.write_offset = write_offset;
        cmd.update_count = update_count;
        cmd.effect_enabled = effect_info.is_enabled();
    }

    generate_end<CaptureCommand>(cmd);
}

void CommandBuffer::generate_compressor_command(s16 buffer_offset, EffectInfoBase& effect_info,
                                              s32 node_id) {
    auto& cmd{generate_start<CompressorCommand, CommandId::Compressor>(node_id)};

    auto& parameter{
        *reinterpret_cast<CompressorInfo::ParameterVersion2*>(effect_info.get_parameter())};
    auto state{reinterpret_cast<CompressorInfo::State*>(effect_info.get_state_buffer())};

    if (is_channel_count_valid(parameter.channel_count)) {
        auto state_buffer{memory_pool->translate(CpuAddr(state), sizeof(CompressorInfo::State))};
        if (state_buffer) {
            for (u16 channel = 0; channel < parameter.channel_count; channel++) {
                cmd.inputs[channel] = buffer_offset + parameter.inputs[channel];
                cmd.outputs[channel] = buffer_offset + parameter.outputs[channel];
            }
            cmd.parameter = parameter;
            cmd.workbuffer = state_buffer;
            cmd.enabled = effect_info.is_enabled();
        }
    }

    generate_end<CompressorCommand>(cmd);
}

} // namespace armigo::audio::renderer
