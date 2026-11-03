// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/behavior/behavior_info.h"
#include "renderer/command/command_buffer.h"
#include "renderer/command/command_generator.h"
#include "renderer/command/command_list_header.h"
#include "renderer/effect/aux_.h"
#include "renderer/effect/biquad_filter.h"
#include "renderer/effect/buffer_mixer.h"
#include "renderer/effect/capture.h"
#include "renderer/effect/effect_context.h"
#include "renderer/effect/light_limiter.h"
#include "renderer/mix/mix_context.h"
#include "renderer/performance/detail_aspect.h"
#include "renderer/performance/entry_aspect.h"
#include "renderer/sink/device_sink_info.h"
#include "renderer/sink/sink_context.h"
#include "renderer/splitter/splitter_context.h"
#include "renderer/voice/voice_context.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

CommandGenerator::CommandGenerator(CommandBuffer& command_buffer_,
                                   const CommandListHeader& command_list_header_,
                                   const AudioRendererSystemContext& render_context_,
                                   VoiceContext& voice_context_, MixContext& mix_context_,
                                   EffectContext& effect_context_, SinkContext& sink_context_,
                                   SplitterContext& splitter_context_,
                                   PerformanceManager* performance_manager_)
    : command_buffer{command_buffer_}, command_header{command_list_header_},
      render_context{render_context_}, voice_context{voice_context_}, mix_context{mix_context_},
      effect_context{effect_context_}, sink_context{sink_context_},
      splitter_context{splitter_context_}, performance_manager{performance_manager_} {
    command_buffer.generate_clear_mix_command(InvalidNodeId);
}

void CommandGenerator::generate_data_source_command(VoiceInfo& voice_info,
                                                 const VoiceState& voice_state, const s8 channel) {
    if (voice_info.mix_id == UnusedMixId) {
        if (voice_info.splitter_id != UnusedSplitterId) {
            auto destination{splitter_context.get_destination_data(voice_info.splitter_id, 0)};
            u32 dest_id{0};
            while (destination != nullptr) {
                if (destination->is_configured()) {
                    auto mix_id{destination->get_mix_id()};
                    if (mix_id < mix_context.get_count() && mix_id != UnusedSplitterId) {
                        auto mix_info{mix_context.get_info(mix_id)};
                        command_buffer.generate_depop_prepare_command(
                            voice_info.node_id, voice_state, render_context.depop_buffer,
                            mix_info->buffer_count, mix_info->buffer_offset,
                            voice_info.was_playing);
                    }
                }
                dest_id++;
                destination = splitter_context.get_destination_data(voice_info.splitter_id, dest_id);
            }
        }
    } else if (voice_info.mix_id < static_cast<u32>(mix_context.get_count())) {
        auto mix_info{mix_context.get_info(voice_info.mix_id)};
        command_buffer.generate_depop_prepare_command(
            voice_info.node_id, voice_state, render_context.depop_buffer, mix_info->buffer_count,
            mix_info->buffer_offset, voice_info.was_playing);
    }

    if (voice_info.was_playing) {
        return;
    }

    if (render_context.behavior->is_wave_buffer_ver2_supported()) {
        switch (voice_info.sample_format) {
        case SampleFormat::PcmInt16:
            command_buffer.generate_pcm_int16_version2_command(
                voice_info.node_id, voice_info, voice_state, render_context.mix_buffer_count,
                channel);
            break;
        case SampleFormat::PcmFloat:
            command_buffer.generate_pcm_float_version2_command(
                voice_info.node_id, voice_info, voice_state, render_context.mix_buffer_count,
                channel);
            break;
        case SampleFormat::Adpcm:
            command_buffer.generate_adpcm_version2_command(voice_info.node_id, voice_info, voice_state,
                                                        render_context.mix_buffer_count, channel);
            break;
        default:
            AUDIO_LOG_ERROR("Invalid SampleFormat {}",
                      static_cast<u32>(voice_info.sample_format));
            break;
        }
    } else {
        switch (voice_info.sample_format) {
        case SampleFormat::PcmInt16:
            command_buffer.generate_pcm_int16_version1_command(
                voice_info.node_id, *command_buffer.memory_pool, voice_info, voice_state,
                render_context.mix_buffer_count, channel);
            break;
        case SampleFormat::PcmFloat:
            command_buffer.generate_pcm_float_version1_command(
                voice_info.node_id, *command_buffer.memory_pool, voice_info, voice_state,
                render_context.mix_buffer_count, channel);
            break;
        case SampleFormat::Adpcm:
            command_buffer.generate_adpcm_version1_command(
                voice_info.node_id, *command_buffer.memory_pool, voice_info, voice_state,
                render_context.mix_buffer_count, channel);
            break;
        default:
            AUDIO_LOG_ERROR("Invalid SampleFormat {}",
                      static_cast<u32>(voice_info.sample_format));
            break;
        }
    }
}

void CommandGenerator::generate_voice_mix_command(std::span<const f32> mix_volumes,
                                               std::span<const f32> prev_mix_volumes,
                                               const VoiceState& voice_state, s16 output_index,
                                               const s16 buffer_count, const s16 input_index,
                                               const s32 node_id) {
    u8 precision{15};
    if (render_context.behavior->is_volume_mix_parameter_precision_q23_supported()) {
        precision = 23;
    }

    if (buffer_count > 8) {
        const auto prev_samples{render_context.memory_pool_info->translate(
            CpuAddr(voice_state.previous_samples.data()), buffer_count * sizeof(s32))};
        command_buffer.generate_mix_ramp_grouped_command(node_id, buffer_count, input_index,
                                                     output_index, mix_volumes, prev_mix_volumes,
                                                     prev_samples, precision);
    } else {
        for (s16 i = 0; i < buffer_count; i++, output_index++) {
            const auto prev_samples{render_context.memory_pool_info->translate(
                CpuAddr(&voice_state.previous_samples[i]), sizeof(s32))};

            command_buffer.generate_mix_ramp_command(node_id, buffer_count, input_index, output_index,
                                                  mix_volumes[i], prev_mix_volumes[i], prev_samples,
                                                  precision);
        }
    }
}

void CommandGenerator::generate_biquad_filter_command_for_voice(VoiceInfo& voice_info,
                                                           const VoiceState& voice_state,
                                                           const s16 buffer_count, const s8 channel,
                                                           const s32 node_id) {
    const bool both_biquads_enabled{voice_info.biquads[0].enabled && voice_info.biquads[1].enabled};
    const auto use_float_processing{render_context.behavior->use_biquad_filter_float_processing()};

    if (both_biquads_enabled && render_context.behavior->use_multi_tap_biquad_filter_processing() &&
        use_float_processing) {
        command_buffer.generate_multitap_biquad_filter_command(node_id, voice_info, voice_state,
                                                           buffer_count, channel);
    } else {
        for (u32 i = 0; i < MaxBiquadFilters; i++) {
            if (voice_info.biquads[i].enabled) {
                command_buffer.generate_biquad_filter_command(node_id, voice_info, voice_state,
                                                           buffer_count, channel, i,
                                                           use_float_processing);
            }
        }
    }
}

void CommandGenerator::generate_voice_command(VoiceInfo& voice_info) {
    u8 precision{15};
    if (render_context.behavior->is_volume_mix_parameter_precision_q23_supported()) {
        precision = 23;
    }

    for (s8 channel = 0; channel < voice_info.channel_count; channel++) {
        const auto resource_id{voice_info.channel_resource_ids[channel]};
        auto& voice_state{voice_context.get_dsp_shared_state(resource_id)};
        auto& channel_resource{voice_context.get_channel_resource(resource_id)};

        PerformanceDetailType detail_type{PerformanceDetailType::Invalid};
        switch (voice_info.sample_format) {
        case SampleFormat::PcmInt16:
            detail_type = PerformanceDetailType::Unk1;
            break;
        case SampleFormat::PcmFloat:
            detail_type = PerformanceDetailType::Unk10;
            break;
        default:
            detail_type = PerformanceDetailType::Unk2;
            break;
        }

        DetailAspect data_source_detail(*this, PerformanceEntryType::Voice, voice_info.node_id,
                                        detail_type);
        generate_data_source_command(voice_info, voice_state, channel);

        if (data_source_detail.initialized) {
            command_buffer.generate_performance_command(data_source_detail.node_id,
                                                      PerformanceState::Stop,
                                                      data_source_detail.performance_entry_address);
        }

        if (voice_info.was_playing) {
            voice_info.prev_volume = 0.0f;
            continue;
        }

        if (!voice_info.has_any_connection()) {
            continue;
        }

        DetailAspect biquad_detail_aspect(*this, PerformanceEntryType::Voice, voice_info.node_id,
                                          PerformanceDetailType::Unk4);
        generate_biquad_filter_command_for_voice(
            voice_info, voice_state, render_context.mix_buffer_count, channel, voice_info.node_id);

        if (biquad_detail_aspect.initialized) {
            command_buffer.generate_performance_command(
                biquad_detail_aspect.node_id, PerformanceState::Stop,
                biquad_detail_aspect.performance_entry_address);
        }

        DetailAspect volume_ramp_detail_aspect(*this, PerformanceEntryType::Voice,
                                               voice_info.node_id, PerformanceDetailType::Unk3);
        command_buffer.generate_volume_ramp_command(
            voice_info.node_id, voice_info, render_context.mix_buffer_count + channel, precision);
        if (volume_ramp_detail_aspect.initialized) {
            command_buffer.generate_performance_command(
                volume_ramp_detail_aspect.node_id, PerformanceState::Stop,
                volume_ramp_detail_aspect.performance_entry_address);
        }

        voice_info.prev_volume = voice_info.volume;

        if (voice_info.mix_id == UnusedMixId) {
            if (voice_info.splitter_id != UnusedSplitterId) {
                auto i{channel};
                auto destination{splitter_context.get_destination_data(voice_info.splitter_id, i)};
                while (destination != nullptr) {
                    if (destination->is_configured()) {
                        const auto mix_id{destination->get_mix_id()};
                        if (mix_id < mix_context.get_count() &&
                            static_cast<s32>(mix_id) != UnusedSplitterId) {
                            auto mix_info{mix_context.get_info(mix_id)};
                            generate_voice_mix_command(
                                destination->get_mix_volume(), destination->get_mix_volume_prev(),
                                voice_state, mix_info->buffer_offset, mix_info->buffer_count,
                                render_context.mix_buffer_count + channel, voice_info.node_id);
                            destination->mark_as_need_to_update_internal_state();
                        }
                    }
                    i += voice_info.channel_count;
                    destination = splitter_context.get_destination_data(voice_info.splitter_id, i);
                }
            }
        } else if (voice_info.mix_id < static_cast<u32>(mix_context.get_count())) {
            // A mix id outside the mix table routes the voice nowhere.
            DetailAspect volume_mix_detail_aspect(*this, PerformanceEntryType::Voice,
                                                  voice_info.node_id, PerformanceDetailType::Unk3);
            auto mix_info{mix_context.get_info(voice_info.mix_id)};
            generate_voice_mix_command(channel_resource.mix_volumes, channel_resource.prev_mix_volumes,
                                    voice_state, mix_info->buffer_offset, mix_info->buffer_count,
                                    render_context.mix_buffer_count + channel, voice_info.node_id);
            if (volume_mix_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    volume_mix_detail_aspect.node_id, PerformanceState::Stop,
                    volume_mix_detail_aspect.performance_entry_address);
            }

            channel_resource.prev_mix_volumes = channel_resource.mix_volumes;
        }
        voice_info.biquad_initialized[0] = voice_info.biquads[0].enabled;
        voice_info.biquad_initialized[1] = voice_info.biquads[1].enabled;
    }
}

void CommandGenerator::generate_voice_commands() {
    const auto voice_count{voice_context.get_count()};

    for (u32 i = 0; i < voice_count; i++) {
        auto sorted_info{voice_context.get_sorted_info(i)};

        if (sorted_info->should_skip() || !sorted_info->update_for_command_generation(voice_context)) {
            continue;
        }

        EntryAspect voice_entry_aspect(*this, PerformanceEntryType::Voice, sorted_info->node_id);

        generate_voice_command(*sorted_info);

        if (voice_entry_aspect.initialized) {
            command_buffer.generate_performance_command(voice_entry_aspect.node_id,
                                                      PerformanceState::Stop,
                                                      voice_entry_aspect.performance_entry_address);
        }
    }

    splitter_context.update_internal_state();
}

void CommandGenerator::generate_buffer_mixer_command(const s16 buffer_offset,
                                                  EffectInfoBase& effect_info, const s32 node_id) {
    u8 precision{15};
    if (render_context.behavior->is_volume_mix_parameter_precision_q23_supported()) {
        precision = 23;
    }

    if (effect_info.is_enabled()) {
        const auto& parameter{
            *reinterpret_cast<BufferMixerInfo::ParameterVersion1*>(effect_info.get_parameter())};
        for (u32 i = 0; i < parameter.mix_count; i++) {
            if (parameter.volumes[i] != 0.0f) {
                command_buffer.generate_mix_command(node_id, buffer_offset + parameter.inputs[i],
                                                  buffer_offset + parameter.outputs[i],
                                                  buffer_offset, parameter.volumes[i], precision);
            }
        }
    }
}

void CommandGenerator::generate_delay_command(const s16 buffer_offset, EffectInfoBase& effect_info,
                                            const s32 node_id) {
    command_buffer.generate_delay_command(node_id, effect_info, buffer_offset);
}

void CommandGenerator::generate_reverb_command(const s16 buffer_offset, EffectInfoBase& effect_info,
                                             const s32 node_id,
                                             const bool long_size_pre_delay_supported) {
    command_buffer.generate_reverb_command(node_id, effect_info, buffer_offset,
                                         long_size_pre_delay_supported);
}

void CommandGenerator::generate_i3dl2_reverb_effect_command(const s16 buffer_offset,
                                                        EffectInfoBase& effect_info,
                                                        const s32 node_id) {
    command_buffer.generate_i3dl2_reverb_command(node_id, effect_info, buffer_offset);
}

void CommandGenerator::generate_aux_command(const s16 buffer_offset, EffectInfoBase& effect_info,
                                          const s32 node_id) {

    if (effect_info.is_enabled()) {
        effect_info.get_workbuffer(0);
        effect_info.get_workbuffer(1);
    }

    if (effect_info.get_send_buffer() != 0 && effect_info.get_return_buffer() != 0) {
        const auto& parameter{
            *reinterpret_cast<AuxInfo::ParameterVersion1*>(effect_info.get_parameter())};
        auto channel_index{parameter.mix_buffer_count - 1};
        u32 write_offset{0};
        for (u32 i = 0; i < parameter.mix_buffer_count; i++, channel_index--) {
            auto new_update_count{command_header.sample_count + write_offset};
            const auto update_count{channel_index > 0 ? 0 : new_update_count};
            command_buffer.generate_aux_command(node_id, effect_info, parameter.inputs[i],
                                              parameter.outputs[i], buffer_offset, update_count,
                                              parameter.count_max, write_offset);
            write_offset = new_update_count;
        }
    }
}

void CommandGenerator::generate_biquad_filter_effect_command(const s16 buffer_offset,
                                                         EffectInfoBase& effect_info,
                                                         const s32 node_id) {
    EffectInfoBase::ParameterState param_state{};
    s8 channel_count = 0;

    if (render_context.behavior->is_effect_info_version2_supported()) {
        const auto* parameter =
            reinterpret_cast<const BiquadFilterInfo::ParameterVersion2*>(effect_info.get_parameter());
        param_state = parameter->state;
        channel_count = parameter->channel_count;
    } else {
        const auto* parameter =
            reinterpret_cast<const BiquadFilterInfo::ParameterVersion1*>(effect_info.get_parameter());
        param_state = parameter->state;
        channel_count = parameter->channel_count;
    }

    if (channel_count > 0 && param_state <= EffectInfoBase::ParameterState::Updated) {
        if (effect_info.is_enabled()) {
            bool needs_init{false};

            switch (param_state) {
            case EffectInfoBase::ParameterState::Initialized:
                needs_init = true;
                break;
            case EffectInfoBase::ParameterState::Updating:
                needs_init = !render_context.behavior->is_biquad_filter_effect_state_clear_bug_fixed();
                break;
            case EffectInfoBase::ParameterState::Updated:
                needs_init = false;
                break;
            }

            const bool use_float_processing =
                render_context.behavior->use_biquad_filter_float_processing();

            for (s8 channel = 0; channel < channel_count; channel++) {
                command_buffer.generate_biquad_filter_command(node_id, effect_info, buffer_offset, channel,
                                                           needs_init, use_float_processing);
            }
        } else {
            for (s8 channel = 0; channel < channel_count; channel++) {
                command_buffer.generate_copy_mix_buffer_command(node_id, effect_info, buffer_offset,
                                                            channel);
            }
        }
    }
}

void CommandGenerator::generate_light_limiter_effect_command(const s16 buffer_offset,
                                                         EffectInfoBase& effect_info,
                                                         const s32 node_id,
                                                         const u32 effect_index) {

    const auto& state{*reinterpret_cast<LightLimiterInfo::State*>(effect_info.get_state_buffer())};

    if (render_context.behavior->is_effect_info_version2_supported()) {
        const auto& parameter{
            *reinterpret_cast<LightLimiterInfo::ParameterVersion2*>(effect_info.get_parameter())};
        const auto& result_state{*reinterpret_cast<LightLimiterInfo::StatisticsInternal*>(
            &effect_context.get_dsp_shared_result_state(effect_index))};
        command_buffer.generate_light_limiter_command(node_id, buffer_offset, parameter, result_state,
                                                   state, effect_info.is_enabled(),
                                                   effect_info.get_workbuffer(-1));
    } else {
        const auto& parameter{
            *reinterpret_cast<LightLimiterInfo::ParameterVersion1*>(effect_info.get_parameter())};
        command_buffer.generate_light_limiter_command(node_id, buffer_offset, parameter, state,
                                                   effect_info.is_enabled(),
                                                   effect_info.get_workbuffer(-1));
    }
}

void CommandGenerator::generate_capture_command(const s16 buffer_offset, EffectInfoBase& effect_info,
                                              const s32 node_id) {
    if (effect_info.is_enabled()) {
        effect_info.get_workbuffer(0);
    }

    if (effect_info.get_send_buffer()) {
        const auto& parameter{
            *reinterpret_cast<AuxInfo::ParameterVersion1*>(effect_info.get_parameter())};
        auto channel_index{parameter.mix_buffer_count - 1};
        u32 write_offset{0};
        for (u32 i = 0; i < parameter.mix_buffer_count; i++, channel_index--) {
            auto new_update_count{command_header.sample_count + write_offset};
            const auto update_count{channel_index > 0 ? 0 : new_update_count};
            command_buffer.generate_capture_command(node_id, effect_info, parameter.inputs[i],
                                                  parameter.outputs[i], buffer_offset, update_count,
                                                  parameter.count_max, write_offset);
            write_offset = new_update_count;
        }
    }
}

void CommandGenerator::generate_compressor_command(const s16 buffer_offset,
                                                 EffectInfoBase& effect_info, const s32 node_id) {
    command_buffer.generate_compressor_command(buffer_offset, effect_info, node_id);
}

void CommandGenerator::generate_effect_command(MixInfo& mix_info) {
    const auto effect_count{effect_context.get_count()};
    for (u32 i = 0; i < effect_count; i++) {
        const auto effect_index{mix_info.effect_order_buffer[i]};
        if (effect_index == -1) {
            break;
        }

        auto& effect_info = effect_context.get_info(effect_index);
        if (effect_info.should_skip()) {
            continue;
        }

        const auto entry_type{mix_info.mix_id == FinalMixId ? PerformanceEntryType::FinalMix
                                                            : PerformanceEntryType::SubMix};

        switch (effect_info.get_type()) {
        case EffectInfoBase::Type::Mix: {
            DetailAspect mix_detail_aspect(*this, entry_type, mix_info.node_id,
                                           PerformanceDetailType::Unk5);
            generate_buffer_mixer_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (mix_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    mix_detail_aspect.node_id, PerformanceState::Stop,
                    mix_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::Aux: {
            DetailAspect aux_detail_aspect(*this, entry_type, mix_info.node_id,
                                           PerformanceDetailType::Unk7);
            generate_aux_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (aux_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    aux_detail_aspect.node_id, PerformanceState::Stop,
                    aux_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::Delay: {
            DetailAspect delay_detail_aspect(*this, entry_type, mix_info.node_id,
                                             PerformanceDetailType::Unk6);
            generate_delay_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (delay_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    delay_detail_aspect.node_id, PerformanceState::Stop,
                    delay_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::Reverb: {
            DetailAspect reverb_detail_aspect(*this, entry_type, mix_info.node_id,
                                              PerformanceDetailType::Unk8);
            generate_reverb_command(mix_info.buffer_offset, effect_info, mix_info.node_id,
                                  render_context.behavior->is_long_size_pre_delay_supported());
            if (reverb_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    reverb_detail_aspect.node_id, PerformanceState::Stop,
                    reverb_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::I3dl2Reverb: {
            DetailAspect i3dl2_detail_aspect(*this, entry_type, mix_info.node_id,
                                             PerformanceDetailType::Unk9);
            generate_i3dl2_reverb_effect_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (i3dl2_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    i3dl2_detail_aspect.node_id, PerformanceState::Stop,
                    i3dl2_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::BiquadFilter: {
            DetailAspect biquad_detail_aspect(*this, entry_type, mix_info.node_id,
                                              PerformanceDetailType::Unk4);
            generate_biquad_filter_effect_command(mix_info.buffer_offset, effect_info,
                                              mix_info.node_id);
            if (biquad_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    biquad_detail_aspect.node_id, PerformanceState::Stop,
                    biquad_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::LightLimiter: {
            DetailAspect light_limiter_detail_aspect(*this, entry_type, mix_info.node_id,
                                                     PerformanceDetailType::Unk11);
            generate_light_limiter_effect_command(mix_info.buffer_offset, effect_info, mix_info.node_id,
                                              effect_index);
            if (light_limiter_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    light_limiter_detail_aspect.node_id, PerformanceState::Stop,
                    light_limiter_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::Capture: {
            DetailAspect capture_detail_aspect(*this, entry_type, mix_info.node_id,
                                               PerformanceDetailType::Unk12);
            generate_capture_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (capture_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    capture_detail_aspect.node_id, PerformanceState::Stop,
                    capture_detail_aspect.performance_entry_address);
            }
        } break;

        case EffectInfoBase::Type::Compressor: {
            DetailAspect capture_detail_aspect(*this, entry_type, mix_info.node_id,
                                               PerformanceDetailType::Unk13);
            generate_compressor_command(mix_info.buffer_offset, effect_info, mix_info.node_id);
            if (capture_detail_aspect.initialized) {
                command_buffer.generate_performance_command(
                    capture_detail_aspect.node_id, PerformanceState::Stop,
                    capture_detail_aspect.performance_entry_address);
            }
        } break;

        default:
            AUDIO_LOG_ERROR("Invalid effect type {}",
                      static_cast<u32>(effect_info.get_type()));
            break;
        }

        effect_info.update_for_command_generation();
    }
}

void CommandGenerator::generate_mix_commands(MixInfo& mix_info) {
    u8 precision{15};
    if (render_context.behavior->is_volume_mix_parameter_precision_q23_supported()) {
        precision = 23;
    }

    if (!mix_info.has_any_connection()) {
        return;
    }

    if (mix_info.dst_mix_id == UnusedMixId) {
        if (mix_info.dst_splitter_id != UnusedSplitterId) {
            s16 dest_id{0};
            auto destination{
                splitter_context.get_destination_data(mix_info.dst_splitter_id, dest_id)};
            while (destination != nullptr) {
                if (destination->is_configured()) {
                    auto splitter_mix_id{destination->get_mix_id()};
                    if (splitter_mix_id < mix_context.get_count()) {
                        auto splitter_mix_info{mix_context.get_info(splitter_mix_id)};
                        const s16 input_index{static_cast<s16>(mix_info.buffer_offset +
                                                               (dest_id % mix_info.buffer_count))};
                        for (s16 i = 0; i < splitter_mix_info->buffer_count; i++) {
                            auto volume{mix_info.volume * destination->get_mix_volume(i)};
                            if (volume != 0.0f) {
                                command_buffer.generate_mix_command(
                                    mix_info.node_id, input_index,
                                    splitter_mix_info->buffer_offset + i, mix_info.buffer_offset,
                                    volume, precision);
                            }
                        }
                    }
                }
                dest_id++;
                destination =
                    splitter_context.get_destination_data(mix_info.dst_splitter_id, dest_id);
            }
        }
    } else if (mix_info.dst_mix_id >= 0 &&
               mix_info.dst_mix_id < static_cast<s32>(mix_context.get_count())) {
        auto dest_mix_info{mix_context.get_info(mix_info.dst_mix_id)};
        for (s16 i = 0; i < mix_info.buffer_count; i++) {
            for (s16 j = 0; j < dest_mix_info->buffer_count; j++) {
                auto volume{mix_info.volume * mix_info.mix_volumes[i][j]};
                if (volume != 0.0f) {
                    command_buffer.generate_mix_command(mix_info.node_id, mix_info.buffer_offset + i,
                                                      dest_mix_info->buffer_offset + j,
                                                      mix_info.buffer_offset, volume, precision);
                }
            }
        }
    }
}

void CommandGenerator::generate_sub_mix_command(MixInfo& mix_info) {
    command_buffer.generate_depop_for_mix_buffers_command(mix_info.node_id, mix_info,
                                                     render_context.depop_buffer);
    generate_effect_command(mix_info);

    DetailAspect mix_detail_aspect(*this, PerformanceEntryType::SubMix, mix_info.node_id,
                                   PerformanceDetailType::Unk5);

    generate_mix_commands(mix_info);

    if (mix_detail_aspect.initialized) {
        command_buffer.generate_performance_command(mix_detail_aspect.node_id, PerformanceState::Stop,
                                                  mix_detail_aspect.performance_entry_address);
    }
}

void CommandGenerator::generate_sub_mix_commands() {
    const auto submix_count{mix_context.get_count()};
    for (s32 i = 0; i < submix_count; i++) {
        auto sorted_info{mix_context.get_sorted_info(i)};
        if (!sorted_info->in_use || sorted_info->mix_id == FinalMixId) {
            continue;
        }

        EntryAspect submix_entry_aspect(*this, PerformanceEntryType::SubMix, sorted_info->node_id);

        generate_sub_mix_command(*sorted_info);

        if (submix_entry_aspect.initialized) {
            command_buffer.generate_performance_command(
                submix_entry_aspect.node_id, PerformanceState::Stop,
                submix_entry_aspect.performance_entry_address);
        }
    }
}

void CommandGenerator::generate_final_mix_command() {
    auto& final_mix_info{*mix_context.get_final_mix_info()};

    command_buffer.generate_depop_for_mix_buffers_command(final_mix_info.node_id, final_mix_info,
                                                     render_context.depop_buffer);
    generate_effect_command(final_mix_info);

    u8 precision{15};
    if (render_context.behavior->is_volume_mix_parameter_precision_q23_supported()) {
        precision = 23;
    }

    for (s16 i = 0; i < final_mix_info.buffer_count; i++) {
        DetailAspect volume_aspect(*this, PerformanceEntryType::FinalMix, final_mix_info.node_id,
                                   PerformanceDetailType::Unk3);
        command_buffer.generate_volume_command(final_mix_info.node_id, final_mix_info.buffer_offset,
                                             i, final_mix_info.volume, precision);
        if (volume_aspect.initialized) {
            command_buffer.generate_performance_command(volume_aspect.node_id, PerformanceState::Stop,
                                                      volume_aspect.performance_entry_address);
        }
    }
}

void CommandGenerator::generate_final_mix_commands() {
    auto final_mix_info{mix_context.get_final_mix_info()};
    EntryAspect final_mix_entry(*this, PerformanceEntryType::FinalMix, final_mix_info->node_id);
    generate_final_mix_command();
    if (final_mix_entry.initialized) {
        command_buffer.generate_performance_command(final_mix_entry.node_id, PerformanceState::Stop,
                                                  final_mix_entry.performance_entry_address);
    }
}

void CommandGenerator::generate_sink_commands() {
    const auto sink_count{sink_context.get_count()};

    for (u32 i = 0; i < sink_count; i++) {
        auto sink_info{sink_context.get_info(i)};
        if (sink_info->is_used() && sink_info->get_type() == SinkInfoBase::Type::DeviceSink) {
            auto state{reinterpret_cast<DeviceSinkInfo::DeviceState*>(sink_info->get_state())};
            if (command_header.sample_rate != TargetSampleRate &&
                state->upsampler_info == nullptr) {
                auto device_state{sink_info->get_device_state()};
                device_state->upsampler_info = render_context.upsampler_manager->allocate();
            }

            EntryAspect device_sink_entry(*this, PerformanceEntryType::Sink,
                                          sink_info->get_node_id());
            auto final_mix{mix_context.get_final_mix_info()};
            generate_sink_command(final_mix->buffer_offset, *sink_info);

            if (device_sink_entry.initialized) {
                command_buffer.generate_performance_command(
                    device_sink_entry.node_id, PerformanceState::Stop,
                    device_sink_entry.performance_entry_address);
            }
        }
    }

    for (u32 i = 0; i < sink_count; i++) {
        auto sink_info{sink_context.get_info(i)};
        if (sink_info->is_used() && sink_info->get_type() == SinkInfoBase::Type::CircularBufferSink) {
            EntryAspect circular_buffer_entry(*this, PerformanceEntryType::Sink,
                                              sink_info->get_node_id());
            auto final_mix{mix_context.get_final_mix_info()};
            generate_sink_command(final_mix->buffer_offset, *sink_info);

            if (circular_buffer_entry.initialized) {
                command_buffer.generate_performance_command(
                    circular_buffer_entry.node_id, PerformanceState::Stop,
                    circular_buffer_entry.performance_entry_address);
            }
        }
    }
}

void CommandGenerator::generate_sink_command(const s16 buffer_offset, SinkInfoBase& sink_info) {
    if (sink_info.should_skip()) {
        return;
    }

    switch (sink_info.get_type()) {
    case SinkInfoBase::Type::DeviceSink:
        generate_device_sink_command(buffer_offset, sink_info);
        break;

    case SinkInfoBase::Type::CircularBufferSink:
        command_buffer.generate_circular_buffer_sink_command(sink_info.get_node_id(), sink_info,
                                                         buffer_offset);
        break;

    default:
        AUDIO_LOG_ERROR("Invalid sink type {}", static_cast<u32>(sink_info.get_type()));
        break;
    }

    sink_info.update_for_command_generation();
}

void CommandGenerator::generate_device_sink_command(const s16 buffer_offset, SinkInfoBase& sink_info) {
    auto& parameter{
        *reinterpret_cast<DeviceSinkInfo::DeviceInParameter*>(sink_info.get_parameter())};
    auto state{*reinterpret_cast<DeviceSinkInfo::DeviceState*>(sink_info.get_state())};

    if (render_context.channels == 2 && parameter.downmix_enabled) {
        command_buffer.generate_down_mix6ch_to2ch_command(InvalidNodeId, parameter.inputs,
                                                      buffer_offset, parameter.downmix_coeff);
    }

    if (state.upsampler_info != nullptr) {
        command_buffer.generate_upsample_command(
            InvalidNodeId, buffer_offset, *state.upsampler_info, parameter.input_count,
            parameter.inputs, command_header.buffer_count, command_header.sample_count,
            command_header.sample_rate);
    }

    command_buffer.generate_device_sink_command(InvalidNodeId, buffer_offset, sink_info,
                                             render_context.session_id,
                                             command_header.samples_buffer);
}

void CommandGenerator::generate_performance_command(
    s32 node_id, PerformanceState state, const PerformanceEntryAddresses& entry_addresses) {
    command_buffer.generate_performance_command(node_id, state, entry_addresses);
}

} // namespace armigo::audio::renderer
