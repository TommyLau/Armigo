// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/common/feature_support.h"
#include "renderer/common/support.h"
#include "renderer/behavior/behavior_info.h"
#include "renderer/behavior/info_updater.h"
#include "renderer/effect/effect_context.h"
#include "renderer/effect/effect_reset.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/mix/mix_context.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/sink/circular_buffer_sink_info.h"
#include "renderer/sink/device_sink_info.h"
#include "renderer/sink/sink_context.h"
#include "renderer/splitter/splitter_context.h"
#include "renderer/voice/voice_context.h"

namespace armigo::audio::renderer {

InfoUpdater::InfoUpdater(std::span<const u8> input_, std::span<u8> output_,
                         GuestMemory* memory_, BehaviorInfo& behaviour_)
    : input{input_.data() + sizeof(UpdateDataHeader)},
      input_origin{input_}, output{output_.data() + sizeof(UpdateDataHeader)},
      output_origin{output_}, in_header{reinterpret_cast<const UpdateDataHeader*>(
                                  input_origin.data())},
      out_header{reinterpret_cast<UpdateDataHeader*>(output_origin.data())},
      expected_input_size{input_.size()}, expected_output_size{output_.size()},
      memory{memory_}, behaviour{behaviour_} {
    std::construct_at<UpdateDataHeader>(out_header, behaviour.get_process_revision());
}

bool InfoUpdater::input_has(const u64 size) const {
    const auto consumed{static_cast<u64>(input - input_origin.data())};
    return consumed <= input_origin.size() && size <= input_origin.size() - consumed;
}

bool InfoUpdater::output_has(const u64 size) const {
    const auto produced{static_cast<u64>(output - output_origin.data())};
    return produced <= output_origin.size() && size <= output_origin.size() - produced;
}

Result InfoUpdater::update_voice_channel_resources(VoiceContext& voice_context) {
    const auto voice_count{voice_context.get_count()};
    if (!input_has(u64{voice_count} * sizeof(VoiceChannelResource::InParameter))) {
        return ResultInvalidUpdateInfo;
    }
    std::span<const VoiceChannelResource::InParameter> in_params{
        reinterpret_cast<const VoiceChannelResource::InParameter*>(input), voice_count};

    for (u32 i = 0; i < voice_count; i++) {
        auto& resource{voice_context.get_channel_resource(i)};
        resource.in_use = in_params[i].in_use;
        if (in_params[i].in_use) {
            resource.mix_volumes = in_params[i].mix_volumes;
        }
    }

    const auto consumed_input_size{voice_count *
                                   static_cast<u32>(sizeof(VoiceChannelResource::InParameter))};
    if (consumed_input_size != in_header->voice_resources_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect voice resource size, header size={}, consumed={}",
                  in_header->voice_resources_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    input += consumed_input_size;
    return ResultSuccess;
}

Result InfoUpdater::update_voices(VoiceContext& voice_context,
                                 std::span<MemoryPoolInfo> memory_pools,
                                 const u32 memory_pool_count) {
    const PoolMapper pool_mapper(memory, memory_pools, memory_pool_count,
                                 behaviour.is_memory_force_mapping_enabled());
    const auto voice_count{voice_context.get_count()};
    std::span<VoiceInfo::OutStatus> out_params{reinterpret_cast<VoiceInfo::OutStatus*>(output),
                                               voice_count};

    for (u32 i = 0; i < voice_count; i++) {
        auto& voice_info{voice_context.get_info(i)};
        voice_info.in_use = false;
    }

    u32 new_voice_count{0};

    // Two input formats exist: legacy (0x170) and v2 with float biquad (0x188).
    const bool use_v2 = behaviour.is_voice_in_parameter_v2_supported();
    const u32 in_stride = use_v2 ? 0x188u : static_cast<u32>(sizeof(VoiceInfo::InParameter));
    if (!input_has(u64{voice_count} * in_stride) ||
        !output_has(u64{voice_count} * sizeof(VoiceInfo::OutStatus))) {
        return ResultInvalidUpdateInfo;
    }

    for (u32 i = 0; i < voice_count; i++) {
        VoiceInfo::InParameter local_in{};
        std::array<VoiceInfo::BiquadFilterParameter2, MaxBiquadFilters> float_biquads{};

        if (!use_v2) {
            const auto* in_param_ptr = reinterpret_cast<const VoiceInfo::InParameter*>(
                input + i * sizeof(VoiceInfo::InParameter));
            local_in = *in_param_ptr;
        } else {
            // The revision 15 layout: float biquad coefficients widen the
            // filters by 0x18 bytes and every later field moves with them.
            static_assert(sizeof(VoiceInfo::InParameter2) == 0x188);
            const auto* vin = reinterpret_cast<const VoiceInfo::InParameter2*>(input + i * in_stride);
            local_in.id = vin->id;
            local_in.node_id = vin->node_id;
            local_in.is_new = vin->is_new;
            local_in.in_use = vin->in_use;
            local_in.play_state = vin->play_state;
            local_in.sample_format = vin->sample_format;
            local_in.sample_rate = vin->sample_rate;
            local_in.priority = static_cast<s32>(vin->priority);
            local_in.sort_order = static_cast<s32>(vin->sort_order);
            local_in.channel_count = vin->channel_count;
            local_in.pitch = vin->pitch;
            local_in.volume = vin->volume;

            // Revision 15 carries float biquad coefficients. The voice keeps them
            // for float processing; the Q14 conversion serves the fixed paths.
            for (size_t filter_idx = 0; filter_idx < MaxBiquadFilters; filter_idx++) {
                const auto& src = vin->biquads[filter_idx];
                auto& dst = local_in.biquads[filter_idx];
                dst.enabled = src.enabled;
                dst.b[0] = static_cast<s16>(std::clamp(src.numerator[0] * 16384.0f, -32768.0f, 32767.0f));
                dst.b[1] = static_cast<s16>(std::clamp(src.numerator[1] * 16384.0f, -32768.0f, 32767.0f));
                dst.b[2] = static_cast<s16>(std::clamp(src.numerator[2] * 16384.0f, -32768.0f, 32767.0f));
                dst.a[0] = static_cast<s16>(std::clamp(src.denominator[0] * 16384.0f, -32768.0f, 32767.0f));
                dst.a[1] = static_cast<s16>(std::clamp(src.denominator[1] * 16384.0f, -32768.0f, 32767.0f));

                float_biquads[filter_idx] = src;
            }
            local_in.wave_buffer_count = vin->wave_buffer_count;
            local_in.wave_buffer_index = static_cast<u16>(vin->wave_buffer_index);
            local_in.src_data_address = static_cast<CpuAddr>(vin->src_data_address);
            local_in.src_data_size = vin->src_data_size;
            local_in.mix_id = static_cast<u32>(vin->mix_id);
            local_in.splitter_id = vin->splitter_id;
            local_in.wave_buffer_internal = vin->wave_buffer_internal;
            for (u32 channel = 0; channel < MaxChannels; channel++) {
                local_in.channel_resource_ids[channel] =
                    static_cast<u32>(vin->channel_resource_ids[channel]);
            }
            local_in.clear_voice_drop = vin->clear_voice_drop;
            local_in.flush_buffer_count = vin->flush_buffer_count;
            local_in.flags = vin->flags;
            local_in.src_quality = vin->src_quality;
        }
        const auto& in_param = local_in;
        std::array<VoiceState*, MaxChannels> voice_states{};

        if (!in_param.in_use) {
            continue;
        }

        // Out-of-range ids or channel counts would index past the renderer's
        // voice tables; such an entry is ignored.
        if (in_param.id >= voice_count || in_param.channel_count > MaxChannels) {
            AUDIO_LOG_ERROR("Voice {} has an invalid id {} or channel count {}", i, in_param.id,
                            in_param.channel_count);
            continue;
        }
        bool valid_resources{true};
        for (u32 channel = 0; channel < in_param.channel_count; channel++) {
            valid_resources &= in_param.channel_resource_ids[channel] < voice_count;
        }
        if (!valid_resources) {
            AUDIO_LOG_ERROR("Voice {} names a channel resource outside the voice table", i);
            continue;
        }

        auto& voice_info{voice_context.get_info(in_param.id)};

        for (u32 channel = 0; channel < in_param.channel_count; channel++) {
            voice_states[channel] = &voice_context.get_state(in_param.channel_resource_ids[channel]);
        }

        if (in_param.is_new) {
            voice_info.initialize();

            for (u32 channel = 0; channel < in_param.channel_count; channel++) {
                *voice_states[channel] = {};
            }
        }

        BehaviorInfo::ErrorInfo update_error{};
        voice_info.update_parameters(update_error, in_param, pool_mapper, behaviour);

        // For REV15+, store the native float biquad coefficients
        if (use_v2) {
            voice_info.use_float_biquads = true;
            voice_info.biquads_float = float_biquads;
        } else {
            voice_info.use_float_biquads = false;
        }

        if (!update_error.error_code.is_success()) {
            behaviour.append_error(update_error);
        }

        std::array<std::array<BehaviorInfo::ErrorInfo, 2>, MaxWaveBuffers> wavebuffer_errors{};
        voice_info.update_wave_buffers(wavebuffer_errors, MaxWaveBuffers * 2, in_param, voice_states,
                                     pool_mapper, behaviour);

        for (auto& wavebuffer_error : wavebuffer_errors) {
            for (auto& error : wavebuffer_error) {
                if (error.error_code.is_error()) {
                    behaviour.append_error(error);
                }
            }
        }

        voice_info.write_out_status(out_params[i], in_param, voice_states);
        new_voice_count += in_param.channel_count;
    }

    auto consumed_input_size{voice_count * in_stride};
    auto consumed_output_size{voice_count * static_cast<u32>(sizeof(VoiceInfo::OutStatus))};
    if (consumed_input_size != in_header->voices_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect voices size, header size={}, consumed={}",
                  in_header->voices_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    out_header->voices_size = consumed_output_size;
    out_header->size += consumed_output_size;
    input += consumed_input_size;
    output += consumed_output_size;

    voice_context.set_active_count(new_voice_count);

    return ResultSuccess;
}

Result InfoUpdater::update_effects(EffectContext& effect_context, const bool renderer_active,
                                  std::span<MemoryPoolInfo> memory_pools,
                                  const u32 memory_pool_count) {
    if (behaviour.is_effect_info_version2_supported()) {
        return update_effects_version2(effect_context, renderer_active, memory_pools,
                                     memory_pool_count);
    } else {
        return update_effects_version1(effect_context, renderer_active, memory_pools,
                                     memory_pool_count);
    }
}

Result InfoUpdater::update_effects_version1(EffectContext& effect_context, const bool renderer_active,
                                          std::span<MemoryPoolInfo> memory_pools,
                                          const u32 memory_pool_count) {
    PoolMapper pool_mapper(memory, memory_pools, memory_pool_count,
                           behaviour.is_memory_force_mapping_enabled());

    const auto effect_count{effect_context.get_count()};

    if (!input_has(u64{effect_count} * sizeof(EffectInfoBase::InParameterVersion1)) ||
        !output_has(u64{effect_count} * sizeof(EffectInfoBase::OutStatusVersion1))) {
        return ResultInvalidUpdateInfo;
    }
    std::span<const EffectInfoBase::InParameterVersion1> in_params{
        reinterpret_cast<const EffectInfoBase::InParameterVersion1*>(input), effect_count};
    std::span<EffectInfoBase::OutStatusVersion1> out_params{
        reinterpret_cast<EffectInfoBase::OutStatusVersion1*>(output), effect_count};

    for (u32 i = 0; i < effect_count; i++) {
        auto effect_info{&effect_context.get_info(i)};
        if (effect_info->get_type() != in_params[i].type) {
            effect_info->force_unmap_buffers(pool_mapper);
            reset_effect(effect_info, in_params[i].type);
        }

        BehaviorInfo::ErrorInfo error_info{};
        effect_info->update(error_info, in_params[i], pool_mapper);
        if (error_info.error_code.is_error()) {
            behaviour.append_error(error_info);
        }

        effect_info->store_status(out_params[i], renderer_active);
    }

    auto consumed_input_size{effect_count *
                             static_cast<u32>(sizeof(EffectInfoBase::InParameterVersion1))};
    auto consumed_output_size{effect_count *
                              static_cast<u32>(sizeof(EffectInfoBase::OutStatusVersion1))};
    if (consumed_input_size != in_header->effects_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect effects size, header size={}, consumed={}",
                  in_header->effects_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    out_header->effects_size = consumed_output_size;
    out_header->size += consumed_output_size;
    input += consumed_input_size;
    output += consumed_output_size;

    return ResultSuccess;
}

Result InfoUpdater::update_effects_version2(EffectContext& effect_context, const bool renderer_active,
                                          std::span<MemoryPoolInfo> memory_pools,
                                          const u32 memory_pool_count) {
    PoolMapper pool_mapper(memory, memory_pools, memory_pool_count,
                           behaviour.is_memory_force_mapping_enabled());

    const auto effect_count{effect_context.get_count()};

    if (!input_has(u64{effect_count} * sizeof(EffectInfoBase::InParameterVersion2)) ||
        !output_has(u64{effect_count} * sizeof(EffectInfoBase::OutStatusVersion2))) {
        return ResultInvalidUpdateInfo;
    }
    std::span<const EffectInfoBase::InParameterVersion2> in_params{
        reinterpret_cast<const EffectInfoBase::InParameterVersion2*>(input), effect_count};
    std::span<EffectInfoBase::OutStatusVersion2> out_params{
        reinterpret_cast<EffectInfoBase::OutStatusVersion2*>(output), effect_count};

    for (u32 i = 0; i < effect_count; i++) {
        auto effect_info{&effect_context.get_info(i)};
        if (effect_info->get_type() != in_params[i].type) {
            effect_info->force_unmap_buffers(pool_mapper);
            reset_effect(effect_info, in_params[i].type);
        }

        BehaviorInfo::ErrorInfo error_info{};
        effect_info->update(error_info, in_params[i], pool_mapper);

        if (error_info.error_code.is_error()) {
            behaviour.append_error(error_info);
        }

        effect_info->store_status(out_params[i], renderer_active);

        if (in_params[i].is_new) {
            effect_info->initialize_result_state(effect_context.get_dsp_shared_result_state(i));
            effect_info->initialize_result_state(effect_context.get_result_state(i));
        }
        effect_info->update_result_state(out_params[i].result_state,
                                       effect_context.get_result_state(i));
    }

    auto consumed_input_size{effect_count *
                             static_cast<u32>(sizeof(EffectInfoBase::InParameterVersion2))};
    auto consumed_output_size{effect_count *
                              static_cast<u32>(sizeof(EffectInfoBase::OutStatusVersion2))};
    if (consumed_input_size != in_header->effects_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect effects size, header size={}, consumed={}",
                  in_header->effects_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    out_header->effects_size = consumed_output_size;
    out_header->size += consumed_output_size;
    input += consumed_input_size;
    output += consumed_output_size;

    return ResultSuccess;
}

Result InfoUpdater::update_mixes(MixContext& mix_context, const u32 mix_buffer_count,
                                EffectContext& effect_context, SplitterContext& splitter_context) {
    s32 mix_count{0};
    u32 consumed_input_size{0};
    u32 input_mix_size{0};

    if (behaviour.is_mix_in_parameter_dirty_only_update_supported()) {
        if (!input_has(sizeof(MixInfo::InDirtyParameter))) {
            return ResultInvalidUpdateInfo;
        }
        auto in_dirty_params{reinterpret_cast<const MixInfo::InDirtyParameter*>(input)};
        mix_count = in_dirty_params->count;

        // Validate against expected header size to ensure structure is correct
        if (mix_count < 0 || mix_count > 0x100) {
            AUDIO_LOG_ERROR("Invalid mix count from dirty parameter: count={}, magic={:#x}, expected_size={}",
                mix_count, in_dirty_params->magic, in_header->mix_size);
            return ResultInvalidUpdateInfo;
        }

        consumed_input_size += static_cast<u32>(sizeof(MixInfo::InDirtyParameter));
        input += sizeof(MixInfo::InDirtyParameter);
    } else {
        mix_count = mix_context.get_count();
    }

    input_mix_size = static_cast<u32>(mix_count * sizeof(MixInfo::InParameter));
    consumed_input_size += input_mix_size;


    if (mix_buffer_count == 0) {
        return ResultInvalidUpdateInfo;
    }

    if (!input_has(input_mix_size)) {
        return ResultInvalidUpdateInfo;
    }

    std::span<const MixInfo::InParameter> in_params{
        reinterpret_cast<const MixInfo::InParameter*>(input), static_cast<size_t>(mix_count)};

    u32 total_buffer_count{0};
    for (s32 i = 0; i < mix_count; i++) {
        const auto& params{in_params[i]};

        if (params.in_use) {
            total_buffer_count += params.buffer_count;
            if (params.dest_mix_id > static_cast<s32>(mix_context.get_count()) &&
                params.dest_mix_id != UnusedMixId && params.mix_id != FinalMixId) {
                return ResultInvalidUpdateInfo;
            }
        }
    }

    if (total_buffer_count > mix_buffer_count) {
        return ResultInvalidUpdateInfo;
    }

    bool mix_dirty{false};
    for (s32 i = 0; i < mix_count; i++) {
        const auto& params{in_params[i]};

        s32 mix_id{i};
        if (behaviour.is_mix_in_parameter_dirty_only_update_supported()) {
            mix_id = params.mix_id;
        }

        if (mix_id < 0 || mix_id >= static_cast<s32>(mix_context.get_count())) {
            return ResultInvalidUpdateInfo;
        }
        auto mix_info{mix_context.get_info(mix_id)};
        if (mix_info->in_use != params.in_use) {
            mix_info->in_use = params.in_use;
            if (!params.in_use) {
                mix_info->clear_effect_processing_order();
            }
            mix_dirty = true;
        }

        if (params.in_use) {
            mix_dirty |= mix_info->update(mix_context.get_edge_matrix(), params, effect_context,
                                          splitter_context, behaviour);
        }
    }

    if (mix_dirty) {
        if (behaviour.is_splitter_supported() && splitter_context.using_splitter()) {
            if (!mix_context.t_sort_info(splitter_context)) {
                return ResultInvalidUpdateInfo;
            }
        } else {
            mix_context.sort_info();
        }
    }

    if (consumed_input_size != in_header->mix_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect mixes size, header size={}, consumed={}",
                  in_header->mix_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    input += mix_count * sizeof(MixInfo::InParameter);

    return ResultSuccess;
}

Result InfoUpdater::update_sinks(SinkContext& sink_context, std::span<MemoryPoolInfo> memory_pools,
                                const u32 memory_pool_count) {
    PoolMapper pool_mapper(memory, memory_pools, memory_pool_count,
                           behaviour.is_memory_force_mapping_enabled());

    if (!input_has(u64{sink_context.get_count()} * sizeof(SinkInfoBase::InParameter)) ||
        !output_has(u64{sink_context.get_count()} * sizeof(SinkInfoBase::OutStatus))) {
        return ResultInvalidUpdateInfo;
    }
    std::span<const SinkInfoBase::InParameter> in_params{
        reinterpret_cast<const SinkInfoBase::InParameter*>(input), memory_pool_count};
    std::span<SinkInfoBase::OutStatus> out_params{
        reinterpret_cast<SinkInfoBase::OutStatus*>(output), memory_pool_count};

    const auto sink_count{sink_context.get_count()};

    for (u32 i = 0; i < sink_count; i++) {
        const auto& params{in_params[i]};
        auto sink_info{sink_context.get_info(i)};

        if (sink_info->get_type() != params.type) {
            sink_info->clean_up();
            switch (params.type) {
            case SinkInfoBase::Type::Invalid:
                std::construct_at<SinkInfoBase>(reinterpret_cast<SinkInfoBase*>(sink_info));
                break;
            case SinkInfoBase::Type::DeviceSink:
                std::construct_at<DeviceSinkInfo>(reinterpret_cast<DeviceSinkInfo*>(sink_info));
                break;
            case SinkInfoBase::Type::CircularBufferSink:
                std::construct_at<CircularBufferSinkInfo>(
                    reinterpret_cast<CircularBufferSinkInfo*>(sink_info));
                break;
            default:
                AUDIO_LOG_ERROR("Invalid sink type {}", static_cast<u32>(params.type));
                break;
            }
        }

        BehaviorInfo::ErrorInfo error_info{};
        sink_info->update(error_info, out_params[i], params, pool_mapper);

        if (error_info.error_code.is_error()) {
            behaviour.append_error(error_info);
        }
    }

    const auto consumed_input_size{sink_count *
                                   static_cast<u32>(sizeof(SinkInfoBase::InParameter))};
    const auto consumed_output_size{sink_count * static_cast<u32>(sizeof(SinkInfoBase::OutStatus))};
    if (consumed_input_size != in_header->sinks_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect sinks size, header size={}, consumed={}",
                  in_header->sinks_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    input += consumed_input_size;
    output += consumed_output_size;
    out_header->sinks_size = consumed_output_size;
    out_header->size += consumed_output_size;

    return ResultSuccess;
}

Result InfoUpdater::update_memory_pools(std::span<MemoryPoolInfo> memory_pools,
                                      const u32 memory_pool_count) {
    PoolMapper pool_mapper(memory, memory_pools, memory_pool_count,
                           behaviour.is_memory_force_mapping_enabled());
    if (!input_has(u64{memory_pool_count} * sizeof(MemoryPoolInfo::InParameter)) ||
        !output_has(u64{memory_pool_count} * sizeof(MemoryPoolInfo::OutStatus))) {
        return ResultInvalidUpdateInfo;
    }
    std::span<const MemoryPoolInfo::InParameter> in_params{
        reinterpret_cast<const MemoryPoolInfo::InParameter*>(input), memory_pool_count};
    std::span<MemoryPoolInfo::OutStatus> out_params{
        reinterpret_cast<MemoryPoolInfo::OutStatus*>(output), memory_pool_count};

    for (size_t i = 0; i < memory_pool_count; i++) {
        auto state{pool_mapper.update(memory_pools[i], in_params[i], out_params[i])};
        if (state != MemoryPoolInfo::ResultState::Success &&
            state != MemoryPoolInfo::ResultState::BadParam &&
            state != MemoryPoolInfo::ResultState::MapFailed &&
            state != MemoryPoolInfo::ResultState::InUse) {
            AUDIO_LOG_WARNING("Invalid ResultState from updating memory pools");
            return ResultInvalidUpdateInfo;
        }
    }

    const auto consumed_input_size{memory_pool_count *
                                   static_cast<u32>(sizeof(MemoryPoolInfo::InParameter))};
    const auto consumed_output_size{memory_pool_count *
                                    static_cast<u32>(sizeof(MemoryPoolInfo::OutStatus))};
    if (consumed_input_size != in_header->memory_pool_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect memory pool size, header size={}, consumed={}",
                  in_header->memory_pool_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    input += consumed_input_size;
    output += consumed_output_size;
    out_header->memory_pool_size = consumed_output_size;
    out_header->size += consumed_output_size;
    return ResultSuccess;
}

Result InfoUpdater::update_performance_buffer(std::span<u8> performance_output,
                                            const u64 performance_output_size,
                                            PerformanceManager* performance_manager) {
    if (!input_has(sizeof(PerformanceManager::InParameter)) ||
        !output_has(sizeof(PerformanceManager::OutStatus))) {
        return ResultInvalidUpdateInfo;
    }
    auto in_params{reinterpret_cast<const PerformanceManager::InParameter*>(input)};
    auto out_params{reinterpret_cast<PerformanceManager::OutStatus*>(output)};

    if (performance_manager != nullptr) {
        out_params->history_size =
            performance_manager->copy_histories(performance_output.data(), performance_output_size);
        performance_manager->set_detail_target(in_params->target_node_id);
    } else {
        out_params->history_size = 0;
    }

    const auto consumed_input_size{static_cast<u32>(sizeof(PerformanceManager::InParameter))};
    const auto consumed_output_size{static_cast<u32>(sizeof(PerformanceManager::OutStatus))};
    if (consumed_input_size != in_header->performance_buffer_size) {
        AUDIO_LOG_ERROR("Consumed an incorrect performance size, header size={}, consumed={}",
                  in_header->performance_buffer_size, consumed_input_size);
        return ResultInvalidUpdateInfo;
    }

    input += consumed_input_size;
    output += consumed_output_size;
    out_header->performance_buffer_size = consumed_output_size;
    out_header->size += consumed_output_size;
    return ResultSuccess;
}

Result InfoUpdater::update_behavior_info(BehaviorInfo& behaviour_) {
    if (!input_has(sizeof(BehaviorInfo::InParameter))) {
        return ResultInvalidUpdateInfo;
    }
    const auto in_params{reinterpret_cast<const BehaviorInfo::InParameter*>(input)};

    if (!check_valid_revision(in_params->revision)) {
        return ResultInvalidUpdateInfo;
    }

    if (in_params->revision != behaviour_.get_user_revision()) {
        return ResultInvalidUpdateInfo;
    }

    behaviour_.clear_error();
    behaviour_.update_flags(in_params->flags);

    if (in_header->behaviour_size != sizeof(BehaviorInfo::InParameter)) {
        return ResultInvalidUpdateInfo;
    }

    input += sizeof(BehaviorInfo::InParameter);
    return ResultSuccess;
}

Result InfoUpdater::update_error_info(const BehaviorInfo& behaviour_) {
    if (!output_has(sizeof(BehaviorInfo::OutStatus))) {
        return ResultInvalidUpdateInfo;
    }
    auto out_params{reinterpret_cast<BehaviorInfo::OutStatus*>(output)};
    behaviour_.copy_error_info(out_params->errors, out_params->error_count);

    const auto consumed_output_size{static_cast<u32>(sizeof(BehaviorInfo::OutStatus))};

    output += consumed_output_size;
    out_header->behaviour_size = consumed_output_size;
    out_header->size += consumed_output_size;
    return ResultSuccess;
}

Result InfoUpdater::update_splitter_info(SplitterContext& splitter_context) {
    if (!input_has(in_header->splitter_size)) {
        return ResultInvalidUpdateInfo;
    }
    if (!splitter_context.update(input, in_header->splitter_size)) {
        return ResultInvalidUpdateInfo;
    }

    input += in_header->splitter_size;

    return ResultSuccess;
}

Result InfoUpdater::update_renderer_info(const u64 elapsed_frames) {
    struct RenderInfo {
        /* 0x00 */ u64 frames_elapsed;
        /* 0x08 */ char unk08[0x8];
    };
    static_assert(sizeof(RenderInfo) == 0x10, "RenderInfo has the wrong size!");

    if (!output_has(sizeof(RenderInfo))) {
        return ResultInvalidUpdateInfo;
    }
    auto out_params{reinterpret_cast<RenderInfo*>(output)};
    out_params->frames_elapsed = elapsed_frames;

    const auto consumed_output_size{static_cast<u32>(sizeof(RenderInfo))};

    output += consumed_output_size;
    out_header->render_info_size = consumed_output_size;
    out_header->size += consumed_output_size;

    return ResultSuccess;
}

Result InfoUpdater::check_consumed_size() {
    if (CpuAddr(input) - CpuAddr(input_origin.data()) != expected_input_size) {
        return ResultInvalidUpdateInfo;
    } else if (CpuAddr(output) - CpuAddr(output_origin.data()) != expected_output_size) {
        return ResultInvalidUpdateInfo;
    }
    return ResultSuccess;
}

} // namespace armigo::audio::renderer
