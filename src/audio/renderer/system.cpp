// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <chrono>
#include <span>

#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/common/common.h"
#include "renderer/common/feature_support.h"
#include "renderer/common/workbuffer_allocator.h"
#include "renderer/behavior/info_updater.h"
#include "renderer/command/command_buffer.h"
#include "renderer/command/command_generator.h"
#include "renderer/command/command_list_header.h"
#include "renderer/effect/effect_info_base.h"
#include "renderer/effect/effect_result_state.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/memory/pool_mapper.h"
#include "renderer/mix/mix_info.h"
#include "renderer/nodes/edge_matrix.h"
#include "renderer/nodes/node_states.h"
#include "renderer/sink/sink_info_base.h"
#include "renderer/system.h"
#include "renderer/upsampler/upsampler_info.h"
#include "renderer/voice/voice_channel_resource.h"
#include "renderer/voice/voice_info.h"
#include "renderer/voice/voice_state.h"
#include "renderer/common/support.h"

#include <armigo/audio/pcm_output.h>

#include <algorithm>
#include <cstring>

namespace armigo::audio::renderer {

u64 System::get_work_buffer_size(const AudioRendererParameterInternal& params) {
    BehaviorInfo behavior;
    behavior.set_user_lib_revision(params.revision);

    u64 size{0};

    size += align_up(params.mixes * sizeof(s32), 0x40);
    size += params.sub_mixes * MaxEffects * sizeof(s32);
    size += (params.sub_mixes + 1) * sizeof(MixInfo);
    size += params.voices * (sizeof(VoiceInfo) + sizeof(VoiceChannelResource) + sizeof(VoiceState));
    size += align_up((params.sub_mixes + 1) * sizeof(MixInfo*), 0x10);
    size += align_up(params.voices * sizeof(VoiceInfo*), 0x10);
    size += align_up(((params.sinks + params.sub_mixes) * TargetSampleCount * sizeof(s32) +
                             params.sample_count * sizeof(s32)) *
                                (params.mixes + MaxChannels),
                            0x40);

    if (behavior.is_splitter_supported()) {
        const auto node_size{NodeStates::get_work_buffer_size(params.sub_mixes + 1)};
        const auto edge_size{EdgeMatrix::get_work_buffer_size(params.sub_mixes + 1)};
        size += align_up(node_size + edge_size, 0x10);
    }

    size += SplitterContext::calc_work_buffer_size(behavior, params);
    size += (params.effects + params.voices * MaxWaveBuffers) * sizeof(MemoryPoolInfo);

    if (behavior.is_effect_info_version2_supported()) {
        size += params.effects * sizeof(EffectResultState);
    }
    size += 0x50;

    size = align_up(size, 0x40);

    size += (params.sinks + params.sub_mixes) * sizeof(UpsamplerInfo);
    size += params.effects * sizeof(EffectInfoBase);
    size += align_up(params.voices * sizeof(VoiceState), 0x40);
    size += params.sinks * sizeof(SinkInfoBase);

    if (behavior.is_effect_info_version2_supported()) {
        size += params.effects * sizeof(EffectResultState);
    }

    if (params.perf_frames > 0) {
        auto perf_size{PerformanceManager::get_required_buffer_size_for_performance_metrics_per_frame(
            behavior, params)};
        size += align_up(perf_size * (params.perf_frames + 1) + 0xC0, 0x100);
    }

    if (behavior.is_variadic_command_buffer_size_supported()) {
        size += CommandGenerator::calculate_command_buffer_size(behavior, params) + (0x40 - 1) * 2;
    } else {
        size += 0x18000 + (0x40 - 1) * 2;
    }

    size = align_up(size, 0x1000);
    return size;
}

System::~System() { finalize(); }

Result System::initialize(const AudioRendererParameterInternal& params,
                          const u64 transfer_memory_size, std::shared_ptr<GuestMemoryPins> pins,
                          std::shared_ptr<PcmOutput> output_, const s32 session_id_) {
    if (!check_valid_revision(params.revision)) {
        return ResultInvalidRevision;
    }

    if (get_work_buffer_size(params) > transfer_memory_size) {
        return ResultInsufficientBuffer;
    }

    if (!pins) {
        return ResultInvalidHandle;
    }

    behavior.set_user_lib_revision(params.revision);

    guest_memory.reset(std::move(pins));
    output = std::move(output_);
    session_id = session_id_;

    sample_rate = params.sample_rate;
    sample_count = params.sample_count;
    mix_buffer_count = static_cast<s16>(params.mixes);
    voice_channels = MaxChannels;
    upsampler_count = params.sinks + params.sub_mixes;
    memory_pool_count = params.effects + params.voices * MaxWaveBuffers;
    render_device = params.rendering_device;
    execution_mode = params.execution_mode;

    // The renderer's workbuffer lives in host memory: nothing in it is shared
    // with the guest, so the guest-supplied transfer memory is left untouched.
    workbuffer = std::make_unique<u8[]>(transfer_memory_size);
    workbuffer_size = transfer_memory_size;

    PoolMapper pool_mapper(&guest_memory, false);
    pool_mapper.initialize_system_pool(memory_pool_info, workbuffer.get(), workbuffer_size);

    WorkbufferAllocator allocator({workbuffer.get(), workbuffer_size}, workbuffer_size);

    samples_workbuffer =
        allocator.allocate<s32>((voice_channels + mix_buffer_count) * sample_count, 0x10);
    if (samples_workbuffer.empty()) {
        return ResultInsufficientBuffer;
    }

    auto upsampler_workbuffer{allocator.allocate<s32>(
        (voice_channels + mix_buffer_count) * TargetSampleCount * upsampler_count, 0x10)};
    if (upsampler_workbuffer.empty()) {
        return ResultInsufficientBuffer;
    }

    depop_buffer =
        allocator.allocate<s32>(align_up(static_cast<u32>(mix_buffer_count), 0x40), 0x40);
    if (depop_buffer.empty()) {
        return ResultInsufficientBuffer;
    }

    // invalidate samples_workbuffer DSP cache

    auto voice_infos{allocator.allocate<VoiceInfo>(params.voices, 0x10)};
    for (auto& voice_info : voice_infos) {
        std::construct_at<VoiceInfo>(&voice_info);
    }

    if (voice_infos.empty()) {
        return ResultInsufficientBuffer;
    }

    auto sorted_voice_infos{allocator.allocate<VoiceInfo*>(params.voices, 0x10)};
    if (sorted_voice_infos.empty()) {
        return ResultInsufficientBuffer;
    }

    std::memset(sorted_voice_infos.data(), 0, sorted_voice_infos.size_bytes());

    auto voice_channel_resources{allocator.allocate<VoiceChannelResource>(params.voices, 0x10)};
    u32 i{0};
    for (auto& voice_channel_resource : voice_channel_resources) {
        std::construct_at<VoiceChannelResource>(&voice_channel_resource, i++);
    }

    if (voice_channel_resources.empty()) {
        return ResultInsufficientBuffer;
    }

    auto voice_cpu_states{allocator.allocate<VoiceState>(params.voices, 0x10)};
    if (voice_cpu_states.empty()) {
        return ResultInsufficientBuffer;
    }

    for (auto& voice_state : voice_cpu_states) {
        voice_state = {};
    }

    auto mix_infos{allocator.allocate<MixInfo>(params.sub_mixes + 1, 0x10)};

    if (mix_infos.empty()) {
        return ResultInsufficientBuffer;
    }

    u32 effect_process_order_count{0};
    std::span<s32> effect_process_order_buffer{};

    if (params.effects > 0) {
        effect_process_order_count = params.effects * (params.sub_mixes + 1);
        effect_process_order_buffer = allocator.allocate<s32>(effect_process_order_count, 0x10);
        if (effect_process_order_buffer.empty()) {
            return ResultInsufficientBuffer;
        }
    }

    i = 0;
    for (auto& mix_info : mix_infos) {
        std::construct_at<MixInfo>(
            &mix_info, effect_process_order_buffer.subspan(i * params.effects, params.effects),
            params.effects, this->behavior);
        i++;
    }

    auto sorted_mix_infos{allocator.allocate<MixInfo*>(params.sub_mixes + 1, 0x10)};
    if (sorted_mix_infos.empty()) {
        return ResultInsufficientBuffer;
    }

    std::memset(sorted_mix_infos.data(), 0, sorted_mix_infos.size_bytes());

    if (behavior.is_splitter_supported()) {
        u64 node_state_size{NodeStates::get_work_buffer_size(params.sub_mixes + 1)};
        u64 edge_matrix_size{EdgeMatrix::get_work_buffer_size(params.sub_mixes + 1)};

        auto node_states_workbuffer{allocator.allocate<u8>(node_state_size, 1)};
        auto edge_matrix_workbuffer{allocator.allocate<u8>(edge_matrix_size, 1)};

        if (node_states_workbuffer.empty() || edge_matrix_workbuffer.size() == 0) {
            return ResultInsufficientBuffer;
        }

        mix_context.initialize(sorted_mix_infos, mix_infos, params.sub_mixes + 1,
                               effect_process_order_buffer, effect_process_order_count,
                               node_states_workbuffer, node_state_size, edge_matrix_workbuffer,
                               edge_matrix_size);
    } else {
        mix_context.initialize(sorted_mix_infos, mix_infos, params.sub_mixes + 1,
                               effect_process_order_buffer, effect_process_order_count, {}, 0, {},
                               0);
    }

    upsampler_manager = allocator.allocate<UpsamplerManager>(1, 0x10).data();
    if (upsampler_manager == nullptr) {
        return ResultInsufficientBuffer;
    }

    memory_pool_workbuffer = allocator.allocate<MemoryPoolInfo>(memory_pool_count, 0x10);
    for (auto& memory_pool : memory_pool_workbuffer) {
        std::construct_at<MemoryPoolInfo>(&memory_pool, MemoryPoolInfo::Location::DSP);
    }

    if (memory_pool_workbuffer.empty() && memory_pool_count > 0) {
        return ResultInsufficientBuffer;
    }

    if (!splitter_context.initialize(behavior, params, allocator)) {
        return ResultInsufficientBuffer;
    }

    std::span<EffectResultState> effect_result_states_cpu{};
    if (behavior.is_effect_info_version2_supported() && params.effects > 0) {
        effect_result_states_cpu = allocator.allocate<EffectResultState>(params.effects, 0x10);
        if (effect_result_states_cpu.empty()) {
            return ResultInsufficientBuffer;
        }
        std::memset(effect_result_states_cpu.data(), 0, effect_result_states_cpu.size_bytes());
    }

    allocator.align(0x40);

    unk_2B0 = allocator.get_size() - allocator.get_current_offset();
    unk_2A8 = {&workbuffer[allocator.get_current_offset()], unk_2B0};

    upsampler_infos = allocator.allocate<UpsamplerInfo>(upsampler_count, 0x40);
    for (auto& upsampler_info : upsampler_infos) {
        std::construct_at<UpsamplerInfo>(&upsampler_info);
    }

    std::construct_at<UpsamplerManager>(upsampler_manager, upsampler_count, upsampler_infos,
                                        upsampler_workbuffer);

    if (upsampler_infos.empty()) {
        return ResultInsufficientBuffer;
    }

    auto effect_infos{allocator.allocate<EffectInfoBase>(params.effects, 0x40)};
    for (auto& effect_info : effect_infos) {
        std::construct_at<EffectInfoBase>(&effect_info);
    }

    if (effect_infos.empty() && params.effects > 0) {
        return ResultInsufficientBuffer;
    }

    std::span<EffectResultState> effect_result_states_dsp{};
    if (behavior.is_effect_info_version2_supported() && params.effects > 0) {
        effect_result_states_dsp = allocator.allocate<EffectResultState>(params.effects, 0x40);
        if (effect_result_states_dsp.empty()) {
            return ResultInsufficientBuffer;
        }
        std::memset(effect_result_states_dsp.data(), 0, effect_result_states_dsp.size_bytes());
    }

    effect_context.initialize(effect_infos, params.effects, effect_result_states_cpu,
                              effect_result_states_dsp, effect_result_states_dsp.size());

    auto sinks{allocator.allocate<SinkInfoBase>(params.sinks, 0x10)};
    for (auto& sink : sinks) {
        std::construct_at<SinkInfoBase>(&sink);
    }

    if (sinks.empty()) {
        return ResultInsufficientBuffer;
    }

    sink_context.initialize(sinks, params.sinks);

    auto voice_dsp_states{allocator.allocate<VoiceState>(params.voices, 0x40)};
    if (voice_dsp_states.empty()) {
        return ResultInsufficientBuffer;
    }

    for (auto& voice_state : voice_dsp_states) {
        voice_state = {};
    }

    voice_context.initialize(sorted_voice_infos, voice_infos, voice_channel_resources,
                             voice_cpu_states, voice_dsp_states, params.voices);

    if (params.perf_frames > 0) {
        const auto perf_workbuffer_size{
            PerformanceManager::get_required_buffer_size_for_performance_metrics_per_frame(behavior,
                                                                                   params) *
                (params.perf_frames + 1) +
            0xC};
        performance_workbuffer = allocator.allocate<u8>(perf_workbuffer_size, 0x40);
        if (performance_workbuffer.empty()) {
            return ResultInsufficientBuffer;
        }
        std::memset(performance_workbuffer.data(), 0, performance_workbuffer.size_bytes());
        performance_manager.initialize(performance_workbuffer, performance_workbuffer.size_bytes(),
                                       params, behavior, memory_pool_info);
    }

    render_time_limit_percent = 100;
    drop_voice = params.voice_drop_enabled && params.execution_mode == ExecutionMode::Auto;
    drop_voice_param = 1.0f;
    num_voices_dropped = 0;

    allocator.align(0x40);
    command_workbuffer_size = allocator.get_remaining_size();
    command_workbuffer = allocator.allocate<u8>(command_workbuffer_size, 0x40);
    if (command_workbuffer.empty()) {
        return ResultInsufficientBuffer;
    }

    command_buffer_size = 0;
    reset_command_buffers = true;

    // The workbuffer is host memory shared with nothing, so no data cache flush is needed.

    // Command processing time is not estimated; see CommandProcessingTimeEstimator.

    initialized = true;
    return ResultSuccess;
}

void System::finalize() {
    if (!initialized) {
        return;
    }

    if (active) {
        stop();
    }

    std::scoped_lock l{lock};
    // The system pool is the renderer's own host workbuffer: nothing to unpin.
    memory_pool_info.set_cpu_address(0, 0);
    memory_pool_info.set_dsp_address(0);

    PoolMapper pool_mapper(&guest_memory, false);
    pool_mapper.clear_use_state(memory_pool_workbuffer, memory_pool_count);
    for (auto& memory_pool : memory_pool_workbuffer) {
        if (memory_pool.is_mapped()) {
            pool_mapper.unmap(memory_pool);
        }
    }
    guest_memory.unmap_all();
    initialized = false;
}

void System::start() {
    std::scoped_lock l{lock};
    frames_elapsed = 0;
    state = State::Started;
    active = true;
    if (output) {
        output->set_active(true);
    }
}

void System::stop() {
    // Rendering happens under the lock, so once it is released no frame is in
    // flight and none starts until the next start.
    std::scoped_lock l{lock};
    state = State::Stopped;
    active = false;
    if (output) {
        output->set_active(false);
    }
}

Result System::update(std::span<const u8> input, std::span<u8> performance,
                      std::span<u8> output_status, const std::function<void()>& clear_event) {
    std::scoped_lock l{lock};

    const auto start_time{host_time_ns()};
    std::memset(output_status.data(), 0, output_status.size());

    InfoUpdater info_updater(input, output_status, &guest_memory, behavior);

    auto result{info_updater.update_behavior_info(behavior)};
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update BehaviorInfo!");
        return result;
    }

    result = info_updater.update_memory_pools(memory_pool_workbuffer, memory_pool_count);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update MemoryPools!");
        return result;
    }

    result = info_updater.update_voice_channel_resources(voice_context);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update VoiceChannelResources!");
        return result;
    }

    result = info_updater.update_voices(voice_context, memory_pool_workbuffer, memory_pool_count);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update Voices!");
        return result;
    }

    result = info_updater.update_effects(effect_context, active, memory_pool_workbuffer,
                                         memory_pool_count);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update Effects!");
        return result;
    }

    if (behavior.is_splitter_supported()) {
        result = info_updater.update_splitter_info(splitter_context);
        if (result.is_error()) {
            AUDIO_LOG_ERROR("Failed to update SplitterInfo!");
            return result;
        }
    }

    result =
        info_updater.update_mixes(mix_context, mix_buffer_count, effect_context, splitter_context);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update Mixes!");
        return result;
    }

    result = info_updater.update_sinks(sink_context, memory_pool_workbuffer, memory_pool_count);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update Sinks!");
        return result;
    }

    PerformanceManager* perf_manager{nullptr};
    if (performance_manager.is_initialized()) {
        perf_manager = &performance_manager;
    }

    result =
        info_updater.update_performance_buffer(performance, performance.size_bytes(), perf_manager);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update PerformanceBuffer!");
        return result;
    }

    result = info_updater.update_error_info(behavior);
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Failed to update ErrorInfo!");
        return result;
    }

    if (behavior.is_elapsed_frame_count_supported()) {
        result = info_updater.update_renderer_info(frames_elapsed);
        if (result.is_error()) {
            AUDIO_LOG_ERROR("Failed to update RendererInfo!");
            return result;
        }
    }

    result = info_updater.check_consumed_size();
    if (result.is_error()) {
        AUDIO_LOG_ERROR("Invalid consume size!");
        return result;
    }

    if (clear_event) {
        clear_event();
    }
    num_times_updated++;

    const auto end_time{host_time_ns()};
    ticks_spent_updating += end_time - start_time;

    return ResultSuccess;
}

u32 System::get_rendering_time_limit() const {
    std::scoped_lock l{lock};
    return render_time_limit_percent;
}

void System::set_rendering_time_limit(u32 limit) {
    std::scoped_lock l{lock};
    render_time_limit_percent = limit;
}

u32 System::get_session_id() const {
    return session_id;
}

u32 System::get_sample_rate() const {
    return sample_rate;
}

u32 System::get_sample_count() const {
    return sample_count;
}

u32 System::get_mix_buffer_count() const {
    return mix_buffer_count;
}

ExecutionMode System::get_execution_mode() const {
    return execution_mode;
}

u32 System::get_rendering_device() const {
    return render_device;
}

bool System::is_active() const {
    return active;
}

bool System::prepare_frame() {
    if (!initialized || !active) {
        return false;
    }

    // The DSP part runs synchronously after generation, so no commands of the
    // previous list remain and a new list is generated every frame.
    const auto command_size{generate_command(command_workbuffer, command_workbuffer_size)};

    auto translated_addr{
        memory_pool_info.translate(CpuAddr(command_workbuffer.data()), command_size)};

    command_list_processor.initialize(guest_memory, translated_addr, command_size, output.get());
    // The host output already starts empty when the renderer starts, which is
    // what the first list's buffer reset asks for.
    reset_command_buffers = false;
    command_buffer_size = command_size;
    signalled_tick = host_time_ns();
    return true;
}

void System::process_frame() {
    // Processing time limits are not enforced: the list is processed whole.
    const auto start_time{host_time_us()};
    command_list_processor.process(static_cast<u32>(session_id));
    render_time_taken_us = host_time_us() - start_time;
}

u64 System::generate_command(std::span<u8> in_command_buffer,
                            [[maybe_unused]] u64 command_buffer_size_) {
    PoolMapper::clear_use_state(memory_pool_workbuffer, memory_pool_count);
    const auto start_time{host_time_ns()};

    auto command_list_header{reinterpret_cast<CommandListHeader*>(in_command_buffer.data())};

    command_list_header->buffer_count = static_cast<s16>(voice_channels + mix_buffer_count);
    command_list_header->sample_count = sample_count;
    command_list_header->sample_rate = sample_rate;
    command_list_header->samples_buffer = samples_workbuffer;

    const auto performance_initialized{performance_manager.is_initialized()};
    if (performance_initialized) {
        performance_manager.tap_frame(adsp_behind, num_voices_dropped, render_start_tick);
        adsp_behind = false;
        num_voices_dropped = 0;
        render_start_tick = 0;
    }

    // The host output device is stereo.
    s8 channel_count{2};

    AudioRendererSystemContext render_context{
        .session_id = session_id,
        .channels = channel_count,
        .mix_buffer_count = mix_buffer_count,
        .behavior = &behavior,
        .depop_buffer = depop_buffer,
        .upsampler_manager = upsampler_manager,
        .memory_pool_info = &memory_pool_info,
    };

    CommandBuffer command_buffer{
        .command_list = in_command_buffer,
        .sample_count = sample_count,
        .sample_rate = sample_rate,
        .size = sizeof(CommandListHeader),
        .count = 0,
        .estimated_process_time = 0,
        .memory_pool = &memory_pool_info,
        .time_estimator = &command_processing_time_estimator,
        .behavior = &behavior,
    };

    PerformanceManager* perf_manager{nullptr};
    if (performance_initialized) {
        perf_manager = &performance_manager;
    }

    CommandGenerator command_generator{command_buffer, *command_list_header, render_context,
                                       voice_context,  mix_context,          effect_context,
                                       sink_context,   splitter_context,     perf_manager};

    voice_context.sort_info();
    command_generator.generate_voice_commands();

    const auto start_estimated_time{drop_voice_param *
                                    static_cast<f32>(command_buffer.estimated_process_time)};

    command_generator.generate_sub_mix_commands();
    command_generator.generate_final_mix_commands();
    command_generator.generate_sink_commands();

    if (drop_voice) {
        f32 time_limit_percent{70.0f};
        if (render_context.behavior->is_audio_renderer_processing_time_limit80_percent_supported()) {
            time_limit_percent = 80.0f;
        } else if (render_context.behavior
                       ->is_audio_renderer_processing_time_limit75_percent_supported()) {
            time_limit_percent = 75.0f;
        } else {
            // result is ignored
            render_context.behavior->is_audio_renderer_processing_time_limit70_percent_supported();
            time_limit_percent = 70.0f;
        }

        const auto end_estimated_time{drop_voice_param *
                                      static_cast<f32>(command_buffer.estimated_process_time)};

        const auto dsp_time_limit{((time_limit_percent / 100.0f) * 2'880'000.0f) *
                                  (static_cast<f32>(render_time_limit_percent) / 100.0f)};

        const auto estimated_time{start_estimated_time - end_estimated_time};

        const auto time_limit{static_cast<u32>((std::max)(dsp_time_limit + estimated_time, 0.0f))};
        num_voices_dropped =
            drop_voices(command_buffer, static_cast<u32>(start_estimated_time), time_limit);
    }

    command_list_header->buffer_size = command_buffer.size;
    command_list_header->command_count = command_buffer.count;

    voice_context.update_state_by_dsp_shared();

    if (render_context.behavior->is_effect_info_version2_supported()) {
        effect_context.update_state_by_dsp_shared();
    }

    const auto end_time{host_time_ns()};
    total_ticks_elapsed += end_time - start_time;
    num_command_lists_generated++;
    render_start_tick = (1000 * render_time_taken_us) + signalled_tick;
    frames_elapsed++;

    return command_buffer.size;
}

f32 System::get_voice_drop_parameter() const {
    std::scoped_lock l{lock};
    return drop_voice_param;
}

void System::set_voice_drop_parameter(f32 voice_drop_) {
    std::scoped_lock l{lock};
    drop_voice_param = voice_drop_;
}

u32 System::drop_voices(CommandBuffer& command_buffer, u32 estimated_process_time, u32 time_limit) {
    u32 i{0};
    auto command_list{command_buffer.command_list.data() + sizeof(CommandListHeader)};
    ICommand* cmd{nullptr};

    // Find a first valid voice to drop
    while (i < command_buffer.count) {
        cmd = reinterpret_cast<ICommand*>(command_list);
        if (cmd->type == CommandId::Performance ||
            cmd->type == CommandId::DataSourcePcmInt16Version1 ||
            cmd->type == CommandId::DataSourcePcmInt16Version2 ||
            cmd->type == CommandId::DataSourcePcmFloatVersion1 ||
            cmd->type == CommandId::DataSourcePcmFloatVersion2 ||
            cmd->type == CommandId::DataSourceAdpcmVersion1 ||
            cmd->type == CommandId::DataSourceAdpcmVersion2) {
            break;
        }
        command_list += cmd->size;
        i++;
    }

    if (cmd == nullptr || command_buffer.count == 0 || i >= command_buffer.count) {
        return 0;
    }

    auto voices_dropped{0};
    while (i < command_buffer.count) {
        const auto node_id{cmd->node_id};
        const auto node_id_type{cmd->node_id >> 28};
        const auto node_id_base{(cmd->node_id >> 16) & 0xFFF};

        // If the new estimated process time falls below the limit, we're done dropping.
        if (estimated_process_time <= time_limit) {
            break;
        }

        if (node_id_type != 1) {
            break;
        }

        // Don't drop voices marked with the highest priority.
        auto& voice_info{voice_context.get_info(node_id_base)};
        if (voice_info.priority == HighestVoicePriority) {
            break;
        }

        voices_dropped++;
        voice_info.voice_dropped = true;

        // First iteration should drop the voice, and then iterate through all of the commands tied
        // to the voice. We don't need reverb on a voice which we've just removed, for example.
        // Depops can't be removed otherwise we'll introduce audio popping, and we don't
        // remove perf commands. Lower the estimated time for each command dropped.
        while (i < command_buffer.count && cmd->node_id == node_id) {
            if (cmd->type == CommandId::DepopPrepare) {
                cmd->enabled = true;
            } else if (cmd->enabled && cmd->type != CommandId::Performance) {
                cmd->enabled = false;
                estimated_process_time -= static_cast<u32>(
                    drop_voice_param * static_cast<f32>(cmd->estimated_process_time));
            }
            command_list += cmd->size;
            cmd = reinterpret_cast<ICommand*>(command_list);
            i++;
        }
        i++;
    }
    return voices_dropped;
}

} // namespace armigo::audio::renderer
