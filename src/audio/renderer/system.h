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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <span>

#include "renderer/behavior/behavior_info.h"
#include "renderer/command/command_processing_time_estimator.h"
#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/common/guest_memory.h"
#include "renderer/common/result.h"
#include "renderer/dsp/command_list_processor.h"
#include "renderer/effect/effect_context.h"
#include "renderer/memory/memory_pool_info.h"
#include "renderer/mix/mix_context.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/sink/sink_context.h"
#include "renderer/splitter/splitter_context.h"
#include "renderer/upsampler/upsampler_manager.h"
#include "renderer/voice/voice_context.h"

namespace armigo::audio {
class PcmOutput;
}

namespace armigo::audio::renderer {

class CommandBuffer;

// The audio renderer system: the guest-visible renderer state (voices, mixes,
// effects, sinks, splitters, memory pools, performance metrics) updated by
// RequestUpdate, and the per-frame command list generation and processing.
//
// The firmware splits a frame between the system manager thread, which
// generates the command list and signals the renderer event, and the DSP,
// which processes the list. Here render_frame does both on the caller's
// thread under the system lock, in the same order: generate, signal, process.
// Processing under the lock means RequestUpdate (which may detach pools) never
// runs while the DSP part reads guest memory.
class System {
    enum class State {
        Started = 0,
        Stopped = 2,
    };

public:
    System() = default;
    ~System();

    // Calculate the total size required for all audio render workbuffers.
    static u64 get_work_buffer_size(const AudioRendererParameterInternal& params);

    // Initialize the renderer system: allocate the workbuffers and initialize
    // everything to a default state, ready to receive a RequestUpdate. The
    // workbuffer is host memory of the guest-declared transfer memory size.
    Result initialize(const AudioRendererParameterInternal& params, u64 transfer_memory_size,
                      std::shared_ptr<GuestMemoryPins> pins, std::shared_ptr<PcmOutput> output,
                      s32 session_id);

    void finalize();
    void start();
    void stop();

    // Apply one RequestUpdate: input is the guest's update data, output and
    // performance receive the renderer's status.
    // `clear_event` runs under the system lock once the update succeeded, where
    // the firmware clears the renderer event.
    Result update(std::span<const u8> input, std::span<u8> performance, std::span<u8> output,
                  const std::function<void()>& clear_event = {});

    u32 get_rendering_time_limit() const;
    void set_rendering_time_limit(u32 limit);
    u32 get_session_id() const;
    u32 get_sample_rate() const;
    u32 get_sample_count() const;
    u32 get_mix_buffer_count() const;
    ExecutionMode get_execution_mode() const;
    u32 get_rendering_device() const;
    bool is_active() const;
    bool is_initialized() const { return initialized; }
    u64 get_frames_elapsed() const { return frames_elapsed; }

    // Render one frame: while active, generate the command list, report the
    // frame through `signal` (the renderer event), then process the list and
    // feed the device sink to the host output. Returns whether a frame was
    // rendered.
    template <typename Signal>
    bool render_frame(Signal&& signal) {
        std::scoped_lock l{lock};
        if (!prepare_frame()) return false;
        signal();
        process_frame();
        return true;
    }

    // Generate a list of commands for the DSP based on current state.
    u64 generate_command(std::span<u8> command_buffer, u64 command_buffer_size);

    // Drop voices if the estimated processing time exceeds the time limit.
    u32 drop_voices(CommandBuffer& command_buffer, u32 estimated_process_time, u32 time_limit);

    f32 get_voice_drop_parameter() const;
    void set_voice_drop_parameter(f32 voice_drop);

private:
    bool prepare_frame();
    void process_frame();

    bool initialized{};
    std::atomic<bool> active{};
    State state{State::Stopped};
    u32 sample_rate{};
    u32 sample_count{};
    s16 mix_buffer_count{};
    // Workbuffer for mix buffers, used by the DSP
    std::span<s32> samples_workbuffer{};
    // Depop samples for depopping commands
    std::span<s32> depop_buffer{};
    u32 memory_pool_count{};
    std::span<MemoryPoolInfo> memory_pool_workbuffer{};
    // System memory pool: the renderer's own workbuffer
    MemoryPoolInfo memory_pool_info{};
    std::span<u8> command_workbuffer{};
    u64 command_workbuffer_size{};
    u64 command_buffer_size{};
    UpsamplerManager* upsampler_manager{};
    std::span<UpsamplerInfo> upsampler_infos{};
    u32 upsampler_count{};
    VoiceContext voice_context{};
    MixContext mix_context{};
    EffectContext effect_context{};
    SinkContext sink_context{};
    SplitterContext splitter_context{};
    CommandProcessingTimeEstimator command_processing_time_estimator{};
    s32 session_id{};
    // Number of channels in use by voices
    s32 voice_channels{};
    mutable std::mutex lock{};
    // Guest memory reachable through attached pools
    GuestMemory guest_memory{};
    // Host PCM output for the device sink
    std::shared_ptr<PcmOutput> output{};
    // DSP-side processor of the generated command list
    CommandListProcessor command_list_processor{};
    PerformanceManager performance_manager{};
    std::span<u8> performance_workbuffer{};
    // Main workbuffer, from which all other workbuffers allocate
    std::unique_ptr<u8[]> workbuffer{};
    u64 workbuffer_size{};
    std::span<u8> unk_2A8{};
    u64 unk_2B0{};
    u32 render_time_limit_percent{};
    bool drop_voice{};
    // Should the host output have its queued frames flushed?
    bool reset_command_buffers{};
    ExecutionMode execution_mode{ExecutionMode::Auto};
    u32 render_device{};
    BehaviorInfo behavior{};
    u64 total_ticks_elapsed{};
    u64 ticks_spent_updating{};
    u64 num_command_lists_generated{};
    u64 num_times_updated{};
    std::atomic<u64> frames_elapsed{};
    bool adsp_behind{};
    u32 num_voices_dropped{};
    u64 render_start_tick{};
    u64 render_time_taken_us{};
    u64 signalled_tick{};
    f32 drop_voice_param{1.0f};
};

}  // namespace armigo::audio::renderer
