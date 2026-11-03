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
#include "renderer/performance/performance_manager.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
struct AudioRendererSystemContext;

class CommandBuffer;
struct CommandListHeader;
class VoiceContext;
class MixContext;
class EffectContext;
class SplitterContext;
class SinkContext;
class BehaviorInfo;
class VoiceInfo;
struct VoiceState;
class MixInfo;
class SinkInfoBase;

// Generates all commands to build up a command list, which are sent to the AudioRender for
// processing.
class CommandGenerator {
public:
    explicit CommandGenerator(CommandBuffer& command_buffer,
                              const CommandListHeader& command_list_header,
                              const AudioRendererSystemContext& render_context,
                              VoiceContext& voice_context, MixContext& mix_context,
                              EffectContext& effect_context, SinkContext& sink_context,
                              SplitterContext& splitter_context,
                              PerformanceManager* performance_manager);

    // Calculate the buffer size needed for commands.
    static u64 calculate_command_buffer_size(const BehaviorInfo& behavior,
                                          const AudioRendererParameterInternal& params) {
        u64 size{0};

        // Effects
        size += params.effects * sizeof(EffectInfoBase);

        // Voices
        u64 voice_size{0};
        if (behavior.is_wave_buffer_ver2_supported()) {
            voice_size = (std::max)((std::max)(sizeof(AdpcmDataSourceVersion2Command),
                                           sizeof(PcmInt16DataSourceVersion2Command)),
                                  sizeof(PcmFloatDataSourceVersion2Command));
        } else {
            voice_size = (std::max)((std::max)(sizeof(AdpcmDataSourceVersion1Command),
                                           sizeof(PcmInt16DataSourceVersion1Command)),
                                  sizeof(PcmFloatDataSourceVersion1Command));
        }
        voice_size += sizeof(BiquadFilterCommand) * MaxBiquadFilters;
        voice_size += sizeof(VolumeRampCommand);
        voice_size += sizeof(MixRampGroupedCommand);

        size += params.voices * (params.splitter_infos * sizeof(DepopPrepareCommand) + voice_size);

        // Sub mixes
        size += sizeof(DepopForMixBuffersCommand) +
                (sizeof(MixCommand) * MaxMixBuffers) * MaxMixBuffers;

        // Final mix
        size += sizeof(DepopForMixBuffersCommand) + sizeof(VolumeCommand) * MaxMixBuffers;

        // Splitters
        size += params.splitter_destinations * sizeof(MixRampCommand) * MaxMixBuffers;

        // Sinks
        size +=
            params.sinks * (std::max)(sizeof(DeviceSinkCommand), sizeof(CircularBufferSinkCommand));

        // Performance
        size += (params.effects + params.voices + params.sinks + params.sub_mixes + 1 +
                 PerformanceManager::MaxDetailEntries) *
                sizeof(PerformanceCommand);
        return size;
    }

    // Get the current command buffer used to generate commands.
    CommandBuffer& get_command_buffer() {
        return command_buffer;
    }

    // Get the current performance manager,
    PerformanceManager* get_performance_manager() {
        return performance_manager;
    }

    // Generate a data source command.
    // These are the basis for all audio output.
    void generate_data_source_command(VoiceInfo& voice_info, const VoiceState& voice_state,
                                   s8 channel);

    // Generate voice mixing commands.
    // These are used to mix buffers together, to mix one input to many outputs,
    // and also used as copy commands to move data around and prevent it being accidentally
    // overwritten, e.g by another data source command into the same channel.
    void generate_voice_mix_command(std::span<const f32> mix_volumes,
                                 std::span<const f32> prev_mix_volumes,
                                 const VoiceState& voice_state, s16 output_index, s16 buffer_count,
                                 s16 input_index, s32 node_id);

    // Generate a biquad filter command for a voice.
    void generate_biquad_filter_command_for_voice(VoiceInfo& voice_info, const VoiceState& voice_state,
                                             s16 buffer_count, s8 channel, s32 node_id);

    // Generate commands for a voice.
    // Includes a data source, biquad filter, volume and mixing.
    void generate_voice_command(VoiceInfo& voice_info);

    // Generate commands for all voices.
    void generate_voice_commands();

    // Generate a mixing command.
    void generate_buffer_mixer_command(s16 buffer_offset, EffectInfoBase& effect_info_base,
                                    s32 node_id);

    // Generate a delay effect command.
    void generate_delay_command(s16 buffer_offset, EffectInfoBase& effect_info_base, s32 node_id);

    // Generate a reverb effect command.
    void generate_reverb_command(s16 buffer_offset, EffectInfoBase& effect_info_base, s32 node_id,
                               bool long_size_pre_delay_supported);

    // Generate an I3DL2 reverb effect command.
    void generate_i3dl2_reverb_effect_command(s16 buffer_offset, EffectInfoBase& effect_info,
                                          s32 node_id);

    // Generate an aux effect command.
    void generate_aux_command(s16 buffer_offset, EffectInfoBase& effect_info, s32 node_id);

    // Generate a biquad filter effect command.
    void generate_biquad_filter_effect_command(s16 buffer_offset, EffectInfoBase& effect_info,
                                           s32 node_id);

    // Generate a light limiter effect command.
    void generate_light_limiter_effect_command(s16 buffer_offset, EffectInfoBase& effect_info,
                                           s32 node_id, u32 effect_index);

    // Generate a capture effect command.
    // Writes a mix buffer back to game memory.
    void generate_capture_command(s16 buffer_offset, EffectInfoBase& effect_info, s32 node_id);

    // Generate a compressor effect command.
    void generate_compressor_command(s16 buffer_offset, EffectInfoBase& effect_info, s32 node_id);

    // Generate all effect commands for a mix.
    void generate_effect_command(MixInfo& mix_info);

    // Generate all mix commands.
    void generate_mix_commands(MixInfo& mix_info);

    // Generate a submix command.
    // Generates all effects and all mixing commands.
    void generate_sub_mix_command(MixInfo& mix_info);

    // Generate all submix command.
    void generate_sub_mix_commands();

    // Generate the final mix.
    void generate_final_mix_command();

    // Generate the final mix commands.
    void generate_final_mix_commands();

    // Generate all sink commands.
    void generate_sink_commands();

    // Generate a sink command.
    // Sends samples out to the backend, or a game-supplied circular buffer.
    void generate_sink_command(s16 buffer_offset, SinkInfoBase& sink_info);

    // Generate a device sink command.
    // Sends samples out to the backend.
    void generate_device_sink_command(s16 buffer_offset, SinkInfoBase& sink_info);

    // Generate a performance command.
    // Used to report performance metrics of the AudioRenderer back to the game.
    void generate_performance_command(s32 node_id, PerformanceState state,
                                    const PerformanceEntryAddresses& entry_addresses);

private:
    // Commands will be written by this buffer
    CommandBuffer& command_buffer;
    // Header information for the commands generated
    const CommandListHeader& command_header;
    // Various things to control generation
    const AudioRendererSystemContext& render_context;
    // Used for generating voices
    VoiceContext& voice_context;
    // Used for generating mixes
    MixContext& mix_context;
    // Used for generating effects
    EffectContext& effect_context;
    // Used for generating sinks
    SinkContext& sink_context;
    // Used for generating submixes
    SplitterContext& splitter_context;
    // Used for generating performance
    PerformanceManager* performance_manager;
};

} // namespace armigo::audio::renderer
