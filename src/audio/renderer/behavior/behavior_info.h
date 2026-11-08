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

#include <array>
#include <span>

#include "renderer/common/common.h"
#include "renderer/common/types.h"
#include "renderer/common/result.h"

namespace armigo::audio::renderer {
// Holds host and user revisions, checks whether render features can be enabled, and reports errors.
class BehaviorInfo {
    static constexpr u32 MaxErrors = 10;

public:
    struct ErrorInfo {
        /* 0x00 */ Result error_code{0};
        /* 0x04 */ u32 unk_04;
        /* 0x08 */ CpuAddr address;
    };
    static_assert(sizeof(ErrorInfo) == 0x10, "BehaviorInfo::ErrorInfo has the wrong size!");

    struct Flags {
        u64 IsMemoryForceMappingEnabled : 1;
    };

    struct InParameter {
        /* 0x00 */ u32 revision;
        /* 0x08 */ Flags flags;
    };
    static_assert(sizeof(InParameter) == 0x10, "BehaviorInfo::InParameter has the wrong size!");

    struct OutStatus {
        /* 0x00 */ std::array<ErrorInfo, MaxErrors> errors;
        /* 0xA0 */ u32 error_count;
        /* 0xA4 */ char unkA4[0xC];
    };
    static_assert(sizeof(OutStatus) == 0xB0, "BehaviorInfo::OutStatus has the wrong size!");

    BehaviorInfo();

    // Get the host revision as a number.
    u32 get_process_revision_num() const;

    // Get the host revision in chars, e.g REV8.
    // Rev 10 and higher use the ascii characters above 9.
    // E.g:
    //     Rev 10 = REV:
    //     Rev 11 = REV;
    u32 get_process_revision() const;

    // Get the user revision as a number.
    u32 get_user_revision_num() const;

    // Get the user revision in chars, e.g REV8.
    // Rev 10 and higher use the ascii characters above 9. REV: REV; etc.
    u32 get_user_revision() const;

    // Set the user revision.
    void set_user_lib_revision(u32 user_revision);

    // Whether the user library tags its revisions as a Switch 2 library ('REv').
    bool is_native_platform() const {
        return native_platform;
    }

    // Clear the current error count.
    void clear_error();

    // Append an error to the error list.
    void append_error(const ErrorInfo& error);

    // Copy errors to the given output container.
    void copy_error_info(std::span<ErrorInfo> out_errors, u32& out_count) const;

    // Update the behaviour flags.
    void update_flags(Flags flags);

    // Check if memory pools can be forcibly mapped.
    bool is_memory_force_mapping_enabled() const;

    // Check if the ADPCM context bug is fixed.
    // The ADPCM context was not being sent to the AudioRenderer, leading to incorrect scaling being
    // used.
    bool is_adpcm_loop_context_bug_fixed() const;

    // Check if the splitter is supported.
    bool is_splitter_supported() const;

    // Check if the splitter bug is fixed.
    // Update is given the wrong number of splitter destinations, leading to invalid data
    // being processed.
    bool is_splitter_bug_fixed() const;

    // Check if effects version 2 are supported.
    // This gives support for returning effect states from the AudioRenderer, currently only used
    // for Limiter statistics.
    bool is_effect_info_version2_supported() const;

    // Check if a variadic command buffer is supported.
    // As of Rev 5 with the added optional performance metric logging, the command
    // buffer can be a variable size, so take that into account for calculating its size.
    bool is_variadic_command_buffer_size_supported() const;

    // Check if wave buffers version 2 are supported.
    // See WaveBufferVersion1 and WaveBufferVersion2.
    bool is_wave_buffer_ver2_supported() const;

    // Check if long size pre delay is supported.
    // This allows a longer initial delay time for the Reverb command.
    bool is_long_size_pre_delay_supported() const;

    // Check if the command time estimator version 2 is supported.
    bool is_command_processing_time_estimator_version2_supported() const;

    // Check if the command time estimator version 3 is supported.
    bool is_command_processing_time_estimator_version3_supported() const;

    // Check if the command time estimator version 4 is supported.
    bool is_command_processing_time_estimator_version4_supported() const;

    // Check if the command time estimator version 5 is supported.
    bool is_command_processing_time_estimator_version5_supported() const;

    // Check if the AudioRenderer can use up to 70% of the allocated processing timeslice.
    bool is_audio_renderer_processing_time_limit70_percent_supported() const;

    // Check if the AudioRenderer can use up to 75% of the allocated processing timeslice.
    bool is_audio_renderer_processing_time_limit75_percent_supported() const;

    // Check if the AudioRenderer can use up to 80% of the allocated processing timeslice.
    bool is_audio_renderer_processing_time_limit80_percent_supported() const;

    // Check if voice flushing is supported
    // This allowws low-priority voices to be dropped if the AudioRenderer is running behind.
    bool is_flush_voice_wave_buffers_supported() const;

    // Check if counting the number of elapsed frames is supported.
    // This adds extra output to RequestUpdate, returning the number of times the AudioRenderer
    // processed a command list.
    bool is_elapsed_frame_count_supported() const;

    // Check if performance metrics version 2 are supported.
    // This adds extra output to RequestUpdate, returning the number of times the AudioRenderer
    // (Unused?).
    bool is_performance_metrics_data_format_version2_supported() const;

    // Get the supported performance metrics version.
    // Version 2 logs some extra fields in output, such as number of voices dropped,
    // processing start time, if the AudioRenderer exceeded its time, etc.
    size_t get_performance_metrics_data_format() const;

    // Check if skipping voice pitch and sample rate conversion is supported.
    // This speeds up the data source commands by skipping resampling if unwanted.
    // See DecodeFromWaveBuffers
    bool is_voice_pitch_and_src_skipped_supported() const;

    // Check if resetting played sample count at loop points is supported.
    // This resets the number of samples played in a voice state when a loop point is reached.
    // See DecodeFromWaveBuffers
    bool is_voice_played_sample_count_reset_at_loop_point_supported() const;

    // Check if the clear state bug for biquad filters is fixed.
    // The biquad state was not marked as needing re-initialisation when the effect was updated, it
    // was only initialized once with a new effect.
    bool is_biquad_filter_effect_state_clear_bug_fixed() const;

    // Check if Q23 precision is supported for fixed point.
    bool is_volume_mix_parameter_precision_q23_supported() const;

    // Check if float processing for biuad filters is supported.
    bool use_biquad_filter_float_processing() const;

    // Check if dirty-only mix updates are supported.
    // This saves a lot of buffer size as mixes can be large and not change much.
    bool is_mix_in_parameter_dirty_only_update_supported() const;

    // Check if multi-tap biquad filters are supported.
    bool use_multi_tap_biquad_filter_processing() const;

    // Check if device api version 2 is supported.
    // In the SDK but not in any sysmodule? Not sure, left here for completeness anyway.
    bool is_device_api_version2_supported() const;

    // Check if new channel mappings are used for Delay commands.
    // Older commands used:
    //   front left/front right/back left/back right/center/lfe
    // Whereas everywhere else in the code uses:
    //   front left/front right/center/lfe/back left/back right
    // This corrects that and makes everything standardised.
    bool is_delay_channel_mapping_changed() const;

    // Check if new channel mappings are used for Reverb commands.
    // Older commands used:
    //   front left/front right/back left/back right/center/lfe
    // Whereas everywhere else in the code uses:
    //   front left/front right/center/lfe/back left/back right
    // This corrects that and makes everything standardised.
    bool is_reverb_channel_mapping_changed() const;

    // Check if new channel mappings are used for I3dl2Reverb commands.
    // Older commands used:
    //   front left/front right/back left/back right/center/lfe
    // Whereas everywhere else in the code uses:
    //   front left/front right/center/lfe/back left/back right
    // This corrects that and makes everything standardised.
    bool is_i3dl2_reverb_channel_mapping_changed() const;

    // Check if explicit previous mix volume reset is supported for splitters.
    // This allows splitters to explicitly reset their previous mix volumes instead of
    // doing so implicitly on first use.
    bool is_splitter_prev_volume_reset_supported() const;

    // Check if splitter destination v2b parameter format is supported (revision 15+).
    // This uses the extended parameter format with biquad filter fields.
    bool is_splitter_destination_v2b_supported() const;

    // Check if voice input parameter v2 format is supported (revision 15+).
    // This uses the extended parameter format with float biquad filters.
    bool is_voice_in_parameter_v2_supported() const;

    // Check if splitter destinations can carry biquad filter parameters (revision 12+).
    bool is_biquad_filter_parameter_for_splitter_enabled() const;

    // Host version
    u32 process_revision;
    // User version
    u32 user_revision{};
    // The user library tags its revisions as a Switch 2 library ('REv').
    bool native_platform{};
    // Behaviour flags
    Flags flags{};
    // Errors generated and reported during Update
    std::array<ErrorInfo, MaxErrors> errors{};
    // Error count
    u32 error_count{};
};

} // namespace armigo::audio::renderer
