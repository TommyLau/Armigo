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
#include "renderer/behavior/behavior_info.h"

namespace armigo::audio::renderer {

BehaviorInfo::BehaviorInfo() : process_revision{CurrentRevision} {}

u32 BehaviorInfo::get_process_revision_num() const {
    return process_revision;
}

u32 BehaviorInfo::get_process_revision() const {
    return make_magic('R', 'E', 'V',
                             static_cast<char>(static_cast<u8>('0') + process_revision));
}

u32 BehaviorInfo::get_user_revision_num() const {
    return user_revision;
}

u32 BehaviorInfo::get_user_revision() const {
    return make_magic('R', 'E', 'V',
                             static_cast<char>(static_cast<u8>('0') + user_revision));
}

void BehaviorInfo::set_user_lib_revision(const u32 user_revision_) {
    user_revision = get_revision_num(user_revision_);
}

void BehaviorInfo::clear_error() {
    error_count = 0;
}

void BehaviorInfo::append_error(const ErrorInfo& error) {
    AUDIO_LOG_ERROR("Error during RequestUpdate, reporting code {:04X} address {:08X}",
              error.error_code.raw, error.address);
    if (error_count < MaxErrors) {
        errors[error_count++] = error;
    }
}

void BehaviorInfo::copy_error_info(std::span<ErrorInfo> out_errors, u32& out_count) const {
    out_count = (std::min)(error_count, MaxErrors);

    for (size_t i = 0; i < MaxErrors; i++) {
        if (i < out_count) {
            out_errors[i] = errors[i];
        } else {
            out_errors[i] = {};
        }
    }
}

void BehaviorInfo::update_flags(const Flags flags_) {
    flags = flags_;
}

bool BehaviorInfo::is_memory_force_mapping_enabled() const {
    return flags.IsMemoryForceMappingEnabled;
}

bool BehaviorInfo::is_adpcm_loop_context_bug_fixed() const {
    return check_feature_supported(SupportTags::AdpcmLoopContextBugFix, user_revision);
}

bool BehaviorInfo::is_splitter_supported() const {
    return check_feature_supported(SupportTags::Splitter, user_revision);
}

bool BehaviorInfo::is_splitter_bug_fixed() const {
    return check_feature_supported(SupportTags::SplitterBugFix, user_revision);
}

bool BehaviorInfo::is_effect_info_version2_supported() const {
    return check_feature_supported(SupportTags::EffectInfoVer2, user_revision);
}

bool BehaviorInfo::is_variadic_command_buffer_size_supported() const {
    return check_feature_supported(SupportTags::AudioRendererVariadicCommandBufferSize,
                                 user_revision);
}

bool BehaviorInfo::is_wave_buffer_ver2_supported() const {
    return check_feature_supported(SupportTags::WaveBufferVer2, user_revision);
}

bool BehaviorInfo::is_long_size_pre_delay_supported() const {
    return check_feature_supported(SupportTags::LongSizePreDelay, user_revision);
}

bool BehaviorInfo::is_command_processing_time_estimator_version2_supported() const {
    return check_feature_supported(SupportTags::CommandProcessingTimeEstimatorVersion2,
                                 user_revision);
}

bool BehaviorInfo::is_command_processing_time_estimator_version3_supported() const {
    return check_feature_supported(SupportTags::CommandProcessingTimeEstimatorVersion3,
                                 user_revision);
}

bool BehaviorInfo::is_command_processing_time_estimator_version4_supported() const {
    return check_feature_supported(SupportTags::CommandProcessingTimeEstimatorVersion4,
                                 user_revision);
}

bool BehaviorInfo::is_command_processing_time_estimator_version5_supported() const {
    return check_feature_supported(SupportTags::CommandProcessingTimeEstimatorVersion4,
                                 user_revision);
}

bool BehaviorInfo::is_audio_renderer_processing_time_limit70_percent_supported() const {
    return check_feature_supported(SupportTags::AudioRendererProcessingTimeLimit70Percent,
                                 user_revision);
}

bool BehaviorInfo::is_audio_renderer_processing_time_limit75_percent_supported() const {
    return check_feature_supported(SupportTags::AudioRendererProcessingTimeLimit75Percent,
                                 user_revision);
}

bool BehaviorInfo::is_audio_renderer_processing_time_limit80_percent_supported() const {
    return check_feature_supported(SupportTags::AudioRendererProcessingTimeLimit80Percent,
                                 user_revision);
}

bool BehaviorInfo::is_flush_voice_wave_buffers_supported() const {
    return check_feature_supported(SupportTags::FlushVoiceWaveBuffers, user_revision);
}

bool BehaviorInfo::is_elapsed_frame_count_supported() const {
    return check_feature_supported(SupportTags::ElapsedFrameCount, user_revision);
}

bool BehaviorInfo::is_performance_metrics_data_format_version2_supported() const {
    return check_feature_supported(SupportTags::PerformanceMetricsDataFormatVersion2, user_revision);
}

size_t BehaviorInfo::get_performance_metrics_data_format() const {
    if (check_feature_supported(SupportTags::PerformanceMetricsDataFormatVersion2, user_revision)) {
        return 2;
    }
    return 1;
}

bool BehaviorInfo::is_voice_pitch_and_src_skipped_supported() const {
    return check_feature_supported(SupportTags::VoicePitchAndSrcSkipped, user_revision);
}

bool BehaviorInfo::is_voice_played_sample_count_reset_at_loop_point_supported() const {
    return check_feature_supported(SupportTags::VoicePlayedSampleCountResetAtLoopPoint,
                                 user_revision);
}

bool BehaviorInfo::is_biquad_filter_effect_state_clear_bug_fixed() const {
    return check_feature_supported(SupportTags::BiquadFilterEffectStateClearBugFix, user_revision);
}

bool BehaviorInfo::is_volume_mix_parameter_precision_q23_supported() const {
    return check_feature_supported(SupportTags::VolumeMixParameterPrecisionQ23, user_revision);
}

bool BehaviorInfo::use_biquad_filter_float_processing() const {
    return check_feature_supported(SupportTags::BiquadFilterFloatProcessing, user_revision);
}

bool BehaviorInfo::is_mix_in_parameter_dirty_only_update_supported() const {
    return check_feature_supported(SupportTags::MixInParameterDirtyOnlyUpdate, user_revision);
}

bool BehaviorInfo::use_multi_tap_biquad_filter_processing() const {
    return check_feature_supported(SupportTags::MultiTapBiquadFilterProcessing, user_revision);
}

bool BehaviorInfo::is_device_api_version2_supported() const {
    return check_feature_supported(SupportTags::DeviceApiVersion2, user_revision);
}

bool BehaviorInfo::is_delay_channel_mapping_changed() const {
    return check_feature_supported(SupportTags::DelayChannelMappingChange, user_revision);
}

bool BehaviorInfo::is_reverb_channel_mapping_changed() const {
    return check_feature_supported(SupportTags::ReverbChannelMappingChange, user_revision);
}

bool BehaviorInfo::is_i3dl2_reverb_channel_mapping_changed() const {
    return check_feature_supported(SupportTags::I3dl2ReverbChannelMappingChange, user_revision);
}

bool BehaviorInfo::is_splitter_prev_volume_reset_supported() const {
    return check_feature_supported(SupportTags::SplitterPrevVolumeReset, user_revision);
}

bool BehaviorInfo::is_splitter_destination_v2b_supported() const {
    return check_feature_supported(SupportTags::SplitterDestinationV2b, user_revision);
}

bool BehaviorInfo::is_voice_in_parameter_v2_supported() const {
    return check_feature_supported(SupportTags::VoiceInParameterV2, user_revision);
}

bool BehaviorInfo::is_biquad_filter_parameter_for_splitter_enabled() const {
    return check_feature_supported(SupportTags::SplitterBiquadFilterParameter, user_revision);
}

} // namespace armigo::audio::renderer
