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

#include <map>
#include <ranges>
#include <tuple>

#include "renderer/common/support.h"
#include "renderer/common/support.h"
#include "renderer/common/types.h"
#include <ranges>

namespace armigo::audio::renderer {
constexpr u32 CurrentRevision = 15;
// Switch 2 libraries tag their revisions 'REv' (lowercase v) and go one further.
constexpr u32 NativeCurrentRevision = 16;

constexpr bool is_native_revision(u32 revision) {
    return (revision & 0xffffff) == make_magic('R', 'E', 'v', '\0');
}

constexpr u32 encode_revision(u32 revision, bool native) {
    return make_magic('R', 'E', native ? 'v' : 'V', static_cast<char>('0' + revision));
}

enum class SupportTags {
    CommandProcessingTimeEstimatorVersion4,
    CommandProcessingTimeEstimatorVersion3,
    CommandProcessingTimeEstimatorVersion2,
    MultiTapBiquadFilterProcessing,
    EffectInfoVer2,
    WaveBufferVer2,
    BiquadFilterFloatProcessing,
    VolumeMixParameterPrecisionQ23,
    MixInParameterDirtyOnlyUpdate,
    BiquadFilterEffectStateClearBugFix,
    VoicePlayedSampleCountResetAtLoopPoint,
    VoicePitchAndSrcSkipped,
    SplitterBugFix,
    FlushVoiceWaveBuffers,
    ElapsedFrameCount,
    AudioRendererVariadicCommandBufferSize,
    PerformanceMetricsDataFormatVersion2,
    AudioRendererProcessingTimeLimit80Percent,
    AudioRendererProcessingTimeLimit75Percent,
    AudioRendererProcessingTimeLimit70Percent,
    AdpcmLoopContextBugFix,
    Splitter,
    LongSizePreDelay,
    AudioUsbDeviceOutput,
    DeviceApiVersion2,
    DelayChannelMappingChange,
    ReverbChannelMappingChange,
    I3dl2ReverbChannelMappingChange,
    SplitterPrevVolumeReset,
    SplitterBiquadFilterParameter,
    SplitterDestinationV2b,
    VoiceInParameterV2,

    // Not a real tag, just here to get the count.
    Size
};

// The revision number of a tag, or ~0 for a tag of neither platform.
constexpr u32 get_revision_num(u32 user_revision) {
    if (user_revision < 0x100) {
        return user_revision;
    }
    if (!is_native_revision(user_revision) &&
        (user_revision & 0xffffff) != make_magic('R', 'E', 'V', '\0')) {
        return ~u32{0};
    }
    const u32 encoded = user_revision >> 24;
    return encoded >= '0' ? encoded - '0' : ~u32{0};
};

constexpr bool check_feature_supported(SupportTags tag, u32 user_revision) {
    constexpr std::array<std::pair<SupportTags, u32>, static_cast<u32>(SupportTags::Size)> features{
        {
            {SupportTags::AudioRendererProcessingTimeLimit70Percent, 1},
            {SupportTags::Splitter, 2},
            {SupportTags::AdpcmLoopContextBugFix, 2},
            {SupportTags::LongSizePreDelay, 3},
            {SupportTags::AudioUsbDeviceOutput, 4},
            {SupportTags::AudioRendererProcessingTimeLimit75Percent, 4},
            {SupportTags::VoicePlayedSampleCountResetAtLoopPoint, 5},
            {SupportTags::VoicePitchAndSrcSkipped, 5},
            {SupportTags::SplitterBugFix, 5},
            {SupportTags::FlushVoiceWaveBuffers, 5},
            {SupportTags::ElapsedFrameCount, 5},
            {SupportTags::AudioRendererProcessingTimeLimit80Percent, 5},
            {SupportTags::AudioRendererVariadicCommandBufferSize, 5},
            {SupportTags::PerformanceMetricsDataFormatVersion2, 5},
            {SupportTags::CommandProcessingTimeEstimatorVersion2, 5},
            {SupportTags::BiquadFilterEffectStateClearBugFix, 6},
            {SupportTags::BiquadFilterFloatProcessing, 7},
            {SupportTags::VolumeMixParameterPrecisionQ23, 7},
            {SupportTags::MixInParameterDirtyOnlyUpdate, 7},
            {SupportTags::WaveBufferVer2, 8},
            {SupportTags::CommandProcessingTimeEstimatorVersion3, 8},
            {SupportTags::EffectInfoVer2, 9},
            {SupportTags::CommandProcessingTimeEstimatorVersion4, 10},
            {SupportTags::MultiTapBiquadFilterProcessing, 10},
            {SupportTags::DelayChannelMappingChange, 11},
            {SupportTags::ReverbChannelMappingChange, 11},
            {SupportTags::I3dl2ReverbChannelMappingChange, 11},
            {SupportTags::SplitterBiquadFilterParameter, 12},
            {SupportTags::SplitterPrevVolumeReset, 13},
            {SupportTags::SplitterDestinationV2b, 15},
            {SupportTags::VoiceInParameterV2, 15},
        }};

    const auto& feature =
        std::ranges::find_if(features, [tag](const auto& entry) { return entry.first == tag; });
    if (feature == features.cend()) {
        AUDIO_LOG_ERROR("Invalid SupportTag {}!", static_cast<u32>(tag));
        return false;
    }
    user_revision = get_revision_num(user_revision);
    return user_revision != ~u32{0} && (*feature).second <= user_revision;
}

constexpr bool check_valid_revision(u32 user_revision) {
    return get_revision_num(user_revision) <=
           (is_native_revision(user_revision) ? NativeCurrentRevision : CurrentRevision);
};

} // namespace armigo::audio::renderer
