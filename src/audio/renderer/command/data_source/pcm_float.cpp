// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/data_source/decode.h"
#include "renderer/command/data_source/pcm_float.h"

namespace armigo::audio::renderer {

void PcmFloatDataSourceVersion1Command::process(
    const CommandListProcessor& processor) {
    auto out_buffer = processor.mix_buffer(output_index * processor.sample_count,
                                                    processor.sample_count);

    for (auto& wave_buffer : wave_buffers) {
        wave_buffer.loop_start_offset = wave_buffer.start_offset;
        wave_buffer.loop_end_offset = wave_buffer.end_offset;
        wave_buffer.loop_count = wave_buffer.loop ? -1 : 0;
    }

    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::PcmFloat,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = channel_index,
        .channel_count = channel_count,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = 0,
        .data_size = 0,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    decode_from_wave_buffers(*processor.memory, args);
}

bool PcmFloatDataSourceVersion1Command::verify(
    [[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

void PcmFloatDataSourceVersion2Command::process(
    const CommandListProcessor& processor) {
    auto out_buffer = processor.mix_buffer(output_index * processor.sample_count,
                                                    processor.sample_count);

    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::PcmFloat,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = channel_index,
        .channel_count = channel_count,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = 0,
        .data_size = 0,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    decode_from_wave_buffers(*processor.memory, args);
}

bool PcmFloatDataSourceVersion2Command::verify(
    [[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
