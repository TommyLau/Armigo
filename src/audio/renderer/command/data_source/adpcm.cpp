// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <span>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/data_source/adpcm.h"
#include "renderer/command/data_source/decode.h"

namespace armigo::audio::renderer {

void AdpcmDataSourceVersion1Command::process(const CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffer(output_index * processor.sample_count,
                                                  processor.sample_count)};

    for (auto& wave_buffer : wave_buffers) {
        wave_buffer.loop_start_offset = wave_buffer.start_offset;
        wave_buffer.loop_end_offset = wave_buffer.end_offset;
        wave_buffer.loop_count = wave_buffer.loop ? -1 : 0;
    }

    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::Adpcm,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = 0,
        .channel_count = 1,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = data_address,
        .data_size = data_size,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    decode_from_wave_buffers(*processor.memory, args);
}

bool AdpcmDataSourceVersion1Command::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

void AdpcmDataSourceVersion2Command::process(const CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffer(output_index * processor.sample_count,
                                                  processor.sample_count)};

    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::Adpcm,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = 0,
        .channel_count = 1,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = data_address,
        .data_size = data_size,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    decode_from_wave_buffers(*processor.memory, args);
}

bool AdpcmDataSourceVersion2Command::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
