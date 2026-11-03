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
#include "renderer/common/wave_buffer.h"
#include "renderer/voice/voice_state.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

class GuestMemory;

struct DecodeFromWaveBuffersArgs {
    SampleFormat sample_format;
    std::span<s32> output;
    VoiceState* voice_state;
    std::span<WaveBufferVersion2> wave_buffers;
    s8 channel;
    s8 channel_count;
    SrcQuality src_quality;
    f32 pitch;
    u32 source_sample_rate;
    u32 target_sample_rate;
    u32 sample_count;
    CpuAddr data_address;
    u64 data_size;
    bool IsVoicePlayedSampleCountResetAtLoopPointSupported;
    bool IsVoicePitchAndSrcSkippedSupported;
};

struct DecodeArg {
    CpuAddr buffer;
    u64 buffer_size;
    u32 start_offset;
    u32 end_offset;
    s8 channel_count;
    std::array<s16, 16> coefficients;
    VoiceState::AdpcmContext* adpcm_context;
    s8 target_channel;
    u32 offset;
    u32 samples_to_read;
};

// Decode wavebuffers according to the given args.
void decode_from_wave_buffers(GuestMemory& memory, const DecodeFromWaveBuffersArgs& args);

} // namespace armigo::audio::renderer
