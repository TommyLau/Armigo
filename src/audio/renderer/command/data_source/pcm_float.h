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

#include <string>

#include "renderer/common/wave_buffer.h"
#include "renderer/command/icommand.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command to decode PCM float-encoded version 1 wavebuffers
// into the output_index mix buffer.
struct PcmFloatDataSourceVersion1Command : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Quality used for sample rate conversion
    SrcQuality src_quality;
    // Mix buffer index for decoded samples
    s16 output_index;
    // Flags to control decoding (see VoiceInfo::Flags)
    u16 flags;
    // Wavebuffer sample rate
    u32 sample_rate;
    // Pitch used for sample rate conversion
    f32 pitch;
    // Target channel to read within the wavebuffer
    s8 channel_index;
    // Number of channels within the wavebuffer
    s8 channel_count;
    // Wavebuffers containing the wavebuffer address, context address, looping information etc
    std::array<WaveBufferVersion2, MaxWaveBuffers> wave_buffers;
    // Voice state, updated each call and written back to game
    CpuAddr voice_state;
};

// AudioRenderer command to decode PCM float-encoded version 2 wavebuffers
// into the output_index mix buffer.
struct PcmFloatDataSourceVersion2Command : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Quality used for sample rate conversion
    SrcQuality src_quality;
    // Mix buffer index for decoded samples
    s16 output_index;
    // Flags to control decoding (see VoiceInfo::Flags)
    u16 flags;
    // Wavebuffer sample rate
    u32 sample_rate;
    // Pitch used for sample rate conversion
    f32 pitch;
    // Target channel to read within the wavebuffer
    s8 channel_index;
    // Number of channels within the wavebuffer
    s8 channel_count;
    // Wavebuffers containing the wavebuffer address, context address, looping information etc
    std::array<WaveBufferVersion2, MaxWaveBuffers> wave_buffers;
    // Voice state, updated each call and written back to game
    CpuAddr voice_state;
};

} // namespace armigo::audio::renderer
