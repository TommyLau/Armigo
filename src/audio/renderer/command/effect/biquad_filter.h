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

#include "renderer/command/icommand.h"
#include "renderer/voice/voice_info.h"
#include "renderer/voice/voice_state.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for applying a biquad filter to the input mix buffer, saving the results to
// the output mix buffer.
struct BiquadFilterCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Input mix buffer index
    s16 input;
    // Output mix buffer index
    s16 output;
    // Input parameters for biquad
    VoiceInfo::BiquadFilterParameter biquad;
    // Input parameters for biquad (REV15+ native float)
    VoiceInfo::BiquadFilterParameter2 biquad_float;
    // Biquad state, updated each call
    CpuAddr state;
    // If true, reset the state
    bool needs_init;
    // If true, use float processing rather than int
    bool use_float_processing;
    // If true, use native float coefficients (REV15+)
    bool use_float_coefficients;
};

// Biquad filter float implementation.
void apply_biquad_filter_float(std::span<s32> output, std::span<const s32> input,
                            std::array<s16, 3>& b, std::array<s16, 2>& a,
                            VoiceState::BiquadFilterState& state, const u32 sample_count);

// Biquad filter float implementation with native float coefficients (SDK REV15+).
void apply_biquad_filter_float2(std::span<s32> output, std::span<const s32> input,
                             std::array<f32, 3>& b, std::array<f32, 2>& a,
                             VoiceState::BiquadFilterState& state, const u32 sample_count);

} // namespace armigo::audio::renderer
