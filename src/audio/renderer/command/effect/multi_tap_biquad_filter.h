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
#include <string>

#include "renderer/command/icommand.h"
#include "renderer/voice/voice_info.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for applying multiple biquads at once.
struct MultiTapBiquadFilterCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Input mix buffer index
    s16 input;
    // Output mix buffer index
    s16 output;
    // Biquad parameters
    std::array<VoiceInfo::BiquadFilterParameter, MaxBiquadFilters> biquads;
    // Biquad parameters (REV15+ native float)
    std::array<VoiceInfo::BiquadFilterParameter2, MaxBiquadFilters> biquads_float;
    // Biquad states, updated each call
    std::array<CpuAddr, MaxBiquadFilters> states;
    // If each biquad needs initialisation
    std::array<bool, MaxBiquadFilters> needs_init;
    // Number of active biquads
    u8 filter_tap_count;
    // If true, use native float coefficients (REV15+)
    bool use_float_coefficients;
};

} // namespace armigo::audio::renderer
