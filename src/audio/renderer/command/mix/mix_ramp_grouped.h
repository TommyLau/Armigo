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
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for mixing multiple input mix buffers to multiple output mix buffers, with
// a volume applied to the input, and volume ramping to smooth out the transition.
struct MixRampGroupedCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Fixed point precision
    u8 precision;
    // Number of mix buffers to mix
    u32 buffer_count;
    // Input mix buffer indexes for each mix buffer
    std::array<s16, MaxMixBuffers> inputs;
    // Output mix buffer indexes for each mix buffer
    std::array<s16, MaxMixBuffers> outputs;
    // Previous mix volumes for each mix buffer
    std::array<f32, MaxMixBuffers> prev_volumes;
    // Current mix volumes for each mix buffer
    std::array<f32, MaxMixBuffers> volumes;
    // Pointer to the previous sample buffer, used for depop
    CpuAddr previous_samples;
};

} // namespace armigo::audio::renderer
