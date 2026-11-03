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

#include <span>
#include <string>

#include "renderer/command/icommand.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for mixing an input mix buffer to an output mix buffer, with a volume
// applied to the input, and volume ramping to smooth out the transition.
struct MixRampCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Fixed point precision
    u8 precision;
    // Input mix buffer index
    s16 input_index;
    // Output mix buffer index
    s16 output_index;
    // Previous mix volume
    f32 prev_volume;
    // Current mix volume
    f32 volume;
    // Pointer to the previous sample buffer, used for depopping
    CpuAddr previous_sample;
};

// Mix input mix buffer into output mix buffer, with volume applied to the input.
template <size_t Q>
s32 apply_mix_ramp(std::span<s32> output, std::span<const s32> input, f32 volume_, f32 ramp_,
                 u32 sample_count);

} // namespace armigo::audio::renderer
