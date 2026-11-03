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
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for applying volume to a mix buffer, with ramping for the volume to smooth
// out the transition.
struct VolumeRampCommand : ICommand {
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
    // Previous mix volume applied to the input
    f32 prev_volume;
    // Current mix volume applied to the input
    f32 volume;
};

} // namespace armigo::audio::renderer
