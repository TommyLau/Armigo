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
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for downmixing 6 channels to 2.
// Channel layout (SMPTE):
//     0 - front left
//     1 - front right
//     2 - center
//     3 - lfe
//     4 - back left
//     5 - back right
struct DownMix6chTo2chCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Input mix buffer offsets for each channel
    std::array<s16, MaxChannels> inputs;
    // Output mix buffer offsets for each channel
    std::array<s16, MaxChannels> outputs;
    // Coefficients used for downmixing
    std::array<FixedPoint<48, 16>, 4> down_mix_coeff;
};

} // namespace armigo::audio::renderer
