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
#include "renderer/effect/reverb.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for a Reverb effect. Apply a reverb to inputs mix buffer, outputs receives
// the results.
struct ReverbCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Input mix buffer offsets for each channel
    std::array<s16, MaxChannels> inputs;
    // Output mix buffer offsets for each channel
    std::array<s16, MaxChannels> outputs;
    // Input parameters
    ReverbInfo::ParameterVersion2 parameter;
    // State, updated each call
    CpuAddr state;
    // Game-supplied workbuffer (Unused)
    CpuAddr workbuffer;
    // Is this effect enabled?
    bool effect_enabled;
    // Is a longer pre-delay time supported?
    bool long_size_pre_delay_supported;
};

} // namespace armigo::audio::renderer
