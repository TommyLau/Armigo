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
#include "renderer/effect/compressor.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for limiting volume between a high and low threshold.
// Version 1.
struct CompressorCommand : ICommand {
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
    CompressorInfo::ParameterVersion2 parameter;
    // State, updated each call
    CpuAddr state;
    // Game-supplied workbuffer (Unused)
    CpuAddr workbuffer;
    // Is this effect enabled?
    bool effect_enabled;
};

} // namespace armigo::audio::renderer
