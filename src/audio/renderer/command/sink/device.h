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
#include <string>

#include "renderer/command/icommand.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for sinking samples to an output device.
struct DeviceSinkCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Device name
    char name[0x100];
    // System session id (unused)
    s32 session_id;
    // Sample buffer to sink
    std::span<s32> sample_buffer;
    // Number of input channels
    u32 input_count;
    // Mix buffer indexes for each channel
    std::array<s16, MaxChannels> inputs;
};

} // namespace armigo::audio::renderer
