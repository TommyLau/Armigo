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

// AudioRenderer command for preparing depop.
// Adds the previously output last samples to the depop buffer.
struct DepopPrepareCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Depop buffer offset for each mix buffer
    std::array<s16, MaxMixBuffers> inputs;
    // Pointer to the previous mix buffer samples
    CpuAddr previous_samples;
    // Number of mix buffers to use
    u32 buffer_count;
    // Pointer to the current depop values
    CpuAddr depop_buffer;
};

} // namespace armigo::audio::renderer
