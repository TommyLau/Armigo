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

// AudioRenderer command for upsampling a mix buffer to 48Khz.
// Input must be 8Khz, 16Khz or 32Khz, and output will be 48Khz.
struct UpsampleCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Pointer to the output samples buffer.
    CpuAddr samples_buffer;
    // Pointer to input mix buffer indexes.
    CpuAddr inputs;
    // Number of input mix buffers.
    u32 buffer_count;
    // Unknown, unused.
    u32 unk_20;
    // Source data sample count.
    u32 source_sample_count;
    // Source data sample rate.
    u32 source_sample_rate;
    // Pointer to the upsampler info for this command.
    CpuAddr upsampler_info;
};

} // namespace armigo::audio::renderer
