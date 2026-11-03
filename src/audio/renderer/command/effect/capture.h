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

// AudioRenderer command for capturing a mix buffer. That is, writing it back to a given game memory
// address.
struct CaptureCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Input mix buffer index
    s16 input;
    // Output mix buffer index
    s16 output;
    // Meta info for writing
    CpuAddr send_buffer_info;
    // Game memory write buffer
    CpuAddr send_buffer;
    // Max samples to read/write
    u32 count_max;
    // Current read/write offset
    u32 write_offset;
    // Number of samples to update per call
    u32 update_count;
    // is this effect enabled?
    bool effect_enabled;
};

} // namespace armigo::audio::renderer
