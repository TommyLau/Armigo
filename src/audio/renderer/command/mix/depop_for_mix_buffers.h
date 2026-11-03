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

// AudioRenderer command for depopping a mix buffer.
// Adds a cumulation of previous samples to the current mix buffer with a decay.
struct DepopForMixBuffersCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // Starting input mix buffer index
    u32 input;
    // Number of mix buffers to depop
    u32 count;
    // Amount to decay the depop sample for each new sample
    FixedPoint<49, 15> decay;
    // Address of the depop buffer, holding the last sample for every mix buffer
    CpuAddr depop_buffer;
};

} // namespace armigo::audio::renderer
