// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/mix/depop_prepare.h"
#include "renderer/voice/voice_state.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {

void DepopPrepareCommand::process([[maybe_unused]] const CommandListProcessor& processor) {
    auto samples{reinterpret_cast<s32*>(previous_samples)};
    auto buffer{reinterpret_cast<s32*>(depop_buffer)};

    for (u32 i = 0; i < buffer_count; i++) {
        if (samples[i]) {
            buffer[inputs[i]] += samples[i];
            samples[i] = 0;
        }
    }
}

bool DepopPrepareCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
