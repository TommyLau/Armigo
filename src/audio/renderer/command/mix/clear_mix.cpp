// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <string>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/mix/clear_mix.h"

namespace armigo::audio::renderer {

void ClearMixBufferCommand::process(const CommandListProcessor& processor) {
    memset(processor.mix_buffers.data(), 0, processor.mix_buffers.size_bytes());
}

bool ClearMixBufferCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
