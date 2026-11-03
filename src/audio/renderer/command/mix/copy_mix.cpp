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
#include "renderer/command/mix/copy_mix.h"

namespace armigo::audio::renderer {

void CopyMixBufferCommand::process(const CommandListProcessor& processor) {
    auto output{processor.mix_buffer(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffer(input_index * processor.sample_count,
                                             processor.sample_count)};
    std::memcpy(output.data(), input.data(), processor.sample_count * sizeof(s32));
}

bool CopyMixBufferCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
