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
#include "renderer/command/mix/volume.h"
#include "renderer/common/fixed_point.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {
// Apply volume to the input mix buffer, saving to the output buffer.
template <size_t Q>
static void apply_uniform_gain(std::span<s32> output, std::span<const s32> input, const f32 volume,
                             const u32 sample_count) {
    if (volume == 1.0f) {
        std::memcpy(output.data(), input.data(), input.size_bytes());
    } else {
        const FixedPoint<64 - Q, Q> gain{volume};
        for (u32 i = 0; i < sample_count; i++) {
            output[i] = (input[i] * gain).to_int();
        }
    }
}

void VolumeCommand::process(const CommandListProcessor& processor) {
    // If input and output buffers are the same, and the volume is 1.0f, this won't do
    // anything, so just skip.
    if (input_index == output_index && volume == 1.0f) {
        return;
    }

    auto output{processor.mix_buffer(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffer(input_index * processor.sample_count,
                                             processor.sample_count)};

    switch (precision) {
    case 15:
        apply_uniform_gain<15>(output, input, volume, processor.sample_count);
        break;

    case 23:
        apply_uniform_gain<23>(output, input, volume, processor.sample_count);
        break;

    default:
        AUDIO_LOG_ERROR("Invalid precision {}", precision);
        break;
    }
}

bool VolumeCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
