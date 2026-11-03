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
#include "renderer/command/mix/volume_ramp.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
// Apply volume with ramping to the input mix buffer, saving to the output buffer.
template <size_t Q>
static void apply_linear_envelope_gain(std::span<s32> output, std::span<const s32> input,
                                    const f32 volume, const f32 ramp_, const u32 sample_count) {
    if (volume == 0.0f && ramp_ == 0.0f) {
        std::memset(output.data(), 0, output.size_bytes());
    } else if (volume == 1.0f && ramp_ == 0.0f) {
        std::memcpy(output.data(), input.data(), output.size_bytes());
    } else if (ramp_ == 0.0f) {
        const FixedPoint<64 - Q, Q> gain{volume};
        for (u32 i = 0; i < sample_count; i++) {
            output[i] = (input[i] * gain).to_int();
        }
    } else {
        FixedPoint<64 - Q, Q> gain{volume};
        const FixedPoint<64 - Q, Q> ramp{ramp_};
        for (u32 i = 0; i < sample_count; i++) {
            output[i] = (input[i] * gain).to_int();
            gain += ramp;
        }
    }
}

void VolumeRampCommand::process(const CommandListProcessor& processor) {
    auto output{processor.mix_buffer(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffer(input_index * processor.sample_count,
                                             processor.sample_count)};
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};

    // If input and output buffers are the same, and the volume is 1.0f, and there's no ramping,
    // this won't do anything, so just skip.
    if (input_index == output_index && prev_volume == 1.0f && ramp == 0.0f) {
        return;
    }

    switch (precision) {
    case 15:
        apply_linear_envelope_gain<15>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    case 23:
        apply_linear_envelope_gain<23>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    default:
        AUDIO_LOG_ERROR("Invalid precision {}", precision);
        break;
    }
}

bool VolumeRampCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
