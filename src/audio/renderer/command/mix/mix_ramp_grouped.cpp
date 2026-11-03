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
#include "renderer/command/mix/mix_ramp.h"
#include "renderer/command/mix/mix_ramp_grouped.h"

namespace armigo::audio::renderer {

void MixRampGroupedCommand::process(const CommandListProcessor& processor) {
    std::span<s32> prev_samples = {reinterpret_cast<s32*>(previous_samples), MaxMixBuffers};

    for (u32 i = 0; i < buffer_count; i++) {
        auto last_sample{0};
        if (prev_volumes[i] != 0.0f || volumes[i] != 0.0f) {
            const auto output{processor.mix_buffer(outputs[i] * processor.sample_count,
                                                            processor.sample_count)};
            const auto input{processor.mix_buffer(inputs[i] * processor.sample_count,
                                                           processor.sample_count)};
            const auto ramp{(volumes[i] - prev_volumes[i]) /
                            static_cast<f32>(processor.sample_count)};

            if (prev_volumes[i] == 0.0f && ramp == 0.0f) {
                prev_samples[i] = 0;
                continue;
            }

            switch (precision) {
            case 15:
                last_sample =
                    apply_mix_ramp<15>(output, input, prev_volumes[i], ramp, processor.sample_count);
                break;
            case 23:
                last_sample =
                    apply_mix_ramp<23>(output, input, prev_volumes[i], ramp, processor.sample_count);
                break;
            default:
                AUDIO_LOG_ERROR("Invalid precision {}", precision);
                break;
            }
        }

        prev_samples[i] = last_sample;
    }
}

bool MixRampGroupedCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
