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
#include "renderer/common/fixed_point.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

template <size_t Q>
s32 apply_mix_ramp(std::span<s32> output, std::span<const s32> input, const f32 volume_,
                 const f32 ramp_, const u32 sample_count) {
    FixedPoint<64 - Q, Q> volume{volume_};
    FixedPoint<64 - Q, Q> sample{0};

    if (ramp_ == 0.0f) {
        for (u32 i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
        }
    } else {
        FixedPoint<64 - Q, Q> ramp{ramp_};
        for (u32 i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
            volume += ramp;
        }
    }
    return sample.to_int();
}

template s32 apply_mix_ramp<15>(std::span<s32>, std::span<const s32>, f32, f32, u32);
template s32 apply_mix_ramp<23>(std::span<s32>, std::span<const s32>, f32, f32, u32);

void MixRampCommand::process(const CommandListProcessor& processor) {
    auto output{processor.mix_buffer(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffer(input_index * processor.sample_count,
                                             processor.sample_count)};
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    auto prev_sample_ptr{reinterpret_cast<s32*>(previous_sample)};

    // If previous volume and ramp are both 0, nothing will be added to the output, so just skip.
    if (prev_volume == 0.0f && ramp == 0.0f) {
        *prev_sample_ptr = 0;
        return;
    }

    switch (precision) {
    case 15:
        *prev_sample_ptr =
            apply_mix_ramp<15>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    case 23:
        *prev_sample_ptr =
            apply_mix_ramp<23>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    default:
        AUDIO_LOG_ERROR("Invalid precision {}", precision);
        break;
    }
}

bool MixRampCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
