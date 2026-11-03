// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <algorithm>
#include <limits>
#include <span>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/mix/mix.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
// Mix input mix buffer into output mix buffer, with volume applied to the input.
template <size_t Q>
static void apply_mix(std::span<s32> output, std::span<const s32> input, const f32 volume_,
                     const u32 sample_count) {
    const FixedPoint<64 - Q, Q> volume{volume_};
    for (u32 i = 0; i < sample_count; i++) {
        output[i] = (output[i] + input[i] * volume).to_int();
    }
}

void MixCommand::process(const CommandListProcessor& processor) {
    auto output{processor.mix_buffer(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffer(input_index * processor.sample_count,
                                             processor.sample_count)};

    // If volume is 0, nothing will be added to the output, so just skip.
    if (volume == 0.0f) {
        return;
    }

    switch (precision) {
    case 15:
        apply_mix<15>(output, input, volume, processor.sample_count);
        break;

    case 23:
        apply_mix<23>(output, input, volume, processor.sample_count);
        break;

    default:
        AUDIO_LOG_ERROR("Invalid precision {}", precision);
        break;
    }
}

bool MixCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
