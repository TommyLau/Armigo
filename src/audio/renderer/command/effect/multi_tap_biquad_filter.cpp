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
#include "renderer/command/effect/biquad_filter.h"
#include "renderer/command/effect/multi_tap_biquad_filter.h"

namespace armigo::audio::renderer {

void MultiTapBiquadFilterCommand::process(const CommandListProcessor& processor) {
    if (filter_tap_count > MaxBiquadFilters) {
        AUDIO_LOG_ERROR("Too many filter taps! {}", filter_tap_count);
        filter_tap_count = MaxBiquadFilters;
    }

    auto input_buffer{
        processor.mix_buffer(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffer(output * processor.sample_count, processor.sample_count)};

    // The taps are chained: the first filters the input into the output and
    // every later tap filters the output in place. Voices filter their own
    // buffer in place, so input and output are normally the same.
    for (u32 i = 0; i < filter_tap_count; i++) {
        if (i > 0) {
            input_buffer = output_buffer;
        }
        auto state{reinterpret_cast<VoiceState::BiquadFilterState*>(states[i])};
        if (needs_init[i]) {
            *state = {};
        }

        // REV15+: Use native float coefficients if available
        if (use_float_coefficients) {
            apply_biquad_filter_float2(output_buffer, input_buffer, biquads_float[i].numerator,
                                    biquads_float[i].denominator, *state, processor.sample_count);
        } else {
            apply_biquad_filter_float(output_buffer, input_buffer, biquads[i].b, biquads[i].a, *state,
                                   processor.sample_count);
        }
    }
}

bool MultiTapBiquadFilterCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
