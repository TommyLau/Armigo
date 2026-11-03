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
#include "renderer/voice/voice_state.h"
#include <numeric>
#include <bit>

namespace armigo::audio::renderer {
// Biquad filter float implementation.
void apply_biquad_filter_float(std::span<s32> output, std::span<const s32> input,
                            std::array<s16, 3>& b_, std::array<s16, 2>& a_,
                            VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr f64 min{(std::numeric_limits<s32>::min)()};
    constexpr f64 max{(std::numeric_limits<s32>::max)()};
    std::array<f64, 3> b{FixedPoint<50, 14>::from_base(b_[0]).to_double(),
                         FixedPoint<50, 14>::from_base(b_[1]).to_double(),
                         FixedPoint<50, 14>::from_base(b_[2]).to_double()};
    std::array<f64, 2> a{FixedPoint<50, 14>::from_base(a_[0]).to_double(),
                         FixedPoint<50, 14>::from_base(a_[1]).to_double()};
    std::array<f64, 4> s{std::bit_cast<f64>(state.s0), std::bit_cast<f64>(state.s1),
                         std::bit_cast<f64>(state.s2), std::bit_cast<f64>(state.s3)};

    for (u32 i = 0; i < sample_count; i++) {
        f64 in_sample{static_cast<f64>(input[i])};
        auto sample{in_sample * b[0] + s[0] * b[1] + s[1] * b[2] + s[2] * a[0] + s[3] * a[1]};

        output[i] = static_cast<s32>(std::clamp(sample, min, max));

        s[1] = s[0];
        s[0] = in_sample;
        s[3] = s[2];
        s[2] = sample;
    }

    state.s0 = std::bit_cast<s64>(s[0]);
    state.s1 = std::bit_cast<s64>(s[1]);
    state.s2 = std::bit_cast<s64>(s[2]);
    state.s3 = std::bit_cast<s64>(s[3]);
}

// Biquad filter float implementation with native float coefficients.
void apply_biquad_filter_float2(std::span<s32> output, std::span<const s32> input,
                             std::array<f32, 3>& b, std::array<f32, 2>& a,
                             VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr f64 min{std::numeric_limits<s32>::min()};
    constexpr f64 max{std::numeric_limits<s32>::max()};

    std::array<f64, 3> b_double{static_cast<f64>(b[0]), static_cast<f64>(b[1]),
                                static_cast<f64>(b[2])};
    std::array<f64, 2> a_double{static_cast<f64>(a[0]), static_cast<f64>(a[1])};
    std::array<f64, 4> s{std::bit_cast<f64>(state.s0), std::bit_cast<f64>(state.s1),
                         std::bit_cast<f64>(state.s2), std::bit_cast<f64>(state.s3)};

    for (u32 i = 0; i < sample_count; i++) {
        f64 in_sample{static_cast<f64>(input[i])};
        auto sample{in_sample * b_double[0] + s[0] * b_double[1] + s[1] * b_double[2] +
                    s[2] * a_double[0] + s[3] * a_double[1]};

        output[i] = static_cast<s32>(std::clamp(sample, min, max));

        s[1] = s[0];
        s[0] = in_sample;
        s[3] = s[2];
        s[2] = sample;
    }

    state.s0 = std::bit_cast<s64>(s[0]);
    state.s1 = std::bit_cast<s64>(s[1]);
    state.s2 = std::bit_cast<s64>(s[2]);
    state.s3 = std::bit_cast<s64>(s[3]);
}

// Biquad filter s32 implementation.
static void apply_biquad_filter_int(std::span<s32> output, std::span<const s32> input,
                                 std::array<s16, 3>& b, std::array<s16, 2>& a,
                                 VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr s64 min{(std::numeric_limits<s32>::min)()};
    constexpr s64 max{(std::numeric_limits<s32>::max)()};

    for (u32 i = 0; i < sample_count; i++) {
        const s64 in_sample{input[i]};
        const s64 sample{in_sample * b[0] + state.s0};
        const s64 out_sample{std::clamp<s64>((sample + (1 << 13)) >> 14, min, max)};

        output[i] = static_cast<s32>(out_sample);

        state.s0 = state.s1 + b[1] * in_sample + a[0] * out_sample;
        state.s1 = b[2] * in_sample + a[1] * out_sample;
    }
}

void BiquadFilterCommand::process(const CommandListProcessor& processor) {
    auto state_{reinterpret_cast<VoiceState::BiquadFilterState*>(state)};
    if (needs_init) {
        *state_ = {};
    }

    auto input_buffer{
        processor.mix_buffer(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffer(output * processor.sample_count, processor.sample_count)};

    if (use_float_processing) {
        // REV15+: Use native float coefficients if available
        if (use_float_coefficients) {
            apply_biquad_filter_float2(output_buffer, input_buffer, biquad_float.numerator,
                                    biquad_float.denominator, *state_, processor.sample_count);
        } else {
            apply_biquad_filter_float(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                                   processor.sample_count);
        }
    } else {
        apply_biquad_filter_int(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                             processor.sample_count);
    }
}

bool BiquadFilterCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
