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
#include "renderer/command/effect/light_limiter.h"

namespace armigo::audio::renderer {
// Update the LightLimiterInfo state according to the given parameters.
// A no-op.
static void update_light_limiter_effect_parameter([[maybe_unused]] const LightLimiterInfo::ParameterVersion2& params,
                                              [[maybe_unused]] LightLimiterInfo::State& state) {}

// Initialize a new LightLimiterInfo state according to the given parameters.
static void initialize_light_limiter_effect(const LightLimiterInfo::ParameterVersion2& params,
                                         LightLimiterInfo::State& state, [[maybe_unused]] const CpuAddr workbuffer) {
    state = {};
    state.samples_average.fill(0.0f);
    state.compression_gain.fill(1.0f);
    state.look_ahead_sample_offsets.fill(0);
    for (u32 i = 0; i < params.channel_count; i++) {
        state.look_ahead_sample_buffers[i].resize(params.look_ahead_samples_max, 0.0f);
    }
}

// Apply a light limiter effect if enabled, according to the current state, on the input mix
// buffers, saving the results to the output mix buffers.
static void apply_light_limiter_effect(const LightLimiterInfo::ParameterVersion2& params,
                                    LightLimiterInfo::State& state, const bool enabled,
                                    std::span<std::span<const s32>> inputs,
                                    std::span<std::span<s32>> outputs, const u32 sample_count,
                                    LightLimiterInfo::StatisticsInternal* statistics) {
    constexpr s64 min{(std::numeric_limits<s32>::min)()};
    constexpr s64 max{(std::numeric_limits<s32>::max)()};

    const auto recip_estimate = [](f64 a) -> f64 {
        s32 q, s;
        f64 r;
        q = (s32)(a * 512.0);               /* a in units of 1/512 rounded down */
        r = 1.0 / (((f64)q + 0.5) / 512.0); /* reciprocal r */
        s = (s32)(256.0 * r + 0.5);         /* r in units of 1/256 rounded to nearest */
        return ((f64)s / 256.0);
    };

    if (enabled) {
        if (statistics && params.statistics_reset_required) {
            for (u32 i = 0; i < params.channel_count; i++) {
                statistics->channel_compression_gain_min[i] = 1.0f;
                statistics->channel_max_sample[i] = 0;
            }
        }

        for (u32 sample_index = 0; sample_index < sample_count; sample_index++) {
            for (u32 channel = 0; channel < params.channel_count; channel++) {
                auto sample{(FixedPoint<49, 15>(inputs[channel][sample_index]) /
                             FixedPoint<49, 15>::one) *
                            params.input_gain};
                auto abs_sample{sample};
                if (sample < 0.0f) {
                    abs_sample = -sample;
                }
                auto coeff{abs_sample > state.samples_average[channel] ? params.attack_coeff
                                                                       : params.release_coeff};
                state.samples_average[channel] +=
                    ((abs_sample - state.samples_average[channel]) * coeff).to_float();

                // Reciprocal estimate
                auto new_average_sample{FixedPoint<49, 15>(
                    recip_estimate(state.samples_average[channel].to_double()))};
                if (params.processing_mode != LightLimiterInfo::ProcessingMode::Mode1) {
                    // Two Newton-Raphson steps
                    auto temp{2.0 - (state.samples_average[channel] * new_average_sample)};
                    new_average_sample = 2.0 - (state.samples_average[channel] * temp);
                }

                auto above_threshold{state.samples_average[channel] > params.threshold};
                auto attenuation{above_threshold ? params.threshold * new_average_sample : 1.0f};
                coeff = attenuation < state.compression_gain[channel] ? params.attack_coeff
                                                                      : params.release_coeff;
                state.compression_gain[channel] +=
                    (attenuation - state.compression_gain[channel]) * coeff;

                auto lookahead_sample{
                    state.look_ahead_sample_buffers[channel]
                                                   [state.look_ahead_sample_offsets[channel]]};

                state.look_ahead_sample_buffers[channel][state.look_ahead_sample_offsets[channel]] =
                    sample;
                state.look_ahead_sample_offsets[channel] =
                    (state.look_ahead_sample_offsets[channel] + 1) % params.look_ahead_samples_min;

                outputs[channel][sample_index] = static_cast<s32>(
                    std::clamp((lookahead_sample * state.compression_gain[channel] *
                                params.output_gain * FixedPoint<49, 15>::one)
                                   .to_long(),
                               min, max));

                if (statistics) {
                    statistics->channel_max_sample[channel] =
                        (std::max)(statistics->channel_max_sample[channel], abs_sample.to_float());
                    statistics->channel_compression_gain_min[channel] =
                        (std::min)(statistics->channel_compression_gain_min[channel],
                                 state.compression_gain[channel].to_float());
                }
            }
        }
    } else {
        for (u32 i = 0; i < params.channel_count; i++) {
            if (params.inputs[i] != params.outputs[i]) {
                std::memcpy(outputs[i].data(), inputs[i].data(), outputs[i].size_bytes());
            }
        }
    }
}

void LightLimiterVersion1Command::process(const CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (u32 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffer(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffer(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<LightLimiterInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == LightLimiterInfo::ParameterState::Updating) {
            update_light_limiter_effect_parameter(parameter, *state_);
        } else if (parameter.state == LightLimiterInfo::ParameterState::Initialized) {
            initialize_light_limiter_effect(parameter, *state_, workbuffer);
        }
    }

    LightLimiterInfo::StatisticsInternal* statistics{nullptr};
    apply_light_limiter_effect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                            processor.sample_count, statistics);
}

bool LightLimiterVersion1Command::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

void LightLimiterVersion2Command::process(const CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (u32 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffer(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffer(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<LightLimiterInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == LightLimiterInfo::ParameterState::Updating) {
            update_light_limiter_effect_parameter(parameter, *state_);
        } else if (parameter.state == LightLimiterInfo::ParameterState::Initialized) {
            initialize_light_limiter_effect(parameter, *state_, workbuffer);
        }
    }

    auto statistics{reinterpret_cast<LightLimiterInfo::StatisticsInternal*>(result_state)};
    apply_light_limiter_effect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                            processor.sample_count, statistics);
}

bool LightLimiterVersion2Command::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
