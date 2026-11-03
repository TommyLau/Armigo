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
#include "renderer/command/effect/delay.h"

namespace armigo::audio::renderer {
// Update the DelayInfo state according to the given parameters.
static void set_delay_effect_parameter(const DelayInfo::ParameterVersion1& params,
                                    DelayInfo::State& state) {
    auto channel_spread{params.channel_spread};
    state.feedback_gain = params.feedback_gain * 0.97998046875f;
    state.delay_feedback_gain = state.feedback_gain * (1.0f - channel_spread);
    if (params.channel_count == 4 || params.channel_count == 6) {
        channel_spread >>= 1;
    }
    state.delay_feedback_cross_gain = channel_spread * state.feedback_gain;
    state.lowpass_feedback_gain = params.lowpass_amount * 0.949951171875f;
    state.lowpass_gain = 1.0f - state.lowpass_feedback_gain;
}

// Initialize a new DelayInfo state according to the given parameters.
static void initialize_delay_effect(const DelayInfo::ParameterVersion1& params,
                                  DelayInfo::State& state,
                                  [[maybe_unused]] const CpuAddr workbuffer) {
    state = {};

    for (u32 channel = 0; channel < params.channel_count; channel++) {
        FixedPoint<32, 32> sample_count_max{0.064f};
        sample_count_max *= params.sample_rate.to_int_floor() * params.delay_time_max;

        FixedPoint<18, 14> delay_time{params.delay_time};
        delay_time *= params.sample_rate / 1000;
        FixedPoint<32, 32> sample_count{delay_time};

        if (sample_count > sample_count_max) {
            sample_count = sample_count_max;
        }

        state.delay_lines[channel].sample_count_max = sample_count_max.to_int_floor();
        state.delay_lines[channel].sample_count = sample_count.to_int_floor();
        state.delay_lines[channel].buffer.resize(state.delay_lines[channel].sample_count, 0);
        if (state.delay_lines[channel].sample_count == 0) {
            state.delay_lines[channel].buffer.push_back(0);
        }
        state.delay_lines[channel].buffer_pos = 0;
        state.delay_lines[channel].decay_rate = 1.0f;
    }

    set_delay_effect_parameter(params, state);
}

// Delay effect impl, according to the parameters and current state, on the input mix buffers,
// saving the results to the output mix buffers.
template <size_t NumChannels>
static void apply_delay(const DelayInfo::ParameterVersion1& params, DelayInfo::State& state,
                       std::span<std::span<const s32>> inputs, std::span<std::span<s32>> outputs,
                       const u32 sample_count) {
    for (u32 sample_index = 0; sample_index < sample_count; sample_index++) {
        std::array<FixedPoint<50, 14>, NumChannels> input_samples{};
        for (u32 channel = 0; channel < NumChannels; channel++) {
            input_samples[channel] = inputs[channel][sample_index] * 64;
        }

        std::array<FixedPoint<50, 14>, NumChannels> delay_samples{};
        for (u32 channel = 0; channel < NumChannels; channel++) {
            delay_samples[channel] = state.delay_lines[channel].read();
        }

        // clang-format off
        std::array<std::array<FixedPoint<18, 14>, NumChannels>, NumChannels> matrix{};
        if constexpr (NumChannels == 1) {
            matrix = {{
                {state.feedback_gain},
            }};
        } else if constexpr (NumChannels == 2) {
            matrix = {{
                {state.delay_feedback_gain, state.delay_feedback_cross_gain},
                {state.delay_feedback_cross_gain, state.delay_feedback_gain},
            }};
        } else if constexpr (NumChannels == 4) {
            matrix = {{
                {state.delay_feedback_gain, state.delay_feedback_cross_gain, state.delay_feedback_cross_gain, 0.0f},
                {state.delay_feedback_cross_gain, state.delay_feedback_gain, 0.0f, state.delay_feedback_cross_gain},
                {state.delay_feedback_cross_gain, 0.0f, state.delay_feedback_gain, state.delay_feedback_cross_gain},
                {0.0f, state.delay_feedback_cross_gain, state.delay_feedback_cross_gain, state.delay_feedback_gain},
            }};
        } else if constexpr (NumChannels == 6) {
            matrix = {{
                {state.delay_feedback_gain, 0.0f, state.delay_feedback_cross_gain, 0.0f, state.delay_feedback_cross_gain, 0.0f},
                {0.0f, state.delay_feedback_gain, state.delay_feedback_cross_gain, 0.0f, 0.0f, state.delay_feedback_cross_gain},
                {state.delay_feedback_cross_gain, state.delay_feedback_cross_gain, state.delay_feedback_gain, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, params.feedback_gain, 0.0f, 0.0f},
                {state.delay_feedback_cross_gain, 0.0f, 0.0f, 0.0f, state.delay_feedback_gain, state.delay_feedback_cross_gain},
                {0.0f, state.delay_feedback_cross_gain, 0.0f, 0.0f, state.delay_feedback_cross_gain, state.delay_feedback_gain},
            }};
        }
        // clang-format on

        std::array<FixedPoint<50, 14>, NumChannels> gained_samples{};
        for (u32 channel = 0; channel < NumChannels; channel++) {
            FixedPoint<50, 14> delay{};
            for (u32 j = 0; j < NumChannels; j++) {
                delay += delay_samples[j] * matrix[j][channel];
            }
            gained_samples[channel] = input_samples[channel] * params.in_gain + delay;
        }

        for (u32 channel = 0; channel < NumChannels; channel++) {
            state.lowpass_z[channel] = gained_samples[channel] * state.lowpass_gain +
                                       state.lowpass_z[channel] * state.lowpass_feedback_gain;
            state.delay_lines[channel].write(state.lowpass_z[channel]);
        }

        for (u32 channel = 0; channel < NumChannels; channel++) {
            outputs[channel][sample_index] = (input_samples[channel] * params.dry_gain +
                                              delay_samples[channel] * params.wet_gain)
                                                 .to_int_floor() /
                                             64;
        }
    }
}

// Apply a delay effect if enabled, according to the parameters and current state, on the input mix
// buffers, saving the results to the output mix buffers.
static void apply_delay_effect(const DelayInfo::ParameterVersion1& params, DelayInfo::State& state,
                             const bool enabled, std::span<std::span<const s32>> inputs,
                             std::span<std::span<s32>> outputs, const u32 sample_count) {

    if (!is_channel_count_valid(params.channel_count)) {
        AUDIO_LOG_ERROR("Invalid delay channels {}", params.channel_count);
        return;
    }

    if (enabled) {
        switch (params.channel_count) {
        case 1:
            apply_delay<1>(params, state, inputs, outputs, sample_count);
            break;
        case 2:
            apply_delay<2>(params, state, inputs, outputs, sample_count);
            break;
        case 4:
            apply_delay<4>(params, state, inputs, outputs, sample_count);
            break;
        case 6:
            apply_delay<6>(params, state, inputs, outputs, sample_count);
            break;
        default:
            for (u32 channel = 0; channel < params.channel_count; channel++) {
                if (inputs[channel].data() != outputs[channel].data()) {
                    std::memcpy(outputs[channel].data(), inputs[channel].data(),
                                sample_count * sizeof(s32));
                }
            }
            break;
        }
    } else {
        for (u32 channel = 0; channel < params.channel_count; channel++) {
            if (inputs[channel].data() != outputs[channel].data()) {
                std::memcpy(outputs[channel].data(), inputs[channel].data(),
                            sample_count * sizeof(s32));
            }
        }
    }
}

void DelayCommand::process(const CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (s16 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffer(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffer(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<DelayInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == DelayInfo::ParameterState::Updating) {
            set_delay_effect_parameter(parameter, *state_);
        } else if (parameter.state == DelayInfo::ParameterState::Initialized) {
            initialize_delay_effect(parameter, *state_, workbuffer);
        }
    }
    apply_delay_effect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                     processor.sample_count);
}

bool DelayCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
