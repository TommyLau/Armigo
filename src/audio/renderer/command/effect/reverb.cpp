// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <numbers>
#include <ranges>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/effect/reverb.h"
#include <ranges>

namespace armigo::audio::renderer {

constexpr std::array<f32, ReverbInfo::MaxDelayLines> FdnMaxDelayLineTimes = {
    53.9532470703125f,
    79.19256591796875f,
    116.23876953125f,
    170.61529541015625f,
};

constexpr std::array<f32, ReverbInfo::MaxDelayLines> DecayMaxDelayLineTimes = {
    7.0f,
    9.0f,
    13.0f,
    17.0f,
};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayTaps + 1>, ReverbInfo::NumEarlyModes>
    EarlyDelayTimes = {
        {{0.000000f, 3.500000f, 2.799988f, 3.899963f, 2.699951f, 13.399963f, 7.899963f, 8.399963f,
          9.899963f, 12.000000f, 12.500000f},
         {0.000000f, 11.799988f, 5.500000f, 11.199951f, 10.399963f, 38.099976f, 22.199951f,
          29.599976f, 21.199951f, 24.799988f, 40.000000f},
         {0.000000f, 41.500000f, 20.500000f, 41.299988f, 0.000000f, 29.500000f, 33.799988f,
          45.199951f, 46.799988f, 0.000000f, 50.000000f},
         {33.099976f, 43.299988f, 22.799988f, 37.899963f, 14.899963f, 35.299988f, 17.899963f,
          34.199951f, 0.000000f, 43.299988f, 50.000000f},
         {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
          0.000000f, 0.000000f, 0.000000f}},
};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayTaps>, ReverbInfo::NumEarlyModes>
    EarlyDelayGains = {{
        {0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f,
         0.679993f, 0.679993f},
        {0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.679993f, 0.679993f,
         0.679993f, 0.679993f},
        {0.500000f, 0.699951f, 0.699951f, 0.679993f, 0.500000f, 0.679993f, 0.679993f, 0.699951f,
         0.679993f, 0.000000f},
        {0.929993f, 0.919983f, 0.869995f, 0.859985f, 0.939941f, 0.809998f, 0.799988f, 0.769958f,
         0.759949f, 0.649963f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
         0.000000f, 0.000000f},
    }};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayLines>, ReverbInfo::NumLateModes>
    FdnDelayTimes = {{
        {53.953247f, 79.192566f, 116.238770f, 130.615295f},
        {53.953247f, 79.192566f, 116.238770f, 170.615295f},
        {5.000000f, 10.000000f, 5.000000f, 10.000000f},
        {47.029968f, 71.000000f, 103.000000f, 170.000000f},
        {53.953247f, 79.192566f, 116.238770f, 170.615295f},
    }};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayLines>, ReverbInfo::NumLateModes>
    DecayDelayTimes = {{
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
        {1.000000f, 1.000000f, 1.000000f, 1.000000f},
        {7.000000f, 7.000000f, 13.000000f, 9.000000f},
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
    }};

// Update the ReverbInfo state according to the given parameters.
static void update_reverb_effect_parameter(const ReverbInfo::ParameterVersion2& params,
                                        ReverbInfo::State& state) {
    const auto pow_10 = [](f32 val) -> f32 {
        return (val >= 0.0f) ? 1.0f : (val <= -5.3f) ? 0.0f : std::pow(10.0f, val);
    };

    const auto cos = [](f32 degrees) -> f32 {
        return std::cos(degrees * std::numbers::pi_v<f32> / 180.0f);
    };

    static bool unk_initialized{false};
    static FixedPoint<50, 14> unk_value{};

    auto sample_rate = FixedPoint<50, 14>::from_base(params.sample_rate);
    if (sample_rate.to_float() < 1.0f) sample_rate = FixedPoint<50, 14>(1.0f);

    auto pre_delay_time = FixedPoint<50, 14>::from_base(params.pre_delay);
    if (pre_delay_time.to_float() < 0.0f) pre_delay_time = FixedPoint<50, 14>(0.0f);

    for (u32 i = 0; i < ReverbInfo::MaxDelayTaps; i++) {
        int target_delay = ((pre_delay_time + EarlyDelayTimes[params.early_mode][i]) * sample_rate).to_int();
        if (target_delay < 0) target_delay = 0;
        if (target_delay > state.pre_delay_line.sample_count_max) target_delay = state.pre_delay_line.sample_count_max;

        int old_delay = state.early_delay_times[i] - 1;
        int smooth_delay = old_delay + (target_delay - old_delay) / 4;
        state.early_delay_times[i] = smooth_delay + 1;

        auto target_gain = FixedPoint<50, 14>::from_base(params.early_gain) * EarlyDelayGains[params.early_mode][i];
        if (target_gain.to_float() < 0.0f) target_gain = FixedPoint<50, 14>(0.0f);
        if (target_gain.to_float() > 1.0f) target_gain = FixedPoint<50, 14>(1.0f);

        state.early_gains[i] = state.early_gains[i] + (target_gain - state.early_gains[i]) / 4;
    }

    if (params.channel_count == 2) {
        state.early_gains[4] *= FixedPoint<50, 14>(0.5f);
        state.early_gains[5] *= FixedPoint<50, 14>(0.5f);
    }

    int target_pre_time = ((pre_delay_time + EarlyDelayTimes[params.early_mode][10]) * sample_rate).to_int();
    if (target_pre_time < 0) target_pre_time = 0;
    if (target_pre_time > state.pre_delay_line.sample_count_max) target_pre_time = state.pre_delay_line.sample_count_max;

    int old_pre_time = state.pre_delay_time;
    state.pre_delay_time = old_pre_time + (target_pre_time - old_pre_time) / 4;

    if (!unk_initialized) {
        unk_value = cos((1280.0f / sample_rate).to_float());
        unk_initialized = true;
    }

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        int target_fdn_delay = (FdnDelayTimes[params.late_mode][i] * sample_rate).to_int();
        if (target_fdn_delay < 0) target_fdn_delay = 0;
        if (target_fdn_delay > state.fdn_delay_lines[i].sample_count_max) target_fdn_delay = state.fdn_delay_lines[i].sample_count_max;

        int old_fdn = state.fdn_delay_lines[i].sample_count;
        state.fdn_delay_lines[i].sample_count = old_fdn + (target_fdn_delay - old_fdn) / 4;
        state.fdn_delay_lines[i].buffer_end = &state.fdn_delay_lines[i].buffer[state.fdn_delay_lines[i].sample_count - 1];

        int target_decay_delay = (DecayDelayTimes[params.late_mode][i] * sample_rate).to_int();
        if (target_decay_delay < 0) target_decay_delay = 0;
        if (target_decay_delay > state.decay_delay_lines[i].sample_count_max) target_decay_delay = state.decay_delay_lines[i].sample_count_max;

        int old_decay = state.decay_delay_lines[i].sample_count;
        state.decay_delay_lines[i].sample_count = old_decay + (target_decay_delay - old_decay) / 4;
        state.decay_delay_lines[i].buffer_end = &state.decay_delay_lines[i].buffer[state.decay_delay_lines[i].sample_count - 1];

        auto colouration = FixedPoint<50, 14>::from_base(params.colouration);
        if (colouration.to_float() < 0.0f) colouration = FixedPoint<50, 14>(0.0f);
        if (colouration.to_float() > 1.0f) colouration = FixedPoint<50, 14>(1.0f);

        state.decay_delay_lines[i].decay = state.decay_delay_lines[i].decay + (0.5999755859375f * (1.0f - colouration) - state.decay_delay_lines[i].decay) / 4;

        auto decay_time_fp = FixedPoint<50, 14>::from_base(params.decay_time);
        if (decay_time_fp.to_float() <= 0.0f) decay_time_fp = FixedPoint<50, 14>(0.0001f);

        auto a = (FixedPoint<50, 14>(state.fdn_delay_lines[i].sample_count_max) + state.decay_delay_lines[i].sample_count_max) * -3;
        auto b = a / (decay_time_fp * sample_rate);

        auto hf_decay_ratio = FixedPoint<50, 14>::from_base(params.high_freq_decay_ratio);
        if (hf_decay_ratio.to_float() < 0.001f) hf_decay_ratio = FixedPoint<50, 14>(0.001f);
        if (hf_decay_ratio.to_float() > 0.999f) hf_decay_ratio = FixedPoint<50, 14>(0.999f);

        FixedPoint<50, 14> c{0.0f}, d{0.0f};
        if (hf_decay_ratio.to_float() > 0.9949f) {
            c = 0.0f;
            d = 1.0f;
        } else {
            auto e = pow_10(((((1.0f / hf_decay_ratio.to_float()) - 1.0f) * 2) / 100 * (b / 10)).to_float());
            if (e < 0.0001f) e = 0.0001f;
            if (e > 1.0f) e = 1.0f;

            auto f = 1.0f - e;
            if (f < 0.0001f) f = 0.0001f;

            auto g = 2.0f - (unk_value.to_float() * e * 2);

            auto h_sq = g * g - f * f * 4;
            if (h_sq < 0.0f) h_sq = 0.0f;

            auto h = std::sqrt(h_sq);

            c = (g - h) / (f * 2.0f);
            d = 1.0f - c;
        }

        state.hf_decay_prev_gain[i] = state.hf_decay_prev_gain[i] + (c - state.hf_decay_prev_gain[i]) / 4;
        state.hf_decay_gain[i] = state.hf_decay_gain[i] + (pow_10((b / 1000).to_float()) * d * 0.70709228515625f - state.hf_decay_gain[i]) / 4;

        state.prev_feedback_output[i] = 0;
    }
}

// Initialize a new ReverbInfo state according to the given parameters.
static void initialize_reverb_effect(const ReverbInfo::ParameterVersion2& params,
                                   ReverbInfo::State& state, [[maybe_unused]] const CpuAddr workbuffer,
                                   const bool long_size_pre_delay_supported) {
    state = {};

    auto delay{FixedPoint<50, 14>::from_base(params.sample_rate)};

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        auto fdn_delay_time{(FdnMaxDelayLineTimes[i] * delay).to_uint_floor()};
        state.fdn_delay_lines[i].initialize(fdn_delay_time, 1.0f);

        auto decay_delay_time{(DecayMaxDelayLineTimes[i] * delay).to_uint_floor()};
        state.decay_delay_lines[i].initialize(decay_delay_time, 0.0f);
    }

    const auto pre_delay{long_size_pre_delay_supported ? 350.0f : 150.0f};
    const auto pre_delay_line{(pre_delay * delay).to_uint_floor()};
    state.pre_delay_line.initialize(pre_delay_line, 1.0f);

    const auto center_delay_time{(5 * delay).to_uint_floor()};
    state.center_delay_line.initialize(center_delay_time, 1.0f);

    update_reverb_effect_parameter(params, state);

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        std::ranges::fill(state.fdn_delay_lines[i].buffer, 0);
        std::ranges::fill(state.decay_delay_lines[i].buffer, 0);
    }
    std::ranges::fill(state.center_delay_line.buffer, 0);
    std::ranges::fill(state.pre_delay_line.buffer, 0);
}

// Pass-through the effect, copying input to output directly, with no reverb applied.
static void apply_reverb_effect_bypass(std::span<std::span<const s32>> inputs,
                                    std::span<std::span<s32>> outputs, const u32 channel_count,
                                    [[maybe_unused]] const u32 sample_count) {
    for (u32 i = 0; i < channel_count; i++) {
        if (inputs[i].data() != outputs[i].data()) {
            std::memcpy(outputs[i].data(), inputs[i].data(), outputs[i].size_bytes());
        }
    }
}

// Tick the delay lines, reading and returning their current output, and writing a new decaying
// sample (mix).
static FixedPoint<50, 14> axfx2_all_pass_tick(ReverbInfo::ReverbDelayLine& decay,
                                                   ReverbInfo::ReverbDelayLine& fdn,
                                                   const FixedPoint<50, 14> mix) {
    const auto val{decay.read()};
    const auto mixed{mix - (val * decay.decay)};
    const auto out{decay.tick(mixed) + (mixed * decay.decay)};

    fdn.tick(out);
    return out;
}

// Impl. Apply a Reverb according to the current state, on the input mix buffers,
// saving the results to the output mix buffers.
template <size_t NumChannels>
static void apply_reverb_effect(const ReverbInfo::ParameterVersion2& params, ReverbInfo::State& state,
                              std::span<std::span<const s32>> inputs,
                              std::span<std::span<s32>> outputs, const u32 sample_count) {
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes1Ch{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes2Ch{
        0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes4Ch{
        0, 0, 1, 1, 0, 1, 2, 2, 3, 3,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes6Ch{
        0, 0, 1, 1, 2, 2, 4, 4, 5, 5,
    };

    std::span<const u8> tap_indexes{};
    if constexpr (NumChannels == 1) {
        tap_indexes = OutTapIndexes1Ch;
    } else if constexpr (NumChannels == 2) {
        tap_indexes = OutTapIndexes2Ch;
    } else if constexpr (NumChannels == 4) {
        tap_indexes = OutTapIndexes4Ch;
    } else if constexpr (NumChannels == 6) {
        tap_indexes = OutTapIndexes6Ch;
    }

    for (u32 sample_index = 0; sample_index < sample_count; sample_index++) {
        std::array<FixedPoint<50, 14>, NumChannels> output_samples{};

        for (u32 early_tap = 0; early_tap < ReverbInfo::MaxDelayTaps; early_tap++) {
            const auto sample{state.pre_delay_line.tap_out(state.early_delay_times[early_tap]) *
                              state.early_gains[early_tap]};
            output_samples[tap_indexes[early_tap]] += sample;
            if constexpr (NumChannels == 6) {
                output_samples[static_cast<u32>(Channels::LFE)] += sample;
            }
        }

        if constexpr (NumChannels == 6) {
            output_samples[static_cast<u32>(Channels::LFE)] *= 0.2f;
        }

        FixedPoint<50, 14> input_sample{};
        for (u32 channel = 0; channel < NumChannels; channel++) {
            input_sample += inputs[channel][sample_index];
        }

        input_sample *= 64;
        input_sample *= FixedPoint<50, 14>::from_base(params.base_gain);
        state.pre_delay_line.write(input_sample);

        for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
            state.prev_feedback_output[i] =
                state.prev_feedback_output[i] * state.hf_decay_prev_gain[i] +
                state.fdn_delay_lines[i].read() * state.hf_decay_gain[i];
        }

        FixedPoint<50, 14> pre_delay_sample{
            state.pre_delay_line.tap_out(state.pre_delay_time) *
            FixedPoint<50, 14>::from_base(params.late_gain)};

        std::array<FixedPoint<50, 14>, ReverbInfo::MaxDelayLines> mix_matrix{
            state.prev_feedback_output[2] + state.prev_feedback_output[1] + pre_delay_sample,
            -state.prev_feedback_output[0] - state.prev_feedback_output[3] + pre_delay_sample,
            state.prev_feedback_output[0] - state.prev_feedback_output[3] + pre_delay_sample,
            state.prev_feedback_output[1] - state.prev_feedback_output[2] + pre_delay_sample,
        };

        std::array<FixedPoint<50, 14>, ReverbInfo::MaxDelayLines> allpass_samples{};
        for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
            allpass_samples[i] = axfx2_all_pass_tick(state.decay_delay_lines[i],
                                                  state.fdn_delay_lines[i], mix_matrix[i]);
        }

        const auto dry_gain{FixedPoint<50, 14>::from_base(params.dry_gain)};
        const auto wet_gain{FixedPoint<50, 14>::from_base(params.wet_gain)};

        if constexpr (NumChannels == 6) {
            const std::array<FixedPoint<50, 14>, MaxChannels> allpass_outputs{
                allpass_samples[0], allpass_samples[1], allpass_samples[2] - allpass_samples[3],
                allpass_samples[3], allpass_samples[2], allpass_samples[3],
            };

            for (u32 channel = 0; channel < NumChannels; channel++) {
                auto in_sample{inputs[channel][sample_index] * dry_gain};

                FixedPoint<50, 14> allpass{};
                if (channel == static_cast<u32>(Channels::Center)) {
                    allpass = state.center_delay_line.tick(allpass_outputs[channel] * 0.5f);
                } else {
                    allpass = allpass_outputs[channel];
                }

                auto out_sample{((output_samples[channel] + allpass) * wet_gain) / 64};
                outputs[channel][sample_index] = (in_sample + out_sample).to_int();
            }
        } else {
            for (u32 channel = 0; channel < NumChannels; channel++) {
                auto in_sample{inputs[channel][sample_index] * dry_gain};
                auto out_sample{((output_samples[channel] + allpass_samples[channel]) * wet_gain) /
                                64};
                outputs[channel][sample_index] = (in_sample + out_sample).to_int();
            }
        }
    }
}

// Apply a Reverb if enabled, according to the current state, on the input mix buffers,
// saving the results to the output mix buffers.
static void apply_reverb_effect(const ReverbInfo::ParameterVersion2& params, ReverbInfo::State& state,
                              const bool enabled, std::span<std::span<const s32>> inputs,
                              std::span<std::span<s32>> outputs, const u32 sample_count) {
    if (enabled) {
        switch (params.channel_count) {
        case 0:
            return;
        case 1:
            apply_reverb_effect<1>(params, state, inputs, outputs, sample_count);
            break;
        case 2:
            apply_reverb_effect<2>(params, state, inputs, outputs, sample_count);
            break;
        case 4:
            apply_reverb_effect<4>(params, state, inputs, outputs, sample_count);
            break;
        case 6:
            apply_reverb_effect<6>(params, state, inputs, outputs, sample_count);
            break;
        default:
            apply_reverb_effect_bypass(inputs, outputs, params.channel_count, sample_count);
            break;
        }
    } else {
        apply_reverb_effect_bypass(inputs, outputs, params.channel_count, sample_count);
    }
}

void ReverbCommand::process(const CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (u32 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffer(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffer(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<ReverbInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == ReverbInfo::ParameterState::Updating) {
            update_reverb_effect_parameter(parameter, *state_);
        } else if (parameter.state == ReverbInfo::ParameterState::Initialized) {
            initialize_reverb_effect(parameter, *state_, workbuffer, long_size_pre_delay_supported);
        }
    }
    apply_reverb_effect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                      processor.sample_count);
}

bool ReverbCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
