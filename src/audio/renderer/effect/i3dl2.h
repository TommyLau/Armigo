// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <vector>

#include "renderer/common/common.h"
#include "renderer/effect/effect_info_base.h"
#include "renderer/common/types.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {

class I3dl2ReverbInfo : public EffectInfoBase {
public:
    struct ParameterVersion1 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ u16 channel_count_max;
        /* 0x0E */ u16 channel_count;
        /* 0x10 */ char unk10[0x4];
        /* 0x14 */ u32 sample_rate;
        /* 0x18 */ f32 room_HF_gain;
        /* 0x1C */ f32 reference_HF;
        /* 0x20 */ f32 late_reverb_decay_time;
        /* 0x24 */ f32 late_reverb_HF_decay_ratio;
        /* 0x28 */ f32 room_gain;
        /* 0x2C */ f32 reflection_gain;
        /* 0x30 */ f32 reverb_gain;
        /* 0x34 */ f32 late_reverb_diffusion;
        /* 0x38 */ f32 reflection_delay;
        /* 0x3C */ f32 late_reverb_delay_time;
        /* 0x40 */ f32 late_reverb_density;
        /* 0x44 */ f32 dry_gain;
        /* 0x48 */ ParameterState state;
        /* 0x49 */ char unk49[0x3];
    };
    static_assert(sizeof(ParameterVersion1) <= sizeof(EffectInfoBase::InParameterVersion1),
                  "I3dl2ReverbInfo::ParameterVersion1 has the wrong size!");

    struct ParameterVersion2 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ u16 channel_count_max;
        /* 0x0E */ u16 channel_count;
        /* 0x10 */ char unk10[0x4];
        /* 0x14 */ u32 sample_rate;
        /* 0x18 */ f32 room_HF_gain;
        /* 0x1C */ f32 reference_HF;
        /* 0x20 */ f32 late_reverb_decay_time;
        /* 0x24 */ f32 late_reverb_HF_decay_ratio;
        /* 0x28 */ f32 room_gain;
        /* 0x2C */ f32 reflection_gain;
        /* 0x30 */ f32 reverb_gain;
        /* 0x34 */ f32 late_reverb_diffusion;
        /* 0x38 */ f32 reflection_delay;
        /* 0x3C */ f32 late_reverb_delay_time;
        /* 0x40 */ f32 late_reverb_density;
        /* 0x44 */ f32 dry_gain;
        /* 0x48 */ ParameterState state;
        /* 0x49 */ char unk49[0x3];
    };
    static_assert(sizeof(ParameterVersion2) <= sizeof(EffectInfoBase::InParameterVersion2),
                  "I3dl2ReverbInfo::ParameterVersion2 has the wrong size!");

    static constexpr u32 MaxDelayLines = 4;
    static constexpr u32 MaxDelayTaps = 20;

    struct I3dl2DelayLine {
        void initialize(const s32 delay_time) {
            max_delay = delay_time;
            buffer.resize(delay_time + 1, 0);
            buffer_end = &buffer[delay_time];
            output = &buffer[0];
            set_delay(delay_time);
            wet_gain = 0.0f;
        }

        void set_delay(const s32 delay_time) {
            if (max_delay < delay_time) {
                return;
            }
            delay = delay_time;
            input = &buffer[(output - buffer.data() + delay) % (max_delay + 1)];
        }

        FixedPoint<50, 14> tick(const FixedPoint<50, 14> sample) {
            write(sample);

            auto out_sample{read()};

            output++;
            if (output >= buffer_end) {
                output = buffer.data();
            }

            return out_sample;
        }

        FixedPoint<50, 14> read() const {
            return *output;
        }

        void write(const FixedPoint<50, 14> sample) {
            *input = sample;
            input++;
            if (input >= buffer_end) {
                input = buffer.data();
            }
        }

        FixedPoint<50, 14> tap_out(const s32 index) const {
            auto out{input - (index + 1)};
            if (out < buffer.data()) {
                out += max_delay + 1;
            }
            return *out;
        }

        std::vector<FixedPoint<50, 14>> buffer{};
        FixedPoint<50, 14>* buffer_end{};
        s32 max_delay{};
        FixedPoint<50, 14>* input{};
        FixedPoint<50, 14>* output{};
        s32 delay{};
        f32 wet_gain{};
    };

    struct State {
        f32 lowpass_0;
        f32 lowpass_1;
        f32 lowpass_2;
        I3dl2DelayLine early_delay_line;
        std::array<s32, MaxDelayTaps> early_tap_steps;
        f32 early_gain;
        f32 late_gain;
        s32 early_to_late_taps;
        std::array<I3dl2DelayLine, MaxDelayLines> fdn_delay_lines;
        std::array<I3dl2DelayLine, MaxDelayLines> decay_delay_lines0;
        std::array<I3dl2DelayLine, MaxDelayLines> decay_delay_lines1;
        f32 last_reverb_echo;
        I3dl2DelayLine center_delay_line;
        std::array<std::array<f32, 3>, MaxDelayLines> lowpass_coeff;
        std::array<f32, MaxDelayLines> shelf_filter;
        f32 dry_gain;
    };
    static_assert(sizeof(State) <= sizeof(EffectInfoBase::State),
                  "I3dl2ReverbInfo::State is too large!");

    // Update the info with new parameters, version 1.
    void update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion1& in_params,
                const PoolMapper& pool_mapper) override;

    // Update the info with new parameters, version 2.
    void update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion2& in_params,
                const PoolMapper& pool_mapper) override;

    // Update the info after command generation. Usually only changes its state.
    void update_for_command_generation() override;

    // Initialize a new result state. Version 2 only, unused.
    void initialize_result_state(EffectResultState& result_state) override;

    // Update the host-side state with the ADSP-side state. Version 2 only, unused.
    void update_result_state(EffectResultState& cpu_state, EffectResultState& dsp_state) override;

    // Get a workbuffer assigned to this effect with the given index.
    CpuAddr get_workbuffer(s32 index) override;
};

} // namespace armigo::audio::renderer
