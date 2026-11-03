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

class ReverbInfo : public EffectInfoBase {
public:
    struct ParameterVersion1 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ u16 channel_count_max;
        /* 0x0E */ u16 channel_count;
        /* 0x10 */ u32 sample_rate;
        /* 0x14 */ u32 early_mode;
        /* 0x18 */ s32 early_gain;
        /* 0x1C */ s32 pre_delay;
        /* 0x20 */ s32 late_mode;
        /* 0x24 */ s32 late_gain;
        /* 0x28 */ s32 decay_time;
        /* 0x2C */ s32 high_freq_Decay_ratio;
        /* 0x30 */ s32 colouration;
        /* 0x34 */ s32 base_gain;
        /* 0x38 */ s32 wet_gain;
        /* 0x3C */ s32 dry_gain;
        /* 0x40 */ ParameterState state;
    };
    static_assert(sizeof(ParameterVersion1) <= sizeof(EffectInfoBase::InParameterVersion1),
                  "ReverbInfo::ParameterVersion1 has the wrong size!");

    struct ParameterVersion2 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ u16 channel_count_max;
        /* 0x0E */ u16 channel_count;
        /* 0x10 */ u32 sample_rate;
        /* 0x14 */ u32 early_mode;
        /* 0x18 */ s32 early_gain;
        /* 0x1C */ s32 pre_delay;
        /* 0x20 */ s32 late_mode;
        /* 0x24 */ s32 late_gain;
        /* 0x28 */ s32 decay_time;
        /* 0x2C */ s32 high_freq_decay_ratio;
        /* 0x30 */ s32 colouration;
        /* 0x34 */ s32 base_gain;
        /* 0x38 */ s32 wet_gain;
        /* 0x3C */ s32 dry_gain;
        /* 0x40 */ ParameterState state;
    };
    static_assert(sizeof(ParameterVersion2) <= sizeof(EffectInfoBase::InParameterVersion2),
                  "ReverbInfo::ParameterVersion2 has the wrong size!");

    static constexpr u32 MaxDelayLines = 4;
    static constexpr u32 MaxDelayTaps = 10;
    static constexpr u32 NumEarlyModes = 5;
    static constexpr u32 NumLateModes = 5;

    struct ReverbDelayLine {
        void initialize(const s32 delay_time, const f32 decay_rate) {
            buffer.resize(delay_time + 1, 0);
            buffer_end = &buffer[delay_time];
            output = &buffer[0];
            decay = decay_rate;
            sample_count_max = delay_time;
            set_delay(delay_time);
        }

        void set_delay(const s32 delay_time) {
            if (sample_count_max < delay_time) {
                return;
            }
            sample_count = delay_time;
            input = &buffer[0];
        }

        FixedPoint<50, 14> tick(const FixedPoint<50, 14> sample) {
            auto out_sample{read()};

            output++;
            if (output >= buffer_end) {
                output = buffer.data();
            }

            write(sample);
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
                out += sample_count;
            }
            return *out;
        }

        s32 sample_count{};
        s32 sample_count_max{};
        std::vector<FixedPoint<50, 14>> buffer{};
        FixedPoint<50, 14>* buffer_end;
        FixedPoint<50, 14>* input{};
        FixedPoint<50, 14>* output{};
        FixedPoint<50, 14> decay{};
    };

    struct State {
        ReverbDelayLine pre_delay_line;
        ReverbDelayLine center_delay_line;
        std::array<s32, MaxDelayTaps> early_delay_times;
        std::array<FixedPoint<50, 14>, MaxDelayTaps> early_gains;
        s32 pre_delay_time;
        std::array<ReverbDelayLine, MaxDelayLines> decay_delay_lines;
        std::array<ReverbDelayLine, MaxDelayLines> fdn_delay_lines;
        std::array<FixedPoint<50, 14>, MaxDelayLines> hf_decay_gain;
        std::array<FixedPoint<50, 14>, MaxDelayLines> hf_decay_prev_gain;
        std::array<FixedPoint<50, 14>, MaxDelayLines> prev_feedback_output;
    };
    static_assert(sizeof(State) <= sizeof(EffectInfoBase::State),
                  "ReverbInfo::State is too large!");

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
