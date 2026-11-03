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

class DelayInfo : public EffectInfoBase {
public:
    struct ParameterVersion1 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ u16 channel_count_max;
        /* 0x0E */ u16 channel_count;
        /* 0x10 */ u32 delay_time_max;
        /* 0x14 */ u32 delay_time;
        /* 0x18 */ FixedPoint<18, 14> sample_rate;
        /* 0x1C */ FixedPoint<18, 14> in_gain;
        /* 0x20 */ FixedPoint<18, 14> feedback_gain;
        /* 0x24 */ FixedPoint<18, 14> wet_gain;
        /* 0x28 */ FixedPoint<18, 14> dry_gain;
        /* 0x2C */ FixedPoint<18, 14> channel_spread;
        /* 0x30 */ FixedPoint<18, 14> lowpass_amount;
        /* 0x34 */ ParameterState state;
    };
    static_assert(sizeof(ParameterVersion1) <= sizeof(EffectInfoBase::InParameterVersion1),
                  "DelayInfo::ParameterVersion1 has the wrong size!");

    struct ParameterVersion2 {
        /* 0x00 */ std::array<s8, MaxChannels> inputs;
        /* 0x06 */ std::array<s8, MaxChannels> outputs;
        /* 0x0C */ s16 channel_count_max;
        /* 0x0E */ s16 channel_count;
        /* 0x10 */ s32 delay_time_max;
        /* 0x14 */ s32 delay_time;
        /* 0x18 */ s32 sample_rate;
        /* 0x1C */ s32 in_gain;
        /* 0x20 */ s32 feedback_gain;
        /* 0x24 */ s32 wet_gain;
        /* 0x28 */ s32 dry_gain;
        /* 0x2C */ s32 channel_spread;
        /* 0x30 */ s32 lowpass_amount;
        /* 0x34 */ ParameterState state;
    };
    static_assert(sizeof(ParameterVersion2) <= sizeof(EffectInfoBase::InParameterVersion2),
                  "DelayInfo::ParameterVersion2 has the wrong size!");

    struct DelayLine {
        FixedPoint<50, 14> read() const {
            return buffer[buffer_pos];
        }

        void write(const FixedPoint<50, 14> value) {
            buffer[buffer_pos] = value;
            buffer_pos = static_cast<u32>((buffer_pos + 1) % buffer.size());
        }

        s32 sample_count_max{};
        s32 sample_count{};
        std::vector<FixedPoint<50, 14>> buffer{};
        u32 buffer_pos{};
        FixedPoint<18, 14> decay_rate{};
    };

    struct State {
        /* 0x000 */ std::array<s32, 8> unk_000;
        /* 0x020 */ std::array<DelayLine, MaxChannels> delay_lines;
        /* 0x0B0 */ FixedPoint<18, 14> feedback_gain;
        /* 0x0B4 */ FixedPoint<18, 14> delay_feedback_gain;
        /* 0x0B8 */ FixedPoint<18, 14> delay_feedback_cross_gain;
        /* 0x0BC */ FixedPoint<18, 14> lowpass_gain;
        /* 0x0C0 */ FixedPoint<18, 14> lowpass_feedback_gain;
        /* 0x0C4 */ std::array<FixedPoint<50, 14>, MaxChannels> lowpass_z;
    };
    static_assert(sizeof(State) <= sizeof(EffectInfoBase::State),
                  "DelayInfo::State has the wrong size!");

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
