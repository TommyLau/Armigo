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

#include "renderer/common/common.h"
#include "renderer/effect/effect_info_base.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

class BufferMixerInfo : public EffectInfoBase {
public:
    struct ParameterVersion1 {
        /* 0x00 */ std::array<s8, MaxMixBuffers> inputs;
        /* 0x18 */ std::array<s8, MaxMixBuffers> outputs;
        /* 0x30 */ std::array<f32, MaxMixBuffers> volumes;
        /* 0x90 */ u32 mix_count;
    };
    static_assert(sizeof(ParameterVersion1) <= sizeof(EffectInfoBase::InParameterVersion1),
                  "BufferMixerInfo::ParameterVersion1 has the wrong size!");

    struct ParameterVersion2 {
        /* 0x00 */ std::array<s8, MaxMixBuffers> inputs;
        /* 0x18 */ std::array<s8, MaxMixBuffers> outputs;
        /* 0x30 */ std::array<f32, MaxMixBuffers> volumes;
        /* 0x90 */ u32 mix_count;
    };
    static_assert(sizeof(ParameterVersion2) <= sizeof(EffectInfoBase::InParameterVersion2),
                  "BufferMixerInfo::ParameterVersion2 has the wrong size!");

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
};

} // namespace armigo::audio::renderer
