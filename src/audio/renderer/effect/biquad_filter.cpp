// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/biquad_filter.h"

namespace armigo::audio::renderer {

void BiquadFilterInfo::update(BehaviorInfo::ErrorInfo& error_info,
                              const InParameterVersion1& in_params, [[maybe_unused]] const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(ParameterVersion1));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;

    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void BiquadFilterInfo::update(BehaviorInfo::ErrorInfo& error_info,
                              const InParameterVersion2& in_params, [[maybe_unused]] const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion2*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion2*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(ParameterVersion2));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;

    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void BiquadFilterInfo::update_for_command_generation() {
    usage_state = enabled ? UsageState::Enabled : UsageState::Disabled;

    auto* params_v1 = reinterpret_cast<ParameterVersion1*>(parameter.data());
    auto* params_v2 = reinterpret_cast<ParameterVersion2*>(parameter.data());

    const auto raw_state_v1 = static_cast<u8>(params_v1->state);
    const auto raw_state_v2 = static_cast<u8>(params_v2->state);

    if (raw_state_v1 <= static_cast<u8>(ParameterState::Updated)) {
        params_v1->state = ParameterState::Updated;
    } else if (raw_state_v2 <= static_cast<u8>(ParameterState::Updated)) {
        params_v2->state = ParameterState::Updated;
    } else {
        params_v1->state = ParameterState::Updated;
        params_v2->state = ParameterState::Updated;
    }
}

void BiquadFilterInfo::initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

void BiquadFilterInfo::update_result_state([[maybe_unused]] EffectResultState& cpu_state,
                                         [[maybe_unused]] EffectResultState& dsp_state) {}

} // namespace armigo::audio::renderer
