// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/buffer_mixer.h"

namespace armigo::audio::renderer {

void BufferMixerInfo::update(BehaviorInfo::ErrorInfo& error_info,
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

void BufferMixerInfo::update(BehaviorInfo::ErrorInfo& error_info,
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

void BufferMixerInfo::update_for_command_generation() {
    if (enabled) {
        usage_state = UsageState::Enabled;
    } else {
        usage_state = UsageState::Disabled;
    }
}

void BufferMixerInfo::initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

void BufferMixerInfo::update_result_state([[maybe_unused]] EffectResultState& cpu_state,
                                        [[maybe_unused]] EffectResultState& dsp_state) {}

} // namespace armigo::audio::renderer
