// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/delay.h"

namespace armigo::audio::renderer {

void DelayInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion1& in_params,
                       const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};

    if (is_channel_count_valid(in_specific->channel_count_max)) {
        const auto old_state{params->state};
        std::memcpy(params, in_specific, sizeof(ParameterVersion1));
        mix_id = in_params.mix_id;
        process_order = in_params.process_order;
        enabled = in_params.enabled;

        if (!is_channel_count_valid(in_specific->channel_count)) {
            params->channel_count = params->channel_count_max;
        }

        if (!is_channel_count_valid(in_specific->channel_count) ||
            old_state != ParameterState::Updated) {
            params->state = old_state;
        }

        if (buffer_unmapped || in_params.is_new) {
            usage_state = UsageState::New;
            params->state = ParameterState::Initialized;
            buffer_unmapped = !pool_mapper.try_attach_buffer(
                error_info, workbuffers[0], in_params.workbuffer, in_params.workbuffer_size);
            return;
        }
    }
    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void DelayInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion2& in_params,
                       const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};

    if (is_channel_count_valid(in_specific->channel_count_max)) {
        const auto old_state{params->state};
        std::memcpy(params, in_specific, sizeof(ParameterVersion1));
        mix_id = in_params.mix_id;
        process_order = in_params.process_order;
        enabled = in_params.enabled;

        if (!is_channel_count_valid(in_specific->channel_count)) {
            params->channel_count = params->channel_count_max;
        }

        if (!is_channel_count_valid(in_specific->channel_count) ||
            old_state != ParameterState::Updated) {
            params->state = old_state;
        }

        if (buffer_unmapped || in_params.is_new) {
            usage_state = UsageState::New;
            params->state = ParameterState::Initialized;
            buffer_unmapped = !pool_mapper.try_attach_buffer(
                error_info, workbuffers[0], in_params.workbuffer, in_params.workbuffer_size);
            return;
        }
    }
    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void DelayInfo::update_for_command_generation() {
    if (enabled) {
        usage_state = UsageState::Enabled;
    } else {
        usage_state = UsageState::Disabled;
    }

    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};
    params->state = ParameterState::Updated;
}

void DelayInfo::initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

void DelayInfo::update_result_state([[maybe_unused]] EffectResultState& cpu_state, [[maybe_unused]] EffectResultState& dsp_state) {}

CpuAddr DelayInfo::get_workbuffer(s32 index) {
    return get_single_buffer(index);
}

} // namespace armigo::audio::renderer
