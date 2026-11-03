// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/aux_.h"

namespace armigo::audio::renderer {

void AuxInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion1& in_params,
                     const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(ParameterVersion1));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;
    if (buffer_unmapped || in_params.is_new) {
        const bool send_unmapped{!pool_mapper.try_attach_buffer(
            error_info, workbuffers[0], in_specific->send_buffer_info_address,
            sizeof(AuxBufferInfo) + in_specific->count_max * sizeof(s32))};
        const bool return_unmapped{!pool_mapper.try_attach_buffer(
            error_info, workbuffers[1], in_specific->return_buffer_info_address,
            sizeof(AuxBufferInfo) + in_specific->count_max * sizeof(s32))};

        buffer_unmapped = send_unmapped || return_unmapped;

        if (!buffer_unmapped) {
            auto send{workbuffers[0].get_reference(false)};
            send_buffer_info = send + sizeof(AuxInfoDsp);
            send_buffer = send + sizeof(AuxBufferInfo);

            auto ret{workbuffers[1].get_reference(false)};
            return_buffer_info = ret + sizeof(AuxInfoDsp);
            return_buffer = ret + sizeof(AuxBufferInfo);
        }
    } else {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }
}

void AuxInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion2& in_params,
                     const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion2*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion2*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(ParameterVersion2));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;

    if (buffer_unmapped || in_params.is_new) {
        const bool send_unmapped{!pool_mapper.try_attach_buffer(
            error_info, workbuffers[0], params->send_buffer_info_address,
            sizeof(AuxBufferInfo) + params->count_max * sizeof(s32))};
        const bool return_unmapped{!pool_mapper.try_attach_buffer(
            error_info, workbuffers[1], params->return_buffer_info_address,
            sizeof(AuxBufferInfo) + params->count_max * sizeof(s32))};

        buffer_unmapped = send_unmapped || return_unmapped;

        if (!buffer_unmapped) {
            auto send{workbuffers[0].get_reference(false)};
            send_buffer_info = send + sizeof(AuxInfoDsp);
            send_buffer = send + sizeof(AuxBufferInfo);

            auto ret{workbuffers[1].get_reference(false)};
            return_buffer_info = ret + sizeof(AuxInfoDsp);
            return_buffer = ret + sizeof(AuxBufferInfo);
        }
    } else {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }
}

void AuxInfo::update_for_command_generation() {
    if (enabled) {
        usage_state = UsageState::Enabled;
    } else {
        usage_state = UsageState::Disabled;
    }
}

void AuxInfo::initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

void AuxInfo::update_result_state([[maybe_unused]] EffectResultState& cpu_state, [[maybe_unused]] EffectResultState& dsp_state) {}

CpuAddr AuxInfo::get_workbuffer(s32 index) {
    return workbuffers[index].get_reference(true);
}

} // namespace armigo::audio::renderer
