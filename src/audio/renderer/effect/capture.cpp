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
#include "renderer/effect/capture.h"

namespace armigo::audio::renderer {

void CaptureInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion1& in_params,
                         const PoolMapper& pool_mapper) {
    auto in_specific{
        reinterpret_cast<const AuxInfo::ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<AuxInfo::ParameterVersion1*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(AuxInfo::ParameterVersion1));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;
    if (buffer_unmapped || in_params.is_new) {
        buffer_unmapped = !pool_mapper.try_attach_buffer(
            error_info, workbuffers[0], in_specific->send_buffer_info_address,
            in_specific->count_max * sizeof(s32) + sizeof(AuxInfo::AuxBufferInfo));

        if (!buffer_unmapped) {
            const auto send_address{workbuffers[0].get_reference(false)};
            send_buffer_info = send_address + sizeof(AuxInfo::AuxInfoDsp);
            send_buffer = send_address + sizeof(AuxInfo::AuxBufferInfo);
            return_buffer_info = 0;
            return_buffer = 0;
        }
    } else {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }
}

void CaptureInfo::update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion2& in_params,
                         const PoolMapper& pool_mapper) {
    auto in_specific{
        reinterpret_cast<const AuxInfo::ParameterVersion2*>(in_params.specific.data())};
    auto params{reinterpret_cast<AuxInfo::ParameterVersion2*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(AuxInfo::ParameterVersion2));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;

    if (buffer_unmapped || in_params.is_new) {
        buffer_unmapped = !pool_mapper.try_attach_buffer(
            error_info, workbuffers[0], params->send_buffer_info_address,
            params->count_max * sizeof(s32) + sizeof(AuxInfo::AuxBufferInfo));

        if (!buffer_unmapped) {
            const auto send_address{workbuffers[0].get_reference(false)};
            send_buffer_info = send_address + sizeof(AuxInfo::AuxInfoDsp);
            send_buffer = send_address + sizeof(AuxInfo::AuxBufferInfo);
            return_buffer_info = 0;
            return_buffer = 0;
        }
    } else {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }
}

void CaptureInfo::update_for_command_generation() {
    if (enabled) {
        usage_state = UsageState::Enabled;
    } else {
        usage_state = UsageState::Disabled;
    }
}

void CaptureInfo::initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

void CaptureInfo::update_result_state([[maybe_unused]] EffectResultState& cpu_state, [[maybe_unused]] EffectResultState& dsp_state) {}

CpuAddr CaptureInfo::get_workbuffer(s32 index) {
    return workbuffers[index].get_reference(true);
}

} // namespace armigo::audio::renderer
