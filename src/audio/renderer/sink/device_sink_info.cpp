// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/sink/device_sink_info.h"
#include "renderer/upsampler/upsampler_manager.h"

namespace armigo::audio::renderer {

DeviceSinkInfo::DeviceSinkInfo() {
    state.fill(0);
    parameter.fill(0);
    type = Type::DeviceSink;
}

void DeviceSinkInfo::clean_up() {
    auto state_{reinterpret_cast<DeviceState*>(state.data())};

    if (state_->upsampler_info) {
        state_->upsampler_info->manager->free(state_->upsampler_info);
        state_->upsampler_info = nullptr;
    }

    parameter.fill(0);
    type = Type::Invalid;
}

void DeviceSinkInfo::update(BehaviorInfo::ErrorInfo& error_info, OutStatus& out_status,
                            const InParameter& in_params,
                            [[maybe_unused]] const PoolMapper& pool_mapper) {

    const auto device_params{reinterpret_cast<const DeviceInParameter*>(&in_params.device)};
    auto current_params{reinterpret_cast<DeviceInParameter*>(parameter.data())};

    if (in_use == in_params.in_use) {
        current_params->downmix_enabled = device_params->downmix_enabled;
        current_params->downmix_coeff = device_params->downmix_coeff;
    } else {
        type = in_params.type;
        in_use = in_params.in_use;
        node_id = in_params.node_id;
        *current_params = *device_params;
    }

    auto current_state{reinterpret_cast<DeviceState*>(state.data())};

    for (size_t i = 0; i < current_state->downmix_coeff.size(); i++) {
        current_state->downmix_coeff[i] = current_params->downmix_coeff[i];
    }

    std::memset(&out_status, 0, sizeof(OutStatus));
    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void DeviceSinkInfo::update_for_command_generation() {}

} // namespace armigo::audio::renderer
