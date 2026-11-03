// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/compressor.h"

namespace armigo::audio::renderer {

void CompressorInfo::update([[maybe_unused]] BehaviorInfo::ErrorInfo& error_info,
                            [[maybe_unused]] const InParameterVersion1& in_params, [[maybe_unused]] const PoolMapper& pool_mapper) {}

void CompressorInfo::update(BehaviorInfo::ErrorInfo& error_info,
                            const InParameterVersion2& in_params, [[maybe_unused]] const PoolMapper& pool_mapper) {
    auto in_specific{reinterpret_cast<const ParameterVersion1*>(in_params.specific.data())};
    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};

    std::memcpy(params, in_specific, sizeof(ParameterVersion1));
    mix_id = in_params.mix_id;
    process_order = in_params.process_order;
    enabled = in_params.enabled;

    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void CompressorInfo::update_for_command_generation() {
    if (enabled) {
        usage_state = UsageState::Enabled;
    } else {
        usage_state = UsageState::Disabled;
    }

    auto params{reinterpret_cast<ParameterVersion1*>(parameter.data())};
    params->state = ParameterState::Updated;
}

CpuAddr CompressorInfo::get_workbuffer(s32 index) {
    return get_single_buffer(index);
}

} // namespace armigo::audio::renderer
