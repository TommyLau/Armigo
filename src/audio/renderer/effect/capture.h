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

class CaptureInfo : public EffectInfoBase {
public:
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
