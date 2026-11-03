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

#include <span>

#include "renderer/effect/effect_info_base.h"
#include "renderer/effect/effect_result_state.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

class EffectContext {
public:
    // Initialize the effect context
    void initialize(std::span<EffectInfoBase> effect_infos_, u32 effect_count_,
                    std::span<EffectResultState> result_states_cpu_,
                    std::span<EffectResultState> result_states_dsp_, size_t dsp_state_count);

    // Get the EffectInfo for a given index
    EffectInfoBase& get_info(const u32 index);

    // Get the CPU result state for a given index
    EffectResultState& get_result_state(const u32 index);

    // Get the DSP result state for a given index
    EffectResultState& get_dsp_shared_result_state(const u32 index);

    // Get the number of effects in this context
    u32 get_count() const;

    // Update the CPU and DSP result states for all effects
    void update_state_by_dsp_shared();

private:
    // Workbuffer for all of the effects
    std::span<EffectInfoBase> effect_infos{};
    // Number of effects in the workbuffer
    u32 effect_count{};
    // Workbuffer of states for all effects, kept host-side and not directly modified, dsp states
    // are copied here on the next render frame
    std::span<EffectResultState> result_states_cpu{};
    // Workbuffer of states for all effects, used by the AudioRenderer to track effect state
    // between calls
    std::span<EffectResultState> result_states_dsp{};
    // Number of result states in the workbuffers
    size_t dsp_state_count{};
};

} // namespace armigo::audio::renderer
