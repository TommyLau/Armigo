// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/effect/effect_context.h"

namespace armigo::audio::renderer {

void EffectContext::initialize(std::span<EffectInfoBase> effect_infos_, const u32 effect_count_,
                               std::span<EffectResultState> result_states_cpu_,
                               std::span<EffectResultState> result_states_dsp_,
                               const size_t dsp_state_count_) {
    effect_infos = effect_infos_;
    effect_count = effect_count_;
    result_states_cpu = result_states_cpu_;
    result_states_dsp = result_states_dsp_;
    dsp_state_count = dsp_state_count_;
}

EffectInfoBase& EffectContext::get_info(const u32 index) {
    return effect_infos[index];
}

EffectResultState& EffectContext::get_result_state(const u32 index) {
    return result_states_cpu[index];
}

EffectResultState& EffectContext::get_dsp_shared_result_state(const u32 index) {
    return result_states_dsp[index];
}

u32 EffectContext::get_count() const {
    return effect_count;
}

void EffectContext::update_state_by_dsp_shared() {
    for (size_t i = 0; i < dsp_state_count; i++) {
        effect_infos[i].update_result_state(result_states_cpu[i], result_states_dsp[i]);
    }
}

} // namespace armigo::audio::renderer
