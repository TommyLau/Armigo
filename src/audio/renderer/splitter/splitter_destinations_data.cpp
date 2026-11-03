// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/splitter/splitter_destinations_data.h"

namespace armigo::audio::renderer {

SplitterDestinationData::SplitterDestinationData(const s32 id_) : id{id_} {}

void SplitterDestinationData::clear_mix_volume() {
    mix_volumes.fill(0.0f);
    prev_mix_volumes.fill(0.0f);
}

s32 SplitterDestinationData::get_id() const {
    return id;
}

bool SplitterDestinationData::is_configured() const {
    return in_use && destination_id != UnusedMixId;
}

s32 SplitterDestinationData::get_mix_id() const {
    return destination_id;
}

f32 SplitterDestinationData::get_mix_volume(const u32 index) const {
    if (index >= mix_volumes.size()) {
        AUDIO_LOG_ERROR("SplitterDestinationData::GetMixVolume Invalid index {}", index);
        return 0.0f;
    }
    return mix_volumes[index];
}

std::span<f32> SplitterDestinationData::get_mix_volume() {
    return mix_volumes;
}

f32 SplitterDestinationData::get_mix_volume_prev(const u32 index) const {
    if (index >= prev_mix_volumes.size()) {
        AUDIO_LOG_ERROR("SplitterDestinationData::GetMixVolumePrev Invalid index {}",
                  index);
        return 0.0f;
    }
    return prev_mix_volumes[index];
}

std::span<f32> SplitterDestinationData::get_mix_volume_prev() {
    return prev_mix_volumes;
}

void SplitterDestinationData::update(const InParameter& params) {
    if (params.id != id || params.magic != get_splitter_send_data_magic()) {
        return;
    }

    destination_id = params.mix_id;
    mix_volumes = params.mix_volumes;

    if (!in_use && params.in_use) {
        prev_mix_volumes = mix_volumes;
        need_update = false;
    }

    in_use = params.in_use;
}

void SplitterDestinationData::mark_as_need_to_update_internal_state() {
    need_update = true;
}

void SplitterDestinationData::update_internal_state() {
    if (in_use && need_update) {
        prev_mix_volumes = mix_volumes;
    }
    need_update = false;
}

SplitterDestinationData* SplitterDestinationData::get_next() const {
    return next;
}

void SplitterDestinationData::set_next(SplitterDestinationData* next_) {
    next = next_;
}

std::span<SplitterDestinationData::BiquadFilterParameter2>
SplitterDestinationData::get_biquad_filters() {
    return biquad_filters;
}

std::span<const SplitterDestinationData::BiquadFilterParameter2>
SplitterDestinationData::get_biquad_filters() const {
    return biquad_filters;
}

} // namespace armigo::audio::renderer
