// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/splitter/splitter_info.h"

namespace armigo::audio::renderer {

SplitterInfo::SplitterInfo(const s32 id_) : id{id_} {}

void SplitterInfo::initialize_infos(SplitterInfo* splitters, const u32 count) {
    if (splitters == nullptr) {
        return;
    }

    for (u32 i = 0; i < count; i++) {
        auto& splitter{splitters[i]};
        splitter.destinations = nullptr;
        splitter.destination_count = 0;
        splitter.new_connection = true;
    }
}

u32 SplitterInfo::update(const InParameter* params) {
    if (params->id != id) {
        return 0;
    }
    sample_rate = params->sample_rate;
    new_connection = true;
    return static_cast<u32>((sizeof(InParameter) + 3 * sizeof(s32)) +
                            params->destination_count * sizeof(s32));
}

SplitterDestinationData* SplitterInfo::get_data(const u32 destination_id) {
    auto out_destination{destinations};
    u32 i{0};
    while (i < destination_id) {
        if (out_destination == nullptr) {
            break;
        }
        out_destination = out_destination->get_next();
        i++;
    }

    return out_destination;
}

u32 SplitterInfo::get_destination_count() const {
    return destination_count;
}

void SplitterInfo::set_destination_count(const u32 count) {
    destination_count = count;
}

bool SplitterInfo::has_new_connection() const {
    return new_connection;
}

void SplitterInfo::clear_new_connection_flag() {
    new_connection = false;
}

void SplitterInfo::set_new_connection_flag() {
    new_connection = true;
}

void SplitterInfo::update_internal_state() {
    auto destination{destinations};
    while (destination != nullptr) {
        destination->update_internal_state();
        destination = destination->get_next();
    }
}

void SplitterInfo::set_destinations(SplitterDestinationData* destinations_) {
    destinations = destinations_;
}

} // namespace armigo::audio::renderer
