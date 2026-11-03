// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/behavior/behavior_info.h"
#include "renderer/effect/effect_context.h"
#include "renderer/mix/mix_info.h"
#include "renderer/nodes/edge_matrix.h"
#include "renderer/splitter/splitter_context.h"

namespace armigo::audio::renderer {

MixInfo::MixInfo(std::span<s32> effect_order_buffer_, s32 effect_count_, BehaviorInfo& behavior)
    : effect_order_buffer{effect_order_buffer_}, effect_count{effect_count_},
      long_size_pre_delay_supported{behavior.is_long_size_pre_delay_supported()} {
    clear_effect_processing_order();
}

void MixInfo::cleanup() {
    mix_id = UnusedMixId;
    dst_mix_id = UnusedMixId;
    dst_splitter_id = UnusedSplitterId;
}

void MixInfo::clear_effect_processing_order() {
    for (s32 i = 0; i < effect_count; i++) {
        effect_order_buffer[i] = -1;
    }
}

bool MixInfo::update(EdgeMatrix& edge_matrix, const InParameter& in_params,
                     EffectContext& effect_context, SplitterContext& splitter_context,
                     const BehaviorInfo& behavior) {
    volume = in_params.volume;
    sample_rate = in_params.sample_rate;
    buffer_count = static_cast<s16>(in_params.buffer_count);
    in_use = in_params.in_use;
    mix_id = in_params.mix_id;
    node_id = in_params.node_id;
    mix_volumes = in_params.mix_volumes;

    bool sort_required{false};
    if (behavior.is_splitter_supported()) {
        sort_required = update_connection(edge_matrix, in_params, splitter_context);
    } else {
        if (dst_mix_id != in_params.dest_mix_id) {
            dst_mix_id = in_params.dest_mix_id;
            sort_required = true;
        }
        dst_splitter_id = UnusedSplitterId;
    }

    clear_effect_processing_order();

    // Check all effects, and set their order if they belong to this mix.
    const auto count{effect_context.get_count()};
    for (u32 i = 0; i < count; i++) {
        const auto& info{effect_context.get_info(i)};
        if (mix_id == info.get_mix_id()) {
            const auto processing_order{info.get_processing_order()};
            if (processing_order < 0 || processing_order >= effect_count) {
                break;
            }
            effect_order_buffer[processing_order] = i;
        }
    }

    return sort_required;
}

bool MixInfo::update_connection(EdgeMatrix& edge_matrix, const InParameter& in_params,
                               SplitterContext& splitter_context) {
    auto has_new_connection{false};
    if (dst_splitter_id != UnusedSplitterId) {
        auto& splitter_info{splitter_context.get_info(dst_splitter_id)};
        has_new_connection = splitter_info.has_new_connection();
    }

    // Check if this mix matches the input parameters.
    // If everything is the same, don't bother updating.
    if (dst_mix_id == in_params.dest_mix_id && dst_splitter_id == in_params.dest_splitter_id &&
        !has_new_connection) {
        return false;
    }

    // Reset the mix in the graph, as we're about to update it.
    edge_matrix.remove_edges(mix_id);

    if (in_params.dest_mix_id == UnusedMixId) {
        if (in_params.dest_splitter_id != UnusedSplitterId) {
            // If the splitter is used, connect this mix to each active destination.
            auto& splitter_info{splitter_context.get_info(in_params.dest_splitter_id)};
            auto const destination_count{splitter_info.get_destination_count()};

            for (u32 i = 0; i < destination_count; i++) {
                auto destination{
                    splitter_context.get_destination_data(in_params.dest_splitter_id, i)};

                if (destination) {
                    const auto destination_id{destination->get_mix_id()};
                    if (destination_id != UnusedMixId) {
                        edge_matrix.connect(mix_id, destination_id);
                    }
                }
            }
        }
    } else {
        // If the splitter is not used, only connect this mix to its destination.
        edge_matrix.connect(mix_id, in_params.dest_mix_id);
    }

    dst_mix_id = in_params.dest_mix_id;
    dst_splitter_id = in_params.dest_splitter_id;
    return true;
}

bool MixInfo::has_any_connection() const {
    return dst_mix_id != UnusedMixId || dst_splitter_id != UnusedSplitterId;
}

} // namespace armigo::audio::renderer
