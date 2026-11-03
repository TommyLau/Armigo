// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <ranges>

#include "renderer/mix/mix_context.h"
#include "renderer/splitter/splitter_context.h"
#include <ranges>

namespace armigo::audio::renderer {

void MixContext::initialize(std::span<MixInfo*> sorted_mix_infos_, std::span<MixInfo> mix_infos_,
                            const u32 count_, std::span<s32> effect_process_order_buffer_,
                            const u32 effect_count_, std::span<u8> node_states_workbuffer,
                            const u64 node_buffer_size, std::span<u8> edge_matrix_workbuffer,
                            const u64 edge_matrix_size) {
    count = count_;
    sorted_mix_infos = sorted_mix_infos_;
    mix_infos = mix_infos_;
    effect_process_order_buffer = effect_process_order_buffer_;
    effect_count = effect_count_;

    if (node_states_workbuffer.size() > 0 && edge_matrix_workbuffer.size() > 0) {
        node_states.initialize(node_states_workbuffer, node_buffer_size, count);
        edge_matrix.initialize(edge_matrix_workbuffer, edge_matrix_size, count);
    }

    for (s32 i = 0; i < count; i++) {
        sorted_mix_infos[i] = &mix_infos[i];
    }
}

MixInfo* MixContext::get_sorted_info(const s32 index) {
    return sorted_mix_infos[index];
}

void MixContext::set_sorted_info(const s32 index, MixInfo& mix_info) {
    sorted_mix_infos[index] = &mix_info;
}

MixInfo* MixContext::get_info(const s32 index) {
    return &mix_infos[index];
}

MixInfo* MixContext::get_final_mix_info() {
    return &mix_infos[0];
}

s32 MixContext::get_count() const {
    return count;
}

void MixContext::update_distances_from_final_mix() {
    for (s32 i = 0; i < count; i++) {
        mix_infos[i].distance_from_final_mix = InvalidDistanceFromFinalMix;
    }

    for (s32 i = 0; i < count; i++) {
        auto& mix_info{mix_infos[i]};
        sorted_mix_infos[i] = &mix_info;

        if (!mix_info.in_use) {
            continue;
        }

        auto mix_id{mix_info.mix_id};
        auto distance_to_final_mix{FinalMixId};

        while (distance_to_final_mix < count) {
            if (mix_id == FinalMixId) {
                break;
            }

            if (mix_id == UnusedMixId) {
                distance_to_final_mix = InvalidDistanceFromFinalMix;
                break;
            }

            auto distance_from_final_mix{mix_infos[mix_id].distance_from_final_mix};
            if (distance_from_final_mix != InvalidDistanceFromFinalMix) {
                distance_to_final_mix = distance_from_final_mix + 1;
                break;
            }

            distance_to_final_mix++;
            mix_id = mix_infos[mix_id].dst_mix_id;
        }

        if (distance_to_final_mix >= count) {
            distance_to_final_mix = InvalidDistanceFromFinalMix;
        }
        mix_info.distance_from_final_mix = distance_to_final_mix;
    }
}

void MixContext::sort_info() {
    update_distances_from_final_mix();

    std::ranges::sort(sorted_mix_infos, [](const MixInfo* lhs, const MixInfo* rhs) {
        return lhs->distance_from_final_mix > rhs->distance_from_final_mix;
    });

    calc_mix_buffer_offset();
}

void MixContext::calc_mix_buffer_offset() {
    s16 offset{0};
    for (s32 i = 0; i < count; i++) {
        auto mix_info{sorted_mix_infos[i]};
        if (mix_info->in_use) {
            const auto buffer_count{mix_info->buffer_count};
            mix_info->buffer_offset = offset;
            offset += buffer_count;
        }
    }
}

bool MixContext::t_sort_info(const SplitterContext& splitter_context) {
    if (!splitter_context.using_splitter()) {
        calc_mix_buffer_offset();
        return true;
    }

    if (!node_states.tsort(edge_matrix)) {
        return false;
    }

    auto sorted_results{node_states.get_sorted_resuls()};
    const auto result_size{(std::min)(count, static_cast<s32>(sorted_results.second))};
    for (s32 i = 0; i < result_size; i++) {
        sorted_mix_infos[i] = &mix_infos[sorted_results.first[i]];
    }

    calc_mix_buffer_offset();
    return true;
}

EdgeMatrix& MixContext::get_edge_matrix() {
    return edge_matrix;
}

} // namespace armigo::audio::renderer
