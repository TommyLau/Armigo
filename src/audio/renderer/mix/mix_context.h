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

#include "renderer/mix/mix_info.h"
#include "renderer/nodes/edge_matrix.h"
#include "renderer/nodes/node_states.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class SplitterContext;

/*
 * Manages mixing states, sorting and building a node graph to describe a mix order.
 */
class MixContext {
public:
    // Initialize the mix context.
    void initialize(std::span<MixInfo*> sorted_mix_infos, std::span<MixInfo> mix_infos, u32 count_,
                    std::span<s32> effect_process_order_buffer, u32 effect_count,
                    std::span<u8> node_states_workbuffer, u64 node_buffer_size,
                    std::span<u8> edge_matrix_workbuffer, u64 edge_matrix_size);

    // Get a sorted mix at the given index.
    MixInfo* get_sorted_info(s32 index);

    // Set the sorted info at the given index.
    void set_sorted_info(s32 index, MixInfo& mix_info);

    // Get a mix at the given index.
    MixInfo* get_info(s32 index);

    // Get the final mix.
    MixInfo* get_final_mix_info();

    // Get the current number of mixes.
    s32 get_count() const;

    // Update all of the mixes' distance from the final mix.
    // Needs to be called after altering the mix graph.
    void update_distances_from_final_mix();

    // Non-splitter sort, sorts the sorted mixes based on their distance from the final mix.
    void sort_info();

    // Re-calculate the mix buffer offsets for each mix after altering the mix.
    void calc_mix_buffer_offset();

    // Splitter sort, traverse the splitter node graph and sort the sorted mixes from results.
    bool t_sort_info(const SplitterContext& splitter_context);

    // Get the edge matrix used for the mix graph.
    EdgeMatrix& get_edge_matrix();

private:
    // Array of sorted mixes
    std::span<MixInfo*> sorted_mix_infos{};
    // Array of mixes
    std::span<MixInfo> mix_infos{};
    // Number of active mixes
    s32 count{};
    // Array of effect process orderings
    std::span<s32> effect_process_order_buffer{};
    // Number of effects in the process ordering buffer
    u64 effect_count{};
    // Node states used in splitter sort
    NodeStates node_states{};
    // Edge matrix for connected nodes used in splitter sort
    EdgeMatrix edge_matrix{};
};

} // namespace armigo::audio::renderer
