// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/nodes/node_states.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

void NodeStates::initialize(std::span<u8> buffer_, [[maybe_unused]] const u64 node_buffer_size,
                            const u32 count) {
    u64 num_blocks{align_up(count, 0x40) / sizeof(u64)};
    u64 offset{0};

    node_count = count;

    nodes_found.buffer.resize(count);
    nodes_found.size = count;
    nodes_found.reset();

    offset += num_blocks;

    nodes_complete.buffer.resize(count);
    nodes_complete.size = count;
    nodes_complete.reset();

    offset += num_blocks;

    results = {reinterpret_cast<u32*>(&buffer_[offset]), count};

    offset += count * sizeof(u32);

    stack.stack = {reinterpret_cast<u32*>(&buffer_[offset]), count * count};
    stack.size = count * count;
    stack.unk_10 = count * count;

    offset += count * count * sizeof(u32);
}

bool NodeStates::tsort(const EdgeMatrix& edge_matrix) {
    return depth_first_search(edge_matrix, stack);
}

bool NodeStates::depth_first_search(const EdgeMatrix& edge_matrix, Stack& stack_) {
    reset_state();

    for (u32 node_id = 0; node_id < node_count; node_id++) {
        if (get_state(node_id) == SearchState::Unknown) {
            stack_.push(node_id);
        }

        while (stack_.count() > 0) {
            auto current_node{stack_.top()};
            switch (get_state(current_node)) {
            case SearchState::Unknown:
                set_state(current_node, SearchState::Found);
                break;
            case SearchState::Found:
                set_state(current_node, SearchState::Complete);
                push_tsort_result(current_node);
                stack_.pop();
                continue;
            case SearchState::Complete:
                stack_.pop();
                continue;
            }

            const auto edge_count{edge_matrix.get_node_count()};
            for (u32 edge_id = 0; edge_id < edge_count; edge_id++) {
                if (!edge_matrix.connected(current_node, edge_id)) {
                    continue;
                }

                switch (get_state(edge_id)) {
                case SearchState::Unknown:
                    stack_.push(edge_id);
                    break;
                case SearchState::Found:
                    AUDIO_LOG_ERROR("Cycle detected in the node graph, graph is not a DAG! "
                              "Bailing to avoid an infinite loop");
                    reset_state();
                    return false;
                case SearchState::Complete:
                    break;
                }
            }
        }
    }

    return true;
}

NodeStates::SearchState NodeStates::get_state(const u32 id) const {
    if (nodes_found.buffer[id]) {
        return SearchState::Found;
    } else if (nodes_complete.buffer[id]) {
        return SearchState::Complete;
    }
    return SearchState::Unknown;
}

void NodeStates::push_tsort_result(const u32 id) {
    results[result_pos++] = id;
}

void NodeStates::set_state(const u32 id, const SearchState state) {
    switch (state) {
    case SearchState::Complete:
        nodes_found.buffer[id] = false;
        nodes_complete.buffer[id] = true;
        break;
    case SearchState::Found:
        nodes_found.buffer[id] = true;
        nodes_complete.buffer[id] = false;
        break;
    case SearchState::Unknown:
        nodes_found.buffer[id] = false;
        nodes_complete.buffer[id] = false;
        break;
    default:
        AUDIO_LOG_ERROR("Unknown node SearchState {}", static_cast<u32>(state));
        break;
    }
}

void NodeStates::reset_state() {
    nodes_found.reset();
    nodes_complete.reset();
    std::fill(results.begin(), results.end(), -1);
    result_pos = 0;
}

u32 NodeStates::get_node_count() const {
    return node_count;
}

std::pair<std::span<u32>::reverse_iterator, size_t> NodeStates::get_sorted_resuls() const {
    return {results.rbegin(), result_pos};
}

} // namespace armigo::audio::renderer
