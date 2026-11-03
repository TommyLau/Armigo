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
#include <vector>

#include "renderer/nodes/edge_matrix.h"
#include "renderer/common/support.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Graph utility functions for sorting and getting results from the DAG.
class NodeStates {
    // State of a node in the depth first search.
    enum class SearchState {
        Unknown,
        Found,
        Complete,
    };

    // Stack used for a depth first search.
    struct Stack {
        // Calculate the workbuffer size required for this stack.
        static u32 calc_buffer_size(u32 count) {
            return count * sizeof(u32);
        }

        // Reset the stack back to default.
        void reset(u32* buffer_, u32 size_) {
            stack = {buffer_, size_};
            size = size_;
            pos = 0;
            unk_10 = size_;
        }

        // Get the current stack position.
        u32 count() const {
            return pos;
        }

        // Push a new node to the stack.
        void push(u32 data) {
            stack[pos++] = data;
        }

        // Pop a node from the stack.
        u32 pop() {
            return stack[--pos];
        }

        // Get the top of the stack without popping.
        u32 top() const {
            return stack[pos - 1];
        }

        // Buffer for the stack
        std::span<u32> stack{};
        // Size of the stack buffer
        u32 size{};
        // Current stack position
        u32 pos{};
        // Unknown
        u32 unk_10{};
    };

public:
    // Calculate the workbuffer size required for the node states.
    static u64 get_work_buffer_size(u32 count) {
        return (align_up(count, 0x40) / sizeof(u64)) * 2 + count * sizeof(BitArray) +
               count * Stack::calc_buffer_size(count);
    }

    // Initialize the node states.
    void initialize(std::span<u8> buffer_, u64 node_buffer_size, u32 count);

    // Sort the graph. Only calls DepthFirstSearch.
    bool tsort(const EdgeMatrix& edge_matrix);

    // Sort the graph via depth first search.
    bool depth_first_search(const EdgeMatrix& edge_matrix, Stack& stack);

    // Get the search state of a given node.
    SearchState get_state(u32 id) const;

    // Push a node id to the results buffer when found in the DFS.
    void push_tsort_result(u32 id);

    // Set the state of a node.
    void set_state(u32 id, SearchState state);

    // Reset the nodes found, complete and the results.
    void reset_state();

    // Get the number of nodes in the graph.
    u32 get_node_count() const;

    // Get the sorted results from the DFS.
    std::pair<std::span<u32>::reverse_iterator, size_t> get_sorted_resuls() const;

private:
    // Number of nodes in the graph
    u32 node_count{};
    // Position in results buffer
    u32 result_pos{};
    // List of nodes found
    BitArray nodes_found{};
    // List of nodes completed
    BitArray nodes_complete{};
    // List of results from the depth first search
    std::span<u32> results{};
    // Stack used during the depth first search
    Stack stack{};
};

} // namespace armigo::audio::renderer
