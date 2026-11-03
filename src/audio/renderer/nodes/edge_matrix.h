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

#include "renderer/nodes/bit_array.h"
#include "renderer/common/support.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// An edge matrix, holding the connections for each node to every other node in the graph.
class EdgeMatrix {
public:
    // Calculate the size required for its workbuffer.
    static u64 get_work_buffer_size(u32 count) {
        return align_up(count * count, 0x40) / sizeof(u64);
    }

    // Initialize this edge matrix.
    void initialize(std::span<u8> buffer, u64 node_buffer_size, u32 count);

    // Check if a node is connected to another.
    bool connected(u32 id, u32 destination_id) const;

    // Connect a node to another.
    void connect(u32 id, u32 destination_id);

    // Disconnect a node from another.
    void disconnect(u32 id, u32 destination_id);

    // Remove all connections for a given node.
    void remove_edges(u32 id);

    // Get the number of nodes in the graph.
    u32 get_node_count() const;

private:
    // Edges for the current graph
    BitArray edges;
    // Number of nodes (not edges) in the graph
    u32 count;
};

} // namespace armigo::audio::renderer
