// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/nodes/edge_matrix.h"

namespace armigo::audio::renderer {

void EdgeMatrix::initialize([[maybe_unused]] std::span<u8> buffer,
                            [[maybe_unused]] const u64 node_buffer_size, const u32 count_) {
    count = count_;
    edges.buffer.resize(count_ * count_);
    edges.size = count_ * count_;
    edges.reset();
}

bool EdgeMatrix::connected(const u32 id, const u32 destination_id) const {
    return edges.buffer[count * id + destination_id];
}

void EdgeMatrix::connect(const u32 id, const u32 destination_id) {
    edges.buffer[count * id + destination_id] = true;
}

void EdgeMatrix::disconnect(const u32 id, const u32 destination_id) {
    edges.buffer[count * id + destination_id] = false;
}

void EdgeMatrix::remove_edges(const u32 id) {
    for (u32 dest = 0; dest < count; dest++) {
        disconnect(id, dest);
    }
}

u32 EdgeMatrix::get_node_count() const {
    return count;
}

} // namespace armigo::audio::renderer
