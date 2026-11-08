// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_memory.h"
#include <algorithm>

namespace armigo::shader_native {
std::vector<StageComponent> output_components(std::span<const std::byte> header) {
    std::vector<StageComponent> out;
    const auto byte = [&](std::size_t at) { return std::to_integer<unsigned>(header[at]); };
    if ((byte(0x35) >> 3) & 1) out.push_back({StageComponent::Kind::point_size, 0, 0});
    for (std::uint8_t c = 0; c < 4; ++c)
        if ((byte(0x35) >> (4 + c)) & 1) out.push_back({StageComponent::Kind::position, 0, c});
    for (std::uint8_t n = 0; n < 32; ++n)
        for (std::uint8_t c = 0; c < 4; ++c)
            if ((byte(0x36 + n / 2) >> (4 * (n % 2) + c)) & 1) out.push_back({StageComponent::Kind::generic, n, c});
    return out;
}

std::variant<StageMemory, std::string> stage_memory(const Program& program) {
    if ((program.stage != Stage::task && program.stage != Stage::mesh) || !program.mesh_launch || program.header.size() != 128)
        return std::string("stage buffer requires a task or mesh program");
    const auto& launch = *program.mesh_launch;
    if (launch.payload_bytes > metal_payload_bytes) return std::string("task payload exceeds a Metal object payload");
    StageMemory m;
    // The hardware writes a mesh workgroup's number at input word 0, so a
    // mesh stage under a task stage has an input word whatever its payload.
    const auto input_bytes = program.stage == Stage::mesh && launch.task ? std::max<std::uint32_t>(launch.payload_bytes, 4) : launch.payload_bytes;
    m.output = m.input + ((input_bytes + 15) & ~15u);
    if (program.stage == Stage::task) {
        // Word 1 is the mesh launch count; a payload shorter than 8 bytes
        // does not hold it and launches none (mesh_stage.cpp).
        m.bytes = m.output + std::max<std::uint32_t>((launch.payload_bytes + 3) & ~3u, 8);
    } else {
        if (launch.max_vertices > metal_mesh_vertices || launch.max_primitives > metal_mesh_primitives)
            return std::string("mesh vertices or primitives exceed a Metal mesh");
        m.vertex = output_components(program.header);
        if (launch.primitive_header)
            for (const auto& component : output_components(program.image.subspan(*launch.primitive_header, 128)))
                if (component.kind == StageComponent::Kind::generic) m.primitive.push_back(component);
        const auto blocks = [](std::uint32_t count) { return (std::max<std::uint32_t>(count, 1) + stage_block - 1) / stage_block; };
        const auto vertex_blocks = blocks(launch.max_vertices), primitive_blocks = blocks(launch.max_primitives);
        const auto rows = std::max<std::uint32_t>(vertex_blocks * static_cast<std::uint32_t>(m.vertex.size()) +
                                                  primitive_blocks * static_cast<std::uint32_t>(m.primitive.size()), 1);
        m.primitives = m.output + rows * stage_block * 4;
        // A count of zero wraps to every lane of every primitive block, so
        // whole blocks of index bytes are reserved.
        m.bytes = m.primitives + ((4 + launch.primitive_vertices * primitive_blocks * stage_block + 3) & ~3u);
    }
    if (m.bytes > metal_threadgroup_bytes) return std::string("stage buffer exceeds Metal workgroup memory");
    return m;
}
} // namespace armigo::shader_native
