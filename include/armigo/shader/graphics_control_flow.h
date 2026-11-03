// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader/graphics.h>

#include <array>
#include <cstddef>
#include <vector>

namespace armigo::shader::graphics {

// Upper bound on blocks per control-flow graph program.
inline constexpr std::size_t max_blocks = 4096;

// Explicit control-flow program description. Nodes, exports, and blocks are
// validated as a whole; make_graph is the only way to attach block metadata
// to a Program.
struct GraphSpec {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::array<std::byte, 80> sph{};
    std::vector<Node> nodes;
    std::vector<Export> exports;
    std::vector<BasicBlock> blocks;
};

// Validates the specification and returns an immutable graph Program.
// Rejections carry a specific Error code:
//   GraphFormat   block ranges do not cover the nodes exactly once, or a
//                 block, terminator target, phi predecessor, or node
//                 reference is outside the graph;
//   GraphShape    a block is unreachable, a target is not strictly forward,
//                 or the unique return block is missing or not last;
//   GraphTypes    an operand, phi arm, export, or branch condition has the
//                 wrong value type;
//   GraphDataFlow a definition is unavailable on a path that reads it, phi
//                 arms are missing, duplicated, or labelled with a
//                 non-predecessor, or an export does not dominate the
//                 return block.
[[nodiscard]] GraphResult make_graph(const GraphSpec& spec, Limits limits = {});

}  // namespace armigo::shader::graphics
