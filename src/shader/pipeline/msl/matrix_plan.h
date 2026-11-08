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
#include "../structured.h"
#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace armigo::shader::pipeline {
// Where each warp matrix operation of a compute stage keeps its operands on
// the SIMD-group matrix hardware. A matrix stands for words only when it is
// made by the one operation that assigns those words (an LDSM's tiles, a
// multiply's D), so it holds their value wherever they are read, or when an
// earlier operation of the same block, which the emitter writes in one scope,
// made it from the same single-assignment words.
struct MatrixPlan {
    // An 8x8 binary16 matrix: quadrant `index` of the A or B matrices
    // multiply `op` placed through the scratch block (own), block `index` of
    // the tiles matrix load `op` read from shared memory (load), or half
    // `index` of multiply `op`'s D (product), which is an A quadrant as it
    // stands.
    struct Tile {
        enum class Source : std::uint8_t { own, load, product } source{Source::own};
        OperationId op;
        std::uint8_t index{};
    };
    struct Multiply {
        // A's quadrants by A word: (row 0, k 0), (row 8, k 0), (row 0, k 8),
        // (row 8, k 8). They are this operation's own scratch quadrants
        // unless every A word is a word a matrix load or a multiply returned
        // or the A words are those of an earlier multiply of the block.
        std::array<Tile, 4> a{};
        // B's halves by B word: k 0 and k 8, from matrix loads, an earlier
        // multiply of the block or this operation's scratch.
        std::array<Tile, 2> b{};
        // C is the lanes' words, literal zero, or the D matrices of c_from
        // (a chain).
        enum class Accumulator : std::uint8_t { lanes, zero, chain } c{Accumulator::lanes};
        OperationId c_from;
        // Whether D's words are read back into the lanes: false when only
        // multiplies read them, as matrices.
        bool lanes{true};
        // Whether another multiply takes D as matrices (D is then declared
        // for the whole function, as its reader may be in another block).
        bool kept{};
    };
    struct Load {
        // Blocks read as A quadrants and as B halves (each declared for the
        // whole function), and whether the lanes still need their words.
        std::uint8_t a_tiles{}, b_tiles{};
        bool lanes{true};
    };
    std::unordered_map<std::uint32_t, Multiply> multiplies;
    std::unordered_map<std::uint32_t, Load> loads;
};

// Plans every matrix_multiply_add and matrix_load of the program, calling
// `step` once for each unit of work so the emitter's analysis budget bounds it
// (`step` throws when the budget is spent).
[[nodiscard]] MatrixPlan plan_matrices(const StructuredProgram&, const std::function<void()>& step);
} // namespace armigo::shader::pipeline
