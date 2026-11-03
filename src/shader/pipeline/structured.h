// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "resources.h"

namespace armigo::shader::pipeline {
struct SequenceRegion { std::vector<RegionId> children; };
struct BlockRegion { BlockId block; }; // Emit operations only; AST owns control flow.
struct IfRegion { Operand condition; RegionId when_true, when_false; };
struct LoopRegion { RegionId body, continuing; };
struct BreakRegion { RegionId loop; };
struct ContinueRegion { RegionId loop; };
struct ReturnRegion { std::vector<Operand> values; };
using Region = std::variant<SequenceRegion, BlockRegion, IfRegion, LoopRegion,
                            BreakRegion, ContinueRegion, ReturnRegion>;
struct StructuredFunction { FunctionId function; RegionId root; };
struct StructuredProgram {
    // Phi destruction has made ValueIds mutable locals. Parallel copies are lowered
    // with temporaries in dedicated edge blocks, splitting critical edges first.
    // Mutable destinations are marked ValueKind::mutable_local; phis are empty.
    // BlockRegion emits each owned block exactly once; retained CFG terminators
    // are provenance only. Structured control flow is exclusively in regions.
    Program ir;
    ResourceRequirements requirements;
    MetalProfile profile;
    std::vector<Region> regions;
    std::vector<StructuredFunction> functions;
};
// Checks region ownership, lexical temporary scope, loop targets and post-phi
// payloads. Mutable reaching definitions are established by the SSA input phase.
[[nodiscard]] Status validate(const StructuredProgram&);
} // namespace armigo::shader::pipeline
