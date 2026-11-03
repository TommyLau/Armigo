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
#include "decoder.h"
#include "structured.h"

#include <string>

namespace armigo::shader::pipeline {
struct EmitOptions {
    std::string entry_point; // Empty selects the stage-specific armigo_*_main entry.
    bool source_locations{};
    bool fragment_output_types_from_render_targets{};
    FragmentPositionConvention fragment_position{FragmentPositionConvention::unspecified};
    bool emulate_depth_bias{};
};
struct MslArtifact {
    std::string source;
    std::string entry_point;
    ResourcePlan reflection; // Includes all consumed specialization dependencies.
};
// Only decode is also declared in decoder.h. These are contracts, not stub phases.
[[nodiscard]] Result<RegisterProgram> emit_ir(const DecodedProgram&, FactReader&, Limits);
[[nodiscard]] Result<SsaProgram> construct_ssa(RegisterProgram, Limits);
[[nodiscard]] Result<LegalProgram> legalize(SsaProgram, FactReader&, const MetalProfile&);
[[nodiscard]] Result<StructuredProgram> structure(LegalProgram, Limits);
[[nodiscard]] Result<ResourcePlan> plan_resources(const StructuredProgram&, const MetalProfile&);
[[nodiscard]] Result<MslArtifact> emit_msl(const StructuredProgram&, const ResourcePlan&, const EmitOptions&);

static_assert(std::is_same_v<decltype(&decode), Result<DecodedProgram> (*)(const StageImage&, FactReader&, Limits)>);
static_assert(std::is_same_v<decltype(&emit_ir), Result<RegisterProgram> (*)(const DecodedProgram&, FactReader&, Limits)>);
static_assert(std::is_same_v<decltype(&construct_ssa), Result<SsaProgram> (*)(RegisterProgram, Limits)>);
static_assert(std::is_same_v<decltype(&legalize), Result<LegalProgram> (*)(SsaProgram, FactReader&, const MetalProfile&)>);
static_assert(std::is_same_v<decltype(&structure), Result<StructuredProgram> (*)(LegalProgram, Limits)>);
static_assert(std::is_same_v<decltype(&plan_resources), Result<ResourcePlan> (*)(const StructuredProgram&, const MetalProfile&)>);
static_assert(std::is_same_v<decltype(&emit_msl), Result<MslArtifact> (*)(const StructuredProgram&, const ResourcePlan&, const EmitOptions&)>);
} // namespace armigo::shader::pipeline
