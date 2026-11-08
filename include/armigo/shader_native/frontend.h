// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader_native/program.h>
#include "pipeline/phases.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace armigo::shader_native {
namespace ir = armigo::shader::pipeline;
enum class PixelRateState { unknown, disabled, enabled };
inline constexpr std::string_view disabled_pixel_rate_profile = "armigo-native-metal4-vprs-disabled";
struct FrontendOptions {
    // Clip convention is draw state, not a property inferred from shader bytes.
    std::optional<ir::DepthConvention> vertex_depth;
    bool vertex_fetch = true;
    ir::FragmentPositionConvention fragment_position{ir::FragmentPositionConvention::unspecified};
    // This is a caller-supplied draw-state fact, never inferred from a zero index.
    PixelRateState pixel_rate{PixelRateState::unknown};
    // A mesh stage's paired fragment stage: the components (bit c of entry n)
    // of each generic input it reads. With it the mesh outputs exactly those
    // components, one its output maps lack holding what a stage input no
    // stage wrote holds: 0 for x, y and z, 1 for w. Without it the mesh
    // outputs every component its output maps enable.
    std::optional<std::array<std::uint8_t, 32>> mesh_fragment_reads;
    // Read a texture handle no constant-buffer word or handle program
    // resolves from the texture and sampler pools at run time
    // (the project documentation). The caller sets it when it binds pool
    // tables; otherwise such a handle is refused.
    bool texture_pools = false;
};
// The generic components (bit c of entry n) a mesh stage's output maps
// enable, per vertex and per primitive together.
std::array<std::uint8_t, 32> mesh_generic_outputs(const Program&);
// The frontend produces the existing register IR directly. No intermediate
// machine code is generated, and unsupported native semantics remain errors.
ir::Result<ir::RegisterProgram> emit_native_ir(const Program&, ir::Limits = {});
ir::Result<ir::MslArtifact> translate_native(const Program&, const FrontendOptions& = {});
} // namespace armigo::shader_native
