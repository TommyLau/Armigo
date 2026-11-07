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
};
// The frontend produces the existing register IR directly. No intermediate
// machine code is generated, and unsupported native semantics remain errors.
ir::Result<ir::RegisterProgram> emit_native_ir(const Program&, ir::Limits = {});
ir::Result<ir::MslArtifact> translate_native(const Program&, const FrontendOptions& = {});
} // namespace armigo::shader_native
