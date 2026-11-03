// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "pipeline_state.h"
#import <Metal/Metal.h>

namespace armigo::platform::metal {

// render-target-v1 reserves one uint constant per written logical location.
// The exact little-endian values also belong to the pipeline cache identity.
inline std::vector<SpecializationConstant> fragment_output_specialization(
    const CompiledGraphicsStage& stage) {
    std::vector<SpecializationConstant> result;
    if (!stage.fragment_output_types_from_render_targets) return result;
    for (unsigned location = 0; location < 8; ++location) {
        for (const auto& output : stage.color_outputs) {
            if (output.location != location) continue;
            result.push_back({location, "u32", {std::byte(output.scalar),
                std::byte{0}, std::byte{0}, std::byte{0}}});
        }
    }
    return result;
}

inline MTLFunctionConstantValues* fragment_output_constants(
    const CompiledGraphicsStage& stage, bool shader_depth_bias = false) {
    if (!stage.fragment_output_types_from_render_targets && !stage.emulates_depth_bias) return nil;
    auto* constants = [MTLFunctionConstantValues new];
    if (stage.fragment_output_types_from_render_targets) {
        for (const auto& output : stage.color_outputs) {
            const auto value = static_cast<std::uint32_t>(output.scalar);
            [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:output.location];
        }
    }
    if (stage.emulates_depth_bias)
        [constants setConstantValue:&shader_depth_bias type:MTLDataTypeBool atIndex:8];
    return constants;
}

inline MTL4FunctionDescriptor* specialize_fragment_output(
    MTL4LibraryFunctionDescriptor* function, const CompiledGraphicsStage& stage,
    bool shader_depth_bias = false) {
    if (!stage.fragment_output_types_from_render_targets && !stage.emulates_depth_bias) return function;
    auto* specialized = [MTL4SpecializedFunctionDescriptor new];
    specialized.functionDescriptor = function;
    // Constant 8 removes the depth output whenever the fallback is inactive.
    // PipelineKey.shader_depth_bias identifies this descriptor-baked choice.
    specialized.constantValues = fragment_output_constants(stage, shader_depth_bias);
    return specialized;
}

} // namespace armigo::platform::metal
