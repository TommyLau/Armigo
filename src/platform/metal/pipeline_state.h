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

#include <armigo/graphics/draw_plan.h>
#include <armigo/graphics/render_state.h>
#include <armigo/platform/metal_draw.h>
#include <string>

namespace armigo::platform::metal {

struct PipelineContextKey {
    std::uint64_t device = 0;
    std::string compiler_chain, capability_profile;
    bool operator==(const PipelineContextKey&) const = default;
};
struct SpecializationConstant {
    std::uint32_t index = 0;
    // Canonical type tag and exact bits; sorted unique indices. Unknown types
    // or conflicting values are Unsupported, never silently ignored constants.
    std::string type;
    std::vector<std::byte> value;
    bool operator==(const SpecializationConstant&) const = default;
};
struct ShaderFunctionKey {
    // Hash of the fields below when the provider precomputed it (zero
    // otherwise). First, so equality rejects different functions early.
    std::uint64_t hash = 0;
    // Exact validated package/variant/artifact identities, NOT host pointers
    // or GPU program addresses. Adapter derives these from admitted artifacts.
    std::string package, variant, artifact, entry_point;
    std::vector<SpecializationConstant> specialization;
    bool operator==(const ShaderFunctionKey&) const = default;
};
// Hash of a function key's identity fields, for ShaderFunctionKey::hash.
[[nodiscard]] std::uint64_t function_key_hash(const ShaderFunctionKey& key);
// Created only by GraphicsStageProvider after package and live program checks.
// Keys and immutable stages travel together through pipeline and binding APIs.
struct AdmittedGraphicsStages {
    CompiledGraphicsPair stages;
    ShaderFunctionKey vertex_key, fragment_key;
};
// One compiled compute stage admitted for a queue-meta-data launch.
struct AdmittedComputeStage {
    std::shared_ptr<const CompiledGraphicsStage> stage;
    ShaderFunctionKey key;
};
struct VertexAttributeKey {
    std::uint32_t location = 0, buffer_index = 0, offset = 0;
    graphics::VertexAttribFormat format{};
    bool operator==(const VertexAttributeKey&) const = default;
};
struct VertexBufferLayoutKey {
    std::uint32_t buffer_index = 0, stride = 0;
    graphics::VertexStep step = graphics::VertexStep::PerVertex;
    std::uint32_t rate = 1;
    bool operator==(const VertexBufferLayoutKey&) const = default;
};
struct PipelineColorKey {
    std::optional<graphics::TextureFormat> format;
    graphics::ColorBlendState blend;
    bool operator==(const PipelineColorKey&) const = default;
};
struct PipelineKey {
    PipelineContextKey context;
    ShaderFunctionKey vertex, fragment;
    // Canonical location/buffer order. Covers all descriptor-baked fetch state.
    std::vector<VertexAttributeKey> attributes;
    std::vector<VertexBufferLayoutKey> buffers;
    std::array<PipelineColorKey, 8> colors;
    std::optional<graphics::TextureFormat> depth_format, stencil_format;
    graphics::CoverageState coverage;
    graphics::PrimitiveClass primitive = graphics::PrimitiveClass::Triangle;
    bool rasterization_enabled = true;
    // Function constant 8: export depth only while the shader fallback is active.
    bool shader_depth_bias = false;
    std::uint64_t shader_emulation = 0;
    bool operator==(const PipelineKey&) const = default;
};
struct ResidentPipeline {
    PipelineKey key;
    graphics::RetainedResource pipeline;
    // Retains libraries/functions as well as PSOs through GPU completion.
    AdmittedGraphicsStages stages;
};
// Compute pipelines depend only on the compiled function: launch state is
// bound per dispatch, and the threadgroup memory is declared in the kernel.
struct ComputePipelineKey {
    PipelineContextKey context;
    ShaderFunctionKey function;
    bool operator==(const ComputePipelineKey&) const = default;
};
struct ResidentComputePipeline {
    ComputePipelineKey key;
    graphics::RetainedResource pipeline;
    AdmittedComputeStage stage;
};

// L9: device-owned bounded caches; hashes select buckets, equality selects an
// entry. Validate key/artifact/reflection consistency, stage linkage, typed
// color outputs, depth/sample exports and device capabilities before creation.
// Unknown blend/format/sample/emulation combinations return structured errors.
// No runtime shader translation or substitute pipeline on a miss.
// Addresses, attachment identity, uniforms and clear values NEVER enter the
// PSO key. Depth-bias values remain dynamic; shader fallback activation is
// specialized above. Viewport/scissor, blend constants, cull/winding/fill and
// stencil references are dynamic. Any backend that bakes additional state must
// extend the frozen key via the integrator before admitting that state.
class PipelineStateCache {
public:
    virtual ~PipelineStateCache() = default;
    virtual const PipelineContextKey& context() const noexcept = 0;
    virtual graphics::RendererResult<ResidentPipeline> acquire(
        const PipelineKey& key, const AdmittedGraphicsStages& stages) = 0;
    virtual graphics::RendererResult<ResidentComputePipeline> acquire_compute(
        const ComputePipelineKey&, const AdmittedComputeStage&) {
        return graphics::RendererError{{graphics::CommandStatus::Unsupported,
                                        "pipeline cache has no compute pipelines"}};
    }
    virtual graphics::RendererResult<graphics::RetainedResource> depth_stencil(
        const graphics::DepthStencilState& key) = 0;
    // Compare every admitted SamplerDescriptor field (excluding raw_words).
    // Finite float values and canonical disabled-state values are mandatory.
    virtual graphics::RendererResult<graphics::RetainedResource> sampler(
        const graphics::SamplerDescriptor& descriptor) = 0;
    virtual graphics::RendererStatus trim(std::uint64_t budget_bytes) = 0;
};

} // namespace armigo::platform::metal
