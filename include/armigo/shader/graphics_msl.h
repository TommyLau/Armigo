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

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::graphics_msl {

// Storage formats are admitted explicitly by each backend.
enum class VertexFormat : std::uint8_t { Float32x3, Float32x2 };
enum class VertexDepthConvention : std::uint8_t {
    Unspecified,
    ZeroToOne,
    NegativeOneToOne,
};
enum class FragmentPositionConvention : std::uint8_t {
    Unspecified,
    MetalPositionWMatchesGuestPositionW,
};

struct VertexAttributeOption {
    std::uint8_t generic_slot = 0;
    std::uint8_t attribute_index = 0;
    VertexFormat format = VertexFormat::Float32x3;
};

struct EmitOptions {
    std::vector<VertexAttributeOption> vertex_attributes;
    VertexDepthConvention vertex_depth = VertexDepthConvention::Unspecified;
    FragmentPositionConvention fragment_position =
        FragmentPositionConvention::Unspecified;
    // A vertex stage reads its generic inputs from the bound vertex buffers
    // itself, through a per-draw fetch table (pipeline/vertex_fetch.h), so its
    // pipelines do not depend on the vertex layout.
    bool vertex_fetch = false;
    std::size_t max_source_bytes = 1024u * 1024u;
    std::size_t max_texture_resources = 16;
};

struct ConstantBufferBinding {
    std::uint8_t guest_binding = 0;
    std::uint32_t minimum_size = 0;
    std::uint8_t metal_buffer_index = 0;
};

struct TextureBinding {
    // Texel reads and size queries do not consume a sampler argument.
    static constexpr std::uint8_t no_sampler = 0xff;
    std::uint32_t handle_byte_offset = 0;
    std::uint8_t texture_index = 0;
    std::uint8_t sampler_index = 0;
};

struct GenericInterface {
    std::uint8_t slot = 0;
    std::uint8_t component_mask = 0;
    graphics::Interpolation interpolation = graphics::Interpolation::ScreenLinear;
};

struct StageInterface {
    std::uint8_t position_input_mask = 0;
    std::uint8_t position_output_mask = 0;
    std::uint32_t color_output_mask = 0;
    std::vector<GenericInterface> generic_inputs;
    std::vector<GenericInterface> generic_outputs;
};

struct EmittedStage {
    std::string source;
    std::string entry_point;
    shader_input::Stage stage = shader_input::Stage::Vertex;
    VertexDepthConvention vertex_depth = VertexDepthConvention::Unspecified;
    std::vector<ConstantBufferBinding> constant_buffers;
    std::vector<TextureBinding> textures;
    StageInterface interface;
};

struct EmitError {
    enum class Code : std::uint8_t {
        UnsupportedStage,
        InvalidProgram,
        MissingVertexFormat,
        DuplicateVertexFormat,
        UnsupportedVertexFormat,
        UnsupportedInterface,
        DepthConventionRequired,
        PositionConventionRequired,
        ResourceLimit,
        SourceLimit,
        Allocation,
        GraphUnsupported,
    };
    Code code = Code::InvalidProgram;
    std::uint32_t node = 0;
    std::uint32_t detail = 0;
    [[nodiscard]] const char* message() const noexcept;
};

using EmitResult = std::variant<EmittedStage, EmitError>;

[[nodiscard]] EmitResult emit(const graphics::Program& program,
                              const EmitOptions& options = {});

}  // namespace armigo::shader::graphics_msl
