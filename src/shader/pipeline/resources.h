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
#include "ir.h"

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::shader::pipeline {
inline constexpr std::uint32_t constant_buffer_slot_count = 18;
inline constexpr std::uint32_t constant_buffer_window_bytes = 65536;
inline constexpr std::string_view constant_buffer_support_schema = "armigo-cb-sizes";
inline constexpr std::uint32_t constant_buffer_support_version = 1;
// ABI payload: uint32_t size_bytes[constant_buffer_slot_count].
inline constexpr std::uint32_t constant_buffer_support_bytes = 72;
static_assert(constant_buffer_support_bytes == constant_buffer_slot_count * sizeof(std::uint32_t));
// ABI payload: one uint32_t containing the original guest first instance.
inline constexpr std::string_view draw_support_schema = "armigo-draw-parameters";
inline constexpr std::uint32_t draw_support_version = 1;
inline constexpr std::uint32_t draw_support_bytes = 4;

struct ConstantBufferSource { std::uint8_t slot{}; };
struct StorageBufferSource { TextureKey descriptor; };
struct SupportBufferSource { std::string schema; std::uint32_t version{}; };
struct VertexStreamSource { std::uint16_t slot{}; };
using BufferSource = std::variant<ConstantBufferSource, StorageBufferSource,
                                  SupportBufferSource, VertexStreamSource>;
// Append-only: serialized tools outside this target retain the first two values.
enum class SizeKind : std::uint8_t {
    accessed_extent = 0,
    full_binding = 1,
    constant_buffer_window = 2,
    // Bind the descriptor's complete size (third word), rounded up to words,
    // with at least four bytes backing even when the runtime descriptor is empty.
    storage_descriptor = 3,
};
enum class ConstantBufferPolicy : std::uint8_t { none = 0, runtime_zero_words_v1 = 1 };
struct SizeRequirement {
    SizeKind kind{SizeKind::accessed_extent};
    std::uint64_t bytes{}; // Proven extent, binding size, or explicit bounded CB window.
    std::uint32_t alignment{1};
    ConstantBufferPolicy constant_buffer_policy{ConstantBufferPolicy::none};
};
struct BufferUse { BufferSource source; Access access{Access::read}; SizeRequirement size; };
enum class ResourceTextureKind : std::uint8_t { sampled, storage_image };
struct TextureUse {
    ResourceTextureKind kind{ResourceTextureKind::sampled};
    bool coherent{}, atomic{};
    TextureKey key;
    TextureDescriptor descriptor;
    Access access{Access::read};
    std::uint32_t array_length{1};
    bool dynamically_indexed{};
    std::vector<TextureMode> modes;
};
struct SamplerUse {
    TextureKey key; // May differ from the texture key for separate samplers.
    bool comparison{};
    bool normalized_coordinates{};
    std::uint32_t array_length{1};
};
// Storage class of a generic vertex input, derived from the IR alone by the
// post-SSA use analysis (resources/vertex_class.cpp), never from pipeline
// state: a Maxwell attribute register carries float bits for float/unorm/snorm
// guest formats and raw extended integer bits otherwise, so the declaration
// only needs the class (float4/uint4/int4) and the host vertex descriptor
// converts the concrete guest format at draw time.
enum class VertexAttributeClass : std::uint8_t { float_class, uint_class, sint_class };
[[nodiscard]] constexpr ScalarType vertex_class_scalar(VertexAttributeClass cls) {
    switch (cls) {
    case VertexAttributeClass::sint_class: return ScalarType::s32;
    case VertexAttributeClass::uint_class: return ScalarType::u32;
    default: return ScalarType::f32;
    }
}
struct IoUse {
    IoSemantic semantic{IoSemantic::generic};
    std::uint16_t location{};
    std::uint8_t mask{1};
    IoType type;
    Interpolation interpolation;
    Access access{Access::read};
    std::optional<AttributeFormat> vertex_format;
    std::optional<std::uint16_t> vertex_stream;
    std::uint32_t byte_offset{};
    std::uint32_t byte_stride{};
    // Recorded instead of vertex_format when no vertex_attribute_format fact
    // was supplied for this location; io.type.scalar follows this class.
    std::optional<VertexAttributeClass> vertex_class;
};
using ResourceUse = std::variant<BufferUse, TextureUse, SamplerUse>;
struct ResourceRequirements {
    // ResourceId indexes uses. Legalization resolves all handles/descriptors/bounds.
    std::vector<ResourceUse> uses;
    std::vector<IoUse> io;
    std::optional<ComputeConfiguration> compute;
};

enum class MetalNamespace : std::uint8_t { buffer, texture, sampler };
struct MetalRange { MetalNamespace space{MetalNamespace::buffer}; std::uint32_t first{}, count{}; };
// Metal 3 direct-layout fields contain consecutive eight-byte resource handles.
struct ArgumentBufferMember { std::uint32_t byte_offset{}, count{}, stride{8}; };
using BindingLocation = std::variant<MetalRange, ArgumentBufferMember>;
struct StageArgumentBufferLayout {
    std::uint32_t abi_version{1}, buffer_index{}, byte_extent{}, alignment{8};
};
struct MetalProfile {
    // An explicit, versioned host policy. Zero capabilities are not a usable default.
    std::string identity;
    std::uint32_t version{};
    std::uint16_t buffer_slots{}, texture_slots{}, sampler_slots{};
    bool supports_fp16{}, supports_fp64{}, supports_query_lod{}, supports_fragment_atomics{};
    bool supports_demote{}, supports_simd_shuffle{};
    bool operator==(const MetalProfile&) const = default;
};
struct LegalProgram {
    Program ir;
    ResourceRequirements requirements;
    MetalProfile profile; // Legalization decisions are valid only for this exact profile.
};
struct PlannedResource { ResourceId resource; ResourceUse use; BindingLocation location; };
// coverage-v1 uses one 64-byte record per admitted texture/sampler pair.
// Each record holds a coverage texture handle, border colour and final swizzle.
struct CustomBorderPair { ResourceId texture, sampler; };
struct CustomBorderSupport {
    std::uint32_t buffer_index{};
    std::vector<CustomBorderPair> pairs;
};
struct ResourcePlan {
    std::uint32_t reflection_version{3};
    Stage stage{Stage::vertex};
    MetalProfile profile;
    std::vector<PlannedResource> resources;
    std::vector<IoUse> io;
    std::vector<Dependency> dependencies;
    // Emission host convention selected for this stage (EmitOptions), part of its identity.
    FragmentPositionConvention fragment_position{FragmentPositionConvention::unspecified};
    std::optional<ComputeConfiguration> compute;
    bool emulate_depth_bias{};
    std::optional<StageArgumentBufferLayout> argument_buffer;
    std::optional<CustomBorderSupport> custom_border;
    // Present only for a vertex stage consuming InstanceId. The native base
    // may include a split continuation; this input retains the original base.
    std::optional<std::uint32_t> draw_support_buffer_index;
    // Present only for a vertex stage translated for shader-side vertex fetch:
    // the buffer that holds its per-draw fetch table (vertex_fetch.h).
    std::optional<std::uint32_t> vertex_fetch_buffer_index;
    // A vertex stage whose program never exports the guest point size (Maxwell
    // output attribute 0x6c) and is not layered: the emitter writes the fixed
    // size 1.0 to a synthesized [[point_size]] output.
    bool fixed_point_size{};
    // A fragment stage reading the guest point coordinate (attribute 0x2e0,
    // [[point_coord]]); function constant 11 selects a lower-left origin.
    bool reads_point_coord{};
};
} // namespace armigo::shader::pipeline
