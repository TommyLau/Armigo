// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Storage-buffer descriptors carry the guest address, size, and write access
// required to bind the reflected shader resource.

#pragma once

#include <armigo/graphics/engines.h>
#include <armigo/graphics/vertex_input.h>
#include <armigo/shader/graphics_msl.h>
#include <cstdint>
#include <memory>
#include <optional>

namespace armigo::platform {

// The provider validates chain/source/artifact digests and bound-program identity.
// Digests alone do not establish authenticity. No source compilation occurs here.
enum class TextureFactDimension : std::uint8_t { d1, d2, d3, cube };
enum class TextureSampledScalar : std::uint8_t { f32, u32, s32 };
enum class TextureComponentFormat : std::uint8_t {
    floating, unorm, snorm, unsigned_integer, signed_integer, depth
};

struct DerivedTextureFact {
    // The texture_descriptor fact key: the bound handle this shape was proven
    // from. The backend re-checks the complete descriptor contract against
    // the live TIC and any consumed TSC before acquiring a resident texture.
    std::uint8_t cbuf_slot = 0;
    std::uint32_t handle_byte_offset = 0;
    bool normalized_coordinates = true;
    bool comparison = false;
    TextureFactDimension dimension = TextureFactDimension::d2;
    bool array = false;
    bool multisample = false;
    TextureSampledScalar scalar = TextureSampledScalar::f32;
    // Floating means float-sampled, including normalized and sRGB formats.
    TextureComponentFormat component_format = TextureComponentFormat::floating;
};

// Runtime word-zero windows are separate from legacy minimum-size bindings.
// Each window has four-byte alignment and runtime_zero_words_v1 semantics.
struct ConstantBufferWindow {
    std::uint8_t guest_binding = 0;
    std::uint32_t window_bytes = 0;
    std::uint8_t metal_buffer_index = 0;
};

// Component type of one reflected fragment colour output. Render targets admit
// float or raw 32-bit integer exports; anything else stops translation.
enum class ColorOutputScalar : std::uint8_t { f32, u32, s32 };

// One written render target in the v2 fragment interface: its location, the
// components actually exported, and the type the fragment_output_type fact
// proved for the whole target. The legacy v1 schema is synthesized internally
// as its known target-0 float32x4 export.
struct CompiledColorOutput {
    std::uint8_t location = 0;
    std::uint8_t component_mask = 0;
    ColorOutputScalar scalar = ColorOutputScalar::f32;
    std::uint8_t components = 0;
};

enum class CompiledShaderResourceKind : std::uint8_t {
    SampledTexture,
    StorageImage,
    Sampler,
};

enum class CompiledShaderResourceAccess : std::uint8_t { Read, Write, ReadWrite };

struct CompiledStorageBufferBinding {
    std::uint8_t cbuf_slot = 0;
    std::uint32_t descriptor_byte_offset = 0;
    std::uint8_t metal_buffer_index = 0;
    CompiledShaderResourceAccess access = CompiledShaderResourceAccess::Read;
    std::uint32_t alignment = 4;
};

enum class CompiledShaderResourceLocationKind : std::uint8_t {
    Direct,
    ArgumentBufferMember,
};
enum class CompiledShaderResourceNamespace : std::uint8_t { Texture, Sampler };
enum class CompiledShaderTextureDimension : std::uint8_t { d1, d2, d3, cube, buffer };
enum class CompiledShaderTextureScalar : std::uint8_t { f16, f32, u32, s32 };

struct CompiledShaderTextureDescriptor {
    CompiledShaderTextureDimension dimension = CompiledShaderTextureDimension::d2;
    bool array = false;
    bool multisample = false;
    CompiledShaderTextureScalar scalar = CompiledShaderTextureScalar::f32;
    TextureComponentFormat component_format = TextureComponentFormat::floating;
    bool normalized_coordinates = true;
    bool comparison = false;
};

struct CompiledShaderResourceLocation {
    CompiledShaderResourceLocationKind kind = CompiledShaderResourceLocationKind::Direct;
    CompiledShaderResourceNamespace resource_namespace =
        CompiledShaderResourceNamespace::Texture;
    // Direct locations use first/count. Argument-buffer members use
    // byte_offset/count/stride and derive their namespace from the resource.
    std::uint32_t first = 0;
    std::uint32_t byte_offset = 0;
    std::uint32_t count = 0;
    std::uint32_t stride = 0;
};

struct CompiledShaderResource {
    std::uint32_t resource_id = 0;
    std::uint32_t array_length = 0;
    std::uint8_t cbuf_slot = 0;
    std::uint32_t handle_byte_offset = 0;
    CompiledShaderResourceKind kind = CompiledShaderResourceKind::SampledTexture;
    CompiledShaderResourceAccess access = CompiledShaderResourceAccess::Read;
    bool comparison = false;
    bool normalized_coordinates = true;
    std::optional<CompiledShaderTextureDescriptor> texture_descriptor;
    CompiledShaderResourceLocation location;
};

struct CompiledArgumentBufferLayout {
    std::uint32_t abi_version = 0;
    std::uint8_t buffer_index = 0;
    std::uint32_t byte_extent = 0;
    std::uint32_t alignment = 0;
};

struct CompiledCustomBorderPair {
    std::uint32_t texture_resource = 0, sampler_resource = 0;
};
// coverage-v1: immutable 64-byte records in the reflected buffer slot.
struct CompiledCustomBorderSupport {
    std::uint8_t buffer_index = 0;
    std::vector<CompiledCustomBorderPair> pairs;
};

struct CompiledGuestMemory {
    std::uint8_t buffer_index{};
    std::uint32_t tag{};
    // The stage stores to guest memory or applies an atomic or reduction to
    // it; a stage that only loads reads its pools.
    bool writes{true};
    bool operator==(const CompiledGuestMemory&) const = default;
};

// An NVN2 task (Metal object) or mesh stage's launch: its threadgroup size
// and the payload bytes each task threadgroup passes to the mesh threadgroups
// it launches (a task stage's) or a mesh threadgroup reads (a mesh stage's).
// launched_by_task: a mesh stage translated to run under a task stage.
struct CompiledMeshLaunch {
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
    std::uint32_t payload_bytes = 0;
    bool launched_by_task = false;
    bool operator==(const CompiledMeshLaunch&) const = default;
};

struct CompiledGraphicsStage {
    std::vector<std::byte> metallib;
    // A stage translated at run time carries its Metal source instead of a
    // compiled library (metallib stays empty); its user compiles the source.
    // The source is held here, or stays in msl_source_file until it is
    // compiled, when its SHA-256 must equal msl_source_digest (see
    // platform::stage_source).
    std::string msl_source;
    std::string msl_source_file;
    std::string msl_source_digest;
    // Guest program bytes the stage was identified by; zero when unknown.
    std::uint64_t guest_code_bytes = 0;
    // What names this stage across package rebuilds (program digest plus its
    // variant discriminator); empty for stages not loaded from a package.
    std::string identity;
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::string entry_point;
    shader::graphics_msl::VertexDepthConvention vertex_depth =
        shader::graphics_msl::VertexDepthConvention::Unspecified;
    shader::graphics_msl::FragmentPositionConvention fragment_position =
        shader::graphics_msl::FragmentPositionConvention::Unspecified;
    std::vector<shader::graphics_msl::VertexAttributeOption> vertex_attributes;
    // Reflection v2 vertex inputs with a code-derived storage class: location
    // and class per generic input, mutually exclusive with vertex_attributes.
    std::vector<graphics::VertexClassBinding> vertex_class_inputs;
    std::vector<shader::graphics_msl::ConstantBufferBinding> constant_buffers;
    std::vector<shader::graphics_msl::TextureBinding> textures;
    // Reflection v3 preserves texture and sampler identity independently.
    // The legacy paired list above is synthesized only for exactly
    // representable direct scalar resources.
    std::vector<CompiledShaderResource> resources;
    // Host-evaluated handle programs (shader/pipeline/handle_program.h). A
    // texture or sampler whose cbuf_slot is 31 takes program
    // handle_byte_offset / 4 from this list. Replay runs it when encoding the
    // draw or dispatch, not when the command is recorded.
    std::vector<std::vector<std::uint32_t>> handle_programs;
    std::optional<CompiledArgumentBufferLayout> argument_buffer;
    std::optional<CompiledCustomBorderSupport> custom_border;
    bool legacy_texture_bindings_compatible = true;
    shader::graphics_msl::StageInterface interface;
    // Typed per-target fragment colour exports. The summary mask above stays
    // exactly their four-bit-per-target union so both schemas agree.
    std::vector<CompiledColorOutput> color_outputs;
    // render-target-v1 exports keep raw uint32 words offline. Package lookup
    // resolves each color_outputs scalar from the bound logical attachment;
    // Metal function constant <location> selects f32=0, u32=1 or s32=2.
    bool fragment_output_types_from_render_targets = false;
    // shader-v1: fragment buffer 30 is float4(constant, slope, clamp, enabled).
    // This stage capability is immutable; values remain dynamic draw state.
    // Function constant 8 enables its depth output only for an active fallback.
    bool emulates_depth_bias = false;
    // Explicit fragment-depth and sample-mask export flags (v2 only). The MSL
    // emitter supports both; the current single-target draw backend stops on
    // any stage that sets them. Legacy v1 stages cannot express either.
    bool writes_fragment_depth = false;
    bool writes_sample_mask = false;
    // y-direction-v1: the stage reads S2R SR_YDIRECTION and declares the
    // armigo_y_negate function constant (index 9). Draw time specializes it
    // from the guest YControl negate bit and keys the pipeline on that bit.
    bool consumes_y_direction = false;
    // The vertex stage writes [[point_size]]: its program's own export, or (when
    // fixed_point_size) the fixed size 1.0; point draws with another fixed size
    // are refused. Such pipelines leave the topology class unspecified.
    bool writes_point_size = false;
    bool fixed_point_size = false;
    // The fragment stage reads the guest point coordinate (attribute 0x2e0,
    // Metal [[point_coord]]); function constant 11 selects a lower-left origin.
    bool reads_point_coord = false;
    // A vertex stage with no outputs, whose entry returns void: it draws only
    // with rasterizer discard.
    bool requires_rasterizer_discard = false;
    // The texture_handle_cbuf fact this stage consumed offline: the constant
    // buffer its bound handles were proven to come from. Draw time must stop
    // unless the 3D texture buffer selector register (0x982) still selects it;
    // absent for v1 stages, which resolve handles from the live register.
    std::optional<std::uint8_t> texture_handle_cbuf;
    // Complete reflected texture contracts (v2 only), including consumed
    // texture_descriptor facts. Draw time verifies live TIC state and, for
    // bindings that consume a sampler, live TSC state before acquisition.
    std::vector<DerivedTextureFact> texture_facts;
    std::vector<ConstantBufferWindow> constant_buffer_windows;
    std::vector<CompiledStorageBufferBinding> storage_buffers;
    // armigo-cb-sizes v1: 18 little-endian uint32 programmed byte sizes, zero
    // for disabled slots. Present exactly when runtime windows are present.
    std::optional<std::uint8_t> constant_buffer_support_index;
    // armigo-draw-parameters: original guest first instance as one uint32
    // (v1, vertex stages); a fragment stage takes v2, which adds the
    // render-target height as a second word. Native baseInstance still
    // includes the split continuation index.
    std::optional<std::uint8_t> draw_support_buffer_index;
    // armigo-guest-memory v2: the writable guest-memory pool table and the
    // tag the stage records on its first miss.
    std::optional<CompiledGuestMemory> guest_memory;
    // Task and mesh stages only: the launch a mesh draw dispatches with. A
    // task stage reads the draw's first task workgroup, and a mesh stage
    // without a task stage its first mesh workgroup, as word 0 of the
    // draw-parameter record at draw_support_buffer_index.
    std::optional<CompiledMeshLaunch> task, mesh;
    // armigo-vertex-fetch v1: a vertex stage that reads its generic inputs
    // through a per-draw fetch table (armigo/shader/vertex_fetch.h) at this
    // buffer index instead of a vertex descriptor.
    std::optional<std::uint8_t> vertex_fetch_buffer_index;
    // Compute stages only (schema armigo-shader-stage-compute-v1): the local
    // workgroup size and static threadgroup memory bytes this artifact was
    // specialized for. Package lookup matches them against the decoded queue
    // meta data exactly; a miss is a structured error naming both sides.
    std::array<std::uint32_t, 3> compute_local_size{1, 1, 1};
    std::uint32_t compute_shared_memory_bytes = 0;
    // False for the compute-v1 schema, whose constant buffers are not
    // reflected: the replay cannot name every guest range such a stage reads.
    bool constant_buffers_reflected = true;
};

struct CompiledGraphicsPair {
    std::shared_ptr<const CompiledGraphicsStage> vertex;
    std::shared_ptr<const CompiledGraphicsStage> fragment;
};
using CompiledGraphicsLookupResult = std::variant<CompiledGraphicsPair, graphics::CommandResult>;
using CompiledGraphicsLookup = CompiledGraphicsLookupResult (*)(
    void*, const graphics::RenderStop&, const graphics::GpuAddressSpace&);

// Compute dispatch lookup over the same inputs: the single compiled compute
// stage for a RenderKind::Compute stop, or a structured CommandResult refusal
// (a miss names the program and the requested local size/shared memory).
using CompiledComputeStageResult =
    std::variant<std::shared_ptr<const CompiledGraphicsStage>, graphics::CommandResult>;
using CompiledComputeLookup = CompiledComputeStageResult (*)(
    void*, const graphics::RenderStop&, const graphics::GpuAddressSpace&);

struct MetalDrawStatistics {
    std::uint64_t completed_draws = 0;
    std::uint64_t written_pixels = 0;
};
struct MetalDrawContext {
    CompiledGraphicsLookup lookup = nullptr;
    void* lookup_context = nullptr;
    MetalDrawStatistics* statistics = nullptr;
};

// Validates the existing synchronous adapter state domain without GPU work.
// The first persistent activation uses this same gate and specific diagnostics.
[[nodiscard]] graphics::CommandResult validate_metal_draw_state(
    const graphics::RenderStop& snapshot) noexcept;

// Synchronous GPU completion and guest writeback; borrowed memory never escapes.
// The caller serializes context/statistics and guest target access during a draw.
[[nodiscard]] graphics::CommandResult execute_metal_draw(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept;

}  // namespace armigo::platform
