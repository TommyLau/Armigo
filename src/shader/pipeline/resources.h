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
// Version 2 appends the render-target height in pixels as a second word; a
// version-2 record also satisfies version-1 readers.
inline constexpr std::string_view draw_support_schema = "armigo-draw-parameters";
inline constexpr std::uint32_t draw_support_version = 1;
inline constexpr std::uint32_t draw_support_bytes = 4;
inline constexpr std::uint32_t draw_support_height_version = 2;
inline constexpr std::uint32_t draw_support_height_bytes = 8;

struct ConstantBufferSource { std::uint8_t slot{}; };
struct StorageBufferSource { TextureKey descriptor; };
// Guest-memory pool table: the GPU-visible guest memory pools, sorted by
// guest address, with the Metal device address of each, for
// AddressSpace::guest. Little-endian words; a header of eight words, then
// guest_memory_entries entries of eight words at byte 32:
//   word 0      entry count N, the empty entry 0 included (1..8192)
//   word 1      misses: accesses outside every pool or not naturally
//               aligned, counted atomically (the count may wrap)
//   word 2      tag of the program whose access missed first
//   words 3..4  first missing guest address (low, high)
//   word 5      first-miss latch: nonzero once any access has missed; the
//               access that sets it alone writes words 2..4
//   words 6..7  zero
//   entry i     guest start (low, high), guest end (exclusive; low, high),
//               delta = device address of the start - guest start (low,
//               high, wrapping), hint, partial hint
//   write log   after the entries, at byte guest_memory_write_log_byte: one
//               bit per entry (bit e % 32 of word e / 32), set by every
//               store, atomic or reduction that reaches entry e
// Entry 0 is empty; entries 1..N-1 are sorted by start and disjoint. The
// hint of entry (c & 8191) is (c << 13) | e for a 2 MiB chunk c = address >>
// 21 below 2^19 that lies wholly inside entry e, or 0. Its partial hint is
// (c << 13) | e for such a chunk that entry e is the first to overlap and
// that no entry wholly holds, or 0 (as a table without them leaves it). A
// lookup takes the hinted entry when its chunk matches, else, when the
// partial hint's chunk matches, the first of up to four entries from it that
// holds the address, else binary-searches the last entry starting at or
// below the address. A whole access must lie in one entry at
// its natural alignment; otherwise it reads zero and drops its stores or
// atomic. A write tests its entry's log bit and sets it when clear, with a
// relaxed atomic OR. The host clears words 1..5 and the write log before the
// work, reports a set latch and reads the pools written once it completes,
// and keeps every listed pool resident.
inline constexpr std::string_view guest_memory_schema = "armigo-guest-memory";
inline constexpr std::uint32_t guest_memory_version = 3;
inline constexpr std::uint32_t guest_memory_entries = 8192;
inline constexpr std::uint32_t guest_memory_header_words = 8;
inline constexpr std::uint32_t guest_memory_entry_words = 8;
inline constexpr std::uint32_t guest_memory_write_log_byte = (guest_memory_header_words + guest_memory_entries * guest_memory_entry_words) * 4;
inline constexpr std::uint32_t guest_memory_bytes = guest_memory_write_log_byte + guest_memory_entries / 8;
inline constexpr std::uint32_t guest_memory_alignment = 8;
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
enum class ConstantBufferPolicy : std::uint8_t {
    none = 0, runtime_zero_words_v1 = 1, padded_zero_words_v2 = 2, proven_zero_words_v2 = 3
};
inline constexpr std::uint32_t constant_buffer_zero_padding_bytes = 16;
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
    // A texture pool of this descriptor's element type, bound as a pool table
    // in the buffer namespace (the project documentation).
    bool pool{};
};
struct SamplerUse {
    TextureKey key; // May differ from the texture key for separate samplers.
    bool comparison{};
    bool normalized_coordinates{};
    std::uint32_t array_length{1};
    // The sampler pool, bound as a pool table in the buffer namespace.
    bool pool{};
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
    // Legalization also moves implicit-LOD and bias samples out of their
    // predicate guards: a title's opt-in (the project documentation).
    bool hoist_implicit_samples{};
    // A read-only texture handle no constant-buffer word or handle program
    // resolves is read from the texture and sampler pools at run time
    // (the project documentation).
    bool texture_pools{};
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
    // Present only for a vertex stage consuming InstanceId or a fragment stage
    // reading the render-target height. The native base may include a split
    // continuation; this input retains the original base.
    std::optional<std::uint32_t> draw_support_buffer_index;
    // Present only for a vertex stage translated for shader-side vertex fetch:
    // the buffer that holds its per-draw fetch table (vertex_fetch.h).
    std::optional<std::uint32_t> vertex_fetch_buffer_index;
    // A vertex stage whose program never exports the guest point size (Maxwell
    // output attribute 0x6c) and is not layered: the emitter writes the fixed
    // size 1.0 to a synthesized [[point_size]] output.
    bool fixed_point_size{};
    // A vertex stage with no outputs at all: its entry returns void, which
    // Metal admits only with rasterization disabled, so draws must discard
    // primitives before rasterization.
    bool requires_rasterizer_discard{};
    // Optional static void entry for the same guest body, selected only when
    // draw state explicitly disables rasterization.
    std::string discard_entry_point;
    // A fragment stage reading the guest point coordinate (attribute 0x2e0,
    // [[point_coord]]); function constant 11 selects a lower-left origin.
    bool reads_point_coord{};
    // A fragment stage reading the render-target height: its draw support
    // record is version 2 and the height is word 1.
    bool reads_render_target_height{};
    // Program tag a guest-memory miss records (EmitOptions::guest_memory_tag).
    std::uint32_t guest_memory_tag{};
    // The handle programs of textures and samplers keyed at handle_program_slot
    // (handle_program.h), by the key's byte offset / 4.
    std::vector<std::vector<std::uint32_t>> handle_programs;
    // The program stores to guest memory or applies an atomic or reduction
    // to it; a program that only loads reads its pools.
    bool guest_memory_writes{};
};
} // namespace armigo::shader::pipeline
