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
#include "handle_program.h"
#include "types.h"

#include <array>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::pipeline {

enum class OriginKind : std::uint8_t { validated_header, captured_state, explicit_option, derived };
struct EvidenceOrigin {
    OriginKind kind{OriginKind::captured_state};
    // Stable evidence identity, not a host path. Audit equality includes origin;
    // specialization identity serializes only key/value, not the evidence label.
    std::string identity;
    auto operator<=>(const EvidenceOrigin&) const = default;
};
template <class T> struct Known {
    T value;
    EvidenceOrigin origin;
    bool operator==(const Known&) const = default;
};
struct Unknown {
    std::string reason;
    bool operator==(const Unknown&) const = default;
};
template <class T> using Knowledge = std::variant<Unknown, Known<T>>;

enum class VertexFormat : std::uint8_t { u8, s8, unorm8, snorm8, u16, s16, unorm16, snorm16, f16, u32, s32, f32 };
struct AttributeFormat {
    VertexFormat format{VertexFormat::f32};
    std::uint8_t components{1};
    auto operator<=>(const AttributeFormat&) const = default;
};
enum class ComponentFormat : std::uint8_t { unorm, snorm, unsigned_integer, signed_integer, floating, depth };
struct TextureDescriptor {
    TextureShape shape;
    ScalarType scalar{ScalarType::f32};
    ComponentFormat format{ComponentFormat::floating};
    bool normalized_coordinates{};
    bool comparison{};
    // A format code is meaningful only in the named, versioned descriptor namespace.
    std::uint32_t format_code{};
    std::string format_namespace;
    bool operator==(const TextureDescriptor&) const = default;
};
enum class Topology : std::uint8_t { points, lines, triangles };
enum class DepthConvention : std::uint8_t { zero_to_one, negative_one_to_one };
enum class OriginConvention : std::uint8_t { upper_left, lower_left };
enum class Compare : std::uint8_t { never, less, equal, less_equal, greater, not_equal, greater_equal, always };
// Launch dimensions are supplied by QMD/launch evidence, never inferred from code.
struct ComputeLocalSize {
    std::array<std::uint32_t, 3> dimensions{};
    bool operator==(const ComputeLocalSize&) const = default;
};
struct ComputeConfiguration {
    ComputeLocalSize local_size;
    std::uint32_t shared_memory_bytes{};
    bool operator==(const ComputeConfiguration&) const = default;
};
[[nodiscard]] constexpr bool valid_compute_configuration(const ComputeConfiguration& c) {
    const auto& d = c.local_size.dimensions;
    return d[0] && d[1] && d[2] && d[0] <= 1024 && d[1] <= 1024 && d[2] <= 64 &&
        std::uint64_t(d[0]) * d[1] * d[2] <= 1024 &&
        c.shared_memory_bytes <= 32768 && c.shared_memory_bytes % 4 == 0;
}
struct AlphaTest {
    Compare comparison{Compare::always};
    std::uint32_t reference_bits{};
    bool operator==(const AlphaTest&) const = default;
};

// index is the guest slot/location; byte_offset is always in bytes, never words.
enum class FactKind : std::uint8_t {
    // The mask and size identities are deprecated draw-state facts. Keep their
    // serialized values and every later enumerator stable for external tools.
    local_memory_bytes, constant_buffer_bound_mask, constant_buffer_size,
    constant_buffer_word, texture_handle_cbuf, texture_descriptor, texture_array_length,
    vertex_attribute_format, fragment_output_type, input_interpolation,
    topology, depth_convention, origin_convention, negate_y, alpha_test,
    early_z, alpha_to_coverage, dual_source_blend, draw_parameters,
    compute_local_size, compute_shared_memory_bytes, count
};
struct FactKey {
    FactKind kind{};
    std::uint32_t index{};
    std::uint32_t byte_offset{};
    std::uint8_t component{}; // Only input_interpolation uses this field (0..3).
    constexpr auto operator<=>(const FactKey&) const = default;
};
[[nodiscard]] constexpr bool valid_fact_key(FactKey key) {
    if (key.component > 3 || (key.kind != FactKind::input_interpolation && key.component != 0)) return false;
    switch (key.kind) {
    case FactKind::constant_buffer_word: case FactKind::texture_array_length: return key.index < 18 && key.byte_offset % 4 == 0;
    // A texture keyed at handle_program_slot takes its handle from a program.
    case FactKind::texture_descriptor: return (key.index < 18 || key.index == handle_program_slot) && key.byte_offset % 4 == 0;
    case FactKind::constant_buffer_size: return key.index < 18 && key.byte_offset == 0;
    case FactKind::vertex_attribute_format: case FactKind::fragment_output_type:
    case FactKind::input_interpolation: return key.index <= 65535 && key.byte_offset == 0;
    case FactKind::local_memory_bytes: case FactKind::constant_buffer_bound_mask:
    case FactKind::texture_handle_cbuf: case FactKind::topology: case FactKind::depth_convention:
    case FactKind::origin_convention: case FactKind::negate_y: case FactKind::alpha_test:
    case FactKind::early_z: case FactKind::alpha_to_coverage: case FactKind::dual_source_blend:
    case FactKind::draw_parameters: case FactKind::compute_local_size:
    case FactKind::compute_shared_memory_bytes: return key.index == 0 && key.byte_offset == 0;
    case FactKind::count: return false;
    }
    return false;
}
using FactValue = std::variant<std::uint32_t, bool, AttributeFormat, IoType, Interpolation,
                               TextureDescriptor, Topology, DepthConvention,
                               OriginConvention, AlphaTest, ComputeLocalSize>;
struct Dependency {
    FactKey key;
    FactValue value;
    EvidenceOrigin origin;
    bool operator==(const Dependency&) const = default;
};

template <FactKind K> struct FactTraits;
#define ARMIGO_FACT_TYPE(kind, type) \
    template <> struct FactTraits<FactKind::kind> { using value_type = type; }
ARMIGO_FACT_TYPE(local_memory_bytes, std::uint32_t);
ARMIGO_FACT_TYPE(compute_local_size, ComputeLocalSize);
ARMIGO_FACT_TYPE(compute_shared_memory_bytes, std::uint32_t);
ARMIGO_FACT_TYPE(constant_buffer_bound_mask, std::uint32_t);
ARMIGO_FACT_TYPE(constant_buffer_size, std::uint32_t);
ARMIGO_FACT_TYPE(constant_buffer_word, std::uint32_t);
ARMIGO_FACT_TYPE(texture_handle_cbuf, std::uint32_t);
ARMIGO_FACT_TYPE(texture_descriptor, TextureDescriptor);
ARMIGO_FACT_TYPE(texture_array_length, std::uint32_t);
ARMIGO_FACT_TYPE(vertex_attribute_format, AttributeFormat);
ARMIGO_FACT_TYPE(fragment_output_type, IoType);
ARMIGO_FACT_TYPE(input_interpolation, Interpolation);
ARMIGO_FACT_TYPE(topology, Topology);
ARMIGO_FACT_TYPE(depth_convention, DepthConvention);
ARMIGO_FACT_TYPE(origin_convention, OriginConvention);
ARMIGO_FACT_TYPE(negate_y, bool);
ARMIGO_FACT_TYPE(alpha_test, AlphaTest);
ARMIGO_FACT_TYPE(early_z, bool);
ARMIGO_FACT_TYPE(alpha_to_coverage, bool);
ARMIGO_FACT_TYPE(dual_source_blend, bool);
ARMIGO_FACT_TYPE(draw_parameters, bool);
#undef ARMIGO_FACT_TYPE

template <FactKind K> struct FactQuery {
    using value_type = typename FactTraits<K>::value_type;
    std::uint32_t index{};
    std::uint32_t byte_offset{};
    std::uint8_t component{};
    [[nodiscard]] constexpr FactKey key() const { return {K, index, byte_offset, component}; }
};
namespace fact_query {
[[nodiscard]] constexpr auto constant_word(std::uint8_t slot, std::uint32_t byte_offset) {
    return FactQuery<FactKind::constant_buffer_word>{slot, byte_offset};
}
[[nodiscard]] constexpr auto texture(TextureKey key) {
    return FactQuery<FactKind::texture_descriptor>{key.cbuf_slot, key.handle_byte_offset};
}
[[nodiscard]] constexpr auto interpolation(std::uint16_t location, std::uint8_t component) {
    return FactQuery<FactKind::input_interpolation>{location, 0, component};
}
} // namespace fact_query
static_assert(std::is_same_v<FactQuery<FactKind::texture_descriptor>::value_type, TextureDescriptor>);
} // namespace armigo::shader::pipeline
