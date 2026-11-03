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

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace armigo::shader {

// Shader-side vertex fetch. A vertex stage translated with
// EmitOptions::vertex_fetch declares no vertex attributes: it reads each
// generic input from the bound vertex buffers itself, through one table per
// draw at the Metal buffer index its reflection names ("vertex_fetch"), so
// its pipelines do not depend on the vertex layout. The table describes the
// streams and attributes of the draw's vertex state; the shader converts
// each attribute as a Metal vertex descriptor would.
inline constexpr std::string_view vertex_fetch_schema = "armigo-vertex-fetch";
inline constexpr std::uint32_t vertex_fetch_version = 1;
inline constexpr std::size_t vertex_fetch_streams = 16;
inline constexpr std::size_t vertex_fetch_attributes = 16;

// How a stream advances. Any other step value n reads per instance:
// index = base instance + (instance id - base instance) / n. A stream with
// stride zero reads the same bytes for every index.
inline constexpr std::uint32_t vertex_fetch_step_vertex = 0;

// The storage of one attribute. The fetch computes the 32-bit register words
// the format writes, as the GPU's vertex fetch does, and the input reads them
// as its own type: integer storage writes the integer, or its float value when
// the attribute is scaled (an integer-to-float format); every other kind
// writes float bits. Missing components read (0, 0, 0, 1), the 1 being an
// integer for unscaled integer storage and 1.0 otherwise.
enum class VertexFetchKind : std::uint32_t {
    disabled,   // reads zero in every component
    unorm8, snorm8, uint8, sint8,
    unorm16, snorm16, uint16, sint16, float16,
    float32, uint32, sint32,
    unorm10_10_10_2, snorm10_10_10_2, float11_11_10,
    unorm8_bgra, // four unorm8 components stored blue, green, red, alpha
};

// One draw's table, uploaded to a constant buffer.
struct VertexFetchTable {
    std::uint64_t stream[vertex_fetch_streams];   // GPU address of the stream's first byte
    std::uint32_t stride[vertex_fetch_streams];
    std::uint32_t step[vertex_fetch_streams];
    std::uint32_t limit[vertex_fetch_streams];    // bytes readable from the stream address
    std::uint32_t attribute[vertex_fetch_attributes]; // vertex_fetch_attribute codes
};
static_assert(sizeof(VertexFetchTable) == 384);
inline constexpr std::uint32_t vertex_fetch_bytes = sizeof(VertexFetchTable);

// An attribute code: kind in bits 0-4, component count (1-4) in bits 5-7,
// stream in bits 8-11, byte offset within the vertex in bits 12-30, and bit 31
// set for an integer kind scaled to float.
inline constexpr std::uint32_t vertex_fetch_offset_limit = 1u << 19;
[[nodiscard]] constexpr std::uint32_t vertex_fetch_attribute(VertexFetchKind kind, std::uint32_t components,
                                                             std::uint32_t stream, std::uint32_t offset,
                                                             bool scaled = false) {
    return static_cast<std::uint32_t>(kind) | (components & 7u) << 5 | (stream & 15u) << 8 |
           (offset & (vertex_fetch_offset_limit - 1)) << 12 | (scaled ? 1u << 31 : 0u);
}

// The MSL the translator emits for a fetching stage: the table type and the
// functions armigo_vertex_fetch_float / _uint / _int(table, location,
// vertex id, instance id, base instance), each returning four components.
[[nodiscard]] std::string_view vertex_fetch_msl();

} // namespace armigo::shader
