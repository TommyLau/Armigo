// ============================================================================
// Armigo Graphics - Vertex Input Fetch
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Vertex attribute words are decoded into owned C++ enums. At draw time, the
// host vertex descriptor converts the guest format while the shader declares
// only its storage class. Unsupported encodings return structured errors.

#pragma once

#include <armigo/graphics/engines.h>
#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace armigo::graphics {

// One direct, per-vertex float32 XYZ attribute. Addresses remain GPU virtual
// addresses; planning does not copy or establish readability of guest bytes.
struct VertexFloat3Plan {
    std::uint32_t attribute = 0;
    std::uint32_t buffer = 0;
    std::uint64_t address = 0;
    std::uint64_t inclusive_limit = 0;
    std::uint32_t stride = 0;
    std::uint32_t offset = 0;
};

using VertexFloat3PlanResult = std::variant<VertexFloat3Plan, CommandResult>;
using VertexFloat3Result = std::variant<std::array<std::uint32_t, 3>, CommandResult>;

// The caller selects only attributes consumed by the bound vertex program.
// Other formats, constant attributes and instancing remain explicit stops.
[[nodiscard]] VertexFloat3PlanResult plan_vertex_float3(
    const RenderStop& snapshot, std::uint32_t attribute) noexcept;

// vertex_index is the already resolved vertex number, including any base index.
// Each call reads current guest bytes and preserves their raw binary32 bits.
[[nodiscard]] VertexFloat3Result fetch_vertex_float3(
    const VertexFloat3Plan& plan, std::uint64_t vertex_index,
    const GpuAddressSpace& memory) noexcept;

// Storage class of a vertex value. Maxwell attribute registers carry float
// bits for float/unorm/snorm formats and raw extended integer bits otherwise,
// so the host vertex descriptor must map a format whose class matches the
// declaration the translated shader was emitted with.
enum class VertexFormatClass : std::uint8_t { float_class, uint_class, sint_class };

// Guest vertex attribute formats that Metal can fetch natively, named after
// their Maxwell register encoding. Scaled formats have no Metal equivalent and
// stay unsupported instead of silently substituting an integer format.
enum class VertexAttribFormat : std::uint8_t {
    R8Unorm, R8Snorm, R8Uint, R8Sint,
    R16Float, R16Unorm, R16Snorm, R16Uint, R16Sint,
    R32Float, R32Uint, R32Sint,
    R8G8Unorm, R8G8Snorm, R8G8Uint, R8G8Sint,
    R16G16Float, R16G16Unorm, R16G16Snorm, R16G16Uint, R16G16Sint,
    R32G32Float, R32G32Uint, R32G32Sint,
    R8G8B8Unorm, R8G8B8Snorm, R8G8B8Uint, R8G8B8Sint,
    R16G16B16Float, R16G16B16Unorm, R16G16B16Snorm, R16G16B16Uint, R16G16B16Sint,
    R32G32B32Float, R32G32B32Uint, R32G32B32Sint,
    R8G8B8A8Unorm, R8G8B8A8Snorm, R8G8B8A8Uint, R8G8B8A8Sint,
    R16G16B16A16Float, R16G16B16A16Unorm, R16G16B16A16Snorm, R16G16B16A16Uint, R16G16B16A16Sint,
    R32G32B32A32Float, R32G32B32A32Uint, R32G32B32A32Sint,
    // Metal offers no raw integer 1010102 vertex format, so only the
    // normalized packed forms are admitted.
    A2B10G10R10Unorm, A2B10G10R10Snorm,
    B10G11R11Float,
};

struct VertexAttribFormatInfo {
    VertexAttribFormat format{};
    VertexFormatClass format_class = VertexFormatClass::float_class;
    std::uint8_t components = 0;
    std::uint8_t size_bytes = 0;
};

// encoded is the packed Maxwell format field of an attribute register: a six
// bit size and three bit type at bits 21..30. Unsupported encodings yield
// std::nullopt; callers turn that into a structured error, never a substitute.
[[nodiscard]] std::optional<VertexAttribFormatInfo> decode_vertex_attrib_format(
    std::uint32_t encoded) noexcept;

// The same guest bytes fetched for a shader declared with another storage
// class. Only formats whose components are all 32-bit words qualify: Maxwell
// keeps binary32 float bits and raw 32-bit integers unconverted in the
// attribute register, so a shader that reinterprets a float attribute as
// integer bits (or the reverse) reads the guest words exactly when the host
// fetches them with the matching 32-bit format of the declared class.
// Narrower formats are normalized, converted or extended by the fetch and
// yield std::nullopt.
[[nodiscard]] std::optional<VertexAttribFormatInfo> vertex_format_as_class(
    const VertexAttribFormatInfo& format, VertexFormatClass format_class) noexcept;

// One decoded guest attribute register: format identity plus where the value
// lives (stream, byte offset) and whether the register marks it constant.
struct VertexAttributePlan {
    std::uint32_t location = 0;
    std::uint32_t stream = 0;
    std::uint32_t offset = 0;
    bool constant = false;
    VertexAttribFormatInfo format;
};

// The caller supplies the shader-declared locations only; registers of
// unconsumed attributes are never decoded.
using VertexAttributePlanResult = std::variant<VertexAttributePlan, CommandResult>;
[[nodiscard]] VertexAttributePlanResult plan_vertex_attribute(
    const RenderStop& snapshot, std::uint32_t location) noexcept;

// One decoded, enabled guest vertex stream. The inclusive limit is the last
// valid byte address. Divisor follows the guest instancing flag of the same
// stream: zero steps per vertex, a nonzero value divides the instance counter
// (the attribute index advances once per that many instances).
struct VertexStreamPlan {
    std::uint32_t stream = 0;
    std::uint64_t address = 0;
    std::uint64_t inclusive_limit = 0;
    std::uint32_t stride = 0;
    std::uint32_t divisor = 0;
};

using VertexStreamPlanResult = std::variant<VertexStreamPlan, CommandResult>;
[[nodiscard]] VertexStreamPlanResult plan_vertex_stream(
    const RenderStop& snapshot, std::uint32_t stream) noexcept;

// A translated generic vertex input, recorded by reflection v2 from the
// code-derived storage class: location is both the guest attribute register
// index and the host vertex-descriptor attribute index.
struct VertexClassBinding {
    std::uint8_t location = 0;
    VertexFormatClass input_class = VertexFormatClass::float_class;
};

// One direct, per-vertex float32 XY attribute. Same fields and meaning as
// VertexFloat3Plan; addresses remain GPU virtual addresses and planning does
// not copy or establish readability of guest bytes.
struct VertexFloat2Plan {
    std::uint32_t attribute = 0;
    std::uint32_t buffer = 0;
    std::uint64_t address = 0;
    std::uint64_t inclusive_limit = 0;
    std::uint32_t stride = 0;
    std::uint32_t offset = 0;
};

using VertexFloat2PlanResult = std::variant<VertexFloat2Plan, CommandResult>;
using VertexFloat2Result = std::variant<std::array<std::uint32_t, 2>, CommandResult>;

// The caller selects only attributes consumed by the bound vertex program.
// Other formats, constant attributes and instancing remain explicit stops.
[[nodiscard]] VertexFloat2PlanResult plan_vertex_float2(
    const RenderStop& snapshot, std::uint32_t attribute) noexcept;

// vertex_index is the already resolved vertex number, including any base index.
// Each call reads current guest bytes and preserves their raw binary32 bits.
[[nodiscard]] VertexFloat2Result fetch_vertex_float2(
    const VertexFloat2Plan& plan, std::uint64_t vertex_index,
    const GpuAddressSpace& memory) noexcept;

}  // namespace armigo::graphics
