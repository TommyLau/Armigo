// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/renderer_types.h>
#include <armigo/graphics/vertex_input.h>
#include <vector>

namespace armigo::graphics {

enum class PrimitiveTopology : std::uint8_t {
    Points, Lines, LineStrip, Triangles, TriangleStrip,
    LineLoop, TriangleFan, Quads, QuadStrip
};
enum class PrimitiveClass : std::uint8_t { Point, Line, Triangle };
enum class IndexFormat : std::uint8_t { U8, U16, U32 };
enum class VertexStep : std::uint8_t { PerVertex, PerInstance, Constant };
struct VertexStreamStep {
    std::uint32_t stream = 0;
    VertexStep step = VertexStep::PerVertex;
    std::uint32_t rate = 1;
    bool operator==(const VertexStreamStep&) const = default;
};
struct VertexInputPlan {
    // Reuse existing format/class/stream validation; don't create a competing
    // attribute decoder. Steps extend the existing stream's divisor contract.
    std::vector<VertexAttributePlan> attributes;
    std::vector<VertexStreamPlan> streams;
    std::vector<VertexStreamStep> steps;
};
struct IndexPlan {
    IndexFormat source_format = IndexFormat::U16;
    IndexFormat encoded_format = IndexFormat::U16;
    std::uint64_t address = 0, inclusive_limit = 0;
    bool restart_enabled = false;
    std::uint32_t restart_index = 0;
    // Nonempty for frontend inline streams. Raw, zero-extended elements are
    // owned separately from encoded output; address/limit then name no memory.
    std::vector<std::uint32_t> inline_source;
    // Nonempty for normalization/widening or inline input. Owned little-endian bytes,
    // never a guest span. Empty means a checked direct retained buffer is used.
    std::vector<std::byte> converted;
    // address/limit name the whole guest index buffer and the single segment
    // starts at the draw's first element (native base vertex assembly).
    bool whole_buffer = false;
};
struct DrawSegment {
    std::uint32_t first = 0, count = 0;
    std::int32_t base_vertex = 0;
    // Native bindings consume B+i; i remains available to recover guest B.
    std::uint32_t base_instance = 0, instance_count = 1;
    std::uint32_t instance_index = 0;
};
struct DrawPlan {
    DrawDescription source;
    PrimitiveTopology topology = PrimitiveTopology::Triangles;
    PrimitiveClass primitive_class = PrimitiveClass::Triangle;
    VertexInputPlan vertex;
    std::optional<IndexPlan> indices;
    std::vector<DrawSegment> segments;
    // Nonzero names an explicitly admitted shader variant for builtin-ID
    // preservation after normalization. No unadvertised shader patching.
    std::uint64_t builtin_emulation = 0;
};

// L8: consume the frontend's event-latched DrawDescription, not stale count
// registers. Decode Begin and topology override encodings separately. Preserve
// restart BEFORE signed base vertex, disabled-restart sentinel indices, strip
// winding/degenerates and vertex/instance builtins. U8 requires tested widening;
// loops/fans/quads require tested normalization. Adjacency/patches/unknown Begin
// controls stop. Check inclusive index/fetch limits and arithmetic for every
// consumed attribute, including per-instance divisors. Zero count is a real
// no-coverage plan, never fabricated GPU completion. memory is borrowed only
// for this call; native residency is acquired separately before acceptance.
// Inline input is explicit, starts at element zero and must match draw.count.
// It is copied into owned planning data; no synthetic guest address is used.
// native_base_vertex leaves an indexed draw's signed base vertex to the native
// draw call (segment base_vertex) instead of biasing a converted index copy;
// normalized topologies still bias during expansion.
// plan_vertex = false leaves plan.vertex empty and skips the per-draw vertex
// fetch reach checks, for a caller that reuses an earlier draw's vertex layout.
[[nodiscard]] RendererResult<DrawPlan> assemble_draw(const RenderStop& stop,
    std::span<const VertexClassBinding> inputs, const GpuAddressSpace& memory,
    std::span<const std::uint32_t> inline_indices = {}, bool native_base_vertex = false,
    bool plan_vertex = true);

} // namespace armigo::graphics
