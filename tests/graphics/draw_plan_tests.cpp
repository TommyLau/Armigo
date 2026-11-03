// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/draw_plan.h>
#include <armigo/graphics/vertex_input.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

namespace {
using namespace armigo::graphics;
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); } } while (false)

// Independently authored synthetic fixture: one mapped region at 0x100000
// backed by a local array, holding index bytes and vertex payload bytes.
struct Fixture {
    std::vector<std::byte> bytes{std::vector<std::byte>(0x10000)};
    bool fail = false;

    // The mapping exposes guest base 0x8000; callbacks address guest bytes.
    static bool read(void* context, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<Fixture*>(context);
        if (self.fail || address < 0x8000 ||
            address - 0x8000 > self.bytes.size() ||
            destination.size() > self.bytes.size() - (address - 0x8000)) {
            return false;
        }
        std::copy_n(self.bytes.begin() + (address - 0x8000),
                    destination.size(), destination.begin());
        return true;
    }

    void put16(std::uint64_t offset, std::uint32_t value) {
        bytes[offset] = std::byte(value & 255u);
        bytes[offset + 1] = std::byte((value >> 8) & 255u);
    }

    void put32(std::uint64_t offset, std::uint32_t value) {
        put16(offset, value & 65535u);
        put16(offset + 2, (value >> 16) & 65535u);
    }
};

std::size_t encoded_size(IndexFormat format) {
    switch (format) {
    case IndexFormat::U8: return 1;
    case IndexFormat::U16: return 2;
    case IndexFormat::U32: return 4;
    }
    return 1;
}

std::vector<std::uint32_t> decoded(const IndexPlan& indices) {
    const std::size_t size = encoded_size(indices.encoded_format);
    std::vector<std::uint32_t> values;
    const auto& bytes = indices.converted;
    values.reserve(bytes.size() / size);
    for (std::size_t offset = 0; offset + size <= bytes.size(); offset += size) {
        std::uint32_t value = 0;
        for (std::size_t i = 0; i != size; ++i) {
            value |= std::uint32_t{std::to_integer<std::uint32_t>(bytes[offset + i])}
                     << (i * 8);
        }
        values.push_back(value);
    }
    return values;
}

// One float32 XYZ attribute on stream 0 plus the draw registers of the
// snapshot; tests extend from here.
RenderStop base_snapshot() {
    RenderStop stop{};
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    // Attribute 3: R32G32B32 float at offset 0 on stream 0.
    stop.registers[0x458 + 3] = (0u << 7) | 0u | (0x12u << 21) | (7u << 27);
    // Stream 0: enabled, stride 12, address 0x200000, end 0x200fff.
    stop.registers[0x700] = 0x1000 | 12;
    stop.registers[0x701] = 0;
    stop.registers[0x702] = 0x200000;
    stop.registers[0x7c0] = 0;
    stop.registers[0x7c1] = 0x200fff;
    stop.registers[0x703] = 0;
    return stop;
}

DrawDescription triangle_arrays(std::uint32_t first, std::uint32_t count) {
    DrawDescription draw;
    draw.indexed = false;
    draw.topology = 4;
    draw.first = first;
    draw.count = count;
    draw.base_vertex = 0;
    draw.base_instance = 0;
    draw.instance_count = 1;
    return draw;
}

DrawDescription indexed(std::uint32_t topology, std::uint32_t first,
                        std::uint32_t count, std::int32_t base_vertex,
                        std::uint32_t base_instance, std::uint32_t instances) {
    DrawDescription draw;
    draw.indexed = true;
    draw.topology = topology;
    draw.first = first;
    draw.count = count;
    draw.base_vertex = base_vertex;
    draw.base_instance = base_instance;
    draw.instance_count = instances;
    return draw;
}

// Index buffer state: U16 at 0x100000..0x100fff unless overridden.
void set_index_buffer(RenderStop& stop, std::uint32_t type, std::uint64_t start,
                      std::uint64_t end) {
    stop.registers[0x5f2] = std::uint32_t(start >> 32);
    stop.registers[0x5f3] = std::uint32_t(start & 0xffffffffu);
    stop.registers[0x5f4] = std::uint32_t(end >> 32);
    stop.registers[0x5f5] = std::uint32_t(end & 0xffffffffu);
    stop.registers[0x5f6] = type;
}

// Returns by value so a reference never points into the call's temporary.
DrawPlan plan_or_fail(const RendererResult<DrawPlan>& result) {
    static const DrawPlan empty{};
    if (const auto* plan = std::get_if<DrawPlan>(&result)) return *plan;
    const auto& error = std::get<RendererError>(result);
    const auto command = error.command();
    std::fprintf(stderr, "FAIL line context: unexpected stop status %d: %.*s\n",
                 int(command.status),
                 int(command.detail.size()), command.detail.data());
    ++failures;
    return empty;
}

bool stopped_with(const RendererResult<DrawPlan>& result, CommandStatus status) {
    if (const auto* error = std::get_if<RendererError>(&result)) {
        return error->command().status == status;
    }
    return false;
}

constexpr VertexClassBinding float_binding(std::uint8_t location) {
    return VertexClassBinding{location, VertexFormatClass::float_class};
}

RendererResult<DrawPlan> assemble(
    const RenderStop& stop, std::initializer_list<VertexClassBinding> inputs,
    const GpuAddressSpace& memory) {
    return assemble_draw(stop, inputs, memory);
}

void test_arrays_and_index_formats(Fixture& fixture, const GpuAddressSpace& memory) {
    // Non-indexed triangles: a single direct segment with the array window.
    auto stop = base_snapshot();
    stop.draw = triangle_arrays(7, 9);
    const auto& plan = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(plan.segments.size() == 1);
    if (plan.segments.size() == 1) {
        CHECK(plan.segments[0].first == 7 && plan.segments[0].count == 9);
        CHECK(plan.segments[0].base_vertex == 0);
        CHECK(plan.segments[0].base_instance == 0 && plan.segments[0].instance_count == 1);
    }
    CHECK(!plan.indices);
    CHECK(plan.topology == PrimitiveTopology::Triangles);
    CHECK(plan.primitive_class == PrimitiveClass::Triangle);
    CHECK(plan.builtin_emulation == 0);
    CHECK(plan.vertex.attributes.size() == 1 && plan.vertex.streams.size() == 1);
    CHECK(plan.vertex.steps.size() == 1 &&
          plan.vertex.steps[0].step == VertexStep::PerVertex);

    // U16 indices in range bind directly: no converted bytes.
    set_index_buffer(stop, 1, 0x100000, 0x100fff);
    stop.draw = indexed(4, 2, 6, 0, 0, 1);
    for (unsigned element = 0; element != 8; ++element) fixture.put16(element * 2, element * 3);
    const auto& direct = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(direct.indices);
    if (direct.indices) {
        CHECK(direct.indices->source_format == IndexFormat::U16);
        CHECK(direct.indices->encoded_format == IndexFormat::U16);
        CHECK(direct.indices->address == 0x100000 + 2 * 2);
        CHECK(direct.indices->inclusive_limit == 0x100000 + (2 + 6) * 2 - 1);
        CHECK(direct.indices->converted.empty());
        CHECK(direct.indices->restart_enabled == false);
    }
    CHECK(direct.segments.size() == 1);
    if (direct.segments.size() == 1) {
        // Segment offsets address the encoded buffer, whose element zero is
        // the draw's first element.
        CHECK(direct.segments[0].first == 0 && direct.segments[0].count == 6);
    }

    // U32 indices bind directly as well.
    set_index_buffer(stop, 2, 0x100000, 0x100fff);
    stop.draw = indexed(4, 0, 3, 0, 0, 1);
    fixture.put32(0, 10);
    fixture.put32(4, 11);
    fixture.put32(8, 12);
    const auto& wide = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(wide.indices && wide.indices->encoded_format == IndexFormat::U32);
    CHECK(wide.indices && wide.indices->converted.empty());

    // U8 indices always widen to U16 for Metal, one element per source byte;
    // the encoded buffer holds the drawn range only.
    set_index_buffer(stop, 0, 0x100000, 0x100fff);
    stop.draw = indexed(4, 1, 4, 0, 0, 1);
    const std::array<std::uint8_t, 6> byte_indices{{5, 0, 9, 7, 1, 3}};
    for (unsigned element = 0; element != byte_indices.size(); ++element) {
        fixture.bytes[element] = std::byte(byte_indices[element]);
    }
    const auto& widened = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(widened.indices && widened.indices->source_format == IndexFormat::U8);
    CHECK(widened.indices && widened.indices->encoded_format == IndexFormat::U16);
    if (widened.indices) {
        const auto values = decoded(*widened.indices);
        CHECK(values.size() == 4);
        const std::vector<std::uint32_t> expected{0, 9, 7, 1};
        CHECK(values == expected);
    }
    CHECK(widened.segments.size() == 1);
    if (widened.segments.size() == 1) {
        CHECK(widened.segments[0].first == 0 && widened.segments[0].count == 4);
    }
    CHECK(widened.indices && widened.indices->address == 0x100000 + 1);

    // Out-of-range element windows and bad ranges stop with a memory fault.
    stop.draw = indexed(4, 0x800, 0x801, 0, 0, 1);
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::MemoryFault));
    set_index_buffer(stop, 3, 0x100000, 0x100fff);
    stop.draw = indexed(4, 0, 1, 0, 0, 1);
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    set_index_buffer(stop, 1, 0, 0x100fff);
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::MemoryFault));
    set_index_buffer(stop, 1, 0x100fff, 0x100000);
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::MemoryFault));
    fixture.fail = true;
    set_index_buffer(stop, 0, 0x100000, 0x100fff);
    stop.draw = indexed(4, 0, 4, 0, 0, 1);  // U8 forces the index scan
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::MemoryFault));
    fixture.fail = false;
}

void test_restart_and_base_vertex(Fixture& fixture, const GpuAddressSpace& memory) {
    auto stop = base_snapshot();
    // Bias tests reach past a small window, so widen stream 0's end address.
    stop.registers[0x7c1] = 0x2fffff;
    set_index_buffer(stop, 1, 0x100000, 0x100fff);
    // 0 1 2 | 0xffff 0xffff | 5 6 7, restart index 0xffff, U16.
    const std::vector<std::uint32_t> raw{0, 1, 2, 0xffff, 0xffff, 5, 6, 7};
    for (unsigned element = 0; element != raw.size(); ++element) {
        fixture.put16(element * 2, raw[element]);
    }
    stop.registers[0x591] = 1;     // restart enable
    stop.registers[0x592] = 0xffff;  // restart index
    stop.draw = indexed(5, 0, 8, 0, 0, 1);  // triangle strip
    const auto& split = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(split.indices && split.indices->restart_enabled);
    CHECK(split.indices && split.indices->converted.empty());
    CHECK(split.segments.size() == 2);
    if (split.segments.size() == 2) {
        CHECK(split.segments[0].first == 0 && split.segments[0].count == 3);
        CHECK(split.segments[1].first == 5 && split.segments[1].count == 3);
    }

    // Restart before base vertex: the raw value decides the split; biased
    // entries then widen to U32 when the signed sum exceeds sixteen bits.
    stop.draw = indexed(4, 0, 8, 0x10000, 0, 1);  // triangles, base 0x10000
    const auto& biased = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(biased.segments.size() == 2);
    if (biased.segments.size() == 2) {
        CHECK(biased.segments[0].first == 0 && biased.segments[0].count == 3);
        CHECK(biased.segments[1].first == 5 && biased.segments[1].count == 3);
        CHECK(biased.segments[0].base_vertex == 0);
    }
    CHECK(biased.indices && biased.indices->encoded_format == IndexFormat::U32);
    if (biased.indices) {
        const auto values = decoded(*biased.indices);
        // Restart entries keep their raw value; live entries carry the base.
        const std::vector<std::uint32_t> expected{
            0x10000, 0x10001, 0x10002, 0xffff, 0xffff,
            0x10005, 0x10006, 0x10007};
        CHECK(values == expected);
    }

    // A negative sum stops per element.
    stop.draw = indexed(4, 0, 3, -1, 0, 1);
    fixture.put16(0, 0);
    fixture.put16(2, 1);
    fixture.put16(4, 2);
    stop.registers[0x591] = 0;
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::MemoryFault));

    // A base that fits sixteen bits keeps the encoded format at U16.
    stop.draw = indexed(4, 0, 3, 100, 0, 1);
    const auto& small_bias = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(small_bias.indices && small_bias.indices->encoded_format == IndexFormat::U16);
    if (small_bias.indices) {
        const auto values = decoded(*small_bias.indices);
        CHECK((values == std::vector<std::uint32_t>{100, 101, 102}));
    }

    // Restart disabled: a sentinel-valued element stays an ordinary index.
    stop.draw = indexed(4, 0, 8, 0, 0, 1);
    for (unsigned element = 0; element != raw.size(); ++element) {
        fixture.put16(element * 2, raw[element]);
    }
    const auto& sentinel = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(sentinel.segments.size() == 1);
    if (sentinel.segments.size() == 1) {
        CHECK(sentinel.segments[0].first == 0 && sentinel.segments[0].count == 8);
    }
    CHECK(sentinel.indices && sentinel.indices->converted.empty());

    // Array restart (non-indexed) needs the separate array control and splits
    // on vertex numbers.
    stop = base_snapshot();
    stop.registers[0x591] = 1;
    stop.registers[0x592] = 5;
    stop.registers[0x37a] = 1;
    stop.draw = triangle_arrays(3, 6);  // vertices 3..8, restart at 5
    const auto& arrays = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(arrays.segments.size() == 2);
    if (arrays.segments.size() == 2) {
        CHECK(arrays.segments[0].first == 3 && arrays.segments[0].count == 2);
        CHECK(arrays.segments[1].first == 6 && arrays.segments[1].count == 3);
    }
    // Without the array control, restart does not apply to arrays.
    stop.registers[0x37a] = 0;
    const auto& plain_arrays = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(plain_arrays.segments.size() == 1);
    // Array draws reject a signed base vertex.
    stop.draw = triangle_arrays(0, 3);
    stop.draw->base_vertex = 1;
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
}

void test_topology_conversions(Fixture& fixture, const GpuAddressSpace& memory) {
    // Non-indexed fan: winding-preserving triples anchored at vertex `first`.
    auto stop = base_snapshot();
    stop.draw = triangle_arrays(10, 5);
    stop.draw->topology = 6;  // Begin triangle fan
    stop.draw->base_instance = 11;
    stop.draw->instance_index = 3;
    const auto& fan = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(fan.topology == PrimitiveTopology::Triangles);
    CHECK(fan.primitive_class == PrimitiveClass::Triangle);
    CHECK(fan.indices && fan.indices->encoded_format == IndexFormat::U32);
    if (fan.indices) {
        const auto values = decoded(*fan.indices);
        const std::vector<std::uint32_t> expected{
            10, 11, 12, 10, 12, 13, 10, 13, 14};
        CHECK(values == expected);
    }
    CHECK(fan.segments.size() == 1);
    if (fan.segments.size() == 1) {
        CHECK(fan.segments[0].count == 9);
        CHECK(fan.segments[0].base_instance == 14);
        CHECK(fan.segments[0].instance_index == 3);
    }
    CHECK(fan.builtin_emulation == 0);

    // Begin polygon is planned as a fan.
    stop.draw->topology = 9;
    const auto& polygon = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(polygon.topology == PrimitiveTopology::Triangles);
    CHECK(polygon.indices && polygon.indices->converted.size() == 9 * 4);

    // Non-indexed quads split along the 0-2 diagonal; partial quads vanish.
    stop.draw = triangle_arrays(0, 8);
    stop.draw->topology = 7;  // quads
    const auto& quads = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    if (quads.indices) {
        const auto values = decoded(*quads.indices);
        const std::vector<std::uint32_t> expected{
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
        CHECK(values == expected);
    } else {
        CHECK(false);
    }
    stop.draw = triangle_arrays(0, 7);  // one full quad plus a leftover vertex
    stop.draw->topology = 7;
    const auto& partial = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(partial.indices && partial.indices->converted.size() == 6 * 4);

    // Quad strip keeps the guest 1-3 diagonal per quad: cyclic order 0,1,3,2.
    stop.draw = triangle_arrays(0, 6);
    stop.draw->topology = 8;  // quad strip
    const auto& strip = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    if (strip.indices) {
        const auto values = decoded(*strip.indices);
        const std::vector<std::uint32_t> expected{
            0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4};
        CHECK(values == expected);
    } else {
        CHECK(false);
    }
    CHECK(strip.segments.size() == 1 && strip.segments[0].count == 12);

    // Non-indexed line loop: line list with the closing wrap segment.
    stop.draw = triangle_arrays(4, 3);
    stop.draw->topology = 2;  // line loop
    const auto& loop = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(loop.topology == PrimitiveTopology::Lines);
    CHECK(loop.primitive_class == PrimitiveClass::Line);
    if (loop.indices) {
        const auto values = decoded(*loop.indices);
        const std::vector<std::uint32_t> expected{4, 5, 5, 6, 6, 4};
        CHECK(values == expected);
    } else {
        CHECK(false);
    }

    // Incomplete normalized primitives cover nothing but stay real plans.
    stop.draw = triangle_arrays(0, 2);
    stop.draw->topology = 6;  // fan with fewer than three vertices
    const auto& stub_fan = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(stub_fan.segments.empty() && !stub_fan.indices);
    stop.draw = triangle_arrays(0, 1);
    stop.draw->topology = 2;  // loop with fewer than two vertices
    const auto& stub_loop = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(stub_loop.segments.empty());

    // Indexed conversions carry the guest index values, including the fan
    // anchor and the loop wrap.
    set_index_buffer(stop, 1, 0x100000, 0x100fff);
    const std::vector<std::uint32_t> guest{20, 21, 22, 23, 24};
    for (unsigned element = 0; element != guest.size(); ++element) {
        fixture.put16(element * 2, guest[element]);
    }
    stop.draw = indexed(6, 0, 5, 0, 0, 1);  // indexed fan
    const auto& indexed_fan = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(indexed_fan.topology == PrimitiveTopology::Triangles);
    if (indexed_fan.indices) {
        const auto values = decoded(*indexed_fan.indices);
        const std::vector<std::uint32_t> expected{
            20, 21, 22, 20, 22, 23, 20, 23, 24};
        CHECK(values == expected);
    } else {
        CHECK(false);
    }
    stop.draw = indexed(2, 0, 4, 0, 0, 1);  // indexed line loop
    const auto& indexed_loop = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(indexed_loop.topology == PrimitiveTopology::Lines);
    if (indexed_loop.indices) {
        const auto values = decoded(*indexed_loop.indices);
        const std::vector<std::uint32_t> expected{20, 21, 21, 22, 22, 23, 23, 20};
        CHECK(values == expected);
    } else {
        CHECK(false);
    }

    // Indexed quad strip with a base vertex keeps values and diagonal.
    stop.draw = indexed(8, 0, 4, 2, 0, 1);  // quad strip, base 2
    fixture.put16(0, 7);
    fixture.put16(2, 8);
    fixture.put16(4, 9);
    fixture.put16(6, 10);
    const auto& indexed_strip = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(indexed_strip.indices && indexed_strip.indices->encoded_format == IndexFormat::U32);
    if (indexed_strip.indices) {
        const auto values = decoded(*indexed_strip.indices);
        const std::vector<std::uint32_t> expected{9, 10, 12, 9, 12, 11};
        CHECK(values == expected);
    }

    // Adjacency, patches and unknown Begin encodings stop.
    for (const auto begin : {10u, 11u, 12u, 13u}) {
        stop.draw = indexed(begin, 0, 4, 0, 0, 1);
        CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                           CommandStatus::Unsupported));
    }
    stop.draw = triangle_arrays(0, 4);
    stop.draw->topology = 14;  // patches
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    stop.draw->topology = 15;
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    stop.draw->topology = (1u << 26) | 4u;  // instance control bits
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));

    // The topology override decodes from its own table.
    stop.draw = triangle_arrays(10, 5);
    stop.draw->topology = 4;
    stop.registers[0x652] = 1;      // override enable
    stop.registers[0x65c] = 0x1015;  // override fan
    const auto& override_fan = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(override_fan.topology == PrimitiveTopology::Triangles);
    CHECK(override_fan.indices && override_fan.indices->converted.size() == 9 * 4);
    stop.registers[0x65c] = 7;  // quads are absent from the override table
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    stop.registers[0x65c] = 14;  // patches through the override
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    // Restart during normalization stops instead of dropping restarts.
    stop.registers[0x652] = 0;
    stop.draw = indexed(7, 0, 8, 0, 0, 1);  // indexed quads
    stop.registers[0x591] = 1;
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    stop.registers[0x591] = 0;

    // Array restart during normalization stops under the same documented
    // limitation instead of silently expanding across the restart vertex.
    stop.draw = triangle_arrays(0, 4);
    stop.draw->topology = 6;  // triangle fan
    stop.registers[0x591] = 1;
    stop.registers[0x592] = 2;
    stop.registers[0x37a] = 1;
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
}

void test_instancing_and_builtins(Fixture& fixture, const GpuAddressSpace& memory) {
    auto stop = base_snapshot();
    // A second attribute on instanced stream 1 with divisor 3: R32 float at
    // offset 0, uint class... keep float class with R32 float.
    stop.registers[0x458 + 4] = (0u << 7) | 1u | (0x12u << 21) | (7u << 27);
    stop.registers[0x620 + 1] = 1;  // instancing flag
    stop.registers[0x707] = 3;      // divisor register of stream 1
    stop.registers[0x700 + 1 * 4] = 0x1000 | 16;
    stop.registers[0x700 + 1 * 4 + 1] = 0;
    stop.registers[0x700 + 1 * 4 + 2] = 0x300000;  // address 0x300000
    stop.registers[0x7c0 + 1 * 2] = 0;
    stop.registers[0x7c0 + 1 * 2 + 1] = 0x300fff;  // end 0x300fff
    stop.draw = triangle_arrays(0, 6);
    stop.draw->base_instance = 7;
    stop.draw->instance_index = 2;
    stop.draw->instance_count = 5;
    stop.registers[0x7c0 + 1 * 2 + 1] = 0x300043;
    const auto& plan = plan_or_fail(
        assemble(stop, {float_binding(3), float_binding(4)}, memory));
    CHECK(plan.vertex.streams.size() == 2 && plan.vertex.steps.size() == 2);
    CHECK(plan.vertex.steps[0].step == VertexStep::PerVertex);
    CHECK(plan.vertex.steps[0].rate == 1);
    CHECK(plan.vertex.steps[1].step == VertexStep::PerInstance);
    CHECK(plan.vertex.steps[1].rate == 3);
    CHECK(plan.vertex.steps[1].stream == 1);
    // Native segment binding receives B+i while the continuation index remains
    // available separately for recovering the original guest base B.
    CHECK(plan.segments.size() == 1);
    if (plan.segments.size() == 1) {
        CHECK(plan.segments[0].base_instance == 9);
        CHECK(plan.segments[0].instance_count == 5);
        CHECK(plan.segments[0].instance_index == 2);
    }
    CHECK(plan.source.base_instance == 7 && plan.source.instance_index == 2);
    CHECK(plan.builtin_emulation == 0);

    // Per-instance reach uses B+i: last fetch index (7 + 2 + 5 - 1) / 3 = 4,
    // whose inclusive byte reach is 4 * 16 + 3 = 67.
    stop.registers[0x7c0 + 1 * 2 + 1] = 0x300042;
    CHECK(stopped_with(
        assemble(stop, {float_binding(3), float_binding(4)}, memory),
        CommandStatus::MemoryFault));
    stop.registers[0x7c0 + 1 * 2 + 1] = 0x300fff;
    stop.draw->base_instance = 0xffffffc0u;
    stop.draw->instance_index = 0;
    stop.draw->instance_count = 0x3fu;
    CHECK(stopped_with(
        assemble(stop, {float_binding(3), float_binding(4)}, memory),
        CommandStatus::MemoryFault));

    // Manually supplied descriptions must not wrap B+i before planning.
    stop.draw->base_instance = std::numeric_limits<std::uint32_t>::max();
    stop.draw->instance_index = 1;
    stop.draw->instance_count = 1;
    CHECK(stopped_with(
        assemble(stop, {float_binding(3), float_binding(4)}, memory),
        CommandStatus::BudgetExceeded));

    // An instanced flagged stream with divisor zero steps per vertex.
    stop.draw->base_instance = 0;
    stop.draw->instance_index = 0;
    stop.draw->instance_count = 1;
    stop.registers[0x707] = 0;
    const auto& divisor_zero = plan_or_fail(
        assemble(stop, {float_binding(3), float_binding(4)}, memory));
    CHECK(divisor_zero.vertex.steps[1].step == VertexStep::PerVertex);
    // A zero stride stream reads one constant address.
    stop.registers[0x700 + 1 * 4] = 0x1000;
    const auto& constant = plan_or_fail(
        assemble(stop, {float_binding(3), float_binding(4)}, memory));
    CHECK(constant.vertex.steps[1].step == VertexStep::Constant);

    // A 32-bit float register feeding a uint declaration refetches the same
    // words with the uint format; narrower formats stop instead.
    stop.registers[0x700 + 1 * 4] = 0x1000 | 16;
    const auto& refetched = plan_or_fail(assemble(stop,
        {VertexClassBinding{3, VertexFormatClass::uint_class}}, memory));
    CHECK(refetched.vertex.attributes.size() == 1 &&
          refetched.vertex.attributes[0].format.format == VertexAttribFormat::R32Uint &&
          refetched.vertex.attributes[0].format.format_class == VertexFormatClass::uint_class);
    const auto float_register = stop.registers[0x458 + 3];
    stop.registers[0x458 + 3] = (0u << 7) | 0u | (0x0au << 21) | (2u << 27);
    CHECK(stopped_with(
        assemble(stop,
                 {VertexClassBinding{3, VertexFormatClass::uint_class}},
                 memory),
        CommandStatus::Unsupported));
    stop.registers[0x458 + 3] = float_register;

    // Zero count is a real no-coverage plan.
    stop = base_snapshot();
    stop.draw = triangle_arrays(0, 0);
    const auto& zero = plan_or_fail(assemble(stop, {float_binding(3)}, memory));
    CHECK(zero.segments.empty() && !zero.indices);
    CHECK(zero.topology == PrimitiveTopology::Triangles);
    CHECK(zero.vertex.streams.size() == 1);

    // A missing event-latched description and foreign snapshots stop.
    stop.draw.reset();
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
    stop = base_snapshot();
    stop.kind = RenderKind::Clear;
    stop.draw = triangle_arrays(0, 3);
    CHECK(stopped_with(assemble(stop, {float_binding(3)}, memory),
                       CommandStatus::Unsupported));
}

void test_inline_indices(Fixture& fixture, const GpuAddressSpace& memory) {
    auto stop = base_snapshot();
    stop.draw = indexed(4, 0, 7, -1, 5, 1);
    stop.draw->instance_index = 4;
    stop.registers[0x591] = 1;
    stop.registers[0x592] = 0xffffffffu;
    // Deliberately invalid ordinary index state: inline input has no guest VA.
    set_index_buffer(stop, 99, 0, 0);
    std::vector<std::uint32_t> words{1, 2, 3, 0xffffffffu, 4, 5, 6};
    const std::array bindings{float_binding(3)};
    fixture.fail = true;
    const auto plan = plan_or_fail(assemble_draw(stop, bindings, memory, words));
    CHECK(plan.indices.has_value());
    if (plan.indices) {
        CHECK(plan.indices->address == 0 && plan.indices->inclusive_limit == 0);
        CHECK(plan.indices->inline_source == words);
        CHECK(decoded(*plan.indices) ==
              (std::vector<std::uint32_t>{0, 1, 2, 0xffffffffu, 3, 4, 5}));
        words[0] = 77;
        CHECK(plan.indices->inline_source[0] == 1);
    }
    CHECK(plan.segments.size() == 2);
    if (plan.segments.size() == 2) {
        CHECK(plan.segments[0].first == 0 && plan.segments[0].count == 3);
        CHECK(plan.segments[1].first == 4 && plan.segments[1].count == 3);
        CHECK(plan.segments[0].base_instance == 9 &&
              plan.segments[1].base_instance == 9);
        CHECK(plan.segments[0].instance_index == 4 &&
              plan.segments[1].instance_index == 4);
    }
    stop.draw->first = 1;
    CHECK(stopped_with(assemble_draw(stop, bindings, memory, words),
                       CommandStatus::Unsupported));
    stop.draw->first = 0;
    --stop.draw->count;
    CHECK(stopped_with(assemble_draw(stop, bindings, memory, words),
                       CommandStatus::Unsupported));

    stop.draw = indexed(6, 0, 4, -4, 0, 1);
    stop.registers[0x591] = 0;
    words = {4, 5, 6, 7};
    const auto fan = plan_or_fail(assemble_draw(stop, bindings, memory, words));
    CHECK(fan.indices && decoded(*fan.indices) ==
          (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3}));

    stop.draw = indexed(5, 0, 3, -100, 0, 1);
    stop.registers[0x591] = 1;
    words.assign(3, 0xffffffffu);
    const auto empty = plan_or_fail(assemble_draw(stop, bindings, memory, words));
    CHECK(empty.segments.empty() && !empty.indices);
    fixture.fail = false;
}

}  // namespace

int main() {
    Fixture fixture;
    const std::array mappings{Mapping{0x100000, 0x10000, 0, 0x8000, 0x10000, 0x1000}};
    auto space = make_address_space(mappings, {Fixture::read, nullptr, &fixture});
    if (!std::holds_alternative<GpuAddressSpace>(space)) {
        std::fprintf(stderr, "FAIL: address space setup\n");
        return 1;
    }
    const auto& memory = std::get<GpuAddressSpace>(space);
    test_arrays_and_index_formats(fixture, memory);
    test_restart_and_base_vertex(fixture, memory);
    test_topology_conversions(fixture, memory);
    test_instancing_and_builtins(fixture, memory);
    test_inline_indices(fixture, memory);
    std::printf("Draw plan failures: %u\n", failures);
    return failures == 0 ? 0 : 1;
}
