// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/vertex_input.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
using namespace armigo::graphics;
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); } } while (false)

struct Backing {
    std::array<std::byte, 4096> bytes{};
    bool fail = false;
    unsigned reads = 0;

    static bool read(void* context, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<Backing*>(context);
        ++self.reads;
        if (self.fail || address < 0x8000 || address - 0x8000 > self.bytes.size() ||
            destination.size() > self.bytes.size() - (address - 0x8000)) return false;
        std::copy_n(self.bytes.begin() + (address - 0x8000), destination.size(), destination.begin());
        return true;
    }
    void put(std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte != 4; ++byte)
            bytes[offset + byte] = std::byte((value >> (byte * 8)) & 255u);
    }
};

RenderStop snapshot() {
    RenderStop result{};
    result.kind = RenderKind::Draw;
    result.engine = engine_class::threed;
    // Independently chosen attribute, stream, stride and byte offset.
    result.registers[0x458 + 7] = 3u | (5u << 7) | (2u << 21) | (7u << 27);
    result.registers[0x700 + 3 * 4] = 0x1000 | 24;
    result.registers[0x700 + 3 * 4 + 1] = 9;
    result.registers[0x700 + 3 * 4 + 2] = 0x1000;
    result.registers[0x7c0 + 3 * 2] = 9;
    result.registers[0x7c0 + 3 * 2 + 1] = 0x1fff;
    return result;
}

void test_state() {
    const auto base = snapshot();
    const auto planned = plan_vertex_float3(base, 7);
    CHECK(std::holds_alternative<VertexFloat3Plan>(planned));
    if (const auto* plan = std::get_if<VertexFloat3Plan>(&planned)) {
        CHECK(plan->attribute == 7 && plan->buffer == 3);
        CHECK(plan->address == 0x900001000 && plan->inclusive_limit == 0x900001fff);
        CHECK(plan->stride == 24 && plan->offset == 5);
    }
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(base, 32)));
    for (const auto bit : {5u, 6u, 30u, 31u}) {
        auto changed = base;
        changed.registers[0x458 + 7] |= 1u << bit;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    }
    for (unsigned size = 0; size != 64; ++size) {
        if (size == 2) continue;
        auto changed = base;
        changed.registers[0x458 + 7] ^= (2u ^ size) << 21;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    }
    for (unsigned type = 0; type != 7; ++type) {
        auto changed = base;
        changed.registers[0x458 + 7] ^= (7u ^ type) << 27;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    }
    for (const auto control : {24u, 0x1000u | 24u | (1u << 13)}) {
        auto changed = base;
        changed.registers[0x700 + 3 * 4] = control;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    }
    auto changed = base;
    changed.registers[0x620 + 3] = 1;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    changed = base;
    changed.registers[0x458 + 7] ^= 3u ^ 16u;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    changed = base;
    changed.registers[0x7c0 + 3 * 2 + 1] = 0xfff;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
    changed = base;
    changed.kind = RenderKind::Clear;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(changed, 7)));
}

void test_fetch() {
    Backing backing;
    const std::array mappings{Mapping{0x900001000, 4096, 0, 0x8000, 4096, 4096}};
    auto space = make_address_space(mappings, {Backing::read, nullptr, &backing});
    CHECK(std::holds_alternative<GpuAddressSpace>(space));
    if (!std::holds_alternative<GpuAddressSpace>(space)) return;
    const auto& memory = std::get<GpuAddressSpace>(space);
    auto plan = std::get<VertexFloat3Plan>(plan_vertex_float3(snapshot(), 7));
    constexpr std::array expected{0x80000000u, 0x7fc12345u, 0x3f900000u};
    for (unsigned component = 0; component != 3; ++component)
        backing.put(5 + 2 * 24 + component * 4, expected[component]);
    auto result = fetch_vertex_float3(plan, 2, memory);
    CHECK(std::holds_alternative<std::array<std::uint32_t, 3>>(result));
    if (const auto* value = std::get_if<std::array<std::uint32_t, 3>>(&result))
        CHECK(*value == expected);
    backing.put(5 + 2 * 24, 0x3e800000u);
    result = fetch_vertex_float3(plan, 2, memory);
    CHECK(std::get<std::array<std::uint32_t, 3>>(result)[0] == 0x3e800000u);
    plan.inclusive_limit = plan.address + 5 + 2 * 24 + 11;
    CHECK(std::holds_alternative<std::array<std::uint32_t, 3>>(fetch_vertex_float3(plan, 2, memory)));
    --plan.inclusive_limit;
    const auto reads = backing.reads;
    CHECK(std::get<CommandResult>(fetch_vertex_float3(plan, 2, memory)).status == CommandStatus::MemoryFault);
    CHECK(backing.reads == reads);
    plan.inclusive_limit = 0x900001fff;
    backing.fail = true;
    CHECK(std::get<CommandResult>(fetch_vertex_float3(plan, 2, memory)).status == CommandStatus::MemoryFault);
    backing.fail = false;
    plan.stride = 0;
    backing.put(5, 0x40400000u);
    CHECK(std::get<std::array<std::uint32_t, 3>>(fetch_vertex_float3(plan, 999, memory))[0] == 0x40400000u);
    plan.stride = 24;
    CHECK(std::get<CommandResult>(fetch_vertex_float3(plan, std::numeric_limits<std::uint64_t>::max(), memory)).status == CommandStatus::MemoryFault);
    plan.address = std::numeric_limits<std::uint64_t>::max() - 2;
    plan.inclusive_limit = std::numeric_limits<std::uint64_t>::max();
    CHECK(std::get<CommandResult>(fetch_vertex_float3(plan, 0, memory)).status == CommandStatus::MemoryFault);
    plan.address = 0x900002000;
    plan.inclusive_limit = 0x900002fff;
    CHECK(std::get<CommandResult>(fetch_vertex_float3(plan, 0, memory)).status == CommandStatus::MemoryFault);
}

RenderStop snapshot_float2() {
    RenderStop result{};
    result.kind = RenderKind::Draw;
    result.engine = engine_class::threed;
    // Independently chosen attribute, stream, stride and byte offset.
    result.registers[0x458 + 11] = 5u | (3u << 7) | (4u << 21) | (7u << 27);
    result.registers[0x700 + 5 * 4] = 0x1000 | 16;
    result.registers[0x700 + 5 * 4 + 1] = 7;
    result.registers[0x700 + 5 * 4 + 2] = 0x1000;
    result.registers[0x7c0 + 5 * 2] = 7;
    result.registers[0x7c0 + 5 * 2 + 1] = 0x2fff;
    return result;
}

void test_float2_state() {
    const auto base = snapshot_float2();
    const auto planned = plan_vertex_float2(base, 11);
    CHECK(std::holds_alternative<VertexFloat2Plan>(planned));
    if (const auto* plan = std::get_if<VertexFloat2Plan>(&planned)) {
        CHECK(plan->attribute == 11 && plan->buffer == 5);
        CHECK(plan->address == 0x700001000 && plan->inclusive_limit == 0x700002fff);
        CHECK(plan->stride == 16 && plan->offset == 3);
    }
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(base, 32)));
    for (const auto bit : {5u, 6u, 30u, 31u}) {
        auto changed = base;
        changed.registers[0x458 + 11] |= 1u << bit;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    }
    for (unsigned size = 0; size != 64; ++size) {
        if (size == 4) continue;
        auto changed = base;
        changed.registers[0x458 + 11] ^= (4u ^ size) << 21;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    }
    for (unsigned type = 0; type != 7; ++type) {
        auto changed = base;
        changed.registers[0x458 + 11] ^= (7u ^ type) << 27;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    }
    for (const auto control : {16u, 0x1000u | 16u | (1u << 13)}) {
        auto changed = base;
        changed.registers[0x700 + 5 * 4] = control;
        CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    }
    auto changed = base;
    changed.registers[0x620 + 5] = 1;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    changed = base;
    changed.registers[0x458 + 11] ^= 5u ^ 16u;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    changed = base;
    changed.registers[0x7c0 + 5 * 2 + 1] = 0xfff;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    changed = base;
    changed.kind = RenderKind::Clear;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(changed, 11)));
    // Each planner rejects the other profile's format size word.
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float2(snapshot(), 7)));
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_float3(base, 11)));
}

void test_float2_fetch() {
    Backing backing;
    const std::array mappings{Mapping{0x700001000, 4096, 0, 0x8000, 4096, 4096}};
    auto space = make_address_space(mappings, {Backing::read, nullptr, &backing});
    CHECK(std::holds_alternative<GpuAddressSpace>(space));
    if (!std::holds_alternative<GpuAddressSpace>(space)) return;
    const auto& memory = std::get<GpuAddressSpace>(space);
    auto plan = std::get<VertexFloat2Plan>(plan_vertex_float2(snapshot_float2(), 11));
    // A NaN payload and negative zero must survive bit-exact.
    constexpr std::array expected{0x7fc12345u, 0x80000000u};
    for (unsigned component = 0; component != 2; ++component)
        backing.put(3 + 2 * 16 + component * 4, expected[component]);
    auto result = fetch_vertex_float2(plan, 2, memory);
    CHECK(std::holds_alternative<std::array<std::uint32_t, 2>>(result));
    if (const auto* value = std::get_if<std::array<std::uint32_t, 2>>(&result))
        CHECK(*value == expected);
    backing.put(3 + 2 * 16, 0x3e800000u);
    result = fetch_vertex_float2(plan, 2, memory);
    CHECK(std::get<std::array<std::uint32_t, 2>>(result)[0] == 0x3e800000u);
    CHECK(std::get<std::array<std::uint32_t, 2>>(result)[1] == 0x80000000u);
    plan.inclusive_limit = plan.address + 3 + 2 * 16 + 7;
    CHECK(std::holds_alternative<std::array<std::uint32_t, 2>>(fetch_vertex_float2(plan, 2, memory)));
    --plan.inclusive_limit;
    const auto reads = backing.reads;
    CHECK(std::get<CommandResult>(fetch_vertex_float2(plan, 2, memory)).status == CommandStatus::MemoryFault);
    CHECK(backing.reads == reads);
    plan.inclusive_limit = 0x700002fff;
    backing.fail = true;
    CHECK(std::get<CommandResult>(fetch_vertex_float2(plan, 2, memory)).status == CommandStatus::MemoryFault);
    backing.fail = false;
    plan.stride = 0;
    backing.put(3, 0x40400000u);
    CHECK(std::get<std::array<std::uint32_t, 2>>(fetch_vertex_float2(plan, 999, memory))[0] == 0x40400000u);
    plan.stride = 16;
    CHECK(std::get<CommandResult>(fetch_vertex_float2(plan, std::numeric_limits<std::uint64_t>::max(), memory)).status == CommandStatus::MemoryFault);
    plan.address = std::numeric_limits<std::uint64_t>::max() - 2;
    plan.inclusive_limit = std::numeric_limits<std::uint64_t>::max();
    CHECK(std::get<CommandResult>(fetch_vertex_float2(plan, 0, memory)).status == CommandStatus::MemoryFault);
    plan.address = 0x700002000;
    plan.inclusive_limit = 0x700002fff;
    CHECK(std::get<CommandResult>(fetch_vertex_float2(plan, 0, memory)).status == CommandStatus::MemoryFault);
}
// Every admitted Maxwell encoding decodes to its portable format, class,
// component count and byte size; scaled, unknown and out-of-table encodings
// stay unsupported instead of substituting a host format.
void test_format_table() {
    struct Entry { std::uint32_t size, type; VertexAttribFormat format;
                   VertexFormatClass format_class; unsigned components, bytes; };
    static const std::vector<Entry> table{
        // 8-bit sizes convert through normalized or keep raw bytes.
        {0x1d, 2, VertexAttribFormat::R8Unorm, VertexFormatClass::float_class, 1, 1},
        {0x1d, 1, VertexAttribFormat::R8Snorm, VertexFormatClass::float_class, 1, 1},
        {0x1d, 4, VertexAttribFormat::R8Uint, VertexFormatClass::uint_class, 1, 1},
        {0x1d, 3, VertexAttribFormat::R8Sint, VertexFormatClass::sint_class, 1, 1},
        {0x18, 2, VertexAttribFormat::R8G8Unorm, VertexFormatClass::float_class, 2, 2},
        {0x18, 4, VertexAttribFormat::R8G8Uint, VertexFormatClass::uint_class, 2, 2},
        {0x13, 1, VertexAttribFormat::R8G8B8Snorm, VertexFormatClass::float_class, 3, 3},
        {0x0a, 2, VertexAttribFormat::R8G8B8A8Unorm, VertexFormatClass::float_class, 4, 4},
        {0x0a, 3, VertexAttribFormat::R8G8B8A8Sint, VertexFormatClass::sint_class, 4, 4},
        // 16-bit sizes add half floats.
        {0x1b, 7, VertexAttribFormat::R16Float, VertexFormatClass::float_class, 1, 2},
        {0x0f, 2, VertexAttribFormat::R16G16Unorm, VertexFormatClass::float_class, 2, 4},
        {0x0f, 1, VertexAttribFormat::R16G16Snorm, VertexFormatClass::float_class, 2, 4},
        {0x0f, 4, VertexAttribFormat::R16G16Uint, VertexFormatClass::uint_class, 2, 4},
        {0x05, 7, VertexAttribFormat::R16G16B16Float, VertexFormatClass::float_class, 3, 6},
        {0x03, 3, VertexAttribFormat::R16G16B16A16Sint, VertexFormatClass::sint_class, 4, 8},
        // 32-bit sizes add real floats.
        {0x12, 7, VertexAttribFormat::R32Float, VertexFormatClass::float_class, 1, 4},
        {0x04, 7, VertexAttribFormat::R32G32Float, VertexFormatClass::float_class, 2, 8},
        {0x02, 7, VertexAttribFormat::R32G32B32Float, VertexFormatClass::float_class, 3, 12},
        {0x01, 7, VertexAttribFormat::R32G32B32A32Float, VertexFormatClass::float_class, 4, 16},
        {0x01, 4, VertexAttribFormat::R32G32B32A32Uint, VertexFormatClass::uint_class, 4, 16},
        {0x04, 3, VertexAttribFormat::R32G32Sint, VertexFormatClass::sint_class, 2, 8},
        // Packed normalized formats; raw integer 1010102 has no Metal fetch.
        {0x30, 2, VertexAttribFormat::A2B10G10R10Unorm, VertexFormatClass::float_class, 4, 4},
        {0x30, 1, VertexAttribFormat::A2B10G10R10Snorm, VertexFormatClass::float_class, 4, 4},
        {0x31, 7, VertexAttribFormat::B10G11R11Float, VertexFormatClass::float_class, 3, 4},
    };
    for (const auto& entry : table) {
        const auto decoded = decode_vertex_attrib_format((entry.size << 21) | (entry.type << 27));
        CHECK(decoded.has_value());
        if (!decoded) continue;
        CHECK(decoded->format == entry.format);
        CHECK(decoded->format_class == entry.format_class);
        CHECK(decoded->components == entry.components);
        CHECK(decoded->size_bytes == entry.bytes);
    }
    // Unsupported: scaled types, integer 1010102, unknown size codes, stray
    // high bits, and the all-zero register of an unused attribute.
    for (const auto encoded : {(0x1du << 21) | (5u << 27), (0x1du << 21) | (6u << 27),
                               (0x30u << 21) | (4u << 27), (0x31u << 21) | (4u << 27),
                               (0x33u << 21) | (7u << 27), (7u << 21) | (7u << 27),
                               1u << 31, 0u}) {
        CHECK(!decode_vertex_attrib_format(encoded).has_value());
    }
}

void test_attribute_and_stream_planning() {
    RenderStop stop{};
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    // RGBA8 uint at offset 12 on stream 2, and a constant RGBA8 float attribute.
    stop.registers[0x458 + 3] = 12u << 7 | 2u | (0x0au << 21) | (4u << 27);
    stop.registers[0x458 + 4] = (1u << 6) | (1u << 7) | 5u | (0x0au << 21) | (2u << 27);
    stop.registers[0x700 + 2 * 4] = 0x1000 | 24;
    stop.registers[0x700 + 2 * 4 + 1] = 9;
    stop.registers[0x700 + 2 * 4 + 2] = 0x1000;
    stop.registers[0x7c0 + 2 * 2] = 9;
    stop.registers[0x7c0 + 2 * 2 + 1] = 0x1fff;
    const auto planned = plan_vertex_attribute(stop, 3);
    CHECK(std::holds_alternative<VertexAttributePlan>(planned));
    if (const auto* plan = std::get_if<VertexAttributePlan>(&planned)) {
        CHECK(plan->location == 3 && plan->stream == 2 && plan->offset == 12);
        CHECK(!plan->constant);
        CHECK(plan->format.format == VertexAttribFormat::R8G8B8A8Uint);
        CHECK(plan->format.format_class == VertexFormatClass::uint_class);
        CHECK(plan->format.components == 4 && plan->format.size_bytes == 4);
    }
    const auto constant = plan_vertex_attribute(stop, 4);
    CHECK(std::holds_alternative<VertexAttributePlan>(constant));
    if (const auto* plan = std::get_if<VertexAttributePlan>(&constant)) {
        CHECK(plan->constant && plan->stream == 5 && plan->offset == 1);
        CHECK(plan->format.format == VertexAttribFormat::R8G8B8A8Unorm);
    }
    // An unconsumed attribute register (all zero) fails decoding: the caller
    // only plans locations the shader reads.
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_attribute(stop, 0)));
    const auto stream = plan_vertex_stream(stop, 2);
    CHECK(std::holds_alternative<VertexStreamPlan>(stream));
    if (const auto* plan = std::get_if<VertexStreamPlan>(&stream)) {
        CHECK(plan->address == 0x900001000ull && plan->inclusive_limit == 0x900001fffull);
        CHECK(plan->stride == 24 && plan->divisor == 0);
    }
    // The constant attribute's stream is disabled: planning its stream stops.
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_stream(stop, 5)));
    // Structured stops: scaled format, disabled stream, invalid address range
    // and a non-draw snapshot.
    auto changed = stop;
    changed.registers[0x458 + 3] = 12u << 7 | 2u | (0x0au << 21) | (5u << 27);
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_attribute(changed, 3)));
    changed = stop;
    changed.registers[0x700 + 2 * 4] = 24;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_stream(changed, 2)));
    // The divisor register counts only with the stream's instancing flag set;
    // without the flag a stored divisor stays ignored.
    changed = stop;
    changed.registers[0x700 + 2 * 4 + 3] = 5;
    const auto plain = plan_vertex_stream(changed, 2);
    CHECK(std::holds_alternative<VertexStreamPlan>(plain));
    if (const auto* plan = std::get_if<VertexStreamPlan>(&plain)) {
        CHECK(plan->divisor == 0);
    }
    changed.registers[0x620 + 2] = 2;
    const auto instanced = plan_vertex_stream(changed, 2);
    CHECK(std::holds_alternative<VertexStreamPlan>(instanced));
    if (const auto* plan = std::get_if<VertexStreamPlan>(&instanced)) {
        CHECK(plan->divisor == 5);
        CHECK(plan->stride == 24 && plan->address == 0x900001000ull);
    }
    changed = stop;
    changed.registers[0x7c0 + 2 * 2 + 1] = 0;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_stream(changed, 2)));
    changed.kind = RenderKind::Clear;
    CHECK(std::holds_alternative<CommandResult>(plan_vertex_attribute(changed, 3)));
}
}  // namespace

// Formats of 32-bit words refetch as any class with the same layout; every
// narrower or packed format stays bound to its own class.
void test_class_refetch() {
    const auto as = [](std::uint32_t size, std::uint32_t type, VertexFormatClass format_class) {
        const auto decoded = decode_vertex_attrib_format((size << 21) | (type << 27));
        CHECK(decoded.has_value());
        return decoded ? vertex_format_as_class(*decoded, format_class) : std::nullopt;
    };
    const auto uint4 = as(0x01, 7, VertexFormatClass::uint_class);
    CHECK(uint4 && uint4->format == VertexAttribFormat::R32G32B32A32Uint &&
          uint4->format_class == VertexFormatClass::uint_class && uint4->components == 4 &&
          uint4->size_bytes == 16);
    const auto sint1 = as(0x12, 7, VertexFormatClass::sint_class);
    CHECK(sint1 && sint1->format == VertexAttribFormat::R32Sint && sint1->size_bytes == 4);
    const auto float3 = as(0x02, 4, VertexFormatClass::float_class);
    CHECK(float3 && float3->format == VertexAttribFormat::R32G32B32Float);
    const auto uint2 = as(0x04, 3, VertexFormatClass::uint_class);
    CHECK(uint2 && uint2->format == VertexAttribFormat::R32G32Uint);
    const auto same = as(0x0a, 2, VertexFormatClass::float_class);
    CHECK(same && same->format == VertexAttribFormat::R8G8B8A8Unorm);
    CHECK(!as(0x0a, 2, VertexFormatClass::uint_class));
    CHECK(!as(0x1d, 4, VertexFormatClass::sint_class));
    CHECK(!as(0x1b, 7, VertexFormatClass::uint_class));
    CHECK(!as(0x03, 3, VertexFormatClass::uint_class));
    CHECK(!as(0x30, 2, VertexFormatClass::uint_class));
    CHECK(!as(0x31, 7, VertexFormatClass::uint_class));
}

int main() {
    test_state();
    test_fetch();
    test_float2_state();
    test_float2_fetch();
    test_format_table();
    test_attribute_and_stream_planning();
    test_class_refetch();
    std::printf("Vertex input failures: %u\n", failures);
    return failures == 0 ? 0 : 1;
}
